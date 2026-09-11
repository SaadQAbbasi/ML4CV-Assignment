"""Checkpoint-only retrieval ablations on held-out training frames."""
from pathlib import Path
import torch
from PIL import Image
from torchvision.ops import box_iou
from prw_dataset import preprocess_query_crop
from prw_evaluation import _image_tensor, _load_gallery_target


def detection_counts(boxes, scores, gt, threshold, iou_threshold):
    """Confidence-ordered one-to-one matches, including unlabeled GT people."""
    order = scores.argsort(descending=True)
    overlaps = box_iou(boxes, gt)
    used = set()
    hits = 0
    for index in order.tolist():
        if scores[index] < threshold:
            continue
        for target in overlaps[index].argsort(descending=True).tolist():
            if overlaps[index, target] < iou_threshold:
                break
            if target not in used:
                used.add(target)
                hits += 1
                break
    return hits, int((scores >= threshold).sum()), len(gt)


@torch.inference_mode()
def run_retrieval_diagnostics(model, root, frame_names, device,
                              thresholds=(0., .05, .1, .3, .5), max_queries=100):
    """Cache detections/GT embeddings once, then compare identical queries.

    Uses supplied validation frames only. Queries are original-resolution GT
    crops of identities present in at least two frames. Source frames are
    excluded. All GT people, including unlabeled people, remain distractors.
    Returned examples contain top-five boxes and separate detector confidence.
    """
    root = Path(root)
    names = list(dict.fromkeys(frame_names))
    gallery = []
    was_training, old_threshold = model.training, model.det_score_thresh
    model.eval()
    model.det_score_thresh = 0.
    try:
        for i, name in enumerate(names):
            tensor, scale = _image_tensor(root, name)
            batch = tensor.unsqueeze(0).to(device)
            gt, ids = _load_gallery_target(root, name)
            result = {k: v.cpu() for k, v in model(batch)[0].items()}
            _, _, pyramid = model._extract_pyramid(batch)
            pooled = model.roi_pooler(pyramid, [(gt * scale).to(device)], [tuple(batch.shape[-2:])])
            gt_embeddings = model.reid_head(pooled).cpu()
            result['boxes'] /= scale
            gallery.append(dict(frame=name, gt=gt, ids=ids, gt_embeddings=gt_embeddings, **result))
            if (i + 1) % 50 == 0:
                print(f'Diagnostics: embedded {i + 1}/{len(names)} validation frames', flush=True)
        occurrences = {}
        for i, entry in enumerate(gallery):
            for j, identity in enumerate(entry['ids'].tolist()):
                if identity >= 0:
                    occurrences.setdefault(identity, []).append((i, j))
        queries = [(identity, i, j) for identity, items in sorted(occurrences.items())
                   if len({i for i, _ in items}) >= 2 for i, j in items]
        # Spread a bounded query sample across identities/frames deterministically.
        if max_queries is not None and len(queries) > max_queries:
            queries = [queries[i] for i in torch.linspace(0, len(queries)-1, max_queries).long().tolist()]
        if not queries:
            raise ValueError('No cross-image query identities in supplied validation frames.')
        query_embeddings = []
        for identity, i, j in queries:
            with Image.open(root / 'frames' / f"{gallery[i]['frame']}.jpg") as image:
                crop = preprocess_query_crop(image.crop(tuple(gallery[i]['gt'][j].tolist())))
            query_embeddings.append(model.embed_crop(crop.unsqueeze(0).to(device))[0].cpu())
    finally:
        model.det_score_thresh = old_threshold
        model.train(was_training)

    summaries, examples = [], {}
    for mode, threshold in [('detected', float(t)) for t in thresholds] + [('gt', 0.)]:
        embeddings, metadata = [], []
        totals = {t: [0, 0, 0] for t in (.5, .75)}
        for i, entry in enumerate(gallery):
            boxes = entry['gt'] if mode == 'gt' else entry['boxes']
            scores = torch.ones(len(boxes)) if mode == 'gt' else entry['scores']
            emb = entry['gt_embeddings'] if mode == 'gt' else entry['embeddings']
            overlaps = box_iou(boxes, entry['gt'])
            for t in totals:
                counts = detection_counts(boxes, scores, entry['gt'], threshold, t)
                totals[t] = [a+b for a,b in zip(totals[t], counts)]
            for j in (scores >= threshold).nonzero(as_tuple=True)[0].tolist():
                embeddings.append(emb[j])
                metadata.append((i, j, boxes[j].tolist(), float(scores[j]), overlaps[j]))
        matrix = torch.stack(embeddings) if embeddings else torch.empty(0, query_embeddings[0].numel())
        aps, top1, saved = [], [], []
        for (identity, source, query_box), query in zip(queries, query_embeddings):
            similarities = matrix @ query
            order = similarities.argsort(descending=True).tolist()
            order = [index for index in order if metadata[index][0] != source]
            total_gt = sum(int((e['ids'] == identity).sum()) for i,e in enumerate(gallery) if i != source)
            used, hits, ap, ranked = set(), 0, 0., []
            for rank, index in enumerate(order, 1):
                frame, _, box, confidence, overlaps = metadata[index]
                correct = False
                for gt_index in overlaps.argsort(descending=True).tolist():
                    if overlaps[gt_index] < .5:
                        break
                    if gallery[frame]['ids'][gt_index] == identity and (frame, gt_index) not in used:
                        used.add((frame, gt_index)); correct = True; break
                if correct:
                    hits += 1
                    ap += hits / rank
                if rank <= 5:
                    ranked.append(dict(rank=rank, frame=gallery[frame]['frame'], box=box,
                                       score=float(similarities[index]), person_confidence=confidence if mode == 'detected' else None,
                                       correct=correct))
            aps.append(ap / total_gt)
            top1.append(bool(ranked and ranked[0]['correct']))
            saved.append(dict(identity=identity, query_frame=gallery[source]['frame'],
                              query_box=gallery[source]['gt'][query_box].tolist(), ranked=ranked))
        row = dict(mode=mode, threshold=threshold if mode == 'detected' else None,
                   queries=len(queries), gallery_frames=len(gallery), candidates=len(metadata),
                   mAP=sum(aps)/len(aps), top1=sum(top1)/len(top1))
        for t, (hits, count, gt_count) in totals.items():
            row[f'precision_iou{int(t*100)}'] = hits/max(count,1)
            row[f'recall_iou{int(t*100)}'] = hits/max(gt_count,1)
        summaries.append(row)
        examples[f'{mode}_{threshold}'] = saved
        print(row, flush=True)
    return summaries, examples
