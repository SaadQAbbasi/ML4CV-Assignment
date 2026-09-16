"""Fixed held-out gallery evaluation for a frozen detector and changing ReID."""
from pathlib import Path
import torch
from torchvision.ops import box_iou
from prw_evaluation import _image_tensor, _load_gallery_target, embed_detected_crops


@torch.inference_mode()
def build_validation_gallery(model, root, frame_names, device, score_threshold=.5):
    """Cache boxes only; every evaluation recomputes embeddings. No test files read."""
    names = list(dict.fromkeys(frame_names))
    gallery = []
    mode, threshold = model.training, model.det_score_thresh
    model.eval()
    model.det_score_thresh = score_threshold
    try:
        for name in names:
            image, scale = _image_tensor(Path(root), name)
            result = model(image[None].to(device))[0]
            gt, ids = _load_gallery_target(Path(root), name)
            gallery.append(dict(frame=name, boxes=result['boxes'].detach().cpu()/scale,
                                gt=gt, ids=ids))
    finally:
        model.det_score_thresh = threshold
        model.train(mode)
    queries = [(i,j) for i,g in enumerate(gallery) for j,identity in enumerate(g['ids'].tolist())
               if identity >= 0 and any(bool((h['ids']==identity).any())
                                        for k,h in enumerate(gallery) if k != i)]
    if not queries:
        raise ValueError('Validation gallery has no cross-frame query identities')
    return dict(gallery=gallery, queries=queries, score_threshold=score_threshold)


def rank_validation(matrix, query_embeddings, gallery, queries, iou_threshold=.5):
    """Same best-IoU assignment, duplicate handling and GT denominator as test."""
    labels, keys, frames = [], [], []
    offset = 0
    for i,g in enumerate(gallery):
        n = len(g['boxes'])
        ids = torch.full((n,), -1, dtype=torch.long)
        match_keys = ids.clone()
        if n and len(g['gt']):
            overlaps, indices = box_iou(g['boxes'],g['gt']).max(1)
            valid = (overlaps >= iou_threshold) & (g['ids'][indices] >= 0)
            ids[valid] = g['ids'][indices[valid]]
            match_keys[valid] = indices[valid]+offset
        labels.extend(ids.tolist()); keys.extend(match_keys.tolist()); frames.extend([i]*n)
        offset += len(g['gt'])
    aps, hits = [], []
    for query,(source,box) in zip(query_embeddings,queries):
        identity = int(gallery[source]['ids'][box])
        total = sum(int((g['ids']==identity).sum()) for i,g in enumerate(gallery) if i!=source)
        order = (matrix @ query).argsort(descending=True, stable=True).tolist()
        order = [j for j in order if frames[j]!=source]
        used, positives, ap, first = set(),0,0.,0
        for rank,j in enumerate(order,1):
            if labels[j]==identity and keys[j] not in used:
                used.add(keys[j]); positives+=1; ap+=positives/rank
                if rank==1:first=1
        aps.append(ap/total); hits.append(first)
    return dict(retrieval_mAP=sum(aps)/len(aps), retrieval_top1=sum(hits)/len(hits),
                retrieval_queries=len(queries), retrieval_gallery_frames=len(gallery),
                retrieval_candidates=len(labels))


@torch.inference_mode()
def evaluate_validation_gallery(model, root, cache, device, crop_batch_size=32):
    mode = model.training
    model.eval()
    features, query_features = [], {}
    try:
        for i,g in enumerate(cache['gallery']):
            path = Path(root)/'frames'/f"{g['frame']}.jpg"
            features.append(embed_detected_crops(model,path,g['boxes'],device,crop_batch_size))
            indices = sorted({j for source,j in cache['queries'] if source==i})
            if indices:
                emb = embed_detected_crops(model,path,g['gt'][indices],device,crop_batch_size)
                query_features.update({(i,j):e for j,e in zip(indices,emb)})
        return rank_validation(torch.cat(features),
                               [query_features[q] for q in cache['queries']],
                               cache['gallery'],cache['queries'])
    finally:
        model.train(mode)
