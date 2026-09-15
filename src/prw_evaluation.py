"""PRW gallery/query retrieval evaluation for the person-search model."""

from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import wandb
from PIL import Image
from torchvision.ops import box_iou
from torchvision.transforms.functional import normalize, pil_to_tensor, resize

from prw_dataset import _load_mat_table, _mat_strings, preprocess_query_crop


def _load_query_info(path: Path):
    queries = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 6:
            continue
        identity, x, y, w, h, frame = fields
        queries.append({
            "identity": int(identity),
            "box": [float(x), float(y), float(x) + float(w), float(y) + float(h)],
            "frame": frame,
        })
    return queries


def _load_query_crop(path: Path):
    with Image.open(path) as image:
        return preprocess_query_crop(image)


def _load_gallery_target(root: Path, name: str):
    annotation = _load_mat_table(root / "annotations" / f"{name}.jpg.mat")
    boxes = torch.as_tensor(annotation[:, 1:5], dtype=torch.float32)
    boxes[:, 2:] += boxes[:, :2]
    ids = torch.as_tensor([int(row[0]) if int(row[0]) >= 0 else -1 for row in annotation])
    return boxes, ids


def _image_tensor(root: Path, name: str, size=(256, 384), mean=(.485, .456, .406), std=(.229, .224, .225)):
    image = pil_to_tensor(Image.open(root / "frames" / f"{name}.jpg").convert("RGB")).float() / 255.0
    _, height, width = image.shape
    scale = min(size[0] / height, size[1] / width)
    resized_h, resized_w = round(height * scale), round(width * scale)
    image = normalize(resize(image, [resized_h, resized_w]), mean, std)
    padded = image.new_zeros(3, size[0], size[1])
    padded[:, :resized_h, :resized_w] = image
    return padded, scale


@torch.inference_mode()
def embed_detected_crops(model, image_path, boxes, device, batch_size=32):
    """Embed retained boxes in original-image xyxy coordinates, preserving order.

    Uses the exact same original-pixel preprocessing as standalone queries.
    Empty detections do not run the crop backbone.
    """
    if batch_size < 1:
        raise ValueError("crop batch_size must be positive")
    if not len(boxes):
        return torch.empty((0, model.reid_head.embedding_proj.out_features))
    embeddings = []
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        for start in range(0, len(boxes), batch_size):
            crops = torch.stack([
                preprocess_query_crop(image.crop(tuple(box.tolist())))
                for box in boxes[start:start + batch_size].cpu()
            ])
            embeddings.append(model.embed_crop(crops.to(device)).cpu())
    return torch.cat(embeddings)


@torch.inference_mode()
def evaluate_prw(model, root, device, batch_size=2, max_queries=None, max_gallery=None,
                 score_threshold=.5, iou_threshold=.5, return_records=False,
                 table_queries=100, table_top_k=5, query_batch_size=64,
                 ranking_batch_size=8, gallery_crop_batch_size=32, gallery_mode="detected"):


    
    """Evaluate cosine retrieval over PRW test frames.

    gallery_mode='gt' bypasses detection and embeds every annotated gallery
    person crop, including unlabeled distractors. Queries, source-frame exclusion
    and AP denominators are identical to detected mode. GT identities are used
    only for scoring, never supplied to the embedding model.
    Gallery embeddings use original-resolution detected crops after filtering/NMS.
    Query and gallery crops share preprocessing and the same embedding path.

    A gallery detection is relevant when it overlaps a same-ID annotation by
    at least ``iou_threshold``. The query's source frame is excluded.

    batch_size : int
        Number of gallery frames to process at once. Each frame may contain multiple detections.
    max_queries : int or None
        Limit the number of queries to evaluate. None means all queries.
    max_gallery : int or None
        Limit the number of gallery frames to evaluate. None means all frames.
    score_threshold : float or None
        Minimum detection confidence to retain a gallery box. None means use the model's default.
    iou_threshold : float
        Minimum IoU between a gallery detection and a same-ID annotation to count as relevant.
    return_records : bool
        If True, return a list of query records with ranked retrieval results.
    table_queries : int
        Maximum number of query records to save for W&B table and plotting.
    table_top_k : int
        Maximum number of ranked retrievals to save for W&B table and plotting.
    query_batch_size : int
        Number of query crops to embed at once. Each crop is a single person.
    ranking_batch_size : int
        Number of queries to rank against the gallery at once. Each query is a single person.
    gallery_crop_batch_size : int
        Number of detected gallery crops to embed at once. Each crop is a single person.    

    """
    if gallery_mode not in ("detected", "gt"):
        raise ValueError("gallery_mode must be 'detected' or 'gt'")
    root = Path(root)
    split = _load_mat_table(root / "frame_test.mat")
    # frame_test.mat contains a MATLAB string array under this key.
    names = _mat_strings(split) if split.ndim != 0 else []
    if not names:
        from scipy.io import loadmat
        names = _mat_strings(loadmat(root / "frame_test.mat")["img_index_test"])
    if max_gallery is not None:
        names = names[:max_gallery]

    gallery = []
    model.eval()
    threshold = model.det_score_thresh
    if score_threshold is not None:
        model.det_score_thresh = score_threshold
    try:
        gallery_started = perf_counter()
        for start in range(0, len(names), batch_size):
            batch_names = names[start:start + batch_size]
            tensors = []
            targets = []
            scales = []
            for name in batch_names:
                tensor, scale = _image_tensor(root, name)
                tensors.append(tensor)
                scales.append(scale)
                boxes, ids = _load_gallery_target(root, name)
                boxes[:, 0::2] *= scale
                boxes[:, 1::2] *= scale
                targets.append((boxes, ids))
            if gallery_mode == "gt":
                results = [{"boxes": boxes.clone(), "scores": torch.ones(len(boxes))}
                           for boxes, ids in targets]
            else:
                results = model(torch.stack(tensors).to(device))
            for name, result, target, scale in zip(batch_names, results, targets, scales):
                result = {key: value.cpu() for key, value in result.items()}
                result["embeddings"] = embed_detected_crops(
                    model, root / "frames" / f"{name}.jpg",
                    result["boxes"] / scale, device, gallery_crop_batch_size)
                gallery.append({"frame": name, "result": result, "target": target, "scale": scale})
            if len(gallery) % 500 < batch_size:
                print(f"PRW evaluation: embedded {len(gallery)}/{len(names)} gallery frames", flush=True)
    finally:
        model.det_score_thresh = threshold
    print(f"PRW evaluation: gallery embedding finished in "
          f"{perf_counter() - gallery_started:.1f}s", flush=True)

    # Match each detection to GT once. The previous implementation repeated
    # this IoU work for every query, resulting in millions of tiny CPU calls.
    print("PRW evaluation: matching detections to gallery ground truth", flush=True)
    frame_to_index = {entry["frame"]: index for index, entry in enumerate(gallery)}
    gt_counts = Counter()
    frame_gt_boxes = {}
    detection_scores = []
    detection_embeddings = []
    detection_ids = []
    detection_frames = []
    detection_boxes = []
    detection_gt_keys = []
    gt_offset = 0
    for frame_index, entry in enumerate(gallery):
        gt_boxes, gt_ids = entry["target"]
        for identity in gt_ids[gt_ids >= 0].tolist():
            gt_counts[int(identity)] += 1
        for identity in gt_ids[gt_ids >= 0].unique().tolist():
            boxes = gt_boxes[gt_ids == identity] / entry["scale"]
            frame_gt_boxes[(entry["frame"], int(identity))] = boxes

        result = entry["result"]
        if len(result["boxes"]) == 0:
            gt_offset += len(gt_boxes)
            continue
        matched_ids = torch.full((len(result["boxes"]),), -1, dtype=torch.long)
        matched_gt_keys = torch.full((len(result["boxes"]),), -1, dtype=torch.long)
        if gallery_mode == "gt":
            # Direct annotation assignment avoids ambiguous overlapping GT boxes.
            matched_ids = gt_ids.clone()
            known = gt_ids >= 0
            matched_gt_keys[known] = torch.arange(len(gt_ids))[known] + gt_offset
        elif len(gt_boxes):
            overlaps = box_iou(result["boxes"], gt_boxes)
            best_iou, best_gt = overlaps.max(dim=1)
            matched_ids = gt_ids[best_gt].clone()
            valid_match = (best_iou >= iou_threshold) & (matched_ids >= 0)
            matched_gt_keys[valid_match] = best_gt[valid_match] + gt_offset
            matched_ids[~valid_match] = -1
        detection_scores.append(result["scores"])
        detection_embeddings.append(result["embeddings"])
        detection_ids.append(matched_ids)
        detection_frames.append(torch.full((len(result["boxes"]),), frame_index, dtype=torch.long))
        detection_boxes.append(result["boxes"] / entry["scale"])
        detection_gt_keys.append(matched_gt_keys)
        gt_offset += len(gt_boxes)

    if detection_embeddings:
        detection_scores = torch.cat(detection_scores)
        detection_embeddings = torch.cat(detection_embeddings)
        detection_ids = torch.cat(detection_ids)
        detection_frames = torch.cat(detection_frames)
        detection_boxes = torch.cat(detection_boxes)
        detection_gt_keys = torch.cat(detection_gt_keys)
    else:
        embedding_dim = model.reid_head.embedding_proj.out_features
        detection_embeddings = torch.empty((0, embedding_dim))
        detection_ids = torch.empty(0, dtype=torch.long)
        detection_frames = torch.empty(0, dtype=torch.long)
        detection_boxes = torch.empty((0, 4))
        detection_gt_keys = torch.empty(0, dtype=torch.long)

    queries = _load_query_info(root / "query_info.txt")
    if max_queries is not None:
        queries = queries[:max_queries]
    query_items = []
    for query in queries:
        crop_path = root / "query_box" / f"{query['identity']}_{query['frame']}.jpg"
        if crop_path.exists():
            query_items.append((query, crop_path, _load_query_crop(crop_path)))

    query_embeddings = []
    query_started = perf_counter()
    for start in range(0, len(query_items), query_batch_size):
        batch_items = query_items[start:start + query_batch_size]
        crops = torch.stack([item[2] for item in batch_items]).to(device)
        embeddings = model.embed_crop(crops).cpu()
        query_embeddings.extend(embeddings.unbind())
        print(f"PRW evaluation: embedded {len(query_embeddings)}/{len(query_items)} queries", flush=True)
    print(f"PRW evaluation: query embedding finished in "
          f"{perf_counter() - query_started:.1f}s", flush=True)

    average_precisions = []
    top1_hits = []
    records = []
    if len(detection_embeddings):
        gallery_embeddings_device = detection_embeddings.to(device)
        detection_ids_device = detection_ids.to(device)
        detection_frames_device = detection_frames.to(device)
        detection_gt_keys_device = detection_gt_keys.to(device)
        ranking_started = perf_counter()
        print(f"PRW evaluation: ranking {len(query_items)} queries against "
              f"{len(detection_embeddings)} detections", flush=True)
        for start in range(0, len(query_items), ranking_batch_size):
            end = min(start + ranking_batch_size, len(query_items))
            query_batch = torch.stack(query_embeddings[start:end]).to(device)
            score_batch = query_batch @ gallery_embeddings_device.T
            for offset, (query, crop_path, _) in enumerate(query_items[start:end]):
                identity = query["identity"]
                source_frame_index = frame_to_index.get(query["frame"], -1)
                valid = detection_frames_device != source_frame_index
                scores = score_batch[offset].masked_fill(~valid, float("-inf"))
                order = torch.argsort(scores, descending=True)
                relevant_candidates = (detection_ids_device[order] == identity) & valid[order]
                total_gt_relevant = gt_counts[identity] - len(
                    frame_gt_boxes.get((query["frame"], identity), ()))
                candidate_ranks = relevant_candidates.nonzero(as_tuple=True)[0]
                candidate_gt_keys = detection_gt_keys_device[order[candidate_ranks]].cpu().tolist()
                # Multiple proposals may overlap one person. Only the highest-ranked
                # proposal is a true positive; the rest remain false positives.
                seen_gt = set()
                positive_rank_indices = []
                for candidate_rank, gt_key in zip(candidate_ranks.cpu().tolist(), candidate_gt_keys):
                    if gt_key >= 0 and gt_key not in seen_gt:
                        seen_gt.add(gt_key)
                        positive_rank_indices.append(candidate_rank)
                positive_ranks = torch.as_tensor(
                    positive_rank_indices, device=device, dtype=torch.long) + 1
                if total_gt_relevant > 0:
                    hit_numbers = torch.arange(1, len(positive_ranks) + 1, device=device)
                    average_precisions.append(float((hit_numbers / positive_ranks).sum() / total_gt_relevant))
                    top1_hits.append(int(bool(len(positive_ranks) and positive_ranks[0] == 1)))

                if return_records and len(records) < table_queries:
                    positive_rank_set = set(positive_rank_indices)
                    ranked = []
                    valid_order = order[valid[order]]
                    for rank, detection_index in enumerate(valid_order[:table_top_k].cpu().tolist(), start=1):
                        frame = gallery[detection_frames[detection_index].item()]["frame"]
                        gt_boxes = frame_gt_boxes.get((frame, identity), torch.empty((0, 4)))
                        ranked.append({
                            "rank": rank, "score": float(scores[detection_index]),
                            "person_confidence": float(detection_scores[detection_index]) if gallery_mode == "detected" else None,
                            "correct": rank - 1 in positive_rank_set,
                            "frame": frame, "box": detection_boxes[detection_index].tolist(),
                            "gt_count": len(gt_boxes), "gt_boxes": gt_boxes.tolist(),
                        })
                    records.append({
                        "identity": identity, "frame": query["frame"],
                        "query_path": str(crop_path),
                        "ground_truth_count": total_gt_relevant, "ranked": ranked,
                    })
            print(f"PRW evaluation: ranked {end}/{len(query_items)} queries", flush=True)
        print(f"PRW evaluation: ranking finished in "
              f"{perf_counter() - ranking_started:.1f}s", flush=True)

    # Include queries with relevant GT even when the detector returns no boxes.
    if not len(detection_embeddings):
        for query, _, _ in query_items:
            total = gt_counts[query["identity"]] - len(
                frame_gt_boxes.get((query["frame"], query["identity"]), ()))
            if total > 0:
                average_precisions.append(0.)
                top1_hits.append(0)

    detector_metrics = {}
    if gallery_mode == "detected":
        from retrieval_diagnostics import detection_counts
        for overlap_threshold in (.5, .75):
            hits = predicted = annotated = 0
            for entry in gallery:
                h, n, g = detection_counts(entry["result"]["boxes"],
                    entry["result"]["scores"], entry["target"][0],
                    float("-inf"), overlap_threshold)
                hits += h
                predicted += n
                annotated += g
            suffix = int(overlap_threshold * 100)
            detector_metrics[f"detection_precision_iou{suffix}"] = hits / max(predicted, 1)
            detector_metrics[f"detection_recall_iou{suffix}"] = hits / max(annotated, 1)

    metrics = {
        **detector_metrics,
        "gallery_mode": gallery_mode,
        "retrieval_mAP": float(np.mean(average_precisions)) if average_precisions else 0.0,
        "retrieval_top1": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "retrieval_queries": len(average_precisions),
        "retrieval_gallery_frames": len(gallery),
        "retrieval_detections": sum(len(e["result"]["boxes"]) for e in gallery),
    }
    if return_records:
        metrics["retrieval_records"] = records
    return metrics


def plot_retrieval_query(records, root, query_index=0, top_k=5):
    """Show a query crop, ranked detection crops, and their gallery locations.

    ``query_index`` selects a saved retrieval record; no model inference is needed.
    Green/red indicates evaluation correctness, not a model confidence threshold.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not records:
        raise ValueError("No retrieval records available. Evaluate with return_records=True.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    record = records[query_index]
    ranked = record["ranked"][:top_k]
    fig, axes = plt.subplots(2, len(ranked) + 1, squeeze=False,
                             figsize=(3.5 * (len(ranked) + 1), 7))
    for ax in axes.flat:
        ax.axis("off")
    with Image.open(record["query_path"]) as source:
        axes[0, 0].imshow(source.convert("RGB"))
    axes[0, 0].set_title(f"Query ID {record['identity']}\n{record['frame']}")
    axes[1, 0].text(0.5, 0.5,
                    "Top matches by cosine similarity\nGreen: correct match\nRed: incorrect match"
                    if ranked else "No gallery detections available",
                    ha="center", va="center", transform=axes[1, 0].transAxes)
    for col, item in enumerate(ranked, start=1):
        with Image.open(Path(root) / "frames" / f"{item['frame']}.jpg") as source:
            frame = source.convert("RGB")
        color = "forestgreen" if item["correct"] else "crimson"
        axes[0, col].imshow(frame.crop(tuple(item["box"])))
        axes[0, col].set_title(
            f"Rank {item['rank']} | similarity {item['score']:.3f}\n"
            f"{'Correct' if item['correct'] else 'Incorrect'}", color=color)
        axes[1, col].imshow(frame)
        x1, y1, x2, y2 = item["box"]
        axes[1, col].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=color, linewidth=2))
        axes[1, col].set_title(item["frame"])
    fig.tight_layout()
    return fig


def make_wandb_retrieval_table(records, root):
    """Create one W&B row per query, with saved predictions side by side."""
    root = Path(root)
    max_predictions = max((len(record["ranked"]) for record in records), default=0)
    columns = ["query_id", "query_frame", "query", "same_id_gt_total"]
    prediction_fields = ["predicted_box", "similarity", "person_confidence", "correct", "gallery_frame",
                         "gallery", "box_xyxy", "same_id_gt_in_frame"]
    for rank in range(1, max_predictions + 1):
        columns.extend(f"top_{rank}_{field}" for field in prediction_fields)
    table = wandb.Table(columns=columns)
    rows_added = 0
    total_rows = len(records)
    print(f"W&B retrieval table: preparing {total_rows} rows", flush=True)
    for record in records:
        query_image = wandb.Image(record["query_path"])
        row = [record["identity"], record["frame"], query_image,
               record["ground_truth_count"]]
        for item in record["ranked"]:
            box = item["box"]
            prediction_data = [{
                "position": {"minX": box[0], "minY": box[1],
                             "maxX": box[2], "maxY": box[3]},
                "class_id": 1, "domain": "pixel",
                "scores": {"similarity": item["score"]},
            }]
            ground_truth_data = [{
                "position": {"minX": gt[0], "minY": gt[1],
                             "maxX": gt[2], "maxY": gt[3]},
                "class_id": 1, "domain": "pixel",
            } for gt in item["gt_boxes"]]
            gallery_image = wandb.Image(str(root / "frames" / f"{item['frame']}.jpg"), boxes={
                "predictions": {
                    "box_data": prediction_data,
                    "class_labels": {1: "retrieved person"},
                },
                "ground_truth": {
                    "box_data": ground_truth_data,
                    "class_labels": {1: "person"},
                }
            })
            with Image.open(root / "frames" / f"{item['frame']}.jpg") as source:
                predicted_crop = wandb.Image(source.convert("RGB").crop(tuple(box)),
                                            caption=f"Rank {item['rank']} | similarity {item['score']:.3f}")
            row.extend([predicted_crop, item["score"], item.get("person_confidence"), item["correct"], item["frame"],
                        gallery_image, str(box), item["gt_count"]])
        row.extend([None] * (len(columns) - len(row)))
        table.add_data(*row)
        rows_added += 1
        if rows_added % 50 == 0 or rows_added == total_rows:
            print(f"W&B retrieval table: prepared {rows_added}/{total_rows} rows", flush=True)
    return table
