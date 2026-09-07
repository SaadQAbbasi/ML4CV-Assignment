"""PRW gallery/query retrieval evaluation for the person-search model."""

from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import Image
from torchvision.ops import box_iou
from torchvision.transforms.functional import normalize, pil_to_tensor, resize

from prw_dataset import _load_mat_table, _mat_strings


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


def _load_query_crop(path: Path, size=(128, 64), mean=(.485, .456, .406), std=(.229, .224, .225)):
    image = pil_to_tensor(Image.open(path).convert("RGB")).float() / 255.0
    return normalize(resize(image, list(size)), mean, std)


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
def evaluate_prw(model, root, device, batch_size=2, max_queries=None, max_gallery=None,
                 score_threshold=None, iou_threshold=.5, return_records=False,
                 table_queries=100, table_top_k=5):
    """Evaluate cosine retrieval over PRW test frames.

    A gallery detection is relevant when it overlaps a same-ID annotation by
    at least ``iou_threshold``. The query's source frame is excluded.
    """
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
            results = model(torch.stack(tensors).to(device))
            for name, result, target, scale in zip(batch_names, results, targets, scales):
                result = {key: value.cpu() for key, value in result.items()}
                gallery.append({"frame": name, "result": result, "target": target, "scale": scale})
            if len(gallery) % 500 < batch_size:
                print(f"PRW evaluation: embedded {len(gallery)}/{len(names)} gallery frames", flush=True)
    finally:
        model.det_score_thresh = threshold

    queries = _load_query_info(root / "query_info.txt")
    if max_queries is not None:
        queries = queries[:max_queries]
    embeddings = []
    for query in queries:
        crop_path = root / "query_box" / f"{query['identity']}_{query['frame']}.jpg"
        if not crop_path.exists():
            continue
        crop = _load_query_crop(crop_path).unsqueeze(0).to(device)
        embeddings.append((query, model.embed_crop(crop)[0].cpu()))
        if len(embeddings) % 250 == 0:
            print(f"PRW evaluation: embedded {len(embeddings)}/{len(queries)} queries", flush=True)

    average_precisions = []
    top1_hits = []
    records = []
    for query, query_embedding in embeddings:
        candidates = []
        total_gt_relevant = 0
        for entry in gallery:
            if entry["frame"] == query["frame"]:
                continue
            result = entry["result"]
            gallery_boxes, gallery_ids = entry["target"]
            same_id_boxes = gallery_boxes[gallery_ids == query["identity"]]
            gt_relevant = len(same_id_boxes)
            total_gt_relevant += gt_relevant
            if len(result["boxes"]) == 0:
                continue
            similarities = result["embeddings"] @ query_embedding
            overlaps = box_iou(gallery_boxes, result["boxes"])
            relevant = torch.zeros(len(result["boxes"]), dtype=torch.bool)
            if gt_relevant:
                relevant = ((gallery_ids == query["identity"]).float()[:, None] *
                            (overlaps >= iou_threshold)).any(dim=0)
            for score, is_relevant, box in zip(similarities.cpu().tolist(), relevant.tolist(), result["boxes"]):
                candidates.append((score, bool(is_relevant), box / entry["scale"], entry["frame"],
                                   gt_relevant, same_id_boxes / entry["scale"]))
        candidates.sort(key=lambda item: item[0], reverse=True)
        positives = [int(item[1]) for item in candidates]
        # Normalize AP by relevant ground-truth instances, so missed people
        # reduce AP instead of disappearing from the denominator.
        num_positive = total_gt_relevant
        if return_records and len(records) < table_queries:
            records.append({
                "identity": query["identity"],
                "frame": query["frame"],
                "query_path": str(root / "query_box" / f"{query['identity']}_{query['frame']}.jpg"),
                "ground_truth_count": total_gt_relevant,
                "ranked": [
                    {"rank": rank, "score": score, "correct": is_relevant,
                     "frame": frame, "box": box.tolist(), "gt_count": gt_count,
                     "gt_boxes": gt_boxes.tolist()}
                    for rank, (score, is_relevant, box, frame, gt_count, gt_boxes)
                    in enumerate(candidates[:table_top_k], start=1)
                ],
            })
        if num_positive == 0:
            continue
        hits = 0
        precision_sum = 0.0
        for rank, is_positive in enumerate(positives, start=1):
            if is_positive:
                hits += 1
                precision_sum += hits / rank
        average_precisions.append(precision_sum / num_positive)
        top1_hits.append(positives[0] if positives else 0)

    metrics = {
        "retrieval_mAP": float(np.mean(average_precisions)) if average_precisions else 0.0,
        "retrieval_top1": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "retrieval_queries": len(average_precisions),
        "retrieval_gallery_frames": len(gallery),
        "retrieval_detections": sum(len(e["result"]["boxes"]) for e in gallery),
    }
    if return_records:
        metrics["retrieval_records"] = records
    return metrics


def make_wandb_retrieval_table(records, root):
    """Create an interactive W&B table for inspecting ranked detections."""
    root = Path(root)
    table = wandb.Table(columns=[
        "query_id", "query_frame", "query", "rank", "similarity",
        "gallery_frame", "gallery", "predicted_box", "same_id_gt_in_frame",
        "same_id_gt_total", "correct",
    ])
    for record in records:
        query_image = wandb.Image(record["query_path"])
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
            table.add_data(record["identity"], record["frame"], query_image,
                           item["rank"], item["score"], item["frame"], gallery_image,
                           box, item["gt_count"], record["ground_truth_count"], item["correct"])
    return table
