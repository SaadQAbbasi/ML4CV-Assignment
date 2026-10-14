"""Adapter between this project's evaluation data and the official ``eval_search_prw``.

``eval_function.eval_search_prw`` (provided with the assignment, derived from
SeqNet) expects SeqNet-style dataset objects and per-image numpy arrays. This
module converts the gallery detections, ground truth and query embeddings that
``prw_evaluation.evaluate_prw`` already computes into exactly that format, so
the reported mAP / top-1 come from the official function without modifying it.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from eval_function import eval_search_prw


@dataclass
class SearchDataset:
    """Minimal stand-in for a SeqNet dataset: ``eval_search_prw`` only reads these fields."""
    annotations: list
    img_prefix: str = ""

    def __len__(self):
        return len(self.annotations)


@dataclass
class OfficialInputs:
    """All positional inputs of ``eval_search_prw`` for one evaluation."""
    gallery_dataset: SearchDataset
    query_dataset: SearchDataset
    gallery_dets: list = field(default_factory=list)
    gallery_feats: list = field(default_factory=list)
    query_box_feats: list = field(default_factory=list)


def camera_id(frame_name):
    """PRW frame names start with the camera, e.g. ``c3s1_155583`` -> 3 (-1 if absent).

    Only used when ``ignore_cam_id=False``; the PRW protocol ignores cameras.
    """
    match = re.match(r"c(\d+)", frame_name)
    return int(match.group(1)) if match else -1


def _unit_rows(features):
    """Float32 numpy matrix with L2-normalized rows (cosine similarity = dot product)."""
    features = torch.as_tensor(features, dtype=torch.float32)
    if features.ndim == 1:
        features = features[None]
    return F.normalize(features, dim=1).numpy() if len(features) else features.numpy()


def build_official_inputs(gallery, query_items, query_embeddings, image_root=""):
    """Convert ``evaluate_prw`` intermediate results into ``eval_search_prw`` inputs.

    Parameters
    ----------
    gallery : list of dict
        One entry per gallery frame with keys ``frame``, ``scale``, ``target``
        (GT boxes in resized coordinates, identities with -1 for unknown people) and
        ``result`` (``boxes`` in resized coordinates, ``scores``, ``embeddings``).
    query_items : list of tuple
        ``(query, crop_path, crop)`` where ``query`` has ``identity``, ``frame`` and
        ``box`` (original-image xyxy).
    query_embeddings : list of torch.Tensor
        One embedding per query item, in the same order.
    image_root : str or Path
        Frame directory, only stored as ``img_prefix`` in the official result.

    All boxes are returned in original-image xyxy coordinates.
    """
    gallery_annotations, gallery_dets, gallery_feats = [], [], []
    for entry in gallery:
        gt_boxes, gt_ids = entry["target"]
        result = entry["result"]
        gallery_annotations.append({
            "img_name": entry["frame"],
            "boxes": (gt_boxes / entry["scale"]).numpy().astype(np.float32),
            "pids": gt_ids.numpy().astype(np.int64),
            "cam_id": camera_id(entry["frame"]),
        })
        boxes = (result["boxes"] / entry["scale"]).float()
        scores = result["scores"].float().reshape(-1, 1)
        gallery_dets.append(torch.cat([boxes, scores], dim=1).numpy())
        gallery_feats.append(_unit_rows(result["embeddings"]))

    query_annotations = [{
        "img_name": query["frame"],
        "boxes": np.asarray([query["box"]], dtype=np.float32),
        "pids": int(query["identity"]),
        "cam_id": camera_id(query["frame"]),
    } for query, _, _ in query_items]
    query_feats = [_unit_rows(embedding)[0] for embedding in query_embeddings]

    return OfficialInputs(SearchDataset(gallery_annotations, str(image_root)),
                          SearchDataset(query_annotations, str(image_root)),
                          gallery_dets, gallery_feats, query_feats)


def official_search_metrics(gallery, query_items, query_embeddings, det_thresh, image_root=""):
    """Run the official ``eval_search_prw`` and return its mAP and top-1.

    Queries whose identity has no other occurrence in the evaluated gallery are
    dropped first: the official function divides by that count. On the full PRW
    test split every query has at least one other occurrence, so nothing is dropped.
    """
    frames_by_identity = {}
    for entry in gallery:
        for identity in set(entry["target"][1].tolist()):
            frames_by_identity.setdefault(identity, set()).add(entry["frame"])
    kept = [index for index, (query, _, _) in enumerate(query_items)
            if frames_by_identity.get(query["identity"], set()) - {query["frame"]}]
    inputs = build_official_inputs(gallery, [query_items[i] for i in kept],
                                   [query_embeddings[i] for i in kept], image_root)
    result = eval_search_prw(inputs.gallery_dataset, inputs.query_dataset,
                             inputs.gallery_dets, inputs.gallery_feats,
                             inputs.query_box_feats, det_thresh=det_thresh)
    return {
        "official_mAP": float(result["mAP"]),
        "official_top1": float(result["accs"][0]),
        "official_queries": len(kept),
    }
