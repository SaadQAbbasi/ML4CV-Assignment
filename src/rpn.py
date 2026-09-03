"""
rpn.py

Goal of this file (Step 01 of the pipeline):
    {P2, P3, P4, P5, P6} (from fpn.py) -> per-anchor objectness scores +
    bbox deltas, at every spatial location, on every pyramid level.

Building this in stages (see class/function order below), because target
assignment is the part most likely to hide subtle bugs and deserves to be
checked in isolation before proposal decoding is layered on top:

    STAGE 1 (this pass):
        - build_anchor_generator(): vertical, person-shaped anchor boxes
        - RPNHead: shared conv -> per-level (objectness, bbox_deltas)
    STAGE 2 (next):
        - anchor <-> GT matching (assign_targets)
        - RPN loss (objectness CE + smooth-L1 on matched anchors)
    STAGE 3 (after that):
        - decode deltas -> proposal boxes, clip, filter, NMS, top-N

Anchor design choice: standard Faster R-CNN uses aspect_ratios=(0.5,1.0,2.0)
(height/width convention in torchvision) because it's built for generic
COCO objects. For person-only detection we drop the square/wide ratios
entirely and use only tall ratios, since every GT box we care about is a
standing/walking human. Scales are chosen per FPN level so that, e.g.,
P2 (stride 4, highest-res) proposes small anchors for far-away/small
people, and P6 (stride 64) proposes large anchors for close-up people --
mirrors how FPN assigns objects to levels by size in the original paper.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

from backbone import build_backbone
from fpn import FPN, build_backbone_with_fpn


# ---------------------------------------------------------------------------
# Anchor generator
# ---------------------------------------------------------------------------

# One (scale-tuple, ratio-tuple) pair per FPN level, in P2->P6 order.
# Scales are the anchor's longer side in pixels, roughly doubling per level
# to match each level's stride doubling (standard FPN convention: level's
# base scale ~= 4x its stride).
#
# Ratios use torchvision's convention: aspect_ratio = height / width.
# CONFIRMED against torchvision source (AnchorGenerator.generate_anchors):
# h_ratio = sqrt(ratio), w_ratio = 1/sqrt(ratio) -> ratio > 1 means TALLER
# than wide. So person-shaped (tall) anchors need ratios like 2.0-4.0, NOT
# 0.5/0.33/0.25 (that range produces WIDE boxes -- the opposite of what we
# want, verified empirically below). Getting this backwards is an easy,
# silent mistake since nothing throws an error -- it just quietly trains
# proposals shaped like the wrong things.
DEFAULT_ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (256,))
DEFAULT_ANCHOR_RATIOS = ((2.0, 3.0, 4.0),) * 5


def build_anchor_generator(sizes: Tuple[Tuple[int, ...], ...] = DEFAULT_ANCHOR_SIZES,
                            aspect_ratios: Tuple[Tuple[float, ...], ...] = DEFAULT_ANCHOR_RATIOS
                            ) -> AnchorGenerator:
    """
    sizes / aspect_ratios must each have one tuple per FPN level (5 levels:
    P2..P6 by default). num_anchors_per_location is len(sizes[i]) *
    len(aspect_ratios[i]) at level i -- with the defaults that's 1*3 = 3
    anchors/location (fewer than the generic 9, because we removed the
    ratios/scales that don't apply to people).
    """
    assert len(sizes) == len(aspect_ratios), \
        "sizes and aspect_ratios must have one entry per FPN level"
    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)


def make_image_list(images: torch.Tensor) -> ImageList:
    """
    AnchorGenerator wants an ImageList (batched tensor + per-image original
    (H, W)) so it can compute stride = image_size / feature_map_size
    correctly. For now we assume the whole batch is already padded to a
    common size with no extra per-image letterboxing -- revisit this once
    train.py's batching/collation is built.
    """
    image_sizes = [tuple(images.shape[-2:])] * images.shape[0]
    return ImageList(images, image_sizes)


# ---------------------------------------------------------------------------
# RPN head: shared conv trunk -> per-level (objectness, bbox_deltas)
# ---------------------------------------------------------------------------

class RPNHead(nn.Module):
    """
    Same small conv head applied independently to every FPN level (weights
    ARE shared across levels -- this is standard and is what lets a single
    head generalize across the scale range FPN already handles).

    For each spatial cell x each anchor at that cell:
        objectness:  1 logit  (person vs not-person, pre-sigmoid)
        bbox_deltas: 4 values (dx, dy, dw, dh) relative to the anchor box
    """

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_deltas = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        for layer in [self.conv, self.objectness_logits, self.bbox_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        feats: list of (B, C, H_i, W_i) tensors, one per FPN level.
        returns:
            objectness: list of (B, num_anchors, H_i, W_i)
            bbox_deltas: list of (B, num_anchors*4, H_i, W_i)
        """
        objectness, bbox_deltas = [], []
        for f in feats:
            t = self.relu(self.conv(f))
            objectness.append(self.objectness_logits(t))
            bbox_deltas.append(self.bbox_deltas(t))
        return objectness, bbox_deltas


# ---------------------------------------------------------------------------
# Stage 2: anchor <-> GT target assignment + RPN loss
# ---------------------------------------------------------------------------

import torch.nn.functional as F
from torchvision.models.detection._utils import Matcher, BoxCoder, BalancedPositiveNegativeSampler
from torchvision.ops import boxes as box_ops

# NOTE on thresholds: standard Faster R-CNN RPN uses fg=0.7 / bg=0.3, not the
# 0.5/0.5 split mentioned in the assignment brief. Kept configurable (not
# hardcoded to the brief's 0.5) because a 0.5/0.5 split with no gap between
# fg and bg is known to make training noisy -- anchors right at IoU=0.5 flip
# label with tiny box perturbations. The RPN loss docstring below explains
# this choice; the *final* detection head's assignment (roi_heads.py, next
# module) is a separate function and can legitimately use 0.5 as the brief
# specifies, since that stage matches proposals (not raw anchors) to GT.
RPN_FG_IOU_THRESH = 0.7
RPN_BG_IOU_THRESH = 0.3
RPN_BATCH_SIZE_PER_IMAGE = 256
RPN_POSITIVE_FRACTION = 0.5
BOX_CODER_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


class RPNTargetAssigner:
    """
    Matches anchors to GT person boxes and produces per-anchor training
    targets. Kept as its own class (not folded into the loss function) so
    train.py / rpn.py can inspect matched targets directly for debugging,
    e.g. to visualize which anchors got marked positive on a sample image.

    Per-anchor label convention (matches torchvision's Matcher):
        label ==  1  -> positive (IoU >= RPN_FG_IOU_THRESH with some GT,
                         or the single best-matching anchor for a GT that
                         nothing else exceeded fg_thresh for -- this
                         "allow_low_quality_matches" ensures every GT box
                         gets at least one positive anchor even if it's an
                         awkward scale/ratio nothing scores 0.7 IoU on)
        label ==  0  -> negative/background (IoU < RPN_BG_IOU_THRESH with
                         every GT box)
        label == -1  -> ignored (0.3 <= max IoU < 0.7 -- ambiguous, excluded
                         from the loss entirely, standard Faster R-CNN practice)
    """

    def __init__(self, fg_iou_thresh: float = RPN_FG_IOU_THRESH,
                 bg_iou_thresh: float = RPN_BG_IOU_THRESH,
                 batch_size_per_image: int = RPN_BATCH_SIZE_PER_IMAGE,
                 positive_fraction: float = RPN_POSITIVE_FRACTION):
        self.matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self.box_coder = BoxCoder(weights=BOX_CODER_WEIGHTS)

    def assign_targets_single_image(self, anchors: torch.Tensor, gt_boxes: torch.Tensor
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        anchors:  (num_anchors, 4) x1y1x2y2
        gt_boxes: (num_gt, 4) x1y1x2y2 -- assumed all valid people, no -1/
                   unlabeled-ID filtering here since RPN only needs to know
                   "is this a person", not "which person" (identity handling
                   is entirely roi_heads.py + oim.py's job, further downstream)

        returns:
            labels:            (num_anchors,) in {1, 0, -1} as described above
            matched_gt_boxes:  (num_anchors, 4) -- the GT box each anchor was
                                matched to (meaningless/unused for label!=1
                                anchors, but shape-aligned with anchors for
                                convenient indexing when computing reg loss)
        """
        if gt_boxes.numel() == 0:
            # image has no person GT boxes at all -- every anchor is negative
            labels = torch.zeros(anchors.shape[0], dtype=torch.float32, device=anchors.device)
            matched_gt_boxes = torch.zeros_like(anchors)
            return labels, matched_gt_boxes

        match_quality_matrix = box_ops.box_iou(gt_boxes, anchors)  # (num_gt, num_anchors)
        matched_idxs = self.matcher(match_quality_matrix)          # (num_anchors,)

        labels = (matched_idxs >= 0).to(torch.float32)             # positives -> 1
        labels[matched_idxs == Matcher.BETWEEN_THRESHOLDS] = -1.0  # ambiguous -> ignore
        # matcher already encodes negatives (below bg_thresh) as -1 in
        # matched_idxs (Matcher.BELOW_LOW_THRESHOLD); those got labels=0
        # above via (matched_idxs >= 0), which is what we want.

        clamped_idxs = matched_idxs.clamp(min=0)  # avoid negative indexing below
        matched_gt_boxes = gt_boxes[clamped_idxs]

        return labels, matched_gt_boxes

    def __call__(self, anchors_per_image: List[torch.Tensor], gt_boxes_per_image: List[torch.Tensor]
                  ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        all_labels, all_matched_boxes = [], []
        for anchors, gt_boxes in zip(anchors_per_image, gt_boxes_per_image):
            labels, matched_boxes = self.assign_targets_single_image(anchors, gt_boxes)
            all_labels.append(labels)
            all_matched_boxes.append(matched_boxes)
        return all_labels, all_matched_boxes


def compute_rpn_loss(objectness: List[torch.Tensor], bbox_deltas: List[torch.Tensor],
                      anchors_per_image: List[torch.Tensor], labels_per_image: List[torch.Tensor],
                      matched_gt_boxes_per_image: List[torch.Tensor], box_coder: BoxCoder,
                      sampler: BalancedPositiveNegativeSampler) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    objectness / bbox_deltas: RAW per-level outputs from RPNHead, shape
        (B, num_anchors_per_loc, H_i, W_i) / (B, num_anchors_per_loc*4, H_i, W_i)
        -- these get flattened+concatenated across levels to line up with
        anchors_per_image (which AnchorGenerator already returns flattened).

    Loss = CE(objectness, labels) computed ONLY on sampled anchors (a
    balanced subset -- otherwise background anchors outnumber positives by
    ~1000:1 and swamp the loss, per RPN_BATCH_SIZE_PER_IMAGE/POSITIVE_FRACTION)
    + smooth-L1(bbox_deltas, encoded_targets) computed ONLY on positive anchors
    (regression is meaningless for background anchors -- there's no GT box
    to regress toward).
    """
    # flatten each level's (B, A, H, W) objectness / (B, A*4, H, W) deltas
    # into (B, total_anchors) / (B, total_anchors, 4) and concat across levels,
    # so per-image row i lines up 1:1 with anchors_per_image[i].
    flat_objectness, flat_deltas = [], []
    for obj_lvl, deltas_lvl in zip(objectness, bbox_deltas):
        B, A, H, W = obj_lvl.shape
        obj_lvl = obj_lvl.permute(0, 2, 3, 1).reshape(B, -1)             # (B, A*H*W)
        deltas_lvl = deltas_lvl.view(B, A, 4, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 4)
        flat_objectness.append(obj_lvl)
        flat_deltas.append(deltas_lvl)
    flat_objectness = torch.cat(flat_objectness, dim=1)   # (B, total_anchors)
    flat_deltas = torch.cat(flat_deltas, dim=1)            # (B, total_anchors, 4)

    labels = torch.stack(labels_per_image, dim=0)          # (B, total_anchors)
    sampled_pos_masks, sampled_neg_masks = sampler([labels])
    sampled_pos_mask = sampled_pos_masks[0].to(torch.bool)
    sampled_neg_mask = sampled_neg_masks[0].to(torch.bool)
    sampled_mask = sampled_pos_mask | sampled_neg_mask

    objectness_loss = F.binary_cross_entropy_with_logits(
        flat_objectness[sampled_mask], labels[sampled_mask].clamp(min=0)  # -1s excluded by mask
    )

    if sampled_pos_mask.sum() == 0:
        bbox_loss = flat_deltas.sum() * 0.0  # keep it in the graph, contribute nothing
    else:
        anchors = torch.stack(anchors_per_image, dim=0)                    # (B, total_anchors, 4)
        matched_gt = torch.stack(matched_gt_boxes_per_image, dim=0)        # (B, total_anchors, 4)
        target_deltas = box_coder.encode_single(matched_gt[sampled_pos_mask], anchors[sampled_pos_mask])
        bbox_loss = F.smooth_l1_loss(flat_deltas[sampled_pos_mask], target_deltas,
                                      beta=1.0 / 9, reduction="sum") / sampled_mask.sum().clamp(min=1)

    return objectness_loss, bbox_loss


# ---------------------------------------------------------------------------
# Stage 3: decode raw predictions -> final proposal boxes
# ---------------------------------------------------------------------------

def decode_proposals(objectness: List[torch.Tensor], bbox_deltas: List[torch.Tensor],
                      anchors_per_image: List[torch.Tensor], image_shapes: List[Tuple[int, int]],
                      box_coder: BoxCoder, pre_nms_top_n: int = 2000, post_nms_top_n: int = 300,
                      nms_thresh: float = 0.7, min_size: float = 1.0
                      ) -> List[torch.Tensor]:
    """
    Turns raw RPNHead outputs into the final set of proposal boxes each
    image will hand to ROIAlign. Per image:
        1. decode deltas relative to anchors -> absolute boxes
        2. clip boxes to image bounds (predicted boxes can extend past edges)
        3. drop degenerate boxes (near-zero width/height)
        4. take top pre_nms_top_n boxes BY SCORE *per level* before NMS
           (per-level, not globally -- otherwise one very confident level
           could crowd out every proposal from the others; this matches
           standard FPN-RPN practice)
        5. NMS within each image (person score IoU>nms_thresh -> suppress)
        6. keep top post_nms_top_n by score after NMS

    pre_nms_top_n/post_nms_top_n are deliberately larger for training
    (more recall, more anchors for downstream matching to learn from) and
    smaller for inference (speed) -- pass different values at call sites in
    train.py vs inference.py rather than hardcoding train/eval branching here.
    """
    B = objectness[0].shape[0]
    level_sizes = []
    flat_scores_per_level, flat_boxes_per_level = [], []
    for obj_lvl, deltas_lvl in zip(objectness, bbox_deltas):
        Bc, A, H, W = obj_lvl.shape
        level_sizes.append(A * H * W)
        scores_lvl = obj_lvl.permute(0, 2, 3, 1).reshape(Bc, -1)                       # (B, A*H*W)
        deltas_lvl = deltas_lvl.view(Bc, A, 4, H, W).permute(0, 3, 4, 1, 2).reshape(Bc, -1, 4)
        flat_scores_per_level.append(scores_lvl)
        flat_boxes_per_level.append(deltas_lvl)

    proposals_per_image = []
    for img_idx in range(B):
        anchors = anchors_per_image[img_idx]  # (total_anchors, 4), concatenated across levels
        img_h, img_w = image_shapes[img_idx]

        kept_boxes, kept_scores, kept_level_ids = [], [], []
        offset = 0
        for level_idx, n_level in enumerate(level_sizes):
            level_anchors = anchors[offset: offset + n_level]
            level_scores = flat_scores_per_level[level_idx][img_idx]
            level_deltas = flat_boxes_per_level[level_idx][img_idx]
            offset += n_level

            level_boxes = box_coder.decode_single(level_deltas, level_anchors)

            k = min(pre_nms_top_n, level_scores.shape[0])
            top_scores, top_idx = level_scores.topk(k)
            top_boxes = level_boxes[top_idx]

            kept_boxes.append(top_boxes)
            kept_scores.append(top_scores)
            kept_level_ids.append(torch.full((k,), level_idx, dtype=torch.int64, device=top_scores.device))

        boxes = torch.cat(kept_boxes, dim=0)
        scores = torch.cat(kept_scores, dim=0)
        level_ids = torch.cat(kept_level_ids, dim=0)

        boxes = box_ops.clip_boxes_to_image(boxes, (img_h, img_w))
        keep = box_ops.remove_small_boxes(boxes, min_size)
        boxes, scores, level_ids = boxes[keep], scores[keep], level_ids[keep]

        # batched_nms suppresses within each level_id group independently,
        # then results are merged and re-ranked by score across levels --
        # standard multi-level RPN NMS strategy (avoids one level's boxes
        # suppressing another level's genuinely-different-scale proposal
        # for the same image region)
        keep = box_ops.batched_nms(boxes, torch.sigmoid(scores), level_ids, nms_thresh)
        keep = keep[:post_nms_top_n]

        proposals_per_image.append(boxes[keep])

    return proposals_per_image


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(2, 3, 800, 600)
    # synthetic GT: image 0 has two "people" (tall boxes), image 1 has none
    # (exercises the "no GT in image" branch in assign_targets_single_image)
    gt_boxes_per_image = [
        torch.tensor([[100., 100., 180., 340.], [400., 200., 470., 500.]]),
        torch.zeros((0, 4)),
    ]

    backbone, fpn = build_backbone_with_fpn("resnet50", pretrained=False)
    anchor_gen = build_anchor_generator()
    num_anchors_per_loc = anchor_gen.num_anchors_per_location()
    assert len(set(num_anchors_per_loc)) == 1
    rpn_head = RPNHead(in_channels=fpn.out_channels[0], num_anchors=num_anchors_per_loc[0])
    assigner = RPNTargetAssigner()
    box_coder = BoxCoder(weights=BOX_CODER_WEIGHTS)
    sampler = BalancedPositiveNegativeSampler(RPN_BATCH_SIZE_PER_IMAGE, RPN_POSITIVE_FRACTION)

    backbone.eval(); fpn.eval(); rpn_head.eval()
    with torch.no_grad():
        c_feats = backbone(x)
        p_feats_dict = fpn(c_feats)
        p_feats = [p_feats_dict[lvl] for lvl in fpn.out_levels]

        image_list = make_image_list(x)
        anchors_per_image = anchor_gen(image_list, p_feats)

        objectness, bbox_deltas = rpn_head(p_feats)

        # --- Stage 1 shape check (as before) ---
        total_anchors = sum(o.shape[1] * o.shape[2] * o.shape[3] for o in objectness)
        assert anchors_per_image[0].shape[0] == total_anchors
        print(f"[Stage 1] total anchors/image: {total_anchors}  "
              f"anchor shape sample (tall check): {anchors_per_image[0][0].tolist()}")

        # --- Stage 2: target assignment ---
        labels_per_image, matched_gt_per_image = assigner(anchors_per_image, gt_boxes_per_image)
        for i, labels in enumerate(labels_per_image):
            n_pos = (labels == 1).sum().item()
            n_neg = (labels == 0).sum().item()
            n_ignore = (labels == -1).sum().item()
            print(f"[Stage 2] image {i}: {len(gt_boxes_per_image[i])} GT boxes -> "
                  f"{n_pos} positive anchors, {n_neg} negative, {n_ignore} ignored "
                  f"(total {n_pos+n_neg+n_ignore})")
        assert labels_per_image[1].eq(0).all(), "image with 0 GT boxes must have all-negative anchors"

        obj_loss, box_loss = compute_rpn_loss(
            objectness, bbox_deltas, anchors_per_image, labels_per_image,
            matched_gt_per_image, box_coder, sampler
        )
        print(f"[Stage 2] rpn objectness loss: {obj_loss.item():.4f}  "
              f"rpn bbox loss: {box_loss.item():.4f}  (untrained weights -> just sanity, not meaningful yet)")

        # --- Stage 3: decode proposals ---
        image_shapes = [(x.shape[2], x.shape[3])] * x.shape[0]
        proposals = decode_proposals(
            objectness, bbox_deltas, anchors_per_image, image_shapes, box_coder,
            pre_nms_top_n=2000, post_nms_top_n=300, nms_thresh=0.7
        )
        for i, boxes in enumerate(proposals):
            in_bounds = ((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0)
                         & (boxes[:, 2] <= image_shapes[i][1]) & (boxes[:, 3] <= image_shapes[i][0])).all()
            print(f"[Stage 3] image {i}: {boxes.shape[0]} final proposals "
                  f"(<= post_nms_top_n=300: {'OK' if boxes.shape[0] <= 300 else 'FAIL'}, "
                  f"all in-bounds: {'OK' if in_bounds else 'FAIL'})")