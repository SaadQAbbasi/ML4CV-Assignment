"""
roi_heads.py

Goal of this file (Steps 03-04 of the pipeline):
    RPN proposals + {P2,P3,P4,P5} -> ROIAlign -> per-proposal embeddings
    -> detection head -> refined person/bg score + bbox -> final NMS

This is the second detection stage: RPN's job (rpn.py) was "is there
probably a person somewhere near this anchor" using a shared, coarse head
applied identically everywhere. This stage's job is "look closely at each
specific proposed region and decide for real" -- a proposal-specific FC head
looking at pooled, fixed-size features, which is strictly more accurate.

Design notes:
    - MultiScaleRoIAlign (torchvision) is reused rather than hand-rolling
      the FPN-level-assignment formula (Eq. 1 in the FPN paper: which
      pyramid level a given proposal should be pooled from, based on the
      proposal's area) -- same "reuse well-tested geometry, write the
      person-search-specific logic ourselves" philosophy as rpn.py.
    - Only P2-P5 are used for ROI pooling, NOT P6. P6 exists purely to give
      RPN one extra coarse anchor scale for very large objects; standard
      Faster-RCNN-FPN/Mask-RCNN practice excludes it from the box head.
    - DetectionHead and ReIDHead are TWO SEPARATE modules, each with their
      own FC trunk, both reading the same pooled RoIAlign output
      independently and in parallel (matches the pipeline diagram exactly:
      RoI Align feeds two distinct boxes, "Detection Head" and "ReID head",
      not one head with a bolted-on embedding branch). This also happens to
      be the real fix for a tension flagged during pipeline design:
      detection wants "all people look similar" (compact vs. background)
      while re-ID wants "different identities are separable" -- sharing a
      trunk between both objectives fights itself; separate trunks don't.
    - Both anchor and proposal matching default to IoU >= 0.5.
    - PersonSearchModel never appends GT boxes to the proposal set. GT boxes
      supply matching/regression targets and identity labels only.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.models.detection._utils import Matcher, BoxCoder, BalancedPositiveNegativeSampler

from fpn import build_backbone_with_fpn
from rpn import (build_anchor_generator, RPNHead, RPNTargetAssigner, compute_rpn_loss,
                  decode_proposals, make_image_list, BOX_CODER_WEIGHTS,
                  RPN_BATCH_SIZE_PER_IMAGE, RPN_POSITIVE_FRACTION)

# Single-threshold assignment, matches the assignment brief ("IoU > 0.5").
ROI_FG_IOU_THRESH = 0.7
ROI_BG_IOU_THRESH = 0.3
ROI_BATCH_SIZE_PER_IMAGE = 512
ROI_POSITIVE_FRACTION = 0.25   # fewer positives needed here than at RPN --
                                # proposals are already pre-filtered by RPN,
                                # so positives are much easier to find per image
ROI_ALIGN_OUTPUT_SIZE = 7      # standard Faster R-CNN choice (7x7 pooled grid)


# ---------------------------------------------------------------------------
# ROI pooling: proposals -> fixed-size per-proposal feature maps
# ---------------------------------------------------------------------------

def build_roi_pooler(featmap_names: List[str] = ("P2", "P3", "P4", "P5"),
                      output_size: int = ROI_ALIGN_OUTPUT_SIZE,
                      sampling_ratio: int = 2) -> MultiScaleRoIAlign:
    """
    featmap_names must match the keys of the dict passed into pooler.forward
    (i.e. the P-level names from fpn.py), MINUS P6 (see module docstring).
    """
    return MultiScaleRoIAlign(featmap_names=list(featmap_names),
                               output_size=output_size, sampling_ratio=sampling_ratio)


# ---------------------------------------------------------------------------
# Detection head: pooled features -> person/bg score + bbox refinement
# ---------------------------------------------------------------------------

class DetectionHead(nn.Module):
    """
    Matches the diagram's "Detection Head" box exactly: RoI-Align features
    -> class (person/bg) + bbox refinement. Nothing else -- no embedding
    branch here (see ReIDHead below, a fully separate module).

    Two FC layers over the flattened ROI-Align output (standard Faster
    R-CNN box head), then two small linear heads:
        cls_score:  (N, 2) logits, background vs person
        bbox_pred:  (N, 4) deltas refining the proposal box
    """

    def __init__(self, in_channels: int, roi_output_size: int = ROI_ALIGN_OUTPUT_SIZE,
                 representation_size: int = 1024):
        super().__init__()
        pooled_dim = in_channels * roi_output_size * roi_output_size
        self.fc6 = nn.Linear(pooled_dim, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.cls_score = nn.Linear(representation_size, 2)          # bg, person
        self.bbox_pred = nn.Linear(representation_size, 4)

        for layer in [self.fc6, self.fc7]:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, pooled_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pooled_feats: (N, C, roi_output_size, roi_output_size) from ROIAlign,
                      N = total proposals across the whole batch (MultiScale-
                      RoIAlign returns them concatenated, not per-image)
        returns: cls_logits (N,2), bbox_deltas (N,4)
        """
        x = pooled_feats.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return self.cls_score(x), self.bbox_pred(x)


class ReIDHead(nn.Module):
    """
    Matches the diagram's "ReID head" box exactly: the SAME RoI-Align
    features feed this head independently, in parallel with DetectionHead
    -- not sequentially through it. This is a deliberate architectural
    split, not just a naming change: detection wants "all people look
    similar" (compact vs. background), re-ID wants "different identities
    are separable" (spread out). Giving each its own FC trunk (rather than
    sharing fc6/fc7 and only branching at the last layer, which is what an
    earlier version of this file did) removes that tension at the
    representation level, matching what the diagram actually specifies.

    Output feeds Step 05's metric loss (metric_loss.py) -- L2-normalized,
    since that's what BatchHardTripletLoss/ContrastiveLoss expect.
    """

    def __init__(self, in_channels: int, roi_output_size: int = ROI_ALIGN_OUTPUT_SIZE,
                 representation_size: int = 1024, embedding_dim: int = 256):
        super().__init__()
        pooled_dim = in_channels * roi_output_size * roi_output_size
        self.fc6 = nn.Linear(pooled_dim, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.embedding_proj = nn.Linear(representation_size, embedding_dim)

        for layer in [self.fc6, self.fc7]:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.embedding_proj.weight, std=0.01)
        nn.init.constant_(self.embedding_proj.bias, 0)

    def forward(self, pooled_feats: torch.Tensor) -> torch.Tensor:
        """
        pooled_feats: same (N, C, roi_output_size, roi_output_size) input as
                      DetectionHead gets -- both heads read the SAME RoIAlign
                      output, per the diagram, not one feeding the other.
        returns: embeddings (N, embedding_dim), L2-normalized
        """
        x = pooled_feats.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return F.normalize(self.embedding_proj(x), dim=1)  # L2-normalized, metric loss expects this


# ---------------------------------------------------------------------------
# Target assignment for the detection head (0.5 IoU threshold, per brief)
# ---------------------------------------------------------------------------

class ROITargetAssigner:
    """
    Matches proposals to GT person boxes for the detection head's loss.
    Structurally similar to RPNTargetAssigner but deliberately a SEPARATE
    class (not shared code) because the threshold semantics differ: RPN
    ignores an ambiguous band, this stage does not (see module docstring).
    """

    def __init__(self, fg_iou_thresh: float = ROI_FG_IOU_THRESH,
                 bg_iou_thresh: float = ROI_BG_IOU_THRESH,
                 batch_size_per_image: int = ROI_BATCH_SIZE_PER_IMAGE,
                 positive_fraction: float = ROI_POSITIVE_FRACTION):
        # Low-IoU proposals remain background, with no forced GT matches.
        self.matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self.box_coder = BoxCoder(weights=BOX_CODER_WEIGHTS)

    @staticmethod
    def add_gt_proposals(proposals_per_image: List[torch.Tensor],
                          gt_boxes_per_image: List[torch.Tensor]) -> List[torch.Tensor]:
        return [torch.cat([props, gt]) if gt.numel() else props
                for props, gt in zip(proposals_per_image, gt_boxes_per_image)]

    def assign_and_sample(self, proposals_per_image: List[torch.Tensor],
                           gt_boxes_per_image: List[torch.Tensor],
                           gt_ids_per_image: List[torch.Tensor]
                           ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
                                      List[torch.Tensor], List[torch.Tensor]]:
        """
        For each image, matches proposals to GT, subsamples a balanced set
        (ROI_BATCH_SIZE_PER_IMAGE, ROI_POSITIVE_FRACTION positive), and
        returns only the SAMPLED subset -- unlike RPN (which computes loss
        over all anchors but only backprops through a sampled mask), the
        detection head only ever runs ROIAlign+FC on the sampled proposals,
        since running the FC head over thousands of proposals per image
        every step would be wasteful.

        Returns (all lists, one entry per image, already subsampled):
            sampled_proposals:   (n_sampled, 4)
            cls_labels:          (n_sampled,) in {0,1} -- 0=bg, 1=person
            matched_gt_boxes:    (n_sampled, 4) -- meaningless for bg entries
            matched_gt_ids:      (n_sampled,) -- person identity label for
                                  positives; -1 for background (Step 05/OIM
                                  reads this to know which entries to skip
                                  entirely vs. route to the unlabeled queue)
        """
        sampled_proposals, cls_labels, matched_gt_boxes, matched_gt_ids = [], [], [], []

        for proposals, gt_boxes, gt_ids in zip(proposals_per_image, gt_boxes_per_image, gt_ids_per_image):
            if gt_boxes.numel() == 0 or proposals.numel() == 0:
                labels = torch.zeros(proposals.shape[0], dtype=torch.int64, device=proposals.device)
                matched_boxes = torch.zeros_like(proposals)
                matched_ids = torch.full((proposals.shape[0],), -1, dtype=torch.int64, device=proposals.device)
            else:
                iou = box_ops.box_iou(gt_boxes, proposals)   # (num_gt, num_proposals)
                matched_idxs = self.matcher(iou)
                labels = (matched_idxs >= 0).to(torch.int64)  # bg(-1)->0, matched->1
                labels[matched_idxs == Matcher.BETWEEN_THRESHOLDS] = -1  # ignore ambiguous IoU
                clamped = matched_idxs.clamp(min=0)
                matched_boxes = gt_boxes[clamped]
                matched_ids = gt_ids[clamped]
                matched_ids = torch.where(labels == 1, matched_ids,
                                           torch.full_like(matched_ids, -1))

            pos_mask, neg_mask = self.sampler([labels])
            keep = (pos_mask[0] | neg_mask[0]).nonzero(as_tuple=True)[0]

            sampled_proposals.append(proposals[keep])
            cls_labels.append(labels[keep])
            matched_gt_boxes.append(matched_boxes[keep])
            matched_gt_ids.append(matched_ids[keep])

        return sampled_proposals, cls_labels, matched_gt_boxes, matched_gt_ids


def compute_detection_loss(cls_logits: torch.Tensor, bbox_deltas: torch.Tensor,
                            cls_labels: torch.Tensor, matched_gt_boxes: torch.Tensor,
                            proposals: torch.Tensor, box_coder: BoxCoder
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    All inputs already flattened/concatenated across the batch (N = total
    sampled proposals across all images -- matches how MultiScaleRoIAlign
    and the DetectionHead process proposals: as one flat batch, not
    per-image loops, since the FC head has no notion of "which image").
    """
    cls_loss = (F.cross_entropy(cls_logits, cls_labels) if cls_labels.numel()
                else cls_logits.sum() * 0.0)

    pos_mask = cls_labels == 1
    if pos_mask.sum() == 0:
        bbox_loss = bbox_deltas.sum() * 0.0
    else:
        target_deltas = box_coder.encode_single(matched_gt_boxes[pos_mask], proposals[pos_mask])
        bbox_loss = F.smooth_l1_loss(bbox_deltas[pos_mask], target_deltas,
                                      beta=1.0 / 9, reduction="sum") / cls_labels.numel()
    return cls_loss, bbox_loss


# ---------------------------------------------------------------------------
# Inference-time: decode detection-head outputs -> final boxes, second NMS
# ---------------------------------------------------------------------------

def postprocess_detections(cls_logits: torch.Tensor, bbox_deltas: torch.Tensor,
                            embeddings: torch.Tensor, proposals_per_image: List[torch.Tensor],
                            image_shapes: List[Tuple[int, int]], box_coder: BoxCoder,
                            score_thresh: float = 0.5, nms_thresh: float = 0.4,
                            max_detections_per_image: int = 100
                            ) -> List[Dict[str, torch.Tensor]]:
    """
    This is the SECOND NMS in the pipeline (first was inside RPN's
    decode_proposals). Necessary because the detection head refines each
    proposal's box independently -- several originally-distinct RPN
    proposals can converge onto nearly the same final box after refinement,
    which the first NMS (run before refinement) had no way to catch.

    Splits the flat (N_total_sampled_or_all,) tensors back into per-image
    chunks using each image's proposal count, since everything upstream of
    this function operates on the whole batch flattened together.
    """
    boxes_per_image_counts = [p.shape[0] for p in proposals_per_image]
    cls_logits_split = cls_logits.split(boxes_per_image_counts)
    bbox_deltas_split = bbox_deltas.split(boxes_per_image_counts)
    embeddings_split = embeddings.split(boxes_per_image_counts)

    results = []
    for img_logits, img_deltas, img_emb, proposals, (img_h, img_w) in zip(
        cls_logits_split, bbox_deltas_split, embeddings_split, proposals_per_image, image_shapes
    ):
        scores = F.softmax(img_logits, dim=1)[:, 1]   # person-class probability
        boxes = box_coder.decode_single(img_deltas, proposals)
        boxes = box_ops.clip_boxes_to_image(boxes, (img_h, img_w))

        valid_size = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
        keep = (scores >= score_thresh) & valid_size
        boxes, scores, img_emb = boxes[keep], scores[keep], img_emb[keep]

        keep = box_ops.nms(boxes, scores, nms_thresh)
        keep = keep[:max_detections_per_image]

        results.append({"boxes": boxes[keep], "scores": scores[keep], "embeddings": img_emb[keep]})
    return results


# ---------------------------------------------------------------------------
# Wiring: full backbone -> FPN -> RPN -> ROI heads, for shape/logic verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(2, 3, 800, 600)
    gt_boxes_per_image = [
        torch.tensor([[100., 100., 180., 340.], [400., 200., 470., 500.]]),
        torch.tensor([[50., 50., 120., 300.]]),
    ]
    # identity labels per GT box: -1 = detected-but-unlabeled (as in
    # PRW/CUHK-SYSU), >=0 = a real labeled identity
    gt_ids_per_image = [torch.tensor([3, -1]), torch.tensor([7])]

    backbone, fpn = build_backbone_with_fpn("resnet50", pretrained=False)
    anchor_gen = build_anchor_generator()
    rpn_head = RPNHead(in_channels=fpn.out_channels[0],
                        num_anchors=anchor_gen.num_anchors_per_location()[0])
    rpn_assigner = RPNTargetAssigner()
    box_coder = BoxCoder(weights=BOX_CODER_WEIGHTS)
    rpn_sampler = BalancedPositiveNegativeSampler(RPN_BATCH_SIZE_PER_IMAGE, RPN_POSITIVE_FRACTION)

    roi_pooler = build_roi_pooler()
    det_head = DetectionHead(in_channels=fpn.out_channels[0])
    reid_head = ReIDHead(in_channels=fpn.out_channels[0])
    roi_assigner = ROITargetAssigner()

    backbone.eval(); fpn.eval(); rpn_head.eval(); det_head.eval(); reid_head.eval()
    with torch.no_grad():
        c_feats = backbone(x)
        p_feats_dict = fpn(c_feats)
        p_feats = [p_feats_dict[lvl] for lvl in fpn.out_levels]

        image_list = make_image_list(x)
        anchors_per_image = anchor_gen(image_list, p_feats)
        objectness, bbox_deltas_rpn = rpn_head(p_feats)

        image_shapes = [(x.shape[2], x.shape[3])] * x.shape[0]
        proposals = decode_proposals(objectness, bbox_deltas_rpn, anchors_per_image,
                                      image_shapes, box_coder,
                                      pre_nms_top_n=2000, post_nms_top_n=300, nms_thresh=0.7)
        print(f"[RPN] proposals per image: {[p.shape[0] for p in proposals]}")

        # --- ROI heads stage ---
        proposals = roi_assigner.add_gt_proposals(proposals, gt_boxes_per_image)
        print(f"[ROI] proposals after add_gt_proposals: {[p.shape[0] for p in proposals]}")

        sampled_props, cls_labels, matched_boxes, matched_ids = roi_assigner.assign_and_sample(
            proposals, gt_boxes_per_image, gt_ids_per_image
        )
        for i, (labels, ids) in enumerate(zip(cls_labels, matched_ids)):
            n_pos = (labels == 1).sum().item()
            n_neg = (labels == 0).sum().item()
            print(f"[ROI] image {i}: sampled {labels.shape[0]} proposals "
                  f"({n_pos} positive, {n_neg} negative); "
                  f"identity labels of positives: {ids[labels == 1].tolist()}")

        # ROIAlign needs a dict keyed by the SAME names as featmap_names,
        # and only P2-P5 (P6 deliberately excluded, see module docstring)
        p_feats_for_roi = {lvl: p_feats_dict[lvl] for lvl in ("P2", "P3", "P4", "P5")}
        pooled = roi_pooler(p_feats_for_roi, sampled_props, image_shapes)
        print(f"[ROI] pooled features shape: {tuple(pooled.shape)} "
              f"(N=total sampled proposals across batch, C={fpn.out_channels[0]}, "
              f"{ROI_ALIGN_OUTPUT_SIZE}x{ROI_ALIGN_OUTPUT_SIZE})")

        # DetectionHead and ReIDHead both read the SAME pooled features
        # independently, per the diagram -- not one feeding the other.
        cls_logits, bbox_deltas_det = det_head(pooled)
        embeddings = reid_head(pooled)
        print(f"[ROI] cls_logits: {tuple(cls_logits.shape)}  "
              f"bbox_deltas: {tuple(bbox_deltas_det.shape)}  "
              f"embeddings: {tuple(embeddings.shape)} "
              f"(L2-normalized check: norms={embeddings.norm(dim=1)[:3].tolist()})")

        flat_cls_labels = torch.cat(cls_labels)
        flat_matched_boxes = torch.cat(matched_boxes)
        flat_sampled_props = torch.cat(sampled_props)
        cls_loss, bbox_loss = compute_detection_loss(
            cls_logits, bbox_deltas_det, flat_cls_labels, flat_matched_boxes,
            flat_sampled_props, box_coder
        )
        print(f"[ROI] detection cls loss: {cls_loss.item():.4f}  "
              f"bbox loss: {bbox_loss.item():.4f}  (untrained -> sanity only)")

        # --- final (second) NMS / postprocessing, inference-style ---
        # re-run pooling+head on the ORIGINAL (unsampled) RPN proposals to
        # simulate what inference actually does (no GT, no sampling)
        inference_proposals = decode_proposals(objectness, bbox_deltas_rpn, anchors_per_image,
                                                image_shapes, box_coder,
                                                pre_nms_top_n=2000, post_nms_top_n=300, nms_thresh=0.7)
        pooled_inf = roi_pooler(p_feats_for_roi, inference_proposals, image_shapes)
        cls_logits_inf, bbox_deltas_inf = det_head(pooled_inf)
        embeddings_inf = reid_head(pooled_inf)
        final = postprocess_detections(cls_logits_inf, bbox_deltas_inf, embeddings_inf,
                                        inference_proposals, image_shapes, box_coder,
                                        score_thresh=0.0, nms_thresh=0.4)  # thresh=0 since untrained
        for i, res in enumerate(final):
            print(f"[Final] image {i}: {res['boxes'].shape[0]} final detections after 2nd NMS, "
                  f"embeddings shape {tuple(res['embeddings'].shape)}")
