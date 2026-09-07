"""
model.py

Wires every module built so far into a single nn.Module with one entry
point: PersonSearchModel. Nothing new is implemented here -- this file is
pure composition, matching how the whole pipeline was scoped from the start
(backbone.py / fpn.py / rpn.py / roi_heads.py / metric_loss.py each own
their own logic; this file only owns the ORDER they're called in and how
their outputs/losses get combined).

Two modes, controlled by the standard nn.Module .training flag:

    model.train(); losses = model(images, targets)
        images:  (B, 3, H, W) tensor, already resized/padded to a common size
        targets: list[dict], one per image:
                     {"boxes": (N,4) x1y1x2y2, "ids": (N,) int64,
                      -1 for GT persons with no identity annotation}
        returns: dict of 5 scalar losses (caller sums them + .backward())

    model.eval(); results = model(images)
        returns: list[dict], one per image:
                     {"boxes": (M,4), "scores": (M,), "embeddings": (M,D)}
        this is exactly what inference.py will call for building the
        gallery embeddings AND for the query side (when the query is a
        full scene image rather than a pre-cropped box -- see model.py's
        embed_crop() for the pre-cropped-query path, which skips RPN/ROI
        heads' proposal machinery entirely).

Loss composition (5 terms, matches the pipeline spec exactly):
    total = rpn_objectness + rpn_bbox + det_cls + det_bbox + metric_loss
Kept as a dict rather than pre-summed, so train.py/the notebook can log each
term separately -- if metric_loss stays ~0 while detection losses fall,
that's the P×K-batch-sampling problem flagged earlier, not a bug in this file.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision.models.detection._utils import BoxCoder, BalancedPositiveNegativeSampler

from fpn import build_backbone_with_fpn
from rpn import (build_anchor_generator, RPNHead, RPNTargetAssigner, compute_rpn_loss,
                  decode_proposals, make_image_list, BOX_CODER_WEIGHTS,
                  RPN_BATCH_SIZE_PER_IMAGE, RPN_POSITIVE_FRACTION)
from roi_heads import (build_roi_pooler, DetectionHead, ReIDHead, ROITargetAssigner,
                        compute_detection_loss, postprocess_detections, ROI_ALIGN_OUTPUT_SIZE)
from metric_loss import BatchHardTripletLoss, ContrastiveLoss
from torchvision.ops import box_iou


class PersonSearchModel(nn.Module):

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True,
                 fpn_out_channels: int = 256, embedding_dim: int = 256,
                 triplet_margin: float = 0.3,
                 metric_loss: str = "triplet",
                 rpn_pre_nms_top_n_train: int = 2000, rpn_post_nms_top_n_train: int = 2000,
                 rpn_pre_nms_top_n_test: int = 1000, rpn_post_nms_top_n_test: int = 300,
                 rpn_nms_thresh: float = 0.7,
                 det_score_thresh: float = 0.5, det_nms_thresh: float = 0.4,
                 det_max_detections: int = 100,
                 anchor_sizes=((16,), (32,), (64,), (128,)),
                 anchor_ratios=((2.0, 3.0, 4.0),) * 4,
                 rpn_fg_iou_thresh: float = 0.7, rpn_bg_iou_thresh: float = 0.3,
                 roi_fg_iou_thresh: float = 0.7, roi_bg_iou_thresh: float = 0.3):
        super().__init__()

        self.backbone, self.fpn = build_backbone_with_fpn(
            backbone_name, fpn_out_channels=fpn_out_channels, pretrained=pretrained, add_p6=False
        )
        self.anchor_gen = build_anchor_generator(anchor_sizes, anchor_ratios)
        if len(anchor_sizes) != len(self.fpn.out_levels):
            raise ValueError("Provide one anchor-size and ratio tuple per FPN level (P2–P5)")
        num_anchors = self.anchor_gen.num_anchors_per_location()
        assert len(set(num_anchors)) == 1, "RPNHead assumes equal anchor count per level"
        self.rpn_head = RPNHead(in_channels=fpn_out_channels, num_anchors=num_anchors[0])
        self.rpn_assigner = RPNTargetAssigner(rpn_fg_iou_thresh, rpn_bg_iou_thresh)
        self.rpn_sampler = BalancedPositiveNegativeSampler(RPN_BATCH_SIZE_PER_IMAGE, RPN_POSITIVE_FRACTION)

        self.roi_pooler = build_roi_pooler()
        # DetectionHead and ReIDHead are separate modules, per the pipeline
        # diagram: RoI Align feeds both independently, in parallel -- not
        # one head with an embedding branch grafted onto it. See roi_heads.py.
        self.det_head = DetectionHead(in_channels=fpn_out_channels)
        self.reid_head = ReIDHead(in_channels=fpn_out_channels, embedding_dim=embedding_dim)
        self.roi_assigner = ROITargetAssigner(roi_fg_iou_thresh, roi_bg_iou_thresh)
        if metric_loss == "triplet":
            self.metric_loss_fn = BatchHardTripletLoss(margin=triplet_margin)
        elif metric_loss == "contrastive":
            self.metric_loss_fn = ContrastiveLoss(margin=1.0)
        else:
            raise ValueError("metric_loss must be 'triplet' or 'contrastive'")

        self.box_coder = BoxCoder(weights=BOX_CODER_WEIGHTS)

        # train/test differ on proposal budget (recall vs. speed) -- see rpn.py docstring
        self.rpn_pre_nms_top_n_train = rpn_pre_nms_top_n_train
        self.rpn_post_nms_top_n_train = rpn_post_nms_top_n_train
        self.rpn_pre_nms_top_n_test = rpn_pre_nms_top_n_test
        self.rpn_post_nms_top_n_test = rpn_post_nms_top_n_test
        self.rpn_nms_thresh = rpn_nms_thresh

        self.det_score_thresh = det_score_thresh
        self.det_nms_thresh = det_nms_thresh
        self.det_max_detections = det_max_detections

    def _extract_pyramid(self, images: torch.Tensor):
        c_feats = self.backbone(images)
        p_feats_dict = self.fpn(c_feats)
        p_feats_list = [p_feats_dict[lvl] for lvl in self.fpn.out_levels]     # ordered P2..P5
        p_feats_roi = {lvl: p_feats_dict[lvl] for lvl in ("P2", "P3", "P4", "P5")}  # P6 excluded, see roi_heads.py
        return p_feats_dict, p_feats_list, p_feats_roi

    def forward(self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        if self.training:
            assert targets is not None, "targets required in training mode"
            return self._forward_train(images, targets)
        return self._forward_eval(images)

    # ------------------------------------------------------------------ #

    def _forward_train(self, images: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        gt_boxes_per_image = [t["boxes"] for t in targets]
        gt_ids_per_image = [t["ids"] for t in targets]
        image_shapes = [tuple(images.shape[-2:])] * images.shape[0]

        _, p_feats_list, p_feats_roi = self._extract_pyramid(images)

        image_list = make_image_list(images)
        anchors_per_image = self.anchor_gen(image_list, p_feats_list)
        objectness, rpn_bbox_deltas = self.rpn_head(p_feats_list)

        # --- RPN loss ---
        rpn_labels, rpn_matched_boxes = self.rpn_assigner(anchors_per_image, gt_boxes_per_image)
        rpn_obj_loss, rpn_box_loss = compute_rpn_loss(
            objectness, rpn_bbox_deltas, anchors_per_image, rpn_labels, rpn_matched_boxes,
            self.box_coder, self.rpn_sampler
        )

        # --- proposals for the ROI stage (train-time budget) ---
        proposals = decode_proposals(
            objectness, rpn_bbox_deltas, anchors_per_image, image_shapes, self.box_coder,
            pre_nms_top_n=self.rpn_pre_nms_top_n_train, post_nms_top_n=self.rpn_post_nms_top_n_train,
            nms_thresh=self.rpn_nms_thresh
        )
        # Match detector-generated proposals only; never inject GT boxes.

        sampled_props, cls_labels, matched_gt_boxes, matched_gt_ids = self.roi_assigner.assign_and_sample(
            proposals, gt_boxes_per_image, gt_ids_per_image
        )

        # torchvision RoI pooling uses FP32 even when backbone/FPN use BF16.
        with torch.autocast(images.device.type, enabled=False):
            pooled = self.roi_pooler({k: v.float() for k, v in p_feats_roi.items()},
                                     sampled_props, image_shapes)
        # both heads read the SAME pooled features independently (diagram: two
        # parallel boxes off RoI Align), not one head feeding the other
        cls_logits, det_bbox_deltas = self.det_head(pooled)
        flat_cls_labels = torch.cat(cls_labels)
        flat_matched_boxes = torch.cat(matched_gt_boxes)
        flat_matched_ids = torch.cat(matched_gt_ids)
        flat_sampled_props = torch.cat(sampled_props)

        # --- detection loss ---
        det_cls_loss, det_box_loss = compute_detection_loss(
            cls_logits, det_bbox_deltas, flat_cls_labels, flat_matched_boxes,
            flat_sampled_props, self.box_coder
        )

        # --- metric loss ---
        reid_keep = (flat_cls_labels == 1) & (flat_matched_ids >= 0)
        filt_emb = self.reid_head(pooled[reid_keep])
        filt_ids = flat_matched_ids[reid_keep]
        metric_loss, metric_stats = self.metric_loss_fn(filt_emb, filt_ids)
        with torch.no_grad():
            gt_count = sum(len(b) for b in gt_boxes_per_image)
            recalled = sum(
                int((box_iou(gt, props).max(dim=1).values >= 0.5).sum())
                for gt, props in zip(gt_boxes_per_image, proposals) if len(gt) and len(props)
            )

        return {
            "rpn_objectness_loss": rpn_obj_loss,
            "rpn_bbox_loss": rpn_box_loss,
            "det_cls_loss": det_cls_loss,
            "det_bbox_loss": det_box_loss,
            "metric_loss": metric_loss,
            # not a loss, but useful to log alongside the losses so a
            # collapsing metric_loss can be diagnosed (see module docstring)
            "_metric_active_fraction": images.new_tensor(metric_stats.get("active_fraction", 0.0)),
            "_hardest_positive_distance": images.new_tensor(metric_stats.get("hardest_positive_distance", 0.0)),
            "_hardest_negative_distance": images.new_tensor(metric_stats.get("hardest_negative_distance", 0.0)),
            "_rpn_positive_anchors": sum((labels == 1).sum() for labels in rpn_labels).detach(),
            "_proposals": images.new_tensor(sum(len(p) for p in proposals)),
            "_positive_rois": (flat_cls_labels == 1).sum().detach(),
            "_labeled_rois": reid_keep.sum().detach(),
            "_reid_identities": images.new_tensor(filt_ids.unique().numel()),
            "_proposal_recall_iou50": images.new_tensor(recalled / max(gt_count, 1)),
            "_gt_persons": images.new_tensor(gt_count),
            "_metric_valid_anchors": images.new_tensor(float(metric_stats.get("valid_anchors", 0))),
        }

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _forward_eval(self, images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        image_shapes = [tuple(images.shape[-2:])] * images.shape[0]

        _, p_feats_list, p_feats_roi = self._extract_pyramid(images)

        image_list = make_image_list(images)
        anchors_per_image = self.anchor_gen(image_list, p_feats_list)
        objectness, rpn_bbox_deltas = self.rpn_head(p_feats_list)

        proposals = decode_proposals(
            objectness, rpn_bbox_deltas, anchors_per_image, image_shapes, self.box_coder,
            pre_nms_top_n=self.rpn_pre_nms_top_n_test, post_nms_top_n=self.rpn_post_nms_top_n_test,
            nms_thresh=self.rpn_nms_thresh
        )  # no GT injection at eval time -- real proposals only

        pooled = self.roi_pooler(p_feats_roi, proposals, image_shapes)
        cls_logits, det_bbox_deltas = self.det_head(pooled)
        embeddings = self.reid_head(pooled)

        return postprocess_detections(
            cls_logits, det_bbox_deltas, embeddings, proposals, image_shapes, self.box_coder,
            score_thresh=self.det_score_thresh, nms_thresh=self.det_nms_thresh,
            max_detections_per_image=self.det_max_detections
        )

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def embed_crop(self, crop: torch.Tensor) -> torch.Tensor:
        """
        Query-side path for inference.py: the query is normally given as a
        TIGHT CROP of one person (not a full scene needing detection), so
        running the whole RPN/proposal machinery on it would be wrong --
        there's no "which box is the person" ambiguity to resolve, the
        crop IS the box. Instead: backbone -> FPN -> pool the crop's own
        full extent as a single "proposal" -> detection head -> embedding.

        crop: (B, 3, H, W), a batch of person crops (can be B=1 for one query)
        returns: (B, embedding_dim) L2-normalized embeddings
        """
        _, _, p_feats_roi = self._extract_pyramid(crop)
        B, _, H, W = crop.shape
        # one "proposal" per image covering the entire crop
        full_box = torch.tensor([[0., 0., float(W), float(H)]], device=crop.device)
        boxes = [full_box.clone() for _ in range(B)]
        image_shapes = [(H, W)] * B

        # query side only ever needs the embedding, not class/bbox -- so
        # this path calls ReIDHead alone, skipping DetectionHead entirely
        pooled = self.roi_pooler(p_feats_roi, boxes, image_shapes)
        return self.reid_head(pooled)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = PersonSearchModel(backbone_name="resnet50", pretrained=False)

    # ---- training-mode forward + backward sanity check ----
    model.train()
    images = torch.randn(2, 3, 800, 600)
    targets = [
        {"boxes": torch.tensor([[100., 100., 180., 340.], [400., 200., 470., 500.]]),
         "ids": torch.tensor([3, -1])},
        {"boxes": torch.tensor([[50., 50., 120., 300.]]), "ids": torch.tensor([3])},
    ]

    losses = model(images, targets)
    print("[Train] loss dict:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    total_loss = sum(v for k, v in losses.items() if not k.startswith("_"))
    total_loss.backward()

    n_grad, n_total = 0, 0
    for name, p in model.named_parameters():
        n_total += 1
        if p.requires_grad:
            n_grad += 1 if (p.grad is not None and p.grad.abs().sum().item() > 0) else 0
    print(f"\n[Backward] parameters with nonzero grad: {n_grad} "
          f"(out of {n_total} total; some are frozen BN/backbone by design, not a bug)")
    assert n_grad > 0, "backward pass produced no gradients at all -- something is disconnected"

    # ---- eval-mode forward ----
    model.eval()
    with torch.no_grad():
        results = model(images)
    print("\n[Eval] detections per image:")
    for i, r in enumerate(results):
        print(f"  image {i}: {r['boxes'].shape[0]} boxes, "
              f"embeddings {tuple(r['embeddings'].shape)}")

    # ---- query-crop embedding path ----
    with torch.no_grad():
        query_crop = torch.randn(3, 3, 128, 64)  # 3 query crops, tall aspect ratio
        query_emb = model.embed_crop(query_crop)
    print(f"\n[Query] embed_crop output: {tuple(query_emb.shape)} "
          f"(norms: {query_emb.norm(dim=1).tolist()})")
