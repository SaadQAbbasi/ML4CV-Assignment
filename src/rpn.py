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
# Wiring: backbone -> FPN -> anchors + RPN head, for shape verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(2, 3, 800, 600)
    backbone, fpn = build_backbone_with_fpn("resnet50", pretrained=False)
    anchor_gen = build_anchor_generator()
    num_anchors_per_loc = anchor_gen.num_anchors_per_location()
    print(f"Anchors per location, per level (P2..P6): {num_anchors_per_loc}")
    assert len(set(num_anchors_per_loc)) == 1, \
        "RPNHead below assumes the same anchor count at every level"
    rpn_head = RPNHead(in_channels=fpn.out_channels[0], num_anchors=num_anchors_per_loc[0])

    backbone.eval(); fpn.eval(); rpn_head.eval()
    with torch.no_grad():
        c_feats = backbone(x)
        p_feats_dict = fpn(c_feats)
        p_feats = [p_feats_dict[lvl] for lvl in fpn.out_levels]  # ordered P2..P6

        image_list = make_image_list(x)
        anchors_per_image = anchor_gen(image_list, p_feats)     # list[B] of (num_anchors_total, 4)

        objectness, bbox_deltas = rpn_head(p_feats)

    print(f"\nInput: {tuple(x.shape)}")
    total_anchors = 0
    for lvl, stride, obj, deltas in zip(fpn.out_levels, fpn.out_strides, objectness, bbox_deltas):
        n_anchors_level = obj.shape[1] * obj.shape[2] * obj.shape[3]
        total_anchors += n_anchors_level
        print(f"  {lvl} (stride {stride:>2}): objectness={tuple(obj.shape)}  "
              f"bbox_deltas={tuple(deltas.shape)}  anchors_at_this_level={n_anchors_level}")

    print(f"\nTotal anchors/image (summed over levels): {total_anchors}")
    print(f"AnchorGenerator anchors for image 0: {anchors_per_image[0].shape[0]} "
          f"({'OK' if anchors_per_image[0].shape[0] == total_anchors else 'MISMATCH'})")
    print(f"Sample anchor boxes (first 3, level P2, format x1,y1,x2,y2):\n{anchors_per_image[0][:3]}")