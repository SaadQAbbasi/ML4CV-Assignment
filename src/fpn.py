"""
fpn.py

Goal of this file (bridges Step 00 -> Step 01 of the pipeline):
    {C2, C3, C4, C5} (from backbone.py) -> {P2, P3, P4, P5, P6} (fused pyramid)

FPN's job: take the backbone's per-stage feature maps (deep = more semantic
but low-res, shallow = less semantic but high-res) and fuse them top-down so
EVERY output level is both high-resolution-appropriate AND semantically rich.
Without this, RPN on C2 alone would be spatially precise but semantically
weak (bad for classifying "is this a person"), and RPN on C5 alone would be
semantically strong but too coarse to localize small/far people.

This file is intentionally thin: the fusion math (1x1 lateral convs +
nearest-neighbor upsample + add, then 3x3 smoothing conv) is standard and
well-tested in torchvision, so we wrap it rather than reimplement it -- same
philosophy as reusing torchvision's ResNet/Swin in backbone.py. The
person-search-specific logic starts in rpn.py, not here.

P6 is included (extra stride-64 level, made by max-pooling P5) because the
original FPN paper found an extra coarse level helps RPN recall on large
objects; standard in Faster-RCNN-FPN and Mask-RCNN implementations.
"""

from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from backbone import Backbone, build_backbone


class FPN(nn.Module):
    """
    Wraps torchvision's FeaturePyramidNetwork so it consumes/produces the
    same {level_name: tensor} dict convention as Backbone, and carries its
    own out_channels/out_strides so downstream code (rpn.py) never needs to
    hardcode backbone-specific channel counts.
    """

    def __init__(self, backbone: Backbone, out_channels: int = 256,
                 add_p6: bool = True):
        super().__init__()
        self.in_levels = list(backbone.out_levels)          # ["C2","C3","C4","C5"]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(backbone.out_channels),    # e.g. [256,512,1024,2048]
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool() if add_p6 else None,
        )

        # output level names mirror input names but P-prefixed; +P6 if added
        self.out_levels: List[str] = [f"P{lvl[1:]}" for lvl in self.in_levels]
        self.out_strides: List[int] = list(backbone.out_strides)
        if add_p6:
            self.out_levels.append("P6")
            self.out_strides.append(self.out_strides[-1] * 2)   # P6 = P5 stride * 2
        self.out_channels: List[int] = [out_channels] * len(self.out_levels)

    def forward(self, backbone_feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # FeaturePyramidNetwork wants an OrderedDict keyed by whatever names
        # we give it (order matters: shallow -> deep), and returns the same
        # keys back with fused+smoothed maps (plus one extra "pool" key from
        # LastLevelMaxPool if extra_blocks was set).
        ordered = {lvl: backbone_feats[lvl] for lvl in self.in_levels}
        fused = self.fpn(ordered)

        # remap {"C2":P2, "C3":P3, "C4":P4, "C5":P5, "pool":P6} -> P-prefixed names
        out = {}
        for c_lvl, p_lvl in zip(self.in_levels, self.out_levels):
            out[p_lvl] = fused[c_lvl]
        if "pool" in fused:
            out["P6"] = fused["pool"]
        return out


def build_backbone_with_fpn(backbone_name: str = "resnet50", fpn_out_channels: int = 256,
                             add_p6: bool = True, **backbone_kwargs):
    """
    Convenience factory: builds a matched (backbone, fpn) pair in one call.
    Kept as two separate modules (not fused into one class) so you can still
    call the backbone alone -- useful for the query-embedding path at
    inference time, where you may want raw backbone features without RPN.
    """
    backbone = build_backbone(backbone_name, **backbone_kwargs)
    fpn = FPN(backbone, out_channels=fpn_out_channels, add_p6=add_p6)
    return backbone, fpn


if __name__ == "__main__":
    torch.manual_seed(0)

    for name, x in [
        ("resnet50", torch.randn(2, 3, 800, 600)),
        ("swin_t", torch.randn(2, 3, 800, 600)),
    ]:
        backbone, fpn = build_backbone_with_fpn(name, pretrained=False)
        backbone.eval(); fpn.eval()

        with torch.no_grad():
            c_feats = backbone(x)
            p_feats = fpn(c_feats)

        print(f"--- {name} -> FPN | input: {tuple(x.shape)} ---")
        for lvl, stride in zip(fpn.out_levels, fpn.out_strides):
            fmap = p_feats[lvl]
            expected_h = -(-x.shape[2] // stride)
            expected_w = -(-x.shape[3] // stride)
            ok = (fmap.shape[1] == fpn.out_channels[0]
                  and fmap.shape[2] == expected_h and fmap.shape[3] == expected_w)
            status = "OK" if ok else "MISMATCH"
            print(f"  {lvl} (stride {stride}): shape={tuple(fmap.shape)} "
                  f"expected=(B={x.shape[0]}, C={fpn.out_channels[0]}, "
                  f"H={expected_h}, W={expected_w})  [{status}]")
        print()