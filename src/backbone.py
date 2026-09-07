"""
backbone.py

Goal of this file (Step 00 of the person-search pipeline):
    Image (B, 3, H, W) -> dict of multi-resolution feature maps

Design:
    - `Backbone` is an abstract interface. Every backbone (ResNet, DINO, ...)
      implements `forward(x) -> Dict[str, Tensor]` where keys are level names
      ("C2","C3","C4","C5") and each value is (B, C, H_i, W_i).
    - `build_backbone(name=...)` is the single factory function you call from
      the rest of the pipeline. Swapping ResNet50 <-> ResNet101 <-> DINO later
      is a one-line change at the call site, nothing downstream needs to know.
    - FPN is intentionally NOT part of this file. This file's only job is
      "image -> multi-scale feature maps". FPN (the top-down fusion) is the
      next module (fpn.py) and will consume the dict this file produces.

Currently implemented:
    - ResNet18/34/50/101/152 (torchvision, ImageNet-pretrained)

Stubbed (raises NotImplementedError with notes):
    - DINO ViT backbone -- included as a stub because ViT backbones are NOT
      naturally multi-scale the way CNNs are (every transformer block outputs
      the same H/patch_size x W/patch_size grid), so "feature maps at
      different resolutions" needs a different strategy there (e.g. pool a
      single-resolution token grid down to several resolutions, or use a
      multi-scale ViT variant). Left as a clearly marked follow-up so the
      CNN path can be verified/trained first.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter


class Backbone(ABC, nn.Module):
    """Common interface every backbone must satisfy."""

    #: output feature map names, ordered shallow -> deep
    out_levels: List[str] = []
    #: channel count for each entry in out_levels (same order)
    out_channels: List[int] = []
    #: stride relative to the input image for each entry in out_levels
    out_strides: List[int] = []

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...


class ResNetBackbone(Backbone):
    """
    Wraps a torchvision ResNet and exposes C2-C5 feature maps.

        C2: stride 4,  from layer1
        C3: stride 8,  from layer2
        C4: stride 16, from layer3
        C5: stride 32, from layer4
    """

    _CHANNELS = {
        "resnet18": [64, 128, 256, 512],
        "resnet34": [64, 128, 256, 512],
        "resnet50": [256, 512, 1024, 2048],
        "resnet101": [256, 512, 1024, 2048],
        "resnet152": [256, 512, 1024, 2048],
    }

    def __init__(self, name: str = "resnet50", pretrained: bool = True,
                 trainable_layers: int = 3, frozen_bn: bool = True):
        """
        name:              one of _CHANNELS keys
        pretrained:        load ImageNet weights
        trainable_layers:  how many of {layer4, layer3, layer2, layer1, stem}
                            to unfreeze, counting from the deepest. 3 means
                            layer4, layer3, layer2 are trainable; stem and
                            layer1 stay frozen (common Faster R-CNN default).
        frozen_bn:         if True, BatchNorm layers everywhere are frozen
                            (running stats + affine params). Standard practice
                            when fine-tuning detectors with small batch sizes.
        """
        super().__init__()
        if name not in self._CHANNELS:
            raise ValueError(f"Unknown resnet variant '{name}'. "
                              f"Choose from {list(self._CHANNELS)}")

        weights = torchvision.models.get_model_weights(name).DEFAULT if pretrained else None
        net = torchvision.models.get_model(name, weights=weights)
        self.frozen_bn = frozen_bn

        if frozen_bn:
            net = self._freeze_batchnorm(net)

        self._set_trainable_layers(net, trainable_layers)
        if frozen_bn:
            self._freeze_batchnorm(net)

        return_layers = {"layer1": "C2", "layer2": "C3",
                          "layer3": "C4", "layer4": "C5"}
        self.body = IntermediateLayerGetter(net, return_layers=return_layers)

        self.out_levels = ["C2", "C3", "C4", "C5"]
        self.out_channels = self._CHANNELS[name]
        self.out_strides = [4, 8, 16, 32]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_bn:
            self._freeze_batchnorm(self)
        return self

    @staticmethod
    def _freeze_batchnorm(net: nn.Module) -> nn.Module:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        return net

    @staticmethod
    def _set_trainable_layers(net: nn.Module, trainable_layers: int) -> None:
        assert 0 <= trainable_layers <= 5
        layers_deep_to_shallow = ["layer4", "layer3", "layer2", "layer1", "stem"]
        layers_to_train = layers_deep_to_shallow[:trainable_layers]

        # torchvision resnet stem = conv1 + bn1 (no explicit "stem" attr)
        name_to_module = {
            "layer4": [net.layer4], "layer3": [net.layer3],
            "layer2": [net.layer2], "layer1": [net.layer1],
            "stem": [net.conv1, net.bn1],
        }
        for layer_name, modules in name_to_module.items():
            requires_grad = layer_name in layers_to_train
            for module in modules:
                for p in module.parameters():
                    p.requires_grad_(requires_grad)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.body(x)  # {"C2":..., "C3":..., "C4":..., "C5":...}


class SwinBackbone(Backbone):
    """
    Wraps torchvision's Swin Transformer (hierarchical ViT) and exposes
    C2-C5 style feature maps, same contract as ResNetBackbone.

    Unlike plain ViT/DINO, Swin merges patches between stages (like a CNN's
    strided convs), so it produces a genuine 4-level pyramid:
        C2: stride 4,  from end of stage 1
        C3: stride 8,  from end of stage 2
        C4: stride 16, from end of stage 3
        C5: stride 32, from end of stage 4
    That's why this is a real implementation and not a stub like DINO below:
    it satisfies the Backbone interface exactly, no resampling tricks needed.

    Caveat: torchvision's Swin blocks output channels-LAST tensors
    (B, H, W, C), because internally they run LayerNorm over the channel
    dim. FPN / roi_align expect channels-FIRST (B, C, H, W), so this class
    permutes at the point features are collected -- downstream code never
    needs to know Swin's internal layout.
    """

    _CHANNELS = {
        "swin_t": [96, 192, 384, 768],
        "swin_s": [96, 192, 384, 768],
        "swin_b": [128, 256, 512, 1024],
    }
    # index into net.features whose OUTPUT is the end of each stage
    _STAGE_END_INDICES = [1, 3, 5, 7]  # stage1, stage2, stage3, stage4

    def __init__(self, name: str = "swin_t", pretrained: bool = True,
                 trainable_stages: int = 3):
        """
        name:              one of _CHANNELS keys ("swin_t"/"swin_s"/"swin_b")
        pretrained:        load ImageNet weights
        trainable_stages:  how many of the 4 stages to unfreeze, counting
                            from the deepest (stage4 first). Mirrors
                            ResNetBackbone's trainable_layers for consistency.
        """
        super().__init__()
        if name not in self._CHANNELS:
            raise ValueError(f"Unknown swin variant '{name}'. "
                              f"Choose from {list(self._CHANNELS)}")

        weights = "IMAGENET1K_V1" if pretrained else None
        net = torchvision.models.get_model(name, weights=weights)

        self.features = net.features  # nn.Sequential, see _STAGE_END_INDICES
        self._set_trainable_stages(trainable_stages)

        self.out_levels = ["C2", "C3", "C4", "C5"]
        self.out_channels = self._CHANNELS[name]
        self.out_strides = [4, 8, 16, 32]

    def _set_trainable_stages(self, trainable_stages: int) -> None:
        assert 0 <= trainable_stages <= 4
        # stage boundaries in self.features: stage1=[0,1] stage2=[2,3] stage3=[4,5] stage4=[6,7]
        stage_module_indices = [[0, 1], [2, 3], [4, 5], [6, 7]]
        stages_deep_to_shallow = list(reversed(stage_module_indices))
        train_indices = set()
        for stage in stages_deep_to_shallow[:trainable_stages]:
            train_indices.update(stage)

        for i, module in enumerate(self.features):
            requires_grad = i in train_indices
            for p in module.parameters():
                p.requires_grad_(requires_grad)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = {}
        level_names = iter(self.out_levels)
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            if i in self._STAGE_END_INDICES:
                # (B, H, W, C) -> (B, C, H, W) so FPN/roi_align see channels-first
                feats[next(level_names)] = out.permute(0, 3, 1, 2).contiguous()
        return feats


class DINOBackbone(Backbone):
    """
    Placeholder for a DINO ViT backbone.

    NOT a drop-in replacement for ResNetBackbone yet: a ViT produces a single
    (H/patch_size, W/patch_size) grid of patch tokens per block, not a natural
    multi-resolution pyramid like a CNN's stages. Getting "feature maps at
    different resolutions" out of it needs one of:
      (a) take intermediate transformer blocks (all same resolution) and
          avg/max-pool that single grid down to several resolutions, or
      (b) use a hierarchical ViT (Swin, PVT) which *does* downsample stage to
          stage like a CNN, or
      (c) feed multiple crops/scales of the input image through the same ViT.
    Left unimplemented on purpose until the ResNet path is validated end to
    end -- swap in later without touching any other pipeline file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "DINOBackbone is a stub. See class docstring for why ViT "
            "feature maps aren't a plug-and-play swap for ResNet's C2-C5, "
            "and implement (a)/(b)/(c) depending on which tradeoff you want."
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


_REGISTRY = {
    "resnet18": ResNetBackbone, "resnet34": ResNetBackbone,
    "resnet50": ResNetBackbone, "resnet101": ResNetBackbone,
    "resnet152": ResNetBackbone,
    "swin_t": SwinBackbone, "swin_s": SwinBackbone, "swin_b": SwinBackbone,
    "dino": DINOBackbone,
}


def build_backbone(name: str = "resnet50", **kwargs) -> Backbone:
    """
    Single entry point the rest of the pipeline should call.

    Examples:
        backbone = build_backbone("resnet50", pretrained=True, trainable_layers=3)
        backbone = build_backbone("resnet101", pretrained=True)
        backbone = build_backbone("swin_t", pretrained=True, trainable_stages=2)
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    if cls is ResNetBackbone:
        return ResNetBackbone(name=name, **kwargs)
    if cls is SwinBackbone:
        return SwinBackbone(name=name, **kwargs)
    return cls(**kwargs)


def _check_backbone(backbone: Backbone, x: torch.Tensor, label: str) -> None:
    backbone.eval()
    with torch.no_grad():
        feats = backbone(x)

    print(f"--- {label} | input: {tuple(x.shape)} ---")
    for level, expected_c, expected_s in zip(
        backbone.out_levels, backbone.out_channels, backbone.out_strides
    ):
        fmap = feats[level]
        # ceil-div: conv/window layers with padding round the output size up
        expected_h = -(-x.shape[2] // expected_s)
        expected_w = -(-x.shape[3] // expected_s)
        ok = (fmap.shape[1] == expected_c and fmap.shape[2] == expected_h
              and fmap.shape[3] == expected_w)
        status = "OK" if ok else "MISMATCH"
        print(f"  {level}: shape={tuple(fmap.shape)}  "
              f"expected=(B={x.shape[0]}, C={expected_c}, H={expected_h}, W={expected_w})  [{status}]")

    n_trainable = sum(p.requires_grad for p in backbone.parameters())
    n_total = sum(1 for _ in backbone.parameters())
    print(f"  trainable param tensors: {n_trainable}/{n_total}\n")


if __name__ == "__main__":
    torch.manual_seed(0)

    # ResNet: works with arbitrary, non-power-of-2 spatial sizes
    x_resnet = torch.randn(2, 3, 800, 600)
    resnet_backbone = build_backbone("resnet50", pretrained=False, trainable_layers=3)
    _check_backbone(resnet_backbone, x_resnet, "resnet50 (trainable_layers=3)")

    # Swin: window attention wants sizes divisible by patch_size * window_size
    # (4 * 7 = 28) at the input to stage 1 for a clean, unpadded pyramid.
    x_swin = torch.randn(2, 3, 224, 224)
    swin_backbone = build_backbone("swin_t", pretrained=False, trainable_stages=2)
    _check_backbone(swin_backbone, x_swin, "swin_t (trainable_stages=2)")