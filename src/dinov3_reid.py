"""DINOv3 fixed crop features with a triplet-trained linear projection."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class DinoV3CropReID(nn.Module):
    def __init__(self, backbone=None, hub_repo='facebookresearch/dinov3:main',
                 embedding_dim=256, weights_path=None, pretrained=True):
        super().__init__()
        if backbone is None:
            kwargs = dict(pretrained=pretrained, trust_repo=True)
            if pretrained:
                if weights_path is None or not Path(weights_path).is_file():
                    raise ValueError('Provide a downloaded official DINOv3 ViT-S/16 weights_path')
                kwargs['weights'] = str(weights_path)
            backbone = torch.hub.load(hub_repo, 'dinov3_vits16', **kwargs)
        self.backbone = backbone.requires_grad_(False).eval()
        self.embedding_proj = nn.Linear(backbone.embed_dim, embedding_dim)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, crops):
        h, w = crops.shape[-2:]
        crops = F.pad(crops, (0, (-w) % 16, 0, (-h) % 16))
        # no_grad (not inference_mode) allows projection backward to save features.
        with torch.no_grad():
            features = self.backbone.forward_features(crops)['x_norm_clstoken']
        return F.normalize(self.embedding_proj(features), dim=1)


class FrozenDinoV3CLS(nn.Module):
    """Evaluation-only raw CLS embedding, with no learned projection.

    Shares a restored DINO backbone; does not change the source projection.
    embedding_proj is an identity adapter exposing the dimension expected by
    evaluate_prw when a gallery frame has no detections.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.embedding_proj = nn.Identity()
        self.embedding_proj.out_features = backbone.embed_dim

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, crops):
        self.backbone.eval()
        h, w = crops.shape[-2:]
        crops = F.pad(crops, (0, (-w) % 16, 0, (-h) % 16))
        features = self.backbone.forward_features(crops)['x_norm_clstoken']
        return F.normalize(features.float(), dim=1)
