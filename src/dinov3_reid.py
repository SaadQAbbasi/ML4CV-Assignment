"""DINOv3 crop ReID: frozen by default, with optional last-block fine-tuning."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class DinoV3CropReID(nn.Module):
    def __init__(self, backbone=None, hub_repo='facebookresearch/dinov3:main',
                 embedding_dim=256, weights_path=None, pretrained=True, train_last_blocks=0, crop_size=(128, 64), classifier_ids=None, feature_pooling="cls"):
        super().__init__()
        if feature_pooling not in ("cls", "cls_mean_patch", "cls_upper_lower", "cls_three_regions"):
            raise ValueError("feature_pooling must be 'cls', 'cls_mean_patch' 'cls_upper_lower' or 'cls_three_regions'")
        self.feature_pooling = feature_pooling
        self.crop_size = tuple(crop_size)
        if backbone is None:
            kwargs = dict(pretrained=pretrained, trust_repo=True)
            if pretrained:
                if weights_path is None or not Path(weights_path).is_file():
                    raise ValueError('Provide a downloaded official DINOv3 ViT-S/16 weights_path')
                kwargs['weights'] = str(weights_path)
            backbone = torch.hub.load(hub_repo, 'dinov3_vits16', **kwargs)
        self.backbone = backbone.requires_grad_(False).eval()
        self.train_last_blocks = train_last_blocks
        if train_last_blocks < 0 or (train_last_blocks and train_last_blocks > len(backbone.blocks)):
            raise ValueError('Invalid number of trainable DINOv3 blocks')
        if train_last_blocks:
            for block in self.backbone.blocks[-train_last_blocks:]:
                block.requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
        self.embedding_proj = nn.Linear(backbone.embed_dim * {"cls": 1, "cls_mean_patch": 2, "cls_upper_lower": 3, "cls_three_regions": 4}[feature_pooling], embedding_dim)
        self.identity_classifier = None
        if classifier_ids is not None:
            labels = sorted(set(int(i) for i in classifier_ids))
            if not labels or labels[0] < 0:
                raise ValueError('Classifier requires known training identities')
            self.register_buffer('classifier_ids', torch.tensor(labels, dtype=torch.long))
            self.identity_classifier = nn.Linear(embedding_dim, len(labels))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode and self.train_last_blocks:
            for block in self.backbone.blocks[-self.train_last_blocks:]:
                block.train()
        return self

    def forward(self, crops, return_logits=False):
        h, w = crops.shape[-2:]
        crops = F.pad(crops, (0, (-w) % 16, 0, (-h) % 16))
        # Respect outer inference contexts; frozen mode retains the old behavior.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_last_blocks > 0):
            outputs = self.backbone.forward_features(crops)
            features = outputs['x_norm_clstoken']
            if self.feature_pooling == "cls_mean_patch":
                # Official patch output excludes CLS and register/storage tokens.
                patches = outputs['x_norm_patchtokens']
                features = torch.cat((features, patches.mean(dim=1)), dim=1)
            elif self.feature_pooling == "cls_upper_lower":
                patches = outputs['x_norm_patchtokens']
                rows, cols = crops.shape[-2] // 16, crops.shape[-1] // 16
                if rows < 2 or patches.shape[1] != rows * cols:
                    raise ValueError('Upper/lower pooling requires a matching ViT-S/16 grid with at least two rows')
                # Patch tokens are row-major; split height, not alternating tokens.
                grid = patches.reshape(patches.shape[0], rows, cols, patches.shape[-1])
                split = rows // 2
                upper = grid[:, :split].mean(dim=(1, 2))
                lower = grid[:, split:].mean(dim=(1, 2))
                features = torch.cat((features, upper, lower), dim=1)
            elif self.feature_pooling == "cls_three_regions":
                patches = outputs['x_norm_patchtokens']
                rows, cols = crops.shape[-2] // 16, crops.shape[-1] // 16
                if rows < 3 or patches.shape[1] != rows * cols:
                    raise ValueError('Three-region pooling requires a matching ViT-S/16 grid with at least three rows')
                grid = patches.reshape(patches.shape[0], rows, cols, patches.shape[-1])
                # Non-overlapping approximate thirds; 16 rows split into 5/5/6.
                first, second = rows // 3, 2 * rows // 3
                regions = (grid[:, :first], grid[:, first:second], grid[:, second:])
                features = torch.cat((features, *(region.mean(dim=(1, 2)) for region in regions)), dim=1)
        projected = self.embedding_proj(features)
        embeddings = F.normalize(projected, dim=1)
        if return_logits:
            if self.identity_classifier is None:
                raise ValueError('No identity classifier configured')
            return embeddings, self.identity_classifier(projected)
        return embeddings


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
