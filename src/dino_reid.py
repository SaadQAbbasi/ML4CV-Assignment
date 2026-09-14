"""Frozen Swin detector + independent DINOv2 crop ReID (no detector updates)."""
import torch
from torch import nn
from torch.nn import functional as F
from metric_loss import BatchHardTripletLoss, crop_to_crop_reid_loss


class DinoCropReID(nn.Module):
    def __init__(self, backbone=None, hub_repo='facebookresearch/dinov2:main',
                 embedding_dim=256, train_last_blocks=2, pretrained=True):
        super().__init__()
        self.backbone = backbone if backbone is not None else torch.hub.load(
            hub_repo, 'dinov2_vits14', pretrained=pretrained, trust_repo=True)
        if not 0 <= train_last_blocks <= len(self.backbone.blocks):
            raise ValueError('Invalid number of trainable DINO blocks')
        self.train_last_blocks = train_last_blocks
        self.backbone.requires_grad_(False)
        if train_last_blocks:
            for block in self.backbone.blocks[-train_last_blocks:]:
                block.requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
        self.embedding_proj = nn.Linear(self.backbone.embed_dim, embedding_dim)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode and self.train_last_blocks:
            for block in self.backbone.blocks[-self.train_last_blocks:]:
                block.train()
        return self

    def forward(self, crops):
        # Preserve the existing normalized 128x64 crop input for comparability.
        # DINO-S/14 requires patch-divisible dimensions; pad to 140x70 rather
        # than silently discarding border pixels. Zero is mean-color padding.
        h, w = crops.shape[-2:]
        crops = F.pad(crops, (0, (-w) % 14, 0, (-h) % 14))
        features = self.backbone.forward_features(crops)['x_norm_clstoken']
        return F.normalize(self.embedding_proj(features), dim=1)


class DinoPersonSearch(nn.Module):
    """evaluate_prw-compatible adapter. Scene forward only runs frozen detector."""
    def __init__(self, detector, reid):
        super().__init__()
        self.detector = detector.requires_grad_(False).eval()
        self.reid_head = reid

    @property
    def det_score_thresh(self):
        return self.detector.det_score_thresh

    @det_score_thresh.setter
    def det_score_thresh(self, value):
        self.detector.det_score_thresh = value

    def train(self, mode=True):
        super().train(mode)
        self.detector.eval()
        return self

    @torch.no_grad()
    def forward(self, images):
        return self.detector(images)

    def embed_crop(self, crops):
        return self.reid_head(crops)


def crop_epoch(reid, loader, device, optimizer=None, scaler=None, margin=.3,
               run=None, epoch=0, max_batches=None):
    """Train/validate only DINO crop embeddings; no full images sent to GPU."""
    from training import amp_dtype
    training = optimizer is not None
    reid.train(training)
    criterion = BatchHardTripletLoss(margin)
    totals, count, positive_batches = {}, 0, 0
    for step, (_, targets) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        ids = torch.cat([t['ids'][t['ids'] >= 0] for t in targets]).to(device)
        if not len(ids):
            continue
        frames = torch.cat([t['image_id'].reshape(1).expand(int((t['ids'] >= 0).sum()))
                            for t in targets]).to(device)
        crops = torch.cat([t['query_crops'] for t in targets]).to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device.type, enabled=device.type == 'cuda', dtype=amp_dtype(device)):
                emb = reid(crops)
                loss, stats = crop_to_crop_reid_loss(emb, ids, frames, criterion)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite DINO metric loss')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(reid.parameters(), 10., error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
        values = dict(metric_loss=float(loss.detach()), **stats)
        positive_batches += int(stats['num_pos_pairs'] > 0)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.) + value
        count += 1
        if run and training:
            run.log({'epoch': epoch + (step + 1) / len(loader),
                     **{f'train/{k}': v for k,v in values.items()}})
        if step % 20 == 0:
            print(f"{'train' if training else 'val'} batch {step}: loss={float(loss.detach()):.4f}, valid={stats['valid_anchors']}", flush=True)
    if count == 0:
        raise ValueError('No labeled crops in loader')
    result = {k: v / count for k,v in totals.items()}
    result['positive_batch_fraction'] = positive_batches/count
    if run and not training:
        run.log({'epoch': epoch + 1, **{f'val/{k}': v for k,v in result.items()}})
    return result
