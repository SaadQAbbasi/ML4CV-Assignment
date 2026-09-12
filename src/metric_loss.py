"""Identity losses and cross-image crop-to-scene supervision.

PersonSearchModel uses crop_to_crop_reid_loss on standalone labeled GT crops. Same-source positives are excluded. The standalone
triplet/contrastive modules remain available for embedding-only checks.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def filter_labeled_roIs(embeddings: torch.Tensor, cls_labels: torch.Tensor,
                         matched_gt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    embeddings:      (N, D)
    cls_labels:      (N,) in {0, 1} -- from ROITargetAssigner
    matched_gt_ids:  (N,) int, -1 for background OR unlabeled person, >=0 for
                     a real identity -- also from ROITargetAssigner

    returns only the (embedding, id) pairs eligible for metric learning:
    cls_label==1 AND matched_gt_id>=0. Order is preserved but rows are
    dropped, so the returned tensors are shorter than the inputs.
    """
    keep = (cls_labels == 1) & (matched_gt_ids >= 0)
    return embeddings[keep], matched_gt_ids[keep]


class BatchHardTripletLoss(nn.Module):
    """
    Standard batch-hard triplet loss (Hermans et al., 'In Defense of the
    Triplet Loss for Person Re-Identification'). For every anchor in the
    batch:
        hardest positive = same identity, MAX embedding distance
        hardest negative = different identity, MIN embedding distance
        loss = max(0, margin + d(anchor,hardest_pos) - d(anchor,hardest_neg))

    Batch-hard (vs. naively enumerating all triplets) focuses gradient on
    the actually-confusing examples and is the standard choice for ReID.

    Embeddings are assumed already L2-normalized (ReIDHead does this).
    Euclidean distance is used, bounded in [0, 2].
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, ids: torch.Tensor
                ) -> Tuple[torch.Tensor, dict]:
        """
        Returns (loss, stats). stats reports how many anchors actually had
        BOTH a valid positive and a valid negative in the batch -- an
        anchor whose identity appears only once (no positive) or whose
        identity is the only one present (no negative) contributes 0 and is
        excluded from the mean; stats lets the caller notice if that's
        happening for most/all of the batch (see module docstring warning
        about needing P-identities-x-K-images sampling).
        """
        n = embeddings.shape[0]
        if n < 2:
            return embeddings.sum() * 0.0, {"valid_anchors": 0, "total_anchors": n}

        dist = torch.cdist(embeddings.float(), embeddings.float(), p=2)  # (n, n) pairwise distances

        same_id = ids.unsqueeze(0) == ids.unsqueeze(1)          # (n, n)
        diff_id = ~same_id
        not_self = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)

        pos_mask = same_id & not_self
        neg_mask = diff_id

        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        valid = has_pos & has_neg

        if valid.sum() == 0:
            return embeddings.sum() * 0.0, {"valid_anchors": 0, "total_anchors": n}

        dist_pos_only = dist.masked_fill(~pos_mask, float("-inf"))
        hardest_pos, _ = dist_pos_only.max(dim=1)

        dist_neg_only = dist.masked_fill(~neg_mask, float("inf"))
        hardest_neg, _ = dist_neg_only.min(dim=1)

        losses = F.relu(self.margin + hardest_pos - hardest_neg)
        loss = losses[valid].mean()

        return loss, {"valid_anchors": int(valid.sum()), "total_anchors": n,
                      "active_fraction": float((losses[valid] > 0).float().mean().detach()),
                      "hardest_positive_distance": float(hardest_pos[valid].mean().detach()),
                      "hardest_negative_distance": float(hardest_neg[valid].mean().detach())}


class ContrastiveLoss(nn.Module):
    """
    Pairwise contrastive loss over all pairs in the batch (simpler
    alternative to triplet, offered since the pipeline notes mentioned
    'Contrastive loss, Triplet loss etc' -- pick one at the call site):
        same identity:      loss = d(a,b)^2
        different identity: loss = max(0, margin - d(a,b))^2
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, ids: torch.Tensor
                ) -> Tuple[torch.Tensor, dict]:
        n = embeddings.shape[0]
        if n < 2:
            return embeddings.sum() * 0.0, {"num_pairs": 0}

        dist = torch.cdist(embeddings.float(), embeddings.float(), p=2)
        same_id = ids.unsqueeze(0) == ids.unsqueeze(1)
        not_self = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)

        iu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=embeddings.device), diagonal=1)
        pair_mask = iu & not_self  # each unordered pair counted once

        pos_pairs = pair_mask & same_id
        neg_pairs = pair_mask & ~same_id

        pos_loss = dist[pos_pairs].pow(2)
        neg_loss = F.relu(self.margin - dist[neg_pairs]).pow(2)

        all_losses = torch.cat([pos_loss, neg_loss])
        if all_losses.numel() == 0:
            return embeddings.sum() * 0.0, {"num_pairs": 0}

        loss = all_losses.mean()
        return loss, {"num_pairs": int(all_losses.numel()),
                       "num_pos_pairs": int(pos_pairs.sum()), "num_neg_pairs": int(neg_pairs.sum())}


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- Case 1: healthy batch -- identities 1,2,3 each appear >=2x,
    #     plus background (label 0) and unlabeled-person (label 1, id -1)
    #     rows mixed in, exactly like real ROITargetAssigner output.
    cls_labels = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 1])
    matched_gt_ids = torch.tensor([1, 1, 2, 2, 3, 3, -1, -1, -1])  # last 1 = unlabeled person
    embeddings = F.normalize(torch.randn(9, 16), dim=1)

    filt_emb, filt_ids = filter_labeled_roIs(embeddings, cls_labels, matched_gt_ids)
    print(f"[Filter] {embeddings.shape[0]} RoIs -> {filt_emb.shape[0]} labeled-positive RoIs "
          f"(dropped {embeddings.shape[0]-filt_emb.shape[0]}: backgrounds + unlabeled persons)")
    assert filt_emb.shape[0] == 6 and (filt_ids >= 0).all()

    triplet = BatchHardTripletLoss(margin=0.3)
    loss, stats = triplet(filt_emb, filt_ids)
    print(f"[Triplet] loss={loss.item():.4f}  stats={stats}  "
          f"(expect valid_anchors=6/6 -- every id appears twice, so every "
          f"anchor has both a same-id positive and a different-id negative)")
    assert stats["valid_anchors"] == 6

    contrastive = ContrastiveLoss(margin=1.0)
    c_loss, c_stats = contrastive(filt_emb, filt_ids)
    print(f"[Contrastive] loss={c_loss.item():.4f}  stats={c_stats}")

    # --- Case 2: degenerate batch -- every identity appears exactly once
    #     (this is the realistic risk case flagged in chat: small batches /
    #     no P-identities-x-K-images sampler yet). Loss should NOT crash,
    #     should report 0 valid anchors so the caller can detect it.
    degenerate_ids = torch.tensor([10, 11, 12, 13])
    degenerate_emb = F.normalize(torch.randn(4, 16), dim=1)
    d_loss, d_stats = triplet(degenerate_emb, degenerate_ids)
    print(f"\n[Degenerate case] every id unique -> loss={d_loss.item():.4f} "
          f"stats={d_stats} (expect valid_anchors=0 -- no positive pairs exist, "
          f"this is the batch-sampling problem flagged earlier, not a bug here)")
    assert d_stats["valid_anchors"] == 0 and d_loss.item() == 0.0

    # --- Case 3: empty input (all RoIs were background/unlabeled this batch) ---
    empty_emb = torch.zeros((0, 16))
    empty_ids = torch.zeros((0,), dtype=torch.int64)
    e_loss, e_stats = triplet(empty_emb, empty_ids)
    print(f"\n[Empty case] no labeled RoIs at all -> loss={e_loss.item():.4f} "
          f"stats={e_stats} (must not crash)")


def cross_image_reid_loss(scene, crops, ids, image_ids, loss_fn):
    """Cross-path pairs: positives must share identity but not source image.

    Negatives are different identities. Same-occurrence crop/scene pairs are
    excluded rather than accidentally treated as negatives. Both branches
    receive gradients. Contrastive positive and negative groups are balanced.
    """
    zero = (scene.sum() + crops.sum()) * 0.0
    stats = dict(num_pos_pairs=0, num_neg_pairs=0, valid_anchors=0,
                 active_fraction=0., hardest_positive_distance=0.,
                 hardest_negative_distance=0., cross_image_identities=0)
    if not len(ids):
        return zero, stats
    distances = torch.cdist(crops.float(), scene.float())
    same = ids[:, None] == ids[None, :]
    positive = same & (image_ids[:, None] != image_ids[None, :])
    negative = ~same
    valid = positive.any(1) & negative.any(1)
    stats.update(num_pos_pairs=int(positive.sum()), num_neg_pairs=int(negative.sum()),
                 valid_anchors=int(valid.sum()),
                 cross_image_identities=int(ids[positive.any(1)].unique().numel()))
    if positive.any():
        stats["hardest_positive_distance"] = float(
            distances.masked_fill(~positive, -float("inf")).max(1).values[positive.any(1)].mean().detach())
    if negative.any():
        stats["hardest_negative_distance"] = float(
            distances.masked_fill(~negative, float("inf")).min(1).values[negative.any(1)].mean().detach())
    if isinstance(loss_fn, BatchHardTripletLoss):
        if not valid.any():
            return zero, stats
        hardest_pos = distances.masked_fill(~positive, -float("inf")).max(1).values[valid]
        hardest_neg = distances.masked_fill(~negative, float("inf")).min(1).values[valid]
        losses = F.relu(loss_fn.margin + hardest_pos - hardest_neg)
        stats["active_fraction"] = float((losses > 0).float().mean().detach())
        return losses.mean(), stats
    groups = []
    if positive.any():
        groups.append(distances[positive].square().mean())
    if negative.any():
        negative_losses = F.relu(loss_fn.margin - distances[negative]).square()
        groups.append(negative_losses.mean())
        stats["active_fraction"] = float((negative_losses > 0).float().mean().detach())
    return (torch.stack(groups).mean() if groups else zero), stats


def crop_to_crop_reid_loss(embeddings, ids, image_ids, loss_fn):
    """Experiment 04: same-ID positives must originate from different images.

    Reuse the cross-image masks with a single embedding bank. Diagonal and
    same-image same-ID entries are excluded from both terms. Different-ID
    negatives may share a source image. Counts are directed pairs (A,B and
    B,A); their symmetric means preserve the existing balanced loss scale.
    """
    return cross_image_reid_loss(embeddings, embeddings, ids, image_ids, loss_fn)
