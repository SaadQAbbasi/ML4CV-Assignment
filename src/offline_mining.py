"""Periodic training-crop mining and mixed hard/random P-K batch construction."""
import random
from time import perf_counter
import torch
from torch.nn import functional as F
from prw_dataset import PKIdentityBatchSampler


@torch.inference_mode()
def identity_neighbors(embeddings, ids, top_k=20, chunk_size=256):
    """Exact maximum crop cosine per distinct identity pair, computed in chunks.

    Each crop participates. The maximum focuses on visually confusing occurrences,
    rather than averaging them away in an identity centroid. Keep only a bounded
    neighbor list; randomized selection and random identities temper outliers.
    """
    if top_k < 1 or chunk_size < 1:
        raise ValueError('top_k and chunk_size must be positive')
    if len(embeddings) != len(ids) or embeddings.ndim != 2:
        raise ValueError('Expected one identity per embedding row')
    if not torch.isfinite(embeddings).all():
        raise ValueError('Nonfinite mining embeddings')
    labels, inverse = ids.unique(sorted=True, return_inverse=True)
    if (labels < 0).any():
        raise ValueError('Mining requires labeled identities')
    n_ids = len(labels)
    if n_ids < 2:
        return {int(i): [] for i in labels.tolist()}
    embeddings = F.normalize(embeddings.float(), dim=1)
    scores = embeddings.new_full((n_ids, n_ids), -float('inf'))
    for start in range(0, len(ids), chunk_size):
        similarity = embeddings[start:start + chunk_size] @ embeddings.T
        per_target = embeddings.new_full((len(similarity), n_ids), -float('inf'))
        per_target.scatter_reduce_(1, inverse.expand(len(similarity), -1), similarity,
                                   reduce='amax', include_self=True)
        scores.scatter_reduce_(0, inverse[start:start + len(similarity), None].expand(-1, n_ids),
                               per_target, reduce='amax', include_self=True)
    scores.fill_diagonal_(-float('inf'))
    order = scores.argsort(dim=1, descending=True, stable=True)[:, :min(top_k, n_ids - 1)]
    label_list = labels.cpu().tolist()
    return {int(identity): [int(label_list[j]) for j in row]
            for identity, row in zip(label_list, order.cpu().tolist())}


@torch.inference_mode()
def mine_training_neighbors(reid, loader, device, eligible_identities,
                            top_k=20, crop_batch_size=64, similarity_chunk_size=256):
    """loader must traverse the training subset sequentially, NOT the P-K sampler.

    Embeds every labeled crop in that subset. Only K-eligible training identities
    enter neighbor selection. No filesystem embedding cache or test data is used.
    """
    if crop_batch_size < 1:
        raise ValueError('crop_batch_size must be positive')
    started, was_training = perf_counter(), reid.training
    features, labels = [], []
    frame_count = crop_count = 0
    reid.eval()
    try:
        for _, targets in loader:
            frame_count += len(targets)
            crops = torch.cat([t['query_crops'] for t in targets])
            ids = torch.cat([t['ids'][t['ids'] >= 0] for t in targets])
            if len(crops) != len(ids):
                raise ValueError('Crop/identity alignment mismatch in mining loader')
            for start in range(0, len(crops), crop_batch_size):
                features.append(reid(crops[start:start + crop_batch_size].to(device)).float().cpu())
                labels.append(ids[start:start + crop_batch_size].cpu())
            crop_count += len(crops)
            if frame_count % 100 < len(targets):
                print(f'Offline mining: {frame_count} training frames, {crop_count} crops', flush=True)
        if not features:
            raise ValueError('No labeled training crops available for mining')
        features, labels = torch.cat(features), torch.cat(labels)
        keep = torch.isin(labels, torch.tensor(eligible_identities, dtype=labels.dtype))
        if not keep.any():
            raise ValueError('No K-eligible identities in mining embeddings')
        neighbors = identity_neighbors(features[keep].to(device), labels[keep].to(device),
                                       top_k=top_k, chunk_size=similarity_chunk_size)
    finally:
        reid.train(was_training)
    return neighbors, dict(frames=frame_count, crops=crop_count, identities=len(neighbors),
                            seconds=perf_counter() - started)


class MinedPKIdentityBatchSampler(PKIdentityBatchSampler):
    """Same epoch budget as ordinary P-K, with disjoint selected identity groups.

    Pick an anchor, then up to floor((P-1)*hard_fraction) neighbors at random from
    its top-neighbor list. Fill remaining slots randomly. Select K distinct frames
    per identity as usual. As identities are consumed, unavailable neighbors cause
    a random fallback. Never invent duplicates to fill a group.
    """
    def __init__(self, dataset, p=8, k=3, seed=0, hard_fraction=.5):
        super().__init__(dataset, p=p, k=k, seed=seed)
        if not 0 <= hard_fraction < 1:
            raise ValueError('hard_fraction must be in [0,1), preserving random slots')
        self.hard_fraction, self.epoch = hard_fraction, 0
        self.neighbors = {}
        self.last_stats = {}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_neighbors(self, neighbors):
        eligible = set(self.eligible_identities)
        self.neighbors = {int(i): list(dict.fromkeys(int(j) for j in values
                              if int(j) in eligible and int(j) != int(i)))
                          for i, values in neighbors.items() if int(i) in eligible}

    def identity_batches(self):
        rng = random.Random(self.seed + self.epoch)
        remaining = self.eligible_identities.copy()
        rng.shuffle(remaining)
        hard_total, batches = 0, 0
        while len(remaining) >= self.p:
            anchor = remaining.pop()
            hard_pool = [i for i in self.neighbors.get(anchor, []) if i in remaining]
            hard = rng.sample(hard_pool, min(len(hard_pool), int((self.p - 1) * self.hard_fraction)))
            random_pool = [i for i in remaining if i not in hard]
            random_ids = rng.sample(random_pool, self.p - 1 - len(hard))
            group = [anchor] + hard + random_ids
            chosen = set(group)
            remaining = [i for i in remaining if i not in chosen]
            hard_total += len(hard)
            batches += 1
            yield group
        self.last_stats = dict(batches=batches, hard_identity_slots=hard_total,
                               hard_slot_fraction=hard_total / max(batches * (self.p - 1), 1))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch + 100000)
        for group in self.identity_batches():
            frames = []
            for identity in group:
                frames.extend(rng.sample(sorted(set(self.identity_to_indices[identity])), self.k))
            yield list(dict.fromkeys(frames))
