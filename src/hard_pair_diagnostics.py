"""Read-only audit of labeled training crops; no automatic label corrections."""
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from prw_evaluation import _load_gallery_target, embed_detected_crops


@torch.inference_mode()
def select_hard_pairs(embeddings, identities, frame_ids, margin=.3, chunk_size=128):
    """Global hardest cross-frame positive and closest different-ID negative.

    Negatives may share the anchor frame, matching the training loss mask.
    This examines the whole supplied bank, not historical training batches.
    Returns one triplet per anchor with both a positive and a negative.
    """
    if chunk_size < 1 or margin < 0:
        raise ValueError('Require positive chunk_size and nonnegative margin')
    if embeddings.ndim != 2 or len(embeddings) != len(identities) or len(identities) != len(frame_ids):
        raise ValueError('Expected aligned embedding, identity and frame arrays')
    if not torch.isfinite(embeddings).all() or (identities < 0).any():
        raise ValueError('Expected finite embeddings and known identities')
    emb = F.normalize(embeddings.float(), dim=1)
    ids, frames = identities.to(emb.device), frame_ids.to(emb.device)
    rows = []
    for start in range(0, len(emb), chunk_size):
        distances = torch.cdist(emb[start:start+chunk_size], emb)
        same = ids[start:start+chunk_size, None] == ids[None, :]
        positive = same & (frames[start:start+chunk_size, None] != frames[None, :])
        negative = ~same
        valid = positive.any(1) & negative.any(1)
        pos_dist, pos_index = distances.masked_fill(~positive, -torch.inf).max(1)
        neg_dist, neg_index = distances.masked_fill(~negative, torch.inf).min(1)
        for local in valid.nonzero(as_tuple=True)[0].tolist():
            dp, dn = float(pos_dist[local]), float(neg_dist[local])
            rows.append(dict(anchor=start+local, positive=int(pos_index[local]),
                             negative=int(neg_index[local]), positive_distance=dp,
                             negative_distance=dn, gap=dn-dp,
                             triplet_loss=max(0., margin+dp-dn)))
    return sorted(rows, key=lambda row: (-row['triplet_loss'], row['anchor']))


@torch.inference_mode()
def audit_training_pairs(model, root, frame_names, device, margin=.3,
                         crop_batch_size=64, chunk_size=128):
    """Embed valid labeled original-pixel crops from the explicit training subset.

    No detector passes, augmentation, optimization or label changes. Keeps only
    embeddings and crop metadata in memory. Restores the model's training mode.
    """
    root = Path(root)
    names = list(dict.fromkeys(frame_names))
    metadata, features = [], []
    was_training = model.training
    model.eval()
    try:
        for frame_index, name in enumerate(names):
            boxes, ids = _load_gallery_target(root, name)
            keep = (ids >= 0) & (boxes[:, 2:] > boxes[:, :2]).all(1)
            boxes, ids = boxes[keep], ids[keep]
            if len(boxes):
                features.append(embed_detected_crops(model, root / 'frames' / f'{name}.jpg',
                                                     boxes, device, crop_batch_size))
                metadata.extend(dict(frame=name, frame_index=frame_index, identity=int(identity),
                                     box=box.tolist()) for box, identity in zip(boxes, ids))
            if (frame_index+1) % 100 == 0:
                print(f'Hard-pair audit: {frame_index+1}/{len(names)} frames, {len(metadata)} crops', flush=True)
        if not features:
            raise ValueError('No valid labeled training crops')
        embeddings = torch.cat(features).to(device)
        identities = torch.tensor([r['identity'] for r in metadata], device=device)
        frames = torch.tensor([r['frame_index'] for r in metadata], device=device)
        rows = select_hard_pairs(embeddings, identities, frames, margin, chunk_size)
    finally:
        model.train(was_training)
    return dict(scope='global_training_bank_not_historical_batches', margin=margin,
                frames=len(names), crops=len(metadata), valid_anchors=len(rows),
                metadata=metadata, triplets=rows)


def diverse_triplets(audit, limit=50):
    """Worst violation per unordered identity pair, avoiding repeated pair previews."""
    selected, seen = [], set()
    for row in audit['triplets']:
        a, n = (audit['metadata'][row[key]]['identity'] for key in ('anchor', 'negative'))
        pair = tuple(sorted((a, n)))
        if pair not in seen:
            selected.append(row)
            seen.add(pair)
        if len(selected) >= limit:
            break
    return selected if limit > 0 else []


def plot_hard_triplet(audit, row, root):
    """Show original crops; similarity alone is not evidence of a label error."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    for ax, key in zip(axes, ('anchor', 'positive', 'negative')):
        item = audit['metadata'][row[key]]
        with Image.open(Path(root) / 'frames' / f"{item['frame']}.jpg") as image:
            ax.imshow(image.convert('RGB').crop(tuple(item['box'])))
        distance = '' if key == 'anchor' else f"\ndistance={row[key+'_distance']:.4f}"
        ax.set_title(f"{key.title()} | ID {item['identity']}\n{item['frame']}{distance}")
        ax.axis('off')
    fig.suptitle(f"Global hard triplet | loss={row['triplet_loss']:.4f}, gap={row['gap']:.4f}\n"
                 'Different labels do not necessarily mean different people: inspect before judging.')
    fig.tight_layout()
    return fig
