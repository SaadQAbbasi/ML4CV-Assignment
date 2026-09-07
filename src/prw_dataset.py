"""PRW dataset helpers for person-search training and checks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import normalize, pil_to_tensor, resize


def _mat_strings(value) -> List[str]:
    """Flatten MATLAB string/object arrays into Python strings."""
    array = np.asarray(value)
    if array.size == 0:
        return []
    items = []
    for entry in array.reshape(-1):
        if isinstance(entry, bytes):
            items.append(entry.decode("utf-8"))
        elif isinstance(entry, str):
            items.append(entry)
        else:
            nested = np.asarray(entry).reshape(-1)
            if nested.size == 1:
                scalar = nested[0]
                if isinstance(scalar, bytes):
                    items.append(scalar.decode("utf-8"))
                else:
                    items.append(str(scalar))
            else:
                items.append("".join(str(x) for x in nested))
    return items


def _load_mat_table(path: Path) -> np.ndarray:
    """Load the first non-private MATLAB variable from a PRW mat file."""
    mat = loadmat(path)
    for key, value in mat.items():
        if not key.startswith("__") and isinstance(value, np.ndarray):
            return value
    raise ValueError(f"no array data found in {path}")


def _identity_from_box(box_entry: np.ndarray) -> int:
    """Convert PRW annotation IDs to the training convention."""
    identity = int(box_entry[0])
    return -1 if identity < 0 else identity


def _annotation_boxes(annotation: np.ndarray) -> torch.Tensor:
    boxes = torch.as_tensor(annotation[:, 1:5], dtype=torch.float32)
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class PRWPersonSearchDataset(Dataset):
    """PRW frames with per-image person boxes and identity labels."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (256, 384),
        image_mean: tuple[float, float, float] = (.485, .456, .406),
        image_std: tuple[float, float, float] = (.229, .224, .225),
    ):
        self.root = Path(root)
        self.split = split
        self.image_h, self.image_w = image_size
        self.image_mean = image_mean
        self.image_std = image_std

        split_file = self.root / f"frame_{split}.mat"
        split_mat = loadmat(split_file)
        key = f"img_index_{split}"
        if key not in split_mat:
            raise KeyError(f"{split_file} missing expected key {key}")
        self.image_names = _mat_strings(split_mat[key])
        self.frames_dir = self.root / "frames"
        self.annotations_dir = self.root / "annotations"

    def __len__(self) -> int:
        return len(self.image_names)

    def _load_image(self, name: str) -> torch.Tensor:
        image = pil_to_tensor(Image.open(self.frames_dir / f"{name}.jpg").convert("RGB")).float() / 255.0
        _, h, w = image.shape
        scale = min(self.image_h / h, self.image_w / w)
        resized_h, resized_w = round(h * scale), round(w * scale)
        image = resize(image, [resized_h, resized_w])
        image = normalize(image, self.image_mean, self.image_std)
        padded = image.new_zeros(3, self.image_h, self.image_w)
        padded[:, :resized_h, :resized_w] = image
        return padded

    def _load_target(self, name: str, orig_size: tuple[int, int]) -> Dict[str, torch.Tensor]:
        annotation = _load_mat_table(self.annotations_dir / f"{name}.jpg.mat")
        if annotation.ndim != 2 or annotation.shape[1] != 5:
            raise ValueError(f"unexpected PRW annotation shape for {name}: {annotation.shape}")
        ids = torch.as_tensor([_identity_from_box(row) for row in annotation], dtype=torch.long)
        boxes = _annotation_boxes(annotation)
        h, w = orig_size
        scale = min(self.image_h / h, self.image_w / w)
        resized_h, resized_w = round(h * scale), round(w * scale)
        boxes[:, 0::2] *= resized_w / w
        boxes[:, 1::2] *= resized_h / h
        valid = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
        return {"boxes": boxes[valid], "ids": ids[valid]}

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        raw = Image.open(self.frames_dir / f"{name}.jpg").convert("RGB")
        image = pil_to_tensor(raw).float() / 255.0
        _, h, w = image.shape
        target = self._load_target(name, (h, w))
        return self._load_image(name), target

    def identity_to_indices(self):
        mapping = defaultdict(list)
        for idx, name in enumerate(self.image_names):
            annotation = _load_mat_table(self.annotations_dir / f"{name}.jpg.mat")
            for row in annotation:
                identity = _identity_from_box(row)
                if identity >= 0:
                    mapping[identity].append(idx)
        return dict(mapping)


class PRWSubset(Dataset):
    """Index subset that preserves the identity mapping required by PK sampling."""

    def __init__(self, dataset: PRWPersonSearchDataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def identity_to_indices(self):
        base_to_subset = {base_index: subset_index
                          for subset_index, base_index in enumerate(self.indices)}
        mapping = defaultdict(list)
        for identity, base_indices in self.dataset.identity_to_indices().items():
            for base_index in base_indices:
                if base_index in base_to_subset:
                    mapping[identity].append(base_to_subset[base_index])
        return dict(mapping)


class PKIdentityBatchSampler(Sampler):
    """Yield batches covering P identities with K distinct frames per identity."""

    def __init__(self, dataset, p: int = 4, k: int = 2):
        self.p = p
        self.k = k
        self.identity_to_indices = dataset.identity_to_indices()
        self.eligible_identities = [identity for identity, indices in self.identity_to_indices.items()
                                    if len(set(indices)) >= k]
        if len(self.eligible_identities) < p:
            raise ValueError(f"only {len(self.eligible_identities)} identities have at least {k} frames")

    def __iter__(self):
        identities = self.eligible_identities.copy()
        random.shuffle(identities)
        for start in range(0, len(identities) - self.p + 1, self.p):
            batch = []
            for identity in identities[start:start + self.p]:
                batch.extend(random.sample(list(set(self.identity_to_indices[identity])), self.k))
            yield list(dict.fromkeys(batch))

    def __len__(self):
        return len(self.eligible_identities) // self.p


def prw_collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
