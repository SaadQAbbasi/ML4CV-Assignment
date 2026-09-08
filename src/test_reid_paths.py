"""Focused CPU tests: python -m unittest discover -s src -p 'test_reid_paths.py'."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from metric_loss import ContrastiveLoss, BatchHardTripletLoss, cross_image_reid_loss
from prw_dataset import PRWPersonSearchDataset, preprocess_query_crop


class ReIDPathsTest(unittest.TestCase):
    def test_cross_image_pairs_and_gradients(self):
        scene = torch.randn(4, 8, requires_grad=True)
        crops = torch.randn(4, 8, requires_grad=True)
        ids = torch.tensor([1, 1, 2, 2])
        frames = torch.tensor([0, 1, 0, 1])
        for criterion in (ContrastiveLoss(), BatchHardTripletLoss(margin=10)):
            loss, stats = cross_image_reid_loss(scene, crops, ids, frames, criterion)
            self.assertEqual(stats['num_pos_pairs'], 4)
            self.assertEqual(stats['num_neg_pairs'], 8)
            self.assertEqual(stats['cross_image_identities'], 2)
            loss.backward()
            self.assertGreater(scene.grad.abs().sum().item(), 0)
            self.assertGreater(crops.grad.abs().sum().item(), 0)
            scene.grad = crops.grad = None

    def test_same_image_is_not_a_positive(self):
        emb = torch.randn(2, 8, requires_grad=True)
        loss, stats = cross_image_reid_loss(emb, emb, torch.tensor([1, 1]),
                                           torch.tensor([0, 0]), ContrastiveLoss())
        self.assertEqual(stats['num_pos_pairs'], 0)
        self.assertEqual(stats['num_neg_pairs'], 0)
        self.assertEqual(loss.item(), 0)
        loss.backward()

    def test_empty_and_single_identity(self):
        for ids, frames in (([], []), ([1, 1], [0, 1]), ([1, 2], [0, 1])):
            emb = torch.randn(len(ids), 8, requires_grad=True)
            for criterion in (ContrastiveLoss(), BatchHardTripletLoss()):
                loss, _ = cross_image_reid_loss(emb, emb, torch.tensor(ids),
                                               torch.tensor(frames), criterion)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()

    def test_original_resolution_crops_use_query_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = Image.fromarray(np.random.default_rng(7).integers(0, 256, (80, 120, 3), dtype=np.uint8))
            raw.save(root / 'frame.jpg')
            dataset = PRWPersonSearchDataset.__new__(PRWPersonSearchDataset)
            dataset.frames_dir = dataset.annotations_dir = root
            dataset.image_names = ['frame']
            dataset.image_h, dataset.image_w = 32, 48
            dataset.image_mean = (.485, .456, .406)
            dataset.image_std = (.229, .224, .225)
            annotation = np.array([[7, 10, 5, 20, 40], [-1, 40, 5, 20, 40],
                                   [8, 70, 5, 0, 40]], dtype=np.float32)
            with patch('prw_dataset._load_mat_table', side_effect=lambda _: annotation.copy()):
                image, target = dataset[0]
            with Image.open(root / 'frame.jpg') as source:
                expected = preprocess_query_crop(source.crop((10, 5, 30, 45)))
            self.assertEqual(tuple(image.shape), (3, 32, 48))
            self.assertEqual(target['ids'].tolist(), [7, -1])
            self.assertEqual(tuple(target['query_crops'].shape), (1, 3, 128, 64))
            self.assertTrue(torch.equal(target['query_crops'][0], expected))
            self.assertEqual(target['image_id'].item(), 0)


if __name__ == '__main__':
    unittest.main()
