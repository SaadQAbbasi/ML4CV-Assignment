"""Focused CPU tests: python -m unittest discover -s src -p 'test_reid_paths.py'."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from metric_loss import ContrastiveLoss, BatchHardTripletLoss, cross_image_reid_loss
from prw_dataset import PRWPersonSearchDataset, preprocess_query_crop, PKIdentityBatchSampler


class ReIDPathsTest(unittest.TestCase):
    def test_supcon_reference_and_gradients(self):
        from metric_loss import SupervisedNTXentLoss, crop_to_crop_reid_loss
        import torch.nn.functional as F
        emb = torch.randn(5, 8, requires_grad=True)
        ids = torch.tensor([1, 1, 1, 2, 2])
        frames = torch.tensor([0, 0, 1, 0, 1])
        normalized = F.normalize(emb.float(), dim=1)
        values = []
        for i in range(5):
            pos = [j for j in range(5) if ids[i] == ids[j] and frames[i] != frames[j]]
            neg = [j for j in range(5) if ids[i] != ids[j]]
            logits = (normalized[i] @ normalized.T) / .1
            values.append(torch.logsumexp(logits[pos + neg], 0) - logits[pos].mean())
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss, stats = crop_to_crop_reid_loss(emb, ids, frames, SupervisedNTXentLoss(.1))
        self.assertTrue(torch.allclose(loss, torch.stack(values).mean(), atol=1e-5))
        self.assertNotIn('active_fraction', stats)
        loss.backward()
        self.assertTrue(torch.isfinite(emb.grad).all())
        self.assertGreater(emb.grad.abs().sum().item(), 0)

    def test_supcon_degenerate(self):
        from metric_loss import SupervisedNTXentLoss, crop_to_crop_reid_loss
        for labels in ([], [1], [1, 1], [1, 2]):
            emb = torch.randn(len(labels), 4, requires_grad=True)
            loss, _ = crop_to_crop_reid_loss(emb, torch.tensor(labels),
                torch.arange(len(labels)), SupervisedNTXentLoss())
            self.assertEqual(loss.item(), 0.)
            loss.backward()
        for temperature in (0, -1, float('nan')):
            with self.assertRaises(ValueError): SupervisedNTXentLoss(temperature)

    def test_crop_triplet_hard_mining(self):
        from metric_loss import crop_to_crop_reid_loss
        # Anchor 0 must ignore farther same-ID index 1 (same image),
        # and use index 2 from another image as its positive.
        emb = torch.tensor([[0., 0.], [2., 0.], [.2, 0.], [.3, 0.]], requires_grad=True)
        ids = torch.tensor([1, 1, 1, 2])
        frames = torch.tensor([0, 0, 1, 1])
        loss, stats = crop_to_crop_reid_loss(emb, ids, frames, BatchHardTripletLoss(.3))
        # Valid anchors 0,1,2: hinge losses .2, .4, 2.0; singleton ID2 excluded.
        self.assertAlmostEqual(loss.item(), (.2 + .4 + 2.) / 3, places=6)
        self.assertEqual(stats['valid_anchors'], 3)
        self.assertEqual(stats['active_fraction'], 1.)
        loss.backward()
        self.assertTrue(torch.isfinite(emb.grad).all())
        self.assertGreater(emb.grad.abs().sum().item(), 0)

    def test_crop_triplet_missing_pairs_and_satisfied_margin(self):
        from metric_loss import crop_to_crop_reid_loss
        for positions, ids, frames in [
            ([], [], []),
            ([0., .1], [1, 1], [0, 1]),  # no negative
            ([0., .1], [1, 2], [0, 1]),  # no positive
            ([0., .1], [1, 1], [0, 0]),  # same-source only
            ([0., .1, 1., 1.1], [1, 1, 2, 2], [0, 1, 0, 1]),
        ]:
            emb = torch.tensor(positions).reshape(-1, 1).requires_grad_()
            loss, stats = crop_to_crop_reid_loss(
                emb, torch.tensor(ids), torch.tensor(frames), BatchHardTripletLoss(.3))
            self.assertEqual(loss.item(), 0.)
            self.assertEqual(stats['active_fraction'], 0.)
            loss.backward()
            self.assertTrue(torch.isfinite(emb.grad).all())


    def test_crop_to_crop_balancing_and_exclusion(self):
        from metric_loss import crop_to_crop_reid_loss
        emb = torch.tensor([[0., 0.], [.2, 0.], [.4, 0.], [.5, 0.]], requires_grad=True)
        ids = torch.tensor([1, 1, 1, 2])
        frames = torch.tensor([0, 0, 1, 1])
        loss, stats = crop_to_crop_reid_loss(emb, ids, frames, ContrastiveLoss())
        # Same-image pair (0,1) and all self-pairs are absent.
        expected_positive = (.4**2 + .2**2) / 2
        expected_negative = (.5**2 + .7**2 + .9**2) / 3
        self.assertAlmostEqual(loss.item(), (expected_positive + expected_negative) / 2, places=6)
        self.assertEqual(stats['num_pos_pairs'], 4)
        self.assertEqual(stats['num_neg_pairs'], 6)
        loss.backward()
        self.assertTrue(torch.isfinite(emb.grad).all())
        self.assertGreater(emb.grad.abs().sum().item(), 0)


    def test_fixed_pk_validation_preserves_rng_and_identity_coverage(self):
        import random
        class Dataset:
            def identity_to_indices(self):
                return {i: [i, i + 1, i + 2] for i in range(8)}
        dataset = Dataset()
        sampler = PKIdentityBatchSampler(dataset, p=4, k=2, seed=17)
        state = random.getstate()
        batches = list(sampler)
        self.assertEqual(batches, list(sampler))
        self.assertEqual(state, random.getstate())
        self.assertEqual(len(batches), len(sampler))
        for batch in batches:
            self.assertEqual(len(batch), len(set(batch)))
            covered = sum(len(set(frames) & set(batch)) >= 2
                          for frames in dataset.identity_to_indices().values())
            self.assertGreaterEqual(covered, 4)


    def test_validation_distances_exclude_batches_without_pairs(self):
        from training import validate, LOSS_KEYS
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.backbone = torch.nn.Identity()
            def forward(self, images, targets=None):
                if not self.training:
                    return [dict(boxes=torch.empty(0, 4), embeddings=torch.empty(0, 8),
                                 scores=torch.empty(0))]
                self.calls += 1
                result = {key: torch.tensor(0.) for key in LOSS_KEYS}
                positive = self.calls == 2
                result.update(_metric_positive_pairs=torch.tensor(int(positive)),
                              _metric_negative_pairs=torch.tensor(1),
                              _hardest_positive_distance=torch.tensor(.8 if positive else 0.),
                              _hardest_negative_distance=torch.tensor(.5),
                              _metric_active_fraction=torch.tensor(.5))
                return result
        loader = [(torch.zeros(1, 3, 32, 32), [dict(boxes=torch.empty(0, 4))])] * 2
        model = Model()
        state = torch.random.get_rng_state()
        metrics = validate(model, loader, torch.device('cpu'))
        self.assertAlmostEqual(metrics['hardest_positive_distance'], .8)
        self.assertEqual(metrics['positive_batch_fraction'], .5)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(model.training)

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
            dataset.query_crop_size = (128, 64)
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
