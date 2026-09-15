import unittest
from unittest.mock import patch
import torch
from torch.nn.functional import normalize
from hard_pair_diagnostics import select_hard_pairs, diverse_triplets, audit_training_pairs


class HardPairTests(unittest.TestCase):
    def test_matches_brute_force_and_cross_frame_mask(self):
        emb = normalize(torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.99, .01], [-1., 0.]]), dim=1)
        ids = torch.tensor([1, 1, 1, 2, 3])
        frames = torch.tensor([0, 0, 1, 0, 2])
        rows = select_hard_pairs(emb, ids, frames, chunk_size=2)
        self.assertEqual(len(rows), 3)  # singleton IDs cannot be anchors
        distances = torch.cdist(emb, emb)
        for row in rows:
            a = row['anchor']
            positives = [j for j in range(5) if ids[j] == ids[a] and frames[j] != frames[a]]
            negatives = [j for j in range(5) if ids[j] != ids[a]]
            self.assertEqual(row['positive'], max(positives, key=lambda j: distances[a,j]))
            self.assertEqual(row['negative'], min(negatives, key=lambda j: distances[a,j]))
            self.assertAlmostEqual(row['triplet_loss'], max(0., .3+float(distances[a,row['positive']])-float(distances[a,row['negative']])), places=5)
        anchor0 = next(r for r in rows if r['anchor']==0)
        self.assertEqual(anchor0['negative'], 3)  # same-frame different ID is valid
        other = select_hard_pairs(emb, ids, frames, chunk_size=5)
        self.assertEqual(rows, other)
        audit = dict(triplets=rows, metadata=[dict(identity=int(i)) for i in ids])
        self.assertEqual(len(diverse_triplets(audit)), 1)

    def test_empty_and_missing_pairs(self):
        self.assertEqual(select_hard_pairs(torch.empty(0,2), torch.empty(0,dtype=torch.long), torch.empty(0,dtype=torch.long)), [])
        self.assertEqual(select_hard_pairs(torch.eye(2), torch.tensor([1,1]), torch.tensor([0,1])), [])

    def test_audit_filters_unknown_and_invalid_and_restores_mode(self):
        model = torch.nn.Linear(2,2)
        model.train()
        boxes = torch.tensor([[0.,0.,2.,2.], [0.,0.,2.,2.], [0.,0.,0.,2.]])
        with patch('hard_pair_diagnostics._load_gallery_target', return_value=(boxes, torch.tensor([1,-1,2]))), \
             patch('hard_pair_diagnostics.embed_detected_crops', return_value=torch.tensor([[1.,0.]])) as embed:
            result = audit_training_pairs(model, '.', ['a','b'], 'cpu')
        self.assertTrue(model.training)
        self.assertEqual(result['crops'], 2)
        self.assertEqual(result['valid_anchors'], 0)
        self.assertTrue(all(len(call.args[2])==1 for call in embed.call_args_list))
        with patch('hard_pair_diagnostics._load_gallery_target', side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                audit_training_pairs(model, '.', ['a'], 'cpu')
        self.assertTrue(model.training)

if __name__ == '__main__':
    unittest.main()
