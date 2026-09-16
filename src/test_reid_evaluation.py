"""Synthetic retrieval checks: detector misses versus annotated crop retrieval."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import prw_evaluation as evaluation


class FakeModel:
    det_score_thresh = .5
    def eval(self):
        return self
    def __call__(self, batch):
        return [dict(boxes=torch.tensor([[0., 0., 10., 10.]]),
                     scores=torch.tensor([.9])) for _ in batch]
    def embed_crop(self, crops):
        return crops


class RetrievalEvaluationTests(unittest.TestCase):
    def test_gt_bypasses_detector_and_preserves_query_protocol(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.],
                              [40., 0., 50., 10.]])
        identities = torch.tensor([1, 2, -1])
        queries = [dict(identity=i, frame='a', box=boxes[i-1].tolist()) for i in (1, 2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'query_box').mkdir()
            for i in (1, 2):
                (root / 'query_box' / f'{i}_a.jpg').touch()
            with patch.object(evaluation, '_load_mat_table', return_value=torch.zeros(1)), \
                 patch.object(evaluation, '_mat_strings', return_value=['a', 'b']), \
                 patch.object(evaluation, '_image_tensor', return_value=(torch.zeros(3, 8, 8), 1.)), \
                 patch.object(evaluation, '_load_gallery_target', side_effect=lambda *a: (boxes.clone(), identities.clone())), \
                 patch.object(evaluation, '_load_query_info', return_value=queries), \
                 patch.object(evaluation, '_load_query_crop', side_effect=lambda p, size: torch.eye(3)[int(p.name[0])-1]), \
                 patch.object(evaluation, 'embed_detected_crops', side_effect=lambda m,p,b,d,bs: torch.eye(3)[:len(b)]):
                model = FakeModel()
                detected = evaluation.evaluate_prw(model, root, 'cpu')
                with patch.object(FakeModel, '__call__', side_effect=AssertionError('Detector invoked in GT mode')):
                    gt = evaluation.evaluate_prw(model, root, 'cpu', gallery_mode='gt', return_records=True)
        self.assertEqual(detected['retrieval_queries'], 2)
        self.assertEqual(gt['retrieval_queries'], 2)
        self.assertEqual(detected['retrieval_mAP'], .5)
        self.assertEqual(gt['retrieval_mAP'], 1.)
        self.assertEqual(gt['retrieval_top1'], 1.)
        self.assertEqual(gt['retrieval_detections'], 6)  # unknown people retained
        self.assertAlmostEqual(detected['detection_recall_iou50'], 1/3)
        for record in gt['retrieval_records']:
            self.assertTrue(all(r['frame'] == 'b' for r in record['ranked']))
            self.assertTrue(all(r['person_confidence'] is None for r in record['ranked']))


if __name__ == '__main__':
    unittest.main()
