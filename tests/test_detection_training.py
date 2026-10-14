import unittest
from unittest.mock import patch
import torch
from model import PersonSearchModel
from detection_training import detection_metrics

class DetectionTests(unittest.TestCase):
    def test_metrics_misses_and_duplicates(self):
        box=torch.tensor([[0.,0.,10.,10.]])
        metrics=detection_metrics([dict(boxes=box.repeat(2,1),scores=torch.tensor([.9,.8]))],[box.repeat(2,1)])
        self.assertAlmostEqual(metrics['AP50'],1.)
        metrics=detection_metrics([dict(boxes=box,scores=torch.tensor([.9]))],[torch.cat([box,box+20])])
        self.assertEqual(metrics['recall_iou50'],.5)
        self.assertAlmostEqual(metrics['AP50'],51/101)
        empty=detection_metrics([dict(boxes=torch.empty(0,4),scores=torch.empty(0))],[box])
        self.assertEqual(empty['AP50'],0.)

    def test_detection_only_skips_reid_training_and_inference(self):
        model=PersonSearchModel(pretrained=False,detection_only=True,
            rpn_pre_nms_top_n_train=20,rpn_post_nms_top_n_train=10,
            rpn_pre_nms_top_n_test=20,rpn_post_nms_top_n_test=10)
        images=torch.randn(2,3,64,64)
        targets=[dict(boxes=torch.tensor([[8.,5.,35.,60.]]),ids=torch.tensor([-1])) for _ in range(2)]
        with patch.object(model.reid_head,'forward',side_effect=AssertionError('ReID called')):
            losses=model(images,targets)
            self.assertEqual(len(losses),4)
            sum(losses.values()).backward()
            self.assertIsNone(model.reid_head.embedding_proj.weight.grad)
            self.assertIsNotNone(model.det_head.cls_score.weight.grad)
            model.eval()
            with torch.no_grad():self.assertEqual(len(model(images)),2)
