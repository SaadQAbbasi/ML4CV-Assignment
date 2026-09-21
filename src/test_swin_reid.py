import unittest
from unittest.mock import patch
import torch
from model import PersonSearchModel
from swin_reid import SwinSceneReID,SwinCropReID

class SwinReIDTests(unittest.TestCase):
    def test_scene_detector_frozen_one_backbone_pass(self):
        detector=PersonSearchModel(pretrained=False,detection_only=True,
            rpn_pre_nms_top_n_test=10,rpn_post_nms_top_n_test=5)
        model=SwinSceneReID(detector).train()
        x=torch.randn(2,3,64,64)
        boxes=[torch.tensor([[0.,0.,40.,60.]]) for _ in range(2)]
        emb=model.embed_scene(x,boxes)
        emb[:,0].sum().backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in detector.parameters()))
        self.assertIsNotNone(model.reid_head.embedding_proj.weight.grad)
        self.assertFalse(detector.training)
        model.eval()
        with patch.object(detector,'_extract_pyramid',wraps=detector._extract_pyramid) as extract:
            with torch.no_grad():results=model(x)
            self.assertEqual(extract.call_count,1)
        self.assertEqual(len(results),2)
        self.assertEqual(results[0]['embeddings'].shape[1],256)

    def test_crop_encoder_output_and_stage_freezing(self):
        model=SwinCropReID(pretrained=False).eval()
        self.assertFalse(any(p.requires_grad for p in model.features[0].parameters()))
        self.assertTrue(any(p.requires_grad for p in model.features[7].parameters()))
        with torch.no_grad():emb=model(torch.randn(2,3,256,128))
        self.assertEqual(tuple(emb.shape),(2,256))
        self.assertTrue(torch.allclose(emb.norm(dim=1),torch.ones(2),atol=1e-5))
