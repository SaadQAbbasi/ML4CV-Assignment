"""Offline contract tests; no pretrained download required."""
import unittest
import torch
from torch import nn
from dino_reid import DinoCropReID, DinoPersonSearch, crop_epoch
from training import make_grad_scaler

class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.stem = nn.Conv2d(3, 8, 1)
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.norm = nn.LayerNorm(8)
    def forward_features(self, x):
        assert x.shape[-2] % 14 == x.shape[-1] % 14 == 0
        x = self.stem(x).mean((2,3))
        for block in self.blocks:
            x = block(x).relu()
        return {'x_norm_clstoken': self.norm(x)}

class FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.det_score_thresh = .5
    def forward(self,x):
        return [dict(boxes=x.new_empty(0,4), scores=x.new_empty(0), embeddings=x.new_empty(0,4)) for _ in x]

class DinoTest(unittest.TestCase):
    def test_frozen_detector_trainable_crop_path(self):
        reid = DinoCropReID(FakeDino(), embedding_dim=4, train_last_blocks=1)
        detector = FakeDetector()
        model = DinoPersonSearch(detector, reid).train()
        self.assertFalse(detector.training)
        self.assertFalse(detector.weight.requires_grad)
        self.assertFalse(reid.backbone.stem.weight.requires_grad)
        self.assertFalse(reid.backbone.blocks[0].weight.requires_grad)
        emb = model.embed_crop(torch.randn(4,3,128,64))
        self.assertTrue(torch.allclose(emb.norm(dim=1), torch.ones(4), atol=1e-5))
        emb[:,0].sum().backward()
        self.assertIsNotNone(reid.embedding_proj.weight.grad)
        self.assertIsNotNone(reid.backbone.blocks[1].weight.grad)
        self.assertIsNone(detector.weight.grad)
        model.det_score_thresh=.8
        self.assertEqual(detector.det_score_thresh,.8)
        self.assertEqual(len(model(torch.randn(2,3,64,64))),2)
    def test_supcon_epoch_uses_selected_loss_and_updates_head(self):
        from metric_loss import SupervisedNTXentLoss, crop_to_crop_reid_loss
        torch.manual_seed(17)
        reid = DinoCropReID(FakeDino(), embedding_dim=4, train_last_blocks=1)
        targets = [dict(ids=torch.tensor([1, 2]), image_id=torch.tensor(i),
                        query_crops=torch.randn(2, 3, 28, 14)) for i in range(2)]
        loader = [(torch.empty(0), targets)]
        reid.eval()
        with torch.no_grad():
            emb = reid(torch.cat([t['query_crops'] for t in targets]))
            expected, _ = crop_to_crop_reid_loss(emb, torch.tensor([1,2,1,2]),
                torch.tensor([0,0,1,1]), SupervisedNTXentLoss(.2))
        result = crop_epoch(reid, loader, torch.device('cpu'), metric_loss='supcon', temperature=.2)
        self.assertAlmostEqual(result['metric_loss'], float(expected), places=6)
        self.assertNotIn('active_fraction', result)
        before = reid.embedding_proj.weight.detach().clone()
        opt = torch.optim.AdamW([p for p in reid.parameters() if p.requires_grad], lr=.001)
        crop_epoch(reid, loader, torch.device('cpu'), opt, make_grad_scaler(torch.device('cpu')),
                   metric_loss='supcon', temperature=.2)
        self.assertFalse(torch.equal(before, reid.embedding_proj.weight))
        self.assertIsNone(reid.backbone.stem.weight.grad)
        self.assertIsNotNone(reid.backbone.blocks[-1].weight.grad)

    def test_training_validation_checkpoint_roundtrip(self):
        reid = DinoCropReID(FakeDino(), embedding_dim=4, train_last_blocks=0)
        targets=[dict(ids=torch.tensor([1,2]), image_id=torch.tensor(i),query_crops=torch.randn(2,3,128,64)) for i in range(2)]
        loader=[(torch.empty(2,3,64,64),targets)]
        opt=torch.optim.AdamW([p for p in reid.parameters() if p.requires_grad],lr=.001)
        result=crop_epoch(reid,loader,torch.device('cpu'),opt,make_grad_scaler(torch.device('cpu')))
        self.assertEqual(result['valid_anchors'],4)
        result=crop_epoch(reid,loader,torch.device('cpu'))
        self.assertEqual(result['positive_batch_fraction'],1)
        clone=DinoCropReID(FakeDino(),embedding_dim=4,train_last_blocks=0)
        clone.load_state_dict(reid.state_dict());clone.eval()
        x=targets[0]['query_crops']
        self.assertTrue(torch.allclose(reid(x),clone(x)))

if __name__=='__main__':unittest.main()
