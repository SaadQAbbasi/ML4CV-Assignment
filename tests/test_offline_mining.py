import random
import unittest
import torch
from torch import nn
from offline_mining import identity_neighbors, mine_training_neighbors, MinedPKIdentityBatchSampler
from dinov3_reid import DinoV3CropReID
from metric_loss import crop_to_crop_reid_loss, BatchHardTripletLoss


class Dataset:
    def identity_to_indices(self):
        return {i: [i*3, i*3+1, i*3+2] for i in range(16)}


class TinyViT(nn.Module):
    embed_dim = 8
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3,8,1)
        self.blocks = nn.ModuleList([nn.Linear(8,8) for _ in range(3)])
        self.norm = nn.LayerNorm(8)
    def forward_features(self,x):
        out = self.stem(x).mean((2,3))
        for block in self.blocks:
            out = block(out).tanh()
        return {'x_norm_clstoken': self.norm(out)}


class MiningTests(unittest.TestCase):
    def test_neighbors_match_brute_force_and_exclude_self(self):
        torch.manual_seed(9)
        features = torch.randn(12,8)
        ids = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3])
        actual = identity_neighbors(features, ids, top_k=2, chunk_size=2)
        normalized = torch.nn.functional.normalize(features,dim=1)
        for i in range(4):
            scores = {j: float((normalized[ids == i] @ normalized[ids == j].T).max())
                      for j in range(4) if j != i}
            expected = sorted(scores,key=scores.get,reverse=True)[:2]
            self.assertEqual(actual[i],expected)
        self.assertEqual(actual,identity_neighbors(features,ids,top_k=2,chunk_size=20))

    def test_pk_mixing_and_fallback_preserve_coverage(self):
        sampler = MinedPKIdentityBatchSampler(Dataset(),p=8,k=3,seed=7)
        sampler.set_neighbors({i:[j for j in range(16) if i != j]+[999,i] for i in range(16)})
        state = random.getstate()
        groups=list(sampler.identity_batches())
        self.assertEqual(len(groups),len(sampler))
        self.assertEqual(len(set(sum(groups,[]))),16)
        self.assertEqual(sampler.last_stats['hard_identity_slots'],6)
        batches=list(sampler)
        for group, frames in zip(groups,batches):
            self.assertEqual(len(set(frames)),24)
            for identity in group:
                self.assertEqual(sum(f//3 == identity for f in frames),3)
        self.assertEqual(batches,list(sampler))
        self.assertEqual(state,random.getstate())
        sampler.set_neighbors({})
        self.assertEqual(len(list(sampler)),2)
        self.assertEqual(sampler.last_stats['hard_identity_slots'],0)

    def test_mining_embeds_all_provided_crops_and_restores_mode(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__(); self.count=0
            def forward(self,x):
                self.count += len(x)
                return x.mean((2,3))
        encoder=Encoder().train()
        targets = [dict(query_crops=torch.randn(3,3,16,16),ids=torch.tensor([0,1,2,-1]))]
        loader=[(None,targets),(None,targets)]
        neighbors, stats=mine_training_neighbors(encoder,loader,'cpu',[0,1],crop_batch_size=2)
        self.assertEqual(encoder.count,6)
        self.assertEqual(stats['frames'],2)
        self.assertEqual(stats['crops'],6)
        self.assertEqual(set(neighbors),{0,1})
        self.assertTrue(encoder.training)

    def test_v3_last_two_blocks_receive_gradients(self):
        torch.manual_seed(12)
        model=DinoV3CropReID(backbone=TinyViT(),embedding_dim=4,train_last_blocks=2).train()
        ids=torch.tensor([0,0,1,1]); frames=torch.tensor([0,1,0,1])
        emb=model(torch.randn(4,3,32,16))
        loss,_=crop_to_crop_reid_loss(emb,ids,frames,BatchHardTripletLoss(2))
        loss.backward()
        self.assertIsNone(model.backbone.stem.weight.grad)
        self.assertIsNone(model.backbone.blocks[0].weight.grad)
        for block in model.backbone.blocks[1:]:
            self.assertIsNotNone(block.weight.grad)
            self.assertTrue(torch.isfinite(block.weight.grad).all())
            self.assertGreater(block.weight.grad.abs().sum().item(),0)
        self.assertIsNotNone(model.backbone.norm.weight.grad)
        model.eval()
        self.assertFalse(any(b.training for b in model.backbone.blocks))
        with torch.no_grad():self.assertFalse(model(torch.randn(2,3,32,16)).requires_grad)


if __name__=='__main__':unittest.main()
