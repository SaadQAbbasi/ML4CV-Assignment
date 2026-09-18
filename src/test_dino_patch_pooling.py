import unittest
import torch
from torch import nn
from torch.nn import functional as F
from dinov3_reid import DinoV3CropReID

class PatchBackbone(nn.Module):
    embed_dim=4
    def __init__(self):
        super().__init__()
        self.stem=nn.Linear(3,4)
        self.blocks=nn.ModuleList([nn.Linear(4,4),nn.Linear(4,4)])
        self.norm=nn.LayerNorm(4)
    def forward_features(self,x):
        x=self.stem(x.mean((2,3)))
        for block in self.blocks:x=block(x)
        x=self.norm(x)
        return dict(x_norm_clstoken=x, x_norm_patchtokens=torch.stack([x*2,x*.5],dim=1))

class PatchPoolingTests(unittest.TestCase):
    def test_pooling_gradients_restore_and_legacy(self):
        torch.manual_seed(4)
        model=DinoV3CropReID(backbone=PatchBackbone(),embedding_dim=3,train_last_blocks=1,
                            feature_pooling='cls_mean_patch')
        x=torch.randn(2,3,32,16)
        captured=[]
        handle=model.embedding_proj.register_forward_pre_hook(lambda m,args: captured.append(args[0]))
        actual=model(x)
        handle.remove()
        outputs=model.backbone.forward_features(x)
        expected=torch.cat([outputs['x_norm_clstoken'],outputs['x_norm_patchtokens'].mean(1)],1)
        self.assertTrue(torch.allclose(captured[0],expected))
        self.assertEqual(model.embedding_proj.in_features,8)
        self.assertTrue(torch.allclose(actual.norm(dim=1),torch.ones(2)))
        actual[:,0].sum().backward()
        self.assertIsNotNone(model.backbone.blocks[-1].weight.grad)
        self.assertIsNone(model.backbone.blocks[0].weight.grad)
        self.assertIsNotNone(model.embedding_proj.weight.grad)
        clone=DinoV3CropReID(backbone=PatchBackbone(),embedding_dim=3,train_last_blocks=1,
                            feature_pooling='cls_mean_patch')
        clone.load_state_dict(model.state_dict(),strict=True)
        model.eval();clone.eval()
        self.assertTrue(torch.equal(model(x),clone(x)))
        legacy=DinoV3CropReID(backbone=PatchBackbone(),embedding_dim=3)
        self.assertEqual(legacy.embedding_proj.in_features,4)
        with self.assertRaises(ValueError):
            DinoV3CropReID(backbone=PatchBackbone(),feature_pooling='typo')

class SpatialBackbone(PatchBackbone):
    def forward_features(self, image):
        cls=super().forward_features(image)['x_norm_clstoken']
        rows,cols=image.shape[-2]//16,image.shape[-1]//16
        locations=torch.arange(rows*cols,device=image.device,dtype=cls.dtype).reshape(1,-1,1)
        return dict(x_norm_clstoken=cls,x_norm_patchtokens=cls[:,None,:]+locations)

class RegionPoolingTests(unittest.TestCase):
    def test_horizontal_regions_gradients_and_roundtrip(self):
        model=DinoV3CropReID(backbone=SpatialBackbone(),embedding_dim=3,
                            train_last_blocks=1,feature_pooling='cls_upper_lower')
        # 4x2 patch grid: upper mean index 1.5, lower mean index 5.5.
        crops=torch.randn(2,3,64,32)
        captured=[]
        handle=model.embedding_proj.register_forward_pre_hook(lambda m,args:captured.append(args[0]))
        embeddings=model(crops)
        handle.remove()
        cls=model.backbone.forward_features(crops)['x_norm_clstoken']
        self.assertTrue(torch.allclose(captured[0],torch.cat([cls,cls+1.5,cls+5.5],1)))
        self.assertEqual(model.embedding_proj.in_features,12)
        self.assertTrue(torch.allclose(embeddings.norm(dim=1),torch.ones(2),atol=1e-6))
        embeddings[:,0].sum().backward()
        self.assertIsNotNone(model.backbone.blocks[-1].weight.grad)
        self.assertIsNone(model.backbone.blocks[0].weight.grad)
        clone=DinoV3CropReID(backbone=SpatialBackbone(),embedding_dim=3,
                            train_last_blocks=1,feature_pooling='cls_upper_lower')
        clone.load_state_dict(model.state_dict(),strict=True)
        model.eval();clone.eval()
        self.assertTrue(torch.equal(model(crops),clone(crops)))
        with self.assertRaises(ValueError):model(torch.randn(1,3,16,32))
        bad=DinoV3CropReID(backbone=PatchBackbone(),feature_pooling='cls_upper_lower')
        with self.assertRaises(ValueError):bad(crops)
