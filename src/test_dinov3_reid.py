import unittest
from unittest.mock import patch
import torch
from torch import nn
from dinov3_reid import DinoV3CropReID
from metric_loss import BatchHardTripletLoss, crop_to_crop_reid_loss

class TinyBackbone(nn.Module):
    embed_dim = 8
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,8,1)
        self.dropout = nn.Dropout(.5)
    def forward_features(self,x):
        assert x.shape[-2]%16 == x.shape[-1]%16 == 0
        return {'x_norm_clstoken': self.dropout(self.conv(x).mean((2,3)))}

class FrozenV3Tests(unittest.TestCase):
    def test_only_head_updates(self):
        torch.manual_seed(1)
        model=DinoV3CropReID(backbone=TinyBackbone(),embedding_dim=4).train()
        before={k:v.clone() for k,v in model.backbone.state_dict().items()}
        head=model.embedding_proj.weight.detach().clone()
        opt=torch.optim.AdamW(model.embedding_proj.parameters(),lr=.01)
        x=torch.randn(4,3,127,63)
        emb=model(x)
        loss,_=crop_to_crop_reid_loss(emb,torch.tensor([1,1,2,2]),torch.tensor([0,1,0,1]),BatchHardTripletLoss(1))
        loss.backward();opt.step()
        self.assertFalse(model.backbone.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.backbone.state_dict().items()))
        self.assertFalse(torch.equal(head,model.embedding_proj.weight))
        model.eval()
        clone=DinoV3CropReID(backbone=TinyBackbone(),embedding_dim=4).eval()
        clone.load_state_dict(model.state_dict())
        self.assertTrue(torch.allclose(model(x),clone(x)))
    def test_restore_does_not_request_pretrained_weights(self):
        with patch('torch.hub.load',return_value=TinyBackbone()) as load:
            DinoV3CropReID(pretrained=False)
            self.assertFalse(load.call_args.kwargs['pretrained'])
            self.assertNotIn('weights',load.call_args.kwargs)
        with self.assertRaises(ValueError):DinoV3CropReID(weights_path='/nonexistent')

if __name__=='__main__':unittest.main()
