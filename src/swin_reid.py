"""Frozen Swin detection with separately trained scene or crop ReID paths."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import swin_t, Swin_T_Weights
from roi_heads import ReIDHead
from dino_reid import DinoPersonSearch
from metric_loss import SupervisedNTXentLoss, cross_image_reid_loss, crop_to_crop_reid_loss
from training import amp_dtype


class SwinCropReID(nn.Module):
    """Independent ImageNet Swin-T, last two stages trainable by default."""
    def __init__(self, pretrained=True, embedding_dim=256, crop_size=(256,128)):
        super().__init__()
        net=swin_t(weights=Swin_T_Weights.DEFAULT if pretrained else None)
        self.features=net.features
        self.norm=net.norm
        self.crop_size=tuple(crop_size)
        for index,module in enumerate(self.features):
            module.requires_grad_(index>=4)
        self.embedding_proj=nn.Linear(768*3,embedding_dim)
        self.train()

    def train(self,mode=True):
        super().train(mode)
        for module in list(self.features)[:4]:module.eval()
        return self

    def forward(self,crops):
        grid=self.norm(self.features(crops))  # NHWC
        split=grid.shape[1]//2
        features=torch.cat([grid.mean((1,2)),grid[:,:split].mean((1,2)),
                            grid[:,split:].mean((1,2))],1)
        return F.normalize(self.embedding_proj(features),dim=1)


class SwinSceneReID(DinoPersonSearch):
    """Only a fresh shared RoI head learns; detector backbone/FPN remain frozen."""
    def __init__(self,detector,embedding_dim=256,crop_size=(256,128),feature_batch_size=2):
        head=ReIDHead(in_channels=detector.det_head.fc6.in_features//49,embedding_dim=embedding_dim)
        head.crop_size=tuple(crop_size)
        super().__init__(detector,head)
        self.feature_batch_size=feature_batch_size

    def pool(self,images,boxes,pyramid=None):
        with torch.no_grad():
            if pyramid is None:pyramid=self.detector._extract_pyramid(images)[2]
            with torch.autocast(images.device.type,enabled=False):
                pooled=self.detector.roi_pooler({k:v.float() for k,v in pyramid.items()},boxes,
                                               [tuple(images.shape[-2:])]*len(images))
        return pooled

    def embed_scene(self,images,boxes):
        outputs=[]
        for start in range(0,len(images),self.feature_batch_size):
            outputs.append(self.reid_head(self.pool(images[start:start+self.feature_batch_size],
                                             boxes[start:start+self.feature_batch_size])))
        return torch.cat(outputs)

    def embed_crop(self,crops):
        h,w=crops.shape[-2:]
        boxes=[crops.new_tensor([[0,0,w,h]]) for _ in crops]
        return self.embed_scene(crops,boxes)

    def forward(self,images):
        # Exactly one full-scene backbone/FPN extraction, shared by detection and ReID.
        with torch.no_grad():
            pyramid=self.detector._extract_pyramid(images)
            results=self.detector._forward_eval(images,pyramid=pyramid)
        embeddings=self.reid_head(self.pool(images,[r['boxes'] for r in results],pyramid[2]))
        for result,emb in zip(results,embeddings.split([len(r['boxes']) for r in results])):
            result['embeddings']=emb
        return results


def swin_reid_epoch(model,loader,device,strategy,optimizer=None,scaler=None,temperature=.1):
    training=optimizer is not None
    model.train(training)
    criterion=SupervisedNTXentLoss(temperature)
    total=count=0
    for step,(images,targets) in enumerate(loader):
        ids=torch.cat([t['ids'][t['ids']>=0] for t in targets]).to(device)
        if not len(ids):continue
        frames=torch.cat([t['image_id'].reshape(1).expand(int((t['ids']>=0).sum())) for t in targets]).to(device)
        crops=torch.cat([t['query_crops'] for t in targets]).to(device)
        if training:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device.type,enabled=device.type=='cuda',dtype=amp_dtype(device)):
                crop_embeddings=model.embed_crop(crops)
                if strategy=='crop_to_scene':
                    boxes=[t['boxes'][t['ids']>=0].to(device) for t in targets]
                    scene=model.embed_scene(images.to(device),boxes)
                    loss,stats=cross_image_reid_loss(scene,crop_embeddings,ids,frames,criterion)
                elif strategy=='crop_to_crop':
                    loss,stats=crop_to_crop_reid_loss(crop_embeddings,ids,frames,criterion)
                else:raise ValueError('Unknown ReID strategy')
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite ReID loss')
            if training:
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10.,error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update()
        total+=float(loss.detach());count+=1
        if step%20==0:print(f'ReID batch {step}/{len(loader)} loss={float(loss.detach()):.4f}',flush=True)
    if not count:raise ValueError('No labeled batches')
    return {'metric_loss':total/count}
