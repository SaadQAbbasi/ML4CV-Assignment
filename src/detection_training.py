"""Detection-only optimization and explicit 101-point AP at IoU .50/.75."""
import torch
from torchvision.ops import box_iou
from training import amp_dtype


def detection_epoch(model, loader, optimizer, scaler, device):
    model.train()
    totals={}
    for step,(images,targets) in enumerate(loader):
        images=images.to(device)
        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,enabled=device.type=='cuda',dtype=amp_dtype(device)):
            losses=model(images,targets)
            total=sum(losses.values())
        if not torch.isfinite(total):raise RuntimeError('Nonfinite detection loss')
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update()
        for k,v in dict(total=total,**losses).items():totals[k]=totals.get(k,0.)+float(v.detach())
        if step%50==0:print(f'Detection batch {step}/{len(loader)} loss={float(total.detach()):.4f}',flush=True)
    return {k:v/len(loader) for k,v in totals.items()}


def detection_metrics(predictions, targets, operating_threshold=.5):
    """All annotated people, including unknown IDs; no ReID metrics or COCO claim."""
    total_gt=sum(len(t) for t in targets)
    rows=sorted([(float(s),i,j) for i,p in enumerate(predictions)
                 for j,s in enumerate(p['scores'])],reverse=True)
    metrics={}
    overlaps=[box_iou(p['boxes'],t) for p,t in zip(predictions,targets)]
    for threshold in (.5,.75):
        used=[set() for _ in targets]; flags=[]
        for score,i,j in rows:
            match=False
            for k in overlaps[i][j].argsort(descending=True).tolist():
                if overlaps[i][j,k]<threshold:break
                if k not in used[i]:used[i].add(k);match=True;break
            flags.append(float(match))
        tp=torch.tensor(flags).cumsum(0)
        recall=tp/max(total_gt,1)
        precision=tp/torch.arange(1,len(rows)+1)
        ap=sum(float(precision[recall>=r].max()) if (recall>=r).any() else 0.
               for r in torch.linspace(0,1,101))/101
        kept=sum(s>=operating_threshold for s,_,_ in rows)
        hits=sum(flags[:kept]); suffix=int(threshold*100)
        metrics.update({f'AP{suffix}':ap, f'precision_iou{suffix}':hits/max(kept,1),
                        f'recall_iou{suffix}':hits/max(total_gt,1)})
    metrics['mean_AP50_AP75']=(metrics['AP50']+metrics['AP75'])/2
    metrics['gt_people']=total_gt
    metrics['candidates']=len(rows)
    return metrics


@torch.inference_mode()
def evaluate_detector(model,loader,device):
    mode,threshold=model.training,model.det_score_thresh
    model.eval();model.det_score_thresh=.001
    predictions=[];targets=[]
    try:
        for images,batch_targets in loader:
            predictions.extend([{k:v.cpu() for k,v in p.items()} for p in model(images.to(device))])
            targets.extend([t['boxes'].cpu() for t in batch_targets])
        return detection_metrics(predictions,targets)
    finally:
        model.det_score_thresh=threshold;model.train(mode)
