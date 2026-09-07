"""Short train/inference check on real PRW frames; not benchmark evaluation."""
from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torchvision.transforms.functional import pil_to_tensor, normalize, resize
import wandb

from model import PersonSearchModel
from training import start_run, train_one_epoch, validate

ROOT = Path(__file__).resolve().parents[1]


def read_frames(split, count=2):
    metadata = loadmat(ROOT / f'data/frame_{split}.mat')
    names = metadata[f'img_index_{split}'].reshape(-1)
    samples = []
    for raw_name in names[:count]:
        name = str(np.asarray(raw_name).reshape(-1)[0])
        image_path = ROOT / 'data/frames' / f'{name}.jpg'
        image = pil_to_tensor(Image.open(image_path).convert('RGB')).float() / 255
        _, h, w = image.shape
        # Preserve aspect ratio and pad to the common tensor shape.
        scale = min(256/h, 384/w)
        rh, rw = round(h*scale), round(w*scale)
        image = normalize(resize(image, [rh,rw]), [.485,.456,.406], [.229,.224,.225])
        padded = torch.zeros(3,256,384)
        padded[:, :rh, :rw] = image
        contents = loadmat(ROOT / 'data/annotations' / f'{name}.jpg.mat')
        arrays = [v for k,v in contents.items() if not k.startswith('__')
                  and isinstance(v,np.ndarray) and v.ndim == 2 and v.shape[1] == 5]
        if len(arrays) != 1: raise ValueError(f'Unexpected annotation schema: {name}')
        annotation = torch.as_tensor(arrays[0], dtype=torch.float32)
        ids = annotation[:,0].long()
        ids[ids < 0] = -1
        boxes = annotation[:,1:5].clone()
        boxes[:,2:] += boxes[:,:2]
        boxes[:,0::2] = boxes[:,0::2].clamp(0,w) * (rw/w)
        boxes[:,1::2] = boxes[:,1::2].clamp(0,h) * (rh/h)
        valid = (boxes[:,2:] > boxes[:,:2]).all(1)
        samples.append((padded, {'boxes':boxes[valid], 'ids':ids[valid]}, name))
    return samples


def main():
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    device = torch.device('cuda')
    torch.manual_seed(5)
    torch.set_num_threads(4)
    train = read_frames('train')
    test = read_frames('test')
    train_batch = (torch.stack([s[0] for s in train]), [s[1] for s in train])
    test_batch = (torch.stack([s[0] for s in test]), [s[1] for s in test])
    config = dict(pretrained=False, rpn_pre_nms_top_n_train=1000,
                  rpn_post_nms_top_n_train=512, rpn_post_nms_top_n_test=100)
    model = PersonSearchModel(**config).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=1e-3, momentum=.9, weight_decay=1e-4)
    run = start_run({**config, 'data':'PRW smoke check', 'train_frames':[s[2] for s in train],
                     'test_frames':[s[2] for s in test]}, name='prw-train-inference-check', mode='offline')
    try:
        history = train_one_epoch(model, [train_batch]*3, optimizer, device, run=run)
        metrics = validate(model, [test_batch], device, run=run, step=2)
        model.eval()
        with torch.inference_mode():
            results = model(test_batch[0].to(device))
            crop = Image.open(next((ROOT/'data/query_box').glob('*.jpg'))).convert('RGB')
            crop = normalize(resize(pil_to_tensor(crop).float()/255, [128,64]),
                             [.485,.456,.406], [.229,.224,.225])
            query = model.embed_crop(crop[None].to(device))
            assert torch.isfinite(query).all()
            # Also exercise nonempty real-image retrieval at a diagnostic-only
            # zero threshold; retain normal-threshold results for reported recall.
            threshold = model.det_score_thresh
            model.det_score_thresh = 0.0
            diagnostic_results = model(test_batch[0].to(device))
            model.det_score_thresh = threshold
            for result in diagnostic_results:
                assert len(result['boxes']) > 0
                similarities = query @ result['embeddings'].T
                assert torch.isfinite(similarities).all()
                assert torch.allclose(result['embeddings'].norm(dim=1),
                                      torch.ones(len(result['boxes']), device=device), atol=1e-4)
        mean = torch.tensor([.485,.456,.406])[:,None,None]
        std = torch.tensor([.229,.224,.225])[:,None,None]
        annotated = []
        for (image,target,name), result in zip(test,results):
            box_data = [{'position':dict(minX=float(b[0]),minY=float(b[1]),maxX=float(b[2]),maxY=float(b[3])),
                         'class_id':1,'scores':{'confidence':float(score)},'domain':'pixel'}
                        for b,score in zip(result['boxes'].cpu(),result['scores'].cpu())]
            gt_data = [{'position':dict(minX=float(b[0]),minY=float(b[1]),maxX=float(b[2]),maxY=float(b[3])),
                        'class_id':1,'domain':'pixel'} for b in target['boxes']]
            annotated.append(wandb.Image((image*std+mean).clamp(0,1), caption=name,
                             boxes={'predictions':{'box_data':box_data,'class_labels':{1:'person'}},
                                    'ground_truth':{'box_data':gt_data,'class_labels':{1:'person'}}}))
        run.log({'inference/real_frames':annotated, 'inference/query_norm':float(query.norm()),
                 'inference/default_threshold_detections':sum(len(r['boxes']) for r in results),
                 'inference/diagnostic_threshold':0.0,
                 'inference/diagnostic_detections':sum(len(r['boxes']) for r in diagnostic_results)})
        report = {'train':history, 'test':metrics, 'gpu':torch.cuda.get_device_name(0),
                  'query_embedding_shape':list(query.shape), 'wandb_dir':run.dir,
                  'diagnostic_threshold':0.0,
                  'diagnostic_detections':sum(len(r['boxes']) for r in diagnostic_results)}
        (ROOT/'artifacts').mkdir(exist_ok=True)
        (ROOT/'artifacts/prw_check.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally: run.finish()


if __name__ == '__main__': main()
