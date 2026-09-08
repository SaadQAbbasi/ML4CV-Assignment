"""GPU regression checks plus main-notebook execution. Run from the repo root."""
import argparse
import json
from pathlib import Path
import random
import sys

import torch
from torchvision.models.detection._utils import BalancedPositiveNegativeSampler, BoxCoder
from torchvision.ops import box_iou

from model import PersonSearchModel
from rpn import compute_rpn_loss, RPNTargetAssigner
from metric_loss import BatchHardTripletLoss
from training import start_run, train_one_epoch, validate

ROOT = Path(__file__).resolve().parents[1]


def check_regressions(device):
    # Per-image sampling must use both images, not mistakenly index a 2-D label tensor.
    obj = [torch.zeros(2, 1, 1, 4, device=device, requires_grad=True)]
    deltas = [torch.zeros(2, 4, 1, 4, device=device, requires_grad=True)]
    anchors = [torch.tensor([[0., 0., 10., 20.]] * 4, device=device) for _ in range(2)]
    labels = [torch.tensor([1., 0., 0., 0.], device=device),
              torch.tensor([0., 1., 0., 0.], device=device)]
    coder = BoxCoder((1., 1., 1., 1.))
    losses = compute_rpn_loss(obj, deltas, anchors, labels, anchors, coder,
                              BalancedPositiveNegativeSampler(4, .5))
    sum(losses).backward()
    assert (obj[0].grad.abs().flatten(1).sum(1) > 0).all()
    assigner = RPNTargetAssigner()
    weak = torch.tensor([[0., 0., 1., 1.]], device=device)
    match, _ = assigner.assign_targets_single_image(weak, anchors[0][:1])
    assert match.item() == 0, 'Low-quality anchors must not become forced positives'
    emb = torch.randn(4, 8, device=device, requires_grad=True)
    loss, stats = BatchHardTripletLoss(margin=10)(emb, torch.tensor([0, 0, 1, 1], device=device))
    loss.backward()
    assert stats['valid_anchors'] == 4 and emb.grad.abs().sum() > 0
    print('RPN batch sampling, strict matching and triplet backward: PASS', flush=True)


def check_model(device):
    torch.manual_seed(7)
    model = PersonSearchModel(pretrained=False, rpn_pre_nms_top_n_train=500,
                              rpn_post_nms_top_n_train=128,
                              rpn_post_nms_top_n_test=32).to(device)
    model.train()
    bn = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert all(not m.training and not m.weight.requires_grad for m in bn)
    x = torch.randn(2, 3, 128, 96, device=device)
    # Construct GT from the detector's own predictions for a controlled regression
    # fixture: this guarantees positive proposals without injecting boxes into RPN.
    with torch.no_grad():
        from rpn import decode_proposals, make_image_list
        _, feats, _ = model._extract_pyramid(x)
        anchors = model.anchor_gen(make_image_list(x), feats)
        obj, delta = model.rpn_head(feats)
        props = decode_proposals(obj, delta, anchors, [(128,96)]*2, model.box_coder,
                                 pre_nms_top_n=500, post_nms_top_n=128)
    targets = [{'boxes': p[:8].clone(), 'ids': torch.arange(8, device=device) % 2} for p in props]
    for i, target in enumerate(targets):
        target['query_crops'] = torch.randn(8, 3, 128, 64, device=device, requires_grad=True)
        target['image_id'] = torch.tensor(i, device=device)
    received = []
    hook = model.reid_head.register_forward_pre_hook(lambda module, inputs: received.append(len(inputs[0])))
    losses = model(x, targets)
    hook.remove()
    assert received == [16, 16]
    assert losses['_metric_positive_pairs'] == 64
    assert losses['_metric_valid_anchors'] > 0
    sum(v for k,v in losses.items() if not k.startswith('_')).backward()
    for name in ('backbone', 'fpn', 'rpn_head', 'det_head', 'reid_head'):
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads), name
    assert all(t['query_crops'].grad.abs().sum() > 0 for t in targets)
    model.zero_grad(set_to_none=True)
    # Empty GT and completely unlabeled persons should still train detection.
    for targets in ([{'boxes': x.new_empty(0,4), 'ids': torch.empty(0,dtype=torch.long,device=device)}]*2,
                    [{'boxes': p[:1], 'ids': torch.tensor([-1],device=device)} for p in props]):
        for i, target in enumerate(targets):
            target['query_crops'] = x.new_empty((0, 3, 128, 64))
            target['image_id'] = torch.tensor(i, device=device)
        losses = model(x, targets)
        assert all(torch.isfinite(v) for v in losses.values())
        assert losses['metric_loss'] == 0
        sum(v for k,v in losses.items() if not k.startswith('_')).backward()
        model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        # Check both populated detections and a threshold that yields no detections.
        for threshold in (0., 1.1):
            model.det_score_thresh = threshold
            pooled_boxes = []
            hook = model.roi_pooler.register_forward_pre_hook(
                lambda module, inputs: pooled_boxes.append([b.clone() for b in inputs[1]]))
            results = model(x)
            hook.remove()
            assert len(pooled_boxes) == 2
            assert all(torch.equal(b, r['boxes']) for b, r in zip(pooled_boxes[-1], results))
            for result in results:
                assert all(torch.isfinite(v).all() for v in result.values())
                if threshold == 0:
                    assert len(result['boxes']) > 0
                    assert torch.allclose(result['embeddings'].norm(dim=1),
                                          torch.ones(len(result['boxes']), device=device), atol=1e-4)
                else: assert result['boxes'].shape == (0,4)
        query = model.embed_crop(x[:1])
        assert query.shape == (1,256) and torch.isfinite(query).all()
    print('All module gradients, positive-only ReID, frozen BN, empty/unlabeled GT, scene/query inference: PASS', flush=True)
    del model
    torch.cuda.empty_cache()


def execute_notebook():
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(ROOT / 'src/train_person_search.ipynb', as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name='python3',
                            on_cell_start=lambda cell, cell_index: print(f'Notebook cell {cell_index}', flush=True),
                            resources={'metadata': {'path': str(ROOT)}})
    client.execute()
    output = ROOT / 'artifacts/train_person_search.executed.ipynb'
    output.parent.mkdir(exist_ok=True)
    nbformat.write(nb, output)
    for cell in nb.cells:
        for item in cell.get('outputs', []):
            if item.output_type == 'stream': print(item.text, end='')
    print(f'Executed notebook saved: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('GPU required for this verification')
    device = torch.device('cuda')
    torch.set_num_threads(4)
    print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
    check_regressions(device)
    check_model(device)
    if args.notebook: execute_notebook()
