"""Training, validation and W&B monitoring shared by the main notebook and checks."""
import os
import time
from pathlib import Path

import torch
import wandb
from torchvision.ops import box_iou

LOSS_KEYS = ('rpn_objectness_loss', 'rpn_bbox_loss', 'det_cls_loss', 'det_bbox_loss', 'metric_loss')


def amp_dtype(device):
    return torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16


def make_grad_scaler(device):
    return torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and amp_dtype(device) == torch.float16,
                                init_scale=128.)


def start_run(config, name='person-search', mode=None):
    # Never prompt for credentials in the middle of a notebook execution.
    mode = mode or os.environ.get('WANDB_MODE', 'offline')
    run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'person-search'),
                     entity=os.environ.get('WANDB_ENTITY') or None,
                     name=name, config=config, mode=mode,
                     dir=str(Path(__file__).resolve().parents[1]),
                     settings=wandb.Settings(init_timeout=60))
    run.define_metric('train/step')
    run.define_metric('train/*', step_metric='train/step')
    run.define_metric('val/*', step_metric='train/step')
    return run


def move_batch(images, targets, device):
    return images.to(device), [{k: v.to(device) for k, v in t.items()} for t in targets]


def train_one_epoch(model, loader, optimizer, device, max_steps=None, log_every=1,
                    run=None, scaler=None, start_step=0):
    model.train()
    amp = device.type == 'cuda'
    scaler = scaler or make_grad_scaler(device)
    history = {}
    for step, (images, targets) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images, targets = move_batch(images, targets, device)
        if amp:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp, dtype=amp_dtype(device)):
            losses = model(images, targets)
            total = sum(losses[k] for k in LOSS_KEYS)
        if not all(torch.isfinite(v).all() for v in losses.values()):
            raise RuntimeError(f'Non-finite loss/diagnostic at step {start_step + step}')
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if amp:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        values = {k: float(v.detach()) for k, v in losses.items()}
        values.update(total=float(total.detach()), grad_norm=float(grad_norm),
                      lr=optimizer.param_groups[0]['lr'], step_seconds=elapsed,
                      images_per_second=len(images) / elapsed, amp_scale=scaler.get_scale(),
                      optimizer_step_skipped=float(scaler.get_scale() < old_scale),
                      gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 2**20 if amp else 0,
                      gpu_peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 2**20 if amp else 0)
        for k, v in values.items():
            history.setdefault(k, []).append(v)
        if run:
            run.log({'train/step': start_step + step,
                     **{f'train/{k.lstrip("_")}': v for k, v in values.items()}})
        if step % log_every == 0:
            print(f'step {start_step + step}: ' + ' | '.join(f'{k}={values[k]:.4f}' for k in LOSS_KEYS)
                  + f' | valid_triplets={values["_metric_valid_anchors"]:.0f}'
                  + f' | peak_MB={values["gpu_peak_allocated_mb"]:.0f}', flush=True)
    return history


@torch.no_grad()
def validate(model, loader, device, run=None, step=0, max_batches=2):
    """Held-out losses and detection diagnostics, not benchmark mAP/CMC."""
    was_training = model.training
    totals, batches = {}, 0
    gt_total = recalled = detection_count = 0
    latencies, norms = [], []
    try:
        for images, targets in loader:
            if batches >= max_batches:
                break
            images, targets = move_batch(images, targets, device)
            # Loss path requires train mode. Frozen BN stays frozen; preserve RNG
            # so random target sampling does not alter subsequent training batches.
            devices = [device.index or 0] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                model.train()
                losses = model(images, targets)
            for k, value in losses.items():
                if not torch.isfinite(value):
                    raise RuntimeError(f'Non-finite validation {k}')
                totals[k.lstrip('_')] = totals.get(k.lstrip('_'), 0.0) + float(value)
            model.eval()
            if device.type == 'cuda': torch.cuda.synchronize(device)
            started = time.perf_counter()
            results = model(images)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - started) / len(images))
            for result, target in zip(results, targets):
                assert all(torch.isfinite(v).all() for v in result.values())
                boxes, emb = result['boxes'], result['embeddings']
                assert len(boxes) == len(emb) == len(result['scores'])
                detection_count += len(boxes)
                gt_total += len(target['boxes'])
                if len(boxes):
                    assert ((boxes[:, 2:] - boxes[:, :2]) > 0).all()
                    norms.extend(emb.norm(dim=1).tolist())
                    if len(target['boxes']):
                        recalled += int((box_iou(target['boxes'], boxes).max(1).values >= .5).sum())
            batches += 1
        metrics = {k: v / max(batches, 1) for k, v in totals.items()}
        metrics.update(total=sum(metrics.get(k, 0) for k in LOSS_KEYS),
                       detection_recall_iou50=recalled / max(gt_total, 1), detections=detection_count,
                       inference_seconds_per_image=sum(latencies) / max(len(latencies), 1),
                       embedding_norm=sum(norms) / max(len(norms), 1))
        if run: run.log({'train/step': step, **{f'val/{k}': v for k, v in metrics.items()}})
        return metrics
    finally:
        model.train(was_training)
