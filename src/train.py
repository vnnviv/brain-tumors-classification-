"""Training loop.

Two phases. First the backbone is frozen and only the new head trains, which
converges fast and stops the random head from wrecking pretrained weights.
Then everything unfreezes and the backbone trains at 5% of the head's rate.
"""

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import (BACKBONE_LR_MULT, EARLY_STOP_PATIENCE, EPOCHS,
                     EPOCHS_HEAD, FOCAL_GAMMA, HEAD_LR, LABEL_SMOOTHING,
                     WEIGHT_DECAY)

HEAD_KEYS = ("fc", "classifier", "head")


class FocalLoss(nn.Module):
    """Cross entropy that down-weights easy examples.

    The classes here are close to balanced, but glioma and meningioma are the
    confusable pair and focal loss keeps the model working on them instead of
    coasting on notumor.
    """

    def __init__(self, gamma=FOCAL_GAMMA, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.weight,
            reduction="none", label_smoothing=LABEL_SMOOTHING,
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def class_weights(labels, n_classes, device):
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32, device=device)


def _is_head(name):
    return any(k in name for k in HEAD_KEYS)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def train_model(model, train_loader, val_loader, device, weights=None,
                verbose=True):
    criterion = FocalLoss(weight=weights)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc, best_state = 0.0, None

    def run_epoch(optimizer, scheduler=None):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

    def snapshot(acc):
        nonlocal best_acc, best_state
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            return True
        return False

    # Phase 1 - head only
    for name, p in model.named_parameters():
        p.requires_grad = _is_head(name)
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(head_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=HEAD_LR, steps_per_epoch=len(train_loader),
        epochs=EPOCHS_HEAD, pct_start=0.3,
    )
    for epoch in range(EPOCHS_HEAD):
        run_epoch(opt, sched)
        snapshot(accuracy(model, val_loader, device))
        if verbose and (epoch + 1) % 4 == 0:
            print(f"    head {epoch + 1}/{EPOCHS_HEAD}  best val {best_acc:.4f}")

    # Phase 2 - full fine-tune
    for p in model.parameters():
        p.requires_grad = True
    backbone, head = [], []
    for name, p in model.named_parameters():
        (head if _is_head(name) else backbone).append(p)

    remaining = EPOCHS - EPOCHS_HEAD
    opt = optim.AdamW(
        [{"params": backbone, "lr": HEAD_LR * BACKBONE_LR_MULT},
         {"params": head, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining)

    stale = 0
    for epoch in range(remaining):
        run_epoch(opt)
        sched.step()
        stale = 0 if snapshot(accuracy(model, val_loader, device)) else stale + 1
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    full {EPOCHS_HEAD + epoch + 1}/{EPOCHS}  "
                  f"best val {best_acc:.4f}")
        if stale >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"    early stop at {EPOCHS_HEAD + epoch + 1}")
            break
        gc.collect()
        if use_amp:
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc
