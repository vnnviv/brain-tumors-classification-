"""Metrics, test-time augmentation, and the biopsy pre-screener."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import albumentations as A

from .config import IMG_SIZE, MEAN, STD, TARGETS, TTA_PASSES


def metrics(y_true, y_pred):
    """Macro-averaged metrics. Specificity is computed per class from the
    confusion matrix, since sklearn has no direct multi-class version."""
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()

    specificities = []
    for i in range(cm.shape[0]):
        tn = total - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if tn + fp else 0.0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, average="macro",
                                          zero_division=0)),
        "specificity": float(np.mean(specificities)),
        "precision": float(precision_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def print_metrics(name, m):
    print(f"\n  {name}")
    for key, target in TARGETS.items():
        hit = "PASS" if m[key] >= target else "MISS"
        print(f"    {hit}  {key:<12} {m[key]:.1%}  (target {target:.0%})")
    print(f"          f1           {m['f1']:.1%}")


_tta = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(0.05, 0.05, p=0.3),
])
_norm = A.Normalize(mean=MEAN, std=STD)


@torch.no_grad()
def predict_tta(model, loader, device, passes=TTA_PASSES):
    """Average softmax over a few light augmentations of each test image.

    Batched rather than one image at a time, which is where the original
    version spent most of its inference budget.
    """
    model.eval()
    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)

    preds, truths = [], []
    for imgs, labels in loader:
        raw = imgs.permute(0, 2, 3, 1).cpu().numpy()
        raw = np.clip(raw * std + mean, 0, 1)
        raw = (raw * 255).astype(np.uint8)

        acc = None
        for _ in range(passes):
            batch = np.stack([_norm(image=_tta(image=img)["image"])["image"]
                              for img in raw])
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            p = F.softmax(model(tensor), dim=1)
            acc = p if acc is None else acc + p
        preds.append((acc / passes).argmax(1).cpu().numpy())
        truths.append(labels.numpy())

    return np.concatenate(preds), np.concatenate(truths)


class BiopsyPreScreener:
    """Turns a class prediction into a triage recommendation.

    This is a study aid, not a medical device. Nothing here is validated for
    clinical use and no part of it should inform a real referral.
    """

    URGENCY = {
        "glioma": ("URGENT", "Aggressive presentation. Biopsy within 24-48 hours."),
        "meningioma": ("HIGH", "Grade must be confirmed by biopsy."),
        "pituitary": ("MODERATE", "Functional imaging, biopsy may be indicated."),
        "notumor": ("LOW", "No tumor detected. Routine follow-up."),
    }

    def __init__(self, model, encoder, device):
        self.model = model.eval()
        self.encoder = encoder
        self.device = device
        self.classes = [c.lower() for c in encoder.classes_]

    @torch.no_grad()
    def screen(self, image_path):
        img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = _norm(image=np.array(img))["image"]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(tensor), dim=1)[0].cpu().numpy()

        idx = int(probs.argmax())
        label = self.classes[idx]
        if "notumor" in self.classes:
            tumor_prob = 1.0 - float(probs[self.classes.index("notumor")])
        else:
            tumor_prob = 1.0

        urgency, note = self.URGENCY.get(
            label, ("MODERATE", "Unrecognised class. Radiologist review needed.")
        )
        return {
            "type": label,
            "tumor_prob": tumor_prob,
            "confidence": float(probs[idx]),
            "urgency": urgency,
            "biopsy_needed": urgency in ("HIGH", "URGENT"),
            "note": note,
            "probs": dict(zip(self.classes, probs.tolist())),
        }
