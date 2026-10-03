"""Runs the full comparison: 7 configurations x 3 seeds.

Usage:
    python run_experiments.py            # everything
    python run_experiments.py --smoke    # 2 configs, 1 seed, few epochs

Every arm splits by unique image first, so no image is ever in both train and
test. That was not true of the first version of this project and it mattered a
lot; see README and tests/test_splits.py.
"""

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.data import collect_paths, find_dataset, make_loaders
from src.evaluate import metrics, predict_tta, print_metrics
from src.models import CNNClassifier, MobileNetExtractor
from src.train import class_weights, train_model


# name, heavy augmentation, svm head, backbone, training repeats
CONFIGS = [
    ("M1 ResNet50 baseline",             False, False, "resnet50", 1),
    ("M2 EfficientNet-B3 baseline",      False, False, "efficientnet_b3", 1),
    ("M3 MobileNetV2 + SVM",             False, True,  None, 1),
    ("M4 ResNet50 + heavy aug",          True,  False, "resnet50", 2),
    ("M5 EfficientNet-B3 + heavy aug",   True,  False, "efficientnet_b3", 2),
    ("M6 MobileNetV2 + SVM, heavy aug",  True,  True,  None, 2),
    ("M7 EfficientNet-B3 + aug + TTA",   True,  False, "efficientnet_b3", 2),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def extract_features(extractor, loader, device):
    feats, labels = [], []
    for imgs, lbl in loader:
        feats.append(extractor(imgs.to(device, non_blocking=True)).cpu().numpy())
        labels.append(lbl.numpy())
    return np.vstack(feats), np.concatenate(labels)


def run_trial(paths, labels, n_classes, seed, heavy, svm, backbone, repeats,
              device):
    train_ld, val_ld, test_ld, train_labels = make_loaders(
        paths, labels, seed, heavy_aug=heavy, repeats=repeats
    )

    if svm:
        extractor = MobileNetExtractor().to(device).eval()
        x_train, y_train = extract_features(extractor, train_ld, device)
        x_test, y_test = extract_features(extractor, test_ld, device)
        scaler = StandardScaler().fit(x_train)
        clf = SVC(kernel="rbf", C=50, gamma="scale", class_weight="balanced")
        clf.fit(scaler.transform(x_train), y_train)
        y_pred = clf.predict(scaler.transform(x_test))
        del extractor
    else:
        model = CNNClassifier(n_classes, backbone=backbone).to(device)
        weights = class_weights(train_labels, n_classes, device)
        model, _ = train_model(model, train_ld, val_ld, device, weights)
        y_pred, y_test = predict_tta(model, test_ld, device)
        del model

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics(y_test, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="short run for checking the pipeline")
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    if args.smoke:
        config.EPOCHS, config.EPOCHS_HEAD = 2, 1
        config.MAX_PER_CLASS = 40
        seeds, configs = [42], CONFIGS[:2]
    else:
        seeds, configs = config.SEEDS, CONFIGS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    paths, labels, encoder = collect_paths(find_dataset(),
                                           config.MAX_PER_CLASS)
    n_classes = len(encoder.classes_)
    print(f"{len(paths)} images, {n_classes} classes: "
          f"{list(encoder.classes_)}\n")

    rows = []
    started = time.time()

    for name, heavy, svm, backbone, repeats in configs:
        print("=" * 64)
        print(f"  {name}")
        print("=" * 64)

        trials = []
        for i, seed in enumerate(seeds, 1):
            set_seed(seed)
            t0 = time.time()
            m = run_trial(paths, labels, n_classes, seed, heavy, svm,
                          backbone, repeats, device)
            trials.append(m)
            print(f"  trial {i}/{len(seeds)} seed {seed}: "
                  f"acc {m['accuracy']:.1%}  ({(time.time() - t0) / 60:.1f} min)")

        row = {"model": name}
        for key in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
            values = [t[key] for t in trials]
            row[key] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
        rows.append(row)
        print_metrics(f"{name} - mean of {len(seeds)} seeds", row)
        print()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("=" * 64)
    print(f"  done in {(time.time() - started) / 60:.1f} min -> {args.out}")
    print("=" * 64)
    for _, r in df.iterrows():
        print(f"  {r['model']:<34} {r['accuracy']:.1%} +/- {r['accuracy_std']:.1%}")

    with open("results.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
