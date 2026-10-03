"""Model definitions.

Three backbones: ResNet50 and EfficientNet-B3 for the end-to-end CNNs, and
MobileNetV2 as a frozen feature extractor feeding an SVM.
"""

import torch.nn as nn
from torchvision import models


class CNNClassifier(nn.Module):
    """ImageNet backbone with a new classification head."""

    def __init__(self, n_classes, backbone="resnet50", dropout=0.4):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "efficientnet_b3":
            base = models.efficientnet_b3(weights="IMAGENET1K_V1")
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, n_classes),
            )
        elif backbone == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V2")
            base.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(base.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, n_classes),
            )
        else:
            raise ValueError(f"unknown backbone: {backbone}")

        self.net = base

    def forward(self, x):
        return self.net(x)


class MobileNetExtractor(nn.Module):
    """Frozen MobileNetV2 trunk, pooled to a 1280-d vector for the SVM."""

    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)
