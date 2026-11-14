"""L2CS-Net (Abdelrahman et al., 2022) trained on Gaze360.

See https://github.com/Ahmednull/L2CS-Net
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import Bottleneck

from ..core.paths import asset
from .base import AppearanceGazeEstimator, Normalization
from .device import select_device

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BINS = 90
INPUT_SIZE = 448


class L2CSGaze360Estimator(AppearanceGazeEstimator):
    name = "l2cs-gaze360"
    # Same normalized camera as ETH-XGaze; the crop is upscaled to the 448px L2CS was trained on.
    normalization = Normalization(focal_length=960.0, distance=0.6, size=224)

    def __init__(self):
        self.device = select_device()
        self.model = L2CS(Bottleneck, [3, 4, 6, 3], BINS)
        self.model.load_state_dict(torch.load(asset("gaze-model-gaze360.pkl"), map_location="cpu"))
        self.model.to(self.device).eval()
        self.bin_indices = torch.arange(BINS, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict(self, normalized_bgr: np.ndarray) -> tuple[float, float]:
        rgb = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        image = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
        yaw_logits, pitch_logits = self.model(tensor.unsqueeze(0).to(self.device))

        # Expected value over the bins, each bin is 4 degrees wide and the range starts at -180.
        def to_radians(logits: torch.Tensor) -> float:
            degrees = (torch.softmax(logits, dim=1) * self.bin_indices).sum(dim=1) * 4 - 180
            return float(degrees[0].cpu()) * math.pi / 180.0

        return to_radians(pitch_logits), to_radians(yaw_logits)


# Taken from https://github.com/Ahmednull/L2CS-Net/blob/main/l2cs/model.py
# MIT License, Copyright (c) 2022 Ahmed Abdelrahman
class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)
        # Vestigial layer from previous experiments, kept so the checkpoint loads.
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)
