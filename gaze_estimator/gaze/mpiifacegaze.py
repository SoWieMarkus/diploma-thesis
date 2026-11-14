"""MPIIFaceGaze (Zhang et al., 2017), ResNet-simple weights from hysts/pytorch_mpiigaze_demo.

See https://arxiv.org/abs/1611.08860
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from ..core.paths import asset
from .base import AppearanceGazeEstimator, Normalization
from .device import select_device

# The model was trained on BGR images, so mean and std are in BGR order.
BGR_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
BGR_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)


class MPIIFaceGazeEstimator(AppearanceGazeEstimator):
    name = "mpiifacegaze"
    # https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/data/normalized_camera_params/mpiifacegaze.yaml
    normalization = Normalization(focal_length=1600.0, distance=1.0, size=224)

    def __init__(self):
        self.device = select_device()
        self.model = ResNetSimple()
        checkpoint = torch.load(asset("gaze-model-mpiifacegaze.pth"), map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, normalized_bgr: np.ndarray) -> tuple[float, float]:
        image = normalized_bgr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(((image - BGR_MEAN) / BGR_STD).transpose(2, 0, 1))
        prediction = self.model(tensor.unsqueeze(0).to(self.device))[0].cpu().numpy()
        return float(prediction[0]), float(prediction[1])


# Adapted from https://github.com/hysts/pytorch_mpiigaze_demo (ptgaze/models/mpiifacegaze/resnet_simple.py)
# MIT License, Copyright (c) 2017 hysts.
#
# Changes: the backbone is always a ResNet-18 style network with layers [2, 2, 2] (default config of
# ptgaze), and it is no longer initialized with ImageNet weights downloaded from the internet. Those
# weights were overwritten by the checkpoint anyway.
class FeatureExtractor(torchvision.models.ResNet):
    def __init__(self):
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 1])
        del self.layer4
        del self.avgpool
        del self.fc
        self.n_features = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer3(self.layer2(self.layer1(x)))


class ResNetSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        n_channels = self.feature_extractor.n_features
        self.conv = nn.Conv2d(n_channels, 1, kernel_size=1, stride=1, padding=0)
        # The model assumes an input size of 224x224 (feature map 14x14).
        self.fc = nn.Linear(n_channels * 14**2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        y = F.relu(self.conv(x))
        x = (x * y).view(x.size(0), -1)
        return self.fc(x)
