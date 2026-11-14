"""ETH-XGaze (Zhang et al., 2020), ResNet-18 weights from hysts/pytorch_mpiigaze_demo.

See https://arxiv.org/abs/2007.15837
"""

from __future__ import annotations

import cv2
import numpy as np
import timm
import torch

from ..core.paths import asset
from .base import AppearanceGazeEstimator, Normalization
from .device import select_device

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ETHXGazeEstimator(AppearanceGazeEstimator):
    name = "eth-xgaze"
    # https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/data/normalized_camera_params/eth-xgaze.yaml
    normalization = Normalization(focal_length=960.0, distance=0.6, size=224)

    def __init__(self):
        self.device = select_device()
        self.model = timm.create_model("resnet18", num_classes=2, pretrained=False)
        checkpoint = torch.load(asset("gaze-model-ethx.pth"), map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, normalized_bgr: np.ndarray) -> tuple[float, float]:
        rgb = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
        prediction = self.model(tensor.unsqueeze(0).to(self.device))[0].cpu().numpy()
        return float(prediction[0]), float(prediction[1])
