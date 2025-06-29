from ..common.vector_from_pitch_yaw import get_vector_from_pitch_and_yaw
from .gaze import Gaze
from .gaze_detector import GazeDetector

import cv2
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, Lambda, ToTensor, Normalize


class MPIIFaceGazeDetector(GazeDetector):
    """
    A gaze detector based on the MPIIFaceGaze model.
    See: https://arxiv.org/abs/1611.08860
    """

    def __init__(self):
        super().__init__(
            name="mpiifacegaze",
            model_file_name="gaze-model-mpiifacegaze.pth",
            # see: https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/data/normalized_camera_params/mpiifacegaze.yaml
            normalized_distance=1,
            normalized_camera_matrix=numpy.array(
                [[1600.0, 0.0, 112.0], [0.0, 1600.0, 112.0], [0.0, 0.0, 1]],
                dtype="double",
            ),
            normalized_image_height=224,
            normalized_image_width=224,
            transform=Compose(
                [
                    Lambda(lambda x: cv2.resize(x, ([224, 224]))),
                    ToTensor(),
                    Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
                ]
            ),
        )

    def load_model(self):
        self.model = Model()
        device = torch.device("cpu")
        self.model.to(device)

        pth = torch.load(self.model_file_path, map_location="cpu")
        self.model.load_state_dict(pth["model"])
        self.model.eval()

    def detect_gaze(self, face, normalized_frame):
        image = self.transform(normalized_frame).unsqueeze(0)
        device = torch.device("cpu")
        image = image.to(device)

        prediction = self.model(image)
        prediction = prediction.cpu().numpy()

        vector = get_vector_from_pitch_and_yaw(prediction[0][0], prediction[0][1])

        # Apply the face's normalized rotation to the gaze vector
        # This is necessary to align the gaze vector with the face's orientation
        vector = vector @ face.normalized_rotation
        return Gaze(vector)


# Source: https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/models/mpiifacegaze/backbones/resnet_simple.py
# Was modified to use the default configuration of the pt-gaze library https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/data/configs/mpiifacegaze.yaml

# MIT License
# Copyright (c) 2017 hysts
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class FeatureExtractor(torchvision.models.ResNet):
    def __init__(self):

        # CHANGE FROM ORIGINAL CODE:
        # simplified original code to always use BasicBlock
        block = torchvision.models.resnet.BasicBlock

        # hardcoded the layers from the config file
        layers = [2, 2, 2, 1]
        super().__init__(block, layers)
        del self.layer4
        del self.avgpool
        del self.fc

        # fixed bug in original code to load a resnet-18 architecture
        state_dict = torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet18-f37072fd.pth"
        )
        self.load_state_dict(state_dict, strict=False)
        module = self.conv1
        module.weight.data = module.weight.data[:, [2, 1, 0]]

        with torch.no_grad():
            data = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            features = self.forward(data)
            self.n_features = features.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# Source: https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/models/mpiifacegaze/resnet_simple.py
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        n_channels = self.feature_extractor.n_features

        self.conv = nn.Conv2d(n_channels, 1, kernel_size=1, stride=1, padding=0)
        # This model assumes the input image size is 224x224.
        self.fc = nn.Linear(n_channels * 14**2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        y = F.relu(self.conv(x))
        x = x * y
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
