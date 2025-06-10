from ..common.vector_from_pitch_yaw import get_vector_from_pitch_and_yaw
from .gaze_detector import GazeDetector
from .gaze import Gaze
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Normalize
from torchvision.models import resnet

import math
import numpy
import torch
import torch.nn as nn


class Gaze360GazeDetector(GazeDetector):

    def __init__(self):
        super().__init__(
            name="l2cs-gaze-360",
            download_url="https://drive.google.com/uc?id=18S956r4jnHtSeT8z8t3z8AoJZjVnNqPJ&export=download",
            model_file_name="L2CSNet_gaze360.pkl",
            normalized_camera_matrix=numpy.array(
                [[960.0, 0.0, 112.0], [0.0, 960.0, 112.0], [0.0, 0.0, 1]],
                dtype="double",
            ),
            normalized_distance=0.6,
            normalized_image_height=224,
            normalized_image_width=224,
            transform=Compose(
                [
                    ToPILImage(),
                    Resize(448),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device("cpu")
        self.idx_tensor = [idx for idx in range(90)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to("cpu")

    def load_model(self):
        self.model = L2CS(resnet.Bottleneck, [3, 4, 6, 3], 90)
        pkl = torch.load(self.model_file_path, map_location="cpu")
        self.model.load_state_dict(pkl)
        self.model.to(self.device)
        self.model.eval()

    def detect_gaze(self, face, normalized_frame):
        image = self.transform(normalized_frame)
        image = image.to(self.device)

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Predict yaw and pitch
        # Note: L2CS uses a different definition from yaw and pitch
        # Thats why I swapped the angles here
        gaze_yaw, gaze_pitch = self.model(image)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)

        # Get continuous predictions in degrees.
        pitch_predicted = (
            torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        )
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180

        pitch_predicted = pitch_predicted.cpu().detach().numpy() * numpy.pi / 180.0
        yaw_predicted = yaw_predicted.cpu().detach().numpy() * numpy.pi / 180.0

        vector = get_vector_from_pitch_and_yaw(pitch_predicted[0], yaw_predicted[0])

        # Apply the face's normalized rotation to the gaze vector
        # This is necessary to align the gaze vector with the face's orientation

        vector = vector @ face.normalized_rotation
        return Gaze(vector=vector, pitch=pitch_predicted[0], yaw=yaw_predicted[0])


# The following code is provided in the implementation of L2CS from Abdelrahman et al.
# See original: https://github.com/Ahmednull/L2CS-Net/blob/main/l2cs/model.py
# MIT License

# Copyright (c) 2022 Ahmed Abdelrahman

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


class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(L2CS, self).__init__()
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

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        # Renamed i to _ in the loop for the linter
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # gaze
        pre_yaw_gaze = self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze
