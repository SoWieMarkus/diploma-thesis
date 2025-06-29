from ..common.vector_from_pitch_yaw import get_vector_from_pitch_and_yaw
from .gaze_detector import GazeDetector
import numpy
import torch
import cv2
import timm
from .gaze import Gaze
from torchvision.transforms import Compose, Lambda, ToTensor, Normalize


class ETHXGazeDetector(GazeDetector):
    """
    ETH-XGaze detector based on the ResNet18 model.
    This detector is trained on the ETH-XGaze dataset and is designed to estimate gaze direction
    from images captured in a controlled environment.
    See: https://arxiv.org/abs/2007.15837
    """

    def __init__(self):
        super().__init__(
            name="eth-xgaze",
            model_file_name="gaze-model-ethx.pth",
            # Source: https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/data/normalized_camera_params/eth-xgaze.yaml
            normalized_distance=0.6,
            normalized_camera_matrix=numpy.array(
                [[960.0, 0.0, 112.0], [0.0, 960.0, 112.0], [0.0, 0.0, 1]],
                dtype="double",
            ),
            normalized_image_height=224,
            normalized_image_width=224,
            transform=Compose(
                [
                    Lambda(lambda x: cv2.resize(x, ([224, 224]))),
                    Lambda(lambda x: x[:, :, ::-1].copy()),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )

    def load_model(self):
        self.model = timm.create_model("resnet18", num_classes=2)
        device = torch.device("cpu")
        self.model.to(device)

        pth = torch.load(self.model_file_path, map_location="cpu")
        self.model.load_state_dict(pth["model"])
        self.model.eval()

    @torch.no_grad()
    def detect_gaze(self, face, normalized_frame):
        image = self.transform(normalized_frame).unsqueeze(0)
        device = torch.device("cpu")
        image = image.to(device)

        prediction = self.model(image)
        prediction = prediction.cpu().numpy()

        vector = get_vector_from_pitch_and_yaw(prediction[0, 0], prediction[0, 1])

        # Apply the face's normalized rotation to the gaze vector
        # This is necessary to align the gaze vector with the face's orientation
        vector = vector @ face.normalized_rotation_matrix

        return Gaze(
            vector=vector,
            pitch=prediction[0][0],
            yaw=prediction[0][1],
        )
