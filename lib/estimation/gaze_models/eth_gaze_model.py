import cv2
import numpy
import timm
import torch
import torchvision

from .gaze_model import GazeModel



# Based on: https://github.com/hysts/pytorch_mpiigaze_demo/tree/master/ptgaze

class ETHGazeModel(GazeModel):
    def __init__(self):
        super().__init__(
            name="eth-gaze",
            download_url="https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.2.2/eth-xgaze_resnet18.pth",
            file_name="eth-gaze-model.pth",
        )
        self.normalized_camera_matrix = numpy.array(
            [[960.0, 0.0, 112.0], [0.0, 960.0, 112.0], [0.0, 0.0, 1]], dtype="double"
        )

        #source: https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/data/normalized_camera_params/eth-xgaze.yaml
        self.normalized_distance = 0.6
        self.normalized_image_height = 224
        self.normalized_image_width = 224
        self.transform = self.get_transform()

    def get_transform(self):
        size = tuple([224, 224])

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda x: cv2.resize(x, size)),
                torchvision.transforms.Lambda(
                    lambda x: x[:, :, ::-1].copy()
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform

    def load_model(self):
        self.model = timm.create_model("resnet18", num_classes=2)
        device = torch.device("cpu")
        self.model.to(device)

        pth = torch.load(self.get_model_path(), map_location="cpu")
        self.model.load_state_dict(pth["model"])
        self.model.eval()

    @torch.no_grad()
    def detect_gaze(self, image, face, video, face_model_name, frame_number):
        # normalize the image
        normalized_image = self.normalize_input_image(image, face, video)

        # save the normalized image for evaluation
        output_file = self.get_output_directory(video, face_model_name) / f"{frame_number}.jpg"
        cv2.imwrite(str(output_file), normalized_image)

        # predict pitch and yaw with eth xgaze
        image = self.transform(normalized_image).unsqueeze(0)

        device = torch.device("cpu")
        image = image.to(device)
        prediction = self.model(image)
        prediction = prediction.cpu().numpy()

        # convert pitch and yaw to vector
        vector = GazeModel.get_vector_from_pitch_and_yaw(prediction[0][0], prediction[0][1])

        # convert normalized gaze vector to coordinates before normalization
        vector = vector @ face.normalized_rotation

        return prediction[0][0], prediction[0][1], vector
