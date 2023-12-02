import pathlib
from abc import abstractmethod

import cv2
import numpy

from ..model import Model
from ...io.frame_face_data import FrameFaceData

SCRIPT_PATH = pathlib.Path(__file__).resolve().parent
OUTPUT_DIRECTORY = SCRIPT_PATH / "../../.." / "output"


class GazeModel(Model):
    def __init__(self, name, download_url, file_name):
        super().__init__(name, download_url, file_name)
        self.model = None
        self.normalized_camera_matrix = None
        self.normalized_distance = None
        self.normalized_image_height = None
        self.normalized_image_width = None

    def get_output_directory(self, video, face_model_name):
        video_name = str(video.name).split(".")[0]
        return OUTPUT_DIRECTORY / video_name / "gaze" / f"{self.name}_{face_model_name}"

    @abstractmethod
    def detect_gaze(self, image, face, video, face_model_name, frame_number):
        pass

    @staticmethod
    def get_vector_from_pitch_and_yaw(pitch, yaw):
        return -numpy.array(
            [
                numpy.cos(pitch) * numpy.sin(yaw),
                numpy.sin(pitch),
                numpy.cos(pitch) * numpy.cos(yaw),
            ]
        )

    def download(self):
        super().download()
        self.load_model()

    @abstractmethod
    def load_model(self):
        raise NotImplementedError()

    # reference : https://github.com/hysts/pytorch_mpiigaze_demo/blob/47cdf68414d20c8281bbb0a03112a298761aaa9b/ptgaze/head_pose_estimation/head_pose_normalizer.py#L27-L37
    def normalize_input_image(self, image, face: FrameFaceData, video):
        inverse_camera_matrix = numpy.linalg.inv(video.camera_matrix)
        normalized_camera_matrix = self.normalized_camera_matrix

        normalized_distance = self.normalized_distance

        # define scale matrix
        scale = numpy.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, normalized_distance / face.face_distance],
            ],
            dtype="double",
        )

        # calculate conversion matrix
        conversion_matrix = scale @ face.normalized_rotation

        # calculate projection matrix for normalization
        projection_matrix = (
                normalized_camera_matrix @ conversion_matrix @ inverse_camera_matrix
        )

        # normalize the give image with the calculated projection matrix
        return cv2.warpPerspective(
            image,
            projection_matrix,
            (
                self.normalized_image_width,
                self.normalized_image_height,
            ),
        )
