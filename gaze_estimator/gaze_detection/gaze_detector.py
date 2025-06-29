from abc import ABC, abstractmethod
import os
import logging
import torch
import numpy
import cv2

from .gaze import Gaze
from ..common.camera import VideoCaptureMetaData
from ..face_detection import Face
from torchvision.transforms import Compose

ASSETS_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "..", "data", "assets")


class GazeDetector(ABC):

    def __init__(
        self,
        name: str,
        download_url: str | None = None,
        model_file_name: str | None = None,
        normalized_camera_matrix: numpy.array = None,
        normalized_distance: float | None = None,
        normalized_image_height: int | None = None,
        normalized_image_width: int | None = None,
        transform: Compose | None = None,
    ):
        self.download_url = download_url
        self.model_file_name = model_file_name
        self.name = name
        self.model = None
        self.normalized_camera_matrix = normalized_camera_matrix
        self.normalized_distance = normalized_distance
        self.normalized_image_height = normalized_image_height
        self.normalized_image_width = normalized_image_width
        self.transform = transform

        if self.model_file_name is None:
            return

        # Download and load the model if it does not exist
        self.model_file_path = (
            os.path.join(ASSETS_DIRECTORY, model_file_name) if model_file_name else None
        )

        if not os.path.exists(self.model_file_path):
            logging.info(f"Downloading {name} model...")
            torch.hub.download_url_to_file(
                self.download_url, self.model_file_path, progress=True
            )
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the model from the specified file and prepare it."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def detect_gaze(self, face: Face, normalized_frame: numpy.ndarray) -> Gaze:
        """
        Detect the gaze vector from the normalized frame and face data.
        Args:
            face (Face): The face data containing pose and distance information.
            normalized_frame (numpy.ndarray): The normalized image frame.
        Returns:
            Gaze: The detected gaze vector.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    # Reference : https://github.com/hysts/pytorch_mpiigaze_demo/blob/47cdf68414d20c8281bbb0a03112a298761aaa9b/ptgaze/head_pose_estimation/head_pose_normalizer.py#L27-L37
    def normalize_input_image(
        self, image: numpy.ndarray, face: Face, meta: VideoCaptureMetaData
    ) -> numpy.ndarray:
        """Normalize the input image based on the face's pose and camera parameters."""
        inverse_camera_matrix = numpy.linalg.inv(meta.camera_matrix)

        normalized_camera_matrix = self.normalized_camera_matrix
        normalized_distance = self.normalized_distance

        scale = numpy.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, normalized_distance / face.distance],
            ],
            dtype="double",
        )

        # calculate conversion matrix
        conversion_matrix = scale @ face.normalized_rotation_matrix

        # calculate projection matrix for normalization
        projection_matrix = (
            normalized_camera_matrix @ conversion_matrix @ inverse_camera_matrix
        )

        # Normalize the image using the projection matrix
        return cv2.warpPerspective(
            image,
            projection_matrix,
            (
                self.normalized_image_width,
                self.normalized_image_height,
            ),
        )
