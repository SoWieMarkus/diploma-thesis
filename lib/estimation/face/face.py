import cv2
import numpy

from ..face_models.face_model import FaceModel
from ...utils.math import (
    convert_rotation_vector_to_matrix,
)
from ...videos.video import Video


class FaceLandmark:
    """
    Class that represents
    """

    def __init__(self, x: float, y: float, z: float, image_x: int, image_y: int):
        """
        :param x:
        :param y:
        :param z:
        :param image_x:
        :param image_y:
        """
        self.x = x
        self.y = y
        self.z = z
        self.image_x = image_x
        self.image_y = image_y

    def __str__(self):
        return f"{round(self.x, 4)},{round(self.y, 4)},{round(self.z, 4)},{round(self.image_x, 4)},{round(self.image_y, 4)}"


class Face:
    """

    """

    def __init__(self, landmarks, average_blink_score) -> None:
        """

        :param landmarks:
        :param average_blink_score:
        """
        self.distance = None
        self.bounding_box = None
        self.center = None
        self.position = None
        self.head_rotation_matrix = None
        self.landmarks_3d = None
        self.head_rotation_vector = None
        self.landmarks = landmarks
        self.image_points = []
        self.calculate_image_points()
        self.average_blink_score = average_blink_score

    def calculate_image_points(self):
        """

        :return:
        """
        image_points = []
        for landmark in self.landmarks:
            image_points.append((landmark.image_x, landmark.image_y))

        self.image_points = numpy.array(image_points, dtype="double")

    def apply(self, face_model: FaceModel, video: Video):
        rotation_vector = numpy.array([0, 0, 0], dtype="double")
        translation_vector = numpy.array([0, 0, 1], dtype="double")

        # map 2D image points of landmarks to 3d template model
        success, rotation_vector, translation_vector = cv2.solvePnP(
            face_model.template_model.landmarks,
            self.image_points,
            video.camera_matrix,
            video.distortion_coefficients,
            rotation_vector,
            translation_vector,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            raise ValueError(
                "Something unexpected happened while mapping landmarks to template models"
            )

        self.head_rotation_vector = rotation_vector

        # convert rotation vector into rotation matrix
        self.head_rotation_matrix = convert_rotation_vector_to_matrix(rotation_vector)
        self.position = translation_vector

        # transform 3d points of template model to represent the 2d image points rotation and position in 3d space
        self.landmarks_3d = (
                face_model.template_model.landmarks @ self.head_rotation_matrix.T
                + self.position
        )

        # calculate 3d center of the face based on the outer corners of the eyes and nose.
        # see: https://files.ait.ethz.ch/projects/xgaze/xucongzhang2020eccv.pdf
        self.center = self.landmarks_3d[
            numpy.concatenate(
                [
                    face_model.template_model.get_right_eye_indices(),
                    face_model.template_model.get_left_eye_indices(),
                    face_model.template_model.get_nose_indices(),
                ]
            )
        ].mean(axis=0)

        # calculate distance of face to camera
        self.distance = numpy.linalg.norm(self.center)

        # calculate bounding box of face in image
        self.bounding_box = self.get_bounding_box()

    def get_bounding_box(self):
        """
        Get the bounding box of the face in the image based on the detected landmarks
        """
        min_x = numpy.min(self.image_points[:, 0])
        max_x = numpy.max(self.image_points[:, 0])
        min_y = numpy.min(self.image_points[:, 1])
        max_y = numpy.max(self.image_points[:, 1])

        bounding_box = [(min_x, min_y), (max_x, max_y)]

        return bounding_box
