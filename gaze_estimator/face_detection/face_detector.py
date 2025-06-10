from abc import abstractmethod, ABC
from .face import Face, FaceLandmark
from ..common import VideoCaptureMetaData, get_bounding_box
import logging
import csv
import cv2
import numpy


class FaceTemplateModel(ABC):
    """
    Abstract base class for a 3D face template model.
    This class is responsible for loading a 3D face template model from a CSV file
    and providing methods to access specific landmark indices for facial features.
    This model can then be used to estimate a 3-dimensional face position and rotation
    based on 2D landmarks detected in an image solving the PnP problem.
    The template model is expected to be in a CSV format where each row corresponds to a landmark.
    The landmarks are expected to be in the format [x, y, z] for each landmark.
    Args:
        file (str): Path to the CSV file containing the 3D face template model.
    """

    def __init__(self, file: str):
        logging.info(f"Reading 3D face template model: '{file}'")

        with open(file, newline="\n") as csv_file:
            reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            self.landmarks = numpy.array(
                [[float(column) for column in row] for row in reader], dtype="double"
            )
        logging.info(
            f"3D face template model loaded with {len(self.landmarks)} landmarks."
        )

    @abstractmethod
    def get_right_eye_ear_indices(self) -> list[int]:
        """Get indices for the right eye landmarks of the template model for the ear calculation."""
        raise NotImplementedError()

    @abstractmethod
    def get_left_eye_ear_indices(self) -> list[int]:
        """Get indices for the left eye landmarks of the template model for the ear calculation."""
        raise NotImplementedError()

    @abstractmethod
    def get_right_eye_indices(self) -> numpy.array:
        """Get landmark indices that are enclosing the right eye in the template model left and right."""
        raise NotImplementedError()

    @abstractmethod
    def get_left_eye_indices(self) -> numpy.array:
        """Get landmark indices that are enclosing the left eye in the template model left and right."""
        raise NotImplementedError()

    @abstractmethod
    def get_mouth_indices(self) -> numpy.array:
        """Get landmark indices that are enclosing the mouth in the template model left and right."""
        raise NotImplementedError()

    @abstractmethod
    def get_nose_indices(self) -> numpy.array:
        """Get landmark indices that are enclosing the nose in the template model left and right."""
        raise NotImplementedError()

    @abstractmethod
    def get_nose_tip_index(self) -> numpy.array:
        """Get the index of the nose tip landmark in the template model."""
        raise NotImplementedError()

    def get_important_indices(self) -> numpy.array:
        """
        Get all important indices of the template model.
        This includes the indices of the eyes, mouth, and nose.
        Returns:
            numpy.array: Array of important landmark indices.
        """
        return numpy.concatenate(
            [
                self.get_right_eye_indices(),
                self.get_left_eye_indices(),
                self.get_mouth_indices(),
                self.get_nose_indices(),
                [self.get_nose_tip_index()],
            ]
        )


class FaceDetector(ABC):
    """
    Abstract base class for face detection.
    This class defines the interface for face detection algorithms.
    It requires subclasses to implement the `detect_face` method.
    Concrete face detectors are then used to detect face landmarks inside the frame of the video source.
    """

    def __init__(self, name: str, template_model: FaceTemplateModel):
        """
        Abstract base class for face detection.
        Args:
            name (str): Name of the face detector.
            template_model (FaceTemplateModel): The 3D face template model to be used for face detection.
        """
        self.name = name
        self.template_model = template_model

    @abstractmethod
    def detect_face(
        self, frame: numpy.ndarray, meta: VideoCaptureMetaData
    ) -> Face | None:
        """
        Detect a face in the given frame.
        Args:
            frame (numpy.ndarray): The image frame in which to detect the face.
            meta (VideoCaptureMetaData): Metadata about the video capture, such as resolution and frame rate.
        Returns:
            Face | None: Returns a Face object if a face is detected, otherwise returns None.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def process_landmarks(
        self, landmarks: list[FaceLandmark], meta: VideoCaptureMetaData
    ) -> Face | None:
        if not landmarks:
            logging.error("No landmarks provided for face processing.")
            return None

        image_points = numpy.array(
            [(landmark.image_x, landmark.image_y) for landmark in landmarks],
            dtype="double",
        )

        # Calculate the eye aspect ratio for both eyes using the template model's indices
        # This is used to determine the openness of the eyes.
        eye_aspect_ratio_left = get_eye_aspect_ratio(
            landmarks, self.template_model.get_left_eye_ear_indices()
        )
        eye_aspect_ratio_right = get_eye_aspect_ratio(
            landmarks, self.template_model.get_right_eye_ear_indices()
        )

        rotation_vector = numpy.array([0, 0, 0], dtype="double")
        translation_vector = numpy.array([0, 0, 1], dtype="double")

        # Solve the PnP problem to estimate the pose of the face in 3D space.
        # For this, we use the landmarks detected in the image and map them to the 3D template model.
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.template_model.landmarks,
            image_points,
            meta.camera_matrix,
            meta.distortion_coefficients,
            rotation_vector,
            translation_vector,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            logging.error("Failed to solve PnP problem for face landmarks.")
            return None

        # TODO this could lead to an error
        head_rotation_matrix = cv2.Rodrigues(rotation_vector)[0]

        landmarks_3d = (
            self.template_model.landmarks @ head_rotation_matrix.T + translation_vector
        )

        # Calculate 3d center of the face based on the outer corners of the eyes and nose.
        # see: https://files.ait.ethz.ch/projects/xgaze/xucongzhang2020eccv.pdf
        center = landmarks_3d[
            numpy.concatenate(
                [
                    self.template_model.get_right_eye_indices(),
                    self.template_model.get_left_eye_indices(),
                    self.template_model.get_nose_indices(),
                ]
            )
        ].mean(axis=0)

        distance = numpy.linalg.norm(center)

        bounding_box = get_bounding_box(image_points)

        # TODO das vielleicht in eigene Methode auslagern
        z_axis = center / numpy.linalg.norm(center)
        head_x_axis = head_rotation_matrix[:, 0]

        cross_y = numpy.cross(z_axis, head_x_axis)
        y_axis = cross_y / numpy.linalg.norm(cross_y)

        cross_x = numpy.cross(y_axis, z_axis)
        x_axis = cross_x / numpy.linalg.norm(cross_x)

        normalized_rotation_matrix = numpy.vstack([x_axis, y_axis, z_axis])

        return Face(
            landmarks=landmarks,
            image_points=image_points,
            eye_aspect_ratio_left=eye_aspect_ratio_left,
            eye_aspect_ratio_right=eye_aspect_ratio_right,
            head_rotation_vector=rotation_vector,
            head_rotation_matrix=head_rotation_matrix,
            position=translation_vector,
            landmarks_3d=landmarks_3d,
            center=center,
            distance=distance,
            bounding_box=bounding_box,
            normalized_rotation_matrix=normalized_rotation_matrix,
        )


def get_eye_aspect_ratio(landmarks: list[FaceLandmark], eye_landmark_indices) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) based on the eye landmarks.
    The EAR is a measure of the eye's openness and is used to detect blinks.
    See https://ieeexplore.ieee.org/document/9509908
    Returns:
        float: The calculated EAR value.
    """
    p1 = landmarks[eye_landmark_indices[0]]
    p2 = landmarks[eye_landmark_indices[1]]
    p3 = landmarks[eye_landmark_indices[2]]
    p4 = landmarks[eye_landmark_indices[3]]
    p5 = landmarks[eye_landmark_indices[4]]
    p6 = landmarks[eye_landmark_indices[5]]

    vertical_distance_1 = numpy.linalg.norm(
        numpy.array([p2.image_x, p2.image_y]) - numpy.array([p6.image_x, p6.image_y])
    )
    vertical_distance_2 = numpy.linalg.norm(
        numpy.array([p3.image_x, p3.image_y]) - numpy.array([p5.image_x, p5.image_y])
    )
    horizontal_distance = numpy.linalg.norm(
        numpy.array([p1.image_x, p1.image_y]) - numpy.array([p4.image_x, p4.image_y])
    )

    return (vertical_distance_1 + vertical_distance_2) / (2 * horizontal_distance)
