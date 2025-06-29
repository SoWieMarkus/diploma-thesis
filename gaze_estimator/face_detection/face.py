from dataclasses import dataclass
import numpy


@dataclass
class FaceLandmark:
    """
    Represents a single facial landmark with its coordinates and image position.
    """

    normalized_x: int
    """Normalized x-coordinate of the landmark in the range [0, 1] (0, width)."""

    normalized_y: int
    """Normalized y-coordinate of the landmark in the range [0, 1] (0, height)."""

    normalized_z: int
    """
    Normalized z-coordinate of the landmark, typically set to 1 as depth information is not provided.
    Mediapipe has a build in feature to calculate the depth of the face landmarks.
    We store it here to compare it with our 3D face position later.
    But appart from that, it is not used.
    """

    image_x: int
    """x-coordinate (in px) of the landmark in the original image."""

    image_y: int
    """y-coordinate (in px) of the landmark in the original image."""


@dataclass
class Face:
    """
    Represents a detected face with its landmarks and additional properties.
    """

    landmarks: list[FaceLandmark]
    """List of facial landmarks detected on the face."""

    eye_aspect_ratio_left: float
    """Average eye aspect ratio for the left eye, used to estimate blinks."""

    eye_aspect_ratio_right: float
    """Average eye aspect ratio for the right eye, used to estimate blinks."""

    image_points: list[tuple[int, int]]
    """List of image points corresponding to the landmarks in pixel coordinates."""

    distance: float
    """Distance from the camera to the face, calculated during pose estimation."""

    bounding_box: tuple[tuple[int, int], tuple[int, int]]
    """Bounding box of the face in pixels in the format ((x_min, y_min), (x_max, y_max))."""

    center: numpy.array
    """Center of the face calculated from the 3D landmarks of eyes and nose."""

    position: numpy.array
    """Position of the face in 3D space, the translation vector from the camera."""

    head_rotation_matrix: numpy.array
    """Rotation matrix representing the head's orientation in 3D space."""

    head_rotation_vector: numpy.array
    """Rotation vector representing the head's orientation in 3D space."""

    landmarks_3d: numpy.array
    """3D coordinates of the landmarks."""

    normalized_rotation_matrix: numpy.array
    """Normalized rotation matrix for the face, used for gaze estimation."""
