"""Common gaze estimator logic, including the image normalization of Zhang et al. (2018)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from ..core.camera import Camera
from ..core.geometry import normalize, vector_from_pitch_yaw
from ..core.types import Face, Gaze


class GazeEstimator(ABC):
    name: str

    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray, face: Face, camera: Camera) -> Gaze:
        """Estimate the gaze of ``face`` in ``frame_bgr``."""


@dataclass(frozen=True)
class Normalization:
    """Parameters of the normalized camera an appearance based model was trained with."""

    focal_length: float
    distance: float
    size: int

    @property
    def camera_matrix(self) -> np.ndarray:
        c = self.size / 2
        return np.array(
            [[self.focal_length, 0.0, c], [0.0, self.focal_length, c], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def normalize_image(
    frame_bgr: np.ndarray, face: Face, camera: Camera, normalization: Normalization
) -> np.ndarray:
    """Warp the image into a virtual camera looking at the face center from a fixed distance.

    Reference: https://github.com/hysts/pytorch_mpiigaze_demo (head_pose_normalizer.py)
    """
    scale = np.diag([1.0, 1.0, normalization.distance / face.distance])
    conversion = scale @ face.normalizing_rotation
    projection = normalization.camera_matrix @ conversion @ np.linalg.inv(camera.matrix)
    return cv2.warpPerspective(frame_bgr, projection, (normalization.size, normalization.size))


class AppearanceGazeEstimator(GazeEstimator):
    """A CNN that predicts pitch and yaw from a normalized face image."""

    normalization: Normalization

    @abstractmethod
    def predict(self, normalized_bgr: np.ndarray) -> tuple[float, float]:
        """Return ``(pitch, yaw)`` in radians in the normalized camera space."""

    def estimate(self, frame_bgr: np.ndarray, face: Face, camera: Camera) -> Gaze:
        normalized = normalize_image(frame_bgr, face, camera, self.normalization)
        pitch, yaw = self.predict(normalized)
        # Rotate the gaze from the normalized camera back into the real camera coordinates.
        vector = normalize(vector_from_pitch_yaw(pitch, yaw) @ face.normalizing_rotation)
        return Gaze(model=self.name, pitch=float(pitch), yaw=float(yaw), vector=vector)
