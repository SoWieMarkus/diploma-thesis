"""Results of the face detectors and gaze estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Face:
    """A detected face, its landmarks and its estimated 3D head pose (camera coordinates, meters)."""

    detector: str
    landmarks_2d: np.ndarray
    """``(N, 2)`` landmark positions in pixels."""
    landmarks_3d: np.ndarray
    """``(N, 3)`` landmarks of the fitted 3D template in camera coordinates."""
    bbox: tuple[float, float, float, float]
    """``(x_min, y_min, x_max, y_max)`` in pixels."""
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    rotation_matrix: np.ndarray
    center: np.ndarray
    """Center between eye corners and nose (origin of the gaze ray), see ETH-XGaze."""
    normalizing_rotation: np.ndarray
    """Rotation into the normalized camera space used by the appearance based gaze models."""
    ear_left: float
    ear_right: float

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.center))

    @property
    def ear(self) -> float:
        return (self.ear_left + self.ear_right) / 2


@dataclass
class Gaze:
    model: str
    pitch: float
    """Pitch in the normalized space of the model (radians)."""
    yaw: float
    """Yaw in the normalized space of the model (radians)."""
    vector: np.ndarray
    """Unit gaze direction in camera coordinates."""
