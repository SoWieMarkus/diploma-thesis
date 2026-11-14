"""Baseline without a gaze model: the gaze is assumed to follow the head orientation."""

from __future__ import annotations

import numpy as np

from ..core.camera import Camera
from ..core.geometry import normalize, pitch_yaw_from_vector
from ..core.types import Face, Gaze
from .base import GazeEstimator


class HeadRotationEstimator(GazeEstimator):
    name = "head-rotation"

    def estimate(self, frame_bgr: np.ndarray, face: Face, camera: Camera) -> Gaze:
        # The face template looks along -z (towards the camera) in its own coordinate system.
        vector = normalize(face.rotation_matrix @ np.array([0.0, 0.0, -1.0]))
        pitch, yaw = pitch_yaw_from_vector(face.normalizing_rotation @ vector)
        return Gaze(model=self.name, pitch=pitch, yaw=yaw, vector=vector)
