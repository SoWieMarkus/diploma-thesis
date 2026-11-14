"""Common face detector logic: head pose estimation with PnP and eye aspect ratio."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..core.camera import Camera
from ..core.types import Face

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceTemplate:
    """A generic 3D face model (meters) with the indices of important landmarks.

    The templates were taken from https://github.com/hysts/pytorch_mpiigaze_demo (MIT).
    """

    landmarks: np.ndarray
    right_eye: tuple[int, int]
    """Outer and inner corner of the right eye."""
    left_eye: tuple[int, int]
    mouth: tuple[int, int]
    nose: tuple[int, int]
    nose_tip: int
    right_eye_ear: tuple[int, int, int, int, int, int]
    """Six landmarks p1..p6 around the right eye used for the eye aspect ratio."""
    left_eye_ear: tuple[int, int, int, int, int, int]

    @classmethod
    def from_csv(cls, path: Path, expected_landmarks: int, **indices) -> FaceTemplate:
        landmarks = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if landmarks.shape != (expected_landmarks, 3):
            raise ValueError(
                f"3D template '{path}' has shape {landmarks.shape}, expected ({expected_landmarks}, 3)."
            )
        return cls(landmarks=landmarks, **indices)

    @property
    def center_indices(self) -> list[int]:
        return [*self.right_eye, *self.left_eye, *self.nose]

    @property
    def important_indices(self) -> list[int]:
        return [*self.right_eye, *self.left_eye, *self.mouth, *self.nose, self.nose_tip]


def eye_aspect_ratio(points: np.ndarray, indices: tuple[int, ...]) -> float:
    """Eye aspect ratio (Soukupova and Cech, 2016): small values indicate a closed eye."""
    p1, p2, p3, p4, p5, p6 = (points[i] for i in indices)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return float(vertical / (2 * horizontal)) if horizontal > 0 else 0.0


class FaceDetector(ABC):
    name: str
    template: FaceTemplate

    @abstractmethod
    def detect_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Return ``(N, 2)`` landmark positions in pixels matching the template, or ``None``."""

    def detect(self, frame_bgr: np.ndarray, camera: Camera) -> Face | None:
        landmarks = self.detect_landmarks(frame_bgr)
        if landmarks is None:
            return None
        return self.estimate_pose(landmarks, camera)

    def estimate_pose(self, landmarks_2d: np.ndarray, camera: Camera) -> Face | None:
        """Fit the 3D template to the 2D landmarks (PnP) and derive the head pose."""
        image_points = np.ascontiguousarray(landmarks_2d, dtype=np.float64)
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            self.template.landmarks,
            image_points,
            camera.matrix,
            camera.distortion_coefficients,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            logger.debug("solvePnP failed for %s", self.name)
            return None

        rvec, tvec = rvec.reshape(3), tvec.reshape(3)
        rotation = cv2.Rodrigues(rvec)[0]
        landmarks_3d = self.template.landmarks @ rotation.T + tvec

        # Face center as in ETH-XGaze: mean of the eye corners and the nose.
        # See https://files.ait.ethz.ch/projects/xgaze/xucongzhang2020eccv.pdf
        center = landmarks_3d[self.template.center_indices].mean(axis=0)
        if center[2] <= 0:
            return None

        # Normalizing rotation: the virtual camera looks at the face center, its x-axis is aligned
        # with the head's x-axis.
        z_axis = center / np.linalg.norm(center)
        y_axis = np.cross(z_axis, rotation[:, 0])
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        normalizing_rotation = np.vstack([x_axis, y_axis, z_axis])

        x_min, y_min = image_points.min(axis=0)
        x_max, y_max = image_points.max(axis=0)

        return Face(
            detector=self.name,
            landmarks_2d=image_points,
            landmarks_3d=landmarks_3d,
            bbox=(float(x_min), float(y_min), float(x_max), float(y_max)),
            rotation_vector=rvec,
            translation_vector=tvec,
            rotation_matrix=rotation,
            center=center,
            normalizing_rotation=normalizing_rotation,
            ear_left=eye_aspect_ratio(image_points, self.template.left_eye_ear),
            ear_right=eye_aspect_ratio(image_points, self.template.right_eye_ear),
        )

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""
