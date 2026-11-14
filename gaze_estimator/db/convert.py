"""Conversion between pipeline results and database rows."""

from __future__ import annotations

import numpy as np

from ..core import types
from . import models


def _list(array: np.ndarray, decimals: int = 6) -> list:
    return np.round(np.asarray(array, dtype=np.float64), decimals).tolist()


def face_row(
    face: types.Face | None, *, frame_id: int, video_id: int, detector: str, runtime_ms: float
) -> models.Face:
    if face is None:
        return models.Face(
            frame_id=frame_id,
            video_id=video_id,
            detector=detector,
            detected=False,
            runtime_ms=runtime_ms,
        )
    return models.Face(
        frame_id=frame_id,
        video_id=video_id,
        detector=detector,
        detected=True,
        runtime_ms=runtime_ms,
        bbox=_list(face.bbox, 2),
        landmarks_2d=_list(face.landmarks_2d, 2),
        landmarks_3d=_list(face.landmarks_3d, 5),
        rotation_vector=_list(face.rotation_vector),
        translation_vector=_list(face.translation_vector),
        center=_list(face.center),
        normalizing_rotation=_list(face.normalizing_rotation),
        distance=face.distance,
        ear_left=face.ear_left,
        ear_right=face.ear_right,
    )


def gaze_row(gaze: types.Gaze, *, face: models.Face, runtime_ms: float) -> models.Gaze:
    return models.Gaze(
        face_id=face.id,
        frame_id=face.frame_id,
        video_id=face.video_id,
        model=gaze.model,
        pitch=gaze.pitch,
        yaw=gaze.yaw,
        vector=_list(gaze.vector),
        runtime_ms=runtime_ms,
    )
