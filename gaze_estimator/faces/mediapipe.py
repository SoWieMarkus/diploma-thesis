"""Face detection with the MediaPipe Tasks ``FaceLandmarker`` (468 face + 10 iris landmarks)."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from ..core.paths import ASSETS_DIR, asset
from .base import FaceDetector, FaceTemplate

MODEL_FILE = "face-landmarker.task"
TEMPLATE_LANDMARKS = 468


def mediapipe_template() -> FaceTemplate:
    return FaceTemplate.from_csv(
        ASSETS_DIR / "template-model-3d-mediapipe.csv",
        TEMPLATE_LANDMARKS,
        right_eye=(33, 133),
        left_eye=(362, 263),
        mouth=(78, 308),
        nose=(240, 460),
        nose_tip=1,
        right_eye_ear=(33, 160, 158, 133, 153, 144),
        left_eye_ear=(362, 385, 387, 263, 373, 380),
    )


class MediapipeFaceDetector(FaceDetector):
    name = "mediapipe"

    def __init__(self, min_confidence: float = 0.5):
        self.template = mediapipe_template()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(asset(MODEL_FILE))),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None
        # The iris landmarks (468-477) are not part of the 3D template, so they are dropped.
        landmarks = result.face_landmarks[0][:TEMPLATE_LANDMARKS]
        return np.array([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float64)

    def close(self) -> None:
        self.landmarker.close()
