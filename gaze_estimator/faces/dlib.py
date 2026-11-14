"""Face detection with dlib (HOG frontal face detector + 68 landmark shape predictor).

dlib is an optional dependency (``uv sync --extra dlib``) because it is compiled from source.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.paths import ASSETS_DIR, asset
from .base import FaceDetector, FaceTemplate

try:
    import dlib
except ImportError as error:  # pragma: no cover - depends on the environment
    raise ImportError(
        "The dlib face detector needs the optional dependency dlib: `uv sync --extra dlib`."
    ) from error


def dlib_template() -> FaceTemplate:
    return FaceTemplate.from_csv(
        ASSETS_DIR / "template-model-3d-dlib.csv",
        68,
        right_eye=(36, 39),
        left_eye=(42, 45),
        mouth=(48, 54),
        nose=(31, 35),
        nose_tip=30,
        right_eye_ear=(36, 37, 38, 39, 40, 41),
        left_eye_ear=(42, 43, 44, 45, 46, 47),
    )


class DlibFaceDetector(FaceDetector):
    name = "dlib"

    def __init__(self, upsample: int = 0):
        self.template = dlib_template()
        self.upsample = upsample
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(asset("face-model-dlib.dat")))

    def detect_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes = self.detector(rgb, self.upsample)
        if len(boxes) == 0:
            return None
        # On a bicycle there is usually only one person in front of the camera: take the largest face.
        box = max(boxes, key=lambda b: b.width() * b.height())
        shape = self.predictor(rgb, box)
        return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float64)
