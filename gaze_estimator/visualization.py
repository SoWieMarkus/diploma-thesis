"""OpenCV overlays for the live demo."""

from __future__ import annotations

import cv2
import numpy as np

from .core.camera import Camera
from .core.types import Face, Gaze
from .faces.base import FaceTemplate

GREEN = (80, 220, 100)
RED = (70, 70, 230)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
GRAY = (170, 170, 170)


def project(points: np.ndarray, camera: Camera) -> np.ndarray:
    points2d, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 3),
        np.zeros(3),
        np.zeros(3),
        camera.matrix,
        camera.distortion_coefficients,
    )
    return points2d.reshape(-1, 2)


def _pt(point) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def draw_face(frame: np.ndarray, face: Face, template: FaceTemplate, all_landmarks: bool = True):
    x0, y0, x1, y1 = face.bbox
    cv2.rectangle(frame, _pt((x0, y0)), _pt((x1, y1)), GRAY, 1, cv2.LINE_AA)

    if all_landmarks:
        for point in face.landmarks_2d:
            cv2.circle(frame, _pt(point), 1, (212, 234, 94), cv2.FILLED, cv2.LINE_AA)

    highlights = [
        (template.right_eye, (0, 255, 255)),
        (template.left_eye, (255, 0, 255)),
        (template.mouth, (255, 0, 0)),
        (template.nose, (0, 255, 0)),
        ((template.nose_tip,), (0, 255, 0)),
    ]
    for indices, color in highlights:
        for index in indices:
            cv2.circle(frame, _pt(face.landmarks_2d[index]), 3, color, cv2.FILLED, cv2.LINE_AA)


def draw_head_axes(frame: np.ndarray, face: Face, camera: Camera, length: float = 0.05):
    axes = face.center + length * face.rotation_matrix.T
    points = project(np.vstack([face.center, axes]), camera)
    origin = _pt(points[0])
    for point, color in zip(points[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)], strict=True):
        cv2.line(frame, origin, _pt(point), color, 2, cv2.LINE_AA)


def draw_gaze(frame: np.ndarray, face: Face, gaze: Gaze, camera: Camera, length: float = 0.2):
    points = project(np.vstack([face.center, face.center + length * gaze.vector]), camera)
    cv2.arrowedLine(frame, _pt(points[0]), _pt(points[1]), MAGENTA, 2, cv2.LINE_AA, tipLength=0.15)


def draw_text_block(frame: np.ndarray, lines: list[str], origin=(12, 24)):
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        y += 22


def draw_status(frame: np.ndarray, looking: bool | None):
    if looking is None:
        return
    text = "LOOKING AT DISPLAY" if looking else "NOT LOOKING AT DISPLAY"
    color = GREEN if looking else RED
    height, width = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = width - tw - 24, height - 24
    cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + 10), color, cv2.FILLED)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
