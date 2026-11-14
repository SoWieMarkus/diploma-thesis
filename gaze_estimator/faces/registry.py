"""Registry of face detectors. Detectors are imported lazily so optional dependencies stay optional."""

from __future__ import annotations

from importlib import import_module

from .base import FaceDetector

FACE_DETECTORS: dict[str, str] = {
    "mediapipe": "gaze_estimator.faces.mediapipe:MediapipeFaceDetector",
    "dlib": "gaze_estimator.faces.dlib:DlibFaceDetector",
}


def create_face_detector(name: str) -> FaceDetector:
    if name not in FACE_DETECTORS:
        raise ValueError(f"Unknown face detector '{name}'. Available: {', '.join(FACE_DETECTORS)}")
    module, cls = FACE_DETECTORS[name].split(":")
    return getattr(import_module(module), cls)()


def resolve_names(selection: list[str] | None) -> list[str]:
    """Expand ``["all"]`` (or ``None``) into all registered names and validate the others."""
    if not selection or "all" in selection:
        return list(FACE_DETECTORS)
    unknown = [name for name in selection if name not in FACE_DETECTORS]
    if unknown:
        raise ValueError(f"Unknown face detector(s): {', '.join(unknown)}")
    return selection
