"""Registry of gaze estimators. Estimators are imported lazily (torch is slow to import)."""

from __future__ import annotations

from importlib import import_module

from .base import GazeEstimator

GAZE_ESTIMATORS: dict[str, str] = {
    "eth-xgaze": "gaze_estimator.gaze.eth_xgaze:ETHXGazeEstimator",
    "l2cs-gaze360": "gaze_estimator.gaze.l2cs_gaze360:L2CSGaze360Estimator",
    "mpiifacegaze": "gaze_estimator.gaze.mpiifacegaze:MPIIFaceGazeEstimator",
    "head-rotation": "gaze_estimator.gaze.head_rotation:HeadRotationEstimator",
}


def create_gaze_estimator(name: str) -> GazeEstimator:
    if name not in GAZE_ESTIMATORS:
        raise ValueError(f"Unknown gaze model '{name}'. Available: {', '.join(GAZE_ESTIMATORS)}")
    module, cls = GAZE_ESTIMATORS[name].split(":")
    return getattr(import_module(module), cls)()


def resolve_names(selection: list[str] | None) -> list[str]:
    if not selection or "all" in selection:
        return list(GAZE_ESTIMATORS)
    unknown = [name for name in selection if name not in GAZE_ESTIMATORS]
    if unknown:
        raise ValueError(f"Unknown gaze model(s): {', '.join(unknown)}")
    return selection
