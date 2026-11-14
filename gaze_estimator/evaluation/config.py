"""Evaluation configuration (``eval.yaml``). Every value has a default, so the file is optional."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Range:
    """``steps`` values starting at ``min`` with step ``(max - min) / steps`` (as in the thesis)."""

    min: float
    max: float
    steps: int

    @property
    def values(self) -> np.ndarray:
        return self.min + (self.max - self.min) / self.steps * np.arange(self.steps)


@dataclass
class DisplayConfig:
    width: float = 0.07
    height: float = 0.15
    offset: float = 0.01


@dataclass
class EvalConfig:
    max_distance: float = 1.0
    """Faces further away than this (meters) are ignored, as in the thesis."""
    exclude_videos: dict[str, str] = field(default_factory=dict)
    """Videos excluded from the evaluation, mapped to the reason (decide *before* looking at results)."""
    blink_as_not_looking: bool = True
    """Frames labeled as blink count as 'not looking' (thesis behaviour)."""
    optimize: str = "f1"
    """Metric used to pick display parameters in the grid search and cross-validation: f1 or mcc."""
    display: DisplayConfig = field(default_factory=DisplayConfig)
    """Physical display used for the angle score and ROC curves."""
    size: Range = field(default_factory=lambda: Range(0.0, 0.5, 50))
    """Grid of the square display size ``l`` (thesis)."""
    offset: Range = field(default_factory=lambda: Range(-0.1, 0.5, 50))
    """Grid of the camera offset ``p`` (thesis)."""
    ear: Range = field(default_factory=lambda: Range(0.0, 1.0, 100))
    """EAR thresholds for the blink evaluation."""
    smoothing_window: int = 5
    """Window (in processed frames) of the temporal majority filter, <= 1 disables it."""
    per_video_plots: bool = True
    formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 200

    @classmethod
    def load(cls, path: Path | None) -> EvalConfig:
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {', '.join(sorted(unknown))}")
        config = cls(
            **{k: v for k, v in data.items() if k not in {"display", "size", "offset", "ear"}}
        )
        if "display" in data:
            config.display = DisplayConfig(**data["display"])
        for key in ("size", "offset", "ear"):
            if key in data:
                setattr(config, key, Range(**data[key]))
        if config.optimize not in {"f1", "mcc"}:
            raise ValueError("optimize must be 'f1' or 'mcc'")
        return config
