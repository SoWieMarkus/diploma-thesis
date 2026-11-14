"""The actual evaluation: display grid search, cross-validation, angle score, smoothing and blinks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..core.geometry import Display, intersect_camera_plane
from . import metrics
from .config import EvalConfig
from .dataset import Dataset, origins_and_directions
from .metrics import Confusion

ANGLE_THRESHOLDS = np.radians(np.arange(0.0, 45.25, 0.25))


# --------------------------------------------------------------------------------------------------
# predictions
# --------------------------------------------------------------------------------------------------


def grid_predictions(points: np.ndarray, sizes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """``(N, S, P)`` hits of camera-plane points for every square display size and offset."""
    x = points[:, 0][:, None, None]
    y = points[:, 1][:, None, None]
    s = sizes[None, :, None]
    p = offsets[None, None, :]
    with np.errstate(invalid="ignore"):
        return (np.abs(x) <= s / 2) & (y >= -p) & (y <= s - p)


def majority_filter(predictions: np.ndarray, window: int) -> np.ndarray:
    """Centered temporal majority vote along axis 0 (frames must be sorted by time)."""
    if window <= 1 or len(predictions) == 0:
        return predictions
    half = window // 2
    values = predictions.astype(np.int32)
    padded = np.concatenate(
        [
            np.zeros((half,) + values.shape[1:], np.int32),
            values,
            np.zeros((half,) + values.shape[1:], np.int32),
        ]
    )
    cumulative = np.concatenate(
        [np.zeros((1,) + values.shape[1:], np.int32), np.cumsum(padded, axis=0)]
    )
    sums = cumulative[window:] - cumulative[:-window]
    # Number of real frames in each window (shorter at the borders).
    index = np.arange(len(values))
    counts = np.minimum(index + half, len(values) - 1) - np.maximum(index - half, 0) + 1
    counts = counts.reshape((-1,) + (1,) * (values.ndim - 1))
    return sums[: len(values)] * 2 > counts


def angle_scores(samples: pd.DataFrame, display: Display) -> np.ndarray:
    """Angle (radians) between gaze and display, ``pi`` when no face/gaze is available."""
    origins, directions = origins_and_directions(samples)
    angles = np.full(len(samples), np.pi)
    valid = np.isfinite(origins).all(axis=1) & np.isfinite(directions).all(axis=1)
    if valid.any():
        angles[valid] = display.angular_distance(origins[valid], directions[valid])
    return angles


# --------------------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------------------


@dataclass
class GridResult:
    sizes: np.ndarray
    offsets: np.ndarray
    per_video: dict[str, Confusion]
    """Confusion counts with shape ``(S, P)`` for each video."""

    @property
    def pooled(self) -> Confusion:
        total = None
        for confusion in self.per_video.values():
            total = confusion if total is None else total + confusion
        return total

    def mean_metric(self, name: str = "f1") -> np.ndarray:
        """Metric averaged over videos (the thesis averaged the F1 score of all videos)."""
        stacked = np.stack([metrics.METRICS[name](c) for c in self.per_video.values()])
        with np.errstate(all="ignore"):
            return np.nanmean(stacked, axis=0) if len(stacked) else np.full(self.shape, np.nan)

    @property
    def shape(self):
        return len(self.sizes), len(self.offsets)

    def best(self, grid: np.ndarray) -> tuple[int, int]:
        if np.all(np.isnan(grid)):
            return 0, 0
        return np.unravel_index(np.nanargmax(grid), grid.shape)


@dataclass
class Fold:
    video: str
    params: tuple[float, ...]
    confusion: Confusion


@dataclass
class CrossValidation:
    folds: list[Fold]

    @property
    def pooled(self) -> Confusion:
        total = Confusion(0, 0, 0, 0)
        for fold in self.folds:
            total = total + fold.confusion
        return total

    def per_fold(self, name: str) -> np.ndarray:
        return np.array([float(metrics.METRICS[name](f.confusion)) for f in self.folds])


@dataclass
class CombinationResult:
    detector: str
    model: str
    samples: pd.DataFrame
    grid: GridResult
    grid_smoothed: GridResult | None
    cv_grid: CrossValidation
    cv_grid_smoothed: CrossValidation | None
    cv_angle: CrossValidation
    angles: np.ndarray
    roc: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.detector} + {self.model}"

    def best_thesis_params(self) -> tuple[float, float, float]:
        """Display size and offset with the best F1 averaged over videos (thesis method)."""
        mean_f1 = self.grid.mean_metric("f1")
        i, j = self.grid.best(mean_f1)
        return float(self.grid.sizes[i]), float(self.grid.offsets[j]), float(mean_f1[i, j])


def leave_one_video_out(
    per_video: dict[str, Confusion], params: list[tuple[float, ...]], optimize: str
) -> CrossValidation:
    """Pick parameters on all other videos, evaluate them on the held-out video.

    ``per_video`` holds confusion counts for every parameter candidate (any shape), ``params`` maps the
    flattened candidate index to its parameter values.
    """
    folds = []
    videos = list(per_video)
    if len(videos) < 2:
        return CrossValidation([])
    for held_out in videos:
        train = None
        for video in videos:
            if video != held_out:
                train = per_video[video] if train is None else train + per_video[video]
        score = np.asarray(metrics.METRICS[optimize](train), dtype=np.float64).ravel()
        best = int(np.nanargmax(score)) if not np.all(np.isnan(score)) else 0
        test = per_video[held_out]
        folds.append(
            Fold(
                held_out,
                params[best],
                Confusion(
                    *(int(np.ravel(getattr(test, k))[best]) for k in ("tp", "fp", "tn", "fn"))
                ),
            )
        )
    return CrossValidation(folds)


def _grid_result(samples: pd.DataFrame, config: EvalConfig, smoothing: int) -> GridResult:
    sizes, offsets = config.size.values, config.offset.values
    origins, directions = origins_and_directions(samples)
    points = intersect_camera_plane(origins, directions)
    per_video = {}
    for video, index in samples.groupby("video", sort=False).indices.items():
        predictions = grid_predictions(points[index], sizes, offsets)
        predictions = majority_filter(predictions, smoothing)
        per_video[video] = Confusion.from_predictions(samples["y"].to_numpy()[index], predictions)
    return GridResult(sizes, offsets, per_video)


def evaluate_combination(
    dataset: Dataset, detector: str, model: str, config: EvalConfig
) -> CombinationResult:
    samples = dataset.samples(detector, model, config)
    grid = _grid_result(samples, config, smoothing=1)
    grid_params = [(float(s), float(p)) for s in grid.sizes for p in grid.offsets]
    cv_grid = leave_one_video_out(grid.per_video, grid_params, config.optimize)

    grid_smoothed = cv_grid_smoothed = None
    if config.smoothing_window > 1:
        grid_smoothed = _grid_result(samples, config, smoothing=config.smoothing_window)
        cv_grid_smoothed = leave_one_video_out(
            grid_smoothed.per_video, grid_params, config.optimize
        )

    display = Display(config.display.width, config.display.height, config.display.offset)
    angles = angle_scores(samples, display)
    samples = samples.assign(angle=angles)

    per_video_angle = {
        video: Confusion.from_predictions(
            samples["y"].to_numpy()[index], angles[index][:, None] <= ANGLE_THRESHOLDS[None, :]
        )
        for video, index in samples.groupby("video", sort=False).indices.items()
    }
    cv_angle = leave_one_video_out(
        per_video_angle, [(float(t),) for t in ANGLE_THRESHOLDS], config.optimize
    )

    roc = {}
    y = samples["y"].to_numpy()
    if 0 < y.sum() < len(y):
        score = -angles
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        roc = {
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "auc": float(roc_auc_score(y, score)),
            "ap": float(average_precision_score(y, score)),
            "prevalence": float(y.mean()),
        }

    return CombinationResult(
        detector,
        model,
        samples,
        grid,
        grid_smoothed,
        cv_grid,
        cv_grid_smoothed,
        cv_angle,
        angles,
        roc,
    )


# --------------------------------------------------------------------------------------------------
# face detection and blinks
# --------------------------------------------------------------------------------------------------


def face_statistics(dataset: Dataset) -> pd.DataFrame:
    """Detection rate, quality distribution, distance and runtime per video and detector."""
    faces = dataset.faces
    rows = []
    groups = [(video, detector, g) for (video, detector), g in faces.groupby(["video", "detector"])]
    groups += [("total", detector, g) for detector, g in faces.groupby("detector")]
    for video, detector, group in groups:
        detected = group[group["detected"]]
        quality = detected["label_quality"]
        rows.append(
            {
                "video": video,
                "detector": detector,
                "frames": len(group),
                "detected": len(detected),
                "detection_rate": len(detected) / len(group) if len(group) else np.nan,
                "quality_good": int((quality == 0).sum()),
                "quality_inaccurate": int((quality == 1).sum()),
                "quality_wrong": int((quality == 2).sum()),
                "quality_unrated": int(quality.isna().sum()),
                "mean_distance": detected["distance"].mean(),
                "mean_runtime_ms": group["runtime_ms"].mean(),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class BlinkResult:
    detector: str
    thresholds: np.ndarray
    f1_total: np.ndarray
    f1_per_video: dict[str, np.ndarray]

    @property
    def best(self) -> tuple[float, float]:
        if np.all(np.isnan(self.f1_total)):
            return float("nan"), float("nan")
        i = int(np.nanargmax(self.f1_total))
        return float(self.thresholds[i]), float(self.f1_total[i])


def evaluate_blinks(dataset: Dataset, detector: str, config: EvalConfig) -> BlinkResult | None:
    """F1 of 'EAR below threshold' as blink detector, for every threshold."""
    frames = dataset.frames[dataset.frames["label_blink"].notna()]
    faces = dataset.faces[(dataset.faces["detector"] == detector) & dataset.faces["detected"]]
    data = frames.merge(faces[["frame_id", "ear"]], on="frame_id")
    if data.empty or not data["label_blink"].any():
        return None
    thresholds = config.ear.values
    y = data["label_blink"].astype(bool).to_numpy()
    predictions = data["ear"].to_numpy()[:, None] < thresholds[None, :]
    per_video = {
        video: metrics.f1(Confusion.from_predictions(y[index], predictions[index]))
        for video, index in data.groupby("video", sort=False).indices.items()
    }
    return BlinkResult(
        detector, thresholds, metrics.f1(Confusion.from_predictions(y, predictions)), per_video
    )


def errors_by_quality(result: CombinationResult) -> dict[str, int]:
    """False negatives/positives at the thesis parameters, split by the landmark quality label."""
    size, offset, _ = result.best_thesis_params()
    samples = result.samples
    origins, directions = origins_and_directions(samples)
    points = intersect_camera_plane(origins, directions)
    predictions = grid_predictions(points, np.array([size]), np.array([offset]))[:, 0, 0]
    errors = predictions != samples["y"].to_numpy()
    quality = samples["label_quality"]
    counts = {"fp": int((errors & predictions).sum()), "fn": int((errors & ~predictions).sum())}
    for name, mask in {
        "no_face": ~samples["detected"].to_numpy(),
        "good": (quality == 0).fillna(False).to_numpy(),
        "inaccurate": (quality == 1).fillna(False).to_numpy(),
        "wrong": (quality == 2).fillna(False).to_numpy(),
        "unrated": (samples["detected"] & quality.isna()).to_numpy(),
    }.items():
        counts[f"errors_{name}"] = int((errors & mask).sum())
    return counts
