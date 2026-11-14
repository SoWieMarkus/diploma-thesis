"""Graphs of the evaluation (matplotlib). Every function writes one figure into ``out``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..core.geometry import intersect_camera_plane  # noqa: E402
from . import metrics  # noqa: E402
from .analysis import BlinkResult, CombinationResult, grid_predictions  # noqa: E402
from .dataset import Dataset, origins_and_directions  # noqa: E402

GREEN, RED, YELLOW, GRAY, DARK = "#2e9e5b", "#d64545", "#e0a526", "#9aa5b1", "#3d4451"
QUALITY_COLORS = {0: GREEN, 1: YELLOW, 2: RED}
MODEL_COLORS = {
    "eth-xgaze": "#d6457f",
    "l2cs-gaze360": "#3b82f6",
    "mpiifacegaze": "#e0a526",
    "head-rotation": "#7c5cd6",
}

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


class Plotter:
    def __init__(self, out: Path, formats: list[str], dpi: int):
        self.out = Path(out)
        self.formats = formats
        self.dpi = dpi
        self.written: list[Path] = []

    def save(self, fig, name: str) -> None:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        for fmt in self.formats:
            target = path.with_suffix(f".{fmt}")
            fig.savefig(target, dpi=self.dpi, bbox_inches="tight")
            self.written.append(target)
        plt.close(fig)

    # ---------------------------------------------------------------------------------------------
    # display grid search (thesis)
    # ---------------------------------------------------------------------------------------------

    def heatmap(self, grid: np.ndarray, sizes, offsets, title: str, name: str, label="F1 score"):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        mesh = ax.pcolormesh(
            sizes * 100, offsets * 100, grid.T, cmap="coolwarm", vmin=0, vmax=1, shading="nearest"
        )
        fig.colorbar(mesh, ax=ax, label=label)
        if not np.all(np.isnan(grid)):
            i, j = np.unravel_index(np.nanargmax(grid), grid.shape)
            ax.annotate(
                f"l = {sizes[i] * 100:.1f} cm, p = {offsets[j] * 100:.1f} cm\n{label} = {grid[i, j]:.3f}",
                xy=(sizes[i] * 100, offsets[j] * 100),
                xytext=(0.97, 0.97),
                textcoords="axes fraction",
                ha="right",
                va="top",
                bbox={"boxstyle": "square,pad=0.3", "fc": "white", "ec": "black", "lw": 0.7},
                arrowprops={"arrowstyle": "->"},
            )
        ax.set_xlabel("Display size l [cm]")
        ax.set_ylabel("Camera offset p [cm]")
        ax.set_title(title)
        ax.grid(False)
        self.save(fig, name)

    def f1_vs_size(self, results: list[CombinationResult], name: str, video: str | None = None):
        """F1 over the display size at the best offset for each size (thesis 'f1-score-gaze')."""
        fig, axes = plt.subplots(
            1, len({r.detector for r in results}), figsize=(11, 4), squeeze=False
        )
        detectors = sorted({r.detector for r in results})
        for ax, detector in zip(axes[0], detectors, strict=True):
            for result in (r for r in results if r.detector == detector):
                if video is None:
                    grid = result.grid.mean_metric("f1")
                elif video in result.grid.per_video:
                    grid = metrics.f1(result.grid.per_video[video])
                else:
                    continue
                with np.errstate(all="ignore"):
                    best = np.nanmax(np.where(np.isnan(grid), -1, grid), axis=1)
                ax.plot(
                    result.grid.sizes * 100,
                    np.where(best < 0, np.nan, best),
                    label=result.model,
                    color=MODEL_COLORS.get(result.model),
                )
            ax.set_title(detector)
            ax.set_xlabel("Display size l [cm]")
            ax.set_ylabel("F1 score (best offset)")
            ax.set_ylim(0, 1)
            ax.legend()
        fig.suptitle(f"F1 score vs. display size | {video or 'all videos (mean)'}")
        self.save(fig, name)

    # ---------------------------------------------------------------------------------------------
    # timelines (thesis strip plots)
    # ---------------------------------------------------------------------------------------------

    @staticmethod
    def _timeline_rows(result: CombinationResult, video: str, params: tuple[float, float]):
        samples = result.samples[result.samples["video"] == video]
        origins, directions = origins_and_directions(samples)
        points = intersect_camera_plane(origins, directions)
        predictions = grid_predictions(points, np.array([params[0]]), np.array([params[1]]))[
            :, 0, 0
        ]
        truth = [GREEN if y else RED for y in samples["y"]]
        prediction = [
            DARK if not d else (GREEN if p else RED)
            for d, p in zip(samples["detected"], predictions, strict=True)
        ]
        quality = [
            DARK if not d else QUALITY_COLORS.get(q if pd.notna(q) else -1, GRAY)
            for d, q in zip(samples["detected"], samples["label_quality"], strict=True)
        ]
        return samples["frame_index"].to_numpy(), [truth, prediction, quality]

    @staticmethod
    def _draw_strips(ax, frames, rows, step):
        labels = ["Ground truth", "Prediction", "Landmark quality"]
        for r, colors in enumerate(rows):
            ax.broken_barh(
                [(f - step / 2, step) for f in frames], (r + 0.1, 0.8), facecolors=colors
            )
        ax.set_yticks([0.5, 1.5, 2.5], labels)
        ax.set_ylim(0, 3)
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_xlabel("Frame")

    def timeline(self, result: CombinationResult, video: str, params, name: str):
        frames, rows = self._timeline_rows(result, video, params)
        if len(frames) == 0:
            return
        step = np.median(np.diff(frames)) if len(frames) > 1 else 1
        fig, ax = plt.subplots(figsize=(12, 2.6))
        self._draw_strips(ax, frames, rows, step)
        ax.set_title(
            f"Video {video} | {result.name} | l = {params[0] * 100:.1f} cm, p = {params[1] * 100:.1f} cm"
        )
        ax.legend(
            handles=_legend(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.35),
            ncol=6,
            frameon=False,
        )
        self.save(fig, name)

    def timeline_overview(self, results: list[CombinationResult], video: str, name: str):
        results = [r for r in results if video in r.grid.per_video]
        if not results:
            return
        fig, axes = plt.subplots(
            len(results), 1, figsize=(12, 2.2 * len(results)), sharex=True, squeeze=False
        )
        for ax, result in zip(axes[:, 0], results, strict=True):
            size, offset, _ = result.best_thesis_params()
            frames, rows = self._timeline_rows(result, video, (size, offset))
            step = np.median(np.diff(frames)) if len(frames) > 1 else 1
            self._draw_strips(ax, frames, rows, step)
            ax.set_title(result.name, fontsize=10)
        axes[-1, 0].legend(
            handles=_legend(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.45),
            ncol=6,
            frameon=False,
        )
        fig.suptitle(f"Video {video}")
        self.save(fig, name)

    # ---------------------------------------------------------------------------------------------
    # faces
    # ---------------------------------------------------------------------------------------------

    def quality_bars(self, stats: pd.DataFrame, name: str, title: str):
        columns = [
            "quality_good",
            "quality_inaccurate",
            "quality_wrong",
            "quality_unrated",
            "not_detected",
        ]
        data = stats.assign(not_detected=stats["frames"] - stats["detected"]).set_index("detector")[
            columns
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(columns))
        width = 0.8 / max(len(data), 1)
        for k, (detector, row) in enumerate(data.iterrows()):
            ax.bar(x + (k - (len(data) - 1) / 2) * width, row.to_numpy(), width, label=detector)
        ax.set_xticks(x, ["good", "inaccurate", "wrong", "unrated", "no face"])
        ax.set_ylabel("Frames")
        ax.set_title(title)
        ax.legend()
        self.save(fig, name)

    def frame_scatter(
        self, dataset: Dataset, video: str, detector: str, column: str, label: str, name: str
    ):
        faces = dataset.faces[
            (dataset.faces["video"] == video) & (dataset.faces["detector"] == detector)
        ]
        data = dataset.frames[dataset.frames["video"] == video].merge(
            faces[["frame_id", "detected", "label_quality", "distance"]], on="frame_id"
        )
        if data.empty or data[column].isna().all():
            return
        colors = [
            DARK if not d else QUALITY_COLORS.get(q if pd.notna(q) else -1, GRAY)
            for d, q in zip(data["detected"], data["label_quality"], strict=True)
        ]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.scatter(data["frame_index"], data[column], c=colors, s=6)
        average = data[column].mean()
        ax.axhline(average, color=RED, lw=1)
        ax.annotate(
            f"mean: {average:.3f}",
            xy=(0.99, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox={"boxstyle": "square,pad=0.3", "fc": "white", "ec": "black", "lw": 0.7},
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel(label)
        ax.set_title(f"{label} | Video {video} | {detector}")
        ax.legend(
            handles=_quality_legend(),
            loc="upper left",
            fontsize=8,
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0, -0.2),
        )
        self.save(fig, name)

    def distance_histogram(self, dataset: Dataset, name: str, max_distance: float):
        detected = dataset.faces[dataset.faces["detected"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        for detector, group in detected.groupby("detector"):
            ax.hist(group["distance"] * 100, bins=40, alpha=0.6, label=detector)
        ax.axvline(max_distance * 100, color=RED, ls="--", label="max distance")
        ax.set_xlabel("Distance face - camera [cm]")
        ax.set_ylabel("Frames")
        ax.set_title("Estimated face distance")
        ax.legend()
        self.save(fig, name)

    # ---------------------------------------------------------------------------------------------
    # blinks
    # ---------------------------------------------------------------------------------------------

    def blink_f1(self, results: list[BlinkResult], name: str, video: str | None = None):
        fig, ax = plt.subplots(figsize=(7, 4))
        for result in results:
            values = result.f1_total if video is None else result.f1_per_video.get(video)
            if values is None:
                continue
            ax.plot(result.thresholds, values, label=result.detector)
        ax.set_xlabel("EAR threshold")
        ax.set_ylabel("F1 score (blink)")
        ax.set_ylim(0, 1)
        ax.set_title(f"Blink detection | {video or 'all videos'}")
        ax.legend()
        self.save(fig, name)

    # ---------------------------------------------------------------------------------------------
    # new: threshold free and cross-validated evaluation
    # ---------------------------------------------------------------------------------------------

    def roc_pr(self, results: list[CombinationResult], detector: str, name: str):
        results = [r for r in results if r.detector == detector and r.roc]
        if not results:
            return
        fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 4.6))
        for result in results:
            color = MODEL_COLORS.get(result.model)
            roc_ax.plot(
                result.roc["fpr"],
                result.roc["tpr"],
                color=color,
                label=f"{result.model} (AUC {result.roc['auc']:.3f})",
            )
            pr_ax.plot(
                result.roc["recall"],
                result.roc["precision"],
                color=color,
                label=f"{result.model} (AP {result.roc['ap']:.3f})",
            )
        roc_ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
        pr_ax.axhline(results[0].roc["prevalence"], color=GRAY, ls="--", lw=1, label="chance")
        roc_ax.set(
            xlabel="False positive rate", ylabel="True positive rate", title=f"ROC | {detector}"
        )
        pr_ax.set(
            xlabel="Recall",
            ylabel="Precision",
            title=f"Precision-recall | {detector}",
            ylim=(0, 1.02),
        )
        roc_ax.legend(loc="lower right")
        pr_ax.legend(loc="lower left")
        self.save(fig, name)

    def angle_distribution(self, results: list[CombinationResult], detector: str, name: str):
        results = [r for r in results if r.detector == detector]
        if not results:
            return
        fig, axes = plt.subplots(
            1, len(results), figsize=(3.6 * len(results), 3.6), sharey=True, squeeze=False
        )
        bins = np.linspace(0, 60, 41)
        for ax, result in zip(axes[0], results, strict=True):
            angles = np.degrees(result.angles)
            y = result.samples["y"].to_numpy()
            detected = result.samples["detected"].to_numpy()
            ax.hist(
                angles[y & detected],
                bins=bins,
                alpha=0.65,
                color=GREEN,
                label="looking",
                density=True,
            )
            ax.hist(
                angles[~y & detected],
                bins=bins,
                alpha=0.65,
                color=RED,
                label="not looking",
                density=True,
            )
            ax.set_title(result.model, fontsize=10)
            ax.set_xlabel("Angle to display [°]")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].legend()
        fig.suptitle(f"Angle between gaze and display | {detector}")
        self.save(fig, name)

    def cv_boxplot(self, results: list[CombinationResult], metric: str, name: str):
        data, labels = [], []
        for result in results:
            for cv, suffix in ((result.cv_grid, "display"), (result.cv_angle, "angle")):
                values = cv.per_fold(metric)
                if len(values):
                    data.append(values[~np.isnan(values)])
                    labels.append(f"{result.detector}\n{result.model}\n({suffix})")
        if not data:
            return
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(data)), 4.8))
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_ylabel(metric.upper() if metric == "mcc" else "F1 score")
        ax.set_title(f"Leave-one-video-out cross-validation ({metric.upper()} per held-out video)")
        ax.tick_params(axis="x", labelsize=7)
        self.save(fig, name)

    def confusion_matrices(self, results: list[CombinationResult], name: str):
        results = [r for r in results if r.cv_grid.folds]
        if not results:
            return
        detectors = sorted({r.detector for r in results})
        models = sorted({r.model for r in results})
        fig, axes = plt.subplots(
            len(detectors),
            len(models),
            figsize=(2.6 * len(models), 2.6 * len(detectors)),
            squeeze=False,
        )
        for ax in axes.ravel():
            ax.axis("off")
        for result in results:
            ax = axes[detectors.index(result.detector), models.index(result.model)]
            ax.axis("on")
            c = result.cv_grid.pooled
            matrix = np.array([[c.tn, c.fp], [c.fn, c.tp]])
            ax.imshow(matrix, cmap="Blues")
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() / 2 else "black",
                )
            ax.set_xticks([0, 1], ["not", "looking"])
            ax.set_yticks([0, 1], ["not", "looking"])
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Ground truth")
            ax.set_title(f"{result.detector} + {result.model}", fontsize=9)
            ax.grid(False)
        fig.suptitle("Confusion matrices (leave-one-video-out, display method)")
        self.save(fig, name)

    def summary_bars(self, summary: pd.DataFrame, name: str):
        if summary.empty:
            return
        columns = [
            c
            for c in (
                "thesis_f1",
                "cv_f1",
                "cv_mcc",
                "cv_balanced_accuracy",
                "angle_cv_f1",
                "roc_auc",
            )
            if c in summary
        ]
        labels = summary["detector"] + "\n" + summary["model"]
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(summary)), 4.8))
        x = np.arange(len(summary))
        width = 0.8 / len(columns)
        for k, column in enumerate(columns):
            ax.bar(x + (k - (len(columns) - 1) / 2) * width, summary[column], width, label=column)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(-0.1, 1)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title("Model comparison")
        ax.legend(fontsize=8, ncol=3)
        self.save(fig, name)

    def smoothing_effect(self, results: list[CombinationResult], name: str, window: int):
        results = [r for r in results if r.cv_grid_smoothed and r.cv_grid.folds]
        if not results:
            return
        labels = [f"{r.detector}\n{r.model}" for r in results]
        raw = [float(metrics.f1(r.cv_grid.pooled)) for r in results]
        smooth = [float(metrics.f1(r.cv_grid_smoothed.pooled)) for r in results]
        raw_mcc = [float(metrics.mcc(r.cv_grid.pooled)) for r in results]
        smooth_mcc = [float(metrics.mcc(r.cv_grid_smoothed.pooled)) for r in results]
        x = np.arange(len(results))
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(results)), 4.4))
        ax.bar(x - 0.3, raw, 0.2, label="F1", color="#93c5fd")
        ax.bar(x - 0.1, smooth, 0.2, label=f"F1 smoothed ({window})", color="#2563eb")
        ax.bar(x + 0.1, raw_mcc, 0.2, label="MCC", color="#fcd34d")
        ax.bar(x + 0.3, smooth_mcc, 0.2, label=f"MCC smoothed ({window})", color="#d97706")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(-0.1, 1)
        ax.set_title("Effect of temporal smoothing (leave-one-video-out)")
        ax.legend(fontsize=8)
        self.save(fig, name)


def _legend():
    return [
        Patch(color=GREEN, label="looking / good"),
        Patch(color=RED, label="not looking / wrong"),
        Patch(color=YELLOW, label="inaccurate"),
        Patch(color=GRAY, label="unrated"),
        Patch(color=DARK, label="no face"),
    ]


def _quality_legend():
    return [
        Patch(color=GREEN, label="good"),
        Patch(color=YELLOW, label="inaccurate"),
        Patch(color=RED, label="wrong"),
        Patch(color=GRAY, label="unrated"),
        Patch(color=DARK, label="no face"),
    ]
