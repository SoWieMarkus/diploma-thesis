"""Evaluate how well each face detector + gaze model detects 'looking at the smartphone'.

1. Thesis method: grid search over a square virtual display (size l, camera offset p), F1 heatmaps,
   timelines, F1 vs. display size, face detection quality and blink detection graphs.
2. Additions: leave-one-video-out cross-validation of the display parameters (no evaluation on the
   data the parameters were tuned on), a continuous angle-to-display score with ROC/PR curves,
   MCC/balanced accuracy, confusion matrices and temporal smoothing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.paths import DEFAULT_DATABASE, PLOTS_DIR
from ..db.session import create_db_engine
from ..evaluation import analysis, metrics, report
from ..evaluation.config import EvalConfig
from ..evaluation.dataset import Dataset
from ..evaluation.plots import Plotter
from .common import setup_logging

logger = logging.getLogger("gaze-evaluate")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE, help="SQLite database file.")
    parser.add_argument(
        "--config", type=Path, help="Evaluation config YAML (see eval.example.yaml)."
    )
    parser.add_argument(
        "--out", type=Path, default=PLOTS_DIR, help="Output folder for graphs and tables."
    )
    parser.add_argument(
        "--no-per-video", action="store_true", help="Skip the graphs of every single video."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def run(database: Path, config: EvalConfig, out: Path) -> Plotter:
    engine = create_db_engine(database, create=False)
    dataset = Dataset.load(engine, config)
    plotter = Plotter(out, config.formats, config.dpi)
    tables = out / "tables"

    if config.exclude_videos:
        for video, reason in config.exclude_videos.items():
            logger.info("Excluding video %s: %s", video, reason)

    logger.info(
        "%d videos, %d frames, face detectors: %s, gaze models: %s",
        len(dataset.videos),
        len(dataset.frames),
        ", ".join(dataset.face_detectors),
        ", ".join(dataset.gaze_models),
    )

    # ---- face detection -------------------------------------------------------------------------
    face_stats = analysis.face_statistics(dataset)
    report.write(face_stats, tables, "face_detection", "Face detection rate and landmark quality")
    plotter.quality_bars(
        face_stats[face_stats["video"] == "total"],
        "faces/quality_total",
        "Landmark quality (all videos)",
    )
    plotter.distance_histogram(dataset, "faces/distance_total", config.max_distance)
    if config.per_video_plots:
        for video in dataset.videos:
            plotter.quality_bars(
                face_stats[face_stats["video"] == video],
                f"faces/{video}/quality",
                f"Landmark quality | Video {video}",
            )
            for detector in dataset.face_detectors:
                for column, label in (
                    ("distance", "Distance [m]"),
                    ("brightness", "Brightness"),
                    ("contrast", "Contrast"),
                    ("sharpness", "Sharpness"),
                ):
                    plotter.frame_scatter(
                        dataset,
                        video,
                        detector,
                        column,
                        label,
                        f"faces/{video}/{detector}_{column}",
                    )

    # ---- blinks ---------------------------------------------------------------------------------
    blinks = [
        b for d in dataset.face_detectors if (b := analysis.evaluate_blinks(dataset, d, config))
    ]
    if blinks:
        report.write(
            report.blink_table(blinks),
            tables,
            "blinks",
            "Blink detection with the eye aspect ratio",
        )
        plotter.blink_f1(blinks, "blinks/f1_vs_ear_total")
        if config.per_video_plots:
            for video in dataset.videos:
                plotter.blink_f1(blinks, f"blinks/{video}_f1_vs_ear", video)
    else:
        logger.warning("No blink labels yet, skipping the blink evaluation.")

    # ---- gaze -----------------------------------------------------------------------------------
    truth = dataset.ground_truth(config)
    if truth.empty:
        logger.warning("No 'looking' labels yet: label frames in the web app (gaze-serve) first.")
        return plotter
    if truth["y"].all() or not truth["y"].any():
        logger.warning("All labeled frames have the same class, metrics will not be meaningful.")
    if truth["video"].nunique() < 2:
        logger.warning("Cross-validation needs at least two labeled videos, it is skipped.")

    results = []
    for detector, model in dataset.combinations:
        logger.info("Evaluating %s + %s", detector, model)
        results.append(analysis.evaluate_combination(dataset, detector, model, config))

    summary = report.summary_table(results)
    report.write(
        summary,
        tables,
        "summary",
        "Detection of looking at the display per face detector and gaze model",
    )
    report.write(
        report.folds_table(results), tables, "cross_validation_folds", "Leave-one-video-out folds"
    )

    for result in results:
        key = f"{result.detector}_{result.model}"
        plotter.heatmap(
            result.grid.mean_metric("f1"),
            result.grid.sizes,
            result.grid.offsets,
            f"Mean F1 of all videos | {result.name}",
            f"grid/total_{key}",
        )
        if config.per_video_plots:
            size, offset, _ = result.best_thesis_params()
            for video, confusion in result.grid.per_video.items():
                plotter.heatmap(
                    metrics.f1(confusion),
                    result.grid.sizes,
                    result.grid.offsets,
                    f"Video {video} | {result.name}",
                    f"grid/{video}/{key}",
                )
                plotter.timeline(result, video, (size, offset), f"timelines/{video}/{key}")

    plotter.f1_vs_size(results, "grid/f1_vs_size_total")
    if config.per_video_plots:
        for video in truth["video"].unique():
            plotter.f1_vs_size(results, f"grid/{video}/f1_vs_size", video)
            for detector in dataset.face_detectors:
                plotter.timeline_overview(
                    [r for r in results if r.detector == detector],
                    video,
                    f"timelines/{video}/{detector}_overview",
                )

    for detector in dataset.face_detectors:
        plotter.roc_pr(results, detector, f"angle/roc_pr_{detector}")
        plotter.angle_distribution(results, detector, f"angle/distribution_{detector}")
    plotter.cv_boxplot(results, "f1", "cross_validation/f1_per_video")
    plotter.cv_boxplot(results, "mcc", "cross_validation/mcc_per_video")
    plotter.confusion_matrices(results, "cross_validation/confusion_matrices")
    plotter.smoothing_effect(results, "cross_validation/smoothing", config.smoothing_window)
    plotter.summary_bars(summary, "summary/model_comparison")

    with pd.option_context("display.width", 200, "display.max_columns", 12):
        columns = [
            c
            for c in ("detector", "model", "thesis_f1", "cv_f1", "cv_mcc", "angle_cv_f1", "roc_auc")
            if c in summary
        ]
        print(summary[columns].round(3).to_string(index=False))
    return plotter


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)
    config = EvalConfig.load(args.config)
    if args.no_per_video:
        config.per_video_plots = False
    np.seterr(all="ignore")
    plotter = run(args.db, config, args.out)
    logger.info("Wrote %d files to %s", len(plotter.written), args.out.resolve())


if __name__ == "__main__":
    main()
