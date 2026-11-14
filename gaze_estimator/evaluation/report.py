"""Tables of the evaluation (CSV for further processing, LaTeX for the thesis)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .analysis import BlinkResult, CombinationResult, errors_by_quality


def summary_table(results: list[CombinationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        size, offset, thesis_f1 = r.best_thesis_params()
        row = {
            "detector": r.detector,
            "model": r.model,
            "frames": len(r.samples),
            "positives": int(r.samples["y"].sum()),
            "thesis_size_cm": size * 100,
            "thesis_offset_cm": offset * 100,
            "thesis_f1": thesis_f1,
        }
        if r.cv_grid.folds:
            pooled = r.cv_grid.pooled
            f1_folds = r.cv_grid.per_fold("f1")
            row |= {
                "cv_f1": float(metrics.f1(pooled)),
                "cv_f1_fold_mean": float(np.nanmean(f1_folds))
                if np.isfinite(f1_folds).any()
                else np.nan,
                "cv_f1_fold_std": float(np.nanstd(f1_folds))
                if np.isfinite(f1_folds).any()
                else np.nan,
                "cv_mcc": float(metrics.mcc(pooled)),
                "cv_balanced_accuracy": float(metrics.balanced_accuracy(pooled)),
                "cv_precision": float(metrics.precision(pooled)),
                "cv_recall": float(metrics.recall(pooled)),
            }
        if r.cv_grid_smoothed and r.cv_grid_smoothed.folds:
            row |= {
                "cv_smoothed_f1": float(metrics.f1(r.cv_grid_smoothed.pooled)),
                "cv_smoothed_mcc": float(metrics.mcc(r.cv_grid_smoothed.pooled)),
            }
        if r.cv_angle.folds:
            thresholds = [np.degrees(f.params[0]) for f in r.cv_angle.folds]
            row |= {
                "angle_cv_f1": float(metrics.f1(r.cv_angle.pooled)),
                "angle_cv_mcc": float(metrics.mcc(r.cv_angle.pooled)),
                "angle_threshold_deg_median": float(np.median(thresholds)),
            }
        if r.roc:
            row |= {"roc_auc": r.roc["auc"], "average_precision": r.roc["ap"]}
        row |= errors_by_quality(r)
        rows.append(row)
    return pd.DataFrame(rows)


def folds_table(results: list[CombinationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for method, cv in (("display", r.cv_grid), ("angle", r.cv_angle)):
            for fold in cv.folds:
                rows.append(
                    {
                        "detector": r.detector,
                        "model": r.model,
                        "method": method,
                        "held_out_video": fold.video,
                        "params": ", ".join(f"{p:.4f}" for p in fold.params),
                        **metrics.summary(fold.confusion),
                    }
                )
    return pd.DataFrame(rows)


def blink_table(results: list[BlinkResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"detector": r.detector, "best_ear_threshold": r.best[0], "best_f1": r.best[1]}
            for r in results
        ]
    )


def write(table: pd.DataFrame, out: Path, name: str, caption: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{name}.csv"
    tex = out / f"{name}.tex"
    table.to_csv(csv, index=False)
    numeric = table.select_dtypes("number").columns
    formatters = {
        c: (lambda v: "–" if pd.isna(v) else f"{v:.3f}")
        for c in numeric
        if table[c].dtype.kind == "f"
    }
    latex = table.to_latex(
        index=False,
        formatters=formatters,
        caption=caption,
        label=f"tab:{name}",
        escape=True,
        longtable=False,
    )
    tex.write_text(latex, encoding="utf-8")
    return [csv, tex]
