import numpy as np
import pytest

from gaze_estimator.cli import evaluate
from gaze_estimator.db.session import create_db_engine
from gaze_estimator.evaluation import analysis, metrics
from gaze_estimator.evaluation.config import EvalConfig, Range
from gaze_estimator.evaluation.dataset import Dataset
from gaze_estimator.evaluation.metrics import Confusion

from .conftest import TRUE_DISPLAY


def test_metrics_match_hand_computed_values():
    c = Confusion(tp=8, fp=2, tn=85, fn=5)
    assert float(metrics.f1(c)) == pytest.approx(16 / 23)
    expected_mcc = (8 * 85 - 2 * 5) / np.sqrt(10 * 13 * 87 * 90)
    assert float(metrics.mcc(c)) == pytest.approx(expected_mcc)
    assert float(metrics.mcc(Confusion(0, 0, 10, 0))) == 0.0
    assert np.isnan(metrics.f1(Confusion(0, 0, 10, 0)))


def test_confusion_on_grids():
    labels = np.array([True, False, True])
    predictions = np.array([[True, False], [True, False], [False, True]])
    c = Confusion.from_predictions(labels, predictions)
    assert c.tp.tolist() == [1, 1] and c.fp.tolist() == [1, 0] and c.fn.tolist() == [1, 1]


def test_majority_filter_removes_single_frame_flicker():
    raw = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    smoothed = analysis.majority_filter(raw, 3)
    assert smoothed.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


@pytest.fixture
def config():
    # A finer grid than the thesis default keeps the test fast but precise enough.
    return EvalConfig(size=Range(0.1, 0.3, 21), offset=Range(0.0, 0.1, 11), smoothing_window=3)


def test_grid_search_finds_the_true_display(synthetic_db, config):
    dataset = Dataset.load(create_db_engine(synthetic_db, create=False), config)
    assert dataset.combinations == [
        ("dlib", "bad-model"),
        ("dlib", "good-model"),
        ("mediapipe", "bad-model"),
        ("mediapipe", "good-model"),
    ]
    result = analysis.evaluate_combination(dataset, "mediapipe", "good-model", config)
    size, offset, f1 = result.best_thesis_params()
    # F1 slightly prefers a larger display (gaze noise pushes hits over the edge), hence the tolerance.
    assert size == pytest.approx(TRUE_DISPLAY.width, abs=0.06)
    assert offset == pytest.approx(TRUE_DISPLAY.offset, abs=0.03)
    assert f1 > 0.85

    # Cross-validation: one fold per video, and it should generalize on clean synthetic data.
    assert len(result.cv_grid.folds) == 3
    assert float(metrics.f1(result.cv_grid.pooled)) > 0.8
    assert result.roc["auc"] > 0.9


def test_better_model_ranks_higher(synthetic_db, config):
    dataset = Dataset.load(create_db_engine(synthetic_db, create=False), config)
    good = analysis.evaluate_combination(dataset, "mediapipe", "good-model", config)
    bad = analysis.evaluate_combination(dataset, "mediapipe", "bad-model", config)
    assert good.roc["auc"] > bad.roc["auc"]
    assert metrics.mcc(good.cv_grid.pooled) > metrics.mcc(bad.cv_grid.pooled)


def test_excluded_videos_and_distance_filter(synthetic_db, config):
    config.exclude_videos = {"video1": "test"}
    config.max_distance = 0.5
    dataset = Dataset.load(create_db_engine(synthetic_db, create=False), config)
    assert "video1" not in dataset.videos
    samples = dataset.samples("mediapipe", "good-model", config)
    assert (samples["distance"].dropna() <= 0.5).all()


def test_blink_evaluation(synthetic_db, config):
    dataset = Dataset.load(create_db_engine(synthetic_db, create=False), config)
    result = analysis.evaluate_blinks(dataset, "mediapipe", config)
    threshold, f1 = result.best
    assert 0.1 < threshold <= 0.3 and f1 == pytest.approx(1.0)


def test_cli_writes_graphs_and_tables(synthetic_db, tmp_path):
    out = tmp_path / "plots"
    config_file = tmp_path / "eval.yaml"
    config_file.write_text(
        "size: {min: 0.1, max: 0.3, steps: 10}\noffset: {min: 0.0, max: 0.1, steps: 5}\nformats: [png]\ndpi: 50\n"
    )
    evaluate.main(["--db", str(synthetic_db), "--config", str(config_file), "--out", str(out)])

    expected = [
        "tables/summary.csv",
        "tables/summary.tex",
        "tables/face_detection.csv",
        "tables/blinks.csv",
        "grid/total_mediapipe_good-model.png",
        "grid/video0/mediapipe_good-model.png",
        "grid/f1_vs_size_total.png",
        "timelines/video0/mediapipe_good-model.png",
        "timelines/video0/mediapipe_overview.png",
        "faces/quality_total.png",
        "faces/video0/dlib_sharpness.png",
        "faces/distance_total.png",
        "blinks/f1_vs_ear_total.png",
        "angle/roc_pr_mediapipe.png",
        "angle/distribution_dlib.png",
        "cross_validation/f1_per_video.png",
        "cross_validation/confusion_matrices.png",
        "cross_validation/smoothing.png",
        "summary/model_comparison.png",
    ]
    missing = [name for name in expected if not (out / name).is_file()]
    assert not missing, missing
