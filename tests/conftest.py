"""Synthetic data: faces in front of a camera whose gaze hits a known display when 'looking'."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaze_estimator.core.geometry import Display, normalize
from gaze_estimator.db import models
from gaze_estimator.db.session import create_db_engine, open_session

TRUE_DISPLAY = Display.square(0.2, 0.05)


def _gaze_towards(center: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    target = np.array([target_xy[0], target_xy[1], 0.0])
    return normalize(target - center)


def build_synthetic_database(path: Path, videos: int = 3, frames: int = 80, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    engine = create_db_engine(path)
    min_x, max_x, min_y, max_y = TRUE_DISPLAY.bounds
    with open_session(engine) as session:
        for v in range(videos):
            video = models.Video(
                name=f"video{v}",
                path=f"/videos/video{v}.mp4",
                fps=30,
                width=640,
                height=480,
                frame_count=frames * 5,
                camera={"width": 640, "height": 480, "fx": 640, "fy": 640, "cx": 320, "cy": 240},
            )
            session.add(video)
            session.flush()
            looking_run = False
            for i in range(frames):
                # Glances last several frames.
                if rng.random() < 0.15:
                    looking_run = not looking_run
                frame = models.Frame(
                    video_id=video.id,
                    index=i * 5,
                    timestamp_ms=i * 5 / 30 * 1000,
                    image_path="x.jpg",
                    brightness=float(rng.uniform(80, 160)),
                    contrast=float(rng.uniform(20, 60)),
                    sharpness=float(rng.uniform(50, 500)),
                    label_looking=int(looking_run),
                    label_blink=bool(rng.random() < 0.05),
                )
                session.add(frame)
                session.flush()
                center = np.array(
                    [rng.normal(0, 0.05), rng.normal(-0.1, 0.03), rng.uniform(0.35, 0.6)]
                )
                if looking_run:
                    target = rng.uniform([min_x, min_y], [max_x, max_y])
                else:
                    # Somewhere around the display, but not on it.
                    while True:
                        target = rng.uniform([-0.35, -0.3], [0.35, 0.45])
                        if not TRUE_DISPLAY.contains(np.array([*target, 0.0])):
                            break
                for detector, noise in (("mediapipe", 0.01), ("dlib", 0.03)):
                    detected = rng.random() > (0.02 if detector == "mediapipe" else 0.2)
                    face = models.Face(
                        frame_id=frame.id,
                        video_id=video.id,
                        detector=detector,
                        detected=detected,
                        runtime_ms=5.0,
                    )
                    if detected:
                        face.center = center.tolist()
                        face.distance = float(np.linalg.norm(center))
                        face.ear_left = face.ear_right = float(0.1 if frame.label_blink else 0.3)
                        face.label_quality = int(rng.integers(0, 3))
                    session.add(face)
                    session.flush()
                    if not detected:
                        continue
                    for model, scale in (("good-model", 1.0), ("bad-model", 6.0)):
                        noisy = target + rng.normal(0, noise * scale, size=2)
                        vector = _gaze_towards(center, noisy)
                        session.add(
                            models.Gaze(
                                face_id=face.id,
                                frame_id=frame.id,
                                video_id=video.id,
                                model=model,
                                pitch=0.0,
                                yaw=0.0,
                                vector=vector.tolist(),
                            )
                        )
        session.commit()
    engine.dispose()
    return path


@pytest.fixture
def synthetic_db(tmp_path: Path) -> Path:
    return build_synthetic_database(tmp_path / "gaze.db")
