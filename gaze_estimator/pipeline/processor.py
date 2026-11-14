"""Batch processing of videos: every n-th frame, every face detector, every gaze model -> SQLite."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import Engine
from sqlmodel import delete, select
from tqdm import tqdm

from ..core.camera import Camera, find_profile_for_video
from ..db import convert, models
from ..db.session import open_session
from ..faces.base import FaceDetector
from ..faces.registry import create_face_detector
from ..gaze.base import GazeEstimator
from ..gaze.registry import create_gaze_estimator

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def find_videos(directory: Path) -> list[Path]:
    return sorted(
        p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def frame_statistics(frame_bgr: np.ndarray) -> tuple[float, float, float]:
    """Brightness, contrast and sharpness of a frame (used to explain detection failures)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return (
        float(gray.mean()),
        float(gray.std()),
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def iterate_frames(capture: cv2.VideoCapture, step: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, frame)`` for every ``step``-th frame. Skipped frames are only grabbed, not decoded."""
    index = 0
    while True:
        if index % step == 0:
            ok, frame = capture.read()
            if not ok:
                return
            yield index, frame
        elif not capture.grab():
            return
        index += 1


@dataclass
class ProcessOptions:
    step: int = 5
    jpeg_quality: int = 85
    force: bool = False
    """Recompute the selected face detectors and all gazes based on them (frame labels are kept)."""
    camera_profile: Path | None = None


class Processor:
    def __init__(
        self,
        engine: Engine,
        database_path: Path,
        output_dir: Path,
        face_detectors: list[str],
        gaze_models: list[str],
        options: ProcessOptions,
    ):
        self.engine = engine
        self.data_root = Path(database_path).resolve().parent
        self.output_dir = Path(output_dir).resolve()
        self.face_names = face_detectors
        self.gaze_names = gaze_models
        self.options = options
        self._faces: dict[str, FaceDetector] = {}
        self._gazes: dict[str, GazeEstimator] = {}

    # Models are loaded on first use, so a fully processed video does not load anything.
    def face_detector(self, name: str) -> FaceDetector:
        if name not in self._faces:
            logger.info("Loading face detector '%s'", name)
            self._faces[name] = create_face_detector(name)
        return self._faces[name]

    def gaze_estimator(self, name: str) -> GazeEstimator:
        if name not in self._gazes:
            logger.info("Loading gaze model '%s'", name)
            self._gazes[name] = create_gaze_estimator(name)
        return self._gazes[name]

    def close(self) -> None:
        for detector in self._faces.values():
            detector.close()

    def process_video(self, path: Path) -> dict[str, int]:
        path = Path(path).resolve()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video '{path}'")

        stats = {"frames": 0, "faces": 0, "gazes": 0}
        try:
            with open_session(self.engine) as session:
                video = self._get_or_create_video(session, path, capture)
                camera = Camera.from_dict(video.camera)
                frame_dir = self.output_dir / video.name
                frame_dir.mkdir(parents=True, exist_ok=True)

                if self.options.force:
                    self._delete_results(session, video.id)

                frames = {
                    f.index: f
                    for f in session.exec(
                        select(models.Frame).where(models.Frame.video_id == video.id)
                    )
                }
                faces = {
                    (f.frame_id, f.detector): f
                    for f in session.exec(
                        select(models.Face).where(models.Face.video_id == video.id)
                    )
                }
                gazes = {
                    (g.face_id, g.model)
                    for g in session.exec(
                        select(models.Gaze).where(models.Gaze.video_id == video.id)
                    )
                }

                total = video.frame_count // self.options.step + 1 if video.frame_count else None
                progress = tqdm(
                    iterate_frames(capture, self.options.step),
                    total=total,
                    desc=video.name,
                    unit="frame",
                )
                for index, image in progress:
                    frame = frames.get(index)
                    if frame is None:
                        frame = self._create_frame(session, video, index, image, frame_dir)
                        stats["frames"] += 1

                    for detector_name in self.face_names:
                        face = faces.get((frame.id, detector_name))
                        result = None
                        if face is None:
                            face, result = self._detect_face(
                                session, detector_name, frame, image, camera
                            )
                            stats["faces"] += 1
                        if not face.detected:
                            continue
                        for gaze_name in self.gaze_names:
                            if (face.id, gaze_name) in gazes:
                                continue
                            if result is None:
                                result = _face_from_row(face, detector_name)
                            self._estimate_gaze(session, gaze_name, face, result, image, camera)
                            gazes.add((face.id, gaze_name))
                            stats["gazes"] += 1
                    session.commit()
        finally:
            capture.release()
        return stats

    def _get_or_create_video(self, session, path: Path, capture: cv2.VideoCapture) -> models.Video:
        video = session.exec(select(models.Video).where(models.Video.path == str(path))).first()
        if video is not None:
            return video
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        profile = find_profile_for_video(path, self.options.camera_profile)
        camera = Camera.resolve(width, height, profile)
        if profile:
            logger.info("Using camera profile %s for %s", profile, path.name)
        video = models.Video(
            name=path.stem,
            path=str(path),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
            width=width,
            height=height,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            camera=camera.to_dict(),
        )
        session.add(video)
        session.commit()
        return video

    def _delete_results(self, session, video_id: int) -> None:
        logger.warning(
            "--force: deleting the results (and quality labels) of the face detectors %s",
            ", ".join(self.face_names),
        )
        face_ids = select(models.Face.id).where(
            models.Face.video_id == video_id, models.Face.detector.in_(self.face_names)
        )
        session.exec(delete(models.Gaze).where(models.Gaze.face_id.in_(face_ids)))
        session.exec(
            delete(models.Face).where(
                models.Face.video_id == video_id, models.Face.detector.in_(self.face_names)
            )
        )
        session.commit()

    def _create_frame(self, session, video: models.Video, index: int, image, frame_dir: Path):
        image_file = frame_dir / f"{index:06d}.jpg"
        cv2.imwrite(str(image_file), image, [cv2.IMWRITE_JPEG_QUALITY, self.options.jpeg_quality])
        brightness, contrast, sharpness = frame_statistics(image)
        frame = models.Frame(
            video_id=video.id,
            index=index,
            timestamp_ms=index / video.fps * 1000 if video.fps else 0.0,
            image_path=Path(os.path.relpath(image_file, self.data_root)).as_posix(),
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
        )
        session.add(frame)
        session.flush()
        return frame

    def _detect_face(self, session, name: str, frame: models.Frame, image, camera: Camera):
        start = time.perf_counter()
        face = self.face_detector(name).detect(image, camera)
        runtime = (time.perf_counter() - start) * 1000
        row = convert.face_row(
            face, frame_id=frame.id, video_id=frame.video_id, detector=name, runtime_ms=runtime
        )
        session.add(row)
        session.flush()
        return row, face

    def _estimate_gaze(
        self, session, name: str, face_row: models.Face, face, image, camera: Camera
    ):
        start = time.perf_counter()
        gaze = self.gaze_estimator(name).estimate(image, face, camera)
        runtime = (time.perf_counter() - start) * 1000
        session.add(convert.gaze_row(gaze, face=face_row, runtime_ms=runtime))


def _face_from_row(row: models.Face, detector: str):
    """Rebuild the in-memory face from a stored row (when only gaze models are missing)."""
    from ..core.types import Face

    return Face(
        detector=detector,
        landmarks_2d=np.array(row.landmarks_2d),
        landmarks_3d=np.array(row.landmarks_3d),
        bbox=tuple(row.bbox),
        rotation_vector=np.array(row.rotation_vector),
        translation_vector=np.array(row.translation_vector),
        rotation_matrix=cv2.Rodrigues(np.array(row.rotation_vector))[0],
        center=np.array(row.center),
        normalizing_rotation=np.array(row.normalizing_rotation),
        ear_left=row.ear_left,
        ear_right=row.ear_right,
    )
