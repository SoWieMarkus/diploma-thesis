"""Load the database into pandas DataFrames for the evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from .config import EvalConfig


def _query(engine: Engine, sql: str) -> pd.DataFrame:
    with engine.connect() as connection:
        result = connection.execute(text(sql))
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def _vectors(column: pd.Series, prefix: str) -> pd.DataFrame:
    values = column.map(lambda v: json.loads(v) if isinstance(v, str) else v)
    array = np.array([v if v is not None else [np.nan] * 3 for v in values], dtype=np.float64)
    array = array.reshape(len(values), 3)
    return pd.DataFrame(
        array, columns=[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"], index=column.index
    )


@dataclass
class Dataset:
    frames: pd.DataFrame
    """One row per processed frame (with labels)."""
    faces: pd.DataFrame
    """One row per frame and face detector."""
    gazes: pd.DataFrame
    """One row per detected face and gaze model."""

    @property
    def face_detectors(self) -> list[str]:
        return sorted(self.faces["detector"].unique())

    @property
    def gaze_models(self) -> list[str]:
        return sorted(self.gazes["model"].unique())

    @property
    def videos(self) -> list[str]:
        return list(dict.fromkeys(self.frames.sort_values("video")["video"]))

    @property
    def combinations(self) -> list[tuple[str, str]]:
        pairs = self.gazes[["detector", "model"]].drop_duplicates()
        return sorted(map(tuple, pairs.to_numpy()))

    @classmethod
    def load(cls, engine: Engine, config: EvalConfig) -> Dataset:
        frames = _query(
            engine,
            'SELECT f.id AS frame_id, f.video_id, v.name AS video, f."index" AS frame_index, '
            "f.timestamp_ms, f.brightness, f.contrast, f.sharpness, f.label_looking, f.label_blink "
            "FROM frame f JOIN video v ON v.id = f.video_id",
        )
        faces = _query(
            engine,
            "SELECT id AS face_id, frame_id, detector, detected, distance, ear_left, ear_right, "
            "center, label_quality, runtime_ms FROM face",
        )
        gazes = _query(
            engine,
            "SELECT g.frame_id, f.detector, g.model, g.vector, g.runtime_ms "
            "FROM gaze g JOIN face f ON f.id = g.face_id",
        )

        frames = frames[~frames["video"].isin(config.exclude_videos)].copy()
        frames["video"] = frames["video"].astype(str)
        frames["label_blink"] = frames["label_blink"].astype("boolean")
        frames["label_looking"] = frames["label_looking"].astype("Int64")

        faces = faces[faces["frame_id"].isin(frames["frame_id"])].copy()
        faces["detected"] = faces["detected"].astype(bool)
        faces["ear"] = (faces["ear_left"] + faces["ear_right"]) / 2
        faces["label_quality"] = faces["label_quality"].astype("Int64")
        faces = pd.concat(
            [faces.drop(columns="center"), _vectors(faces["center"], "center")], axis=1
        )
        faces = faces.merge(frames[["frame_id", "video", "frame_index"]], on="frame_id")

        gazes = gazes[gazes["frame_id"].isin(frames["frame_id"])].copy()
        gazes = pd.concat([gazes.drop(columns="vector"), _vectors(gazes["vector"], "gaze")], axis=1)
        return cls(
            frames.reset_index(drop=True),
            faces.reset_index(drop=True),
            gazes.reset_index(drop=True),
        )

    def ground_truth(self, config: EvalConfig) -> pd.DataFrame:
        """Frames with a usable 'looking' label (unsure is excluded) and the binary target ``y``."""
        labeled = self.frames[self.frames["label_looking"].isin([0, 1])].copy()
        looking = labeled["label_looking"] == 1
        if config.blink_as_not_looking:
            looking &= ~labeled["label_blink"].fillna(False).astype(bool)
        labeled["y"] = looking.astype(bool)
        return labeled

    def samples(self, detector: str, model: str, config: EvalConfig) -> pd.DataFrame:
        """One row per labeled frame with face center and gaze vector (``nan`` if no face).

        Frames with a face further away than ``max_distance`` are dropped, as in the thesis.
        """
        truth = self.ground_truth(config)
        faces = self.faces[self.faces["detector"] == detector]
        gazes = self.gazes[(self.gazes["detector"] == detector) & (self.gazes["model"] == model)]
        data = truth.merge(
            faces[
                [
                    "frame_id",
                    "detected",
                    "distance",
                    "label_quality",
                    "center_x",
                    "center_y",
                    "center_z",
                ]
            ],
            on="frame_id",
            how="left",
        ).merge(gazes[["frame_id", "gaze_x", "gaze_y", "gaze_z"]], on="frame_id", how="left")
        data["detected"] = data["detected"].fillna(False).astype(bool)
        data = data[~(data["distance"] > config.max_distance)]
        return data.sort_values(["video", "frame_index"]).reset_index(drop=True)


def origins_and_directions(samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    origins = samples[["center_x", "center_y", "center_z"]].to_numpy(dtype=np.float64)
    directions = samples[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=np.float64)
    return origins, directions
