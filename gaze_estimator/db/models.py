"""Database schema (SQLite via SQLModel).

``video`` 1-n ``frame`` 1-n ``face`` (one row per face detector, also when no face was found)
1-n ``gaze`` (one row per gaze model).

Labels live directly on the rows they describe and are ``NULL`` until somebody labeled them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, SQLModel


class LookingLabel(IntEnum):
    """Does the cyclist look at the smartphone display?"""

    NO = 0
    YES = 1
    UNSURE = 2
    """Cannot be decided (e.g. face hidden). Excluded from the evaluation."""


class QualityLabel(IntEnum):
    """Qualitative rating of the landmarks a face detector found."""

    GOOD = 0
    INACCURATE = 1
    """Roughly right, but visibly off (e.g. eyes or mouth misplaced)."""
    WRONG = 2
    """Landmarks do not match the face (or detected something that is not a face)."""


def utcnow() -> datetime:
    return datetime.now(UTC)


class Video(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    path: str = Field(unique=True)
    fps: float
    width: int
    height: int
    frame_count: int
    camera: dict = Field(sa_column=Column(JSON, nullable=False))
    """Intrinsics used for this video, see :class:`gaze_estimator.core.camera.Camera`."""
    created_at: datetime = Field(default_factory=utcnow)


class Frame(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("video_id", "index"),)

    id: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    index: int
    timestamp_ms: float
    image_path: str
    """Path of the JPG, relative to the directory of the database file."""
    brightness: float
    """Mean gray value (0-255)."""
    contrast: float
    """Standard deviation of the gray values."""
    sharpness: float
    """Variance of the Laplacian."""

    label_looking: int | None = None
    """:class:`LookingLabel`"""
    label_blink: bool | None = None
    """True if the eyes are closed (blink)."""
    labeled_at: datetime | None = None


class Face(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("frame_id", "detector"),)

    id: int | None = Field(default=None, primary_key=True)
    frame_id: int = Field(foreign_key="frame.id", index=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    detector: str = Field(index=True)
    detected: bool
    runtime_ms: float = 0.0

    bbox: list[float] | None = Field(default=None, sa_column=Column(JSON))
    landmarks_2d: list[list[float]] | None = Field(default=None, sa_column=Column(JSON))
    landmarks_3d: list[list[float]] | None = Field(default=None, sa_column=Column(JSON))
    rotation_vector: list[float] | None = Field(default=None, sa_column=Column(JSON))
    translation_vector: list[float] | None = Field(default=None, sa_column=Column(JSON))
    center: list[float] | None = Field(default=None, sa_column=Column(JSON))
    normalizing_rotation: list[list[float]] | None = Field(default=None, sa_column=Column(JSON))
    distance: float | None = None
    ear_left: float | None = None
    ear_right: float | None = None

    label_quality: int | None = None
    """:class:`QualityLabel`"""


class Gaze(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("face_id", "model"),)

    id: int | None = Field(default=None, primary_key=True)
    face_id: int = Field(foreign_key="face.id", index=True)
    frame_id: int = Field(foreign_key="frame.id", index=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    model: str = Field(index=True)
    pitch: float
    yaw: float
    vector: list[float] = Field(sa_column=Column(JSON, nullable=False))
    runtime_ms: float = 0.0
