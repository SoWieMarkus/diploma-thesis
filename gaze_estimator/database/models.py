from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Relationship, JSON, Column
from sqlalchemy import Index
import uuid


VIDEO_ID = "Video.id"


class Video(SQLModel, table=True):
    """Video model representing a video file with metadata."""

    __tablename__ = "Video"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    path: str = Field(unique=True)
    name: str


class Frame(SQLModel, table=True):
    """Frame model representing a single frame in a video."""

    __tablename__ = "Frame"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    videoId: str
    index: int

    # Labels for manual annotation
    looksOnScreenLabel: Optional[int] = Field(
        default=None
    )  # 0: not on screen, 1: on screen, 2: not detected
    blinksLabel: Optional[int] = Field(
        default=None
    )  # 0: not blinking, 1: blinking, 2: not detected

    # Index equivalent
    __table_args__ = (Index("idx_frame_video_index", "videoId", "index"),)


class Face(SQLModel, table=True):
    """Face model containing face detection results."""

    __tablename__ = "Face"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    videoId: str
    frameId: str
    faceDetector: str

    qualityLabel: Optional[int] = Field(
        default=None
    )  # 0: good, 1: bad, 2: not detected

    # JSON data column
    data: Dict[str, Any] = Field(sa_column=Column(JSON))

    # Unique constraint equivalent
    __table_args__ = (
        Index("idx_face_unique", "videoId", "frameId", "faceDetector", unique=True),
    )


class Gaze(SQLModel, table=True):
    """Gaze model containing gaze estimation results."""

    __tablename__ = "Gaze"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    faceId: str
    videoId: str
    frameId: str
    gazeDetector: str

    # JSON data column
    data: Dict[str, Any] = Field(sa_column=Column(JSON))

    # Unique constraint equivalent
    __table_args__ = (
        Index(
            "idx_gaze_unique",
            "faceId",
            "videoId",
            "frameId",
            "gazeDetector",
            unique=True,
        ),
    )
