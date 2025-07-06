from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
import uuid

# Constants for foreign key references
GAZE_TABLE_ID = "gaze.id"
FACE_TABLE_ID = "face.id"
VIDEO_TABLE_ID = "video.id"


class Video(SQLModel, table=True):
    """Video model representing a video file with metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    title: str
    path: str

    # Relationships
    frames: List["Frame"] = Relationship(back_populates="video")


class Frame(SQLModel, table=True):
    """Frame model representing a single frame in a video with detection results."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    video_id: str = Field(foreign_key=VIDEO_TABLE_ID, alias="videoId")
    index: int
    timestamp: datetime
    dlib_face_id: Optional[str] = Field(
        default=None, foreign_key=FACE_TABLE_ID, unique=True, alias="dlibFaceId"
    )
    mediapipe_face_id: Optional[str] = Field(
        default=None, foreign_key=FACE_TABLE_ID, unique=True, alias="mediapipeFaceId"
    )
    looks_label: Optional[int] = Field(default=None, alias="looksLabel")
    blink_label: Optional[int] = Field(default=None, alias="blinkLabel")

    # Relationships
    video: Video = Relationship(back_populates="frames")
    dlib_face: Optional["Face"] = Relationship(
        back_populates="dlib_frame",
        sa_relationship_kwargs={"foreign_keys": "[Frame.dlib_face_id]"},
    )
    mediapipe_face: Optional["Face"] = Relationship(
        back_populates="mediapipe_frame",
        sa_relationship_kwargs={"foreign_keys": "[Frame.mediapipe_face_id]"},
    )


class Face(SQLModel, table=True):
    """Face model containing face detection results and landmarks."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    quality_label: Optional[int] = Field(default=None, alias="qualityLabel")
    distance: float
    position: str
    center: str
    ear_left: float = Field(alias="earLeft")
    ear_right: float = Field(alias="earRight")

    # Gaze model foreign keys
    eth_x_gaze_id: Optional[str] = Field(
        default=None, foreign_key=GAZE_TABLE_ID, unique=True, alias="ethXGazeId"
    )
    gaze360_id: Optional[str] = Field(
        default=None, foreign_key=GAZE_TABLE_ID, unique=True, alias="gaze360Id"
    )
    head_rotation_id: Optional[str] = Field(
        default=None, foreign_key=GAZE_TABLE_ID, unique=True, alias="headRotationId"
    )
    mpiifacegaze_id: Optional[str] = Field(
        default=None, foreign_key=GAZE_TABLE_ID, unique=True, alias="mpiifacegazeId"
    )

    # Landmark and rotation data stored as strings
    # The vectors are stored as "<x>,<y>,<z>"
    # The landmarks are stored as "<norm_x>,<norm_y>,<norm_z>,<img_x>,<img_y>,<world_x>,<world_y>,<world_z>"
    # "norm_" is the normalized image coordinate (plus the estimate depth from mediapipe as norm_z)
    # "img_" is the position of the landmark in pixels
    # "world_" is the estimated 3d position
    rotation_vector: str = Field(alias="rotationVector")
    landmark_left_eye_left: str = Field(alias="landmarkLeftEyeLeft")
    landmark_left_eye_right: str = Field(alias="landmarkLeftEyeRight")
    landmark_right_eye_left: str = Field(alias="landmarkRightEyeLeft")
    landmark_right_eye_right: str = Field(alias="landmarkRightEyeRight")
    landmark_mouth_left: str = Field(alias="landmarkMouthLeft")
    landmark_mouth_right: str = Field(alias="landmarkMouthRight")
    landmark_nose_left: str = Field(alias="landmarkNoseLeft")
    landmark_nose_right: str = Field(alias="landmarkNoseRight")
    landmark_nose_tip: str = Field(alias="landmarkNoseTip")

    # Relationships
    dlib_frame: Optional[Frame] = Relationship(
        back_populates="dlib_face",
        sa_relationship_kwargs={"foreign_keys": "[Frame.dlib_face_id]"},
    )
    mediapipe_frame: Optional[Frame] = Relationship(
        back_populates="mediapipe_face",
        sa_relationship_kwargs={"foreign_keys": "[Frame.mediapipe_face_id]"},
    )
    eth_x_gaze: Optional["Gaze"] = Relationship(
        back_populates="eth_x_gaze_face",
        sa_relationship_kwargs={"foreign_keys": "[Face.eth_x_gaze_id]"},
    )
    gaze360: Optional["Gaze"] = Relationship(
        back_populates="gaze360_face",
        sa_relationship_kwargs={"foreign_keys": "[Face.gaze360_id]"},
    )
    head_rotation: Optional["Gaze"] = Relationship(
        back_populates="head_rotation_face",
        sa_relationship_kwargs={"foreign_keys": "[Face.head_rotation_id]"},
    )
    mpiifacegaze: Optional["Gaze"] = Relationship(
        back_populates="mpiifacegaze_face",
        sa_relationship_kwargs={"foreign_keys": "[Face.mpiifacegaze_id]"},
    )


class Gaze(SQLModel, table=True):
    """Gaze model containing gaze estimation results."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    pitch: float
    yaw: float
    direction: str  # The vectors are stored as "<x>,<y>,<z>"
    intersection: str

    # Back references to Face model with named relations
    mpiifacegaze_face: Optional[Face] = Relationship(
        back_populates="mpiifacegaze",
        sa_relationship_kwargs={"foreign_keys": "[Face.mpiifacegaze_id]"},
    )
    eth_x_gaze_face: Optional[Face] = Relationship(
        back_populates="eth_x_gaze",
        sa_relationship_kwargs={"foreign_keys": "[Face.eth_x_gaze_id]"},
    )
    gaze360_face: Optional[Face] = Relationship(
        back_populates="gaze360",
        sa_relationship_kwargs={"foreign_keys": "[Face.gaze360_id]"},
    )
    head_rotation_face: Optional[Face] = Relationship(
        back_populates="head_rotation",
        sa_relationship_kwargs={"foreign_keys": "[Face.head_rotation_id]"},
    )
