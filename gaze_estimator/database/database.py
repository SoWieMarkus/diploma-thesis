import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, create_engine, Session, select
from contextlib import contextmanager
import cv2

from .models import Video, Frame, Face, Gaze
from dotenv import load_dotenv  # Add this import

load_dotenv("../.env")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///gaze_estimation.db")
logging.info(f"Using database URL: {DATABASE_URL}")


class DatabaseManager:
    """Database wrapper for managing video processing data."""

    def __init__(self, database_url: str = None):
        """
        Initialize the database manager.

        Args:
            database_url: SQLite database URL. Defaults to local database.
        """
        if database_url is None:
            database_url = "sqlite:///gaze_estimation.db"

        self.engine = create_engine(database_url, echo=False)

        # Create tables if they don't exist
        self._create_tables()

    def _create_tables(self):
        """Create all database tables."""
        SQLModel.metadata.create_all(self.engine)
        logging.info("Database tables created/verified")

    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = Session(self.engine)
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def get_session_dependency(self):
        """Get a database session for FastAPI dependency injection."""
        session = Session(self.engine)
        try:
            yield session
        finally:
            session.close()

    # Video Management
    def insert_video(self, title: str, path: str) -> Video:
        """
        Insert a new video into the database.

        Args:
            title: Video title/name
            path: Path to the video file

        Returns:
            Video: The created video object
        """
        with self.get_session() as session:
            # Check if video already exists
            existing_video = session.exec(
                select(Video).where(Video.path == path)
            ).first()

            if existing_video:
                logging.info(f"Video already exists: {path}")
                return existing_video

            video = Video(title=title, path=path)
            session.add(video)
            logging.info(f"Inserted video: {title} ({video.id})")
            return video

    # Frame Management
    def insert_frame(self, video_id: str, index: int, timestamp: datetime) -> Frame:
        """
        Insert a new frame into the database.

        Args:
            video_id: ID of the parent video
            index: Frame index in the video
            timestamp: Timestamp of the frame

        Returns:
            Frame: The created frame object
        """
        with self.get_session() as session:
            # Check if frame already exists
            existing_frame = session.exec(
                select(Frame).where(Frame.video_id == video_id, Frame.index == index)
            ).first()

            if existing_frame:
                logging.warning(
                    f"Frame already exists: video {video_id}, index {index}"
                )
                return existing_frame

            frame = Frame(video_id=video_id, index=index, timestamp=timestamp)
            session.add(frame)
            return frame

    def get_frame(self, video_id: str, index: int) -> Optional[Frame]:
        """Get frame by video ID and index."""
        with self.get_session() as session:
            return session.exec(
                select(Frame).where(Frame.video_id == video_id, Frame.index == index)
            ).first()

    def get_frames_by_video(self, video_id: str) -> List[Frame]:
        """Get all frames for a specific video."""
        with self.get_session() as session:
            return session.exec(select(Frame).where(Frame.video_id == video_id)).all()

    # Face Management
    def insert_face(self, face_data: Dict[str, Any]) -> Face:
        """
        Insert a new face detection result.

        Args:
            face_data: Dictionary containing face data

        Returns:
            Face: The created face object
        """
        with self.get_session() as session:
            face = Face(**face_data)
            session.add(face)
            return face

    def update_frame_face(self, frame_id: str, face_id: str, detector_type: str):
        """
        Update frame with face detection result.

        Args:
            frame_id: ID of the frame
            face_id: ID of the detected face
            detector_type: Type of detector ('dlib' or 'mediapipe')
        """
        with self.get_session() as session:
            frame = session.exec(select(Frame).where(Frame.id == frame_id)).first()

            if not frame:
                raise ValueError(f"Frame not found: {frame_id}")

            if detector_type == "dlib":
                frame.dlib_face_id = face_id
            elif detector_type == "mediapipe":
                frame.mediapipe_face_id = face_id
            else:
                raise ValueError(f"Unknown detector type: {detector_type}")

    # Gaze Management
    def insert_gaze(self, gaze_data: Dict[str, Any]) -> Gaze:
        """
        Insert a new gaze estimation result.

        Args:
            gaze_data: Dictionary containing gaze data

        Returns:
            Gaze: The created gaze object
        """
        with self.get_session() as session:
            gaze = Gaze(**gaze_data)
            session.add(gaze)
            return gaze

    def update_face_gaze(self, face_id: str, gaze_id: str, gaze_type: str):
        """
        Update face with gaze estimation result.

        Args:
            face_id: ID of the face
            gaze_id: ID of the gaze estimation
            gaze_type: Type of gaze estimator
        """
        with self.get_session() as session:
            face = session.exec(select(Face).where(Face.id == face_id)).first()

            if not face:
                raise ValueError(f"Face not found: {face_id}")

            if gaze_type == "eth-xgaze":
                face.eth_x_gaze_id = gaze_id
            elif gaze_type == "gaze360":
                face.gaze360_id = gaze_id
            elif gaze_type == "head-rotation":
                face.head_rotation_id = gaze_id
            elif gaze_type == "mpiifacegaze":
                face.mpiifacegaze_id = gaze_id
            else:
                raise ValueError(f"Unknown gaze type: {gaze_type}")

            session.commit()

    # Utility Methods
    def _delete_frame_data(self, session: Session, frame: Frame):
        """Delete all data associated with a frame."""
        # Delete faces and their gaze data
        for face in [frame.dlib_face, frame.mediapipe_face]:
            if face:
                # Delete associated gaze data
                for gaze in [
                    face.eth_x_gaze,
                    face.gaze360,
                    face.head_rotation,
                    face.mpiifacegaze,
                ]:
                    if gaze:
                        session.delete(gaze)
                session.delete(face)

    def clear_video_processing_data(self, video_id: str):
        """
        Clear all processing data for a video (faces, gaze) but keep the video and frames.

        Args:
            video_id: ID of the video to clear
        """
        with self.get_session() as session:
            frames = session.exec(select(Frame).where(Frame.video_id == video_id)).all()

            for frame in frames:
                self._delete_frame_data(session, frame)
                # Reset face IDs
                frame.dlib_face_id = None
                frame.mediapipe_face_id = None

            logging.info(f"Cleared processing data for video: {video_id}")


# Helper function to create a database manager instance
def create_database_manager() -> DatabaseManager:
    """
    Create a database manager instance.
    Returns:
        DatabaseManager: Configured database manager
    """
    return DatabaseManager(DATABASE_URL)
