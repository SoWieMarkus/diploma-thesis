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
    def insert_video(self, path: str, name: str) -> Video:
        """
        Insert a new video into the database.

        Args:
            title: Video title/name
            path: Path to the video file

        Returns:
            Video: The created video object
        """
        # TODO upsert

        with self.get_session() as session:
            # Check if video already exists
            existing_video = session.exec(
                select(Video).where(Video.path == path)
            ).first()

            if existing_video:
                logging.debug(f"Video already exists: {existing_video.id}")
                return existing_video

            video = Video(path=path, name=name)
            session.add(video)
            logging.debug(f"Inserted video: {name} ({video.id})")
            return video

    # Frame Management
    def insert_frame(self, video_id: str, index: int) -> Frame:
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
                logging.debug(f"Frame already exists: {existing_frame.id}")
                return existing_frame

            frame = Frame(video_id=video_id, index=index)
            session.add(frame)
            logging.debug(f"Inserted frame: {video_id} [{index}] ({frame.id})")
            return frame


# Helper function to create a database manager instance
def create_database_manager() -> DatabaseManager:
    """
    Create a database manager instance.
    Returns:
        DatabaseManager: Configured database manager
    """
    return DatabaseManager(DATABASE_URL)
