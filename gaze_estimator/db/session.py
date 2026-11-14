"""SQLite engine and session setup."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel, create_engine

from . import models  # noqa: F401 - registers the tables


def create_db_engine(path: Path, *, create: bool = True) -> Engine:
    path = Path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.is_file():
        raise FileNotFoundError(f"Database '{path}' does not exist. Run `gaze-process` first.")

    engine = create_engine(
        f"sqlite:///{path.as_posix()}", connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    return engine


def open_session(engine: Engine) -> Session:
    # Objects stay usable after commit (no lazy reload), which keeps the pipeline code simple.
    return Session(engine, expire_on_commit=False)
