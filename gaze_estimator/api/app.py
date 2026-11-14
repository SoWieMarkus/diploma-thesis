"""REST API for the web app (viewer + labeling) and static hosting of the built web app."""

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Integer, cast, func
from sqlmodel import Session, select

from ..core.paths import WEB_DIST_DIR
from ..db import models
from ..db.session import create_db_engine, open_session


class FrameLabelUpdate(BaseModel):
    label_looking: int | None = Field(default=None, ge=0, le=2)
    label_blink: bool | None = None


class FaceLabelUpdate(BaseModel):
    label_quality: int | None = Field(default=None, ge=0, le=2)


def create_app(database: Path, web_dist: Path | None = WEB_DIST_DIR) -> FastAPI:
    database = Path(database).resolve()
    engine = create_db_engine(database, create=False)
    data_root = database.parent

    app = FastAPI(title="Gaze Estimator-Inator", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_session():
        with open_session(engine) as session:
            yield session

    SessionDep = Annotated[Session, Depends(get_session)]

    @app.get("/api/health")
    def health():
        return {"status": "ok", "database": str(database)}

    @app.get("/api/models")
    def list_models(session: SessionDep):
        return {
            "faces": sorted(session.exec(select(models.Face.detector).distinct()).all()),
            "gazes": sorted(session.exec(select(models.Gaze.model).distinct()).all()),
        }

    @app.get("/api/videos")
    def list_videos(session: SessionDep):
        frame_stats = {
            row[0]: row[1:]
            for row in session.exec(
                select(
                    models.Frame.video_id,
                    func.count(models.Frame.id),
                    func.count(models.Frame.label_looking),
                    func.count(models.Frame.label_blink),
                    func.min(models.Frame.id),
                ).group_by(models.Frame.video_id)
            )
        }
        face_stats: dict[int, dict[str, dict]] = defaultdict(dict)
        for video_id, detector, total, detected, rated in session.exec(
            select(
                models.Face.video_id,
                models.Face.detector,
                func.count(models.Face.id),
                func.sum(cast(models.Face.detected, Integer)),
                func.count(models.Face.label_quality),
            ).group_by(models.Face.video_id, models.Face.detector)
        ):
            face_stats[video_id][detector] = {
                "frames": total,
                "detected": int(detected or 0),
                "quality_labeled": rated,
            }

        result = []
        for video in session.exec(select(models.Video).order_by(models.Video.name)):
            frames, looking, blink, first_frame = frame_stats.get(video.id, (0, 0, 0, None))
            result.append(
                {
                    **video.model_dump(exclude={"path"}),
                    "processed_frames": frames,
                    "labeled_looking": looking,
                    "labeled_blink": blink,
                    "thumbnail_frame_id": first_frame,
                    "faces": face_stats.get(video.id, {}),
                }
            )
        return result

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: int, session: SessionDep):
        video = session.get(models.Video, video_id)
        if video is None:
            raise HTTPException(404, "Video not found")
        return video.model_dump(exclude={"path"})

    @app.get("/api/videos/{video_id}/frames")
    def list_frames(video_id: int, session: SessionDep):
        """Lightweight list of all processed frames of a video (for timelines and navigation)."""
        detected: dict[int, dict[str, bool]] = defaultdict(dict)
        quality: dict[int, dict[str, int | None]] = defaultdict(dict)
        for frame_id, detector, is_detected, label in session.exec(
            select(
                models.Face.frame_id,
                models.Face.detector,
                models.Face.detected,
                models.Face.label_quality,
            ).where(models.Face.video_id == video_id)
        ):
            detected[frame_id][detector] = is_detected
            quality[frame_id][detector] = label

        frames = session.exec(
            select(models.Frame)
            .where(models.Frame.video_id == video_id)
            .order_by(models.Frame.index)
        )
        return [
            {
                "id": f.id,
                "index": f.index,
                "timestamp_ms": f.timestamp_ms,
                "label_looking": f.label_looking,
                "label_blink": f.label_blink,
                "detected": detected.get(f.id, {}),
                "label_quality": quality.get(f.id, {}),
            }
            for f in frames
        ]

    @app.get("/api/videos/{video_id}/gazes")
    def gaze_track(video_id: int, face: str, gaze: str, session: SessionDep):
        """Face center and gaze vector of every frame for one face detector + gaze model."""
        rows = session.exec(
            select(models.Frame.id, models.Frame.index, models.Face.center, models.Gaze.vector)
            .join(models.Face, models.Face.frame_id == models.Frame.id)
            .join(models.Gaze, models.Gaze.face_id == models.Face.id)
            .where(
                models.Frame.video_id == video_id,
                models.Face.detector == face,
                models.Gaze.model == gaze,
            )
            .order_by(models.Frame.index)
        )
        return [
            {"frame_id": frame_id, "index": index, "center": center, "vector": vector}
            for frame_id, index, center, vector in rows
        ]

    @app.get("/api/frames/{frame_id}")
    def get_frame(frame_id: int, session: SessionDep):
        frame = session.get(models.Frame, frame_id)
        if frame is None:
            raise HTTPException(404, "Frame not found")
        faces = session.exec(select(models.Face).where(models.Face.frame_id == frame_id)).all()
        gazes = session.exec(select(models.Gaze).where(models.Gaze.frame_id == frame_id)).all()
        gazes_by_face: dict[int, list] = defaultdict(list)
        for gaze in gazes:
            gazes_by_face[gaze.face_id].append(gaze.model_dump(exclude={"frame_id", "video_id"}))
        return {
            **frame.model_dump(exclude={"image_path"}),
            "faces": [
                {
                    **face.model_dump(exclude={"frame_id", "video_id"}),
                    "gazes": gazes_by_face.get(face.id, []),
                }
                for face in faces
            ],
        }

    @app.get("/api/frames/{frame_id}/image")
    def get_frame_image(frame_id: int, session: SessionDep):
        frame = session.get(models.Frame, frame_id)
        if frame is None:
            raise HTTPException(404, "Frame not found")
        path = (data_root / frame.image_path).resolve()
        if not path.is_file():
            raise HTTPException(404, "Image file missing")
        return FileResponse(
            path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"}
        )

    @app.patch("/api/frames/{frame_id}")
    def update_frame_labels(frame_id: int, update: FrameLabelUpdate, session: SessionDep):
        frame = session.get(models.Frame, frame_id)
        if frame is None:
            raise HTTPException(404, "Frame not found")
        for field in update.model_fields_set:
            setattr(frame, field, getattr(update, field))
        frame.labeled_at = datetime.now(UTC)
        session.add(frame)
        session.commit()
        return frame.model_dump(exclude={"image_path"})

    @app.patch("/api/faces/{face_id}")
    def update_face_labels(face_id: int, update: FaceLabelUpdate, session: SessionDep):
        face = session.get(models.Face, face_id)
        if face is None:
            raise HTTPException(404, "Face not found")
        if "label_quality" in update.model_fields_set:
            face.label_quality = update.label_quality
        session.add(face)
        session.commit()
        return {"id": face.id, "label_quality": face.label_quality}

    if web_dist is not None and (Path(web_dist) / "index.html").is_file():
        web_dist = Path(web_dist)
        if (web_dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = (web_dist / path).resolve()
            if path and candidate.is_file() and web_dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(web_dist / "index.html")

    return app
