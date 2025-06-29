from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, create_engine, Session, select
from typing import List, Optional
import os
import sys
import uvicorn

# Add the parent directory to the path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from ...database.models import Video, Frame, Face, Gaze

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///../../gaze_estimation.db")
engine = create_engine(DATABASE_URL, echo=True)

# Create FastAPI app
app = FastAPI(
    title="Gaze Estimation API",
    description="API for accessing gaze estimation video data",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get database session
def get_session():
    with Session(engine) as session:
        yield session


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Gaze Estimation API is running"}


@app.get("/videos", response_model=List[Video])
async def get_all_videos(session: Session = Depends(get_session)):
    """Get all videos in the database."""
    try:
        statement = select(Video)
        videos = session.exec(statement).all()
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching videos: {str(e)}")


@app.get("/videos/{video_id}", response_model=Video)
async def get_video_by_id(video_id: str, session: Session = Depends(get_session)):
    """Get a specific video by ID."""
    try:
        video = session.get(Video, video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        return video
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching video: {str(e)}")


@app.get("/videos/{video_id}/frames", response_model=List[Frame])
async def get_video_frames(
    video_id: str,
    skip: int = 0,
    limit: int = 100,
    session: Session = Depends(get_session),
):
    """Get frames for a specific video with pagination."""
    try:
        # Check if video exists
        video = session.get(Video, video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get frames with pagination
        statement = (
            select(Frame).where(Frame.video_id == video_id).offset(skip).limit(limit)
        )
        frames = session.exec(statement).all()
        return frames
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching frames: {str(e)}")


@app.get("/frames/{frame_id}", response_model=Frame)
async def get_frame_by_id(frame_id: str, session: Session = Depends(get_session)):
    """Get a specific frame by ID."""
    try:
        frame = session.get(Frame, frame_id)
        if not frame:
            raise HTTPException(status_code=404, detail="Frame not found")
        return frame
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching frame: {str(e)}")


@app.get("/frames/{frame_id}/faces")
async def get_frame_faces(frame_id: str, session: Session = Depends(get_session)):
    """Get face detections for a specific frame."""
    try:
        frame = session.get(Frame, frame_id)
        if not frame:
            raise HTTPException(status_code=404, detail="Frame not found")

        faces = {"dlib_face": frame.dlib_face, "mediapipe_face": frame.mediapipe_face}
        return faces
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching faces: {str(e)}")


@app.get("/faces/{face_id}", response_model=Face)
async def get_face_by_id(face_id: str, session: Session = Depends(get_session)):
    """Get a specific face detection by ID."""
    try:
        face = session.get(Face, face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Face not found")
        return face
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching face: {str(e)}")


@app.get("/faces/{face_id}/gaze")
async def get_face_gaze(face_id: str, session: Session = Depends(get_session)):
    """Get all gaze estimations for a specific face."""
    try:
        face = session.get(Face, face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Face not found")

        gaze_data = {
            "eth_x_gaze": face.eth_x_gaze,
            "gaze360": face.gaze360,
            "head_rotation": face.head_rotation,
            "mpiifacegaze": face.mpiifacegaze,
        }
        return gaze_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching gaze data: {str(e)}"
        )


@app.get("/gaze/{gaze_id}", response_model=Gaze)
async def get_gaze_by_id(gaze_id: str, session: Session = Depends(get_session)):
    """Get a specific gaze estimation by ID."""
    try:
        gaze = session.get(Gaze, gaze_id)
        if not gaze:
            raise HTTPException(status_code=404, detail="Gaze estimation not found")
        return gaze
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching gaze: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
