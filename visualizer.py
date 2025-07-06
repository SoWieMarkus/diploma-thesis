import os
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from typing import List
from gaze_estimator.database import create_database_manager, Video


database = create_database_manager()

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

static_path = os.path.join(os.path.dirname(__file__), "visualizer", "dist")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the Angular frontend."""
    html_path = os.path.join(
        os.path.dirname(__file__), "visualizer", "dist", "index.html"
    )
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    else:
        raise HTTPException(
            status_code=404,
            detail="Frontend not found. Please build the Angular app first.",
        )


@app.get("/api")
async def root():
    """Health check endpoint."""
    return {"message": "Gaze Estimation API is running"}


@app.get("/api/videos", response_model=List[Video])
async def get_all_videos(session: Session = Depends(database.get_session_dependency)):
    """Get all videos in the database."""
    try:
        statement = select(Video)
        videos = session.exec(statement).all()
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching videos: {str(e)}")


@app.get("/videos/{video_id}", response_model=Video)
async def get_video_by_id(
    video_id: str, session: Session = Depends(database.get_session_dependency)
):
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


if __name__ == "__main__":
    uvicorn.run(app, port=3000)
