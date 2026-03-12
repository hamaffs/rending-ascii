"""
FastAPI Backend
Handles file uploads and coordinates video generation.
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from video_assembler import VideoAssembler


app = FastAPI(title="ASCII Video Generator API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure upload and output directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)

# Store rendering jobs
rendering_jobs = {}


class RenderRequest(BaseModel):
    style: str = "classic"
    mode: str = "geometric"
    fps: int = 30


class RenderResponse(BaseModel):
    job_id: str
    status: str
    message: str
    video_url: Optional[str] = None
    progress: float = 0.0


class ProgressTracker:
    """Track rendering progress for a job."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.progress = 0.0
        self.status = "pending"
        self.output_path = None
        self.error = None
    
    def update(self, progress: float):
        self.progress = progress
        if progress >= 1.0:
            self.status = "completed"


async def render_video_task(
    job_id: str,
    audio_path: str,
    style: str,
    mode: str,
    fps: int,
    objects: list = None
):
    """Background task to render video."""
    tracker = rendering_jobs[job_id]
    
    try:
        tracker.status = "processing"
        
        # Generate unique output filename
        output_filename = f"{job_id}.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        # Create assembler with progress callback
        def progress_callback(progress: float):
            tracker.update(progress)
        
        assembler = VideoAssembler(
            audio_path=audio_path,
            output_path=str(output_path),
            fps=fps,
            style=style,
            mode=mode,
            objects=objects
        )
        assembler.set_progress_callback(progress_callback)
        
        # Render the video
        result_path = assembler.render()
        
        tracker.output_path = result_path
        tracker.progress = 1.0
        tracker.status = "completed"
        
    except Exception as e:
        tracker.status = "failed"
        tracker.error = str(e)
        print(f"Error rendering video: {e}")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    template_path = TEMPLATE_DIR / "index.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), media_type="text/html")
    return {"message": "ASCII Video Generator API", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/render")
async def render_video(
    file: UploadFile = File(...),
    style: str = Form("classic"),
    mode: str = Form("enhanced_3d"),
    fps: int = Form(30),
    object1: str = Form("sphere"),
    object2: str = Form("cube"),
    object3: str = Form("pyramid")
):
    """
    Upload an audio file and start rendering a video.
    
    - **file**: Audio file (MP3/WAV)
    - **style**: Visual style (classic, matrix, high_contrast)
    - **mode**: Visual mode (geometric, waveform, particles, mood, enhanced_3d)
    - **fps**: Frames per second (default 30)
    - **object1, object2, object3**: 3D objects for enhanced_3d mode (sphere, cube, pyramid, torus)
    """
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported: MP3, WAV, OGG, FLAC"
        )
    
    # Validate style
    valid_styles = ["classic", "matrix", "high_contrast"]
    if style not in valid_styles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style. Valid options: {valid_styles}"
        )
    
    # Validate mode
    valid_modes = ["geometric", "waveform", "particles", "mood", "enhanced_3d"]
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Valid options: {valid_modes}"
        )
    
    # Validate objects
    valid_objects = ["sphere", "cube", "pyramid", "torus"]
    objects = [object1, object2, object3]
    for obj in objects:
        if obj not in valid_objects:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid object: {obj}. Valid options: {valid_objects}"
            )
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    audio_filename = f"{job_id}_{file.filename}"
    audio_path = UPLOAD_DIR / audio_filename
    
    with open(audio_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Initialize job tracker
    tracker = ProgressTracker(job_id)
    rendering_jobs[job_id] = tracker
    
    # Start background rendering task
    asyncio.create_task(
        render_video_task(job_id, str(audio_path), style, mode, fps, objects)
    )
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": "Video rendering started",
        "progress": 0.0,
        "mode": mode,
        "objects": objects
    }


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a rendering job."""
    
    if job_id not in rendering_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    tracker = rendering_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": tracker.status,
        "progress": tracker.progress
    }
    
    if tracker.status == "completed" and tracker.output_path:
        # Generate download URL
        video_filename = os.path.basename(tracker.output_path)
        response["video_url"] = f"/api/download/{video_filename}"
    
    if tracker.status == "failed":
        response["error"] = tracker.error
    
    return response


@app.get("/api/download/{filename}")
async def download_video(filename: str):
    """Download the generated video."""
    
    video_path = OUTPUT_DIR / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=filename
    )


@app.get("/api/styles")
async def get_styles():
    """Get available visual styles."""
    return {
        "styles": [
            {"id": "classic", "name": "Classic", "description": "White ASCII on black"},
            {"id": "matrix", "name": "Matrix", "description": "Green matrix style"},
            {"id": "high_contrast", "name": "High Contrast", "description": "Bold black and white"}
        ],
        "modes": [
            {"id": "geometric", "name": "Geometric", "description": "Pulsing shapes"},
            {"id": "waveform", "name": "Waveform", "description": "Sine waves"},
            {"id": "particles", "name": "Particles", "description": "Exploding characters"},
            {"id": "mood", "name": "Mood-Aware", "description": "AI-analyzes song mood for visuals"},
            {"id": "enhanced_3d", "name": "3D Enhanced", "description": "3D objects with depth effects & lyrics mood"}
        ],
        "objects": [
            {"id": "sphere", "name": "Sphere", "description": "3D sphere with shading"},
            {"id": "cube", "name": "Cube", "description": "3D cube with faces"},
            {"id": "pyramid", "name": "Pyramid", "description": "3D pyramid"},
            {"id": "torus", "name": "Torus", "description": "3D ring/torus"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
