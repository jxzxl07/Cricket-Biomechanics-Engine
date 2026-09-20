"""CreaseLab public API."""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from api.analysis import AnalysisEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 12
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


class Mode(str, Enum):
    batting = "batting"
    bowling = "bowling"


class CameraAngle(str, Enum):
    side_on = "side_on"
    front_on = "front_on"
    rear = "rear"
    unknown = "unknown"


ENGINE: AnalysisEngine | None = None
STARTUP_ERROR: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ENGINE, STARTUP_ERROR
    try:
        ENGINE = AnalysisEngine()
        STARTUP_ERROR = None
        logger.info("CreaseLab analysis engine loaded")
    except Exception as error:
        ENGINE = None
        STARTUP_ERROR = f"{type(error).__name__}: {error}"
        logger.exception("Analysis engine failed to load")
    yield
    ENGINE = None


app = FastAPI(
    title="CreaseLab Cricket Analysis API",
    description="Video classification, pose-derived movement metrics, replay timeline, and optional AI coaching.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "CreaseLab", "version": app.version, "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok" if ENGINE else "degraded",
        "engine_ready": ENGINE is not None,
        "error": STARTUP_ERROR,
    }


@app.get("/api/v1/models")
def models():
    return {
        "batting": {
            "id": "efficientnetb0-gru-cricshot10-v1",
            "classes": ["cover_drive", "defence", "flick", "hook", "late_cut", "lofted_shot", "pull", "square_cut", "straight_drive", "sweep"],
            "status": "ready" if ENGINE else "unavailable",
            "experimental": False,
        },
        "bowling": {
            "id": "pose-action-prototype-v1",
            "classes": ["left_arm_pace", "right_arm_pace", "left_arm_spin", "right_arm_spin"],
            "status": "experimental",
            "experimental": True,
            "note": "Broad pace/spin estimate. Fine-grained spin classification awaits a licensed model that passes evaluation.",
        },
    }


@app.post("/api/v1/analyze")
async def analyze(
    mode: Mode = Query(...),
    camera_angle: CameraAngle = Query(CameraAngle.unknown),
    use_ai_coach: bool = Query(False, description="When configured, sends three selected stills and metrics to OpenAI."),
    file: UploadFile = File(...),
):
    if ENGINE is None:
        raise HTTPException(503, "The analysis engine is not ready")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported video type '{suffix or 'unknown'}'. Use MP4, MOV, AVI, WebM, or MKV.")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(400, "The uploaded video is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The clip exceeds the 25 MB limit")

    with tempfile.TemporaryDirectory(prefix="creaselab-") as tmpdir:
        video_path = Path(tmpdir) / f"clip{suffix}"
        video_path.write_bytes(contents)
        duration = await run_in_threadpool(_video_duration, video_path)
        if duration <= 0:
            raise HTTPException(400, "The uploaded video could not be decoded")
        if duration > MAX_DURATION_SECONDS:
            raise HTTPException(400, f"Keep clips under {MAX_DURATION_SECONDS} seconds (received {duration:.1f}s)")
        try:
            return await run_in_threadpool(
                ENGINE.analyze, video_path, mode.value, camera_angle.value, use_ai_coach
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except Exception as error:
            logger.exception("Video analysis failed")
            raise HTTPException(500, "Analysis failed. Try a shorter clip with the full player visible.") from error


def _video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return frames / fps if capture.isOpened() and fps > 0 and frames > 0 else 0.0
    finally:
        capture.release()
