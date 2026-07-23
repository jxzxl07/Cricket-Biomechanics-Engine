import logging
import tempfile
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from ml.classifier import load_model_package, predict_from_landmarks
from vision.pose import extract_landmarks_from_video

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi"}


class Mode(str, Enum):
    bowling = "bowling"
    batting = "batting"


MODEL_STATUS: dict[str, bool] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for mode in Mode:
        try:
            load_model_package(mode.value)
            MODEL_STATUS[mode.value] = True
            logger.info("Loaded %s model", mode.value)
        except FileNotFoundError:
            MODEL_STATUS[mode.value] = False
            logger.warning("No trained %s model found", mode.value)
    yield
    MODEL_STATUS.clear()


app = FastAPI(
    title="Cricket Biomechanics API",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    ready = bool(MODEL_STATUS) and all(MODEL_STATUS.values())
    return {
        "status": "ok" if ready else "degraded",
        "models": MODEL_STATUS,
    }


@app.post("/predict")
async def predict(mode: Mode, file: UploadFile = File(...)):
    if not MODEL_STATUS.get(mode.value):
        raise HTTPException(503, f"No trained {mode.value} model is loaded")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type '{suffix}'")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Clip exceeds the 25 MB limit")

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_path = Path(tmpdir) / f"upload{suffix}"
        clip_path.write_bytes(contents)
        landmarks_path = Path(tmpdir) / "landmarks.json"

        try:
            extract_landmarks_from_video(
                clip_path, mode.value, output_path=landmarks_path
            )
            return predict_from_landmarks(landmarks_path, mode.value)
        except ValueError as error:
            raise HTTPException(400, str(error))
        except FileNotFoundError as error:
            raise HTTPException(503, str(error))