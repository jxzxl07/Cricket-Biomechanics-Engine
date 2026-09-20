# CreaseLab web application image.
# Multi-stage: Node and training tools never enter the runtime image. FastAPI
# serves both the API and the compiled React app, so the public service URL
# cannot accidentally expose the API metadata page instead of the product.

FROM node:22-alpine AS frontend-build

WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS base

# MediaPipe runtime needs these system libraries; OpenCV is headless so no GL is required.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS build

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt \
    && pip install --no-cache-dir --prefix=/install --no-deps mediapipe==0.10.21

FROM base AS runtime

RUN useradd --create-home --uid 10001 creaselab
WORKDIR /app

COPY --from=build /install /usr/local

COPY api/ ./api/
COPY ml/ ./ml/
COPY vision/ ./vision/
COPY config.py .
COPY --from=frontend-build /web/dist ./frontend/dist

# Only the deployable artifacts; no raw video, no research code, no tests.
COPY data/models/pose_landmarker_lite.task ./data/models/
COPY data/models/batting_pose.onnx ./data/models/
COPY data/models/batting_pose.json ./data/models/
COPY data/models/bowling_pose.onnx ./data/models/
COPY data/models/bowling_pose.json ./data/models/
COPY data/models/MODEL_CARD.md ./data/models/

# Verify every shipped artifact against its spec checksum before the image is used.
RUN python -c "from ml.model_spec import load_spec; \
    print('batting pose:', load_spec('batting_pose.json').verify_artifact()[:16]); \
    print('bowling pose:', load_spec('bowling_pose.json').verify_artifact()[:16])"

USER creaselab
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MEDIAPIPE_DISABLE_GPU=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# One worker: the models live in process memory and free-tier RAM is tight.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
