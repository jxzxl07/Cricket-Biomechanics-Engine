# Cricket Biomechanics Engine

A cricket action classifier that records batting and bowling actions, extracts
MediaPipe pose landmarks, trains Random Forest classifiers, and analyses new
clips with confidence scores, key biomechanical metrics, and evaluation reports.

The model is available two ways: a **PyQt desktop app** for recording and
training, and a **containerised FastAPI inference service deployed on Azure
Container Apps** for programmatic access.

> **Live API:** https://cricket-api.wonderfuldesert-1933bece.uksouth.azurecontainerapps.io/docs

## Features

- PyQt desktop interface with four modes:
  - Bowling analysis
  - Batting analysis
  - Train bowling model
  - Train batting model
- Live webcam feed with MediaPipe skeleton overlay.
- 5 second countdown and 3 second clip recording.
- Labelled training capture for batting and bowling classes.
- Landmark extraction and feature-vector generation.
- Random Forest training with model persistence through `joblib`.
- Low-confidence correction loop for improving the next training run.
- Bowling legality pre-check for likely illegal deliveries.
- Evaluation summary with CV accuracy, holdout accuracy, classification report, and confusion matrix.
- HTTP inference API, containerised with Docker and deployed to Azure Container Apps.

## Classes

Bowling:

- `left_arm_leg`
- `left_arm_off`
- `left_arm_pace`
- `right_arm_leg`
- `right_arm_off`
- `right_arm_pace`

Batting:

- `cut`
- `drive`
- `flick`
- `pull`
- `reverse_sweep`
- `scoop`
- `sweep`

## Desktop App Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install desktop dependencies:

```bash
pip install -r requirements-desktop.txt
```

Run the app:

```bash
python3 main.py
```

## API Service

The trained models are also served over HTTP for programmatic access.
Training remains desktop-only; the API performs inference against the
committed model artifacts in `data/models/`.

### Endpoints

`GET /health` — returns service status and whether each model loaded.

    {"status": "ok", "models": {"bowling": true, "batting": true}}

`POST /predict?mode={bowling|batting}` — accepts a video clip and returns
the predicted class, confidence, full probability distribution, and the
extracted biomechanical features.

    curl -X POST "<base-url>/predict?mode=batting" -F "file=@clip.mp4"

Accepts `.mp4`, `.mov`, `.avi` up to 25 MB.

### Running locally

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn api.main:app --reload
```

Interactive docs at http://127.0.0.1:8000/docs

### Tests

```bash
pip install pytest httpx
pytest
```

## Docker & Deployment

The API is containerised and deployed to Azure Container Apps.

### Build and run locally

```bash
docker build -t cricket-api .
docker run -p 8000:8000 cricket-api
```

The image installs the system libraries MediaPipe and OpenCV need at runtime
(`libgl1`, `libglib2.0-0`, `libgles2-mesa`, `libegl1`) and runs uvicorn on
`0.0.0.0:8000`. Only the API dependencies are installed (`requirements-api.txt`);
the PyQt desktop toolkit is excluded from the image.

### Deploy to Azure Container Apps

The image is published to Docker Hub (`jxzxl/cricket-api`) and run on Azure
Container Apps with scale-to-zero:

```bash
az group create --name cricket-rg --location uksouth

az containerapp env create --name cricket-env --resource-group cricket-rg \
  --location uksouth --logs-destination none

az containerapp create --name cricket-api --resource-group cricket-rg \
  --environment cricket-env --image docker.io/jxzxl/cricket-api:latest \
  --target-port 8000 --ingress external --cpu 1 --memory 2Gi \
  --min-replicas 0 --max-replicas 1
```

`--min-replicas 0` scales the service to zero when idle. Tear down all
resources with `az group delete --name cricket-rg --yes --no-wait`.

## Data Privacy

The repository intentionally ignores personal capture data:

- `data/raw/`
- `data/landmarks/`
- `data/processed/`
- `data/evaluation/`
- `data/*.sqlite3`

Those folders can contain webcam videos, pose coordinates, derived movement
features, and local database records. They should stay local.

The committed model artifacts are:

- `data/models/pose_landmarker_lite.task`
- `data/models/batting.joblib`
- `data/models/bowling.joblib`

## Training Workflow

1. Open a training page.
2. Pick the correct label from the dropdown.
3. Record balanced clips for every class.
4. Press **Train Model**.
5. Press **View Evaluation** to inspect accuracy and the confusion matrix.

For a stronger model, keep the dataset balanced. A useful target is at least 20 clips per class.

## Analysis Workflow

1. Open Bowling or Batting mode.
2. Record a clip.
3. The app extracts landmarks and features.
4. Pre-checks run first:
   - no person detected
   - no batting shot / bowling action detected
   - illegal delivery for bowling
5. If the clip passes pre-checks, the trained model predicts the class and confidence.
6. If confidence is low, the app prompts for a corrected label to include in the next training run.

## Project Structure

```text
config.py                 Shared paths, labels, and recording settings
main.py                   Desktop application entry point
api/                      FastAPI service (/health, /predict)
database/                 SQLite schema and database helpers
gui/                      PyQt windows and pages
ml/                       Model training, prediction, and evaluation
vision/                   Camera worker, pose extraction, and feature engineering
tests/                    API tests
data/models/              Model files committed to the repo
Dockerfile                Container image for the API service
requirements-api.txt      API dependencies (no PyQt)
requirements-desktop.txt  Desktop dependencies
```

## Notes

The bowling legality detector is a rule-based 2D pose estimate. It is useful for
catching obvious chucking patterns, but it is not an official umpiring or
lab-grade measurement system.