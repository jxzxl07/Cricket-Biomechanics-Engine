# Cricket Biomechanics Engine

A desktop app for recording cricket batting and bowling actions, extracting MediaPipe pose landmarks, training Random Forest classifiers, and analysing new clips with confidence scores, key biomechanical metrics, and evaluation reports.

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

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python3 main.py
```

## Data Privacy

The repository intentionally ignores personal capture data:

- `data/raw/`
- `data/landmarks/`
- `data/processed/`
- `data/evaluation/`
- `data/*.sqlite3`

Those folders can contain webcam videos, pose coordinates, derived movement features, and local database records. They should stay local.

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
main.py                   Application entry point
database/                 SQLite schema and database helpers
gui/                      PyQt windows and pages
ml/                       Model training, prediction, and evaluation
vision/                   Camera worker, pose extraction, and feature engineering
data/models/              Model files committed to the repo
```

## Notes

The bowling legality detector is a rule-based 2D pose estimate. It is useful for catching obvious chucking patterns, but it is not an official umpiring or lab-grade measurement system.
