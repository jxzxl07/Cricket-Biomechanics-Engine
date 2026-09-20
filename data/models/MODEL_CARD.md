# CreaseLab model card

All figures below come from generated artifacts in `data/evaluation/` and are
copied into the runtime spec by `research/evaluation/publish_benchmark.py`. The
API serves those same numbers, so the product cannot claim more than was measured.

## Batting shot classifier (experimental)

| | |
| --- | --- |
| Artifact | `batting_video.onnx` (21 MB) |
| SHA-256 | `102fb0b9e4402bef4adb192347974fbd66fb67113f9471b2fa1d14969c9051e4` |
| Spec | `batting_video.json` (class map, preprocessing, thresholds, benchmark) |
| Architecture | EfficientNetB0 (ImageNet weights, frozen) in a TimeDistributed wrapper, two GRU layers (256 → 128), Dense(1024), softmax(10) |
| Input | 30 RGB frames, 224 × 224, bilinear resize with centred zero padding, values 0–255 |
| Classes | cover drive, defence, flick, hook, late cut, lofted shot, pull, square cut, straight drive, sweep |
| Source | [RITIK-12/CricketShotClassification](https://github.com/RITIK-12/CricketShotClassification), `model_weights.h5` |

### Measured benchmark

Evaluated with the production code path (`ml.video_classifier`) on the 44
historical phone clips whose stored labels map onto the model's ten classes.
`reverse_sweep` and `scoop` have no model class and are excluded from the headline
numbers. Full output: `data/evaluation/revamp_batting_benchmark.md`.

| Metric | Value |
| --- | --- |
| Top-1 accuracy | **29.6%** (chance 10%) |
| Top-2 accuracy | 52.3% |
| Macro F1 | 0.073 |
| Session-grouped majority accuracy | 25% |
| Prediction spread | `straight_drive` × 37, `flick` × 5, `lofted_shot` × 2 |
| Mean confidence, correct | 0.993 |
| Mean confidence, incorrect | 0.932 |

Selective prediction does not rescue it: keeping only predictions above 0.95
coverage/confidence still yields 36% accuracy on 82% of clips.

The source repository reports 94% test accuracy. That figure is **not** reproduced
here and is not quoted as a product claim. Two independent problems were found:

1. **Preprocessing bug in the upstream demo.** The ONNX graph begins with
   `Rescaling(1/255)` and ImageNet normalisation, so the weights expect 0–255
   input. The published Streamlit app feeds `tf.image.resize_with_pad` output,
   which is float 0–1. Feeding the correct 0–255 range raises top-1 accuracy on
   our clips from 15.9% to 29.6%; the bug is documented in
   `research/evaluation/batting_preprocessing_probe.py`.
2. **Domain mismatch.** The weights were trained on broadcast match footage.
   Phone clips from an indoor net are out of distribution. Zero-training fixes
   were tested and did not help (`research/evaluation/batting_domain_probe.py`):
   pose-guided cropping, action-window sampling and horizontal-flip averaging all
   land between 25% and 34% top-1, and every variant collapses onto one or two
   classes.

### How the product uses it

* Always marked experimental in the API response and the UI.
* Displayed confidence is capped at 0.65 and the measured benchmark is shown next
  to the label, so a 99% softmax score can never read as a 99% claim.
* The label becomes `unknown` when the score is below 0.55 or the margin to the
  runner-up is below 0.12.
* Coaching text is generated from pose measurements only. Shot names never appear
  as evidence in coaching.
* Classification is skipped entirely when the pose/action quality gate fails.

### Provenance and licensing

* Source repository licence: MIT.
* Weights: published in that repository under the same licence.
* Training data: announced as sourced from CricShotClassify (Sen et al.,
  *Sensors* 21(8):2846, 2021). Dataset redistribution and commercial-use terms are
  **unverified**. Treat this artifact as a research/demo baseline, not as a
  commercially cleared component.
* Conversion: Keras HDF5 → ONNX, validated against a reconstructed TensorFlow
  graph (max absolute output difference 1.06 × 10⁻⁹). The unused normalisation
  initializer was removed and the outputs re-verified as bit-identical.

### Upgrade path

A defensible replacement needs a video model trained on phone-like footage with a
permissive licence, evaluated on session-grouped splits. Candidate public data:
CricShot10k (Dihan et al., IEEE Access 2026). Until then this model stays a
labelled hint.

## Bowling family (transparent prototype, experimental)

No redistributable bowling video model passed the evaluation gate, so the product
does not ship a trained bowling classifier. Instead:

* Artifact: none. Spec: `bowling_prototype.json`.
* Deterministic scoring over pose signals: peak wrist speed, arm-path range,
  action duration, shoulder rotation.
* Reports broad pace vs spin and the detected bowling arm only.
* Never distinguishes off-spin from leg-spin, and never makes a bowling-legality
  judgement.
* Confidence is capped at 0.69 and always flagged experimental.
* No accuracy figure is published because no labelled, independent bowling test
  set exists for this heuristic. Publishing one would be inventing evidence.

Neutral bowling measurements remain available as coaching statistics, including
estimated elbow angle at release. That is a movement observation, not a legality
assessment, and the API contains no legality field or verdict.

## Pose model

`pose_landmarker_lite.task` is the MediaPipe Pose Landmarker Lite model
(33 landmarks with estimated depth, up to two poses). Pose-derived values are
approximate single-camera estimates: not laboratory motion capture, not medical
measurement, and not ball tracking.
