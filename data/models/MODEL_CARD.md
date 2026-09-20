# CreaseLab model card

## Batting video classifier

- Artifact: `batting_video.onnx`
- SHA-256: `d88a2811de83172d5aee6fdcf714ecf87ed14879e3b4356742c0f224c100ab7b`
- Architecture: EfficientNetB0 frame encoder followed by two GRU layers
- Input: 30 RGB frames, padded to 224 × 224
- Output: cover drive, defence, flick, hook, late cut, lofted shot, pull,
  square cut, straight drive, and sweep
- Source implementation and weights:
  [RITIK-12/CricketShotClassification](https://github.com/RITIK-12/CricketShotClassification)
- Source code repository license: MIT
- Source-reported accuracy: 94%; this figure has not yet been independently
  reproduced by CreaseLab and is not presented as a deployment benchmark.

The model was converted from its original Keras HDF5 weights to ONNX and its
outputs were compared against the reconstructed TensorFlow graph on a real
clip. Maximum absolute output difference was below `1.1e-9`.

### Limitations

The training material is broadcast-style cricket footage. Phone recordings,
unusual camera angles, partial bodies, indoor nets, occlusion, and non-batting
movements can be out of distribution. The application therefore reports the
model score as confidence, keeps a pose/action gate, and never treats that score
as a guarantee of correctness.

## Bowling classifier

The previous project-specific Random Forest is not used. Until a redistributable
bowling video model passes the evaluation gate, the application exposes a
transparent, deliberately confidence-capped pose prototype for broad pace vs
spin feedback. Its result is always marked experimental. It does not make any
bowling-legality judgement.

## Pose model

`pose_landmarker_lite.task` is the MediaPipe Pose Landmarker Lite model. Pose
measurements are 2D/estimated-depth coaching signals, not lab measurements.
