# Audio Module

## Role In Project V1

The audio module is an offline file-based branch for the multimodal roadmap. It
does not replace the radar simulator. It produces timestamped drone-audio
scores that can later be aligned with radar and vision outputs.

## Runtime Package

Primary package:

```bash
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4
```

Legacy wrapper:

```bash
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4
```

Use the primary package for new work.

## Structure

```text
src/audio/
├── preprocess.py
├── features.py
├── classifier.py
├── persistence.py
├── calibration.py
├── manifest.py
├── report.py
├── video_test.py
├── models/
└── data/
```

## Dependencies

The audio stack needs the Python packages pinned in `requirements.txt` and the
system binary `ffmpeg` on `PATH`.

Use Python `3.10` or `3.11` for the full TensorFlow/YAMNet path.

## Model Notes

`--model baseline` uses the MFCC-style baseline artifact.

The baseline was trained on `3.0 s` clips, so the CLI default window length is
also `3.0 s`. Using much shorter windows can reduce detection confidence because
the MFCC/statistical feature distribution no longer matches training.

`--model yamnet` and `--model both` use TensorFlow Hub YAMNet embeddings. If the
YAMNet classifier fails with a NumPy or scikit-learn unpickle error, re-export
`src/audio/models/sound_yamnet_hgb.joblib` in the pinned environment.

## Fusion Notes

Each `AudioSegmentPrediction` can be converted into an `AudioFusionEvent` with:

```python
event = prediction.to_fusion_event()
```

For final fusion, calibrate thresholds on validation data before comparing audio
scores with radar or vision scores. The helper `audio.calibration.choose_threshold_by_f1`
is a baseline threshold-selection utility, not a replacement for a full
calibration study.

Use `--confirm-m` and `--confirm-n` to apply M/N temporal confirmation when a
single isolated audio spike should not count as a confirmed alert.
