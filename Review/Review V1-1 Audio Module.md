# Review V1-1 Audio Module

## Audio Module Review

**Date:** 2026-05-02  
**Scope reviewed:** `Project v1/src/audio_nodule/`, `Project v1/README.md`, `Project v1/requirements.txt`, `Project v1/notebooks/sound_module_training_colab_v1.ipynb`, `Project v1/results/logs/`, `project_context.md`, `Plans/Plan V1-0.md`, and `Plans/Plan V1-1.md`.

## Post-Review Fix Status

The main engineering issues from this review have been addressed after the
initial review:

1. Primary runtime code now lives in `Project v1/src/audio/`.
2. `audio_nodule` remains only as a compatibility wrapper.
3. The one-file prototype was split into preprocessing, features, classifier,
   report, logging, manifest, calibration, persistence, schema, and CLI modules.
4. Model artifacts were copied into `Project v1/src/audio/models/`.
5. Manifests were copied into `Project v1/src/audio/data/` with remapping support.
6. Optional M/N temporal confirmation was added.
7. A fusion-ready `AudioFusionEvent` schema was added.
8. Audio unit tests were added in `Project v1/tests/test_audio_module.py`.
9. The README and project context now point to `python -m audio.video_test`.
10. YAMNet still needs artifact re-export if the saved classifier remains
    incompatible with the local NumPy/scikit-learn stack.

## Executive Verdict

The new audio module is a useful first runtime bridge from the sound-training notebook to a testable video-file workflow. It can extract audio from a video, slice it into timestamped windows, run a trained baseline classifier, print per-window probabilities, plot a probability curve, and save a timestamped log.

However, it is not yet a stable production-quality project module. It is currently closer to an experiment script than the audio branch described in `Plan V1-1`. The main risks are:

1. The package is named `audio_nodule`, which appears to be a typo and weakens project professionalism.
2. The runtime module is concentrated in one 484-line `video_test.py` file instead of the planned `audio/preprocess.py`, `audio/features.py`, and `audio/classifier.py` structure.
3. The YAMNet artifact does not currently load in the local environment, so `--model both` silently degrades to baseline-only inference.
4. There are no audio unit tests or regression tests.
5. The active shell environment cannot run the project because required packages such as `numpy`, `joblib`, `librosa`, `tensorflow`, and `pytest` are not installed.
6. The training manifests contain Colab absolute paths, so they document the split but are not directly reusable on another machine.

The module should be kept, but it needs restructuring and validation before it becomes part of the final fused radar + vision + audio system.

## What The Module Does Well

1. **It gives the audio branch a real runtime entrypoint.**  
   `PYTHONPATH=src python -m audio_nodule.video_test /path/to/video.mp4` is simple enough for repeated experiments.

2. **It works on video files, not only WAV files.**  
   The use of `ffmpeg` makes the module practical for the offline multimodal direction in `Plan V1-1`, where video and audio are expected to come from files.

3. **It uses time windows, which is the correct shape for fusion.**  
   `SegmentPrediction` includes `start_s` and `end_s`, so the module is already close to the temporal-alignment requirements.

4. **It records model availability.**  
   `LoadedAudioModels` tracks whether baseline and YAMNet are enabled, and the generated reports include load notes.

5. **It writes reproducible logs.**  
   Audio test logs are saved under `Project v1/results/logs/` using the shared radar logging utility. This matches the project habit of keeping experiment outputs inspectable.

6. **It has a reasonable baseline feature set.**  
   The MFCC, delta, delta-delta, spectral, zero-crossing, RMS, and spectral-contrast features are a defensible classical audio baseline.

7. **It handles the known YAMNet serialization problem without crashing when baseline is available.**  
   The fallback behavior keeps demos running while clearly reporting that YAMNet was disabled.

## Major Issues

1. **The package name should be fixed before the project spreads further.**  
   `audio_nodule` should almost certainly be `audio_module` or simply `audio`. The typo appears in code paths, README commands, logs, and project context. Keeping it will make the final project look careless.

2. **The module structure does not match the multimodal plan.**  
   `Plan V1-1` recommends:

   ```text
   src/audio/
   ├── preprocess.py
   ├── features.py
   ├── classifier.py
   └── models/
   ```

   The current implementation puts extraction, feature engineering, model loading, scoring, plotting, CLI parsing, and logging into `video_test.py`. This is acceptable for a prototype, but not for the final v1 architecture.

3. **YAMNet is currently not operational locally.**  
   Existing logs show:

   ```text
   YAMNet classifier could not be loaded from sound_yamnet_hgb.joblib:
   <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module.
   ```

   This means the claimed `both` model mode is not currently true in practice. The system is running baseline-only unless the YAMNet classifier is re-exported in the current dependency stack.

4. **The audio probabilities are not calibrated for fusion.**  
   The code averages baseline and YAMNet scores when both are available. `Plan V1-1` explicitly warns against uncalibrated score averaging. Before fusion, the audio branch needs documented calibration or at least an empirical threshold-selection procedure.

5. **There is no audio testing layer.**  
   The project has radar tests, but no tests for:

   - audio slicing edge cases,
   - feature-vector length,
   - baseline model loading,
   - YAMNet fallback behavior,
   - report formatting,
   - CLI argument validation,
   - no-audio-video failure behavior.

6. **`video_test.py` imports TensorFlow and TensorFlow Hub at module import time.**  
   This makes even baseline-only use pay the TensorFlow import cost and fail if TensorFlow is not installed. For a lightweight baseline path, YAMNet imports should be lazy and only happen when `--model yamnet` or `--model both` needs them.

7. **The README scope is now inconsistent.**  
   `Project v1/README.md` still says v1 does not depend on audio classification, but later documents the audio video test command. The project has moved beyond the radar-only scope, so the README should explicitly state that radar is the core and audio is an added multimodal branch.

8. **The training manifests are not portable.**  
   Manifest paths point to `/content/hdds_audio/...`, which documents the Colab training environment but cannot be replayed locally without path remapping.

9. **There is no common fusion event schema yet.**  
   `SegmentPrediction` is useful, but it is still audio-specific. The fusion plan needs an event shape that can carry `t_start`, `t_end`, `audio_score`, model availability, features, and explanation fields.

10. **The module relies on external `ffmpeg` without environment documentation beyond the README note.**  
    `ffmpeg` is necessary for video extraction but is not installable through `requirements.txt`. The setup docs should call it a system dependency.

## Evidence From Current Logs

Two saved audio runs exist:

1. `20260502_130834_audio_video_test_shahid_strike.txt`
2. `20260502_130938_audio_video_test_bolt_m.txt`

Both runs show:

1. baseline enabled,
2. YAMNet disabled,
3. the same artifact load error,
4. per-segment probability logging working.

The first run produced 14 segments and one positive segment. The second produced 265 segments and zero positive segments. This confirms the baseline inference path and logging path are usable, but it does not validate detection quality because the videos, labels, and expected outcomes are not documented in the project.

## Verification Attempt

I attempted to run the existing tests from `Project v1`:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m pytest -q
```

Both were blocked because the active Python environment does not have `pytest` installed. The active `python` is `3.14.4`, and imports for `numpy`, `librosa`, `joblib`, `sklearn`, `tensorflow`, `tensorflow_hub`, and `pytest` all fail in this shell.

This does not prove the module is broken, but it means the current checkout is not immediately reproducible without creating the intended environment from `requirements.txt`.

## Recommended Fix Plan

### Phase 1: Stabilize Naming And Structure

1. Rename `src/audio_nodule` to `src/audio` or `src/audio_module`.
2. Keep a temporary compatibility wrapper only if old commands must keep working.
3. Split `video_test.py` into:
   - `preprocess.py` for `ffmpeg`, loading, resampling, slicing,
   - `features.py` for baseline features and YAMNet embeddings,
   - `classifier.py` for artifact loading and probability scoring,
   - `video_test.py` for CLI orchestration only.
4. Move model artifacts into `src/audio/models/`.

### Phase 2: Fix Runtime Reliability

1. Re-export `sound_yamnet_hgb.joblib` using the pinned stack in `requirements.txt`.
2. Make TensorFlow and TensorFlow Hub lazy imports so baseline-only inference remains lightweight.
3. Add explicit validation for `window_s`, `hop_s`, and `threshold`.
4. Add clearer errors for videos with no audio stream.
5. Document `ffmpeg` as a system dependency.

### Phase 3: Add Tests

Minimum tests:

1. `slice_audio` returns the expected windows for short, exact-length, and long audio.
2. baseline feature extraction returns the expected fixed feature length.
3. model fallback records a load note when YAMNet fails and baseline succeeds.
4. `build_text_report` includes segment count, model flags, positive count, and per-segment lines.
5. CLI parser accepts `baseline`, `yamnet`, and `both`.

These tests can use tiny synthetic NumPy arrays and mock model objects, so they should not require real videos or TensorFlow downloads.

### Phase 4: Prepare For Fusion

1. Add an audio result schema that maps cleanly into the planned `UnifiedDetectionEvent`.
2. Return one audio score per common fusion time window.
3. Store model availability flags and load notes as metadata, not just text logs.
4. Add an M/N persistence option for audio positives so one isolated spike does not become a fused alert.
5. Choose and document threshold calibration using validation-set results.

## Final Assessment

The audio module is a good prototype and should remain in `Project v1`, but it should not be presented as a finished branch yet. The baseline path is wired, timestamped inference exists, and logs are useful. The work needed now is engineering discipline: fix the package name, split responsibilities, repair the YAMNet artifact, add tests, and expose a fusion-ready event format.

Once those issues are handled, the module can credibly support the larger thesis claim from `Plan V1-1`: audio should reduce radar false alarms by adding persistent drone-acoustic evidence on a shared timeline.
