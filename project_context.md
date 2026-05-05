# HDDS Project Context

This file is the handoff note for the Hybrid Drone Detection System project.
It is meant so another teammate, on another PC, can continue the work without
having to reconstruct the whole chat history.

## Project Goal

Build a Python-only hybrid drone detection system for the graduation project.
The final thesis direction we settled on is:

- radar-only simulation as the core baseline
- then multimodal fusion with vision and audio to reduce false alarms
- no hardware dependency
- all work kept reproducible in Python

## Important Repository Areas

- [Project v1](./Project%20v1)
- [Review](./Review)
- [Plans](./Plans)
- [archive](./archive)
- [archived legacy repo cleanup](./archive/repo_cleanup_20260505)

## What We Did, In Order

### 1. Reviewed the legacy project

We first inspected the older repository material and identified the strongest
and weakest parts of the original work.

Strong parts:

- radar simulation work in the old signal/radar notebooks
- OFDM-like waveform and range-Doppler processing
- CFAR-based target detection
- scenario-based experiments

Weak parts:

- the repo mixed unrelated directions
- radar, audio ML, GUI, and YOLO were not organized as one thesis
- it was hard to defend as a single coherent graduation project

Artifacts created:

- [Review V1-0.md](./Review/Review%20V1-0.md)
- [Plan V1-0.md](./Plans/Plan%20V1-0.md)

### 2. Restarted in a clean project directory

We created a new clean implementation in:

- [Project v1](./Project%20v1)

Important rule:

- do not delete the old work
- keep old material available for reference
- build the new system as a clean separate path

### 3. Built the radar simulation stack

The new radar code lives in:

- [Project v1/src/radar_sim](./Project%20v1/src/radar_sim)

Main modules:

- `constants.py`
- `geometry.py`
- `waveform.py`
- `channel.py`
- `processing.py`
- `detection.py`
- `metrics.py`
- `plotting.py`
- `scenarios.py`
- `runner.py`
- `tui.py`
- `realtime.py`
- `logging_utils.py`

Main capabilities:

- OFDM-like radar waveform generation
- reference and surveillance channel simulation
- delay and Doppler target injection
- range-Doppler processing
- CA-CFAR target detection
- scenario evaluation and truth matching

### 4. Added the first notebook

We created the passive radar notebook:

- [passive_radar_scenarios_v1.ipynb](./Project%20v1/notebooks/passive_radar_scenarios_v1.ipynb)

It explains:

- the radar assumptions
- the signal model
- the processing chain
- the configured scenarios
- the numeric outputs
- the plots for each scenario

### 5. Added a terminal UI

We added a prompt-driven TUI:

- [Project v1/src/radar_sim/tui.py](./Project%20v1/src/radar_sim/tui.py)

What it does:

- asks for a custom scenario name
- asks for noise and clutter levels
- asks for one or more targets
- accepts distance, speed, and amplitude
- runs the scenario
- prints numeric results
- shows plots

### 6. Added realtime moving-target radar mode

We added:

- [Project v1/src/radar_sim/realtime.py](./Project%20v1/src/radar_sim/realtime.py)

What it does:

- creates a scenario with moving targets
- updates the target range over time
- recomputes the simulation frame by frame
- prints live numeric output
- updates live plots

This was the user-facing behavior we wanted:

- "put a drone X meters far"
- "let it move closer at speed V m/s"
- "show changes numerically and graphically"

### 7. Reduced stationary false alarms

We added a velocity gate in CFAR detection so near-zero Doppler returns are
filtered out.

Files changed:

- [Project v1/src/radar_sim/constants.py](./Project%20v1/src/radar_sim/constants.py)
- [Project v1/src/radar_sim/detection.py](./Project%20v1/src/radar_sim/detection.py)
- [Project v1/configs/default.yaml](./Project%20v1/configs/default.yaml)

Current default:

- `min_abs_velocity_mps: 1.0`

Meaning:

- detections with `|velocity| < 1.0 m/s` are suppressed

### 8. Added automatic run logging

We made every main run create a timestamped text file.

Files:

- [Project v1/src/radar_sim/logging_utils.py](./Project%20v1/src/radar_sim/logging_utils.py)
- [Project v1/tests/conftest.py](./Project%20v1/tests/conftest.py)

Log location:

- [Project v1/results/logs](./Project%20v1/results/logs)

Log naming pattern:

- `YYYYMMDD_HHMMSS_runner_<name>.txt`
- `YYYYMMDD_HHMMSS_tui_<name>.txt`
- `YYYYMMDD_HHMMSS_realtime_<name>.txt`
- `YYYYMMDD_HHMMSS_pytest_tests.txt`
- `YYYYMMDD_HHMMSS_audio_video_test_<video>.txt`

### 9. Added multimodal roadmap

We wrote the next phase plan for adding vision and audio to reduce false alarms.

Key idea:

- one fused system
- not three unrelated detectors
- the chain should be:
  - ingestion and timestamps
  - per-modality inference
  - temporal alignment
  - fusion
  - tracking and final alert logic

Artifacts:

- [Plan V1-1.md](./Plans/Plan%20V1-1.md)

### 10. Added Obsidian documentation

We also documented the project in the Obsidian vault under:

- `~/Documents/my-wiki/20 Projects/HDDS`

Important notes created:

- `HDDS.md`
- `Overview.md`
- `Current Status.md`
- `System Architecture.md`
- `Multimodal Fusion Roadmap.md`
- `Dashboard.md`
- `Execution Checklist.md`
- `Project V1 Codebase.md`
- `Plans/Plan V1-0.md`
- `Plans/Plan V1-1.md`
- `Reviews/Review V1-0.md`

Legacy notes were archived instead of deleted.

### 11. Chose vision datasets and wrote vision training notebook

For drone vision detection, the recommended dataset stack was:

- Anti-UAV300
- Drone-vs-Bird Challenge
- DroneSwarms

We created a Google Colab notebook:

- [vision_module_training_colab_v1.ipynb](./Project%20v1/notebooks/vision_module_training_colab_v1.ipynb)

It includes:

- cloning the YOLO repo / Ultralytics path
- dataset acquisition cells
- Drive-based preparation
- YOLO-style conversion
- stage-wise training

### 12. Chose sound datasets and wrote audio training notebook

For drone sound detection, the recommended dataset stack was:

- DADS
- ESC-50 as the background/no-drone set

We created a Google Colab notebook:

- [sound_module_training_colab_v1.ipynb](./Project%20v1/notebooks/sound_module_training_colab_v1.ipynb)

It includes:

- downloading data inside Colab session disk
- extracting materialized WAV subsets
- MFCC baseline model
- YAMNet embedding model
- evaluation and export

### 13. Added audio inference on video files

The trained audio artifacts were originally placed in:

- [Project v1/src/audio_nodule](./Project%20v1/src/audio_nodule)

They have now been copied into the corrected runtime package:

- [Project v1/src/audio](./Project%20v1/src/audio)

Important files:

- `sound_baseline_mfcc_logreg.joblib`
- `sound_yamnet_hgb.joblib`
- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`
- `sound_module_trained.ipynb`
- `video_test.py`

We added a video test tool:

- [Project v1/src/audio/video_test.py](./Project%20v1/src/audio/video_test.py)

The old `audio_nodule.video_test` path remains as a compatibility wrapper.

What it does:

- extracts audio from a video with `ffmpeg`
- resamples to 16 kHz mono
- slices audio into 3-second windows by default to match training
- runs the trained models
- prints per-segment probabilities
- plots the probability curve
- writes a timestamped log
- can apply optional M/N temporal confirmation
- exposes audio predictions as fusion-ready events

### 14. Fixed package conflicts

We updated:

- [Project v1/requirements.txt](./Project%20v1/requirements.txt)

The main dependency issue was:

- `opencv-python 4.12.x` requires `numpy >= 2`
- the audio stack and saved artifacts were not happy with that combination

We resolved it by pinning a compatible stack:

- `numpy==1.26.4`
- `opencv-python-headless==4.10.0.84`
- `scikit-learn==1.6.1`

We also added packages used by the notebooks and audio/video code:

- `joblib`
- `librosa`
- `pandas`
- `soundfile`
- `tensorflow`
- `tensorflow-hub`
- `tqdm`
- `seaborn`
- `datasets[audio]`
- `gdown`
- `ultralytics`

### 15. Debugged audio model loading issues

When testing a video file, the YAMNet classifier `.joblib` failed to unpickle
because it had been serialized with a different `numpy/scikit-learn` combo.

Observed error:

- `ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module`

What we changed:

- audio inference now falls back to the baseline model if YAMNet loading fails
- the package `__init__.py` no longer imports `video_test` eagerly
- the inference report now includes model-load notes
- primary audio runtime code now lives in `src/audio`
- `src/audio_nodule` is only a legacy compatibility wrapper
- audio code is split into preprocessing, feature extraction, classification,
  reporting, manifest loading, calibration helpers, and persistence logic

### 16. Current audio test command

Current command to test a video:

```bash
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4 --model both
```

If YAMNet fails to load, use:

```bash
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4 --model baseline
```

To suppress isolated audio spikes:

```bash
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4 --model baseline --confirm-m 3 --confirm-n 5
```

## Current State Of The Project

### Radar side

- core radar simulation is implemented
- custom scenario TUI works
- realtime moving-target mode works
- stationary false alarms are filtered by a minimum velocity gate
- run logs are written automatically

### Audio side

- trained artifacts exist in `src/audio/models`
- `src/audio_nodule` remains only as a legacy compatibility path
- baseline inference on video is wired
- YAMNet inference may need re-exporting if the current artifact remains incompatible
- logs are written for video tests
- audio unit tests exist in `tests/test_audio_module.py`

### Vision side

- dataset and training notebook exist
- the next code step is to build the runtime `vision/` module inside `Project v1`

### Multimodal side

- the fusion direction is defined
- the intended next architecture is radar + vision + audio with temporal alignment

### Multistatic radar map prototype

- standalone radar-only prototype added in `archive/repo_cleanup_20260505/radar_multistatic_map/`
- uses Leaflet with OpenStreetMap tiles for the map view
- simulates one source tower, multiple surveillance antennas, and one drone
- estimates drone location from noisy bistatic range measurements
- source, drone truth, and surveillance antenna locations/parameters are editable
- this is not integrated into the main dashboard yet

## Key File Paths To Know

- [archived radar_multistatic_map](./archive/repo_cleanup_20260505/radar_multistatic_map)
- [Project v1/README.md](./Project%20v1/README.md)
- [Project v1/docs/methodology.md](./Project%20v1/docs/methodology.md)
- [Project v1/configs/default.yaml](./Project%20v1/configs/default.yaml)
- [Project v1/results/tuiResults.txt](./Project%20v1/results/tuiResults.txt)
- [Project v1/results/logs](./Project%20v1/results/logs)
- [Project v1/tests](./Project%20v1/tests)
- [Project v1/notebooks](./Project%20v1/notebooks)
- [Project v1/src/radar_sim](./Project%20v1/src/radar_sim)
- [Project v1/src/audio](./Project%20v1/src/audio)
- [Project v1/src/audio_nodule](./Project%20v1/src/audio_nodule)

## Known Caveats

- This project is simulation-first; there is no hardware dependency.
- The audio YAMNet classifier artifact may still need to be re-exported if it
  keeps failing in the local inference environment.
- The radar detection threshold for stationary objects is configurable and may
  need tuning if slow drones get filtered too aggressively.
- The notebooks are designed for Colab or a local Jupyter environment with the
  required packages installed.

## Recommended Next Steps

1. Re-export `sound_yamnet_hgb.joblib` in the current environment if you need
   full baseline + YAMNet inference on videos.
2. Implement the runtime `vision` module inside `Project v1`.
3. Add a unified fusion layer that combines radar, vision, and audio scores.
4. Add tracking and temporal confirmation so a single noisy frame does not
   trigger an alert.
5. Keep writing logs for every experiment so debugging stays reproducible.

## Short Version For New Teammates

If someone wants to continue quickly:

- read [Project v1/README.md](./Project%20v1/README.md)
- inspect [Project v1/src/radar_sim](./Project%20v1/src/radar_sim)
- check [Project v1/src/audio](./Project%20v1/src/audio)
- open the notebooks in [Project v1/notebooks](./Project%20v1/notebooks)
- use the logs in [Project v1/results/logs](./Project%20v1/results/logs)
