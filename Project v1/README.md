# Project v1

Python-only hybrid drone detection project for the graduation thesis.

## Scope

This version keeps radar as the technical core and adds offline audio inference
as the first multimodal branch:

- OFDM-like illuminator generation
- Reference and surveillance channel simulation
- Delay and Doppler target injection
- Range-Doppler processing
- CA-CFAR target detection
- Scenario-based evaluation
- Video-file audio extraction and drone-audio scoring
- Timestamped audio windows for later multimodal fusion

This project does **not** depend on hardware or GUI code. Vision and audio are
treated as offline file-based branches for the multimodal roadmap, not as
hardware integrations.

## Status

This is the first clean implementation pass created from the old repository material.

## Directory Layout

```text
Project v1/
├── src/radar_sim/
├── src/audio/
├── configs/
├── docs/
└── requirements.txt
```

Archived support material such as tests, notebooks, and previous run logs was
moved to `../archive/repo_cleanup_20260505/project_v1_trim/Project v1/`.

## Setup

Create a virtual environment, then install:

```bash
pip install -r requirements.txt
```

Use Python `3.10` or `3.11` for the full audio stack. TensorFlow may not support
newer Python versions immediately.

The audio-video command also needs `ffmpeg` installed as a system dependency and
available on `PATH`.

## Run A Scenario

From inside `Project v1`:

```bash
PYTHONPATH=src python -m radar_sim.runner --scenario single_slow --no-plots
```

To show plots:

```bash
PYTHONPATH=src python -m radar_sim.runner --scenario two_targets
```

## Run The Interactive TUI

From inside `Project v1`:

```bash
PYTHONPATH=src python -m radar_sim.tui
```

The TUI will ask for:

- scenario name,
- noise and clutter levels,
- one or more targets,
- distance,
- speed,
- and target strength.

Then it runs the scenario and shows:

- numeric truth/detection results in the terminal,
- and matplotlib plots for the range-Doppler and CFAR outputs.

## Run The Realtime Moving-Target Simulator

From inside `Project v1`:

```bash
PYTHONPATH=src python -m radar_sim.realtime
```

This mode asks for:

- initial distance,
- closing speed,
- simulation duration,
- refresh interval,
- and one or more moving targets.

Then it:

- updates the target range over time,
- prints live numeric status in the terminal,
- and updates the plots frame by frame.

## Run Tests

```bash
PYTHONPATH=src pytest -q
```

## Test The Trained Audio Module On A Video

If you have trained audio models in `src/audio/models/`, you can test a video
file directly:

```bash
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4
```

Useful options:

```bash
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4 --model both --window-s 3.0 --hop-s 0.5
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4 --model baseline --no-plot
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4 --threshold 0.60
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4 --baseline-weight 2.0 --yamnet-weight 1.0
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4 --confirm-m 3 --confirm-n 5
```

What it does:

- extracts audio from the video with `ffmpeg`
- resamples to `16 kHz` mono
- slices audio into time windows
- runs the trained baseline MFCC model and/or the trained YAMNet-based model
- prints per-segment drone probabilities
- plots probability versus time
- saves a timestamped text log in `results/logs/`

Notes:

- `--model both` combines available baseline and YAMNet probabilities with explicit weights
- `--confirm-m` and `--confirm-n` apply M/N temporal confirmation to suppress isolated audio spikes
- the YAMNet path needs `tensorflow` and `tensorflow-hub`
- the first YAMNet run may download the hub model into the local cache
- if the YAMNet artifact fails to unpickle, re-export `src/audio/models/sound_yamnet_hgb.joblib` in the pinned environment or run `--model baseline`
- `configs/audio.yaml` records the current runtime defaults and calibration placeholders

## Run Logs

Runtime logs are still written to `Project v1/results/logs/` when audio or radar
commands are run. Historical logs were archived during repository cleanup.

Every `runner`, `tui`, `realtime`, `pytest`, and `audio_video_test` session now writes a timestamped text log to:

```text
Project v1/results/logs/
```

Filename pattern:

```text
YYYYMMDD_HHMMSS_<run_type>_<name>.txt
```

Examples:

- `20260502_154210_runner_single_slow.txt`
- `20260502_154955_tui_custom_scenario.txt`
- `20260502_160021_realtime_test_approach.txt`
- `20260502_160844_pytest_tests.txt`

## Technical Notes

- The model is simulation-only.
- The reported range axis is treated as bistatic range excess derived from delay.
- Velocity is derived from Doppler using a carrier-frequency assumption.
- The current implementation uses an OFDM-like waveform, not a strict LTE stack.

See [docs/methodology.md](/home/mo/dev/python/HDDS2/Project%20v1/docs/methodology.md) for the modeling assumptions.
