# Detailed Research Report For The Radar, Vision, And Sound Modules

## Abstract

This document records, in research-paper style, what was implemented for the
Hybrid Drone Detection System (HDDS) repository at the current stage of the
project. It is intentionally detailed and written for teammates who need to
understand not only what each module does, but also where the code lives, how
the data flows, which assumptions were encoded, and which implementation
details may affect future work.

The project currently has two distinct layers:

1. A cleaned academic package under `Project v1/`, where the radar module is
   the technical core and the sound module is the first clean multimodal branch.
2. An integrated dashboard under `App/`, where radar, sound, and vision are
   synchronized for scenario playback and simple decision fusion.

This distinction is important. The cleaned `Project v1` package is the best
starting point for reproducible thesis work. The `App/` layer is the practical
integration surface that still contains some legacy and compatibility behavior,
especially for vision and audio.

## Repository Scope And What Was Actually Built

At the time of writing, the main relevant paths are:

```text
Project v1/
├── configs/
├── docs/
├── src/radar_sim/
└── src/audio/

App/
├── main.py
└── scenario_loader.py
```

The implemented work can be summarized as follows:

- The radar module is a software-only passive-radar-inspired simulation chain.
- The sound module is an offline inference pipeline that extracts audio from
  video, slices it into windows, computes learned scores, and emits
  timestamped predictions.
- The vision module is currently implemented inside the integrated app, where a
  YOLOv5 detector is applied to video frames and the results are synchronized
  with audio and radar status.
- The app performs a simple fusion rule over the three modalities.

The remainder of this document is organized as three detailed reports:

1. Radar Module Report
2. Vision Module Report
3. Sound Module Report

---

## Report I: Radar Module

### 1. Research Goal

The radar module was developed as the most physically grounded part of the
project. Its purpose is not to drive a real radar sensor, but to reproduce the
core detection physics in software so that the project can be defended as a
signal-processing system before hardware integration.

The radar branch studies:

- reference waveform generation,
- surveillance-channel synthesis,
- delay and Doppler target injection,
- direct-path and clutter contamination,
- range-Doppler formation,
- CA-CFAR detection,
- scenario-based validation and reporting.

This module lives in `Project v1/src/radar_sim/`.

### 2. Files And Responsibilities

The radar implementation is split cleanly by responsibility:

- `constants.py`: radar, CFAR, scenario, and application configuration objects.
- `waveform.py`: OFDM-like reference waveform generation.
- `channel.py`: surveillance-channel simulation, including direct path, clutter,
  target echoes, and noise.
- `processing.py`: cyclic-prefix removal, range processing, clutter suppression,
  and range-Doppler map formation.
- `geometry.py`: conversions between delay, range, Doppler, and velocity.
- `detection.py`: 2D CA-CFAR thresholding and compact detection extraction.
- `metrics.py`: truth generation and truth-matching.
- `runner.py`: scenario execution, report printing, and logging.
- `scenarios.py`: scenario lookup from configuration.
- `realtime.py`: dynamic moving-target simulation mode.
- `tui.py`: interactive scenario configuration.

This is a strong design choice because it separates physics, configuration,
processing, and presentation instead of mixing them into one script.

### 3. Radar Configuration And Operating Assumptions

The default operating parameters are stored in
`Project v1/configs/default.yaml`. The important defaults are:

```yaml
radar:
  carrier_frequency_hz: 2400000000.0
  sample_rate_hz: 20000000.0
  pri_s: 0.00025
  num_pulses: 128
  num_subcarriers: 256
  cyclic_prefix: 64

cfar:
  guard_cells_range: 2
  guard_cells_doppler: 2
  train_cells_range: 8
  train_cells_doppler: 8
  pfa: 1.0e-5
  min_abs_velocity_mps: 1.0
```

These parameters define the simulated sensing region:

- carrier frequency is assumed to be `2.4 GHz`,
- wavelength is derived from the speed of light and carrier frequency,
- slow-time sampling is controlled by `PRI`,
- fast-time resolution is controlled by `sample_rate_hz`,
- the OFDM-like symbol uses `256` subcarriers plus a `64`-sample cyclic prefix,
- the CFAR detector suppresses near-zero-velocity bins below `1.0 m/s`.

That last point matters. The detector is intentionally biased against
stationary or nearly stationary returns in order to reduce clutter-like alarms.

### 4. Waveform Generation

The reference waveform is generated in `waveform.py`. The core implementation
is short and clear:

```python
def generate_qpsk_symbols(num_subcarriers: int, rng: np.random.Generator) -> np.ndarray:
    bits_i = rng.choice((-1.0, 1.0), size=num_subcarriers)
    bits_q = rng.choice((-1.0, 1.0), size=num_subcarriers)
    return (bits_i + 1j * bits_q) / np.sqrt(2.0)


def generate_ofdm_symbol(config: RadarConfig, rng: np.random.Generator) -> np.ndarray:
    freq_symbols = generate_qpsk_symbols(config.num_subcarriers, rng)
    time_symbol = np.fft.ifft(freq_symbols)
    time_symbol /= np.sqrt(np.mean(np.abs(time_symbol) ** 2))
    cyclic_prefix = time_symbol[-config.cyclic_prefix :]
    return np.concatenate((cyclic_prefix, time_symbol))
```

This code shows four important implementation decisions:

1. QPSK is used for subcarrier population.
2. The symbol is generated in frequency domain and transformed by `IFFT`.
3. The time-domain symbol is normalized to unit average power.
4. A cyclic prefix is copied from the end of the useful symbol.

The pulse matrix is then created by repeating that process across
`config.num_pulses`. In practice, this produces a reference matrix with shape:

```text
(num_pulses, samples_per_pulse)
```

### 5. Surveillance-Channel Model

The radar channel model is implemented in `channel.py`. The code below is one
of the most important excerpts in the entire radar branch:

```python
def synthesize_target_echo(
    reference_pulse: np.ndarray,
    target: Target,
    pulse_index: int,
    config: RadarConfig,
) -> np.ndarray:
    delayed = apply_fractional_delay(
        signal=reference_pulse,
        delay_s=target.delay_s,
        fs_hz=config.sample_rate_hz,
    )
    time_axis = np.arange(reference_pulse.size) / config.sample_rate_hz
    slow_time = pulse_index * config.pri_s
    phase = np.exp(1j * 2.0 * np.pi * target.doppler_hz * (time_axis + slow_time))
    return target.amplitude_linear * delayed * phase
```

This is the target model in concrete form:

- `apply_fractional_delay(...)` produces a delayed version of the reference
  pulse, which means targets are not limited to integer-sample delays.
- the Doppler effect is modeled as a complex exponential phase progression over
  both fast time and slow time,
- the target amplitude is applied in linear scale through
  `target.amplitude_linear`.

The full surveillance channel is then assembled as:

```python
direct_path_amplitude = db_to_linear_amplitude(config.direct_path_amplitude_db)
clutter_amplitude = db_to_linear_amplitude(scenario.clutter_amplitude_db)
noise_sigma = np.sqrt(db_to_linear_power(scenario.noise_power_db) / 2.0)
```

and later:

```python
for pulse_idx in range(num_pulses):
    pulse = direct_path_amplitude * reference[pulse_idx]
    pulse = pulse + stationary_clutter

    for target in scenario.targets:
        pulse = pulse + synthesize_target_echo(...)

    noise = noise_sigma * (
        rng.standard_normal(pulse_len) + 1j * rng.standard_normal(pulse_len)
    )
    surveillance[pulse_idx] = pulse + noise
```

This means each pulse contains:

- direct-path leakage,
- a fixed stationary clutter realization,
- one or more delayed/Doppler-shifted target echoes,
- additive complex Gaussian noise.

From a thesis perspective, this is a well-structured compromise between full RF
hardware realism and mathematically controlled simulation.

### 6. Range And Doppler Processing

Signal processing is handled by `processing.py`. The most important processing
step is:

```python
def form_range_profiles(
    reference: np.ndarray, surveillance: np.ndarray, config: RadarConfig
) -> np.ndarray:
    ref_symbols = remove_cyclic_prefix(reference, config)
    surv_symbols = remove_cyclic_prefix(surveillance, config)

    ref_freq = np.fft.fft(ref_symbols, axis=1)
    surv_freq = np.fft.fft(surv_symbols, axis=1)
    matched = surv_freq / (ref_freq + config.epsilon)
    return np.fft.ifft(matched, axis=1)
```

This function is central because it reveals the passive-radar-inspired
philosophy used in the project:

- the cyclic prefix is removed first,
- both channels are transformed to frequency domain,
- the surveillance channel is divided by the reference channel,
- then the result is transformed back by `IFFT` to produce range profiles.

That division is not a generic DSP step added by habit. It specifically
attempts to suppress random subcarrier modulation so that the useful structure
left in the matched result is more directly related to delay.

After range processing, stationary content is suppressed:

```python
def suppress_stationary_components(range_profiles: np.ndarray) -> np.ndarray:
    stationary_estimate = np.mean(range_profiles, axis=0, keepdims=True)
    return range_profiles - stationary_estimate
```

This is a simple but effective clutter-cancellation step. It assumes that
stationary components remain similar across pulses and can therefore be removed
through pulse-mean subtraction.

Finally, the range-Doppler map is formed:

```python
def form_range_doppler_map(
    filtered_profiles: np.ndarray, config: RadarConfig
) -> np.ndarray:
    slow_time_window = np.hamming(config.num_pulses)[:, np.newaxis]
    fast_time_window = np.hamming(filtered_profiles.shape[1])[np.newaxis, :]
    windowed = filtered_profiles * slow_time_window * fast_time_window
    return np.fft.fftshift(np.fft.fft(windowed, axis=0), axes=0)
```

Two window functions are applied before the slow-time FFT:

- a Hamming window across pulses,
- another Hamming window across fast-time bins.

This reduces sidelobes and produces a cleaner map for CFAR thresholding.

### 7. Geometry And Physical Interpretation

The radar branch does not keep everything in abstract bins. `geometry.py`
converts them into physical quantities:

```python
def delay_to_bistatic_range_m(delay_s: float, config: RadarConfig) -> float:
    return config.speed_of_light * delay_s


def doppler_hz_to_velocity_mps(doppler_hz: float, config: RadarConfig) -> float:
    return doppler_hz * config.wavelength_m / 2.0
```

This is important for two reasons:

1. detections can be discussed in meters and meters per second rather than only
   in bin indices,
2. the system stays explainable to supervisors and teammates who care about the
   physical meaning of the outputs.

The axis builders:

- `build_range_axis_m(...)`
- `build_doppler_axis_hz(...)`
- `build_velocity_axis_mps(...)`

ensure that plots and reports use interpretable axes.

### 8. CFAR Detection

Detection is implemented in `detection.py` using 2D cell-averaging CFAR. The
core logic is:

```python
noise_estimate = signal.convolve2d(
    power_map,
    kernel / num_train,
    mode="same",
    boundary="symm",
)
alpha = num_train * ((config.pfa ** (-1.0 / num_train)) - 1.0)
threshold_map = alpha * noise_estimate
detection_mask = power_map > threshold_map
```

This produces an adaptive threshold based on local neighborhood energy rather
than a fixed global threshold. That is the right choice when clutter and noise
conditions are allowed to vary.

The detector then explicitly suppresses near-zero-velocity bins:

```python
stationary_bins = np.abs(velocity_axis_mps) < config.min_abs_velocity_mps
if np.any(stationary_bins):
    detection_mask = detection_mask.copy()
    detection_mask[stationary_bins, :] = False
```

This is a deliberate design decision. The project is not just looking for any
bright return. It is biased toward moving targets that fit the drone-detection
problem.

After thresholding, connected regions are labeled and compacted into one peak
per region:

```python
labels, num_regions = ndimage.label(detection_mask)
...
doppler_bin, range_bin = np.unravel_index(
    np.argmax(region_power), region_power.shape
)
```

This reduces fragmented masks into practical output detections with:

- range bin,
- Doppler bin,
- range in meters,
- Doppler in Hz,
- velocity in m/s,
- peak power in dB.

### 9. Scenario Execution And Evaluation

`runner.py` provides the clean non-GUI experiment path. The main execution
sequence is:

```python
reference = generate_reference_matrix(config=config.radar, rng=rng)
surveillance = simulate_surveillance_matrix(...)
processing = process_reference_and_surveillance(...)
detections = ca_cfar_2d(...)
truths = scenario_truth(scenario, config.radar)
```

This design is valuable because it gives a single, readable experiment chain
for teammates to reuse in tests or papers.

Truth matching is handled in `metrics.py`:

```python
range_ok = abs(detection.range_m - truth.range_m) <= range_tolerance_m
doppler_ok = abs(detection.doppler_hz - truth.doppler_hz) <= doppler_tolerance_hz
if range_ok and doppler_ok:
    return True
```

The current comparison tolerances are not embedded in the detector itself.
They are evaluation-layer choices, which is the correct separation of concerns.

### 10. Scenario Definitions

The default scenarios in `configs/default.yaml` include:

- `clear_sky`
- `single_slow`
- `single_fast`
- `two_targets`
- `low_snr`

This gives the radar branch both sanity cases and stress cases:

- no targets,
- single target,
- multi-target,
- weak target at lower SNR.

The integrated app also supports richer JSON-driven scenarios through
`App/scenario_loader.py`, including timeline-based motion.

### 11. Strengths, Limitations, And Teammate Notes

Strengths:

- physically interpretable delay/Doppler model,
- clear module separation,
- reproducible seeded simulation,
- adaptive CFAR detection,
- scenario and truth evaluation support,
- multiple front ends: runner, TUI, and realtime.

Limitations:

- simulation-only, not hardware-connected,
- no antenna model, no calibration chain, no RF frontend impairments,
- clutter model is simplified and stationary,
- no track-before-detect or multi-frame tracker,
- zero-velocity suppression may hide slow or hovering targets.

Teammate note:

If the thesis requires the cleanest narrative, build from `Project v1/src/radar_sim`.
If the thesis requires a multimodal demo, use the same radar package through
`App/main.py`.

---

## Report II: Vision Module

### 1. Research Goal

The vision module provides the visual detection branch of the multimodal
system. Its present role is not to be a polished standalone package; instead,
it serves as the real-time frame-level detector used by the integrated hybrid
dashboard.

This is an important architectural fact: the current vision branch is mostly an
integration-layer implementation rather than a clean `Project v1/src/vision`
package.

### 2. Where The Vision Code Actually Lives

The primary vision logic is in:

- `App/main.py`
- `App/scenario_loader.py`

The detector loads YOLOv5 weights from either the active repository root or the
archive fallback:

```python
YOLO_DIR = _first_existing_path(
    REPO_ROOT / "yolov5",
    REPO_ROOT / "archive" / "repo_cleanup_20260505" / "yolov5",
)
YOLO_WEIGHTS = _first_existing_path(
    YOLO_DIR / "best.pt",
    REPO_ROOT / "archive" / "repo_cleanup_20260505" / "yolov5" / "best.pt",
)
```

This tells teammates two things immediately:

1. the detector depends on a local YOLOv5 codebase and a trained `best.pt`
   artifact,
2. the app is resilient to repo cleanup because it can fall back to archived
   assets.

### 3. Practical Dependency Note

`App/main.py` imports:

- `cv2`
- `torch`
- `tkinter`
- `tensorflow`
- `librosa`

The cleaned `Project v1/requirements.txt` includes `opencv-python-headless` and
`ultralytics`, but the integrated app still uses `torch.hub.load(...)` with a
local YOLOv5 repo. Teammates should therefore treat the app as having stricter
runtime dependencies than the cleaned package alone.

### 4. Scenario Pairing And Input Format

The integrated app expects scenario bundles with matching stems:

- video file,
- audio file,
- radar JSON file.

That logic is implemented in `scenario_loader.py`:

```python
for stem, parts in by_stem.items():
    if {"video", "audio", "radar"} <= parts.keys():
        bundles.append(
            ScenarioBundle(
                name=stem,
                video_path=parts["video"],
                audio_path=parts["audio"],
                radar_path=parts["radar"],
            )
        )
```

This means the app is not designed to run vision alone. It is designed to
consume synchronized multimodal scenario sets.

### 5. Model Loading Strategy

At runtime, the app prepares the YOLO detector as follows:

```python
if self.yolo_model is None:
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLOv5 weights not found: {YOLO_WEIGHTS}")
    self.yolo_model = torch.hub.load(
        str(YOLO_DIR),
        "custom",
        path=str(YOLO_WEIGHTS),
        source="local",
        device="cpu",
    )
```

This shows several important implementation choices:

- a custom-trained model is expected, not only a stock pretrained one,
- the model is loaded from local files, not from the network,
- inference is forced onto CPU,
- the app fails early if the trained weights are missing.

From a deployment perspective, CPU execution is convenient for demos but can be
slower on high-frame-rate or high-resolution video.

### 6. Video Processing Loop

The app opens the selected video with OpenCV:

```python
self.cap = cv2.VideoCapture(str(self.bundle.video_path))
```

It then computes effective timing from the actual file metadata:

```python
fps = self.cap.get(cv2.CAP_PROP_FPS)
self.video_fps = fps if fps and fps > 1 else 25.0
frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
```

This is a useful robustness measure because some video files expose incomplete
metadata. The code falls back to `25.0` FPS when needed.

The per-frame processing logic is not naive. It tries to keep visual playback
aligned to wall-clock time:

```python
elapsed_s = max(0.0, time.monotonic() - self.playback_started_at)
target_frame = int(elapsed_s * self.video_fps)
if target_frame > self.frame_index + 1:
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    self.frame_index = target_frame
```

This means the system can skip ahead if processing falls behind real-time
playback. That is a practical synchronization mechanism for the multimodal
dashboard.

### 7. Vision Inference Logic

The actual visual detection routine is:

```python
def _run_vision(self, frame: np.ndarray) -> None:
    self.last_vision_boxes = []
    self.last_vision_detected = False
    if self.yolo_model is None:
        return
    results = self.yolo_model(frame)
    for *xyxy, conf, _cls in results.xyxy[0]:
        confidence = float(conf)
        if confidence < VISION_CONFIDENCE:
            continue
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        self.last_vision_boxes.append((x1, y1, x2, y2, confidence))
    self.last_vision_detected = bool(self.last_vision_boxes)
```

Here the vision module performs:

- full-frame YOLO inference,
- confidence thresholding,
- bounding-box extraction,
- a binary detection decision based on whether any valid boxes remain.

Two constants define the default behavior near the top of `App/main.py`:

```python
VISION_CONFIDENCE = 0.5
VISION_EVERY_N_FRAMES = 3
```

This means:

- only predictions with confidence at least `0.5` are accepted,
- vision inference is run every third frame unless a sync jump forces a refresh.

That second choice reduces CPU cost. The module is therefore designed for
stable multimodal playback rather than maximum per-frame detection density.

### 8. Output Rendering

Accepted detections are rendered back onto the frame:

```python
for x1, y1, x2, y2, confidence in self.last_vision_boxes:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)
    cv2.putText(
        frame,
        f"drone {confidence:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (34, 197, 94),
        2,
    )
```

This part is simple, but it has real value for teammate understanding and demo
communication. It turns model outputs into immediately visible evidence.

### 9. Relation To The Radar And Audio Branches

The vision branch does not operate in isolation. It is a synchronized component
inside a common playback loop:

- video frames come from OpenCV,
- audio decisions come from the audio predictor,
- radar status is refreshed according to scenario time,
- all three are combined in the dashboard state.

This is the correct way to think about the current vision module: it is a
multimodal decision source, not just a detector script.

### 10. Current Strengths And Weaknesses

Strengths:

- practical frame-level drone detection,
- uses trained YOLO weights rather than placeholder logic,
- synchronized with audio and radar,
- skips frames intelligently to maintain timing,
- can visualize detections directly.

Weaknesses:

- not yet cleaned into its own `Project v1/src/vision` package,
- depends on external trained weights,
- depends on local YOLOv5 repository layout,
- CPU-only in current app code,
- current binary decision rule is simply “any valid box means detected.”

Teammate note:

If the team wants a publishable software architecture, the next refactor should
extract the vision branch from `App/main.py` into a standalone module with its
own configuration, schema objects, and batch/offline test entrypoint.

---

## Report III: Sound Module

### 1. Research Goal

The sound module is the first clean multimodal branch added to `Project v1`.
Its job is to turn audio extracted from a video into timestamped drone
probabilities that can later be fused with radar and vision signals.

The module is explicitly offline and file-based. It does not stream from a live
microphone. This is consistent with the repository’s thesis direction: first
achieve a reproducible multimodal pipeline, then consider live hardware later.

### 2. Files And Responsibilities

The sound implementation lives in `Project v1/src/audio/`:

- `preprocess.py`: extraction, loading, slicing, and parameter validation.
- `features.py`: baseline feature extraction and YAMNet embeddings.
- `classifier.py`: model loading, probability prediction, and score fusion.
- `persistence.py`: M/N temporal confirmation logic.
- `schemas.py`: data structures for windows, predictions, and fusion events.
- `report.py`: textual reporting and log generation.
- `video_test.py`: main CLI entrypoint for inference on video files.
- `models/`: trained audio model artifacts.

This structure is clean and teammate-friendly. It keeps the sound branch much
more maintainable than an all-in-one notebook workflow.

### 3. Runtime Entry Point

The clean CLI entrypoint is:

```bash
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4
```

The `run_video_inference(...)` function in `video_test.py` coordinates the full
pipeline:

```python
loaded_models = load_audio_models(
    use_baseline=use_baseline,
    use_yamnet=use_yamnet,
)
y, sr = load_audio_from_video(video_path, target_sr=TARGET_SR)
windows = slice_audio(y=y, sr=sr, window_s=window_s, hop_s=hop_s)
predictions = [
    predict_window(
        window=window,
        sr=sr,
        loaded_models=loaded_models,
        threshold=threshold,
        weights=weights,
    )
    for window in windows
]
```

That block is the sound-module equivalent of the radar runner. It clearly shows
the full inference path:

1. load compatible models,
2. extract audio from video,
3. resample and slice into windows,
4. run prediction for each window,
5. optionally post-process with temporal confirmation,
6. optionally plot and log the results.

### 4. Audio Extraction And Preprocessing

Audio extraction is implemented in `preprocess.py` and relies on `ffmpeg`:

```python
command = [
    "ffmpeg",
    "-y",
    "-i",
    str(video_path),
    "-vn",
    "-ac",
    "1",
    "-ar",
    str(target_sr),
    "-c:a",
    "pcm_s16le",
    str(wav_path),
]
```

This command makes the preprocessing assumptions explicit:

- the video stream is ignored,
- audio is converted to mono,
- audio is resampled to the target rate,
- output is written as PCM WAV for stable downstream loading.

The default target sampling rate is:

```python
TARGET_SR = 16000
```

This is a reasonable choice for compact audio inference and is also compatible
with YAMNet-style workflows.

### 5. Windowing Strategy

Audio is segmented into overlapping windows by `slice_audio(...)`:

```python
window_samples = max(1, int(round(window_s * sr)))
hop_samples = max(1, int(round(hop_s * sr)))
```

The function also includes an important practical behavior:

```python
if total_duration - last_end > 0.25 * window_s:
    windows.append(
        AudioWindow(
            start_s=max(0.0, total_duration - window_s),
            end_s=total_duration,
            samples=samples[-window_samples:],
        )
    )
```

This avoids silently discarding a meaningful tail segment near the end of the
audio file.

### 6. Two Feature Paths: Baseline And YAMNet

The sound module supports two independent learned feature branches.

#### 6.1 Baseline MFCC-Style Path

`features.py` extracts classical features:

```python
mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=40)
delta = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)
```

It then aggregates statistics over:

- MFCCs,
- first derivatives,
- second derivatives,
- spectral centroid,
- spectral bandwidth,
- spectral rolloff,
- zero-crossing rate,
- RMS energy,
- spectral contrast.

This path is useful because it is lightweight, explainable, and compatible
with classical machine-learning classifiers serialized by `joblib`.

#### 6.2 YAMNet Embedding Path

The second feature path uses YAMNet embeddings:

```python
waveform = tf.convert_to_tensor(np.asarray(y, dtype=np.float32), dtype=tf.float32)
_, embeddings, _ = yamnet_model(waveform)
return tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)
```

This means the project is not tied to hand-designed features alone. It also
supports a pretrained audio representation learned at a larger scale.

### 7. Model Loading And Compatibility Behavior

The audio model loader in `classifier.py` is more sophisticated than it first
appears. It tries to load whichever model families the user requested and keeps
useful fallback notes:

```python
if use_baseline:
    baseline_model = joblib.load(baseline_path)
...
if use_yamnet:
    yamnet_classifier = joblib.load(yamnet_path)
    yamnet_model = hub.load(YAMNET_HANDLE)
```

The supported artifacts are:

- `sound_baseline_mfcc_logreg.joblib`
- `sound_yamnet_hgb.joblib`

and the YAMNet handle is:

```python
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
```

This matters operationally because the YAMNet path can fail due to environment
or artifact compatibility issues. The code already anticipates that and either
warns or raises depending on which paths remain available.

### 8. Score Fusion

The sound branch can combine the baseline and YAMNet scores:

```python
def fuse_audio_scores(
    baseline_probability: float | None,
    yamnet_probability: float | None,
    weights: dict[str, float] | None = None,
) -> float:
```

The fusion logic computes either:

- a simple mean when no weights are supplied,
- or a weighted average using user-provided weights.

This is a sensible intermediate design:

- simple enough to explain,
- flexible enough for calibration experiments,
- not pretending to be a final statistically calibrated fusion model.

### 9. Per-Window Prediction Semantics

The final prediction object is built in `predict_window(...)`:

```python
audio_score = fuse_audio_scores(
    baseline_probability=baseline_probability,
    yamnet_probability=yamnet_probability,
    weights=weights,
)
label = "drone" if audio_score >= threshold else "no_drone"
```

The corresponding schema in `schemas.py` is:

```python
@dataclass(frozen=True)
class AudioSegmentPrediction:
    start_s: float
    end_s: float
    baseline_probability: float | None
    yamnet_probability: float | None
    audio_score: float
    label: str
    model_notes: tuple[str, ...] = ()
```

This is one of the best interfaces in the repository because it preserves:

- the time interval,
- model-specific scores,
- the fused score,
- the final label,
- extra notes about model loading or post-processing.

### 10. Temporal Confirmation

The sound branch also supports M/N confirmation to suppress isolated spikes:

```python
def apply_m_of_n_confirmation(
    predictions: list[AudioSegmentPrediction],
    confirm_m: int,
    confirm_n: int,
) -> list[AudioSegmentPrediction]:
```

The core rule is:

```python
window = predictions[max(0, idx - confirm_n + 1) : idx + 1]
positive_count = sum(1 for candidate in window if candidate.label == "drone")
label = "drone" if positive_count >= confirm_m else "no_drone"
```

This is a useful practical addition because raw audio classifiers often produce
isolated high-confidence spikes that should not become confirmed alerts.

### 11. Reporting And Logging

The sound branch records both summary and per-window results. `report.py`
builds readable textual output such as:

```python
lines = [
    "Audio inference report",
    f"Video: {video_path}",
    f"Segments: {len(predictions)}",
    f"Window (s): {window_s:.2f}",
    f"Hop (s): {hop_s:.2f}",
    f"Threshold: {threshold:.2f}",
]
```

Each segment is also printed in detail:

```python
f"{idx:03d}. "
f"{item.start_s:7.2f}-{item.end_s:7.2f}s | "
f"baseline={baseline_text} | "
f"yamnet={yamnet_text} | "
f"audio={item.audio_score:.4f} | "
f"label={item.label}"
```

Logs are written through `audio.logging_utils.write_run_log(...)` into:

```text
Project v1/results/logs/
```

This is good engineering practice because it preserves evidence from
experiments instead of forcing teammates to rely on screenshots or terminal
history.

### 12. Important Difference Between The Clean Audio Module And The App Audio Path

One of the most important teammate notes in the whole repository is that the
integrated app does not use the clean audio module in only one way. It first
tries a legacy Keras CNN model and falls back to the clean project audio
pipeline if needed.

The app-specific loader in `App/main.py` does this:

```python
model = tf.keras.models.load_model(
    AUDIO_MODEL_PATH,
    custom_objects=custom_objects,
    compile=False
)
```

and only on failure does it fall back to:

```python
project_models = load_audio_models(use_baseline=True, use_yamnet=False)
```

There is also a parameter mismatch:

- `Project v1/src/audio/video_test.py` defaults to `--window-s 3.0`
- `App/main.py` sets `AUDIO_WINDOW_S = 2.0`

This means app results and clean CLI results are not automatically identical.
Teammates should not compare them blindly without checking which backend and
windowing policy were active.

### 13. Strengths, Limitations, And Teammate Notes

Strengths:

- clean package structure,
- reproducible offline inference pipeline,
- explicit schemas,
- dual-model support,
- weighted score fusion,
- optional temporal confirmation,
- detailed logs and plots.

Limitations:

- file-based only, not live-streaming,
- YAMNet path depends on TensorFlow Hub availability and artifact compatibility,
- fusion is weighted averaging, not full calibration,
- app integration still contains legacy behavior and backend divergence.

Teammate note:

For experiments, documentation, and reproducibility, prefer the clean
`audio.video_test` path first. Use the app-specific audio backend only when the
integrated dashboard behavior itself is the target of evaluation.

---

## Cross-Module Integration Summary

Although the reports above describe the branches separately, the integrated app
connects them in one synchronized playback loop. The fusion rule in
`App/main.py` is intentionally simple:

```python
positives = sum([self.last_vision_detected, audio_detected, radar_detected])
if positives >= 2:
    fusion_text, fusion_color = "Confirmed", RED
elif positives == 1:
    fusion_text, fusion_color = "Watch", AMBER
else:
    fusion_text, fusion_color = "Clear", GREEN
```

This means the current fusion layer is a decision-level voting rule:

- `Confirmed` if at least two modalities are positive,
- `Watch` if exactly one modality is positive,
- `Clear` if none are positive.

This is not the final scientific fusion solution, but it is a correct and
useful integration baseline because:

- it is transparent,
- teammates can inspect it quickly,
- it avoids hidden learned fusion behavior,
- it gives an understandable demo output.

The app also stores time-stamped snapshots for a simple detection history plot,
which helps visualize how modality agreement evolves over time.

## Reproduction Guide For Teammates

### Clean Radar Experiments

```bash
cd "/home/mo/dev/python/HDDS2/Project v1"
PYTHONPATH=src python -m radar_sim.runner --scenario single_slow --no-plots
PYTHONPATH=src python -m radar_sim.runner --scenario two_targets
PYTHONPATH=src python -m radar_sim.realtime
```

### Clean Audio Experiments

```bash
cd "/home/mo/dev/python/HDDS2/Project v1"
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4 --model both --window-s 3.0 --hop-s 0.5
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4 --confirm-m 3 --confirm-n 5
```

### Integrated Multimodal Demo

```bash
cd "/home/mo/dev/python/HDDS2"
python -m App.main
```

For the integrated app, teammates should verify in advance that:

- matching video/audio/json scenario files exist,
- the YOLOv5 repository and `best.pt` weights are available,
- the audio model artifacts exist,
- the necessary GUI and ML dependencies are installed.

## Final Conclusion

The project already contains a meaningful multimodal foundation.

The radar module is the cleanest and most academically mature part of the
system. It has a strong separation of concerns, a physically interpretable
processing chain, scenario-based evaluation, and multiple execution modes.

The sound module is the cleanest multimodal branch. It already behaves like a
reusable package with explicit schemas, clear preprocessing, multiple model
paths, and practical logging.

The vision module works operationally inside the integrated dashboard and is
already useful for multimodal demonstrations, but it is the branch that most
needs packaging cleanup and separation from the GUI layer.

Taken together, the codebase shows that the team has already implemented:

- a radar-centered detection backbone,
- a learned audio inference branch,
- a YOLO-based visual branch,
- synchronized multimodal playback,
- and a simple but understandable decision-level fusion layer.

That is the real technical achievement of the current stage of the project.
