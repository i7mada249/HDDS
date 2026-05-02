# Plan V1-1

## Multimodal Fusion Plan

**Date:** 2026-04-28  
**Objective:** extend `Project v1` from radar-only simulation into a **unified radar + vision + audio detection system** that reduces false alarms through evidence fusion.

## Executive Decision

Do **not** bolt vision and sound onto the project as side modules.

The correct target is:

**one synchronized multimodal detection system with a shared event timeline, shared scoring logic, and a final fused decision.**

If you do not unify the timing, scoring, and decision rules, the result will look like three demos, not one system.

## Main System Definition

The upgraded project should become:

**Multimodal Drone Detection System Using Passive-Radar-Inspired Sensing, Vision Detection, and Audio Detection with Decision Fusion**

## Why Add Vision and Audio

Radar alone can false-alarm under:

1. clutter leakage,
2. weak moving interference,
3. threshold tuning errors,
4. ambiguous small peaks.

Vision helps confirm:

1. visible airborne object presence,
2. approximate image position,
3. persistence across frames,
4. class-specific evidence.

Audio helps confirm:

1. rotor / propeller acoustic signature,
2. persistent drone-like tonal content,
3. events that radar sees but vision misses in low-resolution frames.

The fusion goal is not:

- `radar OR vision OR audio`

The fusion goal is:

- **high sensitivity from radar**
- **false-alarm suppression from vision and audio**
- **final confidence from calibrated evidence fusion**

## Correct Scope Choice

Because your current project is still **Python-only** and **no hardware**, the right near-term architecture is:

### Mode A: Offline / File-Based Multimodal Fusion

Inputs:

1. radar simulation output,
2. video file,
3. audio track or wav file.

All three streams must be aligned on a common timeline.

This is the best graduation-project scope because:

1. it is testable,
2. it is reproducible,
3. it avoids hardware complexity,
4. and it still demonstrates real systems engineering.

### Mode B: Real-Time Multimodal System

This is a future extension, not the first target.

Do not make real-time fusion the first milestone unless you already have stable per-modality pipelines.

## United System Architecture

The unified system should be built in **five layers**:

### 1. Ingestion Layer

Purpose:

- load radar scenario or radar outputs,
- load video frames,
- load audio waveform,
- assign timestamps.

Outputs:

- `RadarFrame`
- `VideoFrame`
- `AudioWindow`

All must carry:

- `t_start`
- `t_end`
- `stream_id`

### 2. Per-Modality Inference Layer

Purpose:

- run radar detection,
- run object detection / tracking,
- run audio classification.

Outputs:

- `RadarDetections`
- `VisionDetections`
- `AudioDetections`

Each detection must include:

1. timestamp/window,
2. confidence score,
3. modality-specific measurements,
4. optional uncertainty.

### 3. Temporal Alignment Layer

Purpose:

- map radar detections, video detections, and audio detections into a common time grid.

This is mandatory.

Without it, fusion is technically weak.

### 4. Fusion Layer

Purpose:

- combine evidence from the three modalities into one event score.

Outputs:

- fused object/event score,
- fused alert label,
- explanation of which modality contributed.

### 5. Tracking and Decision Layer

Purpose:

- maintain event persistence over time,
- suppress one-frame spikes,
- confirm alerts only after consistent evidence.

Outputs:

- final alert,
- confidence,
- track history,
- false-alarm reduced event list.

## Best Algorithms By Module

The right answer is different for:

1. **best practical algorithm for your project now**
2. **best ambitious research-grade algorithm**

### Radar Module

#### Best practical choice now

1. OFDM-based range-Doppler processing
2. CA-CFAR or OS-CFAR
3. track confirmation using **M/N logic**
4. simple **Kalman filter** for temporal smoothing

Why:

- you already have most of this path,
- it is explainable,
- it is fast,
- and it directly addresses false alarms.

#### Best upgrade path

1. switch from pure CA-CFAR to **OS-CFAR** or hybrid CFAR in clutter-stressed cases,
2. add **track-before-confirm** logic,
3. add **Kalman filtering** or alpha-beta tracking,
4. optionally add clutter-suppression improvements before CFAR.

#### What matters most

Radar false alarms are usually reduced more by:

1. better temporal confirmation,
2. better clutter handling,
3. and better threshold calibration

than by changing the waveform alone.

### Vision Module

#### Best practical choice now

**YOLO11 detect + a tracker such as ByteTrack or BoT-SORT**

Reasoning:

1. easier integration than a transformer detector,
2. strong real-time ecosystem,
3. mature training/export story,
4. good enough for a graduation project.

#### Best ambitious research choice

**RT-DETR**

Reasoning:

1. end-to-end detector,
2. no NMS dependency in the standard design,
3. strong research value.

But integration cost is higher than the YOLO path.

#### Recommendation

Use:

1. **YOLO11** if you want the best practical integration path,
2. **RT-DETR** only if vision becomes the centerpiece of the thesis.

#### Important note

Do not use frame-level detection only.

You need **tracking** because false alarms are reduced by:

1. persistence across frames,
2. trajectory consistency,
3. confidence smoothing.

### Audio Module

#### Best practical choice now

Use your current handcrafted-feature pipeline as a baseline:

1. MFCC
2. deltas
3. spectral features
4. LightGBM or another calibrated tree model

This is still a valid baseline because:

1. it is lightweight,
2. interpretable,
3. and easy to debug.

#### Best upgrade path

Use **pretrained audio embeddings** and train a shallow drone-vs-not-drone classifier on top.

Preferred options:

1. **YAMNet** for a very practical baseline,
2. **PANNs embeddings** for stronger pretrained audio features,
3. **AST / SSAST** if you want the strongest research-oriented audio model path.

#### Recommendation

For your project:

1. keep the current MFCC + LightGBM system as baseline,
2. add **PANNs or AST embeddings + shallow classifier** as the upgraded model,
3. compare both and keep the one that performs better on your actual drone dataset.

Why not jump straight to a heavy end-to-end model?

Because limited drone-specific data often makes:

- pretrained embeddings + small classifier

more practical than full end-to-end training.

## Best Fusion Algorithms

This is the most important design decision.

### Bad fusion strategy

Do not use:

1. raw OR logic,
2. uncalibrated score averaging,
3. “if any detector fires, alert”.

That increases false alarms or makes the system unstable.

### Best practical fusion strategy now

Use **late fusion with calibrated per-modality confidence scores**.

Example:

1. radar produces `P_radar`
2. vision produces `P_vision`
3. audio produces `P_audio`
4. final score:

```text
P_fused = w_r * P_radar + w_v * P_vision + w_a * P_audio
```

Then require:

1. `P_fused > threshold`
2. persistence over `M/N` windows

This is the best first implementation.

### Best stronger fusion strategy

Use a **meta-classifier** on top of modality features:

Inputs:

1. radar peak power
2. radar Doppler stability
3. radar track age
4. vision confidence
5. vision track persistence
6. bounding box size / motion consistency
7. audio confidence
8. audio tonal stability
9. modality availability flags

Classifier choices:

1. Logistic Regression
2. XGBoost / LightGBM
3. small MLP

#### Recommendation

Use:

1. weighted late fusion first,
2. then a small meta-classifier when you have aligned labeled data.

### Best temporal confirmation logic

Use **M/N confirmation**:

- alert only if evidence is present in `M` out of the last `N` windows.

Example:

- 3 out of 5 windows

This is simple and very effective.

## How To Make It One United System

You need a **single shared event representation**.

Define one common structure like:

```text
UnifiedDetectionEvent:
  t_start
  t_end
  radar_score
  vision_score
  audio_score
  fused_score
  radar_features
  vision_features
  audio_features
  final_label
  explanation
```

This is what makes the system one project instead of three scripts.

## Required New Project Structure

Extend `Project v1` like this:

```text
Project v1/
├── src/
│   ├── radar_sim/
│   ├── vision/
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── features.py
│   │   └── models/
│   ├── audio/
│   │   ├── preprocess.py
│   │   ├── features.py
│   │   ├── classifier.py
│   │   └── models/
│   ├── fusion/
│   │   ├── schemas.py
│   │   ├── align.py
│   │   ├── rules.py
│   │   ├── scorer.py
│   │   └── tracker.py
│   └── app/
│       ├── pipeline.py
│       ├── report.py
│       └── tui.py
├── configs/
│   ├── radar.yaml
│   ├── vision.yaml
│   ├── audio.yaml
│   └── fusion.yaml
├── notebooks/
├── tests/
└── results/
```

## Data You Need

You cannot evaluate fusion correctly without synchronized or at least alignable data.

You need:

1. video with visible drone / no drone cases,
2. aligned audio,
3. radar scenario labels or radar simulation truth,
4. timestamps,
5. positive and negative cases.

Minimum dataset design:

1. clear negative scenes,
2. drone visible + audible scenes,
3. drone visible but weak audio scenes,
4. audio-only ambiguous scenes,
5. radar false-alarm scenes,
6. clutter-heavy scenes.

## Evaluation Metrics

Evaluate at two levels:

### Per-modality

1. precision
2. recall
3. F1
4. ROC-AUC if probabilistic
5. false alarms per minute

### System-level

1. fused precision
2. fused recall
3. fused F1
4. false alarms per minute
5. miss rate
6. detection latency
7. alert persistence quality

The key thesis result should be:

**multimodal fusion reduces false alarms relative to radar-only detection**

## Phased Execution Plan

### Phase 1: Freeze the fusion thesis

Decide and document:

1. offline multimodal fusion first,
2. common timeline,
3. late fusion first,
4. track confirmation mandatory.

### Phase 2: Stabilize radar as the high-sensitivity trigger

Tasks:

1. keep current radar pipeline,
2. add M/N confirmation,
3. add simple Kalman or alpha-beta tracking,
4. quantify radar-only false alarms.

### Phase 3: Build the vision branch

Tasks:

1. choose YOLO11 as practical default,
2. add frame-to-frame tracking,
3. expose per-frame score and track persistence,
4. export normalized confidence per time window.

### Phase 4: Build the audio branch

Tasks:

1. keep current MFCC + LightGBM as baseline,
2. add pretrained-embedding pipeline,
3. evaluate both,
4. expose per-window confidence.

### Phase 5: Build temporal alignment

Tasks:

1. choose one time window size,
2. align radar pulses, video frames, and audio windows to that grid,
3. handle missing modality data explicitly.

### Phase 6: Build fusion logic

Tasks:

1. define score normalization,
2. define weighted fusion,
3. define M/N alert rule,
4. log contribution of each modality.

### Phase 7: Build the unified TUI / app

Tasks:

1. configure scenario,
2. choose input files or simulated mode,
3. run all branches,
4. show per-modality results,
5. show fused results,
6. save plots and numeric summary.

### Phase 8: Run ablation study

You need these comparisons:

1. radar only
2. radar + vision
3. radar + audio
4. radar + vision + audio

This is where your thesis claim is proved or disproved.

## What You Must Know Before Building

1. **Fusion only works if scores are calibrated.**  
   Raw model confidence from different modalities is not directly comparable.

2. **Tracking is as important as classification.**  
   Persistent weak evidence is often more valuable than one strong isolated detection.

3. **Vision and audio should confirm radar, not replace it.**  
   Radar should remain the main wide-area trigger in your system narrative.

4. **Negative data quality matters more than positive data quantity for false alarms.**  
   You need many hard negative scenes.

5. **Synchronization is a first-class engineering problem.**  
   If the timestamps are weak, your fusion results are weak.

6. **Explainability matters for a graduation project.**  
   A fused alert should answer:
   - what each modality saw,
   - when it saw it,
   - and why the final system fired.

## Recommended Final Position

If you want the strongest practical plan:

1. keep radar as the core sensor,
2. use YOLO11 + tracking for vision,
3. use MFCC baseline plus PANNs or AST embeddings for audio,
4. use weighted late fusion plus M/N temporal confirmation,
5. prove false-alarm reduction with ablation experiments.

That is coherent, defendable, and realistically achievable.

## Recommended First Implementation Order

Do this in order:

1. add radar temporal confirmation,
2. integrate vision branch,
3. integrate audio branch,
4. implement time alignment,
5. implement weighted fusion,
6. add unified reporting,
7. run ablations,
8. only then consider more advanced fusion models.

## Source Notes

Current recommendations were checked against primary or official sources on 2026-04-28:

1. Ultralytics documentation for current YOLO-family deployment guidance.
2. RT-DETR paper: *DETRs Beat YOLOs on Real-time Object Detection*.
3. TensorFlow Hub YAMNet documentation.
4. PANNs paper: *Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition*.
5. AST paper: *Audio Spectrogram Transformer*.
