# Review V1-0

## Project Review

**Date:** 2026-04-27  
**Scope reviewed:** `signal module/`, `archive/signal/`, `AudioModule/`, `App/`, and the repo structure around them.

## Executive Verdict

This repository shows strong effort and real technical growth, especially in the later radar notebooks and reports. The best part of the project is the **simulation-side radar work**, not the GUI, audio model, or YOLO integration.

Right now the project is **not one coherent graduation project**. It is a mixture of:

1. A passive-radar / OFDM / ISAC simulation track.
2. A drone audio-classification track.
3. A video-detection GUI track built around YOLO.
4. A vendored `yolov5` codebase.

From a DSP / radar-engineering standpoint, the graduation project should be rebuilt around **one clear thesis**. The strongest thesis already present in the repo is:

**Python-only radar signal simulation and detection using OFDM-like illuminators, range-Doppler processing, and CFAR-based detection.**

## What You Are Really Trying To Build

Based on `signal module/v5-3.ipynb`, `archive/signal/projectDoc2.md`, `archive/signal/professional_isac_detector.py`, and `archive/signal/v5-3_IMPROVEMENTS_TECHNICAL_REPORT.md`, the real target is:

- a radar sensing simulator,
- using communication-style waveforms,
- with reference and surveillance channels,
- target delay/Doppler modeling,
- range-Doppler map generation,
- and adaptive detection.

That is the correct technical center of the project.

## Powerful Points

1. **The radar direction is academically strong.**  
   The later material moves toward a serious final-year topic: OFDM-based radar / passive radar simulation with Doppler, clutter, noise, and CFAR.

2. **There is clear iteration and improvement.**  
   The repo shows progression from simple signal generation toward more realistic modeling. That matters. It proves learning, not just code accumulation.

3. **You already identified the right radar problems.**  
   The archived review material correctly calls out key issues:
   - reference vs surveillance separation,
   - correct delay/Doppler modeling,
   - range-Doppler FFT correctness,
   - CFAR instead of naive thresholding,
   - physically meaningful axes and scaling.

4. **Some of the later radar code is modular enough to reuse conceptually.**  
   `archive/signal/professional_isac_detector.py` is much closer to a clean scientific prototype than the GUI/audio code. It already separates:
   - waveform generation,
   - environment simulation,
   - processing,
   - detection,
   - plotting.

5. **Scenario-based evaluation is a good presentation choice.**  
   The scenario structure in `signal module/v5-3.ipynb` and `archive/signal/isac_professional_demo.py` is useful for defense and reporting:
   - clear sky,
   - slow target,
   - fast target,
   - multi-target,
   - weak-target / low-SNR.

6. **The project documentation ambition is higher than average.**  
   `archive/signal/projectDoc2.md` and `archive/signal/v5-3_IMPROVEMENTS_TECHNICAL_REPORT.md` show an attempt to justify the system mathematically, not just visually.

7. **The audio module shows ML experimentation ability.**  
   `AudioModule/DETECTION_PROCESS.md` is reasonably structured. It is not the core of the radar project, but it demonstrates feature engineering and model selection discipline.

## Weak Points

1. **The project scope is fragmented.**  
   Radar, audio classification, YOLO vision, and a GUI are competing for the same graduation-project identity. This weakens the scientific story.

2. **The repo does not currently present one defendable system.**  
   There is no single top-level README, no pinned environment, no single entrypoint, and no agreed system definition.

3. **The radar terminology is not yet disciplined enough.**  
   Some files say passive radar, some say ISAC, and some implementations behave closer to a controlled active / semi-active simulation than a strict passive bistatic radar model.  
   This is a major review risk. In a defense, a supervisor can ask:  
   "Is this truly passive radar, or simulated OFDM radar using your own transmitter model?"

4. **The later radar code is promising but still not a validated scientific package.**  
   It is still mostly notebook/report driven. There is no clean package layout, no reproducible config system, and very limited automated validation.

5. **The project relies heavily on notebooks and duplicated logic.**  
   Similar logic appears across:
   - `signal module/v5-1_final.ipynb`
   - `signal module/v5-2.ipynb`
   - `signal module/v5-3.ipynb`
   - multiple `archive/signal/final*.ipynb`
   - `archive/signal/professional_isac_detector.py`

6. **The repo contains heavy third-party baggage unrelated to the final radar thesis.**  
   Vendoring all of `yolov5/` inside the graduation repo makes the project look unfocused and harder to review.

7. **Reproducibility is weak.**  
   There is no project-level `requirements.txt` or environment file for the actual radar code.  
   I tried running the radar scripts, and they fail immediately because the current Python environment does not even provide `numpy`.

8. **Validation is not yet strong enough for a scientific submission.**  
   Current outputs are mostly demonstrations, not acceptance tests. Missing:
   - deterministic unit tests for delay and Doppler,
   - expected-bin checks,
   - controlled false-alarm validation for CFAR,
   - regression tests across scenarios.

9. **The current signal model still needs a strict architectural decision.**  
   The clean restart must choose one:
   - true passive bistatic radar simulation, or
   - OFDM ISAC / active sensing simulation.  
   Mixing both terms without a hard definition is technically unsafe.

10. **The GUI/audio/video work does not help the radar story enough.**  
    `App/main.py` and `App/main2.py` hardcode external assumptions like `ffmpeg`, `torch.hub`, `joblib`, and a YOLO weight path. This is not aligned with a pure-Python radar thesis.

## Best Existing Material To Keep

Keep these as reference material, not as the new implementation baseline:

1. `archive/signal/projectDoc2.md`  
   Best high-level academic direction and scenario framing.

2. `archive/signal/v5-3_IMPROVEMENTS_TECHNICAL_REPORT.md`  
   Best self-critique of technical issues and physics realism.

3. `archive/signal/professional_isac_detector.py`  
   Best current code organization pattern.

4. `signal module/v5-3.ipynb`  
   Best presentation notebook structure and scenario flow.

5. `AudioModule/DETECTION_PROCESS.md`  
   Useful only as an example of how to write a clean engineering/process note.

## Best Existing Material To Archive

These should not drive the new radar project:

1. `App/main.py`
2. `App/main2.py`
3. `AudioModule/drone_detector.joblib`
4. `yolov5/`

They are separate experiments, not the clean foundation for the radar thesis.

## Final Assessment

The project has **good raw material** and **strong technical ambition**, but the current repository is too mixed and too notebook-heavy to be the final graduation submission.

The correct move is to restart with:

- one clear radar thesis,
- one reproducible Python environment,
- one modular codebase,
- one formal validation strategy,
- and one final notebook/report built on top of tested modules.

That restart is justified. It is not throwing work away. It is extracting the strongest 20% of the current project and turning it into a defendable final system.
