# Plan V1-0

## Restart Plan

**Goal:** rebuild the graduation project from scratch as a **Python-only radar simulation project**.  
**No hardware. No YOLO. No audio classifier in the core scope.**

## Scope Decision

The new project should be defined as:

**Simulation-only bistatic passive radar using an OFDM-like illuminator, with reference/surveillance modeling, range-Doppler processing, and CFAR detection.**

This is the cleanest choice because it matches the best material already in the repo while giving you a defensible radar story.

## What We Keep From Old Work

We will reuse ideas, not old structure.

Use these old files as references:

1. `archive/signal/projectDoc2.md`
   - scenario design,
   - expected processing stages,
   - academic framing.

2. `archive/signal/v5-3_IMPROVEMENTS_TECHNICAL_REPORT.md`
   - list of major technical errors to avoid.

3. `archive/signal/professional_isac_detector.py`
   - modular function breakdown.

4. `signal module/v5-3.ipynb`
   - final notebook presentation flow.

## What We Do Not Carry Forward

1. GUI-first design.
2. YOLO / vision detection as part of the core system.
3. Audio classification as part of the core system.
4. Notebook-only development.
5. Ambiguous passive-radar vs ISAC naming.

## New Project Definition

### Working title

**Python Passive Radar Simulator for Drone-Like Target Detection Using OFDM Illuminators**

### Core deliverables

1. Reproducible Python environment.
2. Modular radar simulation code.
3. Automated validation tests.
4. Scenario runner.
5. Final scientific notebook/report.
6. Clean figures for defense.

## Recommended Clean Directory Design

This is the structure I recommend for the new implementation phase:

```text
project_root/
├── src/
│   └── radar_sim/
│       ├── constants.py
│       ├── waveform.py
│       ├── geometry.py
│       ├── channel.py
│       ├── processing.py
│       ├── detection.py
│       ├── scenarios.py
│       ├── metrics.py
│       └── plotting.py
├── tests/
│   ├── test_waveform.py
│   ├── test_delay_doppler.py
│   ├── test_range_axis.py
│   ├── test_cfar.py
│   └── test_scenarios.py
├── notebooks/
│   └── final_report.ipynb
├── configs/
│   └── default.yaml
├── results/
├── docs/
│   └── methodology.md
├── requirements.txt
└── README.md
```

## Phase Plan

### Phase 1: Freeze the Thesis

**Objective:** remove ambiguity.

Decisions to lock:

1. This is a **simulation-only passive radar project**.
2. The waveform is **OFDM-like**, representing an illuminator of opportunity.
3. The receiver has:
   - a reference channel,
   - a surveillance channel.
4. The detection output is:
   - range,
   - Doppler,
   - CFAR decision map.

**Exit criteria:**

- one-paragraph thesis statement,
- one system block diagram,
- one list of assumptions.

### Phase 2: Build the Reproducible Environment

**Objective:** make the project runnable by anyone.

Tasks:

1. Create `requirements.txt`.
2. Pin the core stack:
   - `numpy`
   - `scipy`
   - `matplotlib`
   - `jupyter`
   - optionally `pydantic` or `pyyaml` for config
3. Add a top-level `README.md` with setup and run steps.

**Exit criteria:**

- fresh environment installs cleanly,
- one command runs a baseline scenario.

### Phase 3: Implement the Scientific Signal Model

**Objective:** establish correct radar math before plotting.

Tasks:

1. Define all physical constants in one place.
2. Define radar assumptions explicitly:
   - carrier frequency,
   - bandwidth,
   - sample rate,
   - CPI,
   - bistatic geometry simplification.
3. Implement waveform generation:
   - OFDM subcarriers,
   - cyclic prefix,
   - power normalization.
4. Implement the strict surveillance model:

```text
y[n] = direct_path + Σ alpha_k x[n - d_k] exp(j 2π f_d,k n / f_s) + clutter + noise
```

**Exit criteria:**

- waveform power is normalized,
- target delay is measurable,
- Doppler injection is measurable.

### Phase 4: Build the DSP Chain

**Objective:** produce a correct range-Doppler map.

Tasks:

1. Reference / surveillance preprocessing.
2. Range compression or cross-ambiguity style processing.
3. Slow-time Doppler FFT.
4. Proper axis construction:
   - bistatic range or range-excess definition,
   - Doppler in Hz,
   - optional velocity conversion.
5. Windowing and scaling policy.

**Exit criteria:**

- a synthetic single target appears in the expected bin,
- axis units are physically explained and consistent.

### Phase 5: Add Detection Logic

**Objective:** replace visual detection with statistical detection.

Tasks:

1. Implement CA-CFAR first.
2. Add guard/training cell configuration.
3. Return:
   - threshold map,
   - binary detection map,
   - extracted peak list.
4. Add simple clustering / peak grouping.

**Exit criteria:**

- clear-sky scenario has controlled false alarms,
- target scenarios generate stable detections.

### Phase 6: Define the Official Scenario Set

Use the best scenario pattern from the old notebooks, but standardize it.

Official scenarios:

1. Noise floor / no target.
2. Single slow target.
3. Single fast target.
4. Two-target separation case.
5. Low-SNR weak target.
6. Clutter-stressed case.

**Exit criteria:**

- each scenario is repeatable,
- each scenario has expected outputs documented.

### Phase 7: Add Validation and Tests

**Objective:** make the project defensible.

Minimum tests:

1. Waveform normalization test.
2. Delay-to-range consistency test.
3. Doppler-to-bin consistency test.
4. CFAR threshold sanity test.
5. Scenario regression test with tolerance bands.

**Exit criteria:**

- tests pass in a clean environment,
- failures are informative.

### Phase 8: Build the Final Notebook and Report

**Objective:** separate implementation from presentation.

Notebook rules:

1. Notebook imports tested modules from `src/`.
2. No core algorithm should live only in notebook cells.
3. Each section explains:
   - model,
   - equations,
   - output interpretation.
4. Final plots must include:
   - time-domain examples,
   - spectrum,
   - range profile,
   - range-Doppler map,
   - CFAR detection map,
   - scenario comparison summary.

**Exit criteria:**

- notebook runs top-to-bottom,
- figures are publication/defense quality.

## Technical Rules For The Restart

1. Use one naming system consistently: `range_bin`, `doppler_bin`, `range_axis_m`, `doppler_axis_hz`.
2. Keep all constants centralized.
3. Every processing block must have:
   - purpose,
   - equation,
   - expected output.
4. Do not call it passive radar unless the geometry and signal model support that claim.
5. If we simplify the geometry, state the simplification clearly.
6. Keep the repo lightweight and reviewable.

## Success Criteria

The restart is successful when the project can demonstrate:

1. scientifically correct signal generation,
2. correct target delay/Doppler injection,
3. correct range-Doppler formation,
4. CFAR-based target detection,
5. repeatable scenario evaluation,
6. clean report-quality presentation.

## Immediate Next Step

The next execution step should be:

**create the clean radar project skeleton and pinned Python environment before writing any more notebook logic.**

That will prevent the new version from falling back into the same notebook-fragmentation problem as the current repo.
