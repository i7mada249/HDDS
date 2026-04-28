# Project v1

Python-only radar simulation project for the graduation thesis.

## Scope

This version focuses on one clear system:

- OFDM-like illuminator generation
- Reference and surveillance channel simulation
- Delay and Doppler target injection
- Range-Doppler processing
- CA-CFAR target detection
- Scenario-based evaluation

This project does **not** depend on hardware, YOLO, audio classification, or GUI code.

## Status

This is the first clean implementation pass created from the old repository material.

## Directory Layout

```text
Project v1/
├── src/radar_sim/
├── tests/
├── notebooks/
├── configs/
├── docs/
├── results/
└── requirements.txt
```

## Setup

Create a virtual environment, then install:

```bash
pip install -r requirements.txt
```

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

## Run Tests

```bash
PYTHONPATH=src pytest -q
```

## Technical Notes

- The model is simulation-only.
- The reported range axis is treated as bistatic range excess derived from delay.
- Velocity is derived from Doppler using a carrier-frequency assumption.
- The current implementation uses an OFDM-like waveform, not a strict LTE stack.

See [docs/methodology.md](/home/mo/dev/python/HDDS2/Project%20v1/docs/methodology.md) for the modeling assumptions.
