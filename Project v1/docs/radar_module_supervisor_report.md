# Scientific Technical Report: Radar Module

## Hybrid Drone Detection System (HDDS)

**Prepared for:** Project Supervisor  
**Project:** Python-Based Hybrid Drone Detection System  
**Subsystem Covered:** Passive-Radar-Inspired Simulation Module  
**Repository Path:** `Project v1/src/radar_sim/`

## 1. Introduction

This report documents the radar module developed for the Hybrid Drone Detection
System (HDDS). The radar subsystem is the technical core of the current thesis
implementation. Its purpose is to provide a fully reproducible Python-based
simulation chain for drone detection before integrating other sensing
modalities such as audio and vision.

The implemented radar is not a hardware-connected field radar. It is a
simulation-only, passive-radar-inspired processing framework built to study:

1. waveform generation,
2. bistatic delay modeling,
3. Doppler modeling,
4. direct-path and clutter contamination,
5. range-Doppler processing,
6. CFAR detection,
7. scenario-based evaluation.

The design is intentionally software-only so that the project remains
defensible as an academic signal-processing system without depending on SDR
hardware, RF front ends, antenna calibration, or site-specific deployment
conditions.

## 2. Thesis Motivation For The Radar Module

Drone detection is difficult because small unmanned aerial vehicles may have:

- low radar cross section,
- variable flight dynamics,
- weak acoustic signatures at long range,
- and visual ambiguity under poor lighting or cluttered backgrounds.

For that reason, the project uses radar as the baseline sensing modality.
Radar contributes two crucial physical observables:

1. **bistatic range excess** from propagation delay,
2. **radial velocity** from Doppler frequency shift.

These observables are physically interpretable and remain available even when
vision or audio degrade. The radar module therefore acts as the backbone of the
hybrid system.

## 3. Passive Radar Concept Used In This Project

### 3.1 Passive Radar Principle

A passive radar does not transmit its own dedicated waveform. Instead, it uses
an external illuminator of opportunity such as a communication transmitter. In
an operational passive radar, two channels are normally observed:

1. **Reference channel:** a direct copy of the illuminator waveform.
2. **Surveillance channel:** the received environment containing:
   - direct-path leakage,
   - reflections from targets,
   - clutter,
   - noise.

The target is detected by comparing the surveillance signal against the
reference signal and estimating delay and Doppler.

### 3.2 Interpretation In This Project

This project models a **passive-radar-inspired** system using an
**OFDM-like illuminator**. It does not claim to be a full standard-compliant
LTE, DVB-T, or 5G passive radar. Instead, it abstracts the essential passive
radar ingredients:

- known reference waveform,
- delayed target replicas,
- Doppler phase progression across pulses,
- stationary clutter,
- additive noise,
- range-Doppler formation,
- CFAR thresholding.

This makes the system mathematically coherent and computationally manageable
for thesis work while preserving the main detection physics.

## 4. Overall Radar Architecture

The implemented processing chain is:

1. Generate an OFDM-like reference waveform.
2. Build a reference pulse matrix across many pulses.
3. Simulate the surveillance channel with:
   - direct path,
   - stationary clutter,
   - delayed and Doppler-shifted target echoes,
   - complex Gaussian noise.
4. Remove the cyclic prefix.
5. Transform the reference and surveillance channels to the frequency domain.
6. Divide surveillance by reference to suppress random subcarrier data
   modulation.
7. Apply IFFT across fast time to produce range profiles.
8. Subtract pulse-mean stationary content to suppress zero-Doppler clutter.
9. Apply slow-time FFT to form the range-Doppler map.
10. Apply 2D CA-CFAR to detect significant peaks.
11. Convert detections to physical units:
    - bistatic range excess in meters,
    - Doppler in Hz,
    - radial velocity in m/s.

The main implementation files are:

- [constants.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/constants.py)
- [waveform.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/waveform.py)
- [channel.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/channel.py)
- [processing.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/processing.py)
- [geometry.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/geometry.py)
- [detection.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/detection.py)
- [metrics.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/metrics.py)
- [runner.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/runner.py)
- [tui.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/tui.py)
- [realtime.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/realtime.py)

## 5. Radar Configuration And Signal Parameters

The default system parameters are loaded from
[default.yaml](/home/mo/dev/python/HDDS2/Project%20v1/configs/default.yaml).

### 5.1 Primary Radar Parameters

| Parameter | Symbol | Value | Meaning |
|---|---:|---:|---|
| Speed of light | `c` | `3.0e8 m/s` | Electromagnetic propagation speed |
| Carrier frequency | `f_c` | `2.4 GHz` | Assumed illuminator carrier |
| Wavelength | `lambda = c/f_c` | `0.125 m` | Used for Doppler-to-velocity conversion |
| Sample rate | `f_s` | `20 MHz` | Fast-time sampling frequency |
| Pulse repetition interval | `PRI` | `250 us` | Time between pulses |
| Pulse repetition frequency | `PRF = 1/PRI` | `4 kHz` | Slow-time sampling frequency |
| Number of pulses | `N_p` | `128` | Coherent slow-time snapshots |
| Number of subcarriers | `N_sc` | `256` | OFDM frequency bins |
| Cyclic prefix length | `N_cp` | `64` samples | Prefix used before each OFDM symbol |
| Samples per pulse | `N_sc + N_cp` | `320` samples | Total pulse length |
| Direct-path amplitude |  | `-35 dB` | Leakage strength |
| Clutter amplitude |  | `-45 dB` | Stationary clutter strength |
| Noise power |  | `-75 dB` | Default noise floor |

### 5.2 Derived Quantities

Using the above parameters:

1. **Useful OFDM symbol duration**

```text
T_u = N_sc / f_s = 256 / 20e6 = 12.8 us
```

2. **Cyclic prefix duration**

```text
T_cp = N_cp / f_s = 64 / 20e6 = 3.2 us
```

3. **Total pulse duration**

```text
T_sym = (N_sc + N_cp) / f_s = 320 / 20e6 = 16 us
```

4. **Coherent processing interval**

```text
CPI = N_p * PRI = 128 * 250 us = 32 ms
```

5. **Range-bin spacing**

The model uses sample-delay spacing:

```text
Delta R = c / f_s = 3e8 / 20e6 = 15 m
```

This is the bistatic range-excess spacing of the discrete delay axis used by
the processor.

6. **Maximum modeled range axis**

With `256` useful samples after cyclic-prefix removal:

```text
R_max approx (256 - 1) * 15 = 3825 m
```

7. **Doppler resolution**

```text
Delta f_d = PRF / N_p = 4000 / 128 = 31.25 Hz
```

8. **Velocity resolution**

```text
Delta v = Delta f_d * lambda / 2
        = 31.25 * 0.125 / 2
        = 1.953125 m/s
```

9. **Maximum unambiguous Doppler**

```text
|f_d,max| = PRF / 2 = 2000 Hz
```

10. **Maximum unambiguous radial velocity**

```text
|v_max| = (PRF / 2) * lambda / 2
        = 125 m/s
```

These values define the physical operating region of the simulated processor.

## 6. Reference Waveform Generation

The reference waveform is generated in
[waveform.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/waveform.py).

### 6.1 QPSK Subcarrier Symbols

Each OFDM symbol starts by generating independent QPSK symbols:

```text
X[k] in {(+1 +/- j)/sqrt(2), (-1 +/- j)/sqrt(2)}
```

This is implemented by drawing random in-phase and quadrature signs from
`{-1, +1}` and normalizing by `sqrt(2)`.

### 6.2 OFDM-Like Symbol Synthesis

For each pulse:

1. QPSK symbols are created for `256` subcarriers.
2. An IFFT is applied to move from frequency domain to time domain.
3. The time-domain symbol is power-normalized:

```text
x[n] <- x[n] / sqrt(E{|x[n]|^2})
```

4. A cyclic prefix of `64` samples is copied from the end of the symbol and
   appended to the front.

The resulting pulse contains `320` complex samples.

### 6.3 Reference Pulse Matrix

The function `generate_reference_matrix()` stacks `128` such pulses into a
matrix:

```text
reference.shape = (num_pulses, samples_per_pulse) = (128, 320)
```

Each row is one pulse and each column is a fast-time sample.

## 7. Surveillance Channel Simulation

The surveillance channel is simulated in
[channel.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/channel.py).

### 7.1 Signal Model

For fast-time sample `n` and pulse index `p`, the surveillance signal is:

```text
y[n, p] = a_direct x[n, p]
        + sum_k a_k x[n - d_k, p] exp(j 2 pi f_d,k t[n, p])
        + c[n]
        + w[n, p]
```

where:

- `x[n, p]` is the reference waveform,
- `a_direct` is the direct-path leakage amplitude,
- `a_k` is the complex-scaled target amplitude,
- `d_k` is the target delay,
- `f_d,k` is target Doppler,
- `c[n]` is stationary clutter,
- `w[n, p]` is complex Gaussian noise.

### 7.2 Direct Path

The direct path is modeled as a scaled copy of the reference waveform:

```text
a_direct = 10^(A_direct_db / 20)
```

With the default `-35 dB`, the direct path is weaker than a full-scale signal
but still significant enough to represent leakage from the illuminator.

### 7.3 Target Echo Model

Each target is described by:

- `delay_s`,
- `doppler_hz`,
- `amplitude_db`.

The target echo is created in two stages:

1. **Fractional delay**
2. **Doppler phase rotation**

#### 7.3.1 Fractional Delay

The code applies delay in the frequency domain:

```text
X(f) -> X(f) exp(-j 2 pi f tau)
```

This is implemented by:

1. FFT of the pulse,
2. multiplication by the delay phase term,
3. IFFT back to time domain.

This allows continuous-valued delays rather than restricting targets to integer
sample shifts.

#### 7.3.2 Doppler Phase

For pulse index `p` and fast-time index `n`, the Doppler term is:

```text
exp(j 2 pi f_d (n/f_s + p*PRI))
```

This introduces:

- intra-pulse phase evolution in fast time,
- inter-pulse coherent phase evolution in slow time.

That slow-time phase progression is what ultimately forms a Doppler peak after
the slow-time FFT.

### 7.4 Stationary Clutter

Stationary clutter is generated once per scenario realization as a complex
random vector over fast time and then reused across all pulses:

```text
clutter[n] = A_clutter * complex_gaussian[n]
```

Because it is reused for every pulse, it is stationary in slow time and
therefore concentrated near zero Doppler. This is intentional. It allows the
processing chain to test stationary-clutter suppression.

### 7.5 Noise

Independent complex Gaussian noise is added for each pulse:

```text
w[n, p] ~ CN(0, sigma^2)
```

with:

```text
sigma = sqrt(P_noise_linear / 2)
```

The division by `2` allocates equal variance to real and imaginary parts.

### 7.6 Surveillance Matrix

The final surveillance matrix has the same shape as the reference:

```text
surveillance.shape = (128, 320)
```

Each row contains:

- direct path,
- clutter,
- all target echoes,
- pulse-specific noise.

## 8. Geometry And Physical Interpretation

The geometric conversions are implemented in
[geometry.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/geometry.py).

### 8.1 Delay To Bistatic Range Excess

The project uses:

```text
R_b = c * tau
```

where:

- `R_b` is bistatic range excess,
- `tau` is target delay,
- `c` is speed of light.

This is appropriate for the modeled passive-radar-style delay axis. It should
be interpreted as the excess propagation path relative to the reference path,
not necessarily monostatic slant range.

### 8.2 Doppler To Velocity

The code uses:

```text
v = f_d * lambda / 2
```

where:

- `f_d` is Doppler frequency,
- `lambda = c / f_c`.

This relation is the standard monostatic-style mapping used here as a practical
kinematic interpretation layer. In a true bistatic geometry, the exact Doppler
relation depends on transmitter-target-receiver geometry, but this simplified
mapping is acceptable for the current thesis simulation scope.

## 9. Signal Processing Chain

The processing is implemented in
[processing.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/processing.py).

### 9.1 Cyclic Prefix Removal

The first step removes the cyclic prefix:

```text
ref_symbols  = reference[:, N_cp:]
surv_symbols = surveillance[:, N_cp:]
```

After this step, each pulse has `256` useful samples.

### 9.2 Frequency-Domain Symbol Transformation

Both channels are transformed into the subcarrier domain:

```text
R[k, p] = FFT{ref_symbols[p]}
S[k, p] = FFT{surv_symbols[p]}
```

### 9.3 Data Modulation Suppression

Because the illuminator subcarrier symbols are random QPSK data, the processor
removes that modulation by dividing the surveillance spectrum by the reference
spectrum:

```text
H[k, p] = S[k, p] / (R[k, p] + epsilon)
```

where `epsilon = 1e-12` avoids numerical instability in case a subcarrier is
close to zero.

This step acts like a frequency-domain matched normalization against the known
reference.

### 9.4 Range Profile Formation

An IFFT is then applied across fast time:

```text
h[n, p] = IFFT_k {H[k, p]}
```

The result is a bank of per-pulse delay profiles, referred to in the code as
`range_profiles`.

### 9.5 Stationary Component Suppression

Stationary clutter and direct leakage tend to concentrate at near-zero Doppler.
The implementation estimates stationary content by averaging across pulses:

```text
mean_profile[n] = (1/N_p) sum_p h[n, p]
```

and subtracts it:

```text
h_filtered[n, p] = h[n, p] - mean_profile[n]
```

This is a simple but effective slow-time clutter canceller.

### 9.6 Windowing Before Doppler FFT

Two Hamming windows are applied:

1. slow-time Hamming window across pulses,
2. fast-time Hamming window across range bins.

This reduces sidelobes in the resulting range-Doppler map.

### 9.7 Range-Doppler Map Formation

The filtered profiles are Fourier transformed along slow time:

```text
RD[m, n] = FFT_p {h_filtered[p, n]}
```

and `fftshift` is applied so that zero Doppler appears in the center of the
Doppler axis.

The final matrix is:

```text
range_doppler_map.shape = (128, 256)
```

with:

- axis 0 = Doppler bins,
- axis 1 = range bins.

## 10. Detection Stage: 2D CA-CFAR

Detection is implemented in
[detection.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/detection.py).

### 10.1 Why CFAR Is Used

A fixed global threshold is not appropriate when noise, clutter, and sidelobe
conditions vary across the map. CFAR provides an adaptive threshold based on
local background power.

### 10.2 Power Map

The detector first computes:

```text
P[m, n] = |RD[m, n]|^2
```

### 10.3 CFAR Window Structure

The default CFAR parameters are:

| Quantity | Value |
|---|---:|
| Guard cells in range | 2 |
| Guard cells in Doppler | 2 |
| Training cells in range | 8 |
| Training cells in Doppler | 8 |
| Probability of false alarm | `1e-5` |

This creates a 2D stencil around each cell under test:

- center cell = candidate detection,
- nearby guard cells = excluded to avoid target leakage into the noise estimate,
- outer training cells = used to estimate local noise/clutter power.

### 10.4 Threshold Computation

The local noise estimate is computed by 2D convolution with the CFAR kernel.

If the number of training cells is `N_train`, the threshold factor is:

```text
alpha = N_train * (P_fa^(-1/N_train) - 1)
```

The adaptive threshold becomes:

```text
T[m, n] = alpha * noise_estimate[m, n]
```

Detection occurs when:

```text
P[m, n] > T[m, n]
```

### 10.5 Stationary Velocity Gate

After CFAR thresholding, a practical false-alarm filter is applied:

```text
|velocity| < 1.0 m/s -> reject
```

This is configured through:

```text
min_abs_velocity_mps = 1.0
```

Its purpose is to suppress detections associated with:

- residual direct path,
- static clutter,
- nearly stationary leakage.

This is an important design decision because a drone detector should not report
every strong stationary return as a moving target.

### 10.6 Region Grouping And Peak Selection

After thresholding, connected detection regions are labeled. For each region:

1. the highest-power pixel is found,
2. its range and Doppler bins are retained,
3. its power is converted to dB,
4. the result is reported as one detection.

Each detection includes:

- range bin,
- Doppler bin,
- bistatic range excess in meters,
- Doppler in Hz,
- velocity in m/s,
- peak power in dB.

## 11. Scenario Definition And Ground Truth

Scenario management is handled by
[constants.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/constants.py),
[scenarios.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/scenarios.py),
and [metrics.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/metrics.py).

### 11.1 Default Scenarios

The repository currently defines these scenarios:

#### Clear Sky

- no targets,
- clutter and noise only.

Purpose:

- verify that the processor does not produce uncontrolled false alarms.

#### Single Slow Target

- `delay = 5 us`
- `doppler = 80 Hz`
- `amplitude = -18 dB`

Physical interpretation:

```text
range = c * tau = 1500 m
velocity = f_d * lambda / 2 = 5.0 m/s
```

#### Single Fast Target

- `delay = 8 us`
- `doppler = -350 Hz`
- `amplitude = -16 dB`

Physical interpretation:

```text
range = 2400 m
velocity = -21.875 m/s
```

#### Two Targets

Target 1:

- `delay = 4.5 us`
- `doppler = 120 Hz`
- `amplitude = -18 dB`
- `range = 1350 m`
- `velocity = 7.5 m/s`

Target 2:

- `delay = 7 us`
- `doppler = -180 Hz`
- `amplitude = -20 dB`
- `range = 2100 m`
- `velocity = -11.25 m/s`

Purpose:

- verify multi-target separation in range-Doppler space.

#### Low SNR Weak Target

- `delay = 6 us`
- `doppler = 60 Hz`
- `amplitude = -24 dB`
- `range = 1800 m`
- `velocity = 3.75 m/s`
- reduced noise floor set to `-82 dB`

Purpose:

- test detection under weaker target conditions.

### 11.2 Truth Matching

The metrics module converts each configured target into truth values in:

- range,
- Doppler,
- velocity.

Detections are then matched using tolerances:

- range tolerance = `30 m`
- Doppler tolerance = `25 Hz`

This allows automated consistency checking between injected target parameters
and recovered target estimates.

## 12. User-Facing Execution Modes

### 12.1 Batch Scenario Runner

Implemented in [runner.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/runner.py).

This mode:

1. loads the YAML configuration,
2. selects a named scenario,
3. runs the full pipeline,
4. prints a detection report,
5. optionally plots the outputs,
6. saves a timestamped log.

### 12.2 Interactive TUI

Implemented in [tui.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/tui.py).

This mode accepts targets in physical units:

- distance,
- speed,
- amplitude,
- noise power,
- clutter amplitude.

It then converts:

```text
distance -> delay
speed    -> Doppler
```

and runs the scenario immediately.

This makes the simulation more intuitive for demonstrations and supervisor
review because targets can be specified in meters and meters per second instead
of raw signal parameters.

### 12.3 Realtime Moving-Target Mode

Implemented in [realtime.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/realtime.py).

This mode simulates target motion over time. For each target:

```text
range(t) = max(0, range_0 - v * t)
```

A fresh instantaneous scenario is rebuilt at every frame, processed, and
displayed.

This provides a simplified dynamic demonstration of how detections move when a
drone approaches or recedes.

## 13. Radar Outputs Produced By The Module

The module generates several important outputs.

### 13.1 Range Profiles

Per-pulse delay-domain profiles after reference normalization.

### 13.2 Filtered Range Profiles

Range profiles after stationary mean subtraction.

### 13.3 Range-Doppler Map

A 2D map with:

- horizontal axis = bistatic range excess,
- vertical axis = radial velocity,
- pixel intensity = signal power.

### 13.4 CFAR Threshold Map

Adaptive threshold used for local detection decisions.

### 13.5 Binary Detection Mask

Logical map of cells that passed CFAR and velocity gating.

### 13.6 Detection List

Final compact detections with physical interpretation.

### 13.7 Text Logs

Each run writes a structured log through
[logging_utils.py](/home/mo/dev/python/HDDS2/Project%20v1/src/radar_sim/logging_utils.py).

This improves experiment reproducibility and auditability.

## 14. Scientific Merits Of The Implemented Radar Design

The current module has several strengths as a thesis radar core:

1. **Physically interpretable observables**  
   The injected parameters and reported outputs map cleanly to delay, Doppler,
   range, and velocity.

2. **Reproducible simulation environment**  
   The YAML configuration and fixed random seed allow repeatable experiments.

3. **Clear passive-radar structure**  
   The split between reference and surveillance channels matches the passive
   radar concept.

4. **Modular implementation**  
   Waveform generation, channel modeling, processing, detection, plotting, and
   metrics are separated cleanly.

5. **Support for scenario studies**  
   The code can test clear sky, single-target, multi-target, weak-target, and
   moving-target cases.

6. **Good bridge to multimodal fusion**  
   The module already produces timestamped, structured detections that can later
   be fused with audio and vision branches.

## 15. Modeling Assumptions And Limitations

For academic honesty, the following limitations should be stated explicitly.

### 15.1 Not A Hardware Radar

This is a simulation-only system. It does not currently include:

- SDR receivers,
- antenna geometry,
- synchronization error,
- oscillator phase noise,
- ADC quantization,
- terrain multipath geometry,
- transmitter waveform estimation from real data.

### 15.2 OFDM-Like, Not Standard-Compliant LTE/DVB

The waveform is inspired by communication signals but is not tied to a full
broadcast or cellular standard stack. This is deliberate: the goal is to study
passive-radar processing principles, not reproduce an entire telecom standard.

### 15.3 Simplified Bistatic Interpretation

Delay is mapped as:

```text
R_b = c * tau
```

and velocity is mapped with a monostatic-style formula:

```text
v = f_d * lambda / 2
```

This is suitable for the present simulation, but a full bistatic deployment
would require geometry-specific localization and Doppler interpretation.

### 15.4 Simplified Clutter

Clutter is modeled as stationary complex random structure that is constant
across pulses. Real clutter can be:

- distributed,
- time-varying,
- terrain-dependent,
- and spectrally colored.

### 15.5 Delay/CP Practicality

The cyclic prefix duration is `3.2 us`, while some configured target delays are
larger than this. In a strict communications receiver, delays beyond the cyclic
prefix can introduce inter-symbol interference. The current simulation still
localizes such delays because it models delay using a simplified fractional
shift and processes each pulse independently. This is acceptable for a thesis
simulation, but it should be acknowledged as a modeling simplification rather
than a fully standard-faithful OFDM channel model.

## 16. Why This Radar Module Is Suitable For The Graduation Project

This radar subsystem is suitable for the graduation project because it shows:

1. solid signal-processing structure,
2. physically meaningful simulation variables,
3. configurable scenarios,
4. quantitative detection logic,
5. a clean path to future multimodal fusion.

It is not a toy script. It is a complete processing chain:

- waveform generation,
- channel synthesis,
- matched normalization,
- range processing,
- Doppler processing,
- adaptive detection,
- truth comparison,
- logging,
- visualization,
- interactive experimentation.

That makes it a strong technical baseline for the final hybrid drone detection
system.

## 17. Recommended Next Steps

To mature the radar module further, the next technically justified steps are:

1. add explicit bistatic geometry with transmitter, receiver, and target
   coordinates,
2. model path loss and radar cross section more realistically,
3. include non-stationary clutter and multipath,
4. introduce transmitter waveform uncertainty instead of perfect reference
   knowledge,
5. test alternative clutter cancellers and detectors,
6. add target tracking over time,
7. fuse radar detections with the audio and vision branches.

## 18. Conclusion

The implemented radar module provides a rigorous, modular, and reproducible
passive-radar-inspired simulation environment for drone detection. It models a
reference channel and a surveillance channel, synthesizes delayed and
Doppler-shifted target echoes, forms range-Doppler maps, and detects targets
using 2D CA-CFAR with stationary-return suppression.

Its main contribution to the thesis is that it establishes the radar backbone
in a technically defensible way. The module already supports controlled
experiments, custom scenario creation, moving-target demonstrations, and
structured output suitable for future multimodal integration. Within the scope
of a graduation project, it is an appropriate and scientifically coherent
implementation of the radar core.
