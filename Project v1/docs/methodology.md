# Methodology

## Thesis Definition

This project models a simulation-only passive-radar-inspired sensing chain using an OFDM-like illuminator.

The system includes:

1. A reference channel containing the illuminator waveform.
2. A surveillance channel containing:
   - direct-path leakage,
   - target echoes with delay and Doppler,
   - clutter,
   - complex Gaussian noise.

## Surveillance Signal Model

For pulse index `p` and fast-time index `n`, the surveillance channel follows:

```text
y[n, p] = a_direct x[n, p]
         + sum_k a_k x[n - d_k, p] exp(j 2 pi f_d,k t[n, p])
         + clutter[n, p]
         + noise[n, p]
```

## Processing Chain

1. Remove cyclic prefix.
2. Transform reference and surveillance symbols to frequency domain.
3. Divide surveillance by reference to suppress data modulation.
4. IFFT across fast-time to form range profiles.
5. Remove zero-Doppler stationary content using pulse-mean subtraction.
6. Apply slow-time FFT to generate a range-Doppler map.
7. Apply CA-CFAR on power.

## Axis Definitions

- Delay is converted to bistatic range excess by:

```text
R_b = c * tau
```

- Doppler is converted to velocity by:

```text
v = f_d * lambda / 2
```

where `lambda = c / f_c`.

## Validation Philosophy

The project should be accepted only if:

1. injected delay is recovered near the expected range bin,
2. injected Doppler is recovered near the expected Doppler bin,
3. clear-sky cases do not explode into false alarms,
4. multi-target cases show separable detections.
