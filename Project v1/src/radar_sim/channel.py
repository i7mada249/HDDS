from __future__ import annotations

import numpy as np

from .constants import RadarConfig, ScenarioConfig, Target


def db_to_linear_amplitude(value_db: float) -> float:
    return 10.0 ** (value_db / 20.0)


def db_to_linear_power(value_db: float) -> float:
    return 10.0 ** (value_db / 10.0)


def apply_fractional_delay(signal: np.ndarray, delay_s: float, fs_hz: float) -> np.ndarray:
    num_samples = signal.size
    freqs = np.fft.fftfreq(num_samples, d=1.0 / fs_hz)
    spectrum = np.fft.fft(signal)
    delayed_spectrum = spectrum * np.exp(-1j * 2.0 * np.pi * freqs * delay_s)
    return np.fft.ifft(delayed_spectrum)


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


def simulate_surveillance_matrix(
    reference: np.ndarray,
    config: RadarConfig,
    scenario: ScenarioConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    num_pulses, pulse_len = reference.shape
    surveillance = np.zeros_like(reference, dtype=np.complex128)

    direct_path_amplitude = db_to_linear_amplitude(config.direct_path_amplitude_db)
    clutter_amplitude = db_to_linear_amplitude(scenario.clutter_amplitude_db)
    noise_sigma = np.sqrt(db_to_linear_power(scenario.noise_power_db) / 2.0)

    stationary_clutter = (
        clutter_amplitude
        * (rng.standard_normal(pulse_len) + 1j * rng.standard_normal(pulse_len))
        / np.sqrt(2.0)
    )

    for pulse_idx in range(num_pulses):
        pulse = direct_path_amplitude * reference[pulse_idx]
        pulse = pulse + stationary_clutter

        for target in scenario.targets:
            pulse = pulse + synthesize_target_echo(
                reference_pulse=reference[pulse_idx],
                target=target,
                pulse_index=pulse_idx,
                config=config,
            )

        noise = noise_sigma * (
            rng.standard_normal(pulse_len) + 1j * rng.standard_normal(pulse_len)
        )
        surveillance[pulse_idx] = pulse + noise

    return surveillance
