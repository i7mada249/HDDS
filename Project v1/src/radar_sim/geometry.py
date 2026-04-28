from __future__ import annotations

import numpy as np

from .constants import RadarConfig


def delay_to_bistatic_range_m(delay_s: float, config: RadarConfig) -> float:
    return config.speed_of_light * delay_s


def bistatic_range_to_delay_s(range_m: float, config: RadarConfig) -> float:
    return range_m / config.speed_of_light


def doppler_hz_to_velocity_mps(doppler_hz: float, config: RadarConfig) -> float:
    return doppler_hz * config.wavelength_m / 2.0


def velocity_mps_to_doppler_hz(velocity_mps: float, config: RadarConfig) -> float:
    return 2.0 * velocity_mps / config.wavelength_m


def build_range_axis_m(num_bins: int, config: RadarConfig) -> np.ndarray:
    delays = np.arange(num_bins) / config.sample_rate_hz
    return delay_to_bistatic_range_m(delays, config)


def build_doppler_axis_hz(num_bins: int, config: RadarConfig) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(num_bins, d=config.pri_s))


def build_velocity_axis_mps(num_bins: int, config: RadarConfig) -> np.ndarray:
    return doppler_hz_to_velocity_mps(build_doppler_axis_hz(num_bins, config), config)
