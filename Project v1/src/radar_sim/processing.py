from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import RadarConfig
from .geometry import (
    build_doppler_axis_hz,
    build_range_axis_m,
    build_velocity_axis_mps,
)


@dataclass(frozen=True)
class ProcessingResult:
    range_profiles: np.ndarray
    filtered_profiles: np.ndarray
    range_doppler_map: np.ndarray
    range_axis_m: np.ndarray
    doppler_axis_hz: np.ndarray
    velocity_axis_mps: np.ndarray


def remove_cyclic_prefix(data: np.ndarray, config: RadarConfig) -> np.ndarray:
    return data[:, config.cyclic_prefix :]


def form_range_profiles(
    reference: np.ndarray, surveillance: np.ndarray, config: RadarConfig
) -> np.ndarray:
    ref_symbols = remove_cyclic_prefix(reference, config)
    surv_symbols = remove_cyclic_prefix(surveillance, config)

    ref_freq = np.fft.fft(ref_symbols, axis=1)
    surv_freq = np.fft.fft(surv_symbols, axis=1)
    matched = surv_freq / (ref_freq + config.epsilon)
    return np.fft.ifft(matched, axis=1)


def suppress_stationary_components(range_profiles: np.ndarray) -> np.ndarray:
    stationary_estimate = np.mean(range_profiles, axis=0, keepdims=True)
    return range_profiles - stationary_estimate


def form_range_doppler_map(
    filtered_profiles: np.ndarray, config: RadarConfig
) -> np.ndarray:
    slow_time_window = np.hamming(config.num_pulses)[:, np.newaxis]
    fast_time_window = np.hamming(filtered_profiles.shape[1])[np.newaxis, :]
    windowed = filtered_profiles * slow_time_window * fast_time_window
    return np.fft.fftshift(np.fft.fft(windowed, axis=0), axes=0)


def process_reference_and_surveillance(
    reference: np.ndarray, surveillance: np.ndarray, config: RadarConfig
) -> ProcessingResult:
    range_profiles = form_range_profiles(reference, surveillance, config)
    filtered_profiles = suppress_stationary_components(range_profiles)
    range_doppler_map = form_range_doppler_map(filtered_profiles, config)

    return ProcessingResult(
        range_profiles=range_profiles,
        filtered_profiles=filtered_profiles,
        range_doppler_map=range_doppler_map,
        range_axis_m=build_range_axis_m(range_profiles.shape[1], config),
        doppler_axis_hz=build_doppler_axis_hz(range_doppler_map.shape[0], config),
        velocity_axis_mps=build_velocity_axis_mps(range_doppler_map.shape[0], config),
    )
