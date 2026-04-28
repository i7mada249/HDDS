from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from .constants import CFARConfig


@dataclass(frozen=True)
class Detection:
    range_bin: int
    doppler_bin: int
    range_m: float
    doppler_hz: float
    velocity_mps: float
    peak_power_db: float


@dataclass(frozen=True)
class DetectionResult:
    power_map: np.ndarray
    threshold_map: np.ndarray
    detection_mask: np.ndarray
    detections: tuple[Detection, ...]


def build_cfar_kernel(config: CFARConfig) -> tuple[np.ndarray, int]:
    rows = 2 * (config.guard_cells_doppler + config.train_cells_doppler) + 1
    cols = 2 * (config.guard_cells_range + config.train_cells_range) + 1

    kernel = np.ones((rows, cols), dtype=float)
    row_start = config.train_cells_doppler
    row_stop = row_start + 2 * config.guard_cells_doppler + 1
    col_start = config.train_cells_range
    col_stop = col_start + 2 * config.guard_cells_range + 1
    kernel[row_start:row_stop, col_start:col_stop] = 0.0

    num_train = int(np.sum(kernel))
    return kernel, num_train


def ca_cfar_2d(
    range_doppler_map: np.ndarray,
    range_axis_m: np.ndarray,
    doppler_axis_hz: np.ndarray,
    velocity_axis_mps: np.ndarray,
    config: CFARConfig,
) -> DetectionResult:
    power_map = np.abs(range_doppler_map) ** 2
    kernel, num_train = build_cfar_kernel(config)

    noise_estimate = signal.convolve2d(
        power_map,
        kernel / num_train,
        mode="same",
        boundary="symm",
    )
    alpha = num_train * ((config.pfa ** (-1.0 / num_train)) - 1.0)
    threshold_map = alpha * noise_estimate
    detection_mask = power_map > threshold_map

    labels, num_regions = ndimage.label(detection_mask)
    detections: list[Detection] = []

    for label_id in range(1, num_regions + 1):
        region = labels == label_id
        if not np.any(region):
            continue

        region_power = np.where(region, power_map, 0.0)
        doppler_bin, range_bin = np.unravel_index(
            np.argmax(region_power), region_power.shape
        )
        peak_power = power_map[doppler_bin, range_bin]

        detections.append(
            Detection(
                range_bin=int(range_bin),
                doppler_bin=int(doppler_bin),
                range_m=float(range_axis_m[range_bin]),
                doppler_hz=float(doppler_axis_hz[doppler_bin]),
                velocity_mps=float(velocity_axis_mps[doppler_bin]),
                peak_power_db=float(10.0 * np.log10(peak_power + 1.0e-18)),
            )
        )

    detections.sort(key=lambda item: item.peak_power_db, reverse=True)

    return DetectionResult(
        power_map=power_map,
        threshold_map=threshold_map,
        detection_mask=detection_mask,
        detections=tuple(detections),
    )
