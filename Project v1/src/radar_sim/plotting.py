from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .detection import DetectionResult
from .processing import ProcessingResult


def detection_power_db(power_map: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(power_map + 1.0e-18)


def plot_processing_summary(
    processing: ProcessingResult,
    detections: DetectionResult,
    scenario_name: str,
) -> None:
    rd_power_db = detection_power_db(detections.power_map)
    threshold_db = detection_power_db(detections.threshold_map)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(np.real(processing.range_profiles[0]))
    axes[0, 0].set_title("Range Profile - Real Part (Pulse 0)")
    axes[0, 0].set_xlabel("Range Bin")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    im1 = axes[0, 1].imshow(
        rd_power_db,
        aspect="auto",
        origin="lower",
        extent=[
            processing.range_axis_m[0],
            processing.range_axis_m[-1],
            processing.velocity_axis_mps[0],
            processing.velocity_axis_mps[-1],
        ],
        cmap="viridis",
    )
    axes[0, 1].set_title(f"Range-Doppler Map - {scenario_name}")
    axes[0, 1].set_xlabel("Bistatic Range Excess (m)")
    axes[0, 1].set_ylabel("Velocity (m/s)")
    fig.colorbar(im1, ax=axes[0, 1], label="Power (dB)")

    im2 = axes[1, 0].imshow(
        threshold_db,
        aspect="auto",
        origin="lower",
        extent=[
            processing.range_axis_m[0],
            processing.range_axis_m[-1],
            processing.velocity_axis_mps[0],
            processing.velocity_axis_mps[-1],
        ],
        cmap="magma",
    )
    axes[1, 0].set_title("CFAR Threshold Map")
    axes[1, 0].set_xlabel("Bistatic Range Excess (m)")
    axes[1, 0].set_ylabel("Velocity (m/s)")
    fig.colorbar(im2, ax=axes[1, 0], label="Threshold (dB)")

    axes[1, 1].imshow(
        detections.detection_mask,
        aspect="auto",
        origin="lower",
        extent=[
            processing.range_axis_m[0],
            processing.range_axis_m[-1],
            processing.velocity_axis_mps[0],
            processing.velocity_axis_mps[-1],
        ],
        cmap="gray_r",
    )
    axes[1, 1].set_title("CFAR Detection Mask")
    axes[1, 1].set_xlabel("Bistatic Range Excess (m)")
    axes[1, 1].set_ylabel("Velocity (m/s)")

    plt.tight_layout()
    plt.show()
