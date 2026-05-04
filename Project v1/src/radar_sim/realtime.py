from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .channel import simulate_surveillance_matrix
from .constants import AppConfig, RadarConfig, ScenarioConfig, Target, load_app_config
from .detection import DetectionResult, ca_cfar_2d
from .geometry import bistatic_range_to_delay_s, velocity_mps_to_doppler_hz
from .logging_utils import write_run_log
from .plotting import detection_power_db
from .processing import ProcessingResult, process_reference_and_surveillance
from .runner import default_config_path
from .waveform import generate_reference_matrix


@dataclass(frozen=True)
class MovingTargetSpec:
    initial_range_m: float
    radial_velocity_mps: float
    amplitude_db: float
    name: str


@dataclass(frozen=True)
class RealtimeScenarioConfig:
    name: str
    duration_s: float
    frame_interval_s: float
    noise_power_db: float
    clutter_amplitude_db: float
    targets: tuple[MovingTargetSpec, ...]


def range_at_time_m(target: MovingTargetSpec, elapsed_s: float) -> float:
    return max(0.0, target.initial_range_m - target.radial_velocity_mps * elapsed_s)


def build_instantaneous_scenario(
    realtime: RealtimeScenarioConfig,
    radar: RadarConfig,
    elapsed_s: float,
) -> ScenarioConfig:
    targets = []
    for target in realtime.targets:
        current_range = range_at_time_m(target, elapsed_s)
        delay_s = bistatic_range_to_delay_s(current_range, radar)
        doppler_hz = velocity_mps_to_doppler_hz(target.radial_velocity_mps, radar)
        targets.append(
            Target(
                delay_s=delay_s,
                doppler_hz=doppler_hz,
                amplitude_db=target.amplitude_db,
            )
        )

    return ScenarioConfig(
        name=realtime.name,
        targets=tuple(targets),
        noise_power_db=realtime.noise_power_db,
        clutter_amplitude_db=realtime.clutter_amplitude_db,
    )


def execute_instantaneous_scenario(
    app_config: AppConfig,
    scenario: ScenarioConfig,
    rng: np.random.Generator,
) -> tuple[ProcessingResult, DetectionResult]:
    reference = generate_reference_matrix(config=app_config.radar, rng=rng)
    surveillance = simulate_surveillance_matrix(
        reference=reference,
        config=app_config.radar,
        scenario=scenario,
        rng=rng,
    )
    processing = process_reference_and_surveillance(
        reference=reference,
        surveillance=surveillance,
        config=app_config.radar,
    )
    detections = ca_cfar_2d(
        range_doppler_map=processing.range_doppler_map,
        range_axis_m=processing.range_axis_m,
        doppler_axis_hz=processing.doppler_axis_hz,
        velocity_axis_mps=processing.velocity_axis_mps,
        config=app_config.cfar,
    )
    return processing, detections


def _prompt_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        print("Value is required.")


def _prompt_float(
    prompt: str,
    default: float | None = None,
    minimum: float | None = None,
) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Enter a numeric value.")
                continue

        if minimum is not None and value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        return value


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")


def _prompt_moving_target(index: int) -> MovingTargetSpec:
    print(f"\nMoving target {index}")
    name = _prompt_text("  Name", default=f"Target {index}")
    initial_range_m = _prompt_float("  Initial distance (m)", minimum=0.0)
    radial_velocity_mps = _prompt_float(
        "  Closing speed (m/s). Positive = approaching, negative = receding",
        default=10.0,
    )
    amplitude_db = _prompt_float(
        "  Target amplitude (dB, less negative = stronger)",
        default=-18.0,
    )
    return MovingTargetSpec(
        initial_range_m=initial_range_m,
        radial_velocity_mps=radial_velocity_mps,
        amplitude_db=amplitude_db,
        name=name,
    )


def build_realtime_scenario(config_path: Path) -> tuple[AppConfig, RealtimeScenarioConfig]:
    app_config = load_app_config(config_path)
    radar = app_config.radar

    print("Passive Radar Realtime TUI")
    print("==========================")
    print("Positive speed means the target is approaching, so distance decreases over time.\n")

    scenario_name = _prompt_text("Scenario name", default="Realtime Moving Target")
    duration_s = _prompt_float("Simulation duration (s)", default=12.0, minimum=0.1)
    frame_interval_s = _prompt_float(
        "Refresh interval (s) for numeric/plot updates",
        default=max(0.25, radar.num_pulses * radar.pri_s),
        minimum=0.05,
    )
    noise_power_db = _prompt_float("Noise power (dB)", default=radar.noise_power_db)
    clutter_amplitude_db = _prompt_float(
        "Clutter amplitude (dB)",
        default=radar.clutter_amplitude_db,
    )

    targets: list[MovingTargetSpec] = []
    index = 1
    while True:
        targets.append(_prompt_moving_target(index))
        index += 1
        if not _prompt_yes_no("Add another moving target?", default=False):
            break

    return app_config, RealtimeScenarioConfig(
        name=scenario_name,
        duration_s=duration_s,
        frame_interval_s=frame_interval_s,
        noise_power_db=noise_power_db,
        clutter_amplitude_db=clutter_amplitude_db,
        targets=tuple(targets),
    )


def realtime_summary_text(realtime: RealtimeScenarioConfig) -> str:
    lines = [
        "Realtime scenario summary",
        "-------------------------",
        f"Name: {realtime.name}",
        f"Duration: {realtime.duration_s:.2f} s",
        f"Refresh interval: {realtime.frame_interval_s:.2f} s",
        f"Noise power (dB): {realtime.noise_power_db:.2f}",
        f"Clutter amplitude (dB): {realtime.clutter_amplitude_db:.2f}",
        f"Targets: {len(realtime.targets)}",
    ]
    for idx, target in enumerate(realtime.targets, start=1):
        lines.append(
            f"  {idx}. {target.name}: "
            f"distance={target.initial_range_m:.2f} m, "
            f"closing_speed={target.radial_velocity_mps:.2f} m/s, "
            f"amplitude={target.amplitude_db:.2f} dB"
        )
    return "\n".join(lines)


def print_realtime_summary(realtime: RealtimeScenarioConfig) -> None:
    print()
    print(realtime_summary_text(realtime))


def _build_live_figure(initial_range_extent: float) -> tuple:
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Passive Radar Realtime Scenario")

    rd_im = axes[0, 0].imshow(
        np.zeros((2, 2)),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axes[0, 0].set_title("Current Range-Doppler Map")
    axes[0, 0].set_xlabel("Bistatic Range Excess (m)")
    axes[0, 0].set_ylabel("Velocity (m/s)")
    fig.colorbar(rd_im, ax=axes[0, 0], label="Power (dB)")

    det_im = axes[0, 1].imshow(
        np.zeros((2, 2)),
        aspect="auto",
        origin="lower",
        cmap="gray_r",
    )
    axes[0, 1].set_title("Current Detection Mask")
    axes[0, 1].set_xlabel("Bistatic Range Excess (m)")
    axes[0, 1].set_ylabel("Velocity (m/s)")

    axes[1, 0].set_title("Range History")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Range (m)")
    axes[1, 0].set_xlim(0.0, 1.0)
    axes[1, 0].set_ylim(0.0, max(initial_range_extent, 1.0))
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Velocity History / Status")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Velocity (m/s)")
    axes[1, 1].set_xlim(0.0, 1.0)
    axes[1, 1].grid(True, alpha=0.3)
    status_text = axes[1, 1].text(
        0.02,
        0.98,
        "",
        va="top",
        ha="left",
        transform=axes[1, 1].transAxes,
        family="monospace",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    plt.tight_layout()
    return fig, axes, rd_im, det_im, status_text


def run_realtime_scenario(
    app_config: AppConfig,
    realtime: RealtimeScenarioConfig,
    show_plots: bool = True,
) -> None:
    rng = np.random.default_rng(app_config.seed)
    times = np.arange(0.0, realtime.duration_s + 1e-12, realtime.frame_interval_s)

    truth_ranges_history: list[list[float]] = [[] for _ in realtime.targets]
    truth_velocity_history: list[list[float]] = [[] for _ in realtime.targets]
    detected_range_history: list[float] = []
    detected_velocity_history: list[float] = []
    detection_count_history: list[int] = []
    log_lines = [
        realtime_summary_text(realtime),
        "",
        "Live updates",
        "------------",
    ]

    fig = axes = rd_im = det_im = status_text = None
    if show_plots:
        max_range = max(target.initial_range_m for target in realtime.targets)
        fig, axes, rd_im, det_im, status_text = _build_live_figure(max_range * 1.1)

    print("\nLive updates")
    print("------------")

    for frame_idx, elapsed_s in enumerate(times):
        scenario = build_instantaneous_scenario(realtime, app_config.radar, elapsed_s)
        processing, detections = execute_instantaneous_scenario(app_config, scenario, rng)

        current_ranges = [
            range_at_time_m(target, elapsed_s) for target in realtime.targets
        ]
        current_velocities = [target.radial_velocity_mps for target in realtime.targets]
        for idx, current_range in enumerate(current_ranges):
            truth_ranges_history[idx].append(current_range)
            truth_velocity_history[idx].append(current_velocities[idx])

        primary_detection = detections.detections[0] if detections.detections else None
        if primary_detection is not None:
            detected_range_history.append(primary_detection.range_m)
            detected_velocity_history.append(primary_detection.velocity_mps)
        else:
            detected_range_history.append(np.nan)
            detected_velocity_history.append(np.nan)
        detection_count_history.append(len(detections.detections))

        target_summary = " | ".join(
            f"{target.name}: R={current_ranges[idx]:.2f}m V={target.radial_velocity_mps:.2f}m/s"
            for idx, target in enumerate(realtime.targets)
        )
        detected_summary = (
            "none"
            if primary_detection is None
            else (
                f"R={primary_detection.range_m:.2f}m "
                f"V={primary_detection.velocity_mps:.2f}m/s "
                f"Power={primary_detection.peak_power_db:.2f}dB"
            )
        )
        frame_line = (
            f"t={elapsed_s:6.2f}s | detections={len(detections.detections):3d} | "
            f"truth: {target_summary} | primary: {detected_summary}"
        )
        print(frame_line)
        log_lines.append(frame_line)

        if show_plots and fig is not None:
            rd_power_db = detection_power_db(detections.power_map)
            rd_im.set_data(rd_power_db)
            rd_im.set_extent(
                [
                    processing.range_axis_m[0],
                    processing.range_axis_m[-1],
                    processing.velocity_axis_mps[0],
                    processing.velocity_axis_mps[-1],
                ]
            )
            axes[0, 0].set_xlim(processing.range_axis_m[0], processing.range_axis_m[-1])
            axes[0, 0].set_ylim(
                processing.velocity_axis_mps[0], processing.velocity_axis_mps[-1]
            )

            det_im.set_data(detections.detection_mask.astype(float))
            det_im.set_extent(
                [
                    processing.range_axis_m[0],
                    processing.range_axis_m[-1],
                    processing.velocity_axis_mps[0],
                    processing.velocity_axis_mps[-1],
                ]
            )
            axes[0, 1].set_xlim(processing.range_axis_m[0], processing.range_axis_m[-1])
            axes[0, 1].set_ylim(
                processing.velocity_axis_mps[0], processing.velocity_axis_mps[-1]
            )

            axes[1, 0].cla()
            axes[1, 0].set_title("Range History")
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Range (m)")
            axes[1, 0].set_xlim(0.0, realtime.duration_s)
            axes[1, 0].set_ylim(
                0.0,
                max(max(target.initial_range_m for target in realtime.targets) * 1.1, 1.0),
            )
            axes[1, 0].grid(True, alpha=0.3)
            for idx, target in enumerate(realtime.targets):
                axes[1, 0].plot(
                    times[: frame_idx + 1],
                    truth_ranges_history[idx],
                    label=f"Truth {target.name}",
                )
            axes[1, 0].plot(
                times[: frame_idx + 1],
                detected_range_history,
                linestyle="--",
                linewidth=2,
                label="Primary detection",
            )
            axes[1, 0].legend(loc="upper right")

            axes[1, 1].cla()
            axes[1, 1].set_title("Velocity History / Status")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Velocity (m/s)")
            axes[1, 1].set_xlim(0.0, realtime.duration_s)
            max_velocity = max(abs(target.radial_velocity_mps) for target in realtime.targets)
            axes[1, 1].set_ylim(-max(max_velocity * 1.3, 1.0), max(max_velocity * 1.3, 1.0))
            axes[1, 1].grid(True, alpha=0.3)
            for idx, target in enumerate(realtime.targets):
                axes[1, 1].plot(
                    times[: frame_idx + 1],
                    truth_velocity_history[idx],
                    label=f"Truth {target.name}",
                )
            axes[1, 1].plot(
                times[: frame_idx + 1],
                detected_velocity_history,
                linestyle="--",
                linewidth=2,
                label="Primary detection",
            )
            axes[1, 1].legend(loc="upper right")
            status_text = axes[1, 1].text(
                0.02,
                0.98,
                (
                    f"frame={frame_idx + 1}/{len(times)}\n"
                    f"time={elapsed_s:.2f}s\n"
                    f"detection_count={len(detections.detections)}\n"
                    f"primary={detected_summary}"
                ),
                va="top",
                ha="left",
                transform=axes[1, 1].transAxes,
                family="monospace",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)

    log_path = write_run_log(
        run_type="realtime",
        name=realtime.name,
        content="\n".join(log_lines),
    )
    print(f"\nSaved run log: {log_path}")

    if show_plots:
        plt.ioff()
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive realtime passive radar simulator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib plots.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    app_config, realtime = build_realtime_scenario(args.config)
    print_realtime_summary(realtime)

    if not _prompt_yes_no("\nRun realtime simulation now?", default=True):
        print("Cancelled.")
        return

    run_realtime_scenario(
        app_config=app_config,
        realtime=realtime,
        show_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
