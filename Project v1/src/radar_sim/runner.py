from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .channel import simulate_surveillance_matrix
from .constants import AppConfig, ScenarioConfig, load_app_config
from .detection import DetectionResult, ca_cfar_2d
from .logging_utils import project_root, write_run_log
from .metrics import TruthTarget, match_detection_to_truth, scenario_truth
from .plotting import plot_processing_summary
from .processing import ProcessingResult, process_reference_and_surveillance
from .scenarios import get_scenario, list_scenarios
from .waveform import generate_reference_matrix


@dataclass(frozen=True)
class ScenarioRunResult:
    scenario: ScenarioConfig
    truths: tuple[TruthTarget, ...]
    processing: ProcessingResult
    detections: DetectionResult

def default_config_path() -> Path:
    return project_root() / "configs" / "default.yaml"


def format_report(
    processing: ProcessingResult,
    detections: DetectionResult,
    truths: tuple,
) -> str:
    lines = [
        "Simulation report",
        f"  Range bins: {processing.range_doppler_map.shape[1]}",
        f"  Doppler bins: {processing.range_doppler_map.shape[0]}",
        f"  Detections: {len(detections.detections)}",
        f"  Truth targets: {len(truths)}",
    ]

    if truths:
        lines.append("  Truth:")
        for idx, truth in enumerate(truths, start=1):
            lines.append(
                "    "
                f"{idx}. range={truth.range_m:.2f} m, "
                f"doppler={truth.doppler_hz:.2f} Hz, "
                f"velocity={truth.velocity_mps:.2f} m/s"
            )

    if detections.detections:
        lines.append("  Detections:")
        for idx, detection in enumerate(detections.detections, start=1):
            matched = match_detection_to_truth(
                detection=detection,
                truths=truths,
                range_tolerance_m=30.0,
                doppler_tolerance_hz=25.0,
            )
            lines.append(
                "    "
                f"{idx}. range={detection.range_m:.2f} m, "
                f"doppler={detection.doppler_hz:.2f} Hz, "
                f"velocity={detection.velocity_mps:.2f} m/s, "
                f"peak={detection.peak_power_db:.2f} dB, "
                f"matched={matched}"
            )
    return "\n".join(lines)


def execute_scenario(
    config: AppConfig,
    scenario: ScenarioConfig,
) -> ScenarioRunResult:
    rng = np.random.default_rng(config.seed)

    reference = generate_reference_matrix(config=config.radar, rng=rng)
    surveillance = simulate_surveillance_matrix(
        reference=reference,
        config=config.radar,
        scenario=scenario,
        rng=rng,
    )
    processing = process_reference_and_surveillance(
        reference=reference,
        surveillance=surveillance,
        config=config.radar,
    )
    detections = ca_cfar_2d(
        range_doppler_map=processing.range_doppler_map,
        range_axis_m=processing.range_axis_m,
        doppler_axis_hz=processing.doppler_axis_hz,
        velocity_axis_mps=processing.velocity_axis_mps,
        config=config.cfar,
    )
    truths = scenario_truth(scenario, config.radar)

    return ScenarioRunResult(
        scenario=scenario,
        truths=truths,
        processing=processing,
        detections=detections,
    )


def run_named_scenario(
    config: AppConfig,
    scenario_name: str,
    show_plots: bool = True,
) -> ScenarioRunResult:
    scenario = get_scenario(config, scenario_name)
    result = execute_scenario(config=config, scenario=scenario)
    report_text = format_report(
        processing=result.processing,
        detections=result.detections,
        truths=result.truths,
    )
    print(report_text)

    log_path = write_run_log(
        run_type="runner",
        name=scenario_name,
        content=(
            f"Scenario key: {scenario_name}\n"
            f"Scenario name: {result.scenario.name}\n\n"
            f"{report_text}"
        ),
    )
    print(f"\nSaved run log: {log_path}")

    if show_plots:
        plot_processing_summary(
            processing=result.processing,
            detections=result.detections,
            scenario_name=result.scenario.name,
        )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a radar simulation scenario.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="single_slow",
        help="Scenario key from the config file.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit.",
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
    config = load_app_config(args.config)

    if args.list_scenarios:
        for name in list_scenarios(config):
            print(name)
        return

    run_named_scenario(
        config=config,
        scenario_name=args.scenario,
        show_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
