from __future__ import annotations

import argparse
from pathlib import Path

from audio.classifier import LoadedAudioModels, load_audio_models, predict_window
from audio.persistence import apply_m_of_n_confirmation, validate_confirmation_params
from audio.preprocess import (
    TARGET_SR,
    load_audio_from_video,
    slice_audio,
    validate_inference_params,
)
from audio.report import build_text_report, save_inference_log
from audio.schemas import AudioSegmentPrediction


SegmentPrediction = AudioSegmentPrediction


def run_video_inference(
    video_path: Path,
    window_s: float,
    hop_s: float,
    threshold: float,
    use_baseline: bool,
    use_yamnet: bool,
    show_plot: bool,
    weights: dict[str, float] | None = None,
    confirm_m: int | None = None,
    confirm_n: int | None = None,
) -> tuple[list[AudioSegmentPrediction], Path, LoadedAudioModels]:
    validate_inference_params(window_s=window_s, hop_s=hop_s, threshold=threshold)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if confirm_m is not None or confirm_n is not None:
        if confirm_m is None or confirm_n is None:
            raise ValueError("confirm_m and confirm_n must be provided together.")
        validate_confirmation_params(confirm_m=confirm_m, confirm_n=confirm_n)

    loaded_models = load_audio_models(
        use_baseline=use_baseline,
        use_yamnet=use_yamnet,
    )
    y, sr = load_audio_from_video(video_path, target_sr=TARGET_SR)
    windows = slice_audio(y=y, sr=sr, window_s=window_s, hop_s=hop_s)
    predictions = [
        predict_window(
            window=window,
            sr=sr,
            loaded_models=loaded_models,
            threshold=threshold,
            weights=weights,
        )
        for window in windows
    ]
    if confirm_m is not None and confirm_n is not None:
        predictions = apply_m_of_n_confirmation(
            predictions=predictions,
            confirm_m=confirm_m,
            confirm_n=confirm_n,
        )

    if show_plot:
        plot_predictions(video_path, predictions, threshold)

    log_path = save_inference_log(
        video_path=video_path,
        predictions=predictions,
        threshold=threshold,
        window_s=window_s,
        hop_s=hop_s,
        loaded_models=loaded_models,
    )
    return predictions, log_path, loaded_models


def plot_predictions(
    video_path: Path,
    predictions: list[AudioSegmentPrediction],
    threshold: float,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    times = [0.5 * (item.start_s + item.end_s) for item in predictions]
    audio_scores = [item.audio_score for item in predictions]
    baseline_scores = [
        np.nan if item.baseline_probability is None else item.baseline_probability
        for item in predictions
    ]
    yamnet_scores = [
        np.nan if item.yamnet_probability is None else item.yamnet_probability
        for item in predictions
    ]

    plt.figure(figsize=(12, 5))
    if not np.all(np.isnan(baseline_scores)):
        plt.plot(times, baseline_scores, label="Baseline MFCC probability", alpha=0.7)
    if not np.all(np.isnan(yamnet_scores)):
        plt.plot(times, yamnet_scores, label="YAMNet probability", alpha=0.7)
    plt.plot(times, audio_scores, label="Audio probability", linewidth=2.0, color="black")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold {threshold:.2f}")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Drone probability")
    plt.title(f"Drone-audio inference for {video_path.name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test the trained audio drone detector on a video file."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--window-s",
        type=float,
        default=3.0,
        help="Segment window length in seconds.",
    )
    parser.add_argument(
        "--hop-s",
        type=float,
        default=0.5,
        help="Sliding hop length in seconds.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Drone decision threshold on audio probability.",
    )
    parser.add_argument(
        "--model",
        choices=("both", "baseline", "yamnet"),
        default="both",
        help="Which trained model(s) to use during inference.",
    )
    parser.add_argument(
        "--baseline-weight",
        type=float,
        default=1.0,
        help="Fusion weight for the baseline model when both models are available.",
    )
    parser.add_argument(
        "--yamnet-weight",
        type=float,
        default=1.0,
        help="Fusion weight for the YAMNet model when both models are available.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable the probability plot.",
    )
    parser.add_argument(
        "--confirm-m",
        type=int,
        default=None,
        help="Require M positive windows before reporting a confirmed audio alert.",
    )
    parser.add_argument(
        "--confirm-n",
        type=int,
        default=None,
        help="Confirmation window size N for --confirm-m.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    use_baseline = args.model in {"both", "baseline"}
    use_yamnet = args.model in {"both", "yamnet"}
    weights = {
        "baseline": args.baseline_weight,
        "yamnet": args.yamnet_weight,
    }

    predictions, log_path, loaded_models = run_video_inference(
        video_path=args.video,
        window_s=args.window_s,
        hop_s=args.hop_s,
        threshold=args.threshold,
        use_baseline=use_baseline,
        use_yamnet=use_yamnet,
        show_plot=not args.no_plot,
        weights=weights,
        confirm_m=args.confirm_m,
        confirm_n=args.confirm_n,
    )
    report = build_text_report(
        video_path=args.video,
        predictions=predictions,
        threshold=args.threshold,
        window_s=args.window_s,
        hop_s=args.hop_s,
        loaded_models=loaded_models,
    )
    print(report)
    print(f"\nSaved run log: {log_path}")


if __name__ == "__main__":
    main()
