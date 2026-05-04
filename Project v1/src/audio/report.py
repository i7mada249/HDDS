from __future__ import annotations

from pathlib import Path

from audio.classifier import LoadedAudioModels
from audio.logging_utils import write_run_log
from audio.schemas import AudioSegmentPrediction


def build_text_report(
    video_path: Path,
    predictions: list[AudioSegmentPrediction],
    threshold: float,
    window_s: float,
    hop_s: float,
    loaded_models: LoadedAudioModels,
) -> str:
    audio_scores = [item.audio_score for item in predictions]
    positive_segments = [item for item in predictions if item.label == "drone"]
    lines = [
        "Audio inference report",
        f"Video: {video_path}",
        f"Segments: {len(predictions)}",
        f"Window (s): {window_s:.2f}",
        f"Hop (s): {hop_s:.2f}",
        f"Threshold: {threshold:.2f}",
        f"Baseline enabled: {loaded_models.baseline_enabled}",
        f"YAMNet enabled: {loaded_models.yamnet_enabled}",
        f"Positive segments: {len(positive_segments)}",
        f"Max audio probability: {max(audio_scores, default=0.0):.4f}",
        f"Mean audio probability: {sum(audio_scores) / len(audio_scores) if audio_scores else 0.0:.4f}",
    ]

    if loaded_models.load_notes:
        lines.extend(
            [
                "",
                "Model load notes",
                "----------------",
                *loaded_models.load_notes,
            ]
        )

    lines.extend(
        [
            "",
            "Per-segment predictions",
            "-----------------------",
        ]
    )

    for idx, item in enumerate(predictions, start=1):
        baseline_text = (
            "n/a" if item.baseline_probability is None else f"{item.baseline_probability:.4f}"
        )
        yamnet_text = (
            "n/a" if item.yamnet_probability is None else f"{item.yamnet_probability:.4f}"
        )
        lines.append(
            f"{idx:03d}. "
            f"{item.start_s:7.2f}-{item.end_s:7.2f}s | "
            f"baseline={baseline_text} | "
            f"yamnet={yamnet_text} | "
            f"audio={item.audio_score:.4f} | "
            f"label={item.label}"
        )

    return "\n".join(lines)


def save_inference_log(
    video_path: Path,
    predictions: list[AudioSegmentPrediction],
    threshold: float,
    window_s: float,
    hop_s: float,
    loaded_models: LoadedAudioModels,
) -> Path:
    report = build_text_report(
        video_path=video_path,
        predictions=predictions,
        threshold=threshold,
        window_s=window_s,
        hop_s=hop_s,
        loaded_models=loaded_models,
    )
    return write_run_log(run_type="audio_video_test", name=video_path.stem, content=report)
