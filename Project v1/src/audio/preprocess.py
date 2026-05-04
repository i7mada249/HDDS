from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from audio.schemas import AudioWindow


TARGET_SR = 16000


def validate_inference_params(window_s: float, hop_s: float, threshold: float) -> None:
    if window_s <= 0:
        raise ValueError("window_s must be greater than 0.")
    if hop_s <= 0:
        raise ValueError("hop_s must be greater than 0.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required for video audio extraction but was not found in PATH. "
            "Install ffmpeg as a system dependency, then retry."
        )


def extract_audio_from_video(
    video_path: Path,
    wav_path: Path,
    target_sr: int = TARGET_SR,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    require_ffmpeg()
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        no_audio_hint = ""
        if "does not contain any stream" in stderr or "Stream map" in stderr:
            no_audio_hint = "\nThe input may not contain an audio stream."
        raise RuntimeError(
            "ffmpeg failed while extracting audio.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr:\n{stderr}{no_audio_hint}"
        )


def load_audio_file(audio_path: Path, target_sr: int = TARGET_SR) -> tuple[Any, int]:
    import librosa

    return librosa.load(audio_path, sr=target_sr, mono=True)


def load_audio_from_video(video_path: Path, target_sr: int = TARGET_SR) -> tuple[Any, int]:
    with tempfile.TemporaryDirectory(prefix="hdds_audio_") as tmp_dir:
        wav_path = Path(tmp_dir) / f"{video_path.stem}_audio.wav"
        extract_audio_from_video(video_path, wav_path, target_sr=target_sr)
        return load_audio_file(wav_path, target_sr=target_sr)


def slice_audio(
    y: Any,
    sr: int,
    window_s: float,
    hop_s: float,
) -> list[AudioWindow]:
    import numpy as np

    if window_s <= 0:
        raise ValueError("window_s must be greater than 0.")
    if hop_s <= 0:
        raise ValueError("hop_s must be greater than 0.")
    if sr <= 0:
        raise ValueError("sr must be greater than 0.")

    samples = np.asarray(y, dtype=np.float32)
    window_samples = max(1, int(round(window_s * sr)))
    hop_samples = max(1, int(round(hop_s * sr)))
    windows: list[AudioWindow] = []

    if len(samples) <= window_samples:
        padded = np.pad(samples, (0, max(0, window_samples - len(samples))))
        return [AudioWindow(start_s=0.0, end_s=len(samples) / sr, samples=padded)]

    for start in range(0, len(samples) - window_samples + 1, hop_samples):
        end = start + window_samples
        windows.append(
            AudioWindow(
                start_s=start / sr,
                end_s=end / sr,
                samples=samples[start:end],
            )
        )

    if windows:
        last_end = windows[-1].end_s
        total_duration = len(samples) / sr
        if total_duration - last_end > 0.25 * window_s:
            windows.append(
                AudioWindow(
                    start_s=max(0.0, total_duration - window_s),
                    end_s=total_duration,
                    samples=samples[-window_samples:],
                )
            )

    return windows
