from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio.features import (
    extract_baseline_features_from_signal,
    extract_yamnet_embedding_from_signal,
)
from audio.schemas import AudioSegmentPrediction, AudioWindow


MODULE_ROOT = Path(__file__).resolve().parent
MODEL_DIR = MODULE_ROOT / "models"
BASELINE_MODEL_NAME = "sound_baseline_mfcc_logreg.joblib"
YAMNET_MODEL_NAME = "sound_yamnet_hgb.joblib"
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"


@dataclass(frozen=True)
class LoadedAudioModels:
    baseline_model: object | None
    yamnet_classifier: object | None
    yamnet_model: object | None
    baseline_enabled: bool
    yamnet_enabled: bool
    load_notes: tuple[str, ...]


def _resolve_model_path(model_name: str, models_dir: Path | None = None) -> Path:
    search_dirs = [models_dir] if models_dir is not None else [MODEL_DIR]
    for directory in search_dirs:
        if directory is None:
            continue
        path = directory / model_name
        if path.exists():
            return path
    return (models_dir or MODEL_DIR) / model_name


def _predict_proba_binary(model: Any, features: Any) -> float:
    import numpy as np

    features_2d = np.asarray(features, dtype=np.float32).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_2d)[0]
        return float(probabilities[-1])

    prediction = model.predict(features_2d)[0]
    return float(prediction)


def fuse_audio_scores(
    baseline_probability: float | None,
    yamnet_probability: float | None,
    weights: dict[str, float] | None = None,
) -> float:
    scores: list[tuple[str, float]] = []
    if baseline_probability is not None:
        scores.append(("baseline", baseline_probability))
    if yamnet_probability is not None:
        scores.append(("yamnet", yamnet_probability))
    if not scores:
        return 0.0

    if not weights:
        return sum(score for _, score in scores) / len(scores)

    numerator = 0.0
    denominator = 0.0
    for name, score in scores:
        weight = max(0.0, float(weights.get(name, 0.0)))
        numerator += weight * score
        denominator += weight
    if denominator <= 0:
        return sum(score for _, score in scores) / len(scores)
    return numerator / denominator


def load_audio_models(
    use_baseline: bool,
    use_yamnet: bool,
    models_dir: Path | None = None,
) -> LoadedAudioModels:
    import joblib

    baseline_model = None
    yamnet_classifier = None
    yamnet_model = None
    baseline_enabled = False
    yamnet_enabled = False
    load_notes: list[str] = []

    if use_baseline:
        baseline_path = _resolve_model_path(BASELINE_MODEL_NAME, models_dir=models_dir)
        try:
            baseline_model = joblib.load(baseline_path)
            baseline_enabled = True
        except Exception as exc:  # pragma: no cover
            message = f"Baseline model could not be loaded from {baseline_path}: {exc}"
            if use_yamnet:
                warnings.warn(message)
                load_notes.append(message)
            else:
                raise RuntimeError(message) from exc

    if use_yamnet:
        yamnet_path = _resolve_model_path(YAMNET_MODEL_NAME, models_dir=models_dir)
        try:
            import tensorflow_hub as hub

            yamnet_classifier = joblib.load(yamnet_path)
            yamnet_model = hub.load(YAMNET_HANDLE)
            yamnet_enabled = True
        except Exception as exc:  # pragma: no cover
            message = (
                f"YAMNet classifier could not be loaded from {yamnet_path}: {exc}. "
                "This usually means the artifact was serialized with a different "
                "numpy/scikit-learn combination, or TensorFlow Hub could not load "
                "the YAMNet model. Re-export it in the current environment or run "
                "with --model baseline."
            )
            if baseline_enabled:
                warnings.warn(message)
                load_notes.append(message)
                yamnet_classifier = None
                yamnet_model = None
                yamnet_enabled = False
            else:
                raise RuntimeError(message) from exc

    if not baseline_enabled and not yamnet_enabled:
        raise RuntimeError("No compatible audio model could be loaded.")

    return LoadedAudioModels(
        baseline_model=baseline_model,
        yamnet_classifier=yamnet_classifier,
        yamnet_model=yamnet_model,
        baseline_enabled=baseline_enabled,
        yamnet_enabled=yamnet_enabled,
        load_notes=tuple(load_notes),
    )


def predict_window(
    window: AudioWindow,
    sr: int,
    loaded_models: LoadedAudioModels,
    threshold: float,
    weights: dict[str, float] | None = None,
) -> AudioSegmentPrediction:
    baseline_probability = None
    yamnet_probability = None

    if loaded_models.baseline_model is not None:
        baseline_features = extract_baseline_features_from_signal(window.samples, sr)
        baseline_probability = _predict_proba_binary(
            loaded_models.baseline_model,
            baseline_features,
        )

    if loaded_models.yamnet_model is not None and loaded_models.yamnet_classifier is not None:
        yamnet_embedding = extract_yamnet_embedding_from_signal(
            window.samples,
            loaded_models.yamnet_model,
        )
        yamnet_probability = _predict_proba_binary(
            loaded_models.yamnet_classifier,
            yamnet_embedding,
        )

    audio_score = fuse_audio_scores(
        baseline_probability=baseline_probability,
        yamnet_probability=yamnet_probability,
        weights=weights,
    )
    label = "drone" if audio_score >= threshold else "no_drone"

    return AudioSegmentPrediction(
        start_s=window.start_s,
        end_s=window.end_s,
        baseline_probability=baseline_probability,
        yamnet_probability=yamnet_probability,
        audio_score=audio_score,
        label=label,
        model_notes=loaded_models.load_notes,
    )
