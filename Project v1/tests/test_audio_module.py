from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audio.classifier import fuse_audio_scores, load_audio_models
from audio.calibration import choose_threshold_by_f1, evaluate_threshold
from audio.manifest import load_manifest, resolve_manifest_path
from audio.persistence import apply_m_of_n_confirmation
from audio.preprocess import slice_audio, validate_inference_params
from audio.report import build_text_report
from audio.schemas import AudioSegmentPrediction
from audio.video_test import build_parser


class FakeModel:
    pass


def test_slice_audio_pads_short_clip() -> None:
    samples = np.ones(4, dtype=np.float32)

    windows = slice_audio(samples, sr=4, window_s=2.0, hop_s=1.0)

    assert len(windows) == 1
    assert windows[0].start_s == 0.0
    assert windows[0].end_s == 1.0
    assert len(windows[0].samples) == 8


def test_slice_audio_adds_tail_window() -> None:
    samples = np.arange(11, dtype=np.float32)

    windows = slice_audio(samples, sr=10, window_s=0.4, hop_s=0.4)

    assert [(round(w.start_s, 2), round(w.end_s, 2)) for w in windows] == [
        (0.0, 0.4),
        (0.4, 0.8),
        (0.7, 1.1),
    ]


def test_validate_inference_params_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        validate_inference_params(window_s=0.0, hop_s=0.5, threshold=0.5)
    with pytest.raises(ValueError):
        validate_inference_params(window_s=1.0, hop_s=-0.5, threshold=0.5)
    with pytest.raises(ValueError):
        validate_inference_params(window_s=1.0, hop_s=0.5, threshold=1.5)


def test_fuse_audio_scores_uses_available_scores_and_weights() -> None:
    assert fuse_audio_scores(0.2, None) == pytest.approx(0.2)
    assert fuse_audio_scores(None, 0.8) == pytest.approx(0.8)
    assert fuse_audio_scores(0.2, 0.8) == pytest.approx(0.5)
    assert fuse_audio_scores(
        0.2,
        0.8,
        weights={"baseline": 3.0, "yamnet": 1.0},
    ) == pytest.approx(0.35)


def test_load_audio_models_falls_back_when_yamnet_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "sound_baseline_mfcc_logreg.joblib").write_text("baseline")
    (tmp_path / "sound_yamnet_hgb.joblib").write_text("yamnet")

    def fake_load(path: Path):
        if path.name == "sound_baseline_mfcc_logreg.joblib":
            return FakeModel()
        raise AssertionError("YAMNet classifier should not load without tensorflow_hub")

    monkeypatch.setitem(sys.modules, "joblib", SimpleNamespace(load=fake_load))
    monkeypatch.setitem(
        sys.modules,
        "tensorflow_hub",
        SimpleNamespace(load=lambda _handle: FakeModel()),
    )

    loaded = load_audio_models(
        use_baseline=True,
        use_yamnet=True,
        models_dir=tmp_path,
    )

    assert loaded.baseline_enabled is True
    assert loaded.yamnet_enabled is False
    assert loaded.baseline_model is not None
    assert loaded.load_notes


def test_build_text_report_includes_model_flags_and_segments() -> None:
    loaded = SimpleNamespace(
        baseline_enabled=True,
        yamnet_enabled=False,
        load_notes=("yamnet unavailable",),
    )
    predictions = [
        AudioSegmentPrediction(
            start_s=0.0,
            end_s=1.0,
            baseline_probability=0.7,
            yamnet_probability=None,
            audio_score=0.7,
            label="drone",
        )
    ]

    report = build_text_report(
        video_path=Path("sample.mp4"),
        predictions=predictions,
        threshold=0.5,
        window_s=1.0,
        hop_s=0.5,
        loaded_models=loaded,
    )

    assert "Baseline enabled: True" in report
    assert "YAMNet enabled: False" in report
    assert "Positive segments: 1" in report
    assert "audio=0.7000" in report
    assert "yamnet unavailable" in report


def test_parser_accepts_model_choices() -> None:
    parser = build_parser()

    args = parser.parse_args(["video.mp4", "--model", "baseline", "--no-plot"])

    assert args.video == Path("video.mp4")
    assert args.model == "baseline"
    assert args.no_plot is True


def test_audio_prediction_converts_to_fusion_event() -> None:
    prediction = AudioSegmentPrediction(
        start_s=2.0,
        end_s=3.0,
        baseline_probability=0.4,
        yamnet_probability=0.8,
        audio_score=0.6,
        label="drone",
        model_notes=("note",),
    )

    event = prediction.to_fusion_event()

    assert event.t_start == 2.0
    assert event.t_end == 3.0
    assert event.audio_score == 0.6
    assert event.final_label == "drone"
    assert event.metadata["baseline_probability"] == 0.4


def test_m_of_n_confirmation_suppresses_isolated_spikes() -> None:
    predictions = [
        AudioSegmentPrediction(i, i + 1, None, None, score, label)
        for i, (score, label) in enumerate(
            [
                (0.9, "drone"),
                (0.1, "no_drone"),
                (0.8, "drone"),
                (0.7, "drone"),
            ]
        )
    ]

    confirmed = apply_m_of_n_confirmation(predictions, confirm_m=2, confirm_n=3)

    assert [item.label for item in confirmed] == [
        "no_drone",
        "no_drone",
        "drone",
        "drone",
    ]


def test_threshold_calibration_selects_useful_threshold() -> None:
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]

    metrics = choose_threshold_by_f1(scores, labels)

    assert metrics.threshold == pytest.approx(0.8)
    assert metrics.f1 == pytest.approx(1.0)
    assert evaluate_threshold(scores, labels, threshold=0.5).true_positives == 2


def test_manifest_remaps_colab_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,label,source,group,category\n"
        "/content/hdds_audio/work/wav_data/dads_drone/a.wav,1,DADS,g,c\n",
        encoding="utf-8",
    )

    rows = load_manifest(manifest, local_data_root=tmp_path / "wav_data")

    assert rows[0].resolved_path == tmp_path / "wav_data" / "dads_drone" / "a.wav"
    assert resolve_manifest_path(rows[0].path) is None
