from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AudioWindow:
    start_s: float
    end_s: float
    samples: Any


@dataclass(frozen=True)
class AudioFusionEvent:
    t_start: float
    t_end: float
    audio_score: float
    final_label: str
    modality: str = "audio"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioSegmentPrediction:
    start_s: float
    end_s: float
    baseline_probability: float | None
    yamnet_probability: float | None
    audio_score: float
    label: str
    model_notes: tuple[str, ...] = ()

    @property
    def fused_probability(self) -> float:
        return self.audio_score

    def to_fusion_event(self) -> AudioFusionEvent:
        return AudioFusionEvent(
            t_start=self.start_s,
            t_end=self.end_s,
            audio_score=self.audio_score,
            final_label=self.label,
            metadata={
                "baseline_probability": self.baseline_probability,
                "yamnet_probability": self.yamnet_probability,
                "model_notes": list(self.model_notes),
            },
        )
