from __future__ import annotations

from audio.schemas import AudioSegmentPrediction


def validate_confirmation_params(confirm_m: int, confirm_n: int) -> None:
    if confirm_m <= 0:
        raise ValueError("confirm_m must be greater than 0.")
    if confirm_n <= 0:
        raise ValueError("confirm_n must be greater than 0.")
    if confirm_m > confirm_n:
        raise ValueError("confirm_m must be less than or equal to confirm_n.")


def apply_m_of_n_confirmation(
    predictions: list[AudioSegmentPrediction],
    confirm_m: int,
    confirm_n: int,
) -> list[AudioSegmentPrediction]:
    validate_confirmation_params(confirm_m=confirm_m, confirm_n=confirm_n)
    confirmed: list[AudioSegmentPrediction] = []

    for idx, item in enumerate(predictions):
        window = predictions[max(0, idx - confirm_n + 1) : idx + 1]
        positive_count = sum(1 for candidate in window if candidate.label == "drone")
        label = "drone" if positive_count >= confirm_m else "no_drone"
        confirmed.append(
            AudioSegmentPrediction(
                start_s=item.start_s,
                end_s=item.end_s,
                baseline_probability=item.baseline_probability,
                yamnet_probability=item.yamnet_probability,
                audio_score=item.audio_score,
                label=label,
                model_notes=(
                    *item.model_notes,
                    f"M/N confirmation: {positive_count}/{len(window)} positives, required {confirm_m}/{confirm_n}",
                ),
            )
        )

    return confirmed
