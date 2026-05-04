from __future__ import annotations

from audio.video_test import (
    SegmentPrediction,
    build_parser,
    build_text_report,
    main,
    plot_predictions,
    run_video_inference,
)

__all__ = [
    "SegmentPrediction",
    "build_parser",
    "build_text_report",
    "main",
    "plot_predictions",
    "run_video_inference",
]


if __name__ == "__main__":
    main()
