# Audio Model Artifacts

This directory contains runtime artifacts used by `python -m audio.video_test`.

- `sound_baseline_mfcc_logreg.joblib`: baseline MFCC-style classifier.
- `sound_yamnet_hgb.joblib`: classifier trained on YAMNet embeddings.

If `sound_yamnet_hgb.joblib` fails to unpickle with a `BitGenerator` error, re-export it in the same Python, NumPy, and scikit-learn stack pinned by `Project v1/requirements.txt`.
