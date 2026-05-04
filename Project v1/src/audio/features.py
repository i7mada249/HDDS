from __future__ import annotations

from typing import Any


def extract_baseline_features_from_signal(y: Any, sr: int):
    import librosa
    import numpy as np

    samples = np.asarray(y, dtype=np.float32)
    features: list[float] = []

    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for feat in (mfcc, delta, delta2):
        features.extend(feat.mean(axis=1).tolist())
        features.extend(feat.std(axis=1).tolist())

    sc = librosa.feature.spectral_centroid(y=samples, sr=sr)
    sb = librosa.feature.spectral_bandwidth(y=samples, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(samples)
    rms = librosa.feature.rms(y=samples)
    contrast = librosa.feature.spectral_contrast(y=samples, sr=sr)

    features.extend([float(sc.mean()), float(sc.std())])
    features.extend([float(sb.mean()), float(sb.std())])
    features.extend([float(rolloff.mean()), float(rolloff.std())])
    features.extend([float(zcr.mean()), float(zcr.std())])
    features.extend([float(rms.mean()), float(rms.std())])
    features.extend(contrast.mean(axis=1).tolist())
    features.extend(contrast.std(axis=1).tolist())

    return np.asarray(features, dtype=np.float32)


def extract_yamnet_embedding_from_signal(y: Any, yamnet_model):
    import numpy as np
    import tensorflow as tf

    waveform = tf.convert_to_tensor(np.asarray(y, dtype=np.float32), dtype=tf.float32)
    _, embeddings, _ = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)
