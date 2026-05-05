# Research Paper: Sound Module For The Hybrid Drone Detection System

## Abstract

This paper documents the sound module used in the Hybrid Drone Detection System
(HDDS), explains the algorithmic decisions behind it, and records the training
evidence that remains in the repository after cleanup. The sound branch is one
of the cleanest parts of the multimodal system because it exists as a reusable
package under `Project v1/src/audio/` and exposes a clear offline inference
workflow for teammate use.

A key repository fact must be stated upfront: the cleaned directory
`Project v1/notebooks/` is not present anymore in the current workspace. The
audio training notebook that remains available is instead located at:

`/home/mo/dev/python/HDDS2/Notebooks/drone_thesis_audio_training.ipynb`

Therefore, this report is based on two evidence sources:

1. the cleaned implementation under `Project v1/src/audio/`,
2. the preserved notebook and model artifacts still present in the repository.

## 1. Problem Definition

The purpose of the sound module is to decide whether a video contains drone
audio and to produce timestamped predictions that can later be fused with radar
and vision outputs.

This is not a live microphone pipeline. It is an offline file-based inference
system. That design is justified for the current project stage because:

- it is easier to reproduce,
- it avoids hardware/audio-driver variability,
- it matches the multimodal scenario playback workflow,
- it allows careful calibration and logging before real-time deployment.

## 2. Repository Evidence Reviewed

The current sound-related evidence in the workspace is:

- Clean package:
  `Project v1/src/audio/`
- Runtime configuration:
  `Project v1/configs/audio.yaml`
- Notebook evidence:
  `/home/mo/dev/python/HDDS2/Notebooks/drone_thesis_audio_training.ipynb`
- Trained artifacts:
  `Project v1/src/audio/drone_sound_model.h5`
  `Project v1/src/audio/models/sound_baseline_mfcc_logreg.joblib`
  `Project v1/src/audio/models/sound_yamnet_hgb.joblib`

Artifact sizes currently on disk are:

- `drone_sound_model.h5`: about `25 MB`
- `sound_baseline_mfcc_logreg.joblib`: about `9.6 KB`
- `sound_yamnet_hgb.joblib`: about `2.1 MB`

These files tell us that the sound branch does not rely on a single model. It
contains at least three preserved outputs:

1. a Keras CNN model,
2. a compact baseline classical model,
3. a YAMNet-based classifier artifact.

## 3. Sound Module Architecture

The cleaned package is structured as follows:

```text
src/audio/
├── __init__.py
├── preprocess.py
├── features.py
├── classifier.py
├── persistence.py
├── schemas.py
├── report.py
├── video_test.py
├── drone_sound_model.h5
└── models/
```

This separation is technically strong. It avoids the common problem where data
loading, feature extraction, model inference, plotting, and logging are all
mixed in a notebook or one long script.

## 4. Training Notebook Findings

The training notebook `drone_thesis_audio_training.ipynb` explicitly states its
goal: build a binary classifier that separates drone and non-drone audio.

The notebook workflow is:

1. clone the dataset repository,
2. scan audio files,
3. assign labels,
4. segment clips into fixed-duration chunks,
5. extract Mel-spectrogram features,
6. train a CNN classifier,
7. evaluate on a test split using classification metrics.

The dataset path used in the notebook is:

```python
AUDIO_DIR = '/content/drone_dataset/Data/Audio'
```

This shows the notebook was designed for a Colab-style or notebook-hosted
training session rather than local packaged training. That is acceptable for
the original experimentation phase, but it also explains why training logic was
later moved into cleaner runtime modules.

## 5. Core Training Algorithm

### 5.1 Feature Representation

The notebook uses log-Mel spectrograms, not raw waveform end-to-end learning.
The code path is:

```python
SAMPLE_RATE = 22050
DURATION = 2  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
```

and later:

```python
mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=N_MELS)
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
```

This is a defendable algorithmic choice.

Why use Mel-spectrograms:

- drone acoustics are largely distinguished by frequency-energy structure over
  time,
- Mel-scaled spectral features are standard and effective for sound
  classification,
- they compress the signal while preserving relevant timbral information,
- they allow a small CNN to learn 2D local patterns in time-frequency space.

In other words, the notebook uses an image-like representation of sound because
it is both efficient and appropriate for harmonic and rotor-noise patterns.

### 5.2 CNN Classifier

The notebook defines the following model:

```python
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
```

This architecture is simple on purpose, and that simplicity is justified.

Why this CNN is reasonable:

- convolutional layers learn local spectral-temporal motifs,
- max pooling reduces sensitivity to small local variation,
- the dense layer provides higher-level class separation,
- dropout reduces overfitting,
- sigmoid output is correct for binary drone versus non-drone classification,
- binary cross-entropy matches the probabilistic binary decision objective.

This is not the most advanced audio architecture possible, but it is a very
good thesis-stage baseline because it is explainable, trainable on limited
resources, and easy for teammates to understand.

### 5.3 Training Procedure

The notebook trains the CNN as follows:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)
```

This training design is defendable for several reasons:

- `train_test_split(..., stratify=y)` preserves class balance,
- a held-out test set is kept separate from the training process,
- a validation split inside training monitors generalization during fitting,
- `epochs=20` is moderate and suitable for an early CNN baseline,
- `batch_size=32` is a standard stable training choice.

The notebook then evaluates with:

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Drone', 'Drone']))
```

This means the intended training outputs were:

- test loss,
- test accuracy,
- predicted test labels,
- a full classification report.

Important repository note:

The saved notebook in the current workspace preserves the code cells, but it
does not preserve the final numeric output text from these evaluation cells.
Therefore, the training procedure is recoverable, but the exact final accuracy,
precision, recall, and F1 values are not retained in the notebook file now
available.

## 6. Transition From Training Notebook To Clean Package

The current `src/audio/` package is more advanced than the notebook-only
baseline because it supports multiple inference backends and explicit data
schemas.

The CLI entrypoint is:

```bash
PYTHONPATH=src python -m audio.video_test /path/to/video.mp4
```

The main execution flow is:

```python
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
```

This is a stronger software design than training directly inside a notebook
because it separates experimentation from runtime inference.

## 7. Algorithm Families Used In The Current Sound Module

The current package supports two main algorithm families and one legacy Keras
artifact used by the integrated app.

### 7.1 Baseline MFCC-Statistical Classifier

The baseline feature extraction in `features.py` is:

```python
mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=40)
delta = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)
```

The module then appends summary statistics over:

- MFCCs,
- MFCC deltas,
- MFCC delta-deltas,
- spectral centroid,
- spectral bandwidth,
- spectral rolloff,
- zero-crossing rate,
- RMS,
- spectral contrast.

Why this baseline is used:

- it is lightweight,
- it is interpretable,
- it works well when dataset size is limited,
- it gives the team a classical benchmark against which deeper methods can be
  compared.

This is a sound engineering decision. A project should not rely only on a
single deep model if a cheaper, simpler baseline can provide reference
performance.

### 7.2 YAMNet Embedding Classifier

The second branch uses a pretrained representation:

```python
waveform = tf.convert_to_tensor(np.asarray(y, dtype=np.float32), dtype=tf.float32)
_, embeddings, _ = yamnet_model(waveform)
return tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)
```

Why YAMNet is defendable:

- it brings transfer learning from large-scale audio pretraining,
- it reduces the burden of learning robust features from scratch,
- it can capture richer audio semantics than handcrafted features alone,
- it is still computationally manageable at inference time.

This branch is especially valuable because drone sounds can be subtle and may
benefit from a stronger representation than classical features alone.

### 7.3 Legacy Keras CNN In The Integrated App

The integrated app in `App/main.py` first tries to load:

```python
model = tf.keras.models.load_model(
    AUDIO_MODEL_PATH,
    custom_objects=custom_objects,
    compile=False
)
```

This indicates the `drone_sound_model.h5` artifact is still part of the live
integration story. It is likely the packaged form of the notebook-era CNN or a
closely related descendant.

The app also uses:

```python
AUDIO_WINDOW_S = 2.0
AUDIO_HOP_S = 0.5
N_MELS = 128
```

These constants line up with the notebook’s 2-second chunking philosophy and
Mel-spectrogram training setup.

## 8. Why Multiple Sound Algorithms Were Worth Keeping

Retaining multiple audio algorithms is not redundancy without reason. It is a
good research design choice.

Why keep the baseline model:

- it is small and easy to run,
- it gives a transparent comparison point,
- it may be more stable in constrained environments.

Why keep the YAMNet branch:

- it leverages pretrained knowledge,
- it may generalize better to unseen recording conditions,
- it supports stronger performance when the baseline saturates.

Why keep the CNN artifact:

- it preserves the original end-to-end thesis training outcome,
- it allows the integrated app to keep working even if the newer package path
  changes,
- it encodes the original Mel-spectrogram deep classifier concept.

From a research standpoint, this is not a weakness. It is a staged evolution of
the audio branch from prototype to cleaner production-style package.

## 9. Score Fusion And Decision Logic

The current package fuses available audio scores using:

```python
def fuse_audio_scores(
    baseline_probability: float | None,
    yamnet_probability: float | None,
    weights: dict[str, float] | None = None,
) -> float:
```

and the final decision is:

```python
label = "drone" if audio_score >= threshold else "no_drone"
```

This weighted-average fusion is defendable because:

- it is simple and transparent,
- it allows teammate-controlled calibration,
- it avoids premature overengineering,
- it supports comparative analysis between model branches.

The package also supports temporal confirmation:

```python
positive_count = sum(1 for candidate in window if candidate.label == "drone")
label = "drone" if positive_count >= confirm_m else "no_drone"
```

This is important for real sound data because audio models often produce
isolated spikes that should not be treated as confirmed detections.

## 10. Current Runtime Configuration

The runtime config in `Project v1/configs/audio.yaml` records:

```yaml
sample_rate_hz: 16000
window_s: 3.0
hop_s: 0.5
threshold: 0.5
```

It also records weighted fusion placeholders and optional M/N confirmation.

This configuration is useful because it turns previously implicit notebook
choices into explicit package-level defaults.

## 11. Training Outputs Preserved In The Repository

The following sound-training outputs are still preserved:

1. `drone_sound_model.h5`
   This is the strongest evidence of a trained deep audio model.
2. `sound_baseline_mfcc_logreg.joblib`
   This is the preserved classical baseline classifier.
3. `sound_yamnet_hgb.joblib`
   This is the preserved classifier trained over YAMNet embeddings.
4. Notebook code that prints:
   - test accuracy
   - classification report
   - predictions on the test set

The following outputs are not preserved in current notebook form:

- numeric test accuracy text,
- exact precision/recall/F1 printout,
- plots or confusion matrices saved as standalone files.

This should be stated honestly in the report. The training story is clearly
recoverable, but not every metric artifact survived repository cleanup.

## 12. Strengths And Limitations

Strengths:

- clean packaged inference code,
- multiple model families,
- explicit data schemas,
- good logging and reporting behavior,
- practical temporal confirmation,
- direct compatibility with multimodal fusion.

Limitations:

- no retained numeric training metrics in the preserved notebook,
- training logic is not yet fully packaged as a reproducible training script,
- integrated app behavior differs from clean package defaults,
- live microphone capture is not implemented.

## 13. Final Defense Of The Sound Module

The sound module is technically justified at this project stage because it
solves a real weakness of radar-only detection. Acoustic signatures can provide
confirmation when the drone is visually ambiguous or when the radar scenario is
synthetic rather than hardware-measured.

The design choices are defensible:

- Mel-spectrograms and a CNN are a valid audio classification strategy,
- MFCC-based features provide a strong classical baseline,
- YAMNet embeddings bring transfer learning into the system,
- weighted score fusion creates a transparent research baseline,
- M/N confirmation reduces alert instability.

The most important point is that the module already moved beyond a notebook
prototype. It now exists as a reusable, teammate-readable software component,
which makes it a legitimate branch in the HDDS multimodal architecture.
