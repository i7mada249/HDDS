# Audio Manifests

The CSV files in this directory document the train, validation, and test split used by the Colab audio-training notebook.

The original rows contain Colab paths such as `/content/hdds_audio/...`. Treat those paths as training-session provenance, not as portable local file paths.

For local use, call `audio.manifest.load_manifest(manifest_path, local_data_root=...)` to remap paths under `/content/hdds_audio/work/wav_data/` to a local dataset root.
