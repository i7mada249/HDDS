from __future__ import annotations

import os
import threading
import time
import tkinter as tk
import warnings
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

Path("/tmp/hdds_matplotlib").mkdir(parents=True, exist_ok=True)
Path("/tmp/hdds_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hdds_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/hdds_cache")

import cv2
import joblib
import librosa
import numpy as np
import torch
import tensorflow as tf

# Workaround for Keras 3 to Keras 2 compatibility (quantization_config error)
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

class CompatibleLayer:
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

class CompatibleDense(CompatibleLayer, Dense): pass
class CompatibleConv2D(CompatibleLayer, Conv2D): pass
class CompatibleMaxPooling2D(CompatibleLayer, MaxPooling2D): pass
class CompatibleFlatten(CompatibleLayer, Flatten): pass
class CompatibleDropout(CompatibleLayer, Dropout): pass

custom_objects = {
    'Dense': CompatibleDense, 'Conv2D': CompatibleConv2D,
    'MaxPooling2D': CompatibleMaxPooling2D, 'Flatten': CompatibleFlatten, 'Dropout': CompatibleDropout
}

try:
    from .scenario_loader import (
        PROJECT_ROOT,
        REPO_ROOT,
        RadarScenarioData,
        ScenarioBundle,
        bundle_from_member,
        discover_scenarios,
        load_radar_scenario,
    )
except ImportError:
    from scenario_loader import (
        PROJECT_ROOT,
        REPO_ROOT,
        RadarScenarioData,
        ScenarioBundle,
        bundle_from_member,
        discover_scenarios,
        load_radar_scenario,
    )

from audio.classifier import LoadedAudioModels, load_audio_models
from audio.classifier import predict_window as predict_project_audio_window
from audio.preprocess import TARGET_SR as PROJECT_AUDIO_TARGET_SR
from audio.preprocess import load_audio_file, slice_audio
from radar_sim.channel import simulate_surveillance_matrix
from radar_sim.constants import load_app_config
from radar_sim.detection import ca_cfar_2d
from radar_sim.processing import process_reference_and_surveillance
from radar_sim.waveform import generate_reference_matrix


LEGACY_AUDIO_TARGET_SR = 22050
AUDIO_WINDOW_S = 2.0  # Matches the new model's training duration
AUDIO_HOP_S = 0.5
N_MELS = 128
VISION_CONFIDENCE = 0.5
VISION_EVERY_N_FRAMES = 3

YOLO_DIR = REPO_ROOT / "yolov5"
YOLO_WEIGHTS = YOLO_DIR / "best.pt"
# Path to the new sound model
AUDIO_MODEL_PATH = REPO_ROOT / "Project v1" / "src" / "audio" / "drone_sound_model.h5"

BG = "#0f172a"
SURFACE = "#111827"
SURFACE_2 = "#182235"
BORDER = "#293548"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
GREEN = "#22c55e"
AMBER = "#f59e0b"
RED = "#ef4444"
BLUE = "#38bdf8"


def default_config_path() -> Path:
    return PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class AudioPrediction:
    start_s: float
    end_s: float
    score: float
    label: str


@dataclass(frozen=True)
class RadarRuntime:
    data: RadarScenarioData
    detected: bool
    detection_count: int
    power_db: np.ndarray
    detection_mask: np.ndarray


@dataclass(frozen=True)
class PlaybackAudio:
    samples: np.ndarray
    sr: int


@dataclass(frozen=True)
class LoadedAudioBackend:
    name: str
    legacy_model: object | None = None
    project_models: LoadedAudioModels | None = None
    load_note: str | None = None


@dataclass(frozen=True)
class DetectionSnapshot:
    time_s: float
    vision_detected: bool
    audio_detected: bool
    radar_detected: bool
    fusion_level: int
    audio_score: float


def extract_features(audio_array: np.ndarray, sr: int) -> np.ndarray:
    if sr != LEGACY_AUDIO_TARGET_SR:
        audio_array = librosa.resample(
            audio_array,
            orig_sr=sr,
            target_sr=LEGACY_AUDIO_TARGET_SR,
        )
    sr = LEGACY_AUDIO_TARGET_SR

    if audio_array.ndim > 1:
        audio_array = librosa.to_mono(audio_array.T)

    target_len = int(AUDIO_WINDOW_S * sr)
    if len(audio_array) > target_len:
        audio_array = audio_array[:target_len]
    else:
        audio_array = np.pad(audio_array, (0, target_len - len(audio_array)))

    # Extract Mel Spectrogram (matches training)
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=N_MELS)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    denom = (log_mel_spec.max() - log_mel_spec.min())
    if denom > 0:
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / denom
    else:
        log_mel_spec = np.zeros_like(log_mel_spec)
        
    return log_mel_spec[..., np.newaxis]


def load_audio_backend() -> LoadedAudioBackend:
    try:
        # Load the new Keras model
        model = tf.keras.models.load_model(
            AUDIO_MODEL_PATH, 
            custom_objects=custom_objects, 
            compile=False
        )
        return LoadedAudioBackend(
            name="keras_cnn",
            legacy_model=model,
        )
    except Exception as exc:
        print(f"New Keras model failed to load from {AUDIO_MODEL_PATH}: {exc}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                project_models = load_audio_models(use_baseline=True, use_yamnet=False)
            note = "Using Project v1 baseline audio model instead."
            return LoadedAudioBackend(
                name="project_baseline",
                project_models=project_models,
                load_note=note,
            )
        except Exception:
             raise RuntimeError(f"No audio backend could be loaded. Last error: {exc}")


def predict_audio_file(backend: LoadedAudioBackend, audio_path: Path) -> list[AudioPrediction]:
    if backend.name == "keras_cnn":
        return _predict_audio_file_with_keras_model(backend.legacy_model, audio_path)
    if backend.project_models is not None:
        return _predict_audio_file_with_project_model(backend.project_models, audio_path)
    raise RuntimeError("No audio backend is available.")


def _predict_audio_file_with_keras_model(
    model: tf.keras.Model,
    audio_path: Path,
) -> list[AudioPrediction]:
    audio, sr = librosa.load(audio_path, sr=LEGACY_AUDIO_TARGET_SR, mono=True)
    window_samples = int(AUDIO_WINDOW_S * sr)
    hop_samples = int(AUDIO_HOP_S * sr)
    if len(audio) == 0:
        return []

    predictions = []
    starts = list(range(0, max(1, len(audio) - window_samples + 1), hop_samples))
    
    for start in starts:
        chunk = audio[start : start + window_samples]
        features = extract_features(chunk, sr)
        # Add batch dimension
        X = features[np.newaxis, ...]
        score = float(model.predict(X, verbose=0)[0][0])
        
        start_s = start / sr
        end_s = min(len(audio) / sr, start_s + AUDIO_WINDOW_S)
        predictions.append(
            AudioPrediction(
                start_s=start_s,
                end_s=end_s,
                score=score,
                label="Detected" if score >= 0.5 else "Clear",
            )
        )
    return predictions


def _predict_audio_file_with_legacy_model(
    model: object,
    audio_path: Path,
) -> list[AudioPrediction]:
    # Legacy handler redirected to Keras
    return _predict_audio_file_with_keras_model(model, audio_path)


def _predict_audio_file_with_project_model(
    models: LoadedAudioModels,
    audio_path: Path,
) -> list[AudioPrediction]:
    audio, sr = load_audio_file(audio_path, target_sr=PROJECT_AUDIO_TARGET_SR)
    windows = slice_audio(
        y=audio,
        sr=sr,
        window_s=AUDIO_WINDOW_S,
        hop_s=AUDIO_HOP_S,
    )
    predictions = []
    for window in windows:
        prediction = predict_project_audio_window(
            window=window,
            sr=sr,
            loaded_models=models,
            threshold=0.5,
        )
        predictions.append(
            AudioPrediction(
                start_s=prediction.start_s,
                end_s=prediction.end_s,
                score=prediction.audio_score,
                label="Detected" if prediction.label == "drone" else "Clear",
            )
        )
    return predictions


def load_playback_audio(audio_path: Path) -> PlaybackAudio:
    try:
        import soundfile as sf

        samples, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception:
        samples, sr = librosa.load(audio_path, sr=None, mono=False)
        if np.asarray(samples).ndim == 2:
            samples = np.asarray(samples).T

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 2:
        raise ValueError(f"Unsupported audio shape for playback: {audio.shape}")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return PlaybackAudio(samples=np.ascontiguousarray(audio), sr=int(sr))


class HybridDetectionDashboard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hybrid Detection Dashboard")
        self.root.geometry("1220x760")
        self.root.minsize(1060, 680)
        self.root.configure(bg=BG)

        self.bundle: ScenarioBundle | None = None
        self.app_config = load_app_config(default_config_path())
        self.audio_backend: LoadedAudioBackend | None = None
        self.yolo_model: object | None = None
        self.audio_predictions: list[AudioPrediction] = []
        self.playback_audio: PlaybackAudio | None = None
        self.radar_runtime: RadarRuntime | None = None
        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.frame_index = 0
        self.video_fps = 25.0
        self.video_frame_count = 0
        self.video_duration_s = 0.0
        self.playback_started_at = 0.0
        self.last_vision_detected = False
        self.last_vision_boxes: list[tuple[int, int, int, int, float]] = []
        self.plot_history: list[DetectionSnapshot] = []

        self._build_style()
        self._build_layout()
        self._set_idle_state()

    def _build_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, foreground=TEXT)
        style.configure(
            "Treeview.Heading",
            background=SURFACE_2,
            foreground=TEXT,
            relief="flat",
        )
        style.map("Treeview", background=[("selected", "#1e3a5f")])

    def _build_layout(self) -> None:
        self.top = tk.Frame(self.root, bg=BG)
        self.top.pack(fill=tk.X, padx=18, pady=(16, 10))

        self.title_label = tk.Label(
            self.top,
            text="No scenario loaded",
            bg=BG,
            fg=TEXT,
            font=("Arial", 20, "bold"),
            anchor="w",
        )
        self.title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.open_file_button = self._button(self.top, "Open Scenario", self.open_scenario)
        self.open_file_button.pack(side=tk.LEFT, padx=(8, 0))
        self.open_folder_button = self._button(self.top, "Open Folder", self.open_folder)
        self.open_folder_button.pack(side=tk.LEFT, padx=(8, 0))
        self.start_button = self._button(self.top, "Start", self.start)
        self.start_button.pack(side=tk.LEFT, padx=(8, 0))
        self.stop_button = self._button(self.top, "Stop", self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        self.status_row = tk.Frame(self.root, bg=BG)
        self.status_row.pack(fill=tk.X, padx=18, pady=(0, 12))
        self.fusion_card = self._status_card(self.status_row, "Fusion", "Idle", MUTED)
        self.vision_card = self._status_card(self.status_row, "Vision", "Idle", MUTED)
        self.audio_card = self._status_card(self.status_row, "Audio", "Idle", MUTED)
        self.radar_card = self._status_card(self.status_row, "Radar", "Idle", MUTED)

        self.body = tk.Frame(self.root, bg=BG)
        self.body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))

        self.video_panel = self._panel(self.body)
        self.video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        self.video_label = tk.Label(self.video_panel, bg="#020617")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.radar_heatmap_panel = self._panel(self.body, width=340)
        self.radar_heatmap_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        self.radar_heatmap_panel.pack_propagate(False)
        tk.Label(
            self.radar_heatmap_panel,
            text="Radar Heatmap",
            bg=SURFACE,
            fg=TEXT,
            font=("Arial", 12, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(12, 8))
        self.radar_heatmap_label = tk.Label(self.radar_heatmap_panel, bg="#020617")
        self.radar_heatmap_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        self.radar_heatmap_caption = tk.Label(
            self.radar_heatmap_panel,
            text="Range-Doppler CFAR",
            bg=SURFACE,
            fg=MUTED,
            font=("Arial", 9),
            anchor="w",
        )
        self.radar_heatmap_caption.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.side_panel = self._panel(self.body, width=390)
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_panel.pack_propagate(False)

        self.metrics = tk.Frame(self.side_panel, bg=SURFACE)
        self.metrics.pack(fill=tk.X, padx=12, pady=(12, 8))
        self.time_label = self._metric(self.metrics, "Time", "0.0 s")
        self.audio_score_label = self._metric(self.metrics, "Audio score", "-")
        self.vision_count_label = self._metric(self.metrics, "Vision boxes", "0")
        self.radar_count_label = self._metric(self.metrics, "Radar detections", "0")

        plot_frame = tk.Frame(self.side_panel, bg=SURFACE)
        plot_frame.pack(fill=tk.X, padx=12, pady=(6, 10))
        tk.Label(
            plot_frame,
            text="Detection Plot",
            bg=SURFACE,
            fg=TEXT,
            font=("Arial", 12, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 8))
        self.plot_canvas = tk.Canvas(
            plot_frame,
            height=168,
            bg="#07111f",
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.plot_canvas.pack(fill=tk.X)
        self.plot_canvas.bind("<Configure>", lambda _event: self._draw_detection_plot())

        table_frame = tk.Frame(self.side_panel, bg=SURFACE)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))
        tk.Label(
            table_frame,
            text="Radar Objects",
            bg=SURFACE,
            fg=TEXT,
            font=("Arial", 12, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 8))
        columns = ("object", "range", "speed", "doppler", "db")
        self.radar_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        headings = {
            "object": "Object",
            "range": "m",
            "speed": "m/s",
            "doppler": "Hz",
            "db": "dB",
        }
        widths = {"object": 110, "range": 70, "speed": 70, "doppler": 70, "db": 55}
        for column in columns:
            self.radar_table.heading(column, text=headings[column])
            self.radar_table.column(column, width=widths[column], stretch=False, anchor=tk.CENTER)
        self.radar_table.pack(fill=tk.BOTH, expand=True)

    def _button(self, parent: tk.Widget, text: str, command: object) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=SURFACE_2,
            fg=TEXT,
            activebackground="#243247",
            activeforeground=TEXT,
            relief=tk.FLAT,
            padx=14,
            pady=9,
            font=("Arial", 10, "bold"),
            cursor="hand2",
        )

    def _panel(self, parent: tk.Widget, width: int | None = None) -> tk.Frame:
        frame = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        if width is not None:
            frame.configure(width=width)
        return frame

    def _status_card(self, parent: tk.Widget, title: str, value: str, color: str) -> tk.Label:
        frame = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(frame, text=title, bg=SURFACE, fg=MUTED, font=("Arial", 10), anchor="w").pack(
            fill=tk.X, padx=14, pady=(10, 0)
        )
        label = tk.Label(
            frame,
            text=value,
            bg=SURFACE,
            fg=color,
            font=("Arial", 18, "bold"),
            anchor="w",
        )
        label.pack(fill=tk.X, padx=14, pady=(0, 10))
        return label

    def _metric(self, parent: tk.Widget, title: str, value: str) -> tk.Label:
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill=tk.X, pady=4)
        tk.Label(row, text=title, bg=SURFACE, fg=MUTED, font=("Arial", 10), anchor="w").pack(
            side=tk.LEFT
        )
        label = tk.Label(row, text=value, bg=SURFACE, fg=TEXT, font=("Arial", 11, "bold"))
        label.pack(side=tk.RIGHT)
        return label

    def open_scenario(self) -> None:
        selected = filedialog.askopenfilename(
            title="Open scenario member",
            filetypes=[
                ("Scenario files", "*.mp4 *.avi *.mov *.mkv *.wav *.flac *.mp3 *.ogg *.json"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        try:
            self.load_bundle(bundle_from_member(Path(selected)))
        except Exception as exc:
            messagebox.showerror("Scenario", str(exc))

    def open_folder(self) -> None:
        selected = filedialog.askdirectory(title="Open scenario folder")
        if not selected:
            return
        bundles = discover_scenarios(Path(selected))
        if not bundles:
            messagebox.showerror("Scenario", "No complete mp4/wav/json scenario set found.")
            return
        if len(bundles) == 1:
            self.load_bundle(bundles[0])
            return
        self._choose_bundle(bundles)

    def _choose_bundle(self, bundles: list[ScenarioBundle]) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Scenarios")
        dialog.configure(bg=BG)
        dialog.geometry("360x320")
        listbox = tk.Listbox(
            dialog,
            bg=SURFACE,
            fg=TEXT,
            selectbackground="#1e3a5f",
            relief=tk.FLAT,
            font=("Arial", 12),
        )
        listbox.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        for bundle in bundles:
            listbox.insert(tk.END, bundle.name)

        def select_current() -> None:
            indexes = listbox.curselection()
            if indexes:
                self.load_bundle(bundles[indexes[0]])
                dialog.destroy()

        listbox.bind("<Double-Button-1>", lambda _event: select_current())
        self._button(dialog, "Load", select_current).pack(fill=tk.X, padx=14, pady=(0, 14))

    def load_bundle(self, bundle: ScenarioBundle) -> None:
        self.stop()
        self.bundle = bundle
        self.audio_predictions = []
        self.playback_audio = None
        self.radar_runtime = None
        self.plot_history = []
        self.title_label.configure(text=bundle.name)
        self._set_card(self.fusion_card, "Ready", BLUE)
        self._set_card(self.vision_card, "Ready", BLUE)
        self._set_card(self.audio_card, "Ready", BLUE)
        self._set_card(self.radar_card, "Ready", BLUE)
        try:
            self._load_radar_runtime()
            self._render_radar_table(0.0)
            self._render_radar_heatmap()
            self._draw_detection_plot()
        except Exception as exc:
            messagebox.showerror("Radar", str(exc))

    def start(self) -> None:
        if self.bundle is None:
            messagebox.showerror("Scenario", "Load a scenario first.")
            return
        self.stop()
        self.running = True
        self._set_card(self.fusion_card, "Loading", AMBER)
        self._set_card(self.vision_card, "Loading", AMBER)
        self._set_card(self.audio_card, "Loading", AMBER)
        self._set_card(self.radar_card, "Loading", AMBER)
        threading.Thread(target=self._prepare_runtime, daemon=True).start()

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._stop_audio_playback()
        self.playback_started_at = 0.0

    def _prepare_runtime(self) -> None:
        try:
            if self.yolo_model is None:
                if not YOLO_WEIGHTS.exists():
                    raise FileNotFoundError(f"YOLOv5 weights not found: {YOLO_WEIGHTS}")
                self.yolo_model = torch.hub.load(
                    str(YOLO_DIR),
                    "custom",
                    path=str(YOLO_WEIGHTS),
                    source="local",
                    device="cpu",
                )
            if self.audio_backend is None:
                self.audio_backend = load_audio_backend()

            assert self.bundle is not None
            self.audio_predictions = predict_audio_file(
                self.audio_backend,
                self.bundle.audio_path,
            )
            self.playback_audio = load_playback_audio(self.bundle.audio_path)
            radar_runtime = self._build_radar_runtime()
            self.root.after(0, lambda: self._finish_prepare(radar_runtime))
        except Exception as exc:
            self.running = False
            self.root.after(0, lambda: messagebox.showerror("Runtime", str(exc)))
            self.root.after(0, self._set_idle_state)

    def _finish_prepare(self, radar_runtime: RadarRuntime) -> None:
        self.radar_runtime = radar_runtime
        self.radar_count_label.configure(text=str(radar_runtime.detection_count))
        self._render_radar_heatmap()
        self._start_video()

    def _build_radar_runtime(self) -> RadarRuntime:
        if self.bundle is None:
            raise RuntimeError("No scenario is loaded.")
        radar_data = load_radar_scenario(self.bundle.radar_path, self.app_config.radar)
        rng = np.random.default_rng(self.app_config.seed)
        reference = generate_reference_matrix(config=self.app_config.radar, rng=rng)
        surveillance = simulate_surveillance_matrix(
            reference=reference,
            config=self.app_config.radar,
            scenario=radar_data.scenario,
            rng=rng,
        )
        processing = process_reference_and_surveillance(
            reference=reference,
            surveillance=surveillance,
            config=self.app_config.radar,
        )
        detections = ca_cfar_2d(
            range_doppler_map=processing.range_doppler_map,
            range_axis_m=processing.range_axis_m,
            doppler_axis_hz=processing.doppler_axis_hz,
            velocity_axis_mps=processing.velocity_axis_mps,
            config=self.app_config.cfar,
        )
        detection_count = len(detections.detections)
        power_db = 10.0 * np.log10(detections.power_map + 1.0e-18)
        return RadarRuntime(
            data=radar_data,
            detected=detection_count > 0,
            detection_count=detection_count,
            power_db=power_db,
            detection_mask=detections.detection_mask,
        )

    def _load_radar_runtime(self) -> None:
        self.radar_runtime = self._build_radar_runtime()
        self.radar_count_label.configure(text=str(self.radar_runtime.detection_count))

    def _start_video(self) -> None:
        if not self.running or self.bundle is None:
            return
        self.cap = cv2.VideoCapture(str(self.bundle.video_path))
        if not self.cap.isOpened():
            self.running = False
            messagebox.showerror("Vision", "Cannot open video file.")
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps and fps > 1 else 25.0
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_frame_count = max(0, frame_count)
        self.video_duration_s = (
            self.video_frame_count / self.video_fps if self.video_frame_count else 0.0
        )
        self.frame_index = 0
        self.plot_history = []
        self.playback_started_at = time.monotonic()
        self._start_audio_playback()
        self._set_card(
            self.radar_card,
            "Detected" if self._radar_detected() else "Clear",
            RED if self._radar_detected() else GREEN,
        )
        self._process_frame()

    def _process_frame(self) -> None:
        if not self.running or self.cap is None:
            return

        elapsed_s = max(0.0, time.monotonic() - self.playback_started_at)
        target_frame = int(elapsed_s * self.video_fps)
        if self.video_frame_count and target_frame >= self.video_frame_count:
            self.stop()
            self._set_card(self.fusion_card, "Finished", MUTED)
            return
        skipped_frames = False
        if target_frame > self.frame_index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.frame_index = target_frame
            skipped_frames = True

        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            self._set_card(self.fusion_card, "Finished", MUTED)
            return

        current_s = self.frame_index / self.video_fps
        frame = cv2.resize(frame, (800, 500))
        if skipped_frames or self.frame_index % VISION_EVERY_N_FRAMES == 0:
            self._run_vision(frame)
        self._draw_vision_boxes(frame)

        audio = self._audio_at(current_s)
        audio_detected = audio is not None and audio.label == "Detected"
        radar_detected = self._radar_detected()
        self._update_dashboard(current_s, audio, audio_detected, radar_detected)
        self._show_frame(frame)

        self.frame_index += 1
        next_due_s = self.frame_index / self.video_fps
        elapsed_after_s = max(0.0, time.monotonic() - self.playback_started_at)
        delay_ms = max(1, int((next_due_s - elapsed_after_s) * 1000))
        self.root.after(delay_ms, self._process_frame)

    def _run_vision(self, frame: np.ndarray) -> None:
        self.last_vision_boxes = []
        self.last_vision_detected = False
        if self.yolo_model is None:
            return
        results = self.yolo_model(frame)
        for *xyxy, conf, _cls in results.xyxy[0]:
            confidence = float(conf)
            if confidence < VISION_CONFIDENCE:
                continue
            x1, y1, x2, y2 = [int(value) for value in xyxy]
            self.last_vision_boxes.append((x1, y1, x2, y2, confidence))
        self.last_vision_detected = bool(self.last_vision_boxes)

    def _draw_vision_boxes(self, frame: np.ndarray) -> None:
        for x1, y1, x2, y2, confidence in self.last_vision_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)
            cv2.putText(
                frame,
                f"drone {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (34, 197, 94),
                2,
            )

    def _show_frame(self, frame: np.ndarray) -> None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            return
        image = tk.PhotoImage(data=encoded.tobytes())
        self.video_label.configure(image=image)
        self.video_label.image = image

    def _render_radar_heatmap(self) -> None:
        if self.radar_runtime is None:
            self.radar_heatmap_label.configure(image="", text="")
            return

        power = self.radar_runtime.power_db
        finite = power[np.isfinite(power)]
        if finite.size == 0:
            normalized = np.zeros(power.shape, dtype=np.uint8)
        else:
            low, high = np.percentile(finite, [5, 99])
            if high <= low:
                high = low + 1.0
            normalized = np.clip((power - low) / (high - low), 0.0, 1.0)
            normalized = (normalized * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        mask = self.radar_runtime.detection_mask
        if mask.shape == normalized.shape and np.any(mask):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(heatmap, contours, -1, (255, 255, 255), 1)

        heatmap = cv2.flip(heatmap, 0)
        heatmap = cv2.resize(heatmap, (316, 316), interpolation=cv2.INTER_LINEAR)
        ok, encoded = cv2.imencode(".png", heatmap)
        if not ok:
            return
        image = tk.PhotoImage(data=encoded.tobytes())
        self.radar_heatmap_label.configure(image=image)
        self.radar_heatmap_label.image = image
        self.radar_heatmap_caption.configure(
            text=f"CFAR detections: {self.radar_runtime.detection_count}"
        )

    def _start_audio_playback(self) -> None:
        if self.playback_audio is None:
            return
        try:
            import sounddevice as sd

            sd.stop()
            sd.play(self.playback_audio.samples, self.playback_audio.sr, blocking=False)
        except Exception as exc:
            print(f"Audio playback error: {exc}")

    def _stop_audio_playback(self) -> None:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def _audio_at(self, current_s: float) -> AudioPrediction | None:
        candidates = [
            item for item in self.audio_predictions if item.start_s <= current_s <= item.end_s
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.score)

    def _radar_detected(self) -> bool:
        return self.radar_runtime is not None and self.radar_runtime.detected

    def _update_dashboard(
        self,
        current_s: float,
        audio: AudioPrediction | None,
        audio_detected: bool,
        radar_detected: bool,
    ) -> None:
        positives = sum([self.last_vision_detected, audio_detected, radar_detected])
        if positives >= 2:
            fusion_text, fusion_color = "Confirmed", RED
        elif positives == 1:
            fusion_text, fusion_color = "Watch", AMBER
        else:
            fusion_text, fusion_color = "Clear", GREEN

        self._set_card(self.fusion_card, fusion_text, fusion_color)
        self._set_card(
            self.vision_card,
            "Detected" if self.last_vision_detected else "Clear",
            RED if self.last_vision_detected else GREEN,
        )
        self._set_card(
            self.audio_card,
            "Detected" if audio_detected else "Clear",
            RED if audio_detected else GREEN,
        )
        self._set_card(
            self.radar_card,
            "Detected" if radar_detected else "Clear",
            RED if radar_detected else GREEN,
        )
        self.time_label.configure(text=f"{current_s:.1f} s")
        self.audio_score_label.configure(text="-" if audio is None else f"{audio.score:.2f}")
        self.vision_count_label.configure(text=str(len(self.last_vision_boxes)))
        self._append_detection_snapshot(
            current_s=current_s,
            vision_detected=self.last_vision_detected,
            audio_detected=audio_detected,
            radar_detected=radar_detected,
            fusion_level=positives,
            audio_score=0.0 if audio is None else audio.score,
        )
        self._draw_detection_plot()
        self._render_radar_table(current_s)

    def _append_detection_snapshot(
        self,
        current_s: float,
        vision_detected: bool,
        audio_detected: bool,
        radar_detected: bool,
        fusion_level: int,
        audio_score: float,
    ) -> None:
        if self.plot_history and current_s <= self.plot_history[-1].time_s:
            return
        self.plot_history.append(
            DetectionSnapshot(
                time_s=current_s,
                vision_detected=vision_detected,
                audio_detected=audio_detected,
                radar_detected=radar_detected,
                fusion_level=fusion_level,
                audio_score=audio_score,
            )
        )
        if len(self.plot_history) > 1800:
            self.plot_history = self.plot_history[-1800:]

    def _draw_detection_plot(self) -> None:
        canvas = self.plot_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        left = 62
        right = width - 12
        top = 16
        bottom = height - 24
        plot_width = max(1, right - left)
        lanes = [
            ("Vision", 0, GREEN),
            ("Audio", 1, BLUE),
            ("Radar", 2, AMBER),
            ("Fusion", 3, RED),
        ]
        lane_gap = (bottom - top) / len(lanes)
        duration = max(
            self.video_duration_s,
            self.plot_history[-1].time_s if self.plot_history else 1.0,
            1.0,
        )

        for idx in range(6):
            x = left + plot_width * idx / 5
            canvas.create_line(x, top, x, bottom, fill="#132033")
        canvas.create_line(left, bottom, right, bottom, fill=BORDER)
        canvas.create_text(left, height - 10, text="0s", fill=MUTED, anchor="w", font=("Arial", 8))
        canvas.create_text(
            right,
            height - 10,
            text=f"{duration:.0f}s",
            fill=MUTED,
            anchor="e",
            font=("Arial", 8),
        )

        for label, lane_index, color in lanes:
            y = top + lane_gap * lane_index + lane_gap * 0.5
            canvas.create_text(10, y, text=label, fill=MUTED, anchor="w", font=("Arial", 9))
            canvas.create_line(left, y, right, y, fill="#1c2a3e")
            values = self._plot_lane_values(label)
            previous: DetectionSnapshot | None = None
            for snapshot in self.plot_history:
                detected = values(snapshot)
                if previous is not None:
                    x1 = left + plot_width * (previous.time_s / duration)
                    x2 = left + plot_width * (snapshot.time_s / duration)
                    if detected:
                        canvas.create_line(x1, y, x2, y, fill=color, width=5)
                previous = snapshot
            latest = self.plot_history[-1] if self.plot_history else None
            if latest is not None and values(latest):
                x = left + plot_width * (latest.time_s / duration)
                canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="")

        if self.plot_history:
            current_x = left + plot_width * (self.plot_history[-1].time_s / duration)
            canvas.create_line(current_x, top - 2, current_x, bottom + 2, fill=TEXT)

    def _plot_lane_values(self, label: str):
        if label == "Vision":
            return lambda item: item.vision_detected
        if label == "Audio":
            return lambda item: item.audio_detected
        if label == "Radar":
            return lambda item: item.radar_detected
        return lambda item: item.fusion_level >= 2

    def _render_radar_table(self, current_s: float) -> None:
        self.radar_table.delete(*self.radar_table.get_children())
        if self.radar_runtime is None:
            return
        for radar_object in self.radar_runtime.data.objects:
            current_range = max(0.0, radar_object.distance_m - radar_object.speed_mps * current_s)
            self.radar_table.insert(
                "",
                tk.END,
                values=(
                    radar_object.name,
                    f"{current_range:.0f}",
                    f"{radar_object.speed_mps:.1f}",
                    f"{radar_object.doppler_hz:.0f}",
                    f"{radar_object.amplitude_db:.1f}",
                ),
            )

    def _set_idle_state(self) -> None:
        self._set_card(self.fusion_card, "Idle", MUTED)
        self._set_card(self.vision_card, "Idle", MUTED)
        self._set_card(self.audio_card, "Idle", MUTED)
        self._set_card(self.radar_card, "Idle", MUTED)
        self.time_label.configure(text="0.0 s")
        self.audio_score_label.configure(text="-")
        self.vision_count_label.configure(text="0")
        self.radar_count_label.configure(text="0")
        self.plot_history = []
        self._render_radar_heatmap()
        self._draw_detection_plot()

    def _set_card(self, label: tk.Label, text: str, color: str) -> None:
        label.configure(text=text, fg=color)


def main() -> None:
    root = tk.Tk()
    app = HybridDetectionDashboard(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
