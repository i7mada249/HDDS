import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
import librosa
import numpy as np
from pathlib import Path
import threading
import joblib
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import subprocess

# Constants (from your notebook)
TARGET_SR = 22050
DURATION = 3.0
N_MFCC = 40
SAMPLES = int(TARGET_SR * DURATION)

# Feature extraction (from your notebook)
def extract_features(audio_array, sr):
    # resample if needed
    if sr != TARGET_SR:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
    sr = TARGET_SR

    # ensure mono
    if audio_array.ndim > 1:
        audio_array = librosa.to_mono(audio_array.T)

    # trim / pad to fixed duration
    target_len = int(DURATION * sr)
    if len(audio_array) > target_len:
        audio_array = audio_array[:target_len]
    else:
        audio_array = np.pad(audio_array, (0, target_len - len(audio_array)))

    features = []

    # 1. MFCCs (40 × mean + std = 80 features)
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=N_MFCC)
    features.extend(mfcc.mean(axis=1).tolist())
    features.extend(mfcc.std(axis=1).tolist())

    # 2. MFCC delta (first-order derivative, mean + std = 80 more)
    delta = librosa.feature.delta(mfcc)
    features.extend(delta.mean(axis=1).tolist())
    features.extend(delta.std(axis=1).tolist())

    # 3. Spectral Centroid (mean + std)
    sc = librosa.feature.spectral_centroid(y=audio_array, sr=sr)
    features.extend([sc.mean(), sc.std()])

    # 4. Spectral Bandwidth (mean + std)
    sb = librosa.feature.spectral_bandwidth(y=audio_array, sr=sr)
    features.extend([sb.mean(), sb.std()])

    # 5. Spectral Rolloff (mean + std)
    sr_feat = librosa.feature.spectral_rolloff(y=audio_array, sr=sr)
    features.extend([sr_feat.mean(), sr_feat.std()])

    # 6. Zero Crossing Rate (mean + std)
    zcr = librosa.feature.zero_crossing_rate(audio_array)
    features.extend([zcr.mean(), zcr.std()])

    # 7. Chroma STFT (12 × mean + std = 24)
    chroma = librosa.feature.chroma_stft(y=audio_array, sr=sr)
    features.extend(chroma.mean(axis=1).tolist())
    features.extend(chroma.std(axis=1).tolist())

    # 8. RMS Energy (mean + std)
    rms = librosa.feature.rms(y=audio_array)
    features.extend([rms.mean(), rms.std()])

    # 9. Spectral Contrast (7 bands × mean + std = 14)
    contrast = librosa.feature.spectral_contrast(y=audio_array, sr=sr)
    features.extend(contrast.mean(axis=1).tolist())
    features.extend(contrast.std(axis=1).tolist())

    return np.array(features, dtype=np.float32)

# Load models
audio_model = joblib.load('AudioModule/drone_detector.joblib')
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt', device='cpu')

# GUI Class
class DroneDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Detection App")
        self.video_path = None
        self.cap = None
        self.audio_status = "Not Detected"
        self.vision_status = "Not Detected"
        self.running = False

        # GUI Elements
        tk.Label(root, text="Select Video File:").pack(pady=10)
        tk.Button(root, text="Browse", command=self.select_video).pack()
        tk.Button(root, text="Start Detection", command=self.start_detection).pack(pady=10)
        tk.Button(root, text="Stop", command=self.stop_detection).pack()

        # Split Screen: Left (Audio), Right (Video)
        self.left_frame = tk.Frame(root, width=400, height=400, bg='white')
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.status_label = tk.Label(self.left_frame, text=self.get_status_text(), font=("Arial", 16))
        self.status_label.pack(expand=True)

        self.right_frame = tk.Frame(root, width=400, height=400)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        self.video_label = tk.Label(self.right_frame)
        self.video_label.pack()

    def get_status_text(self):
        return f"Drone sound: {self.audio_status}\nDrone image: {self.vision_status}"

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            messagebox.showinfo("Selected", f"Video: {self.video_path}")

    def start_detection(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first.")
            return
        self.running = True
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            return

        self.audio_status = "Not Detected"
        self.vision_status = "Not Detected"
        self.status_label.config(text=self.get_status_text())

        # Extract audio to temp WAV file
        import tempfile, subprocess
        self._tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self._tmp_wav.close()
        cmd = [
            'ffmpeg', '-y', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', self._tmp_wav.name
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.audio_data, self.audio_sr = sf.read(self._tmp_wav.name, dtype='float32')
        self.audio_frame_idx = 0
        self.audio_samples_per_frame = int(self.audio_sr / self.cap.get(cv2.CAP_PROP_FPS))

        # Extract and process audio in a thread
        threading.Thread(target=self.process_audio, daemon=True).start()
        # Process video frames
        self.process_video()

    def process_audio(self):
        try:
            audio, sr = librosa.load(self._tmp_wav.name, sr=None)  # Use extracted wav
            print(f"Audio loaded: {len(audio)} samples, sr={sr}")
            for i in range(0, len(audio), SAMPLES):
                if not self.running:
                    break
                chunk = audio[i:i+SAMPLES]
                if len(chunk) < SAMPLES:
                    chunk = np.pad(chunk, (0, SAMPLES - len(chunk)))
                features = extract_features(chunk, sr)
                pred = audio_model.predict([features])[0]
                print(f"Prediction: {pred}")
                if pred == 1:  # Assuming 1 is drone class
                    self.audio_status = "Detected"
                else:
                    self.audio_status = "Not Detected"
                self.root.after(0, lambda: self.status_label.config(text=self.get_status_text()))
        except Exception as e:
            print(f"Audio processing error: {e}")

    def process_video(self):
        if not self.running or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        # Play audio chunk for this frame
        if hasattr(self, 'audio_data') and hasattr(self, 'audio_sr'):
            start = self.audio_frame_idx * self.audio_samples_per_frame
            end = start + self.audio_samples_per_frame
            chunk = self.audio_data[start:end]
            if len(chunk) > 0:
                try:
                    sd.play(chunk, self.audio_sr, blocking=True)
                except Exception as e:
                    print(f"Audio playback error: {e}")
            self.audio_frame_idx += 1

        # YOLOv5 detection
        results = yolo_model(frame)
        detected = False
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"drone {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            detected = True
        prev_status = self.vision_status
        self.vision_status = "Detected" if detected else "Not Detected"
        if self.vision_status != prev_status:
            self.root.after(0, lambda: self.status_label.config(text=self.get_status_text()))

        # Convert to Tkinter-compatible image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (400, 300))
        img = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
        self.video_label.config(image=img)
        self.video_label.image = img

        # Schedule next frame
        self.root.after(1, self.process_video)

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
        # Clean up temp wav file
        if hasattr(self, '_tmp_wav'):
            import os
            try:
                os.remove(self._tmp_wav.name)
            except Exception:
                pass

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = DroneDetectionApp(root)
    root.mainloop()