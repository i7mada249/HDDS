import cv2
import numpy as np
import librosa
import threading
import joblib
import subprocess
import tempfile
import os
import torch
import sounddevice as sd
from tkinter import Tk, filedialog

# ================= CONFIG =================
TARGET_SR = 22050
WINDOW_SEC = 0.5
SAMPLES = int(TARGET_SR * WINDOW_SEC)

# ================= FILE PICKER =================
root = Tk()
root.withdraw()

video_path = filedialog.askopenfilename(title="Select Video")

if not video_path or not os.path.exists(video_path):
    raise RuntimeError("❌ No valid video selected")

# ================= LOAD MODELS =================
print("Loading models...")

audio_model = joblib.load('AudioModule/drone_detector.joblib')

yolo_model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='yolov5/best.pt',
    device='cpu'
)

# ================= AUDIO EXTRACTION =================
print("Extracting audio...")

tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
tmp.close()

cmd = [
    'ffmpeg', '-y', '-i', video_path,
    '-vn', '-acodec', 'pcm_s16le',
    '-ar', str(TARGET_SR), '-ac', '1',
    tmp.name
]

result = subprocess.run(cmd)

if result.returncode != 0 or not os.path.exists(tmp.name) or os.path.getsize(tmp.name) < 1000:
    raise RuntimeError("❌ Audio extraction failed")

audio_data, sr = librosa.load(tmp.name, sr=TARGET_SR)

print("Audio loaded:", len(audio_data), "samples")

# ================= PLAY AUDIO =================
def play_audio():
    try:
        sd.play(audio_data, sr, blocking=False)
    except Exception as e:
        print("Audio playback error:", e)

threading.Thread(target=play_audio, daemon=True).start()

# ================= FEATURE (مطابقة للموديل) =================
def extract_features(audio_array):
    features = []

    mfcc = librosa.feature.mfcc(y=audio_array, sr=TARGET_SR, n_mfcc=40)
    features.extend(mfcc.mean(axis=1))
    features.extend(mfcc.std(axis=1))

    delta = librosa.feature.delta(mfcc)
    features.extend(delta.mean(axis=1))
    features.extend(delta.std(axis=1))

    sc = librosa.feature.spectral_centroid(y=audio_array, sr=TARGET_SR)
    sb = librosa.feature.spectral_bandwidth(y=audio_array, sr=TARGET_SR)
    sr_feat = librosa.feature.spectral_rolloff(y=audio_array, sr=TARGET_SR)

    features.extend([sc.mean(), sc.std(), sb.mean(), sb.std(), sr_feat.mean(), sr_feat.std()])

    zcr = librosa.feature.zero_crossing_rate(audio_array)
    features.extend([zcr.mean(), zcr.std()])

    rms = librosa.feature.rms(y=audio_array)
    features.extend([rms.mean(), rms.std()])

    contrast = librosa.feature.spectral_contrast(y=audio_array, sr=TARGET_SR)
    features.extend(contrast.mean(axis=1))
    features.extend(contrast.std(axis=1))

    return np.array(features, dtype=np.float32)

# ================= AUDIO THREAD =================
class AudioProcessor(threading.Thread):
    def __init__(self, audio):
        super().__init__()
        self.audio = audio
        self.idx = 0
        self.status = "Not Detected"
        self.running = True

    def run(self):
        while self.running and self.idx < len(self.audio):
            chunk = self.audio[self.idx:self.idx + SAMPLES]

            if len(chunk) < SAMPLES:
                chunk = np.pad(chunk, (0, SAMPLES - len(chunk)))

            try:
                feats = extract_features(chunk)
                pred = audio_model.predict([feats])[0]

                # تقليل التذبذب
                if pred == 1:
                    self.status = "Detected"

            except Exception as e:
                print("Audio error:", e)
                self.status = "Not Detected"

            self.idx += SAMPLES

# ================= START =================
audio_thread = AudioProcessor(audio_data)
audio_thread.start()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video")

frame_count = 0
vision_status = "Not Detected"

print("Starting detection...")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 500))

    # ===== YOLO (تقليل الحمل)
    if frame_count % 3 == 0:
        try:
            results = yolo_model(frame)
            detected = False

            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > 0.5:
                    detected = True
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            vision_status = "Detected" if detected else "Not Detected"

        except Exception as e:
            print("Vision error:", e)
            vision_status = "Not Detected"

    # ===== STATUS TEXT
    text = f"AUDIO: {audio_thread.status} | VISION: {vision_status}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Drone Detection System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
audio_thread.running = False

try:
    os.remove(tmp.name)
except:
    pass

print("Finished.")