import os
import numpy as np
import librosa
import tensorflow as tf
import sys
import cv2
import sounddevice as sd
import time
from threading import Thread

# Workaround for Keras 3 to Keras 2 compatibility
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

SAMPLE_RATE = 22050
DURATION = 2
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128

def get_prediction(model, signal):
    if len(signal) < 100: return 0.0
    if len(signal) > SAMPLES_PER_TRACK:
        signal = signal[-SAMPLES_PER_TRACK:] # Take trailing window
    else:
        signal = np.pad(signal, (SAMPLES_PER_TRACK - len(signal), 0))
    
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    denom = (log_mel_spec.max() - log_mel_spec.min())
    if denom > 0:
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / denom
    else:
        log_mel_spec = np.zeros_like(log_mel_spec)
    
    X = log_mel_spec[np.newaxis, ..., np.newaxis]
    return model.predict(X, verbose=0)[0][0]

def play_audio(data, sr):
    sd.play(data, sr)
    sd.wait()

def process_video(model_path, video_path):
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    print("Extracting audio...")
    audio_signal, sr = librosa.load(video_path, sr=SAMPLE_RATE)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Start audio playback in background
    audio_thread = Thread(target=play_audio, args=(audio_signal, SAMPLE_RATE), daemon=True)
    audio_thread.start()

    start_time = time.time()
    frame_count = 0
    
    # Pre-warmup prediction
    get_prediction(model, np.zeros(SAMPLES_PER_TRACK))

    print("Running... Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Sync video with audio clock
        elapsed = time.time() - start_time
        expected_frame = int(elapsed * fps)
        
        # If we are behind, skip frames to catch up (prevents lag)
        if frame_count < expected_frame:
            frame_count += 1
            continue

        # Get audio window for the current time
        end_sample = int(elapsed * SAMPLE_RATE)
        start_sample = max(0, end_sample - SAMPLES_PER_TRACK)
        window = audio_signal[start_sample:end_sample]
        
        score = get_prediction(model, window)
        detected = score > 0.5

        # UI
        status = "DRONE SOUND: DETECTED" if detected else "DRONE SOUND: Not detected"
        color = (0, 0, 255) if detected else (0, 255, 0)
        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Score: {score:.2f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        
        cv2.imshow('Drone Sound Detection Test', frame)
        
        frame_count += 1
        # Syncing waitKey with FPS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # If we are too fast, wait a bit
        actual_elapsed = time.time() - start_time
        time_to_wait = (frame_count / fps) - actual_elapsed
        if time_to_wait > 0:
            time.sleep(time_to_wait)

    sd.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_drone_model.py <model_path> <video_path>")
    else:
        process_video(sys.argv[1], sys.argv[2])
