import pyaudio
import wave
import os
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import keyboard  # Ensure you have installed the keyboard library
import serial
import time

def record_audio(duration=5):
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Press and hold the space key to start recording...")
    frames = []
    is_recording = False

    while True:
        if keyboard.is_pressed("space"):  # Start recording when the space key is pressed
            if not is_recording:
                print("Start Recording...")
                is_recording = True
        else:
            if is_recording:
                print("Recording Stopped.")
                break

        if is_recording:
            data = stream.read(CHUNK)
            frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a file
    filename = "temp_recording.wav"
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    return filename

def extract_features(file_path):
    # Extract MFCC features from the audio file
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def recognize_voice(model, label_encoder):
    # Record a new voice sample
    file_path = record_audio()

    # Extract features from the new voice sample
    features = extract_features(file_path).reshape(1, -1)

    # Predict using the SVM model
    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]
    print("Predicted label:", label)

    # Remove the temporary recording file
    os.remove(file_path)

    labels_result = [0, 1]

    # Print the result
    if label in labels_result:
        print("Yes, your voice is recognized in the dataset.")
        send_alert_signal()
    else:
        print("No, your voice is not recognized in the dataset.")

# Kết nối với Arduino qua cổng COM
arduino = serial.Serial(port='COM7', baudrate=9600, timeout=.1)

def send_alert_signal():
    arduino.write(b'1')  # Gửi tín hiệu "ALERT" đến Arduino
    time.sleep(1)            # Đợi một chút để Arduino xử lý

if __name__ == "__main__":
    # Load the trained SVM model and label encoder
    model = joblib.load('svm_model.pkl')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

    # Recognize the voice
    recognize_voice(model, label_encoder)