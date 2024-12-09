import pyaudio
import wave
import os
import librosa
import numpy as np
import pandas as pd
import keyboard  # Ensure you have installed the keyboard library
from sklearn.preprocessing import LabelEncoder

def record_audio(label, duration=5):
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print(f"Recording for '{label}'...")
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

    # Save the audio file with a name corresponding to the label
    directory = "audio_dataset"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Name the file based on the label
    filename = f"{directory}/{label}_{len(os.listdir(directory)) + 1}.wav"
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    print(f"Saved recording as {filename}")

def extract_features(file_path):
    # Extract MFCC features from the audio file
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def create_recognized_dataset(label):
    """Collect and create dataset for recognized voice"""
    recognized_data = []
    label_encoder = LabelEncoder()
    label_encoder.fit([label])  # Encode the unique label for the speaker

    print("Start recording recognized voice (your voice)...")
    while True:
        record_audio(label)  # Record and save the audio file

        # Extract features from the audio files
        files = [f for f in os.listdir('audio_dataset') if f.endswith('.wav')]
        for file in files:
            file_path = f'audio_dataset/{file}'
            features = extract_features(file_path)
            encoded_label = label_encoder.transform([label])[0]
            recognized_data.append(features.tolist() + [encoded_label])

        more_data = input("Do you want to record more audio for the recognized voice? (y/n): ")
        if more_data.lower() != 'y':
            break  # Stop the loop if 'n' is entered

    # Save the collected data to the dataset
    if recognized_data:
        save_dataset(recognized_data, 'recognized_voice_dataset.csv')
    else:
        print("No recognized data to save.")

def create_non_recognized_dataset():
    """Collect and create dataset for non-recognized voice"""
    non_recognized_data = []
    non_recognized_label = "non_recognized"  # Label for non-recognized voice
    label_encoder = LabelEncoder()
    label_encoder.fit([non_recognized_label])  # Encode the label for non-recognized voice

    print("Start recording unrecognized voice (someone else's voice)...")
    while True:
        record_audio(non_recognized_label)  # Record and save the audio file

        # Extract features from the audio files
        files = [f for f in os.listdir('audio_dataset') if f.endswith('.wav')]
        for file in files:
            file_path = f'audio_dataset/{file}'
            features = extract_features(file_path)
            encoded_label = label_encoder.transform([non_recognized_label])[0]
            non_recognized_data.append(features.tolist() + [encoded_label])

        more_data = input("Do you want to record more audio for the unrecognized voice? (y/n): ")
        if more_data.lower() != 'y':
            break  # Stop the loop if 'n' is entered

    # Save the collected data to the dataset
    if non_recognized_data:
        save_dataset(non_recognized_data, 'non_recognized_voice_dataset.csv')
    else:
        print("No unrecognized data to save.")

def save_dataset(data, file_path='dataset.csv'):
    # If the file exists, read the current data
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        num_features = len(data[0])  # Number of features in each row
        column_names = [f"mfcc_{i + 1}" for i in range(num_features - 1)] + ["label"]
        new_data = pd.DataFrame(data, columns=column_names)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        # If the file does not exist, create new data
        num_features = len(data[0])  # Number of features in each row
        column_names = [f"mfcc_{i + 1}" for i in range(num_features - 1)] + ["label"]
        combined_data = pd.DataFrame(data, columns=column_names)

    # Save the combined data to the file
    combined_data.to_csv(file_path, index=False)
    print(f"Dataset updated and saved as '{file_path}'")

if __name__ == "__main__":
    # Collect data for recognized and non-recognized voices
    label = input("Enter your name or ID: ")
    create_recognized_dataset(label)
    create_non_recognized_dataset()