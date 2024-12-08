import pyaudio
import wave
import os
import librosa
import numpy as np
import pandas as pd
import keyboard  # Đảm bảo bạn đã cài đặt thư viện keyboard
from sklearn.preprocessing import LabelEncoder


def record_audio(label, duration=5):
    # Các thông số âm thanh
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
        if keyboard.is_pressed("space"):  # Bắt đầu ghi âm khi nhấn phím Space
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

    # Lưu file âm thanh với tên tương ứng với label (ví dụ: "person1_1.wav")
    directory = "audio_dataset"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Đặt tên tệp dựa trên nhãn người nói (label)
    filename = f"{directory}/{label}_{len(os.listdir(directory)) + 1}.wav"
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    print(f"Saved recording as {filename}")


def extract_features(file_path):
    # Trích xuất MFCC từ tệp âm thanh
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean


def create_dataset():
    # Nhập tên người nói để gắn nhãn
    label = input("Enter your name or ID: ")

    data = []
    label_encoder = LabelEncoder()
    label_encoder.fit([label])  # Mã hóa label duy nhất cho người nói

    # Lặp để thu thập nhiều giọng nói
    while True:
        # Ghi âm giọng nói cá nhân
        record_audio(label)  # Ghi âm và lưu file âm thanh

        # Trích xuất đặc trưng từ các tệp âm thanh
        files = [f for f in os.listdir(f'audio_dataset') if f.endswith('.wav')]
        for file in files:
            file_path = f'audio_dataset/{file}'
            features = extract_features(file_path)
            # Mã hóa label và thêm vào cuối danh sách đặc trưng
            encoded_label = label_encoder.transform([label])[0]
            data.append(features + [encoded_label])  # Thêm nhãn đã mã hóa vào cuối

        # Hỏi người dùng nếu họ muốn thu thập thêm giọng nói
        more_data = input("Do you want to record more audio for this label? (y/n): ")
        if more_data.lower() != 'y':
            break  # Dừng vòng lặp nếu nhập 'n'

    # Chuyển thành dataframe và lưu thành CSV
    if data:  # Kiểm tra nếu có dữ liệu để lưu
        num_features = len(data[0])  # Số lượng đặc trưng trong mỗi dòng dữ liệu
        column_names = [f"mfcc_{i + 1}" for i in range(num_features - 1)] + ["label"]
        df = pd.DataFrame(data)
        df.columns = column_names  # Cập nhật tên cột phù hợp
        df.to_csv('dataset.csv', index=False)
        print("Dataset saved as 'dataset.csv'")
    else:
        print("No data to save.")


if __name__ == "__main__":
    create_dataset()  # Thu thập dữ liệu và tạo dataset
