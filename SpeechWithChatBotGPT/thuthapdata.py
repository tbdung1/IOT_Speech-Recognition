from sklearn.preprocessing import LabelEncoder
import librosa
import numpy as np
import pandas as pd
import os

def extract_features(file_path):
    # Trích xuất MFCC từ tệp âm thanh
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def save_dataset(data, file_path='dataset.csv'):
    # Nếu tệp tồn tại, đọc dữ liệu hiện tại
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        num_features = len(data[0])  # Số lượng đặc trưng trong mỗi dòng
        column_names = [f"mfcc_{i + 1}" for i in range(num_features - 1)] + ["label"]
        new_data = pd.DataFrame(data, columns=column_names)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        # Nếu tệp chưa tồn tại, tạo dữ liệu mới
        num_features = len(data[0])  # Số lượng đặc trưng trong mỗi dòng
        column_names = [f"mfcc_{i + 1}" for i in range(num_features - 1)] + ["label"]
        combined_data = pd.DataFrame(data, columns=column_names)
    # Ghi dữ liệu hợp nhất vào tệp
    combined_data.to_csv(file_path, index=False)
    print(f"Dataset updated and saved as '{file_path}'")

def create_dataset_from_existing_files(directory="audio_test_dataset"):
    data = []
    label_encoder = LabelEncoder()

    # Lấy tất cả các tệp trong thư mục
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    labels = [file.split('_')[0] for file in files]  # Lấy nhãn từ tên tệp
    label_encoder.fit(labels)  # Mã hóa tất cả các nhãn

    for file in files:
        file_path = f"{directory}/{file}"
        features = extract_features(file_path)  # Trích xuất MFCC
        encoded_label = label_encoder.transform([file.split('_')[0]])[0]  # Mã hóa nhãn
        data.append(features.tolist() + [encoded_label])  # Thêm đặc trưng và nhãn

    # Chuyển dữ liệu thành DataFrame và lưu thành CSV
    if data:
        save_dataset(data)
    else:
        print("No data found in directory!")

if __name__ == "__main__":
    create_dataset_from_existing_files()
