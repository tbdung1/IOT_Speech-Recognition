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

def save_dataset(data, file_path):
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

def create_recognized_dataset_from_existing_files(directory="audio_test_dataset"):
    recognized_data = []
    label_encoder = LabelEncoder()

    # Lấy tất cả các tệp trong thư mục
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    labels = [file.split('_')[0] for file in files]  # Lấy nhãn từ tên tệp
    label_encoder.fit(labels)  # Mã hóa tất cả các nhãn

    for file in files:
        file_path = f"{directory}/{file}"
        features = extract_features(file_path)  # Trích xuất MFCC
        encoded_label = label_encoder.transform([file.split('_')[0]])[0]  # Mã hóa nhãn
        recognized_data.append(features.tolist() + [encoded_label])  # Thêm đặc trưng và nhãn

    # Chuyển dữ liệu thành DataFrame và lưu thành CSV
    if recognized_data:
        save_dataset(recognized_data, 'recognized_voice_dataset.csv')
    else:
        print("No recognized voice data found in directory!")

def create_non_recognized_dataset_from_existing_files(directory="audio_test_dataset"):
    non_recognized_data = []
    non_recognized_label = "non_recognized"  # Gắn nhãn cho giọng nói không nhận dạng được
    label_encoder = LabelEncoder()
    label_encoder.fit([non_recognized_label])  # Mã hóa label cho giọng nói không nhận dạng

    # Lấy tất cả các tệp trong thư mục
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]

    for file in files:
        file_path = f"{directory}/{file}"
        features = extract_features(file_path)  # Trích xuất MFCC
        encoded_label = label_encoder.transform([non_recognized_label])[0]  # Mã hóa nhãn
        non_recognized_data.append(features.tolist() + [encoded_label])  # Thêm đặc trưng và nhãn

    # Chuyển dữ liệu thành DataFrame và lưu thành CSV
    if non_recognized_data:
        save_dataset(non_recognized_data, 'non_recognized_voice_dataset.csv')
    else:
        print("No non-recognized voice data found in directory!")

def create_dataset_from_existing_files(directory="audio_dataset", directory_non="audio_test_dataset"):
    # Tạo dataset cho giọng nói đã đăng ký
    create_recognized_dataset_from_existing_files(directory)

    # Tạo dataset cho giọng nói không đăng ký
    # create_non_recognized_dataset_from_existing_files(directory_non)

if __name__ == "__main__":
    create_dataset_from_existing_files()
