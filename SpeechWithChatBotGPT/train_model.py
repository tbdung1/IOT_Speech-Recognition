import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


def load_data_from_csv(file_paths):
    data = []
    labels = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)  # Đọc dataset từ file CSV
        print(f"Dataset from {file_path} head:\n", df.head())  # Print the first few rows of the dataset
        df = df.fillna(df.mean())  # Thay thế NaN bằng giá trị trung bình của mỗi cột

        X = df.iloc[:, :-1].values  # Đặc trưng (MFCC)
        y = df['label'].values  # Nhãn (labels)

        data.append(pd.DataFrame(X))
        labels.append(pd.Series(y))

    # Kết hợp dữ liệu từ các dataset
    X_combined = pd.concat(data, axis=0, ignore_index=True)  # ignore_index để tránh lỗi khi index không khớp
    y_combined = pd.concat(labels, axis=0, ignore_index=True)

    # Mã hóa nhãn thành số nguyên
    label_encoder = LabelEncoder()
    y_combined = label_encoder.fit_transform(y_combined)

    # Lưu các class nhãn vào file để dùng lại
    np.save('classes.npy', label_encoder.classes_)

    return X_combined.values, y_combined


def train_and_save_tflite_model(X, y):
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

    # Xây dựng model TensorFlow
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),  # Số đặc trưng đầu vào
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  # Số lớp đầu ra
    ])

    # Biên dịch model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Huấn luyện model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Đánh giá mô hình
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Chuyển đổi model sang TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Lưu model dưới dạng file .tflite
    with open('svm_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("TensorFlow Lite model saved as 'svm_model.tflite'")


if __name__ == "__main__":
    file_paths = ['recognized_voice_dataset.csv', 'non_recognized_voice_dataset.csv']  # Đường dẫn tới các dataset
    X, y = load_data_from_csv(file_paths)  # Đọc dữ liệu từ các file CSV
    train_and_save_tflite_model(X, y)  # Huấn luyện và lưu model TensorFlow Lite
