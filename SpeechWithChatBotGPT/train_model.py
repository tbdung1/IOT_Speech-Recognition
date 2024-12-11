import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


def load_data_from_csv(file_paths):
    data = []
    labels = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)  # Đọc dataset từ file CSV
        print(f"Dataset from {file_path} head:\n", df.head())  # Print the first few rows of the dataset
        X = df.iloc[:, :-1].values  # Đặc trưng (MFCC)
        y = df['label'].values  # Nhãn (labels)

        # Thay thế NaN bằng giá trị trung bình của mỗi cột
        df = df.fillna(df.mean())

        X = df.iloc[:, :-1].values  # Đặc trưng (MFCC) sau khi thay thế NaN
        y = df['label'].values  # Nhãn (labels)

        # Chuyển đổi dữ liệu từ ndarray thành DataFrame để có thể concat
        data.append(pd.DataFrame(X))
        labels.append(pd.Series(y))

    # Kết hợp dữ liệu từ các dataset
    X_combined = pd.concat(data, axis=0, ignore_index=True)  # ignore_index để tránh lỗi khi index không khớp
    y_combined = pd.concat(labels, axis=0, ignore_index=True)

    # Kiểm tra và mã hóa nhãn nếu chưa
    label_encoder = LabelEncoder()
    y_combined = label_encoder.fit_transform(y_combined)  # Mã hóa nhãn thành số nguyên

    # Save the label encoder classes for later use
    np.save('classes.npy', label_encoder.classes_)
    return X_combined.values, y_combined  # Trả về numpy arrays


def train_svm(X, y):
    accuracies = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  # Chia dữ liệu thành 10 phần

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Khởi tạo và huấn luyện mô hình SVM
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)

        # Dự đoán và kiểm tra độ chính xác
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Fold {fold} - Accuracy: {accuracy * 100:.2f}%")

    # Vẽ biểu đồ độ chính xác qua từng epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), [acc * 100 for acc in accuracies], marker='o', linestyle='--', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('SVM Model Accuracy Over Epochs')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()

    # Lưu mô hình vào tệp
    model.fit(X, y)  # Huấn luyện lại mô hình với toàn bộ dữ liệu
    joblib.dump(model, 'svm_model.pkl')
    print("Model saved as 'svm_model.pkl'")


if __name__ == "__main__":
    file_paths = ['recognized_voice_dataset.csv', 'non_recognized_voice_dataset.csv']  # Đường dẫn tới các dataset
    X, y = load_data_from_csv(file_paths)  # Đọc dữ liệu từ các file CSV
    train_svm(X, y)  # Huấn luyện mô hình SVM
