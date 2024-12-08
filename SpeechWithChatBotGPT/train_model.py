import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder


def load_data():
    # Đọc dataset đã tạo từ file CSV
    df = pd.read_csv('dataset.csv')
    X = df.iloc[:, :-1].values  # Đặc trưng (MFCC)
    y = df['label'].values  # Nhãn (labels)

    # Kiểm tra và mã hóa nhãn nếu chưa
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Mã hóa nhãn thành số nguyên
    return X, y


def train_svm(X, y):
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Khởi tạo và huấn luyện mô hình SVM
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Dự đoán và kiểm tra độ chính xác
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Lưu mô hình vào tệp
    joblib.dump(model, 'svm_model.pkl')
    print("Model saved as 'svm_model.pkl'")


if __name__ == "__main__":
    X, y = load_data()  # Đọc dữ liệu
    train_svm(X, y)     # Huấn luyện mô hình SVM
