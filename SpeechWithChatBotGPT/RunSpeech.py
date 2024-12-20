import pyaudio
import wave
import os
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import keyboard
import serial
import time
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from dotenv import load_dotenv
import openai
from transformers import pipeline

load_dotenv()
# Tải và thiết lập kết nối với OpenAI API
openai.api_key = os.getenv('API_KEY')

def download_model_from_azure(container_name, blob_name, local_file_name, connection_string):
    # Kết nối với Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # Tải mô hình từ Blob Storage
    blob_client = container_client.get_blob_client(blob_name)
    with open(local_file_name, "wb") as file:
        data = blob_client.download_blob()
        data.readinto(file)

def record_audio(duration=5):
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

    # Chuyển giọng nói sang văn bản
    # transcribed_text = transcribe_audio(file_path)
    # print(f"Transcribed Text: {transcribed_text}")

    # Extract features from the new voice sample
    features = extract_features(file_path).reshape(1, -1)
    # print(features)

    # Predict using the SVM model
    prediction = model.predict(features)
    label = prediction[0]
    print("Predicted label:", prediction[0])

    # Remove the temporary recording file
    os.remove(file_path)

    labels_result = [0, 1]

    # Print the result
    if label == 0:
        print("Yes, your voice is recognized in the dataset.")
        send_alert_signal() # Gửi tín hiệu cảnh báo đến Arduino
        # return transcribed_text  # Trả về văn bản nhận dạng được
    else:
        print("No, your voice is not recognized in the dataset.")
        return None


def get_chatgpt_response(user_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_text}]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error getting response from ChatGPT: {e}")
        return None

def PrintText(text):
    for i in text:
        print(i, end="", flush=True)
        time.sleep(0.04)
    print("\n")


def transcribe_audio(filename):
    # Khởi tạo pipeline cho chuyển giọng nói thành văn bản sử dụng mô hình Whisper
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large")

    # Chuyển đổi âm thanh từ tệp .wav thành văn bản
    output = transcriber(filename)["text"]


    return output
# Kết nối với Arduino qua cổng COM
arduino = serial.Serial(port='COM7', baudrate=9600, timeout=1)
time.sleep(2)

def send_alert_signal():
    if arduino.is_open:
        arduino.write(b'1')  # Gửi tín hiệu "ALERT" đến Arduino
        time.sleep(1)  # Đợi một chút để Arduino xử lý
    else:
        print("Failed to connect to Arduino!")
if __name__ == "__main__":
    # Thông tin kết nối Azure
    load_dotenv()
    connection_string = os.getenv('AZURE_CONNECTION_STRING')
    container_name = "svmmodel"
    model_blob_name = "svm_model.pkl"
    classes_blob_name = "classes.npy"

    # Tải mô hình và encoder từ Azure Blob Storage
    download_model_from_azure(container_name, model_blob_name, "svm_model.pkl", connection_string)
    download_model_from_azure(container_name, classes_blob_name, "classes.npy", connection_string)

    # Load the trained SVM model and label encoder
    model = joblib.load('svm_model.pkl')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

    # Gửi lời chào và yêu cầu nhận dạng giọng nói
    PrintText("Hello, Nice to meet you! Have a good day!")
    # recognize_voice(model, label_encoder)

    # Nhận dạng giọng nói và lưu kết quả
    transcribed_text = recognize_voice(model, label_encoder)

    # if transcribed_text:
    #     # Chuyển văn bản sang ChatGPT nếu nhận dạng thành công
    #     user_text = transcribed_text
    #     print(f"User's transcribed text: {user_text}")
    #
    #     chatgpt_response = get_chatgpt_response(user_text)
    #     if chatgpt_response:
    #         print(f"ChatGPT's response: {chatgpt_response}")
    #     else:
    #         print("No response ChatGPT.")
