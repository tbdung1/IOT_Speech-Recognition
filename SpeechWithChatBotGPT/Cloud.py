from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from dotenv import load_dotenv
import os

# Kết nối đến Blob Storage
load_dotenv()
# Truy xuất giá trị của biến môi trường
connection_string = os.getenv('AZURE_CONNECTION_STRING')
container_name = "svmmodel"

# Các file cần upload
model_blob_name = "svm_model.pkl"
classes_blob_name = "classes.npy"
model_local_path = r"C:\Users\84946\OneDrive\Desktop\Documents PTIT\HK1_nam4\IOT\Speech-Recognition\IOT_Speech-Recognition\SpeechWithChatBotGPT\svm_model.pkl"
classes_local_path = r"C:\Users\84946\OneDrive\Desktop\Documents PTIT\HK1_nam4\IOT\Speech-Recognition\IOT_Speech-Recognition\SpeechWithChatBotGPT\classes.npy"

# Tải mô hình lên
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Tải mô hình svm_model.pkl lên
blob_client = container_client.get_blob_client(model_blob_name)
with open(model_local_path, "rb") as data:
    blob_client.upload_blob(data, overwrite=True)
print("Model uploaded successfully!")

# Tải file classes.npy lên
blob_client = container_client.get_blob_client(classes_blob_name)
with open(classes_local_path, "rb") as data:
    blob_client.upload_blob(data, overwrite=True)
print("Classes file uploaded successfully!")
