import requests

# URL của Blob với SAS token
blob_url = "https://<your_account_name>.blob.core.windows.net/<container_name>/<blob_name>?<SAS_token>"

# Gửi yêu cầu GET đến URL và tải dữ liệu về
response = requests.get(blob_url)

# Kiểm tra trạng thái phản hồi
if response.status_code == 200:
    # Lưu mô hình vào file
    with open("svm_model.pkl", "wb") as file:
        file.write(response.content)
    print("Model downloaded successfully!")
else:
    print(f"Failed to download the model. Status code: {response.status_code}")
