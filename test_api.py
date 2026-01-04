from gradio_client import Client

# 1. Khởi tạo Client trỏ vào Space của bạn
# Space ID lấy từ URL: https://huggingface.co/spaces/Whelxi/bartpho-teencode
client = Client("Whelxi/bartpho-teencode")

print("Đang gửi request lên Space...")

# 2. Gửi dữ liệu (predict)
# api_name="/predict" là mặc định cho gr.Interface
try:
    result = client.predict(
            text="chào mng nhé",  # Input text của bạn
            api_name="/predict"
    )
    
    print("-" * 20)
    print("KẾT QUẢ TRẢ VỀ:")
    print(result)
    print("-" * 20)
    
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")