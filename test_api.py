import sys
import time
from huggingface_hub import InferenceClient
# SỬA LỖI: Chỉ import HfHubHTTPError, bỏ HfTimeoutError
from huggingface_hub.utils import HfHubHTTPError

# --- CẤU HÌNH ---
# 👇 THAY TOKEN CỦA BẠN VÀO DƯỚI ĐÂY (Token quyền 'Read')
HF_TOKEN = "" 

# Tên repo trên Hugging Face
REPO_ID = "whelxi/bartpho-teencode"

print("⏳ Đang kết nối tới Hugging Face Inference API...")

# Khởi tạo Client
client = InferenceClient(model=REPO_ID, token=HF_TOKEN)

def normalize_teencode_api(text):
    try:
        # Gọi API
        response = client.text_generation(
            text,
            max_new_tokens=128,
            do_sample=False, 
            return_full_text=False
        )
        return response

    # --- BẮT LỖI CỤ THỂ ---
    except HfHubHTTPError as e:
        print(f"\n❌ LỖI API (HTTP {e.response.status_code}):")
        
        if e.response.status_code == 401:
            print("👉 Token không hợp lệ hoặc chưa điền HF_TOKEN.")
            print("👉 Hãy lấy token tại: https://huggingface.co/settings/tokens")
        elif e.response.status_code == 404:
            print(f"👉 Không tìm thấy model '{REPO_ID}'.")
        elif e.response.status_code == 503:
            print("👉 Server đang khởi động model (Cold start). Vui lòng đợi 30s rồi thử lại.")
        elif e.response.status_code == 400:
             print("👉 Lỗi Request: Có thể do API không hỗ trợ chạy trực tiếp Adapter LoRA (cần merge).")
        else:
            print(f"👉 Chi tiết lỗi: {e}")
        return None

    # SỬA LỖI: Dùng TimeoutError mặc định của Python hoặc Exception chung
    except TimeoutError:
        print("\n❌ LỖI: Quá thời gian chờ (Timeout). Mạng yếu hoặc server phản hồi chậm.")
        return None
        
    except Exception as e:
        import traceback
        print("\n❌ LỖI CHI TIẾT:")
        traceback.print_exc() # <--- Dòng này sẽ in ra nguyên nhân gốc rễ
        return None

# --- GIAO DIỆN CHAT ---
if __name__ == "__main__":
    if "hf_" not in HF_TOKEN:
        print("\n⚠️  CẢNH BÁO: Bạn chưa điền HF_TOKEN đúng. Code sẽ lỗi 401.")

    print("\n" + "="*60)
    print(f"☁️   CHẠY MODEL TRÊN SERVER HUGGING FACE ({REPO_ID}) ☁️")
    print("="*60)
    print("👉 Gõ 'exit' để thoát.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n📝 Teencode: ").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Tạm biệt!")
                break
            
            if not user_input:
                continue

            print("🚀 Đang gọi API...", end='\r')
            start_time = time.time()
            
            result = normalize_teencode_api(user_input)
            
            end_time = time.time()
            
            if result:
                print(" " * 40, end='\r') 
                print(f"✨ Tiếng Việt: {result}")
                print(f"⏱️  Thời gian: {end_time - start_time:.2f}s")
        except KeyboardInterrupt:
            print("\n👋 Đã dừng chương trình.")
            break