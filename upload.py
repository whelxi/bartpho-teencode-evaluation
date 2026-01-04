import os
from huggingface_hub import HfApi, create_repo

# --- CẤU HÌNH ---
LOCAL_MODEL_PATH = "./bartpho-teencode-lora"  # Thư mục chứa model sau khi train
HF_USERNAME = "whelxi"              # Đổi thành username HF của bạn
REPO_NAME = "bartpho-teencode"  # Tên repo bạn muốn đặt
# ----------------

def upload_to_huggingface():
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    
    print(f"🚀 Đang chuẩn bị upload lên: {repo_id}")
    
    # 1. Tạo repo nếu chưa có
    api = HfApi()
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print("✅ Repo đã sẵn sàng.")
    except Exception as e:
        print(f"⚠️ Lưu ý: {e}")

    # 2. Upload toàn bộ folder
    print("⏳ Đang upload files (adapter, tokenizer config, etc)...")
    api.upload_folder(
        folder_path=LOCAL_MODEL_PATH,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload trained LoRA adapter for teencode normalization"
    )
    
    print(f"🎉 Thành công! Xem model tại: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # Kiểm tra xem folder có tồn tại không
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ Lỗi: Không tìm thấy thư mục '{LOCAL_MODEL_PATH}'")
    else:
        upload_to_huggingface()