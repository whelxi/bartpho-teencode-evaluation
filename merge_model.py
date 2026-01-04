import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# CẤU HÌNH
BASE_MODEL = "vinai/bartpho-syllable"
ADAPTER_PATH = "whelxi/bartpho-teencode"
NEW_REPO_NAME = "whelxi/bartpho-teencode-merged" # Tên model mới sẽ tạo
HF_TOKEN = "" # Thay token Write của bạn vào đây

print("⏳ Đang tải model (việc này tốn RAM)...")

# 1. Load Base Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16, # Dùng float16 cho nhẹ
    device_map="auto"
)

# 2. Load Adapter
print("🔗 Đang tải Adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# 3. Gộp (Merge) Adapter vào Base Model
print("🔄 Đang gộp model (Merge & Unload)...")
model = model.merge_and_unload()

# 4. Upload lên Hugging Face
print(f"☁️  Đang đẩy model mới lên: {NEW_REPO_NAME}...")
try:
    # Login thủ công nếu cần, hoặc truyền token trực tiếp
    model.push_to_hub(NEW_REPO_NAME, token=HF_TOKEN, private=False) # Để private=False để dùng API free
    tokenizer.push_to_hub(NEW_REPO_NAME, token=HF_TOKEN, private=False)
    print("✅ THÀNH CÔNG! Hãy dùng tên model mới này trong file test_api.py")
except Exception as e:
    print(f"❌ Lỗi Upload: {e}")