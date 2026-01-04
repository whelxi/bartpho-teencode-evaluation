import torch
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL = "vinai/bartpho-syllable"
ADAPTER_PATH = "whelxi/bartpho-teencode" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Tokenizer & Model
try:
    print("⏳ Đang tải Tokenizer và Base Model (Public)...")
    
    # Bỏ tham số token=...
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    # 2. Load LoRA Adapter
    print(f"🔗 Đang tải Adapter từ: {ADAPTER_PATH}...")
    
    # Bỏ tham số token=...
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("✅ Đã load model thành công (Chế độ Public)!")

except Exception as e:
    print(f"\n❌ LỖI LOAD MODEL: {e}")
    print("👉 Kiểm tra lại:")
    print("   1. HF_TOKEN đã đúng chưa và có quyền 'Read' không?")
    print("   2. Tên repo 'whelxi/bartpho-teencode' có chính xác không?")
    sys.exit(1)

def normalize_teencode(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True, 
        padding="max_length"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True,
            length_penalty=1.0
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- GIAO DIỆN CHAT TRONG TERMINAL ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("☁️   TEST MODEL TỪ HUGGING FACE CLOUD  ☁️")
    print("="*60)
    print("👉 Hướng dẫn: Nhập câu teencode rồi Enter.")
    print("👉 Gõ 'exit', 'quit' hoặc 'q' để thoát.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n📝 Teencode: ").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Tạm biệt!")
                break
            
            if not user_input:
                continue

            print("☁️  Đang gọi model...", end='\r') 
            
            result = normalize_teencode(user_input)
            
            print(" " * 20, end='\r') 
            print(f"✨ Tiếng Việt: {result}")

        except KeyboardInterrupt:
            print("\n\n👋 Đã dừng chương trình.")
            break
        except Exception as e:
            print(f"❌ Lỗi xử lý: {e}")