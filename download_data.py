import os
import requests
from datasets import load_dataset

# Cấu hình chung
DATA_DIR = "./scientific_data"
os.makedirs(DATA_DIR, exist_ok=True)

print("🚀 BẮT ĐẦU TẢI DỮ LIỆU (CẬP NHẬT MỚI)...")

# 1. Tải Từ điển NSW (Cập nhật nguồn mới)
# Lưu ý: Đã chuyển link từ 'blob' sang 'raw' để tải nội dung file JSON
DICT_URL = "https://raw.githubusercontent.com/AnhHoang0529/vn-nsw-dictionary/main/vi-nsw-dict.json"
dict_path = os.path.join(DATA_DIR, "vi-nsw-dict.json") 

try:
    print("📥 Tải NSW Dictionary (Mới)...")
    resp = requests.get(DICT_URL)
    with open(dict_path, "wb") as f:
        f.write(resp.content)
    print("   ✅ Xong.")
except Exception as e:
    print(f"⚠️ Lỗi tải từ điển: {e}")

# Hàm tải dataset từ HuggingFace
def download_hf(repo_id, save_name, subset=None, split='train'):
    print(f"📥 Tải {repo_id} -> {save_name}...")
    try:
        # Tải về
        ds = load_dataset(repo_id, subset, split=split, trust_remote_code=True)
        # Lưu tên chuẩn xác (.jsonl)
        ds.to_json(os.path.join(DATA_DIR, save_name), force_ascii=False)
        print(f"   ✅ Xong ({len(ds)} dòng).")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")

# --- DANH SÁCH TẢI (ĐÃ CẬP NHẬT) ---

# 1. ViLexNorm (Chuẩn hóa)
download_hf("visolex/vilexnorm", "vilexnorm.jsonl")

# 2. VSEC (Lỗi chính tả)
download_hf("nguyenthanhasia/vsec-vietnamese-spell-correction", "vsec.jsonl")

# 3. ViHSD (Context/Toxic)
download_hf("sonlam1102/vihsd", "vihsd.jsonl")

# 4. VSMEC (Context/Emotion)
download_hf("uit-nlp/vietnamese_students_feedback", "vsmec.jsonl")

# 5. WikiANN NER (Tên riêng)
# WikiANN cần subset='vi' để lấy tiếng Việt
download_hf("wikiann", "wikiann_ner.jsonl", subset="vi", split="train")

# 6. WikiLingua (Dấu câu) - Lấy mẫu 10k để prepare xử lý sau
download_hf("wiki_lingua", "wikilingua.jsonl", subset="vietnamese", split="train[:10000]")

print("\n🎉 ĐÃ TẢI XONG! Chạy tiếp 'prepare_data_final.py'")