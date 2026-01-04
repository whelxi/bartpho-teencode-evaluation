# FILE: prepare_data_advanced.py
import re
import json
import os
import random
import string  # <--- ĐÃ THÊM DÒNG NÀY (QUAN TRỌNG)
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset
from unidecode import unidecode
from collections import defaultdict

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
RAW_DIR = "./scientific_data"
PROCESSED_DIR = "./data"
# Tên file đã khớp với hình ảnh của bạn
DICT_PATH = os.path.join(RAW_DIR, "vi-nsw-dict.json")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# 1. BẢNG MÃ LỖI GÕ (TYPO)
TYPO_MAP = {
    'á': ['a1', 'as'], 'à': ['a2', 'af'], 'ả': ['a3', 'ar'], 'ã': ['a4', 'ax'], 'ạ': ['a5', 'aj'],
    'ă': ['a8', 'aw'], 'ắ': ['a81', 'aws'], 'ằ': ['a82', 'awf'], 'ặ': ['a85', 'awj'],
    'â': ['a6', 'aa'], 'ấ': ['a61', 'aas'], 'ầ': ['a62', 'aaf'], 'ậ': ['a65', 'aaj'],
    'đ': ['d9', 'dd'], 'é': ['e1', 'es'], 'è': ['e2', 'ef'], 'ẻ': ['e3', 'er'], 'ẽ': ['e4', 'ex'], 'ẹ': ['e5', 'ej'],
    'ê': ['e6', 'ee'], 'ế': ['e61', 'ees'], 'ề': ['e62', 'eef'], 'ệ': ['e65', 'eej'],
    'í': ['i1', 'is'], 'ì': ['i2', 'if'], 'ỉ': ['i3', 'ir'], 'ĩ': ['i4', 'ix'], 'ị': ['i5', 'ij'],
    'ó': ['o1', 'os'], 'ò': ['o2', 'of'], 'ỏ': ['o3', 'or'], 'õ': ['o4', 'ox'], 'ọ': ['o5', 'oj'],
    'ô': ['o6', 'oo'], 'ố': ['o61', 'oos'], 'ồ': ['o62', 'oof'], 'ộ': ['o65', 'ooj'],
    'ơ': ['o7', 'ow'], 'ớ': ['o71', 'ows'], 'ờ': ['o72', 'owf'], 'ợ': ['o75', 'owj'],
    'ú': ['u1', 'us'], 'ù': ['u2', 'uf'], 'ủ': ['u3', 'ur'], 'ũ': ['u4', 'ux'], 'ụ': ['u5', 'uj'],
    'ư': ['u7', 'uw'], 'ứ': ['u71', 'uws'], 'ừ': ['u72', 'uwf'], 'ự': ['u75', 'uwj'],
    'ý': ['y1', 'ys'], 'ỳ': ['y2', 'yf'], 'ỷ': ['y3', 'yr'], 'ỹ': ['y4', 'yx'], 'ỵ': ['y5', 'yj']
}

# 2. LOAD & ĐẢO NGƯỢC TỪ ĐIỂN
print("⚙️ ĐANG LOAD VÀ ĐẢO NGƯỢC TỪ ĐIỂN...")
rev_teencode_dict = defaultdict(list)

if os.path.exists(DICT_PATH):
    try:
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, dict):
                for teencode, standard_list in content.items():
                    k = teencode.lower().strip()
                    vals = standard_list if isinstance(standard_list, list) else [standard_list]
                    for v in vals:
                        v_clean = v.lower().strip()
                        rev_teencode_dict[v_clean].append(k)
            else:
                print("⚠️ Từ điển không đúng định dạng dict.")
    except Exception as e:
        print(f"⚠️ Lỗi đọc file từ điển: {e}")
else:
    print("⚠️ Không tìm thấy file từ điển.")

print(f"✅ Đã load {len(rev_teencode_dict)} từ chuẩn có thể map sang teencode.")

# Hàm sinh lỗi gõ
def simulate_typo(word):
    chars = list(word)
    new_chars = []
    has_changed = False
    for char in chars:
        if char in TYPO_MAP and random.random() < 0.3:
            typo_char = random.choice(TYPO_MAP[char])
            new_chars.append(typo_char)
            has_changed = True
        else:
            new_chars.append(char)
    return "".join(new_chars) if has_changed else word

# Hàm Augmentation
def augment_text_advanced(text, ratio=0.5):
    if not isinstance(text, str): return ""
    
    words = text.split()
    new_words = []
    
    for word in words:
        # Cần import string để dùng string.punctuation ở đây
        clean_word = word.strip(string.punctuation).lower()
        rand = random.random()
        
        # 1. Teencode Reverse
        if clean_word in rev_teencode_dict and rand < 0.4:
            teencode_options = rev_teencode_dict[clean_word]
            chosen_teencode = random.choice(teencode_options)
            if word[0].isupper():
                chosen_teencode = chosen_teencode.capitalize()
            
            prefix = word[:len(word)-len(word.lstrip(string.punctuation))]
            suffix = word[len(word.rstrip(string.punctuation)):]
            new_words.append(prefix + chosen_teencode + suffix)
            
        # 2. Unidecode
        elif rand < 0.6:
            new_words.append(unidecode(word).lower())
            
        # 3. Typo
        elif rand < 0.7:
             new_words.append(simulate_typo(word))
             
        # 4. Giữ nguyên
        else:
            new_words.append(word)
            
    return " ".join(new_words)

# ==============================================================================
# XỬ LÝ DỮ LIỆU
# ==============================================================================
all_datasets = []

# --- NHÓM 1: PARALLEL DATA ---
print("🔹 Xử lý Parallel Data (ViLexNorm, VSEC)...")

# 1.1 ViLexNorm
try:
    path_vilex = os.path.join(RAW_DIR, "vilexnorm.jsonl")
    if os.path.exists(path_vilex):
        ds = load_dataset("json", data_files=path_vilex, split="train")
        ds_clean = ds.map(lambda x: {"input": x["original"], "output": x["normalized"]}, 
                          remove_columns=ds.column_names)
        all_datasets.append(ds_clean)
        print(f"   - Đã thêm ViLexNorm: {len(ds_clean)} mẫu.")
except Exception as e: print(f"⚠️ ViLexNorm Error: {e}")

# 1.2 VSEC
try:
    path_vsec = os.path.join(RAW_DIR, "vsec.jsonl")
    if os.path.exists(path_vsec):
        ds_vsec = load_dataset("json", data_files=path_vsec, split="train")
        def map_vsec(x):
            out_text = x["corrected_text"] if x["corrected_text"] else x["text"]
            return {"input": x["text"], "output": out_text}
        
        ds_vsec_clean = ds_vsec.map(map_vsec, remove_columns=ds_vsec.column_names)
        all_datasets.append(ds_vsec_clean)
        print(f"   - Đã thêm VSEC: {len(ds_vsec_clean)} mẫu.")
except Exception as e: print(f"⚠️ VSEC Error: {e}")


# --- NHÓM 2: CONTEXT DATA ---
print("🔹 Xử lý Context Data (ViHSD, VSMEC)...")
context_files = ["vihsd.jsonl", "vsmec.jsonl"]
context_data = []

for fname in context_files:
    fpath = os.path.join(RAW_DIR, fname)
    if os.path.exists(fpath):
        print(f"   -> Đang đọc {fname}...")
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = item.get("free_text") or item.get("sentence") or item.get("text")
                    if text and isinstance(text, str) and len(text.split()) > 3:
                        fake_input = augment_text_advanced(text, ratio=0.8)
                        context_data.append({"input": fake_input, "output": text})
                except Exception as e: 
                    # In lỗi nếu có để debug
                    # print(e) 
                    continue

if context_data:
    limit = min(len(context_data), 30000)
    ds_context = Dataset.from_pandas(pd.DataFrame(context_data)).shuffle(seed=42).select(range(limit))
    all_datasets.append(ds_context)
    print(f"   - Đã tạo giả lập từ Context Data: {len(ds_context)} mẫu.")
else:
    print("⚠️ Cảnh báo: Không tạo được mẫu Context Data nào (Kiểm tra lại import string hoặc cấu trúc file).")


# --- NHÓM 3: KNOWLEDGE DATA ---
print("🔹 Xử lý Knowledge Data (WikiAnn, WikiLingua)...")
knowledge_files = ["wikiann_ner.jsonl", "wikilingua.jsonl"] 
knowledge_data = []

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def detokenize_naive(tokens):
    text = " ".join(tokens)
    text = re.sub(r'\s+([,.:;?!])', r'\1', text)
    return text

for fname in knowledge_files:
    fpath = os.path.join(RAW_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = ""
                    if "article" in item:
                         doc = item["article"].get("document", [])
                         if isinstance(doc, list): text = " ".join(doc)
                         elif isinstance(doc, str): text = doc
                    elif "tokens" in item:
                        text = detokenize_naive(item["tokens"])

                    if text and len(text) > 10:
                        text = text[:600]
                        no_tone_text = unidecode(text).lower()
                        clean_input = remove_punctuation(no_tone_text)
                        clean_input = " ".join(clean_input.split())
                        knowledge_data.append({"input": clean_input, "output": text})
                except: continue

if knowledge_data:
    ds_know = Dataset.from_pandas(pd.DataFrame(knowledge_data)).shuffle(seed=42).select(range(min(len(knowledge_data), 10000)))
    all_datasets.append(ds_know)
    print(f"   - Đã xử lý Knowledge Data: {len(ds_know)} mẫu.")

# 3. LƯU FILE
if not all_datasets:
    print("❌ LỖI: Không có dữ liệu input!")
    exit()

print("🔹 Đang gộp và trộn dữ liệu...")
full_dataset = concatenate_datasets(all_datasets).shuffle(seed=42)
split_ds = full_dataset.train_test_split(test_size=0.1)

print(f"✅ TỔNG CỘNG: {len(full_dataset)} mẫu.")
print(f"   - Train: {len(split_ds['train'])} -> lưu tại {PROCESSED_DIR}/train.jsonl")
print(f"   - Valid: {len(split_ds['test'])} -> lưu tại {PROCESSED_DIR}/valid.jsonl")

split_ds["train"].to_json(os.path.join(PROCESSED_DIR, "train.jsonl"), force_ascii=False)
split_ds["test"].to_json(os.path.join(PROCESSED_DIR, "valid.jsonl"), force_ascii=False)

print("🎉 DONE! Sẵn sàng train.")