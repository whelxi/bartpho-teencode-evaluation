import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import evaluate
import os

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
BASE_MODEL = "vinai/bartpho-syllable"
# Đổi thành đường dẫn local của bạn nếu muốn (vd: "./bartpho-teencode-lora")
# Hoặc dùng repo HF nếu bạn đã push lên
ADAPTER_PATH = "whelxi/bartpho-teencode" 
INPUT_CSV = "test.csv"
OUTPUT_CSV = "evaluation_results.csv"
CHART_FILE = "evaluation_chart.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"⚙️ Thiết bị đang sử dụng: {DEVICE.upper()}")

# ==============================================================================
# 1. LOAD MODEL & METRICS
# ==============================================================================
try:
    print("⏳ Đang tải Model và Metrics...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    
    # Load Adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    # Load Metrics
    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    print("✅ Load thành công!")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    print("👉 Nếu dùng model local, hãy kiểm tra lại đường dẫn ADAPTER_PATH.")
    exit()

# ==============================================================================
# 2. HÀM DỰ ĐOÁN
# ==============================================================================
def predict_batch(texts, batch_size=8):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Đang dịch"):
        batch_texts = texts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_texts, 
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
                early_stopping=True
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results

# ==============================================================================
# 3. CHẠY TRÊN FILE TEST.CSV
# ==============================================================================
if os.path.exists(INPUT_CSV):
    print(f"📂 Đang đọc file {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Kiểm tra cột
    if 'original' not in df.columns or 'normalized' not in df.columns:
        print("❌ File CSV thiếu cột 'original' hoặc 'normalized'.")
        exit()
        
    inputs = df['original'].astype(str).tolist()
    references = df['normalized'].astype(str).tolist()
    
    # Chạy dự đoán
    predictions = predict_batch(inputs)
    
    # Lưu kết quả vào DataFrame
    df['prediction'] = predictions
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"💾 Đã lưu kết quả dự đoán vào: {OUTPUT_CSV}")
    
    # ==========================================================================
    # 4. TÍNH TOÁN CHỈ SỐ (METRICS)
    # ==========================================================================
    print("📊 Đang tính toán metrics...")
    
    # BLEU
    bleu_score = bleu.compute(predictions=predictions, references=references)
    
    # ROUGE
    rouge_score = rouge.compute(predictions=predictions, references=references)
    
    metrics = {
        "BLEU": bleu_score['score'],
        "ROUGE-1": rouge_score['rouge1'] * 100, # Rouge trả về 0-1, nhân 100 cho đẹp
        "ROUGE-2": rouge_score['rouge2'] * 100,
        "ROUGE-L": rouge_score['rougeL'] * 100
    }
    
    print("\n" + "="*40)
    print("KẾT QUẢ ĐÁNH GIÁ MODEL")
    print("="*40)
    for k, v in metrics.items():
        print(f"   - {k}: {v:.2f}")
    print("="*40)

    # ==========================================================================
    # 5. VẼ BIỂU ĐỒ
    # ==========================================================================
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Tạo DataFrame cho chart
    chart_df = pd.DataFrame({
        'Metric': list(metrics.keys()), 
        'Score': list(metrics.values())
    })
    
    ax = sns.barplot(x='Metric', y='Score', data=chart_df, palette="viridis")
    
    # Thêm số liệu lên cột
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontweight='bold')

    plt.title(f"Model Performance Metrics\n(Model: {ADAPTER_PATH})", fontsize=14)
    plt.ylabel("Score (0-100)")
    plt.ylim(0, 110) # Cho dư ra một chút ở trên
    plt.tight_layout()
    
    plt.savefig(CHART_FILE)
    print(f"🖼️  Đã lưu biểu đồ vào: {CHART_FILE}")
    print("🎉 Hoàn tất!")

else:
    print(f"❌ Không tìm thấy file {INPUT_CSV}")