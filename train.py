import numpy as np
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import evaluate
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# --- CÁC HÀM PHỤ TRỢ ---
def compute_metrics(eval_preds, tokenizer, metric):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    
    # === [FIX QUAN TRỌNG] ===
    # Trainer chèn -100 vào preds để padding, BARTpho decode bị lỗi.
    # Cần thay -100 về pad_token_id trước khi decode.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # ========================

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Xử lý labels (bạn đã làm đúng phần này, giữ nguyên)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

def preprocess_function(examples, tokenizer):
    inputs = [str(x) for x in examples["input"]]
    targets = [str(x) for x in examples["output"]]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    # Tokenize labels
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    # Xử lý padding của labels thành -100
    labels_ids = labels["input_ids"]
    labels_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_ids
    ]
    
    model_inputs["labels"] = labels_ids
    return model_inputs

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # CẤU HÌNH
    MODEL_NAME = "vinai/bartpho-syllable"
    OUTPUT_DIR = "./bartpho-teencode-lora"
    DATA_DIR = "./data"

    print(f"🚀 Bắt đầu Training với LoRA trên RTX 3070 Ti...")
    
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)

    # --- [FIX 1] Tắt cache để tránh xung đột với Gradient Checkpointing ---
    model.config.use_cache = False 

    # --- [FIX 2] Chỉ bật input grad, không gọi gradient_checkpointing_enable() thủ công ở đây
    # Hãy để TrainingArguments làm việc đó để đồng bộ config
    model.enable_input_require_grads()

    # --- TÍCH HỢP LORA ---
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=32,           
        lora_alpha=64,  
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"] 
    )
    
    model = get_peft_model(model, peft_config)
    print("📊 Thống kê tham số LoRA:")
    print_trainable_parameters(model)
    
    # 2. Load Data (Giữ nguyên)
    data_files = {"train": os.path.join(DATA_DIR, "train.jsonl"), 
                  "validation": os.path.join(DATA_DIR, "valid.jsonl")}
    dataset = load_dataset("json", data_files=data_files)

    # 3. Preprocess 
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=["input", "output"],
        fn_kwargs={"tokenizer": tokenizer} 
    )
    eval_subset = tokenized_datasets["validation"].shuffle(seed=42).select(range(200))

    # 4. Metric (Giữ nguyên)
    metric = evaluate.load("sacrebleu")
    def compute_metrics_wrapper(eval_preds):
        return compute_metrics(eval_preds, tokenizer, metric)

    # 5. Training Arguments (ĐÃ SỬA)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,  
        eval_strategy="steps",
        save_strategy="steps",
        
        predict_with_generate=True,  
        generation_max_length=128,   # Độ dài câu tối đa khi dịch thử

        # --- [CHANGE 1] GIẢM BATCH SIZE XUỐNG ---
        # 3070 Ti 8GB khá chật chội, giảm train xuống 8 và eval xuống 2
        per_device_train_batch_size=8,  # Giảm từ 16 -> 8
        per_device_eval_batch_size=2,   # Giảm từ 4 -> 2
        
        # --- [CHANGE 2] TĂNG ACCUMULATION ĐỂ BÙ LẠI BATCH SIZE ---
        # Cũ: 16 * 2 = 32 mẫu/lần update. Mới: 8 * 4 = 32 mẫu/lần update.
        # Kết quả train tương đương nhưng tốn ít RAM hơn.
        gradient_accumulation_steps=8,  
        
        # --- [CHANGE 3] QUAN TRỌNG CHO EVALUATION ---
        # Mặc định Trainer sẽ giữ toàn bộ kết quả dự đoán trên GPU cho đến khi eval xong.
        # Set = 1 để nó đẩy kết quả về CPU ngay lập tức sau mỗi step, giải phóng VRAM.
        eval_accumulation_steps=1,
        
        # === [FIX 3] CẤU HÌNH QUAN TRỌNG ĐỂ SỬA LỖI ===
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant': False}, # <--- DÒNG NÀY SẼ FIX LỖI "element 0"
        # ===============================================
        
        fp16=False,             
        bf16=True,              
        optim="adamw_torch",   
        dataloader_num_workers=0, # Windows fix (Giữ nguyên)
        
        group_by_length=True,
        learning_rate=3e-4, 
        num_train_epochs=5,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=eval_subset, 
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer, 
        compute_metrics=compute_metrics_wrapper,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

    # 6. Start Train
    print("🔥 Đang tiến hành training...")
    trainer.train()

    # 7. Save Final
    print("💾 Đang lưu Adapter model...")         
    trainer.save_model(OUTPUT_DIR)        
    tokenizer.save_pretrained(OUTPUT_DIR)   
    print(f"✅ Hoàn tất! Model LoRA đã lưu tại: {OUTPUT_DIR}")