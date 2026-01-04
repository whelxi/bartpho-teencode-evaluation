# Vietnamese Teencode → Vietnamese Standard (BARTpho)

This project fine-tunes **BARTpho-base** to normalize Vietnamese *teencode* (informal / slang chat text) into **standard Vietnamese**.

It is designed to be **simple, reproducible, and suitable for academic projects**.

---

## 1. What this project does

* **Input**: Vietnamese teencode / informal chat text  
* **Output**: Clean, normalized Vietnamese  

Example:

```text
Input : minh2 di hc ve muon qa :(
Output: mình đi học về muộn quá :(
```

Typical use cases:

* Text normalization before **summarization**
* Preprocessing for **chat / social media NLP**

---

## 2. Model & Dataset

### Model
* **vinai/bartpho-base** (~139M parameters)
* Encoder–decoder Transformer (Seq2Seq)

### Dataset
* **ViLexNorm** (Vietnamese lexical normalization)
* Loaded from a **data-only mirror**:
  ```text
  visolex/vilexnorm
  ```

The dataset is converted into JSONL format for training.

---

## 3. Hardware requirements

Recommended minimum:

* **GPU**: NVIDIA RTX 3050 Laptop (4–6 GB VRAM)
* **RAM**: 16 GB
* **Disk**: ~2 GB free

> CPU-only training is possible but very slow.

---

## 4. Setup & Training (run in order)

### Step 1: Create virtual environment
py -3.10 -m venv venv
# 1. Kích hoạt
venv\Scripts\activate

# 2. Cập nhật pip (quan trọng)
python -m pip install --upgrade pip

# 3. Cài lại thư viện (Lúc này sẽ tìm thấy Torch GPU)
pip install -r requirements.txt

### Step 2: Install dependencies

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

---

### Step 3: Login to Hugging Face (optional, for upload)

1. Create an account at https://huggingface.co  
2. Go to **Settings → Access Tokens**
3. Create a token with **Write** permission

Login locally:

```bash
huggingface-cli login
```

---

### Step 4: Prepare the dataset

```bash
python prepare_data.py
```

This will:

* Download ViLexNorm
* Split data into **train / validation (90% / 10%)**
* Create:

```text
data/
 ├── train.jsonl
 └── valid.jsonl
```

Each line format:

```json
{"input": "teencode text", "output": "normalized text"}
```

---

### Step 5: Train the model

```bash
python train.py
```

Default training setup:

* Batch size (per device): 2  
* Gradient accumulation: 8 (effective batch = 16)  
* FP16 enabled  
* Max sequence length: 128  
* Epochs: 3  

Checkpoints are saved to:

```text
outputs/
```

Expected training time (RTX 3050 Laptop): **~20–30 minutes**

---

## 5. Inference (test the model)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "your-hf-username/bartpho-teencode-vilexnorm"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "minh2 di hc ve muon qa"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Expected output:

```text
mình đi học về muộn quá
```

---

## 6. Uploading to Hugging Face (optional)

After training:

```python
model.push_to_hub("bartpho-vilexnorm")
tokenizer.push_to_hub("bartpho-vilexnorm")
```

Your model will appear at:

```text
https://huggingface.co/<your-username>/bartpho-vilexnorm
```

---

## 7. Deployment notes

⚠️ **Not suitable for free-tier web hosting**

* Model size (FP16): ~300 MB  
* Runtime RAM required: >2 GB  

Recommended:

* Hugging Face Inference API
* Hugging Face Spaces
* VPS with ≥4 GB RAM

---

## 8. Project structure

```text
.
├── prepare_data.py
├── train.py
├── data/
│   ├── train.jsonl
│   └── valid.jsonl
├── outputs/
└── README.md
```

---

## 9. License & usage

* Dataset: ViLexNorm (research / educational use)
* Base model: BARTpho (VinAI)
* Fine-tuned weights: **learning & research purposes**

Check original licenses before commercial use.

---

**Purpose**: Academic / learning project  
**Author**: Your Name
