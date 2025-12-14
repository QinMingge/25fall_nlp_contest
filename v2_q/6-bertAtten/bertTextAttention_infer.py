import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast
from safetensors.torch import load_file
from bertTextAttention import BertTextAttention

# ================= 配置 =================
# 路径配置
BASE_DIR = "/data/jinda/qinmingge/BERT/25fallnewsclassify"
TEST_FILE = os.path.join(BASE_DIR, "test_a.csv")
SUBMIT_FILE = os.path.join(BASE_DIR, "submit_attention.csv")

# 模型路径
# 1. 原始预训练 BERT 路径 (用于初始化模型结构)
PRETRAINED_BERT_PATH = os.path.join(BASE_DIR, "bert_small_4096_final")
# 2. 微调后的权重路径 (用于加载训练好的参数)
# 优先尝试 safetensors，如果不存在则尝试 bin
FT_MODEL_PATH_SAFETENSORS = os.path.join(BASE_DIR, "output_ft_attention/final_model/model.safetensors")
FT_MODEL_PATH_BIN = os.path.join(BASE_DIR, "output_ft_attention/final_model/pytorch_model.bin")

# 超参数
# 🚀 多卡并行配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,5,7" # 指定空闲显卡 (GPU 2 有占用，排除)
# Attention 模型比 CNN 稍重，Batch Size 设为 32 * 4 = 128
BATCH_SIZE = 32 * 4 
SEQ_LEN = 4096
NUM_LABELS = 14
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 数据集定义 =================
class TestDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, sep='\t' if csv_file.endswith('.tsv') else ',')
        # 检查列名，如果是 'text' 则使用，否则假设第一列是文本
        if 'text' in self.df.columns:
            self.texts = self.df['text'].tolist()
        else:
            print(f"⚠️ Warning: 'text' column not found in {csv_file}. Using first column.")
            self.texts = self.df.iloc[:, 0].tolist()
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {"text": str(self.texts[idx])} # 确保是字符串

# ================= 主函数 =================
def main():
    print(f"🚀 Starting Inference on {DEVICE}...")
    
    # 1. 加载 Tokenizer
    print(f"Loading tokenizer from {PRETRAINED_BERT_PATH}...")
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_BERT_PATH)
    
    # 2. 准备数据
    print(f"Loading test data from {TEST_FILE}...")
    test_dataset = TestDataset(TEST_FILE)
    print(f"Test set size: {len(test_dataset)}")
    
    # 自定义 collate_fn 处理 Tokenization
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        # 动态 Padding 到当前 Batch 最长
        encoding = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=SEQ_LEN, 
            return_tensors="pt"
        )
        return encoding

    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=16, # 🚀 增加 CPU 进程数以匹配 5 张 GPU 的吞吐
        pin_memory=True
    )
    
    # 3. 初始化模型结构
    print(f"Initializing BertTextAttention model structure...")
    model = BertTextAttention(
        bert_model_path=PRETRAINED_BERT_PATH, 
        num_labels=NUM_LABELS
    )
    
    # 4. 加载微调后的权重
    if os.path.exists(FT_MODEL_PATH_SAFETENSORS):
        print(f"Loading weights from {FT_MODEL_PATH_SAFETENSORS}...")
        state_dict = load_file(FT_MODEL_PATH_SAFETENSORS)
    elif os.path.exists(FT_MODEL_PATH_BIN):
        print(f"Loading weights from {FT_MODEL_PATH_BIN}...")
        state_dict = torch.load(FT_MODEL_PATH_BIN, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found at {FT_MODEL_PATH_SAFETENSORS} or {FT_MODEL_PATH_BIN}")

    # 处理可能的 key 不匹配问题 (例如 DDP 训练可能导致 module. 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    
    # 5. 多卡并行 (DataParallel)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model.to(DEVICE)
    model.eval()
    
    # 6. 推理循环
    all_preds = []
    
    print("Starting prediction loop...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            # 混合精度推理 (加速)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
            
            # 保存 logits
            all_preds.append(logits.cpu().numpy())
            
    # 7. 保存结果
    print("Concatenating logits...")
    all_logits = np.concatenate(all_preds, axis=0)
    
    print(f"Saving logits to {BASE_DIR}/submit_attention_logits.npy ...")
    np.save(os.path.join(BASE_DIR, "submit_attention_logits.npy"), all_logits)
    
    preds = np.argmax(all_logits, axis=1)
    print(f"Saving results to {SUBMIT_FILE}...")
    df_submit = pd.DataFrame({"label": preds})
    df_submit.to_csv(SUBMIT_FILE, index=False)
    
    print("✅ Done!")

if __name__ == "__main__":
    main()
