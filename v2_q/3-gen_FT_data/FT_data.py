import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast
from tqdm import tqdm

# ================= 配置 =================
SEQ_LEN = 4096
VOCAB_FILE = "vocab.txt"
TRAIN_FILE = "train_set.csv"
TEST_FILE = "test_a.csv"
OUTPUT_DIR = "ft_data_stratified" # 更改输出目录名，避免混淆
VAL_SIZE = 0.2 # 验证集比例 20%
SEED = 42

# 智能切割锚点 (与 pretrain_data.py 一致)
STOP_TOKENS = ['900', '2662'] 
BACKOFF_WINDOW = 200 
# =======================================

def smart_split_text(text, max_len):
    """
    复制自 pretrain_data.py 的切分逻辑
    """
    if not isinstance(text, str): 
        return []
        
    words = text.split()
    total_len = len(words)
    
    # Case 0: 如果本身就短于 max_len
    if total_len <= max_len:
        return [' '.join(words)]
    
    result = []
    start = 0

    while start < total_len:
        remaining_len = total_len - start
        
        # === 处理最后一段 ===
        if remaining_len < max_len:
            ideal_start = total_len - max_len
            search_start = ideal_start
            search_end = min(total_len, ideal_start + BACKOFF_WINDOW)
            
            smart_start = -1
            for i in range(search_start, search_end):
                if words[i] in STOP_TOKENS:
                    smart_start = i + 1 
                    break
            
            if smart_start != -1:
                last_segment = words[smart_start:]
            else:
                last_segment = words[-max_len:]
                
            result.append(' '.join(last_segment))
            break
            
        # === 中间段落 ===
        end = start + max_len
        chunk = words[start:end]
        
        search_start = len(chunk) - 1
        search_end = max(-1, len(chunk) - 1 - BACKOFF_WINDOW)
        
        split_idx = -1
        for i in range(search_start, search_end, -1):
            if chunk[i] in STOP_TOKENS:
                split_idx = i
                break 
        
        if split_idx != -1:
            real_end = start + split_idx + 1
        else:
            real_end = end
            
        result.append(' '.join(words[start:real_end]))
        start = real_end
        
    return result

def expand_dataframe(df, desc_text="Processing"):
    """
    将 DataFrame 中的长文本进行切分，并保留原始索引
    """
    expanded_rows = []
    # 使用 tqdm 显示进度
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc_text):
        text = row['text']
        
        # 修正：预留 2 个位置给 [CLS] 和 [SEP]
        # 原始文本是空格分隔的数字，1个数字对应1个Token。
        # 为了防止 Tokenizer 添加特殊符号后超过 4096 导致截断（丢失末尾信息），
        # 我们在切分时严格限制内容长度为 4096 - 2 = 4094。
        segments = smart_split_text(text, SEQ_LEN - 2)
        
        # 获取除 text 外的其他列信息
        base_info = row.to_dict()
        del base_info['text']
        
        for seg_i, seg in enumerate(segments):
            new_row = base_info.copy()
            new_row['text'] = seg
            new_row['original_index'] = idx # 保留原始索引，方便后续分析
            new_row['segment_id'] = seg_i
            expanded_rows.append(new_row)
            
    return pd.DataFrame(expanded_rows)

def process_data():
    # 1. 加载 Tokenizer
    if not os.path.exists(VOCAB_FILE):
        raise FileNotFoundError(f"{VOCAB_FILE} not found.")
    print(f"Loading tokenizer from {VOCAB_FILE}...")
    tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE, do_lower_case=False)

    # 2. 读取数据
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"{TRAIN_FILE} not found.")
    print(f"Reading {TRAIN_FILE}...")
    df = pd.read_csv(TRAIN_FILE, sep='\t')
    
    # 3. 分层切分 (Stratified Split)
    # 替代原来的 KFold，直接切分出训练集和验证集
    print(f"Splitting data into Train and Validation (Val size={VAL_SIZE})...")
    train_df, val_df = train_test_split(
        df, 
        test_size=VAL_SIZE, 
        stratify=df['label'], 
        random_state=SEED
    )
    print(f"Original Train samples: {len(train_df)}")
    print(f"Original Validation samples: {len(val_df)}")

    # 4. 应用智能切分 (Data Expansion)
    # 先切分数据集，再进行长文本扩展，防止同一条数据的不同片段泄露到验证集
    print("Applying smart text splitting...")
    expanded_train_df = expand_dataframe(train_df, "Splitting Train")
    expanded_val_df = expand_dataframe(val_df, "Splitting Validation")
    
    print(f"Expanded Train samples: {len(expanded_train_df)}")
    print(f"Expanded Validation samples: {len(expanded_val_df)}")

    # 5. 处理 Test 数据
    test_dataset = None
    if os.path.exists(TEST_FILE):
        print(f"Reading {TEST_FILE}...")
        test_df = pd.read_csv(TEST_FILE, sep='\t')
        expanded_test_df = expand_dataframe(test_df, "Splitting Test")
        print(f"Expanded Test samples: {len(expanded_test_df)}")
        test_dataset = Dataset.from_pandas(expanded_test_df)

    # 6. 转换为 Dataset
    train_dataset = Dataset.from_pandas(expanded_train_df)
    val_dataset = Dataset.from_pandas(expanded_val_df)

    # 7. Tokenization
    def preprocess_function(examples):
        # Tokenize inputs
        # truncation=True, padding=False, max_length=SEQ_LEN
        # 这里的处理逻辑与预训练保持一致：
        # 1. Tokenizer 会自动添加 [CLS] 和 [SEP]
        # 2. 超过 SEQ_LEN 的会被截断 (虽然我们已经做了 smart split，但加上特殊符号后可能微超)
        # 3. padding=False: 启用动态填充 (Dynamic Padding)，不在此处补 0
        model_inputs = tokenizer(
            examples["text"], 
            max_length=SEQ_LEN, 
            padding=False, # ⚠️ 动态填充关键点
            truncation=True
        )
        
        # 处理 Label (如果有)
        if "label" in examples:
            model_inputs["labels"] = examples["label"]
            
        return model_inputs

    # num_proc 可以根据 CPU 核心数调整
    num_proc = 8
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=["text"])
    tokenized_val = val_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=["text"])
    
    datasets_dict = {
        "train": tokenized_train,
        "validation": tokenized_val
    }
    
    if test_dataset:
        tokenized_test = test_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=["text"])
        datasets_dict["test"] = tokenized_test

    final_datasets = DatasetDict(datasets_dict)

    # 8. 保存
    print(f"Saving datasets to {OUTPUT_DIR}...")
    final_datasets.save_to_disk(OUTPUT_DIR)
    print(f"✅ Done! Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_data()
