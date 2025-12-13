import os
import torch
from datasets import load_dataset
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    BertTokenizerFast, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)

# ========================== 🛠️ 核心配置区 ==========================
# 1. 调试模式开关 (True = 跑通流程 / False = 全力开火)
DEBUG_MODE = False  # <--- ⚠️ 跑正式训练前改成 False

# 2. 显卡指定 (只使用空闲的 0,1,2,3)
# 这一行会让程序只看得到这4张卡，编号会自动重排为 0-3
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,6,7"

# 3. 模型与数据配置
MODEL_SIZE = 'small'   # 'mini' 或 'small'
SEQ_LEN = 4096        # 必须与 pretrain_data.csv 的切分长度一致
VOCAB_SIZE = 10000

# 4. 训练超参 (根据显卡自动调整)
if DEBUG_MODE:
    NUM_EPOCHS = 1             # 调试只跑 1 轮
    MAX_STEPS = 50             # 或者只跑 50 步
    BATCH_SIZE = 8             # 小一点防报错
    SAVE_STRATEGY = "no"       # 调试不保存中间结果
    REPORT_TO = "none"         # 不上传 wandb
else:
    NUM_EPOCHS = 20 if MODEL_SIZE == 'mini' else 40
    MAX_STEPS = -1             # 跑完所有 Epoch
    # A40 显存很大，Mini 可以开到 32，Small 可以开到 12-16
    BATCH_SIZE = 32 if MODEL_SIZE == 'mini' else 16
    SAVE_STRATEGY = "epoch"    # 每轮保存
    REPORT_TO = "none"         # 如果有 wandb账号可改成 "wandb"

# ====================================================================

def get_model_config():
    common_config = {
        "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": SEQ_LEN,
        "type_vocab_size": 2,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    }
    
    if MODEL_SIZE == 'mini':
        return BertConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            **common_config
        )
    elif MODEL_SIZE == 'small':
        return BertConfig(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            **common_config
        )
    else:
        raise ValueError("Unknown MODEL_SIZE")

def main():
    # 1. 检查环境
    n_gpu = torch.cuda.device_count()
    print(f"🚀 Detected {n_gpu} GPUs available for training.")
    print(f"🛠️  Mode: {'[DEBUG]' if DEBUG_MODE else '[FULL TRAINING]'}")
    
    # 2. Tokenizer
    tokenizer = BertTokenizerFast(vocab_file="vocab.txt", do_lower_case=False)
    
    # 3. 加载数据
    data_file = "pretrain_data.csv"
    if DEBUG_MODE:
        print("⚠️ Loading only first 1000 lines for debugging...")
        # split='train[:1000]' 这是一个非常方便的切片写法
        dataset = load_dataset("csv", data_files=data_file, split="train[:1000]")
    else:
        print(f"Loading full data from {data_file}...")
        dataset = load_dataset("csv", data_files=data_file, split="train")
    
    # 4. Tokenization
    def encode(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=SEQ_LEN, 
            padding="max_length"
        )

    # 调试模式用单进程，正式模式用多进程加速
    num_proc = 1 if DEBUG_MODE else 16
    print(f"Tokenizing data (num_proc={num_proc})...")
    tokenized_dataset = dataset.map(encode, batched=True, num_proc=num_proc, remove_columns=["text"])

    # 5. 模型初始化
    config = get_model_config()
    model = BertForMaskedLM(config)
    
    # 6. 训练参数
    output_dir = f"./output_{MODEL_SIZE}_{SEQ_LEN}_debug" if DEBUG_MODE else f"./output_{MODEL_SIZE}_{SEQ_LEN}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS, # 调试时生效
        per_device_train_batch_size=BATCH_SIZE,
        
        # A40 专属加速
        bf16=True, 
        gradient_checkpointing=False, # 显存够大先不开，开了省显存但慢
        
        # DDP 配置
        ddp_find_unused_parameters=False,
        
        # 日志与保存
        save_strategy=SAVE_STRATEGY,
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        dataloader_num_workers=4,
        report_to=REPORT_TO
    )

    # 7. Trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    # 8. Run
    print("🔥 Starting Training...")
    trainer.train()
    
    # 9. Save (仅正式模式或调试跑完后)
    final_path = f"./bert_{MODEL_SIZE}_{SEQ_LEN}_final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ All Done! Model saved to {final_path}")

if __name__ == "__main__":
    main()