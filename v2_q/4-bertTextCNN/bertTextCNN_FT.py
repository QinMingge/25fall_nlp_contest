import os
import sys
import logging
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    Trainer, 
    TrainingArguments, 
    BertTokenizerFast,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
from bertTextCNN import BertTextCNN

# ========================== 🛠️ 核心配置区 ==========================
# 1. 调试模式开关
DEBUG_MODE = False 

# 2. 显卡指定
if DEBUG_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    # ⚠️ 请根据实际空闲显卡修改此处
    # 例如：如果 1,2,3 号卡空闲，则填 "1,2,3"
    # 并在运行 torchrun 时指定 --nproc_per_node=3
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,5" 
    # pass # 建议在命令行通过 CUDA_VISIBLE_DEVICES 控制，不要在代码里写死，防止冲突

# 3. 路径配置
DATA_DIR = "ft_data_stratified" # 🆕 更新为新的数据目录
PRETRAINED_MODEL_PATH = "bert_small_4096_final"

if DEBUG_MODE:
    OUTPUT_DIR = "output_ft_cnn_debug"
else:
    OUTPUT_DIR = "output_ft_cnn"

# 4. 训练超参
SEQ_LEN = 4096
NUM_LABELS = 14

if DEBUG_MODE:
    NUM_EPOCHS = 1
    BATCH_SIZE = 4
    REPORT_TO = "none"
    SAVE_STRATEGY = "steps"
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = 10
    SAVE_STEPS = 10
else:
    NUM_EPOCHS = 10
    # 🚀 激进优化：Batch Size 16 -> 32
    # 显存只用了一半 (20GB/46GB)，直接翻倍填满显卡！
    BATCH_SIZE = 32 
    REPORT_TO = "none"
    
    SAVE_STRATEGY = "steps"
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = 500
    SAVE_STEPS = 500

# ====================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': acc,
        'f1_macro': f1
    }

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in [-1, 0]

    # 0. 设置日志
    # 只有主进程写日志文件，避免多进程写入冲突
    if is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log_file = os.path.join(OUTPUT_DIR, "training_detailed.log")
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode="w")
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging to {log_file}")
    else:
        # 其他进程只输出错误信息，不写文件
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.WARN
        )

    if is_main_process:
        print(f"\n{'='*40}")
        print(f"🚀 Starting Fine-tuning (Stratified Split)...")
        print(f"🛠️  Mode: {'[DEBUG]' if DEBUG_MODE else '[FULL TRAINING]'}")
        print(f"{'='*40}\n")
    
    # 1. 加载数据
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory {DATA_DIR} not found. Run FT_data.py first.")
        
    if is_main_process:
        print(f"Loading data from {DATA_DIR}...")
    
    dataset = load_from_disk(DATA_DIR)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    if DEBUG_MODE:
        if is_main_process:
            print("⚠️ Debug mode: using small subset...")
        train_dataset = train_dataset.select(range(100))
        eval_dataset = eval_dataset.select(range(100))

    if is_main_process:
        print(f"Train size: {len(train_dataset)}")
        print(f"Eval size: {len(eval_dataset)}")

    # 2. 初始化模型
    if is_main_process:
        print(f"Initializing BertTextCNN from {PRETRAINED_MODEL_PATH}...")
    
    model = BertTextCNN(
        bert_model_path=PRETRAINED_MODEL_PATH, 
        num_labels=NUM_LABELS,
        filter_sizes=(2, 3, 4, 5),
        num_filters=256 
    )
    
    # 3. 训练参数
    # 强制禁用 torch.compile
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        eval_strategy=EVAL_STRATEGY, 
        eval_steps=EVAL_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        
        bf16=True, 
        # torch_compile=False, # 🔄 彻底移除该参数，防止 transformers 误判
        gradient_checkpointing=False, 
        dataloader_num_workers=8, # 🚀 增加到 8：利用 96 核 CPU 加速数据加载
        dataloader_pin_memory=True, # 🔄 恢复为 True：加速 CPU 到 GPU 的数据传输
        
        logging_steps=50,
        report_to=REPORT_TO,
        
        ddp_find_unused_parameters=False # 🔄 关闭：我们已经修改了模型，不再有 unused parameters
    )

    # 4. Data Collator (动态填充)
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_PATH)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # 5. Trainer
    # 再次确保 model 没有被编译
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
        
    print(f"🔍 TrainingArguments.torch_compile = {training_args.torch_compile}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=data_collator
    )

    # 6. 开始训练
    if is_main_process:
        print("🔥 Starting Training...")
    trainer.train()
    
    # 7. 保存最终模型
    final_path = f"{OUTPUT_DIR}/final_model"
    trainer.save_model(final_path)
    if is_main_process:
        print(f"✅ Training done! Model saved to {final_path}")

if __name__ == "__main__":
    main()
