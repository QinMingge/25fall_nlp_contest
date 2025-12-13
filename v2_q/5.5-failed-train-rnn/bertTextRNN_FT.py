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
    DataCollatorWithPadding,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, f1_score
from bertTextRNN import BertTextRNN
import datetime
import time

# ========================== � 自定义日志回调 ==========================
class CustomLogCallback(TrainerCallback):
    """
    自定义日志回调，用于输出符合要求的格式：
    Time - INFO - Epoch: X, Step: Y, Train Loss: Z, LR: ..., Speed: ...
    """
    def __init__(self):
        self.last_time = time.time()
        self.last_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            # 计算速度
            current_time = time.time()
            time_delta = current_time - self.last_time
            step_delta = state.global_step - self.last_step
            
            # 避免除以零
            if step_delta > 0 and time_delta > 0:
                steps_per_sec = step_delta / time_delta
                ms_per_step = (time_delta / step_delta) * 1000
                speed_info = f"{ms_per_step:.2f}ms/step"
            else:
                speed_info = "N/A"
            
            # 更新状态
            self.last_time = current_time
            self.last_step = state.global_step

            # 获取当前时间
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 构建日志信息 parts
            msg_parts = []
            
            # Epoch
            if 'epoch' in logs:
                msg_parts.append(f"Epoch: {logs['epoch']:.2f}")
            
            # Step
            msg_parts.append(f"Step: {state.global_step}")
            
            # Train Loss
            if 'loss' in logs:
                msg_parts.append(f"Train Loss: {logs['loss']:.4f}")
            
            # Learning Rate
            if 'learning_rate' in logs:
                msg_parts.append(f"LR: {logs['learning_rate']:.2e}")

            # Batch Size (单卡)
            msg_parts.append(f"Batch: {args.per_device_train_batch_size}")

            # Speed
            msg_parts.append(f"Speed: {speed_info}")

            # Eval Metrics (如果有)
            if 'eval_loss' in logs:
                msg_parts.append(f"Eval Loss: {logs['eval_loss']:.4f}")
            if 'eval_accuracy' in logs:
                msg_parts.append(f"Accuracy: {logs['eval_accuracy']:.4f}")
            if 'eval_f1_macro' in logs:
                msg_parts.append(f"F1: {logs['eval_f1_macro']:.4f}")
                
            # 组合消息
            log_msg = ", ".join(msg_parts)
            
            # 获取 logger
            logger = logging.getLogger(__name__)
            logger.info(log_msg)

# ========================== �🛠️ 核心配置区 ==========================
# 1. 调试模式开关
DEBUG_MODE = False 

# 2. 显卡指定
if DEBUG_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    # ⚠️ 请根据实际空闲显卡修改此处
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,5" 

# 3. 路径配置
DATA_DIR = "ft_data_stratified" # 🆕 统一使用分层切分的数据
PRETRAINED_MODEL_PATH = "bert_small_4096_final"

if DEBUG_MODE:
    OUTPUT_DIR = "output_ft_rnn_debug"
else:
    OUTPUT_DIR = "output_ft_rnn"

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
    # 🚀 优化：RNN 比 CNN 更吃显存 (BPTT)，所以 Batch Size 不能像 CNN 那么大 (32)。
    # A40 (48GB) 跑 4096 LSTM，建议尝试 8-16。
    # 如果 OOM，请减小此值。
    BATCH_SIZE = 8
    REPORT_TO = "tensorboard"
    
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
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.WARN
        )

    if is_main_process:
        print(f"\n{'='*40}")
        print(f"🚀 Starting Fine-tuning (RNN Stratified)...")
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
        print(f"Initializing BertTextRNN from {PRETRAINED_MODEL_PATH}...")
    
    model = BertTextRNN(
        bert_model_path=PRETRAINED_MODEL_PATH, 
        num_labels=NUM_LABELS,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        bidirectional=True
    )
    
    # 3. 训练参数
    # 强制禁用 torch.compile (避免动态 Padding 问题)
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
        
        bf16=False, 
        # 🚀 RNN 必须开启 Gradient Checkpointing 才能跑长序列
        gradient_checkpointing=True, 
        dataloader_num_workers=8, # 利用多核 CPU
        dataloader_pin_memory=True,
        
        logging_steps=50,
        report_to=REPORT_TO,
        
        ddp_find_unused_parameters=False
    )

    # 4. Data Collator (动态填充)
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_PATH)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomLogCallback()],
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
        print(f"✅ Done! Model saved to {final_path}")

if __name__ == "__main__":
    main()
