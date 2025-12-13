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
from bertLINEAR import BertLinear
import datetime

import time

# ========================== 📋 自定义日志回调 ==========================
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
                # 估算 samples per second (batch_size * steps_per_sec * num_gpus)
                # 注意：args.per_device_train_batch_size 是单卡 batch size
                # world_size 可以通过 args.world_size 获取 (如果 Trainer 注入了) 或者手动计算
                # 这里简单打印 ms/step
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


# ========================== 🛠️ 核心配置区 ==========================
# 1. 调试模式开关
DEBUG_MODE = False 

# 2. 显卡指定
if DEBUG_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    # ⚠️ 0,1,2,3,5 号卡空闲
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,5" 

# 3. 路径配置
DATA_DIR = "ft_data_stratified" # 🆕 使用分层抽样数据
PRETRAINED_MODEL_PATH = "bert_small_4096_final"

if DEBUG_MODE:
    OUTPUT_DIR = "output_ft_linear_debug"
else:
    OUTPUT_DIR = "output_ft_linear"

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
    # 🚀 激进优化：Batch Size 32
    # Linear 模型参数少，显存占用低，32 应该很安全
    BATCH_SIZE = 32 
    REPORT_TO = "tensorboard" # 🆕 启用 TensorBoard 可视化
    
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
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.ERROR)

    # 1. 加载数据
    if is_main_process:
        logger.info(f"Loading data from {DATA_DIR}...")
    
    dataset = load_from_disk(DATA_DIR)
    
    # 2. 加载 Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_PATH)

    # 3. 初始化模型
    if is_main_process:
        logger.info("Initializing BertLinear model...")
    
    model = BertLinear(
        bert_model_path=PRETRAINED_MODEL_PATH,
        num_labels=NUM_LABELS
    )

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        
        # 优化器与调度
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        # 评估与保存策略
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        
        # 混合精度与加速
        fp16=False,
        bf16=True, # A40 支持 BF16
        dataloader_num_workers=8, # 提高数据加载速度
        
        # DDP 配置
        ddp_find_unused_parameters=False, # 关键：设为 False 提高速度，因为我们已经移除了 Pooler
        
        # 日志
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to=REPORT_TO,
        
        # 禁用 torch.compile (动态 padding 导致挂起)
        torch_compile=False 
    )

    # 5. Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            CustomLogCallback() # 🆕 添加自定义日志回调
        ]
    )

    # 7. 开始训练
    if is_main_process:
        logger.info("Starting training...")
    
    trainer.train()

    # 8. 保存最终模型
    if is_main_process:
        logger.info(f"Saving final model to {OUTPUT_DIR}/final_model")
        trainer.save_model(f"{OUTPUT_DIR}/final_model")
        # 保存自定义模型权重
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_model/pytorch_model.bin")

if __name__ == "__main__":
    main()
