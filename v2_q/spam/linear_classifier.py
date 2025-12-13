import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import (
    BertModel,
    BertConfig,
    TrainingArguments,
    Trainer
)

# =====================================================
#                 1. 数据集定义
# =====================================================
class NewsDataset(Dataset):
    def __init__(self, df, max_len=512):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = [int(x) for x in self.texts[idx].split()]
        # 截断
        ids = ids[:self.max_len]
        attention = [1] * len(ids)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =====================================================
#            2. 分类模型 (BERT + Linear Head)
# =====================================================
class BertForClassification(nn.Module):
    def __init__(self, pretrained_dir, num_labels=14):
        super().__init__()

        # 加载预训练的 BERT
        print(f"Loading pretrained BERT from: {pretrained_dir}")
        config = BertConfig.from_pretrained(pretrained_dir)
        self.bert = BertModel.from_pretrained(pretrained_dir, config=config)

        hidden = config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # 取 [CLS]

        logits = self.classifier(cls)

        # 训练阶段返回 loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        else:
            return logits


# =====================================================
#                3. 主函数，训练分类器
# =====================================================
def main():

    # 预训练好的模型目录（包含 config.json、pytorch_model.bin）
    PRETRAINED_DIR = "pretrained_model"  # <<< 必须确认路径正确

    TRAIN_PATH = "train_set.csv"    # 含 text 和 label
    DEV_PATH = "dev_set.csv"        # 如果没有可用 train 替代

    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH, sep="\t")
    dev_df = pd.read_csv(DEV_PATH, sep="\t")

    train_dataset = NewsDataset(train_df, max_len=512)
    dev_dataset = NewsDataset(dev_df, max_len=512)

    model = BertForClassification(pretrained_dir=PRETRAINED_DIR, num_labels=14)

    training_args = TrainingArguments(
        output_dir="./cls_output",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    trainer.train()

    print("\n linear-Fine-tuning done! Model saved to ./cls_output")


if __name__ == "__main__":
    main()
