# train_ner.py
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig
from bert_bilstm_crf import BertBiLSTMCRF
from dataset_ner import NERDataset, collate_fn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

# ----------------- 配置区（需修改） -----------------
PRETRAINED_DIR = "output_small_4096"   # 根据实际情况修改（预训练结果文件夹）
TRAIN_PATH = "train_set.csv"
DEV_PATH = "test_a.csv"
OUTPUT_DIR = "bert_bilstm_crf_ckpt"
MAX_LEN = 512
BATCH_SIZE = 8
NUM_EPOCHS = 5
LR = 3e-5
NUM_LABELS = 14
LSTM_HIDDEN = 256
LSTM_LAYERS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN_ID = 0
PAD_LABEL_ID = 0
# ------------------------------------------------------

def load_config_or_default(pretrained_dir=None):
    if pretrained_dir is not None and os.path.isdir(pretrained_dir):
        try:
            cfg = BertConfig.from_pretrained(pretrained_dir)
            print("Loaded config from", pretrained_dir)
            return cfg
        except Exception as e:
            print("warning: cannot load config from dir:", e)

    # fallback: create config consistent with your pretrain.py defaults
    print("Using fallback BertConfig (small). Adjust if needed.")
    cfg = BertConfig(
        vocab_size=10000,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=MAX_LEN
    )
    return cfg

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            preds = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            # preds is list of list; convert to padded array aligning with labels
            for i, seq in enumerate(preds):
                all_preds.extend(seq)
            # flatten true labels (we will compute metrics ignoring pad)
            lbls = batch['labels'].cpu().numpy().tolist()
            for seq in lbls:
                all_labels.extend([l for l in seq]) 

    # Note: above flattening is simplistic. For token-level metrics, 
    # you should filter out PAD_LABEL_ID and align lengths.
    # Here we compute macro F1 on token ids ignoring PAD_LABEL_ID.
    filtered_preds = []
    filtered_labels = []
    # Need to reconstruct preds aligned to labels - simpler approach:
    # re-run per batch to align with mask
    model.eval()
    filtered_preds = []
    filtered_labels = []
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            # out is list of predicted sequences (list of lists)
            labels = batch['labels'].cpu().tolist()
            mask = batch['attention_mask'].cpu().tolist()
            for pred_seq, lab_seq, m in zip(out, labels, mask):
                for p, l, mm in zip(pred_seq, lab_seq, m):
                    if mm == 1:  # not pad
                        filtered_preds.append(p)
                        filtered_labels.append(l)

    p, r, f1, _ = precision_recall_fscore_support(filtered_labels, filtered_preds, average='macro', zero_division=0)
    return p, r, f1

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = load_config_or_default(PRETRAINED_DIR)
    model = BertBiLSTMCRF(
        pretrained_dir=PRETRAINED_DIR if os.path.isdir(PRETRAINED_DIR) else None,
        config=cfg,
        num_labels=NUM_LABELS,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS
    )
    model.to(DEVICE)

    train_dataset = NERDataset(TRAIN_PATH, max_len=MAX_LEN)
    dev_dataset = NERDataset(DEV_PATH, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id=PAD_TOKEN_ID, pad_label_id=PAD_LABEL_ID))
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id=PAD_TOKEN_ID, pad_label_id=PAD_LABEL_ID))

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0.0
        step = 0
        for batch in loop:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            loss = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()

            total_loss += loss.item()
            step += 1
            if step % 10 == 0:
                loop.set_postfix(loss=total_loss / step)

        # eval
        p, r, f1 = evaluate(model, dev_loader)
        print(f"\nEpoch {epoch} Eval -> P:{p:.4f} R:{r:.4f} F1:{f1:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(OUTPUT_DIR, f"epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved", ckpt_path)

    print("Training finished.")

if __name__ == "__main__":
    main()
