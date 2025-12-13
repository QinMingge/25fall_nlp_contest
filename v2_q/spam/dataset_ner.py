# dataset_ner.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class NERDataset(Dataset):
    """
    CSV/TSV should contain columns: 'text' and 'labels'
    text: token ids separated by space, e.g. "101 23 45 900 12"
    labels: tag ids separated by space, e.g. "0 0 3 3 0"
    """
    def __init__(self, path, max_len=512, sep='\t'):
        self.df = pd.read_csv(path, sep=sep)
        self.max_len = max_len
        # Basic checks
        if 'text' not in self.df.columns or 'labels' not in self.df.columns:
            raise ValueError("Input file must contain 'text' and 'labels' columns")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text']).strip()
        labels = str(row['labels']).strip()

        ids = [int(x) for x in text.split() if x != ""]
        lab = [int(x) for x in labels.split() if x != ""]

        # truncate
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            lab = lab[:self.max_len]

        attention_mask = [1] * len(ids)

        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long)
        }
        return item

def collate_fn(batch, pad_id=0, pad_label_id=0):
    """
    batch: list of items from NERDataset.__getitem__
    returns padded tensors
    """
    import torch.nn.utils.rnn as rnn_utils

    max_len = max([item['input_ids'].size(0) for item in batch])

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        l = item['input_ids'].size(0)
        pad_len = max_len - l
        input_ids.append(torch.cat([item['input_ids'], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([item['attention_mask'], torch.zeros((pad_len,), dtype=torch.long)]))
        labels.append(torch.cat([item['labels'], torch.full((pad_len,), pad_label_id, dtype=torch.long)]))

    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    labels = torch.stack(labels, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
