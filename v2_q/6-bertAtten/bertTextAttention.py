import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention Mechanism:
    Computes a weighted sum of the input sequence based on a learned query vector.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)

        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))

        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: (batch_size, seq_len, hidden_size)
        # batch_masks:  (batch_size, seq_len)

        # 1. Linear transformation: key = W * h + b
        # (batch_size, seq_len, hidden_size)
        key = torch.matmul(batch_hidden, self.weight) + self.bias

        # 2. Compute attention scores: scores = key * query
        # (batch_size, seq_len)
        outputs = torch.matmul(key, self.query)

        # 3. Mask padding
        # Fill padding positions with a very small number (-1e4) so softmax becomes 0
        # Note: -1e32 causes overflow in FP16, -1e4 is safe and sufficient for softmax
        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e4))

        # 4. Softmax to get probabilities
        attn_scores = F.softmax(masked_outputs, dim=1)

        # 5. Weighted sum
        # (batch_size, 1, seq_len) x (batch_size, seq_len, hidden_size) -> (batch_size, 1, hidden_size)
        batch_outputs = torch.bmm(attn_scores.unsqueeze(1), batch_hidden).squeeze(1)

        return batch_outputs, attn_scores

import numpy as np

class BertTextAttention(nn.Module):
    def __init__(self, bert_model_path, num_labels=14, dropout=0.1):
        super(BertTextAttention, self).__init__()
        
        # 1. Load BERT
        self.bert = BertModel.from_pretrained(bert_model_path, add_pooling_layer=False)
        
        embedding_dim = self.bert.config.hidden_size 
        print(f"✅ BERT Embedding Dimension Detected: {embedding_dim}")

        # 2. Attention Layer
        self.attention = Attention(embedding_dim)

        # 3. Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_labels)
        self.num_labels = num_labels

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # 1. BERT Encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # 2. Attention Pooling
        # Use attention_mask to ignore padding tokens
        attn_output, attn_scores = self.attention(last_hidden_state, attention_mask)

        # 3. Classification
        logits = self.fc(self.dropout(attn_output))
        
        # 4. Loss Calculation
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}
            
        return {"logits": logits}
