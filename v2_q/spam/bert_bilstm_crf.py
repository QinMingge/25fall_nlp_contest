# bert_bilstm_crf.py
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertConfig, AutoConfig, AutoModel

class BertBiLSTMCRF(nn.Module):
    def __init__(
        self,
        pretrained_dir=None,
        config=None,
        num_labels=10,
        lstm_hidden=256,
        lstm_layers=1,
        dropout=0.1,
        use_transformers_auto=False
    ):
        super().__init__()

        # 优先从 pretrained_dir 加载 config + model；否则使用传入的 config 初始化 BertModel
        if pretrained_dir is not None:
            try:
                # 尝试使用 transformers 自动加载（若你保存为 from_pretrained 形式）
                if use_transformers_auto:
                    self.bert = AutoModel.from_pretrained(pretrained_dir)
                    self.config = AutoConfig.from_pretrained(pretrained_dir)
                else:
                    self.config = BertConfig.from_pretrained(pretrained_dir)
                    self.bert = BertModel.from_pretrained(pretrained_dir, config=self.config)
            except Exception as e:
                print("Warning: failed to load pretrained from", pretrained_dir, "->", e)
                if config is None:
                    raise
                self.config = config
                self.bert = BertModel(self.config)
        else:
            if config is None:
                raise ValueError("Either pretrained_dir or config must be provided")
            self.config = config
            self.bert = BertModel(self.config)

        bert_hidden = self.config.hidden_size

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=bert_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(dropout if lstm_layers > 1 else 0.0)
        )

        # classifier to produce emissions for CRF
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        If labels provided -> returns loss (scalar)
        Else -> returns list of predicted tag sequences
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, H)

        sequence_output = self.dropout(sequence_output)

        lstm_out, _ = self.lstm(sequence_output)  # (B, L, 2*hidden)
        emissions = self.classifier(lstm_out)     # (B, L, num_labels)

        mask = attention_mask.bool() if attention_mask is not None else None

        if labels is not None:
            # Expect labels shape (B, L), mask shape (B, L)
            # CRF loss returns log-likelihood; negative to minimize
            log_likelihood = self.crf(emissions, labels, mask=mask, reduction='mean')
            loss = -log_likelihood
            return loss
        else:
            # decode returns list of lists (len=B) of predicted tag ids (variable length if masked)
            preds = self.crf.decode(emissions, mask=mask)
            return preds
