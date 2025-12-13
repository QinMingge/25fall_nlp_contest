import torch
import torch.nn as nn
from transformers import BertModel

class BertLinear(nn.Module):
    def __init__(self, bert_model_path, num_labels=14, dropout=0.1):
        super(BertLinear, self).__init__()
        
        # 1. 加载 BERT 模型
        # 🆕 add_pooling_layer=False: 不加载 Pooler 层，避免 DDP 报错 "unused parameters"
        self.bert = BertModel.from_pretrained(bert_model_path, add_pooling_layer=False)
        
        # 2. 动态获取嵌入维度
        embedding_dim = self.bert.config.hidden_size 
        print(f"✅ BERT Embedding Dimension (Hidden Size) Detected: {embedding_dim}")

        # 3. 分类层
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_labels)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the underlying BERT model.
        """
        self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # 1. BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 🆕 手动提取 [CLS] 向量 (batch_size, hidden_size)
        # 因为 add_pooling_layer=False，所以 outputs.pooler_output 不可用
        cls_token = outputs.last_hidden_state[:, 0, :]
        pooled_output = cls_token

        # 2. 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        
        # 3. 计算 Loss (如果传入了 labels)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}
            
        return {"logits": logits}
