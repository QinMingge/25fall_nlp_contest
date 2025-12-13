import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class BertTextCNN(nn.Module):
    # bert_model_path 应该指向您预训练保存的目录，
    def __init__(self, bert_model_path, num_labels=14, filter_sizes=(2, 3, 4, 5), num_filters=100):
        super(BertTextCNN, self).__init__()
        
        # 1. 加载 BERT 模型 (会加载您预训练的配置和权重)
        # 🆕 add_pooling_layer=False: 不加载 Pooler 层，避免 DDP 报错 "unused parameters"
        self.bert = BertModel.from_pretrained(bert_model_path, add_pooling_layer=False)
        
        # 2. 动态获取嵌入维度 D （D=256 或 D=512）
        embedding_dim = self.bert.config.hidden_size 
        print(f"✅ BERT Embedding Dimension (Hidden Size) Detected: {embedding_dim}")
        
        # 3. TextCNN 卷积层定义：使用 Conv1d 替代 Conv2d，效率更高
        # Conv1d 输入: (Batch, Hidden, Seq)
        # Conv1d 输出: (Batch, Out_Channels, Seq_Out)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                      out_channels=num_filters, 
                      kernel_size=k) 
            for k in filter_sizes
        ])
        
        # 4. 分类层
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the underlying BERT model.
        """
        self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # 1. BERT 编码
        # 注意：必须传入 attention_mask，否则 BERT 无法区分 PAD
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        bert_output = outputs.last_hidden_state
        
        # 2. TextCNN 处理
        # Conv1d 需要输入 (batch_size, in_channels, seq_len)
        # bert_output 是 (batch_size, seq_len, hidden_size) -> permute -> (batch_size, hidden_size, seq_len)
        bert_output_cnn = bert_output.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            # 卷积: (batch_size, num_filters, seq_len-k+1)
            conv_out = F.relu(conv(bert_output_cnn))
            
            # 池化: Max-over-time pooling
            # (batch_size, num_filters, seq_len-k+1) -> max_pool -> (batch_size, num_filters, 1) -> squeeze -> (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        # 拼接: (batch_size, num_filters * len(filter_sizes))
        concat_output = torch.cat(conv_outputs, 1)
        
        # 3. 分类
        dropout_output = self.dropout(concat_output)
        logits = self.fc(dropout_output)
        
        # 4. 计算 Loss (如果传入了 labels)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}
            
        return {"logits": logits}