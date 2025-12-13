import pandas as pd
import torch
from transformers import BertTokenizerFast
from datasets import load_dataset

# 创建测试数据
test_data = {
    'text': ['100 200 300', '400 500 600 700']
}
df = pd.DataFrame(test_data)
df.to_csv('test_data.csv', index=False)

# 方式1：自定义Dataset
class PretrainDatasetV1(Dataset):
    def __init__(self, csv_file, max_length=10):
        df = pd.read_csv(csv_file, dtype={'text': str})
        self.texts = df['text'].tolist()
        self.max_length = max_length
        self.extended_texts = []
        for text in self.texts:
            self.extended_texts.extend([text] * 2)
    
    def __len__(self):
        return len(self.extended_texts)
    
    def __getitem__(self, idx):
        text = self.extended_texts[idx]
        text = [int(token) for token in text.split()]
        text = [7999] + text + [7998] * (self.max_length - len(text) - 1)
        labels = text.copy()
        rand = torch.rand(len(text))
        mask_arr = (rand < 0.15) * (torch.tensor(text) != 7999) * (torch.tensor(text) != 7998)
        selection = torch.flatten(mask_arr.nonzero()).tolist()
        for sel in selection:
            text[sel] = 7997
        return {
            'input_ids': torch.tensor(text),
            'labels': torch.tensor(labels)
        }

# 方式2：使用Tokenizer
def process_with_tokenizer():
    # 创建vocab文件（模拟）
    vocab_lines = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab_lines.extend([str(i) for i in range(10001)])  # 0-10000
    with open("vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_lines))
    
    tokenizer = BertTokenizerFast(
        vocab_file="vocab.txt", 
        do_lower_case=False,
        model_max_length=10,
        pad_token='[PAD]',
        cls_token='[CLS]',
        sep_token='[SEP]',
        mask_token='[MASK]',
        unk_token='[UNK]'
    )
    
    dataset = load_dataset("csv", data_files="test_data.csv", split="train")
    
    def tokenize_function(examples):
        # 这里需要特别注意：原始文本中的数字token需要映射
        # 例如："100" 应该映射到 105 (因为0->5, 1->6, ..., 100->105)
        tokens = []
        for text in examples['text']:
            num_tokens = [int(x) for x in text.split()]
            # 将数字映射到tokenizer的词汇表位置
            mapped_tokens = [num + 5 for num in num_tokens]  # 数字0对应索引5
            tokens.append([2] + mapped_tokens)  # 2是[CLS]
        
        # 编码并添加特殊标记
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=10,
            return_tensors='pt'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenizer, tokenized_dataset

# 对比函数
def compare_methods():
    print("="*50)
    print("对比两种处理方式")
    print("="*50)
    
    # 方法1
    print("\n1. 自定义Dataset处理:")
    dataset1 = PretrainDatasetV1('test_data.csv', max_length=10)
    
    for i in range(2):  # 只看前两个样本
        sample = dataset1[i]
        text = dataset1.extended_texts[i]
        print(f"\n原始文本: {text}")
        print(f"输入ID: {sample['input_ids'].tolist()}")
        print(f"标签: {sample['labels'].tolist()}")
        print(f"MASK位置: {(sample['input_ids'] == 7997).nonzero().flatten().tolist()}")
        print(f"PAD位置: {(sample['input_ids'] == 7998).nonzero().flatten().tolist()}")
    
    # 方法2
    print("\n" + "="*50)
    print("2. Tokenizer处理:")
    
    tokenizer, dataset2 = process_with_tokenizer()
    
    print(f"\nTokenizer特殊标记映射:")
    print(f"[CLS]: {tokenizer.cls_token_id}")
    print(f"[SEP]: {tokenizer.sep_token_id}")
    print(f"[MASK]: {tokenizer.mask_token_id}")
    print(f"[PAD]: {tokenizer.pad_token_id}")
    print(f"数字0: {tokenizer.encode('0')[1:-1][0]}")  # 去掉[CLS]和[SEP]
    print(f"数字100: {tokenizer.encode('100')[1:-1][0]}")
    
    print("\n处理后的样本:")
    for i in range(2):
        print(f"\n样本{i}:")
        print(f"输入ID: {dataset2[i]['input_ids'].tolist()}")
        print(f"Attention Mask: {dataset2[i]['attention_mask'].tolist()}")
        
        # 解码查看
        decoded = tokenizer.decode(dataset2[i]['input_ids'], skip_special_tokens=False)
        print(f"解码: {decoded}")

# 验证映射关系
def verify_mapping():
    print("\n" + "="*50)
    print("验证映射关系")
    print("="*50)
    
    # 创建测试数字
    test_numbers = [0, 1, 100, 7997, 7998, 7999]
    
    print("\n方法1的映射 (手动):")
    print(f"数字0 -> token: 0")
    print(f"数字100 -> token: 100")
    print(f"特殊标记: CLS=7999, PAD=7998, MASK=7997")
    
    print("\n方法2的映射 (Tokenizer):")
    vocab_lines = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab_lines.extend([str(i) for i in range(10001)])
    
    with open("temp_vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_lines))
    
    tokenizer = BertTokenizerFast(vocab_file="temp_vocab.txt", do_lower_case=False)
    
    for num in test_numbers:
        if num in [7997, 7998, 7999]:
            print(f"数字{num} -> 在Tokenizer中不存在（超出词汇表范围）")
        else:
            encoded = tokenizer.encode(str(num))[1:-1]  # 去掉特殊标记
            print(f"数字{num} -> token: {encoded[0] if encoded else '未找到'}")

# 运行对比
if __name__ == "__main__":
    compare_methods()
    verify_mapping()