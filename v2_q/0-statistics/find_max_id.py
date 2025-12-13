import pandas as pd
from tqdm import tqdm
import os

# 配置：把所有能找到的数据文件都加进去
# 只有把测试集也算进去，才能保证推理时不报错
files = ['train_set.csv', 'test_a.csv'] 

max_id = 0
min_id = 999999

print("正在扫描所有数据中的最大 Token ID...")

for f in files:
    if os.path.exists(f):
        print(f"-> 读取 {f}...")
        df = pd.read_csv(f, sep='\t')
        
        # 逐行扫描
        for text in tqdm(df['text'], desc=f"Scanning {f}"):
            if isinstance(text, str):
                # 将字符串 "375 10 99" 转为数字列表 [375, 10, 99]
                ids = [int(t) for t in text.split()]
                
                if ids: # 确保不是空行
                    local_max = max(ids)
                    local_min = min(ids)
                    
                    if local_max > max_id:
                        max_id = local_max
                    if local_min < min_id:
                        min_id = local_min
    else:
        print(f"⚠️ 文件 {f} 不存在，跳过。")

print("-" * 30)
print(f"【统计结果】")
print(f"最小 ID: {min_id}")
print(f"最大 ID: {max_id}")
print("-" * 30)

# 给出建议
suggested_vocab = max_id + 100 # 留点余量
print(f"建议设置 BertConfig 的 vocab_size >= {suggested_vocab}")
print(f"Rank 1 方案使用了特殊 Token (如 7999)，请确保 vocab_size 能覆盖这些特殊 Token。")