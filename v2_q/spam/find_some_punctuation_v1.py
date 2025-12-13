import pandas as pd
import numpy as np
import os
from collections import Counter
from tqdm import tqdm

# --- 配置 ---
FILES = ['train_set.csv', 'test_a.csv'] 
TOP_N_END = 5    # 打印文末频率前5
TOP_N_GLOBAL = 5 # 打印全局频率前5

# ------------------------------------------------------------------
# 第一阶段：数据读取与全量频率扫描
# ------------------------------------------------------------------
print("=== 第一阶段：全量频率扫描 ===")

df_list = []
for f in FILES:
    if os.path.exists(f):
        print(f"-> 读取 {f} ...")
        df_list.append(pd.read_csv(f, sep='\t'))
df = pd.concat(df_list, ignore_index=True)

total_docs = len(df)
print(f"-> 文档总数 (分母A): {total_docs}")

global_counter = Counter()
last_token_counter = Counter()
total_tokens = 0 # 用于计算全局频率的分母

print("-> 正在扫描所有 Token...")
for text in tqdm(df['text'], desc="Pass 1"):
    if not isinstance(text, str): continue
    tokens = text.split()
    if not tokens: continue
    
    # 累加总字数
    total_tokens += len(tokens)
    
    # 1. 记录文末词
    last_token_counter[tokens[-1]] += 1
    
    # 2. 记录全局频率
    global_counter.update(tokens)

print(f"-> Token 总数 (分母B): {total_tokens}")

# ------------------------------------------------------------------
# 打印第一阶段统计结果 (应你的要求 1)
# ------------------------------------------------------------------
print("\n" + "-" * 60)
print("【统计 A：文末出现频率 Top 5】 (分母 = 文档总数)")
print(f"{'Token':<10} | {'文末次数':<12} | {'文末占比(%)':<12}")
print("-" * 60)
for t, count in last_token_counter.most_common(TOP_N_END):
    ratio = (count / total_docs) * 100
    print(f"{t:<10} | {count:<12} | {ratio:.4f}%")

print("\n" + "-" * 60)
print("【统计 B：全局出现频率 Top 5】 (分母 = Token总数)")
print(f"{'Token':<10} | {'全局次数':<12} | {'全局占比(%)':<12}")
print("-" * 60)
for t, count in global_counter.most_common(TOP_N_GLOBAL):
    ratio = (count / total_tokens) * 100
    print(f"{t:<10} | {count:<12} | {ratio:.4f}%")


# ------------------------------------------------------------------
# 第二阶段：锁定监控名单
# ------------------------------------------------------------------
# 选取文末 Top 5 和 全局 Top 3 进入下一轮深度扫描
suspects_end = [t for t, _ in last_token_counter.most_common(5)]
suspects_global = [t for t, _ in global_counter.most_common(3)]
target_tokens = list(set(suspects_end + suspects_global))

print(f"\n-> 锁定 {len(target_tokens)} 个目标进行间隔计算: {target_tokens}")

# ------------------------------------------------------------------
# 第三阶段：计算间隔分布
# ------------------------------------------------------------------
print("\n=== 第三阶段：间隔分布计算 (深度扫描) ===")

interval_stats = {t: [] for t in target_tokens}
target_set = set(target_tokens)

for text in tqdm(df['text'], desc="Pass 2"):
    if not isinstance(text, str): continue
    tokens = text.split()
    if not tokens: continue
    
    # 记录位置
    positions = {t: [] for t in target_tokens}
    for i, token in enumerate(tokens):
        if token in target_set:
            positions[token].append(i)
            
    # 计算间隔
    for t in target_tokens:
        pos_list = positions[t]
        if len(pos_list) >= 2:
            diffs = np.diff(pos_list)
            interval_stats[t].extend(diffs)

# ------------------------------------------------------------------
# 第四阶段：事实结果输出 (应你的要求 2)
# ------------------------------------------------------------------
print("\n" + "=" * 90)
print(f"{'Token':<8} | {'文末次数':<12} | {'文末占比(%)':<12} | {'全局次数':<12} | {'全局占比(%)':<12} | {'平均间隔':<10}")
print("=" * 90)

# 严格按“文末次数”降序排列
sorted_targets = sorted(target_tokens, key=lambda x: last_token_counter[x], reverse=True)

for t in sorted_targets:
    # 基础数据
    count_end = last_token_counter[t]
    count_global = global_counter[t]
    
    # 占比计算
    ratio_end_doc = (count_end / total_docs) * 100       # 占文档总数
    ratio_global_token = (count_global / total_tokens) * 100 # 占Token总数
    
    # 间隔计算
    intervals = interval_stats[t]
    avg_interval = np.mean(intervals) if intervals else 0.0
    
    print(f"{t:<8} | {count_end:<12} | {ratio_end_doc:<12.4f} | {count_global:<12} | {ratio_global_token:<12.4f} | {avg_interval:<10.1f}")

print("=" * 90)