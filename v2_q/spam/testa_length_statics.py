import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图风格
sns.set(style="whitegrid")
# 如果你的环境没有 SimHei，可能会显示方框，可以尝试更换为 'Arial Unicode MS' (Mac) 或其他中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("1. 读取数据中...")
# 确保文件名和路径正确
df = pd.read_csv('test_a.csv', sep='\t')

print("2. 计算长度...")
doc_lens = df['text'].apply(lambda x: len(x.split()))

# --- 统计核心指标 (新增高分位点) ---
print("-" * 30)
print("3. 核心分位点统计：")

# 定义你想看的所有分位点
percentiles = [50, 90, 95, 96, 97, 98, 99, 99.5]
results = np.percentile(doc_lens, percentiles)

# 把结果存个字典方便后面调用
stats = dict(zip(percentiles, results))

# 打印出来
for p, val in stats.items():
    print(f"P{p}: {int(val)}")

max_len = np.max(doc_lens)
print(f"Max (最大值): {max_len}")
print("-" * 30)

# --- 开始绘图 ---
print("4. 正在生成高精度直方图...")

fig, axes = plt.subplots(2, 1, figsize=(12, 14)) # 图表拉高一点，看得清楚

# [图1] 全景图：看整体和极值
sns.histplot(doc_lens, bins=200, kde=False, ax=axes[0], color='navy')
axes[0].set_title('全量数据长度分布 (全景 - Log Scale)', fontsize=14)
axes[0].set_yscale('log') 
axes[0].set_xlabel('文章长度')
axes[0].set_ylabel('数量 (Log Scale)')
# 在全景图里标一下 P99
axes[0].axvline(stats[99], color='magenta', linestyle='--', label=f'P99: {int(stats[99])}')
axes[0].legend()

# [图2] 高清特写图：拉长视距，看到 P99.5
# 截取 0 到 P99.5 的数据来画图，这样能看到绝大多数数据的细节
limit_len = int(stats[99.5]) 
subset_lens = doc_lens[doc_lens <= limit_len]

# bins 设大一点，增加颗粒度
sns.histplot(subset_lens, bins=120, kde=True, ax=axes[1], color='teal', edgecolor='k', linewidth=0.5)
axes[1].set_title(f'高分位点特写 (0 - {limit_len} 字, 涵盖99.5%数据)', fontsize=14)
axes[1].set_xlabel('文章长度')
axes[1].set_ylabel('数量')

# --- 关键辅助线 (根据你的需求增加了密度) ---
# P50 (中位数) - 蓝色
axes[1].axvline(stats[50], color='blue', linestyle='--', linewidth=1.5, label=f'P50: {int(stats[50])}')
# P95 (常规截断点) - 红色
axes[1].axvline(stats[95], color='red', linestyle='--', linewidth=1.5, label=f'P95: {int(stats[95])}')
# P98 (Rank 1 关注点) - 橙色
axes[1].axvline(stats[98], color='orange', linestyle='--', linewidth=1.5, label=f'P98: {int(stats[98])}')
# P99 (极限点) - 紫色
axes[1].axvline(stats[99], color='purple', linestyle='--', linewidth=1.5, label=f'P99: {int(stats[99])}')

axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('testa_length_distribution_detail.png', dpi=300)
print("图表已保存为 testa_length_distribution_detail.png，请查看。")
# plt.show() # 如果不在 notebook 环境，这行可以注释掉以免卡住