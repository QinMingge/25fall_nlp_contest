import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties # 引入 FontProperties
import matplotlib as mpl # 引入 matplotlib 库

plt.rcParams['font.sans-serif'] = ['SimHei']
# 1. 定义标签映射关系
label_map = {
    0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 
    6: '教育', 7: '财经', 8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 
    12: '彩票', 13: '星座'
}

# 2. 读取、清洗和类型转换
try:
    # 假设文件路径已修正，现在文件位于 './train_set.csv' 或其他正确位置
    # 建议使用 header=None读取，然后进行清洗
    train_df = pd.read_csv('train_set.csv', sep='\t', header=None, names=['label', 'text'])

    # 清洗步骤：删除列名为 'label' 的错误行
    train_df = train_df[train_df['label'] != 'label'].copy()
    
    # **关键修正**：将 'label' 列转换为整数类型，解决 KeyError
    train_df['label'] = train_df['label'].astype(int)

except FileNotFoundError:
    print("错误：无法找到文件。请确保您的文件路径正确。")
    exit()

# 3. 统计标签数量
label_counts = train_df['label'].value_counts().sort_index()

# 4. 打印统计结果
print("--- 修正后的标签数量统计 ---")
# 打印每个类别的数量
for label_id, count in label_counts.items():
    category_name = label_map.get(label_id, f'未知标签{label_id}')
    print(f"[{label_id:2}] {category_name:4}：{count} 条")

print(f"\n总样本数：{label_counts.sum()} 条")

# 5. 生成直方图并保存
plt.figure(figsize=(12, 6))

# 使用标签映射后的名称作为 x 轴
labels = [label_map[i] for i in label_counts.index]
counts = label_counts.values

plt.bar(labels, counts, color='skyblue')
plt.xlabel('新闻类别')
plt.ylabel('样本数量')
plt.title('训练集样本类别分布')
plt.xticks(rotation=45, ha='right')

# 在每个柱子上方显示具体数值
for i, count in enumerate(counts):
    plt.text(i, count + 500, str(count), ha='center')

plt.tight_layout()

# **保存图片**：将图表保存为 'label_distribution.png'
save_path = 'label_distribution.png'
plt.savefig(save_path)
print(f"\n✅ 成功保存直方图至: {save_path}")

# plt.show() # 如果在无图形界面的服务器上运行，可以注释掉此行