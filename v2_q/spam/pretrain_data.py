import pandas as pd
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 数据扩充：Rank 4 策略，全量数据预训练
INPUT_FILES = ['train_set.csv', 'test_a.csv'] 
OUTPUT_FILE = 'pretrain_data.csv'

# 2. 长度设定：Rank 1 策略 长上下文
MAX_LEN = 4096

# 3. 智能切割锚点：基于你的统计结论
# 900: 句号 (主力)
# 2662: 感叹号/问号 (强力结束符)
STOP_TOKENS = ['900', '2662'] 

# 回退搜索范围：在 4096 截断点往前找多少个词？
BACKOFF_WINDOW = 200 
# ===========================================

def smart_split_text(text, max_len):
    """
    融合策略切分算法：
    1. 中间段落：寻找 STOP_TOKENS 进行语义完整切分。
    2. 最后一段：采用 Rank 1 的 Overlap 策略（倒数回退），保证不出现过短文本。
    """
    if not isinstance(text, str): 
        return []
        
    words = text.split()
    total_len = len(words)
    
    # Case 0: 如果本身就短于 max_len，直接作为一条数据
    if total_len <= max_len:
        return [' '.join(words)]
    
    result = []
    start = 0

    while start < total_len:
        remaining_len = total_len - start
        
        # === 处理最后一段 ===
        # 如果剩下的长度不足以填满一个 max_len
        # === 最后一段也要完美开头 ===
        if remaining_len < max_len:
            # 1. 设定“最早”可以接受的开头位置
            # 我们希望最后一段尽量长，所以从 (total_len - max_len) 开始看
            # 但为了找句号，我们允许再往前回退 BACKOFF_WINDOW
            
            ideal_start = total_len - max_len
            
            # 搜索范围：在 ideal_start 附近往后找句号
            # 例如：倒数第 4096 个字之后的 200 字范围内找
            search_start = ideal_start
            search_end = min(total_len, ideal_start + BACKOFF_WINDOW)
            
            smart_start = -1
            
            # 正序查找 (从左往右)句号或感叹号
            for i in range(search_start, search_end):
                if words[i] in STOP_TOKENS:
                    smart_start = i + 1 # 切在句号后面，作为新句子的开头
                    break
            
            if smart_start != -1:
                # 方案 A：找到了句号！
                # 这一段从 smart_start 开始，直到全文结束
                # 长度肯定 < 4096，后续会自动 Pad，没问题！
                last_segment = words[smart_start:]
            else:
                # 方案 B：没找到句号，只能按 Rank 1 的硬回退（保证长度）
                # 从倒数 4096 字开始
                last_segment = words[-max_len:]
                
            result.append(' '.join(last_segment))
            
            # 处理完毕，退出循环
            break
            
        # === Rank 6 逻辑：中间段落的智能标点切分 ===
        # 1. 确定理论硬切点
        end = start + max_len
        
        # 2. 智能回退：寻找句号
        chunk = words[start:end]
        
        # 搜索范围：从 chunk 末尾往前倒推 BACKOFF_WINDOW
        search_start = len(chunk) - 1
        search_end = max(-1, len(chunk) - 1 - BACKOFF_WINDOW)
        
        split_idx = -1
        
        for i in range(search_start, search_end, -1):
            if chunk[i] in STOP_TOKENS:
                split_idx = i
                break 
        
        # 3. 决策
        if split_idx != -1:
            # 找到了句号，完美切分
            real_end = start + split_idx + 1
        else:
            # 没找到句号，被迫硬切
            real_end = end
            
        result.append(' '.join(words[start:real_end]))
        
        # 更新起点
        start = real_end
        
    return result

def main():
    print(f"=== 终极版预训练数据构建 (MAX_LEN={MAX_LEN}) ===")
    print(f"-> 策略: 全量数据 + 智能标点切分 + 尾部Overlap回退")
    print(f"-> 锚点: {STOP_TOKENS}")
    
    all_segments = []
    total_raw_docs = 0
    
    for f in INPUT_FILES:
        if os.path.exists(f):
            print(f"\n正在读取 {f} ...")
            df = pd.read_csv(f, sep='\t')
            total_raw_docs += len(df)
            
            for text in tqdm(df['text'], desc=f"Splitting"):
                segs = smart_split_text(text, MAX_LEN)
                all_segments.extend(segs)
        else:
            print(f"⚠️ 文件 {f} 不存在")
    
    print("-" * 40)
    print(f"处理汇总:")
    print(f"1. 原始文档数: {total_raw_docs}")
    print(f"2. 切分后样本数: {len(all_segments)}")
    print(f"   (膨胀系数: {len(all_segments)/total_raw_docs:.2f}x)")
    
    print("\n正在保存...")
    pd.DataFrame({'text': all_segments}).to_csv(OUTPUT_FILE, index=False, sep='\t')
    print(f"✅ 完成！保存至: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()