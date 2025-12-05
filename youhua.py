import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import joblib

# =============================
# 1. 读取数据
# =============================
print("读取数据中...")
train_df = pd.read_csv('./train_set.csv', sep='\t')
test_df = pd.read_csv('./test_a.csv', sep='\t')

X = train_df['text'].astype(str)
y = train_df['label']
X_test = test_df['text'].astype(str)

# =============================
# 2. Word-token Hashing 特征
# =============================
print("构建 Word-token Hashing 特征...")
vec_word = HashingVectorizer(
    n_features=2**21,       # 2^20 → 2^21 可有效提升
    alternate_sign=False,   # 避免正负抵消
    analyzer='word',
    token_pattern=r'\S+',   # 将数字当独立 token
    ngram_range=(1, 3),     # 一定要有 3gram，效果显著
)

X_word = vec_word.transform(X)

# =============================
# 3. Char-level TF-IDF 特征
# =============================
print("构建 Char-level TF-IDF 特征...")
vec_char = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),      # 人工验证效果最好
    max_features=60000       # 50k~80k 区间内都很稳
)

X_char = vec_char.fit_transform(X)

# =============================
# 4. 融合特征
# =============================
print("拼接 Word + Char 特征...")
X_all = hstack([X_word, X_char])  # 水平拼接

# =============================
# 5. 划分验证集
# =============================
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# =============================
# 6. 训练线性 SVM（SGDClassifier）
# =============================
print("训练 SVM 模型中...")
model = SGDClassifier(
    loss='hinge',        # 线性 SVM
    alpha=3e-6,          # 很关键！建议 1e-5~1e-6 之间调参
    max_iter=30,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# =============================
# 7. 验证
# =============================
print("\n验证集中...")
val_pred = model.predict(X_val)
acc = accuracy_score(y_val, val_pred)
print(f"验证集准确率: {acc:.4f}\n")
print(classification_report(y_val, val_pred))

# =============================
# 8. 使用全部数据重新训练
# =============================
print("\n使用全部训练数据重新训练最终模型...")
model.fit(X_all, y)

# 保存模型与 TF-IDF
joblib.dump(model, "fusion_svm_model.pkl")
joblib.dump(vec_char, "fusion_char_vectorizer.pkl")
print("模型已保存。")

# =============================
# 9. 测试集推理
# =============================
print("对测试集提取特征...")
X_word_test = vec_word.transform(X_test)
X_char_test = vec_char.transform(X_test)
X_test_all = hstack([X_word_test, X_char_test])

print("进行预测...")
test_pred = model.predict(X_test_all)

# =============================
# 10. 生成提交文件
# =============================
if 'id' in test_df.columns:
    sub = pd.DataFrame({'id': test_df['id'], 'label': test_pred})
else:
    sub = pd.DataFrame({'id': range(len(test_pred)), 'label': test_pred})

sub.to_csv('submission_fusion.csv', index=False)
print("提交文件已生成: submission_fusion.csv")
print(sub.head())