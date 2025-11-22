import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

print("1. 正在读取数据...")
# 读取训练集，sep='\t' 是因为数据是用制表符分隔的
train_df = pd.read_csv('train_set.csv', sep='\t')
# 读取测试集
test_df = pd.read_csv('test_a.csv', sep='\t')

print("2. 正在预处理数据...")
# 这里的 text 是数字串，我们直接把它当成普通的文本字符串处理
# fillna 是为了防止有空数据报错
train_text = train_df['text'].fillna("0")
test_text = test_df['text'].fillna("0")
all_text = pd.concat([train_text, test_text])

print("3. 正在将文本转换为数字矩阵 (TF-IDF)...")
# TfidfVectorizer 会计算每个词的重要性
# ngram_range=(1, 2) 意思是不仅看"词"，还看"词组"（比如把"6 57"当做一个特征）
# max_features=5000 限制只取最重要的5000个特征，防止你虚拟机内存爆炸
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
vectorizer.fit(all_text)

train_X = vectorizer.transform(train_text)
test_X = vectorizer.transform(test_text)

print("4. 正在训练模型 (LinearSVC)...")
# 划分一个小验证集来看看效果，验证集占 20%
x_train, x_val, y_train, y_val = train_test_split(train_X, train_df['label'], test_size=0.2)

# LinearSVC 是一个很快的分类器，适合文本分类
clf = LinearSVC()
clf.fit(x_train, y_train)

# 在验证集上看看分数
val_pred = clf.predict(x_val)
score = f1_score(y_val, val_pred, average='macro')
print(f"本地验证集的 F1 分数: {score:.4f}")

print("5. 正在生成提交文件...")
# 对测试集进行预测
test_pred = clf.predict(test_X)

# 生成提交格式 csv
submission = pd.DataFrame({'label': test_pred})
submission.to_csv('submit.csv', index=False)

print("完成！请提交 submit.csv 到天池官网。")