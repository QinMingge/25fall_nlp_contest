# gen_vocab.py
vocab_size = 10000
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

with open("vocab.txt", "w", encoding="utf-8") as f:
    for token in special_tokens:
        f.write(token + "\n")
    for i in range(vocab_size - len(special_tokens)):
        f.write(str(i) + "\n")
print("✅ vocab.txt 生成完毕！")