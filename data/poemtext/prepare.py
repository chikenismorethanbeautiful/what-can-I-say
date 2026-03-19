import os
import requests
import tiktoken
import numpy as np

# 打开数据文件
input_file_path = 'tang_poet.txt'
with open(input_file_path, 'r', encoding="utf-8") as f:
    data = f.read()

# 按9:1划分训练集和测试集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 使用GPT-2 BPE编码
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids)} tokens")
print(f"val has {len(val_ids)} tokens")

# 保存为二进制文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))