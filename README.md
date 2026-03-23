# nanoGPT

极简 GPT 实现，支持诗词生成、小说风格文本生成，适合大模型入门学习。

## 项目介绍

nanoGPT 是基于 PyTorch 实现的轻量级 GPT 模型，代码简洁、易于理解、可复现性强，专为大语言模型入门教育设计。本项目支持自定义文本训练，可快速训练唐诗生成、《天龙八部》风格续写等小参数量 GPT 模型。

## 项目特性

- 轻量极简，核心代码易读易改
- 支持唐诗（5.8 万首）、《天龙八部》（124 万字符）数据集训练
- 一键训练、一键推理
- 兼容 GPU / CPU 运行
- 适合初学者理解 Transformer 与自回归生成

## 环境依赖

```bash
pip install torch<3.0 numpy<3.0 transformers<3.0 datasets<3.0 tiktoken<3.0 tqdm<3.0
```

---

## 快速开始

### 1. 下载项目

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
```

![b](.\image\b.png)

### 2. 准备数据集

在 `data/` 目录下创建对应数据集文件夹：

- 诗词：`data/poemtext/tang_poet.txt`
- 天龙八部：`data/tianlong/tianlong.txt`

### 3. 数据预处理

以诗词数据集为例：在 `data/poemtext/` 下编写并运行 `prepare.py`：

```bash
python data/poemtext/prepare.py
```

生成 `train.bin` 和 `val.bin` 二进制训练文件。

### 4. 训练配置

在 `config/` 下新建训练配置文件 `train_poemtext_char.py`，配置模型参数。

### 5. 启动训练

**GPU 训练**

```bash
python train.py config/train_poemtext_char.py
```

**CPU 训练**

```bash
python train.py config/train_poemtext_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

训练完成后，模型权重保存在 `out-poemtext-char/ckpt.pt`。

### 6. 模型推理（文本生成）

**GPU 推理**

```bash
python sample.py --out_dir=out-poemtext-char
```

![f](.\image\a.png)

**CPU 推理**

```bash
python sample.py --out_dir=out-poemtext-char --device=cpu
```

---

## 《天龙八部》模型训练流程

- **数据集**：`data/tianlong/tianlong.txt`

- **预处理**：`python data/tianlong/prepare.py`

- **训练配置**：新建 `config/train_tianlong_char.py`，修改 `dataset='tianlong'`

- **训练命令**：同上

- **推理**：`python sample.py --out_dir=out-tianlong-char`

  训练截图与效果:

  ![f](.\image\f.png)

  ![d](.\image\d.png)
  

---

## 核心文件说明

| 文件            | 说明                                |
| --------------- | ----------------------------------- |
| `prepare.py`    | 文本编码、数据集划分、生成 bin 文件 |
| `train.py`      | 模型训练主程序                      |
| `sample.py`     | 模型推理与文本生成                  |
| `config/xxx.py` | 训练超参数配置                      |
| `data/xxx`      | 数据集存放目录                      |
| `out-xxx/`      | 模型权重保存目录                    |

---

## 调参策略

- 修改超参数观察生成效果（这里修改了学习率和训练迭代次数）

  ```python
  out_dir = 'out-poemtext-char'
  eval_interval = 250  # keep frequent because we'll overfit
  eval_iters = 200
  log_interval = 10    # don't print too too often
  
  always_save_checkpoint = False
  
  dataset = 'poemtext'
  gradient_accumulation_steps = 1
  batch_size = 64
  block_size = 256     # context of up to 256 previous characters
  
  n_layer = 6
  n_head = 6
  n_embed = 384
  dropout = 0.2
  
  learning_rate = 1e-3   # with baby networks can afford to go a bit higher
  max_iters = 2000
  lr_decay_iters = 2000  # make equal to max_iters usually
  min_lr = 1e-5          # learning_rate / 10 usually
  beta2 = 0.99           # make a bit bigger because number of tokens per iter is small
  
  warmup_iters = 100     # not super necessary potentially
  ```

- 超参数修改后效果

  ![c](.\image\c.png)

---

# 思考题

1. **使用《天龙八部》tianlong.txt数据集训练一个GPT模型，并生成内容看看效果**

   要使用《天龙八部》数据集训练GPT模型，首先需要在nanoGPT/data文件夹下创建tianlong文件夹，并将tianlong.txt放置其中。在tianlong文件夹下创建prepare.py文件进行数据预处理，代码逻辑与诗词数据集相同，主要区别在于《天龙八部》文本约124万个字符，经过tiktoken的GPT-2 BPE编码后，训练集和测试集的token数量会更多。按照9：1的比例划分数据集后，将数据保存为train.bin和val.bin二进制文件。接着在config文件夹下创建train_tianlong_char.py配置文件，将out_dir设置为'out-tianlong-char'，dataset设置为'tianlong'，其他参数如n_layer=6、n_head=6、n_embd=384等保持不变。训练完成后运行采样命令python sample.py --out_dir=out-tianlong-char，即可看到生成的《天龙八部》风格文本，内容会模仿金庸武侠小说的语言风格和人物对话。

   | 参数    | 值                  | 说明             |
   | ------- | ------------------- | ---------------- |
   | out_dir | 'out-tianlong-char' | 模型保存路径     |
   | dataset | 'tianlong'          | 数据集名称       |
   | n_layer | 6                   | Transformer 层数 |
   | n_head  | 6                   | 注意力头数       |
   | n_embd  | 384                 | 嵌入维度         |

2. **思考config/train_poemtext_char.py训练文件中其他参数的含义，修改其参数并重新进行训练，看看其效果怎么样**

   **config/train_poemtext_char.py中的关键参数含义如下：**

   | 参数           | 说明               |
   | -------------- | ------------------ |
   | out_dir        | 指定模型保存路径   |
   | dataset        | 指定数据文件夹     |
   | eval_interval  | 控制评估频率       |
   | eval_iters     | 控制评估迭代次数   |
   | log_interval   | 控制日志打印频率   |
   | batch_size     | 批次大小           |
   | block_size     | 上下文长度         |
   | n_layer        | Transformer 层数   |
   | n_head         | 注意力头数         |
   | n_embd         | 嵌入维度           |
   | dropout        | 防止过拟合         |
   | learning_rate  | 学习率             |
   | min_lr         | 最小学习率         |
   | warmup_iters   | 预热迭代次数       |
   | lr_decay_iters | 学习率衰减迭代次数 |
   | beta2          | Adam优化器的超参数 |

   修改参数后重新训练会发现：增大n_layer、n_head、n_embd会增加模型参数量，可能提升表现但训练更慢；降低dropout可能加速训练但易过拟合；提高learning_rate可能加快收敛但不稳定；减小batch_size虽省显存但梯度更新噪声大。不同参数组合会显著影响生成诗词的质量和多样性。

3. **研究模型采样文件sample.py，思考模型的推理过程是怎么样的**

   sample.py的推理过程如下：首先加载训练好的checkpoint文件ckpt.pt，从中恢复模型参数和配置；然后从meta.pkl加载编码器（encode）和解码器（decode），实现字符与索引之间的相互转换；接着将提示词（如"ROMEO:"）通过encode函数转换为token索引序列，并转换为PyTorch张量；模型进入评估模式后调用generate函数，该函数的核心是循环预测：每次将当前序列输入模型，获取最后一个位置的logits输出，通过temperature参数调整概率分布的平滑度，通过top_k过滤低概率token，然后从调整后的概率分布中采样得到下一个token，将新token拼接到序列末尾；重复这个过程直到达到max_new_tokens或遇到结束标记；最后通过decode函数将生成的token序列转换回文本并打印输出。整个过程体现了自回归生成的特点，即逐个token预测并不断拼接。

   | 参数/组件      | 作用                                   |
   | -------------- | -------------------------------------- |
   | ckpt.pt        | 保存训练好的模型参数和配置             |
   | meta.pkl       | 存储编码器（encode）和解码器（decode） |
   | encode         | 将字符转换为token索引                  |
   | decode         | 将token索引转换回字符                  |
   | temperature    | 控制概率分布平滑度，影响生成随机性     |
   | top_k          | 过滤低概率token，只保留概率最高的k个   |
   | max_new_tokens | 控制生成的最大token数量                |
   | generate       | 核心生成函数，实现自回归预测           |

## License

MIT
```

---
