"""
Sample from a trained model with visualization
"""
import os
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
visualize = True # 是否启用可视化
visualization_style = 'heatmap' # 'heatmap', 'token_distribution', 'attention', 'combined'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TextGenerationVisualizer:
    """文本生成过程可视化器"""

    def __init__(self, decode_func, style='heatmap'):
        self.decode = decode_func
        self.visualization_style = style  # 添加这个属性
        self.token_probs_history = []
        self.generated_tokens = []
        self.top_k_history = []
        self.fig = None
        self.axes = None
        self.is_initialized = False

    def setup_visualization(self):
        """设置可视化界面"""
        if self.visualization_style == 'combined':
            self.fig = plt.figure(figsize=(16, 10))
            # 创建网格布局
            gs = plt.GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)

            # 生成的文本显示区域
            self.ax_text = self.fig.add_subplot(gs[0, :])
            self.ax_text.axis('off')
            self.ax_text.set_title('生成的文本', fontsize=12, fontweight='bold')
            self.text_display = self.ax_text.text(0.05, 0.95, '', transform=self.ax_text.transAxes,
                                                 fontsize=10, verticalalignment='top',
                                                 fontfamily='monospace', wrap=True)

            # 概率分布热力图
            self.ax_heatmap = self.fig.add_subplot(gs[1, 0:2])
            self.ax_heatmap.set_title('Token 概率分布热力图', fontsize=12, fontweight='bold')
            self.heatmap = None

            # Top-K 概率变化曲线
            self.ax_topk = self.fig.add_subplot(gs[1, 2])
            self.ax_topk.set_title('Top-5 概率变化', fontsize=12, fontweight='bold')
            self.ax_topk.set_xlabel('生成步数')
            self.ax_topk.set_ylabel('概率')
            self.topk_lines = []

            # 注意力权重可视化
            self.ax_attention = self.fig.add_subplot(gs[2, :])
            self.ax_attention.set_title('注意力权重可视化', fontsize=12, fontweight='bold')
            self.attention_img = None

        elif self.visualization_style == 'heatmap':
            self.fig, self.ax_heatmap = plt.subplots(figsize=(12, 8))
            self.ax_heatmap.set_title('Token 生成概率热力图', fontsize=14, fontweight='bold')
            self.ax_heatmap.set_xlabel('生成步数')
            self.ax_heatmap.set_ylabel('Token 排名')
            self.heatmap = None

        elif self.visualization_style == 'token_distribution':
            self.fig, (self.ax_dist, self.ax_text) = plt.subplots(2, 1, figsize=(12, 10))
            self.ax_dist.set_title('当前步 Token 概率分布', fontsize=14, fontweight='bold')
            self.ax_dist.set_xlabel('Token')
            self.ax_dist.set_ylabel('概率')
            self.ax_text.axis('off')
            self.ax_text.set_title('生成的文本', fontsize=12, fontweight='bold')
            self.text_display = self.ax_text.text(0.05, 0.95, '', transform=self.ax_text.transAxes,
                                                 fontsize=10, verticalalignment='top')

        plt.ion()  # 开启交互模式
        plt.tight_layout()
        plt.show(block=False)

    def update_visualization(self, token_probs, selected_token, step):
        """更新可视化"""
        if not self.is_initialized:
            self.setup_visualization()
            self.is_initialized = True

        # 记录历史数据
        self.token_probs_history.append(token_probs)
        self.generated_tokens.append(selected_token)

        # 获取 top-5 概率
        top_probs, top_indices = torch.topk(torch.tensor(token_probs), min(5, len(token_probs)))
        top_probs = top_probs.numpy()
        self.top_k_history.append(top_probs)

        # 根据风格更新可视化
        if self.visualization_style == 'combined':
            self._update_combined_view(top_probs, step)
        elif self.visualization_style == 'heatmap':
            self._update_heatmap_view(step)
        elif self.visualization_style == 'token_distribution':
            self._update_distribution_view(top_probs, top_indices)

        # 更新文本显示
        if hasattr(self, 'text_display'):
            generated_text = self.decode(self.generated_tokens)
            display_text = generated_text[-500:] if len(generated_text) > 500 else generated_text
            self.text_display.set_text(display_text)

        plt.draw()
        plt.pause(0.1)  # 短暂暂停以更新显示

    def _update_combined_view(self, top_probs, step):
        """更新组合视图"""
        # 更新热力图
        if self.heatmap is None:
            # 创建概率矩阵
            prob_matrix = np.zeros((min(20, len(self.token_probs_history)), len(self.token_probs_history)))
            for i, probs in enumerate(self.token_probs_history[-20:]):
                prob_matrix[i, :len(probs)] = probs[:20]
            self.heatmap = self.ax_heatmap.imshow(prob_matrix.T, aspect='auto', cmap='YlOrRd')
            self.ax_heatmap.set_xlabel('生成步数')
            self.ax_heatmap.set_ylabel('Token 排名')
            plt.colorbar(self.heatmap, ax=self.ax_heatmap)
        else:
            # 更新热力图数据
            prob_matrix = np.zeros((min(20, len(self.token_probs_history)), len(self.token_probs_history)))
            for i, probs in enumerate(self.token_probs_history[-20:]):
                prob_matrix[i, :len(probs)] = probs[:20]
            self.heatmap.set_data(prob_matrix.T)
            self.ax_heatmap.set_xlim(0, len(self.token_probs_history))

        # 更新 Top-K 曲线
        if not self.topk_lines:
            colors = ['red', 'orange', 'yellow', 'green', 'blue']
            for i in range(min(5, len(top_probs))):
                line, = self.ax_topk.plot([], [], color=colors[i], label=f'Top-{i+1}', linewidth=2)
                self.topk_lines.append(line)
            self.ax_topk.legend()
            self.ax_topk.grid(True, alpha=0.3)

        # 更新曲线数据
        for i, line in enumerate(self.topk_lines):
            if i < len(self.top_k_history[0]):
                y_data = [hist[i] for hist in self.top_k_history]
                x_data = range(len(y_data))
                line.set_data(x_data, y_data)
                self.ax_topk.set_xlim(0, len(self.top_k_history))
                self.ax_topk.set_ylim(0, max([max(hist) for hist in self.top_k_history]) * 1.1)

    def _update_heatmap_view(self, step):
        """更新热力图视图"""
        # 限制显示最近20个token的概率分布
        max_tokens_to_show = 20
        max_steps_to_show = min(50, len(self.token_probs_history))

        # 创建概率矩阵
        prob_matrix = np.zeros((max_tokens_to_show, max_steps_to_show))
        for i in range(max_steps_to_show):
            probs = self.token_probs_history[-(max_steps_to_show - i)]
            for j in range(min(max_tokens_to_show, len(probs))):
                prob_matrix[j, i] = probs[j]

        if self.heatmap is None:
            self.heatmap = self.ax_heatmap.imshow(prob_matrix, aspect='auto', cmap='viridis')
            self.ax_heatmap.set_xlabel('生成步数')
            self.ax_heatmap.set_ylabel('Token 排名')
            plt.colorbar(self.heatmap, ax=self.ax_heatmap)
        else:
            self.heatmap.set_data(prob_matrix)
            self.ax_heatmap.set_xlim(0, max_steps_to_show)
            self.ax_heatmap.set_ylim(0, max_tokens_to_show)

    def _update_distribution_view(self, top_probs, top_indices):
        """更新分布视图"""
        self.ax_dist.clear()
        tokens = [f"Token {idx.item()}" for idx in top_indices]
        bars = self.ax_dist.bar(tokens, top_probs, color='steelblue', alpha=0.7)
        self.ax_dist.set_ylim(0, 1)
        self.ax_dist.set_title(f'第 {len(self.generated_tokens)} 步 Token 概率分布', fontsize=12)
        self.ax_dist.set_xlabel('候选 Token')
        self.ax_dist.set_ylabel('概率')

        # 添加数值标签
        for bar, prob in zip(bars, top_probs):
            height = bar.get_height()
            self.ax_dist.text(bar.get_x() + bar.get_width()/2., height,
                            f'{prob:.3f}', ha='center', va='bottom')

    def close(self):
        """关闭可视化"""
        if self.fig:
            plt.ioff()
            plt.close(self.fig)

def visualize_attention_weights(model, input_ids, layer_idx=0, head_idx=0):
    """可视化注意力权重"""
    model.eval()
    with torch.no_grad():
        # 前向传播获取注意力权重
        x = input_ids
        for block_idx, block in enumerate(model.transformer.h):
            if block_idx == layer_idx:
                # 获取注意力层的输出和权重
                attn_output, attn_weights = block.attn(x, x, x, return_weights=True)
                return attn_weights[0, head_idx].cpu().numpy()
    return None

def generate_with_visualization(model, idx, max_new_tokens, temperature=0.8, top_k=200, style='heatmap'):
    """带可视化的生成函数"""
    visualizer = TextGenerationVisualizer(decode, style)
    generated_tokens = idx[0].tolist()

    for step in range(max_new_tokens):
        # 如果序列太长，截断
        if idx.size(1) <= model.config.block_size:
            logits, _ = model(idx)
        else:
            logits, _ = model(idx[:, -model.config.block_size:])

        # 获取最后一个时间步的 logits
        logits = logits[:, -1, :] / temperature

        # 应用 top-k 采样
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 计算概率
        probs = torch.softmax(logits, dim=-1)

        # 采样下一个 token
        next_token = torch.multinomial(probs, num_samples=1)

        # 获取所有 token 的概率分布用于可视化
        token_probs = probs[0].cpu().numpy()
        selected_token = next_token[0].item()

        # 更新可视化
        visualizer.update_visualization(token_probs, selected_token, step)

        # 添加新 token 到序列
        generated_tokens.append(selected_token)
        idx = torch.cat((idx, next_token), dim=1)

        # 可选：每 N 步显示一次信息
        if step % 50 == 0 and step > 0:
            print(f"\n生成进度: {step}/{max_new_tokens} 步")

    visualizer.close()
    return idx

# -----------------------------------------------------------------------------
# 主程序开始
print("正在初始化...")
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 加载模型
print("加载模型...")
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到模型文件 {ckpt_path}")
        print("请先训练模型或指定正确的 out_dir")
        exit(1)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# 加载编码器
print("加载编码器...")
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"从 {meta_path} 加载编码器...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("使用 GPT-2 编码器...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# 编码提示词
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print(f"\n开始生成 {num_samples} 个样本...")
print(f"提示词: {start}")
print(f"可视化风格: {visualization_style}")
print("=" * 50)

# 生成样本
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            print(f"\n样本 {k+1}/{num_samples}:")
            print("-" * 40)

            if visualize:
                # 使用带可视化的生成
                y = generate_with_visualization(model, x, max_new_tokens,
                                               temperature=temperature, top_k=top_k,
                                               style=visualization_style)
                generated_text = decode(y[0].tolist())
                print(generated_text)
            else:
                # 使用原始生成
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                generated_text = decode(y[0].tolist())
                print(generated_text)

            print('-' * 40)
            print()

print("生成完成！")

# 保存生成结果
output_file = os.path.join(out_dir, 'generated_samples.txt')
os.makedirs(out_dir, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"提示词: {start}\n")
    f.write(f"=" * 50 + "\n")
    f.write(f"生成样本数: {num_samples}\n")
    f.write(f"最大token数: {max_new_tokens}\n")
    f.write(f"温度: {temperature}\n")
    f.write(f"Top-K: {top_k}\n")
    f.write("=" * 50 + "\n\n")

print(f"\n生成结果已保存到: {output_file}")