import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 4  # 每个训练步骤有多少批
context_length = 16  # 每批文本的长度
d_model = 64  # 模型的维度
num_blocks = 8  # transformer blocks循环次数
num_heads = 4  # 多头注意力的头数
learning_rate = 1e-3  # 0.001 学习率
dropout = 0.1  # 最低概率，放弃学习率下限
max_iters = 5000  # 训练迭代总数<-将其更改为较小的数字用于测试
eval_interval = 50  # 多长时间计算一次
eval_iters = 20  # 计算的平均迭代数
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU，如果它是可用的
TORCH_SEED = 1337  # 随机种子
torch.manual_seed(TORCH_SEED)

# 加载训练数据
if not os.path.exists('data/sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    # 如果本地不存在数据文件，从指定的url下载数据并保存到本地文件
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)
# 从本地文件中读取文本数据
with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 使用TikToken(与GPT3相同)来标记源文本
encoding = tiktoken.get_encoding("cl100k_base")  # 使用预训练的 cl100k_base 模型进行文本编码，得到 encoding 对象
tokenized_text = encoding.encode(text)  # 使用得到的 encoding 对象将文本进行编码，得到标记化的文本
max_token_value = max(tokenized_text) + 1  # 标记token的最大值,即词典的整体长度
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long,
                              device=device)  # 将标记文本转换为 PyTorch 张量，并放置在指定的设备上（GPU或CPU）

# 分割训练和验证
split_idx = int(len(tokenized_text) * 0.9)  # 计算划分训练集和验证集的索引，将90%的数据用于训练，10%用于验证
train_data = tokenized_text[:split_idx]  # 切片操作，获取训练集数据
val_data = tokenized_text[split_idx:]  # 切片操作，获取验证集数据


# 定义前馈网络 Define Feed Forward Network
class FeedForward(nn.Module):  # 定义一个名为 FeedForward 的类，继承自 PyTorch 中的 nn.Module 类
    def __init__(self):
        super().__init__()
        self.d_model = d_model  # 将超参数 d_model 赋值给类属性 d_model
        self.dropout = dropout  # 将超参数 dropout 赋值给类属性 dropout
        #  使用 PyTorch 中的 nn.Sequential 定义一个前馈网络，包括线性层、ReLU激活函数和Dropout层
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),  # 输出的张量在维度上放大4倍
            nn.ReLU(),  # ReLU激活函数，把小于0的值变成0
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),  # 再把维度缩回原始维度
            nn.Dropout(dropout),  # 使用 Dropout 层，随机丢弃一定概率的神经元，防止过拟合
        )

    # 默认方法，定义前向传播方法，接收输入 x，并返回经过前馈网络处理后的结果
    def forward(self, x):
        return self.ffn(x)  # 将输入 x(类型为 PyTorch 张量) 传递给前馈网络，并返回处理后的结果


# 定义缩放点积注意力 Define Scaled Dot Product Attention
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        # 使用线性层定义注意力机制的键、查询和值的映射
        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size,
                                   bias=False)  # 定义键映射层，使用线性变换将输入的维度从 d_model 转换为 head_size
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size,
                                     bias=False)  # 定义查询映射层，使用线性变换将输入的维度从 d_model 转换为 head_size
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size,
                                     bias=False)  # 定义值映射层，使用线性变换将输入的维度从 d_model 转换为 head_size
        # 生成一个下三角矩阵，用于生成遮罩
        self.register_buffer('tril', torch.tril(
            torch.ones((self.context_length, self.context_length))))  # Lower triangular mask
        self.dropout_layer = nn.Dropout(self.dropout)  # 定义一个 Dropout 层，用于在计算注意力时进行随机丢弃

    # 定义前向传播方法，接收输入 x，并返回通过注意力机制处理后的结果
    def forward(self, x):
        B, T, C = x.shape  # 获取输入 x 的形状信息，分别表示 Batch size、时间步数（当前上下文长度）和通道数（维度）
        assert T <= self.context_length  # 断言，确保输入的时间步数不超过设定的最大上下文长度 self.context_length
        assert C == self.d_model  # 断言，确保输入的通道数（维度）与模型的维度 self.d_model 一致
        q = self.query_layer(x)  # 通过查询映射层，将输入 x 映射为查询向量 q
        k = self.key_layer(x)  # 通过键映射层，将输入 x 映射为键向量 k。
        v = self.value_layer(x)  # 通过值映射层，将输入 x 映射为值向量 v

        # 计算注意力权重，即 scaled dot product attention 公式: Q @ K^T / sqrt(d_k)
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 应用遮罩，将上三角矩阵中的值置为 -inf，使得注意力只关注当前和之前的位置
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 斜切矩阵，将上三角矩阵中的值置为-inf,0或无限大
        weights = F.softmax(input=weights, dim=-1)  # 使用 softmax 函数将注意力权重转换为概率分布
        weights = self.dropout_layer(weights)  # 应用 Dropout，随机丢弃一定比例的注意力权重。

        # 应用 dot product attention，得到最终的输出: weights @ V （计算注意力权重（weights）和值向量（v）的点积）权重表示模型对输入序列中各个位置的关注程度，而值向量则是对应位置的特征表示
        out = weights @ v
        # 注意力机制加权得到的最终输出。模型对输入序列的关注程度，并根据这种关注程度计算得到的加权和。
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        # 创建多头注意力机制
        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        # 线性变换层，用于将多头注意力的结果映射回原始维度
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        # Dropout 层，防止过拟合
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # 对每个注意力头的输出进行拼接
        out = self.projection_layer(out)  # 通过线性变换映射回原始维度
        out = self.dropout_layer(out)  # 应用 Dropout
        return out  # 经过多头注意力和线性变换后的最终输出后进行前向传播


class TransformerBlock(nn.Module):

    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        # 设置注意力头的大小，确保它可以被 d_model 整除。这是因为在多头注意力机制中，输入的维度 d_model 应该均匀分配给多个注意力头
        self.head_size = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        # 多头注意力层
        self.multi_head_attention_layer = MultiHeadAttention(
            head_size=self.head_size)  # 创建一个多头注意力层，用于 Transformer 模块中进行多头注意力的操作。head_size 参数指定注意力头的大小。MultiHeadAttention 是你定义的多头注意力的类。
        # 前馈网络层
        self.feed_forward_layer = FeedForward()  # 创建一个前馈神经网络层，用于 Transformer 模块中进行非线性变换。FeedForward 是你定义的前馈网络的类
        # Layer Normalization 层，用于在每个子层的输出上应用标准化
        self.layer_norm_1 = nn.LayerNorm(
            normalized_shape=self.d_model)  # 创建一个 Layer Normalization 层，用于在 Transformer 模块的多头注意力层之前进行归一化
        self.layer_norm_2 = nn.LayerNorm(
            normalized_shape=self.d_model)  # 创建另一个 Layer Normalization 层，用于在 Transformer 模块的前馈神经网络层之前进行归一化

    def forward(self, x):
        # 多头注意力层的前向传播
        # 注意：操作的顺序与原始的 Transformer 论文不同
        # 这里的顺序是：LayerNorm -> Multi-head attention -> LayerNorm -> Feed forward
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))  # 残差连接
        x = x + self.feed_forward_layer(self.layer_norm_2(x))  # 残差连接
        return x


# 语言模型，用于生成文本或预测下一个可能的单词
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value
        # 设置标记嵌入查找表
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1,
                                                         embedding_dim=self.d_model)  # 定义标记嵌入查找表，将标记映射到 d_model 维度的嵌入向量。

        # 运行所有 Transformer 块
        # 与原论文不同，这里在所有块之后添加了最终的 LayerNorm
        self.transformer_blocks = nn.Sequential(*(
            # 创建多个 Transformer 块，数量由超参数 num_blocks 决定
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                # 添加最终的 LayerNorm 层
                [nn.LayerNorm(self.d_model)]
        ))
        #  定义语言模型输出的线性层，将 d_model 维度的输出映射为标记的维度
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # 获取输入 idx 的形状信息：Batch size, Time steps (sequence length)

        # 设置位置嵌入查找表，采用原始Transformer论文中的方法（正弦和余弦函数）
        # 创建一个形状为 (context_length, d_model) 的零张量，用于存储位置编码
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        # 创建一个包含位置信息的张量，用于计算位置编码
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        # 计算一个除法项，用于位置编码中的除法运算
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        # 计算位置编码中奇数索引位置的 sin 值
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        # 计算位置编码中偶数索引位置的 cos 值
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)

        # 将 position_encoding_lookup_table 从 (context_length, d_model) 转换为 (T, d_model)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)  # 获取与输入序列长度相匹配的位置嵌入，并将其移动到相同的设备（GPU或CPU）。
        # 获取标记的嵌入向量，并加上位置嵌入,得到输入序列的嵌入表示
        x = self.token_embedding_lookup_table(idx) + position_embedding  # 示例图中的那三份X，向Q\K\V传播的样本数据
        # 通过所有 Transformer 块的前向传播,处理输入序列的嵌入表示
        x = self.transformer_blocks(x)  # 开始进入我们一层一层的训练，每一层都是一个Transformer块
        # "logits" 是我们模型在应用 softmax 之前的输出值
        logits = self.language_model_out_linear_layer(x)  # 仅是权重，还需要做softmax处理转换为概率

        # 设置损失函数参数，转换模型输出 logits 与目标标签 targets 的输出形状
        if targets is not None:
            B, T, C = logits.shape
            # 将 logits 从形状 (B, T, C) 转换为 (B * T, C) 批次大小 * 序列长度 , 维度
            logits_reshaped = logits.view(B * T, C)
            # 将 targets 从形状 (B, T) 转换为 (B * T) 批次大小 * 序列长度，并没有维度要求
            targets_reshaped = targets.view(B * T)

            # 使用交叉熵损失函数计算预测 logits logits_reshaped 和目标标签 targets_reshaped 之间的损失
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)  # 运算后为softmax概率值，要求参数为输入和输出的形状匹配
        else:
            loss = None
        # 返回预测的 logits 和计算得到的损失值
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是当前上下文中的索引数组，形状为 (B, T)
        for _ in range(max_new_tokens):  # 对于每个 max_new_tokens，循环生成新的标记，直到生成 max_new_tokens 个新标记
            # 将 idx 裁剪到我们位置嵌入表的最大大小
            idx_crop = idx[:, -self.context_length:]
            # 获取预测，通过调用模型的 self(idx_crop) 方法获取预测的 logits
            logits, loss = self(idx_crop)
            # 获取 logits 中最后一个时间步的预测，维度为 (B, C)
            logits_last_timestep = logits[:, -1, :]
            # 应用 softmax 获取概率分布得到每个词的概率
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # 从概率分布中采样，使用 torch.multinomial 从概率分布中采样，得到下一个预测的索引 idx_next
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将采样的索引 idx_next 追加到输入序列 idx 的末尾
            idx = torch.cat((idx, idx_next), dim=1)
        # 最终，方法返回包含生成文本的索引数组的 idx
        return idx


# 初始化模型
model = TransformerLanguageModel()
# 将模型移动到指定设备（GPU或CPU）
model = model.to(device)


# 获取输入嵌入批次
def get_batch(split: str):
    data = train_data if split == 'train' else val_data  # 根据指定的数据集类型（训练集或验证集），选择相应的数据集赋值给变量 data。如果 split 为 'train'，则选择训练集 train_data；否则选择验证集 val_data。这是用于获取训练和验证数据的一部分
    # 从数据中随机选择索引
    idxs = torch.randint(low=0, high=len(data) - context_length,
                         size=(batch_size,))  # 生成一个包含 batch_size 个随机整数的张量，这些整数表示从数据中选择的起始索引。这些索引用于构造训练批次
    # 构造输入序列 x 和目标序列 y
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(
        device)  # 从数据中根据随机选择的索引 idxs 中构造输入序列 x，并将其移动到指定的设备上。
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(
        device)  # 从数据中构造目标序列 y，其每个元素都是对应输入序列 x 中对应位置的下一个元素，并将其移动到指定的设备上
    # 函数就返回了一个包含输入序列 x 和目标序列 y 的数据批次用于模型的训练和验证过程
    return x, y


# 计算损失
# @torch.no_grad(): 这是一个装饰器（Decorator），用于在函数执行时禁用梯度计算。在 estimate_loss 函数中使用这个装饰器，是为了确保在计算损失时不会影响模型的梯度，因为这个函数只是用来估算损失而不进行梯度更新
@torch.no_grad()
def estimate_loss():
    out = {}  # 初始化一个空的字典，用于存储损失结果。
    model.eval()  # 将模型设置为评估模式。在评估模式下，模型中的一些特定层（如 Dropout）行为会有所不同，以便在推断时得到稳定的结果
    # 对训练集和验证集分别计算损失
    for split in ['train', 'valid']:  # 遍历训练集和验证集
        losses = torch.zeros(eval_iters)  # 初始化一个长度为 eval_iters 的零张量，用于存储每个批次的损失值
        # 对多个批次计算平均损失，遍历多个批次
        for k in range(eval_iters):
            # 获取当前批次的输入序列 x_batch 和目标序列 y_batch
            x_batch, y_batch = get_batch(split)
            # 使用模型进行前向传播，得到预测 logits 和损失值
            logits, loss = model(x_batch, y_batch)
            # 将当前批次的损失值存储在 losses 张量的第 k 个位置
            losses[k] = loss.item()
        # 计算当前数据集（训练集或验证集）的平均损失，并存储在字典 out 中，使用数据集名称 split 作为键
        out[split] = losses.mean()

    # 恢复模型为训练模式
    model.train()  # 将模型设置为训练模式。在训练模式下，模型中的一些特定层（如 Dropout）行为会有所不同，以便在训练时引入随机性，帮助防止过拟合
    # 将存储着训练集和验证集平均损失的字典 out 返回给调用者
    return out


# 使用 AdamW 优化器
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)  # 对模型参数进行优化，学习率为 learning_rate
# 用于跟踪损失的列表
tracked_losses = list()
# 训练循环
for step in range(max_iters):  # 训练循环，迭代 max_iters 次
    if step % eval_iters == 0 or step == max_iters - 1:  # 每隔 eval_iters 步或在最后一步时，进行一次损失估算并记录。
        # 估算损失并记录
        losses = estimate_loss()  # 调用 estimate_loss 函数估算训练集和验证集的损失，并将结果存储在 losses 变量中
        tracked_losses.append(losses)  # 将损失结果添加到用于跟踪损失的列表中
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))  # 打印当前训练步骤的训练损失和验证损失
    # 获取训练批次
    xb, yb = get_batch('train')  # 获取训练集的输入序列 xb 和目标序列 yb，用于模型的训练
    # 模型前向传播和计算损失
    logits, loss = model(xb, yb)  # 将训练集的输入序列 xb 和目标序列 yb 传递给模型进行前向传播，得到预测的 logits 和计算的损失值
    # 梯度清零
    optimizer.zero_grad(set_to_none=True)  # set_to_none=True 表示将梯度张量设置为 None，以便更高效地释放梯度内存
    # 反向传播
    loss.backward()  # 计算梯度。此操作会将梯度信息传播到模型的参数
    # 参数更新
    optimizer.step()  # 根据梯度更新模型参数。这是优化器执行梯度下降步骤的操作，使模型逐渐收敛到损失函数的最小值

# 保存模型的状态字典
torch.save(model.state_dict(), 'model-ckpt.pt')  # 将模型的状态字典保存到名为 'model-ckpt.pt' 的文件中。这个文件包含了模型的所有参数权重，可以用来在之后重新加载模型

# 生成文本
model.eval()  # 将模型设置为评估模式，以确保在生成文本时不引入额外的随机性

# 设置初始文本
start = 'The salesperson'
start_ids = encoding.encode(start)  # 将初始文本编码为索引序列
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])  # 将编码后的序列转换为张量，并添加额外的维度以匹配模型的输入形状
y = model.generate(x, max_new_tokens=100)  # 使用模型生成新文本

# 打印生成的文本
print('---------------')
print(encoding.decode(y[0].tolist()))  # 将生成的文本解码并打印出来
print('---------------')