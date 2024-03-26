import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
context_length = 128  # 上下文长度
d_model = 512  # 模型维度
num_blocks = 12  # Transformer块的数量
num_heads = 8  # 多头注意力机制中的头数
dropout = 0.1  # 丢弃率，是一种正则化技术，它在训练过程中以一定的概率随机将神经元的输出置零，从而减小模型对于训练数据的过拟合风险
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备： 此参数确定模型是否应在GPU上进行训练和运行（'cuda'），如果可用的话，或者在CPU上进行训练和运行（'cpu'）。
TORCH_SEED = 1337  # PyTorch的随机种子，这是用于初始化PyTorch的随机数生成器的种子值。
torch.manual_seed(TORCH_SEED)  # 设置种子可确保结果的可重复性，即如果使用相同的种子运行代码，每次都应该获得相同的结果。


# 定义前馈神经网络（FeedForwardNetwork）的类，该网络包含两个线性层和激活函数
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 nn.Sequential 定义了一个包含两个线性层（全连接层）和一个激活函数的序列
        self.ffn = nn.Sequential(
            # 第一个线性层 (nn.Linear(d_model, d_model * 4))：输入特征维度为 d_model，输出特征维度为 d_model * 4
            nn.Linear(d_model, d_model * 4),
            # ReLU 激活函数 (nn.ReLU())：应用于线性层的输出
            nn.ReLU(),
            # 第二个线性层 (nn.Linear(d_model * 4, d_model))：输入特征维度为 d_model * 4，输出特征维度为 d_model
            nn.Linear(d_model * 4, d_model),
            # Dropout 层 (nn.Dropout(dropout))：应用丢弃率为 dropout 的 dropout 操作
            nn.Dropout(dropout)
        )

    # 前向传播函数 (forward) 接受输入张量 x，进行前馈神经网络的前向传播，并返回输出张量
    def forward(self, x):
        # 输入 x 经过前馈神经网络 self.ffn，其输出即为整个前馈神经网络的输出
        return self.ffn(x)


# 定义 Scaled Dot Product Attention（缩放点积注意力）的类，该注意力层可以被嵌入到 Transformer 模型中
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # self.Wq, self.Wk, self.Wv: 分别是用于计算查询、键、值的线性层。这些层将输入的维度 d_model 映射到 d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model // num_heads, bias=False)
        self.Wk = nn.Linear(d_model, d_model // num_heads, bias=False)
        self.Wv = nn.Linear(d_model, d_model // num_heads, bias=False)
        # 创建一个下三角矩阵作为掩码（mask），用于在计算注意力权重时屏蔽未来的信息。这个掩码被注册为模型的缓冲区，以确保它在模型的状态字典中被持久化
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))
        # 用于添加 dropout 操作的 Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        获取输入张量 x 的形状信息
        B: 表示批次大小（Batch Size），即输入张量中有多少个样本。
        T: 表示序列长度（Time Steps），即输入序列的长度或时间步数。
        C: 表示通道数（Channels），即输入张量中每个时间步的特征维度。
        """
        B, T, C = x.shape
        # q, k, v: 分别是通过线性层计算的查询、键、值
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # Scaled Dot Product Attention 的权重计算过程
        # 缩放的点积注意力的计算。通过将查询（q）与键（k）的转置相乘，然后除以缩放因子
        weights = q @ k.transpose(-2, -1) / math.sqrt(d_model // num_heads)
        # 引入了一个下三角形的掩码（mask），将未来的信息屏蔽掉。这里使用了 PyTorch 的 masked_fill 函数，将 mask 中值为 0 的位置替换成负无穷（float('-inf')）
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        # 对掩码后的注意力权重矩阵进行 softmax 操作，将其转换为概率分布。这确保了权重的归一化，使得它们的和等于 1
        weights = F.softmax(weights, dim=-1)
        # 应用了 dropout 操作。这一步有助于防止模型对噪声的过度敏感，提高泛化性能。
        weights = self.dropout(weights)
        # 注意力权重矩阵 weights 与值矩阵 v 相乘，得到最终的输出矩阵。每个位置的输出向量是对应位置的值（v）加权求和的结果，权重由注意力机制计算得出
        output = weights @ v
        # 返回经过注意力机制加权的值
        return output


# 定义 Multi-Head Attention（多头注意力）的类,该类包含多个 Attention 层
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # self.heads: 是一个 nn.ModuleList，包含了多个 Attention 层，这些层被认为是“头”（heads）。通过使用列表推导式，创建了 num_heads 个 Attention 头。
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        # self.projection_layer: 是一个线性层，用于将多个 Attention 头的输出进行投影，将它们连接到一个更低维度的表示。
        self.projection_layer = nn.Linear(d_model, d_model)
        # 用于添加 dropout 操作的 Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # head_outputs: 对每个 Attention 头执行前向传播，得到头的输出组成的列表
        head_outputs = [head(x) for head in self.heads]
        # 将多个 Attention 头的输出在最后一个维度上连接，形成一个更大的张量
        head_outputs = torch.cat(head_outputs, dim=-1)
        # 将连接后的张量通过投影层，将维度减小回原始的 d_model。并应用 dropout 操作
        out = self.dropout(self.projection_layer(head_outputs))
        # 最后返回最终输出
        return out


# 定义 Transformer 模型中的基本模块，模块包含了层归一化（Layer Normalization）、多头注意力（Multi-Head Attention）和前馈神经网络（FeedForwardNetwork）
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ln1, self.ln2: 两个层归一化层，分别应用于多头注意力和前馈神经网络的输入。这有助于稳定训练过程，加速模型收敛
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # self.mha: 多头注意力层的实例，用于捕捉输入序列中的不同关系
        self.mha = MultiHeadAttention()
        # self.ffn: 前馈神经网络层的实例，引入非线性变换
        self.ffn = FeedForwardNetwork()

    def forward(self, x):
        # 输入 x 先经过层归一化，然后通过多头注意力层，最后将原始输入 x 与注意力层的输出相加。这是 Transformer 中的残差连接
        x = x + self.mha(self.ln1(x))
        #  上一步的输出再次经过层归一化，然后通过前馈神经网络，最后将原始输入 x 与前馈神经网络的输出相加
        x = x + self.ffn(self.ln2(x))
        # 模块是 Transformer 模型的基础单元，在整个模型中可以被堆叠多次以形成更深层次的架构。每个模块通过残差连接的方式，有助于梯度在训练中的流动，提高模型的训练效果。
        # 每个模块的输出作为下一个模块的输入
        return x


# 基于 Transformer 架构的模型类
class Model(nn.Module):
    # 模型的嵌入层（self.token_embedding_table）的输入，即输入令牌的最大索引值。在自然语言处理中，这通常对应于词汇表的大小，表示模型学习的嵌入向量的数量
    # def __init__(self, max_token_value=100080):
    def __init__(self, max_token_value=100080):

        super().__init__()

        # token_embedding_lookup_table 是一个嵌入层（Embedding Layer），用于将输入的离散令牌映射为对应的 d_model 维度的向量。这个嵌入层的权重矩阵会被训练学习。
        self.token_embedding_lookup_table = nn.Embedding(max_token_value, d_model)
        """
        self.transformer_blocks 是一个包含多个 TransformerBlock 的序列，通过 nn.Sequential 将多个 TransformerBlock 串联在一起，构成整个模型的主体部分。最后添加了一个层归一化层
        [TransformerBlock() for _ in range(num_blocks)]: 使用列表推导式创建了一个包含 num_blocks 个 TransformerBlock 实例的列表。这是模型中多个 Transformer 模块的堆叠。
        [nn.LayerNorm(d_model)]: 创建了一个包含一个层归一化层的列表，这个层归一化层会被添加到 Transformer 模块序列的最后。这一步是为了在模型的每个 TransformerBlock 之后添加一个层归一化操作，以进一步稳定训练过程。
        nn.Sequential(*(列表)): 使用 nn.Sequential 将上述两个列表中的实例串联在一起，形成一个序列。* 操作符用于将列表中的实例展开作为 nn.Sequential 的参数。
        """
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock() for _ in range(num_blocks)] +
                [nn.LayerNorm(d_model)]
        ))
        # 线性层，将 Transformer 模型的输出映射为最终的预测结果，其输出维度为 max_token_value，用于模型在离散输出任务上的预测
        self.model_out_linear_layer = nn.Linear(d_model, max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # 创建了一个位置编码查找表，用于生成位置编码。这个表的大小为 (context_length, d_model)
        position_encoding_lookup_table = torch.zeros(context_length, d_model, device=device)
        # 创建了一个表示位置的张量
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        # 计算了位置编码中的分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 使用 sine 和 cosine 函数生成位置编码的值
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        # 截取 position_encoding_lookup_table 来适应输入序列的长度，从而生成正确形状的位置编码（Position Encoding）
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        # 将输入令牌通过嵌入层映射为词嵌入，然后加上位置编码
        x = self.token_embedding_lookup_table(idx) + position_embedding
        # 将输入序列传递给 Transformer 模块进行处理
        x = self.transformer_blocks(x)
        # 将 Transformer 模块的输出通过线性层映射为最终的预测结果，即模型的 logits
        logits = self.model_out_linear_layer(x)

        # 用于计算损失的逻辑，根据是否提供了目标标签 targets 来判断是否需要计算损失
        if targets is not None:
            # 取 logits 的形状信息，其中 B 表示批次大小，T 表示序列长度，C 表示类别数量
            B, T, C = logits.shape
            # 将 logits 的形状调整为二维，以便与目标标签匹配。这是为了使用 PyTorch 的交叉熵损失函数 F.cross_entropy，它要求输入是二维的 logits
            logits_reshape = logits.view(B * T, C)
            # 将目标标签也调整为一维，以便与 logits 的形状匹配
            targets_reshape = targets.view(B * T)
            # 使用 PyTorch 的交叉熵损失函数计算损失。这个函数比较了模型的预测 logits 和实际的目标标签，通过对数概率的差异来计算损失
            loss = F.cross_entropy(input=logits_reshape, target=targets_reshape)
        else:
            # 如果未提供目标标签，则将损失设置为 None
            loss = None
        # 返回模型的 logits 和相应的损失值（如果计算了损失）。在训练过程中，优化器将利用损失值来调整模型的参数，以提高模型在任务上的性能
        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #  截取输入序列 idx 的最后 context_length 个令牌，确保输入的长度不超过模型的最大位置编码表的大小
            idx_crop = idx[:, -context_length:]
            # 使用模型的前向传播方法，得到对输入序列的预测
            logits, loss = self.forward(idx_crop)
            # 从 logits 中获取每个样本的最后一个时间步的预测结果
            logits_last_timestep = logits[:, -1, :]
            # 对最后一个时间步的 logits 应用 softmax 操作，得到对应于每个类别的概率分布
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # 从概率分布中使用多项式分布进行抽样，得到下一个生成的令牌的索引
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将生成的令牌索引追加到输入序列中，形成新的输入序列
            idx = torch.cat((idx, idx_next), dim=1)
        # 循环过程中每次生成一个新的令牌，最多生成 max_new_tokens 个令牌。最终，返回更新后的输入序列 idx，其中包含了生成的新令牌
        return idx