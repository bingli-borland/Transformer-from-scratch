import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import tiktoken
import pandas as pd
import math

# 读取数据
if not os.path.exists('/data/sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/blob/main/sales_textbook.txt?download=true'
    with open('sales_textbook.txt', 'wb') as f:
        f.write(requests.get(url).content)

with open('sales_textbook.txt', 'r') as f:
    text = f.read()

# 超参数
batch_size = 4  # batch_size数据批次
context_length = 16  # context_length截取长度
d_model = 64  # d_model维度（官方为512）
num_heads = 4  # num_heads头数

# 使用tiktoken对文本进行编码，以便于后续的模型训练
encoding = tiktoken.get_encoding("cl100k_base")  # cl100k_base是tiktoken的预训练模型，可以直接使用，不需要自己训练，速度更快，准确度更高
tokenized_text = encoding.encode(text)  # 将文本进行编码
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)  # 将编码后的文本转化为张量
max_token_value = tokenized_text.max().item()  # 获取文本中最大的token值

# 划分训练集和验证集，90%作为训练集，10%作为验证集，以此类推，直到训练集和验证集的总长度相等，才停止，避免过拟合，增加模型的泛化能力，提高准确率，降低模型的过拟合。
train_idex = int(len(tokenized_text) * 0.9)  # 90%作为训练集，10%作为验证集
train_data = tokenized_text[:train_idex]  # 划分训练集，从0到train_idex，作为训练集，train_data维度为(train_idex, )，表示训练集
valid_data = tokenized_text[train_idex:]  # 划分验证集，从train_idex到len(tokenized_text),作为验证集

# 数据预处理，将数据转化为模型输入，每个数据包含context_length个token，预测下一个token，每个数据包含context_length个token
data = train_data
idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))  # 随机选取batch_size个数据
x_batch = torch.stack(
    [data[idx: idx + context_length] for idx in idxs])  # 根据idxs选取数据，并拼接，作为模型输入，x_batch维度为(batch_size, context_length)
y_batch = torch.stack([data[idx + 1: idx + context_length + 1] for idx in
                       idxs])  # 根据idxs选取数据，并拼接，作为模型输入，y_batch维度为(batch_size, context_length)

# 定义输入嵌入表
input_embeding_lookup_table = nn.Embedding(max_token_value + 1,
                                           d_model)  # 输入嵌入表，d_model维度，max_token_value + 1表示文本中最大的token值
x_batch_embedding = input_embeding_lookup_table(x_batch)  # 从输入嵌入表中获取x_batch输入嵌入 torch.Size([4, 16, 64])
y_batch_embedding = input_embeding_lookup_table(y_batch)  # 从输入嵌入表中获取y_batch输入嵌入 torch.Size([4, 16, 64])

# 获取位置编码
position_encoding_lookup_table = torch.zeros(context_length, d_model)  # torch.Size([16, 64])位置编码表（16 x 64 全0形状矩阵）
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(
    1)  # tensor([0, 1, 2, 3, 4, 5....15])，数据类型torch.float32,torch.arange(start=0, end=context_length)的结果并不包含context_length，0-15，不包括16
# 应用sin和cos函数，计算位置编码
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(
    10000.0) / d_model))  # PE(2i) = sin(pos / 10000^(2i/d_model))，PE(2i+1) = cos(pos / 10000^(2i/d_model))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)  # 偶数位，sin
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)  # 奇数位，cos
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, -1,
                                                                                    -1)  # 添加批次维度，扩展维度为(batch_size, 16, 64)，方便与x_batch_embedding和y_batch_embedding相加
# 添加位置编码，以便于后续的模型训练
x = x_batch_embedding + position_encoding_lookup_table  # 产生位置编码后的x_batch,共有4个批次，每个批次16个样本，每个样本64个token
y = y_batch_embedding + position_encoding_lookup_table  # 产生位置编码后的y_batch,共有4个批次，每个批次16个样本，每个样本64个token

# 获取Q, K, V
Wq = nn.Linear(d_model, d_model)  # Q层
Wk = nn.Linear(d_model, d_model)  # K层
Wv = nn.Linear(d_model, d_model)  # V层

Q = Wq(x)  # Q层
K = Wk(x)  # K层
V = Wv(x)  # V层

# 多头注意力计算，4个头和64个token进行计算，第个头的维度就是d_model // num_heads，切换以后就成了[4, 16, 4, 16]
Q = Q.reshape(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1,
                                                                                   3)  # Q维度变换 torch.Size([4, 16, 4, 16])to([4, 4, 16, 16])
K = K.reshape(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1,
                                                                                   3)  # K维度变换 torch.Size([4, 16, 4, 16])to([4, 4, 16, 16])
V = V.reshape(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1,
                                                                                   3)  # V维度变换 torch.Size([4, 16, 4, 16])to([4, 4, 16, 16])
# [4个批次， 4个头， 16个字符， 64/4=16个维度] 这里的4是指的那4个头，每个头包括16个维度，一共64个维度
# 为什么要旋转后再进行点积运算？因为两个矩阵在不加上批次的情况都是16个token与64个维度下的矩阵，如果不旋转进行点积运算就变成了token与维度的关链性，所以我们需要将K旋转90度，这样运算得到了token与token的关链性。
# Q与K的点积相乘
output = Q @ K.transpose(-2, -1) / math.sqrt(d_model // num_heads)  # 计算QKt, K.transpose是对K进行90度矩阵旋转

# 生成遮罩
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()  # 生成上三角矩阵，torch.Size([16, 16])，所有值都为1
output = output.masked_fill(mask, float('-inf'))  # 斜切矩阵，将上三角矩阵中的值置为-inf,0或无限大

# softmax计算
attention_score = F.softmax(output, dim=-1)  # 将概率数值转换为0至1之间的百分比

# apply attention @ V
A = attention_score @ V

# apply concatenate
A = A.transpose(1, 2).reshape(batch_size, -1, d_model)  # A维度变换 torch.Size([4, 16, 64])回到初始的维度， -1为自动计算
Wo = nn.Linear(d_model, d_model)  # 输出层
output = Wo(A)  # 输出层

# apply residual connection
output = output + x  # 残差连接，最终结果加上原始的x值

# apply layer normalization 输入应用层归一化
layer_norm = nn.LayerNorm(d_model)  # layer norm
layer_norm_output = layer_norm(output)  # layer norm

# apply feed forward network 前馈网络
output = nn.Linear(d_model, d_model * 4)(layer_norm_output)  # 输出的张量在维度上放大4倍
output = nn.ReLU()(output)  # ReLU激活函数，把小于0的值变成0
output = nn.Linear(d_model * 4, d_model)(output)  # 再把维度缩回原始维度

# 再次进行残差连接
output = output + layer_norm_output

# 再次进行层归化
output = layer_norm(output)  # layer norm

# apply final liner layer 线性变换，与之前不同，其维度为整个词汇表的维度，即max_token_value + 1
output = nn.Linear(d_model, max_token_value + 1)(output)  # 线性计算作用于output(torch.Size([4, 16, 99088]))
logits = F.softmax(output, dim=-1)  # softmax计算,最终的概率值

predicted_index = torch.argmax(logits[0, 0]).item()  # 获取最大概率对应的索引
print(encoding.decode([5487, predicted_index]))
print(output.shape)