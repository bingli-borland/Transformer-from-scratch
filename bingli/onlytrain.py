import os

import requests
import tiktoken
import torch

from bingli.model import Model

# 超参数
batch_size = 4  # 每个训练步骤有多少批
context_length = 16  # 每批文本的长度
learning_rate = 1e-3  # 0.001 学习率
max_iters = 5000  # 训练迭代总数<-将其更改为较小的数字用于测试
eval_iters = 20  # 计算的平均迭代数
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU，如果它是可用的
TORCH_SEED = 1337  # 随机种子
torch.manual_seed(TORCH_SEED)


# 加载训练数据
if not os.path.exists('../data/sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    # 如果本地不存在数据文件，从指定的url下载数据并保存到本地文件
    with open('../data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)
# 从本地文件中读取文本数据
with open('../data/sales_textbook.txt', 'r', encoding='utf-8') as f:
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

# Initialize the model
model = Model()
model = model.to(device)


# Get input embedding batch
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


# Calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
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
torch.save(model.state_dict(), 'bingli-model-ckpt.pt')  # 将模型的状态字典保存到名为 'model-ckpt.pt' 的文件中。这个文件包含了模型的所有参数权重，可以用来在之后重新加载模型