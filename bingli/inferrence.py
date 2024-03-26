import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from model import TransformerLanguageModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED=1337
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

encoding = tiktoken.get_encoding("cl100k_base")

model = TransformerLanguageModel()
model.load_state_dict(torch.load('model-ckpt1.pt'))
model.eval()
model.to(device)

# 计算模型中的默认参数个数
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("模型中的默认参数个数：", total_params)

start = 'The salesperson'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    y = model.generate(x, max_new_tokens=100)
    print('---------------')
    print(encoding.decode(y[0].tolist()))
    print('---------------')
