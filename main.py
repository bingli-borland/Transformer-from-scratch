from fastapi import FastAPI
import uvicorn
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from model import TransformerLanguageModel
from transformers import pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED=1337
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

encoding = tiktoken.get_encoding("cl100k_base")

model = TransformerLanguageModel()
model.load_state_dict(torch.load('model-ckpt1.pt'))
model.eval()
model.to(device)

app = FastAPI(title="超级模型开发能力平台")

question_answerer = pipeline('question-answering')
classifier = pipeline('sentiment-analysis')

@app.get("/gpt", summary='对话', tags=['NLP方向'])
def gpt(text: str = None):
    start_ids = encoding.encode(text)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=100)
        result = encoding.decode(y[0].tolist())
        return {"code": 200, "result": result}

@app.get("/sentiment", summary='情绪分析', tags=['NLP方向'])
def qa(text: str = None):
    result = classifier(text)
    result = result[0]['label'] # 解析结果，只要需要的
    return {"code": 200, "result": result}

@app.get("/qa", summary='文本问答', tags=['NLP方向'])
def qa(text: str = None, q_text: str = None):
    result = question_answerer({'question': q_text, 'context': text})
    result = result['answer'] # 解析结果，只要需要的
    return {"code": 200, "result": result}

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8000)