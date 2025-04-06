import torch

model_name = "gpt2"  # 修改为 GPT-2 模型
max_length = 1024  # GPT-2 的上下文窗口大小
device = "cuda" if torch.cuda.is_available() else "cpu"
