from datasets import load_dataset
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import atexit
from pyarrow.fs import S3FileSystem, initialize_s3, finalize_s3
from tqdm import tqdm
from utils import log_info, log_inference_stats  # 引入日志记录函数
from config import model_name, device, max_length

# 初始化 S3
initialize_s3()
atexit.register(finalize_s3)  # 确保程序退出时清理 S3 资源

# 示例：使用 S3 文件系统
s3 = S3FileSystem(region="us-west-2")

# 加载 WikiText-2 数据集
dataset = load_dataset("openwebtext", split="test", trust_remote_code=True, cache_dir="D:/Graduate_ Design/Transformers_Inference_Optimization/cache")
texts = dataset["text"][:100]  # 选取前100条测试样本

# 初始化模型与分词器（在 CPU 上测试）
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# 生成推理时间统计
def measure_inference_time(text, max_new_tokens=50):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # 使用 KV Cache 进行推理
    start_time = time.time()
    _ = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)
    return time.time() - start_time

# 添加进度条显示推理进度
times = []
for text in tqdm(texts, desc="推理进度", unit="样本"):
    if len(text.strip()) > 0:
        inference_time = measure_inference_time(text)
        times.append(inference_time)

avg_time = sum(times) / len(times)
print(f"平均推理时间：{avg_time:.4f} 秒")
log_inference_stats(sum(times), len(times))
                    