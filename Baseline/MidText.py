import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载预训练模型和分词器
model_name = "gpt2"  # 你可以替换为其他模型，比如 "bert-base-uncased" 或 "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 设置模型为评估模式
model.eval()

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备输入数据集
texts = [
    "This is a test sentence to measure inference speed.",
    "Another example sentence for testing.",
    "This is yet another sentence to evaluate the model.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world."
]  # 你可以替换为从文件读取的文本列表

# 预热模型（避免初次运行的加载时间影响结果）
inputs = tokenizer(texts[0], return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}
for _ in range(5):
    _ = model(**inputs)

# 测量推理时间
num_runs = len(texts)  # 测试运行次数等于文本数量
start_time = time.time()

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

end_time = time.time()

# 计算平均推理时间
total_time = end_time - start_time
average_time = total_time / num_runs

print(f"模型: {model_name}")
print(f"设备: {device}")
print(f"推理次数: {num_runs}")
print(f"总推理时间: {total_time:.4f} 秒")
print(f"平均推理时间: {average_time:.4f} 秒/次")