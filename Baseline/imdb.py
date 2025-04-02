import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# 加载预训练模型和分词器
model_name = "bert-base-uncased"  
# model_name = "roberta-base"  
# model_name = "distilbert-base-uncased"
# 你可以替换为其他模型，比如 "bert-base-uncased" 或 "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 设置模型为评估模式
model.eval()

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备输入数据集
dataset = load_dataset("imdb")
texts = dataset["test"]["text"]  # 获取测试集中的句子

# 过滤超长文本
max_length = 512  # 模型的最大序列长度
filtered_texts = [text for text in texts if len(tokenizer(text, truncation=False)["input_ids"]) <= max_length]

print(f"原始文本数量: {len(texts)}")
print(f"过滤后文本数量: {len(filtered_texts)}")

# 使用过滤后的文本进行推理
texts = filtered_texts[:1000]  # 取前1000个文本进行测试


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