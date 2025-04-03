import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from config import model_name, max_length, device, dataset_name, test_sample_size
from utils import measure_time, log_info

@measure_time
def run_inference():
    # 加载模型和分词器
    log_info(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # 加载数据集
    log_info(f"加载数据集: {dataset_name}")
    dataset = load_dataset(dataset_name)
    texts = dataset["test"]["text"]

    # 过滤超长文本
    log_info("过滤超长文本...")
    filtered_texts = [text for text in texts if len(tokenizer(text, truncation=False)["input_ids"]) <= max_length]
    log_info(f"原始文本数量: {len(texts)}, 过滤后文本数量: {len(filtered_texts)}")

    # 使用过滤后的文本进行推理
    texts = filtered_texts[:test_sample_size]
    log_info(f"开始推理，样本数量: {len(texts)}")

    # 测量推理时间
    start_time = time.time()
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
    end_time = time.time()

    # 输出推理时间
    total_time = end_time - start_time
    average_time = total_time / len(texts)
    log_info(f"总推理时间: {total_time:.4f} 秒")
    log_info(f"平均推理时间: {average_time:.4f} 秒/次")

if __name__ == "__main__":
    run_inference()