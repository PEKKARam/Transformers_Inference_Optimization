import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer  # 修改为支持 GPT-2 的类
from datasets import load_dataset
from config import model_name, max_length, device, dataset_name, test_sample_size, use_kv_cache
from utils import measure_time, log_info

@measure_time
def run_inference():
    # 加载模型和分词器
    log_info(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)  # 加载 GPT-2 模型
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
    past_key_values = None  # 初始化 KV Cache
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)  # 确保输入不超长
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            if use_kv_cache:
                outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values  # 更新 KV Cache
            else:
                outputs = model(**inputs)
    end_time = time.time()

    # 输出推理时间
    total_time = end_time - start_time
    average_time = total_time / len(texts)
    log_info(f"总推理时间: {total_time:.4f} 秒")
    log_info(f"平均推理时间: {average_time:.4f} 秒/次")

if __name__ == "__main__":
    run_inference()