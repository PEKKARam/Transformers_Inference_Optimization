import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from KVCache import KVCacheManager
from utils import measure_time, log_info
from config import model_name, max_length, device, dataset_name, test_sample_size

# 初始化 KVCache 管理器
kvcache_manager = KVCacheManager(device=device)

# 加载模型和分词器
log_info(f"加载模型: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# 加载 IMDB 数据集
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

# 模拟 KVCache 的生成和管理
@measure_time
def run_inference_with_kvcache():
    """使用 KVCache 优化的推理"""
    for i, text in enumerate(texts):
        # 模拟生成 KVCache
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            kv_cache = outputs.past_key_values  # 获取 KVCache

        # 将部分 KVCache 移动到 CPU（模拟不常用的 KVCache）
        if i % 2 == 0:  # 偶数索引的 KVCache 移动到 CPU
            for layer, (key, value) in enumerate(kv_cache):
                kvcache_manager.add_cache(f"layer_{layer}_key_{i}", key, to_gpu=False)
                kvcache_manager.add_cache(f"layer_{layer}_value_{i}", value, to_gpu=False)

        # 模拟后续推理时使用 KVCache
        for layer, (key, value) in enumerate(kv_cache):
            if i % 2 == 0:  # 从 CPU 恢复到 GPU
                key = kvcache_manager.get_cache(f"layer_{layer}_key_{i}").to(device)
                value = kvcache_manager.get_cache(f"layer_{layer}_value_{i}").to(device)
            # 模拟使用 KVCache 的推理
            _ = model(**inputs, past_key_values=((key, value),))

# 测试推理速度
log_info("开始测试使用 KVCache 优化的推理速度")
run_inference_with_kvcache()
