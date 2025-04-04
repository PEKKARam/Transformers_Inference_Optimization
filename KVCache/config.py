import torch

model_name = "bert-base-uncased"
max_length = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据集配置
dataset_name = "imdb"
test_sample_size = 1000          # 测试样本数量

# KVCache 配置
kvcache_device = device  # 默认设备