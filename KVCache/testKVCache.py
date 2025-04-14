import torch
from utils import log_info  # 引入日志记录功能

class KVCacheManager:
    def __init__(self, device="cuda"):
        self.device = device
        self.cache = {}  # 存储 KVCache 的字典
        log_info(f"KVCacheManager 初始化，默认设备: {device}")

    def add_cache(self, key, value, to_gpu=True):
        if to_gpu and torch.cuda.is_available():
            self.cache[key] = value.to("cuda")
            log_info(f"缓存 {key} 已存储到 GPU")
        else:
            self.cache[key] = value.to("cpu")
            log_info(f"缓存 {key} 已存储到 CPU")

    def get_cache(self, key):
        if key in self.cache:
            log_info(f"获取缓存 {key}")
            return self.cache[key]
        else:
            raise KeyError(f"缓存中不存在键: {key}")

    def move_to_cpu(self, key):
        if key in self.cache:
            self.cache[key] = self.cache[key].to("cpu")
            log_info(f"缓存 {key} 已移动到 CPU")
        else:
            raise KeyError(f"缓存中不存在键: {key}")

    def move_to_gpu(self, key):
        if key in self.cache:
            if torch.cuda.is_available():
                self.cache[key] = self.cache[key].to("cuda")
                log_info(f"缓存 {key} 已移动到 GPU")
            else:
                raise RuntimeError("CUDA 不可用，无法将缓存移动到 GPU")
        else:
            raise KeyError(f"缓存中不存在键: {key}")

    def clear_cache(self, key=None):
        if key is None:
            self.cache.clear()
            log_info("所有缓存已清除")
        elif key in self.cache:
            del self.cache[key]
            log_info(f"缓存 {key} 已清除")
        else:
            raise KeyError(f"缓存中不存在键: {key}")
