import time
import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # 创建日志目录

# 配置日志记录
log_file = os.path.join(log_dir, "inference.log")
logging.basicConfig(
    filename=log_file,  # 日志文件名
    filemode="a",              # 追加模式写入日志
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    level=logging.INFO,         # 日志级别
    encoding="utf-8"          # 日志文件编码
)

def measure_time(func):
    """装饰器，用于测量函数运行时间"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"函数 {func.__name__} 运行时间: {elapsed_time:.4f} 秒")
        return result
    return wrapper

def log_info(message):
    """记录信息日志"""
    logging.info(message)