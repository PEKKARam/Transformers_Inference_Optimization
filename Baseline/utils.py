import time

def measure_time(func):
    """装饰器，用于测量函数运行时间"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"运行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper