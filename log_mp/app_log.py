import logging
import multiprocessing
from logging.handlers import QueueHandler, QueueListener
# from cloghandler import ConcurrentRotatingFileHandler

from multiprocessing import Process, Queue

def config_formatter():
    # 配置日志格式
    log_format = "%(asctime)s | %(levelname)-8s | %(processName)-16s | %(message)s"
    formatter = logging.Formatter(log_format)
    return formatter

# 主进程日志初始化
def setup_main_logger(log_queue, log_file="app.log"):
    # 主进程的日志处理器（写入文件）
    file_handler = logging.FileHandler(log_file)
    # file_handler = ConcurrentRotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5) # 异步写入
    file_handler.setFormatter(config_formatter())

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(config_formatter())

    # 创建 QueueListener 监听队列并分发日志
    queue_listener = QueueListener(log_queue, file_handler, console_handler)
    queue_listener.start()

    return queue_listener

# 子进程日志配置
def setup_worker_logger(log_queue):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 清除所有现有处理器（避免默认的 root logger 处理器）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 子进程仅通过 QueueHandler 发送日志到队列
    queue_handler = QueueHandler(log_queue)

    logger.addHandler(queue_handler)
    return logger

# 子进程任务
def worker_task(num):
    logger = setup_worker_logger(log_queue)
    logger.info(f"子进程 {num} 启动")
    try:
        # 模拟业务逻辑
        1 / (num % 2)  # 偶数会触发除零错误
    except Exception as e:
        logger.error(f"子进程 {num} 发生错误: {str(e)}")
    logger.info(f"子进程 {num} 结束")

if __name__ == "__main__":
    # 创建多进程安全的队列
    log_queue = Queue()

    # 主进程初始化日志监听
    listener = setup_main_logger(log_queue)

    # 创建子进程池
    processes = []
    for i in range(3):
        p = Process(target=worker_task, args=(i,), name=f"Worker-{i}")
        processes.append(p)
        p.start()

    # 等待所有子进程结束
    for p in processes:
        p.join()

    # 停止日志监听并清理资源
    listener.stop()