import logging
from multiprocessing import Lock, Manager
from multiprocessing.shared_memory import SharedMemory
from typing import List, Optional

import numpy as np
import array

logging.basicConfig(level=logging.INFO)

class SharedMemoryPool:
    
    def __init__(self, block_size: int, num_blocks: int = 100, logger = None):
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # 使用Manager创建跨进程队列
        self.manager = Manager()
        self.available = self.manager.Queue()  # 可用内存块名称队列
        self.all_shm = []  # 主进程持有的共享内存对象

        self.logger = logger
        
        # 初始化共享内存块
        for _ in range(num_blocks):
            shm = SharedMemory(create=True, size=block_size)
            self.all_shm.append(shm)
            self.available.put(shm.name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def acquire_from_queue(self) -> Optional[str]:
        """从队列获取内存块名称 (子进程调用)"""
        try:
            return self.available.get(timeout=5)
        except Exception as e:
            return None
    
    def release_to_queue(self, name: str):
        """释放内存块回队列 (子进程调用)"""
        self.available.put(name)
    
    def cleanup(self):
        """主进程负责最终清理，释放所有共享资源"""
        try:
            # 先清理共享内存块
            for shm in self.all_shm:
                if shm:
                    shm.close()
                    shm.unlink()
            self.all_shm = []  # 清空引用列表
            
            # 清理队列资源（尝试清空队列）
            while not self.available.empty():
                try:
                    self.available.get_nowait()
                except Exception:
                    break
            
            # 释放Manager资源（关键修复点）
            if hasattr(self, 'manager') and self.manager:
                self.manager.shutdown()  # 关闭Manager进程
            
        except Exception as e:
            pass

class SharedMemoryContext:
    """共享内存上下文管理器，支持读写模式"""
    
    def __init__(self, shm_name, shape=None, dtype=None, mode='read'):
        """
        初始化上下文
        
        :param shm_name: 共享内存名称
        :param shape: 图像形状 (仅读取模式需要)
        :param dtype: 图像数据类型 (仅读取模式需要)
        :param mode: 'read' 或 'write'
        """
        self.shm_name = shm_name
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.shm = None
    
    def __enter__(self):
        """进入上下文，根据模式返回不同对象"""
        self.shm = SharedMemory(name=self.shm_name)
        
        if self.mode == 'read':
            if self.shape is None or self.dtype is None:
                raise ValueError("Shape and dtype must be provided for read mode")
            return self._read_from_shm()
        else:
            # 写入模式返回共享内存对象本身
            return self.shm
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，关闭共享内存"""
        if self.shm:
            self.shm.close()
    
    def _read_from_shm(self):
        """从共享内存读取数据"""
        # 创建目标数组视图
        return np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self.shm.buf)


def write_to_shm_directly(data: np.ndarray, shm: SharedMemory):
    """
    直接将NumPy数组数据拷贝到共享内存
    
    参数：
    data : 要写入的NumPy数组
    shm  : 已创建的共享内存块
    """
    # 验证共享内存大小是否足够
    required_bytes = data.nbytes
    if shm.size < required_bytes:
        raise ValueError(f"共享内存不足，需要 {required_bytes} 字节，当前为 {shm.size} 字节")

    # 创建目标数组视图（直接使用共享内存缓冲区）
    target_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    target_arr.fill(0)

    # 使用np.copyto进行高效拷贝
    np.copyto(target_arr, data)


def read_from_shm_directly(shm: SharedMemory, shape, data_type):
    target = np.ndarray(shape=shape, dtype=data_type, buffer=shm.buf)
    return target


# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    # 主进程初始化
    with SharedMemoryPool(block_size=1024, num_blocks=4) as pool:
        # 启动子进程
        from multiprocessing import Process
        
        def child_process(available):
            # 子进程通过队列操作
            while True:
                shm_name = available.get()
                print(f"子进程获取内存块: {shm_name}")
                
                # 使用共享内存...
                shm = SharedMemory(name=shm_name)
                arr = array.array('b', shm.buf[:5])
                print(arr)
                
                available.put(shm_name)  # 释放回池
        
        p = Process(target=child_process, args=(pool.available,))
        p.start()
        p.join()







