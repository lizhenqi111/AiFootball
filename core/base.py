import time
import logging
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
import numpy as np

from config import load_config
from memory.memory import write_to_shm_directly, read_from_shm_directly, SharedMemoryContext
from log_mp.app_log import config_formatter, setup_main_logger, setup_worker_logger
from utils.cuda import CUDAContextManager



class BaseNode(Process, ABC):
    """处理节点基类，封装通用功能"""
    
    def __init__(self, 
                 node_type: str,
                 cam_id: int,
                 input_queue: Queue = None,
                 output_queue: Queue = None,
                 shm_queue: Queue = None,
                 log_queue: Queue = None,
                 config: dict = None):
        """
        初始化节点
        
        :param node_type: 节点类型名称 (如 "camera", "detection")
        :param cam_id: 摄像头ID
        :param input_queue: 输入队列
        :param output_queue: 输出队列
        :param shm_queue: 共享内存队列
        :param log_queue: 日志队列
        :param config: 节点配置字典
        """
        super().__init__()
        self.node_type = node_type
        self.cam_id = cam_id
        self.name = f"{node_type}_{cam_id}"
        
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shm_queue = shm_queue
        self.log_queue = log_queue
        self.config = config
        
        # 通用设置
        self.output_name = self.config.get('output_name', [])
        self.device = self.config.get('model', {}).get('device', 'cuda:0')
        self.daemon = True
        self.logger = None
        self.ctx = None  # CUDA上下文
    
    @abstractmethod
    def initialize(self):
        """初始化节点特定资源（在子进程中执行）"""
        pass
    
    @abstractmethod
    def process(self, frame: np.ndarray, metadata: dict) -> dict:
        """
        处理一帧数据
        
        :param frame: 图像帧
        :param metadata: 元数据字典
        :return: 更新后的元数据
        """
        pass
    
    def handle_queue_full(self):
        """处理输出队列满的情况"""
        if self.output_queue and self.output_queue.full():
            frameid_, shm_name_, _, _, _ = self.output_queue.get()
            self.shm_queue.put(shm_name_)
    
    def setup_cuda_context(self):
        """设置CUDA上下文（如果需要）"""

        self.ctx = CUDAContextManager(self.device)
        self.ctx.__enter__()  # 手动进入上下文
        self.logger.info(f"Initialized CUDA context on {self.device}")
    
    def release_resources(self):
        """释放资源（包括CUDA上下文）"""
        if self.ctx:
            self.ctx.__exit__(None, None, None)  # 手动退出上下文
            self.logger.info(f"Released CUDA context on {self.device}")
    
    def run(self):
        """节点主循环（在子进程中执行）"""
        try:
            self.logger = setup_worker_logger(self.log_queue)
            self.logger.info(f"Starting {self.name} node")
            
            # 设置CUDA上下文（如果需要）
            self.setup_cuda_context()
            
            # 初始化节点特定资源
            self.initialize()
            
            # 主处理循环
            while True:
                self.process_frame()
                time.sleep(0.001)
        except Exception as e:
            self.logger.exception(f"Unhandled exception in {self.name}")
        finally:
            self.release_resources()
            self.logger.info(f"Exiting {self.name} node")
    
    def process_frame(self):
        """处理单帧数据"""
        try:
            # 获取输入数据
            if self.input_queue:
                frameid, shm_name, frame_shape, frame_dtype, metadata = self.input_queue.get()
            else:
                # 对于无输入队列的节点（如相机）
                frameid, shm_name, frame_shape, frame_dtype, metadata = self.generate_input()
            
            # 读取共享内存
            with SharedMemoryContext(shm_name, frame_shape, frame_dtype) as frame:
                # 处理计时
                start_time = time.time()
                
                # 执行处理逻辑
                metadata = self.process(frame, metadata)
                
                # 记录处理时间
                process_time = (time.time() - start_time) * 1000
                self.logger.info(f"{self.node_type} frameid: {frameid} | {process_time:.2f}ms")
            
            # 处理输出队列满的情况
            self.handle_queue_full()
            
            # 发送输出
            if self.output_queue:
                self.output_queue.put((
                    frameid,
                    shm_name,
                    frame_shape,
                    frame_dtype,
                    metadata
                ))
            else:
                # 对于无输出队列的节点（如可视化）
                self.shm_queue.put(shm_name)
        
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            # 确保在异常情况下释放共享内存
            if 'shm_name' in locals():
                self.shm_queue.put(shm_name)
            time.sleep(0.001)  # 防止错误循环占用CPU