
import multiprocessing as mp
import cv2
import time

from memory.memory import write_to_shm_directly

from core.base import BaseNode
from memory.memory import SharedMemoryContext


class CameraProcessor(BaseNode):
    def __init__(self, cam_id, output_queue, shm_queue, log_queue, config):
        super().__init__(
            node_type="camera",
            cam_id=cam_id,
            output_queue=output_queue,
            shm_queue=shm_queue,
            log_queue=log_queue,
            config=config
        )
        self.cap = None
        self.frame_id = 0
    
    def initialize(self):
        """初始化相机"""
        self.logger.info(f"Initializing camera {self.cam_id}...")
        
        # 获取相机配置
        cam_config = self.config.get(f"cam_{self.cam_id}", self.config.get("default", {}))
        
        # 初始化相机
        if "path" in cam_config and cam_config["path"]:
            self.cap = cv2.VideoCapture(cam_config["path"])
        else:
            self.cap = cv2.VideoCapture(self.cam_id)
        
        # 设置相机参数
        if "fps" in cam_config:
            self.frame_time = 1 / cam_config["fps"]
        else:
            self.frame_time = 100 
        if "width" in cam_config and "height" in cam_config:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config["height"])
        
        # 检查相机是否成功打开
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.cam_id}")
        
        self.logger.info(f"Camera {self.cam_id} initialized")
        
        # 等待其他节点初始化
        self.logger.info("Waiting for other nodes to initialize...")
        time.sleep(10)
    
    def generate_input(self):
        """生成输入数据（针对相机节点）"""
        start_time = time.time()

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Camera {self.cam_id} failed to read frame")
        
        # 从池中获取内存块
        shm_name = self.shm_queue.get()
        
        # 写入共享内存
        with SharedMemoryContext(shm_name, mode='write') as shm:
            write_to_shm_directly(frame, shm)
        
        # 返回元数据
        self.frame_id += 1

        # 控制帧率
        end_time = time.time()
        wait_time = (self.frame_time - (end_time - start_time))
        if wait_time > 0:
            time.sleep(wait_time)

        return self.frame_id, shm_name, frame.shape, frame.dtype, {"frameid": self.frame_id}
    
    def process(self, frame, metadata):
        """相机节点不需要处理，直接返回元数据"""
        # time.sleep(0.1)
        return metadata
    
    def release_resources(self):
        """释放相机资源"""
        if self.cap:
            self.cap.release()
            self.logger.info(f"Camera {self.cam_id} released")
        super().release_resources()