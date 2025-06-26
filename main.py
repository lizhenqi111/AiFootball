from memory.memory import SharedMemoryPool
import multiprocessing as mp
import time
import numpy as np
import logging
from typing import Dict, List, Type, Any

from core.Camera.image_capture import CameraProcessor
from core.Detection.detection_worker import DetectionPredictor
from core.Pitch.pitch_worker import PitchPredictor
from core.Keypoint.hrnet_worker import HRNetPredictor
from core.Reid.KPR.kpr_worker import KPRPredictor
from core.Track.cameltrack.cameltrack_worker import CAMELTrackWorker
from core.visual.visual_worker import Visualize
from log_mp.app_log import config_formatter, setup_main_logger, setup_worker_logger
from config import load_config

class AnaSystem:
    def __init__(self, config_path="configs/config.yaml"):
        self.config = load_config(config_path)
        self.cameras = self.config["cam_indexs"]
        
        # 创建日志队列和监听器
        self.log_queue = mp.Queue()
        self.listener = setup_main_logger(self.log_queue)
        self.main_logger = setup_worker_logger(self.log_queue)
        
        # 创建共享内存池
        image_size = tuple(self.config["SharedMemoryPool"]["image_size"])
        self.memory_pool = SharedMemoryPool(
            block_size=np.prod(image_size),
            num_blocks=self.config["SharedMemoryPool"]["num_blocks"],
            logger=self.main_logger
        )
        
        # 创建处理流水线队列
        self.pipeline_queues = {}
        for camera_id in self.cameras:
            self.pipeline_queues[camera_id] = {
                "image": mp.Queue(5),
                "detection": mp.Queue(5),
                "pitch": mp.Queue(5),
                "keypoints": mp.Queue(5),
                "reid": mp.Queue(5),
                "track": mp.Queue(5),
            }
        
        # 性能监控
        self.start_time = time.time()
        self.processed_frames = 0

    def build_pipeline(self, camera_id):
        """为指定摄像头构建处理流水线"""
        queues = self.pipeline_queues[camera_id]
        
        # 创建节点实例
        nodes = [
            CameraProcessor(
                cam_id=camera_id,
                output_queue=queues["image"],
                shm_queue=self.memory_pool.available,
                log_queue=self.log_queue,
                config=load_config("configs/core/Camera/camera.yaml")
            ),
            DetectionPredictor(
                cam_id=camera_id,
                input_queue=queues["image"],
                output_queue=queues["detection"],
                shm_queue=self.memory_pool.available,
                log_queue=self.log_queue,
                config=load_config("configs/core/Detection/yolo.yaml")
            ),
            PitchPredictor(
                cam_id=camera_id,
                input_queue=queues["detection"],
                output_queue=queues["pitch"],
                shm_queue=self.memory_pool.available,
                log_queue=self.log_queue,
                config=load_config("configs/core/Pitch/pitch.yaml")
            ),
            HRNetPredictor(
                cam_id=camera_id,
                input_queue=queues["pitch"],
                output_queue=queues["keypoints"],
                shm_queue=self.memory_pool.available,
                log_queue=self.log_queue,
                config=load_config("configs/core/Keypoint/hrnet.yaml")
            ),
            KPRPredictor(
                cam_id=camera_id,
                input_queue=queues["keypoints"],
                output_queue=queues["reid"],
                shm_queue=self.memory_pool.available,
                log_queue=self.log_queue,
                config=load_config("configs/core/Reid/KPR/KPR.yaml")
            ),
            CAMELTrackWorker(
                cam_id=camera_id,
                input_queue=queues["reid"],
                output_queue=queues["track"],
                shm_queue=self.memory_pool.available,
                log_queue=self.log_queue,
                config=load_config("configs/core/Track/cameltrack/cameltrack.yaml"),
            ),
            Visualize(
                cam_id=camera_id,
                input_queue=queues["track"],
                shm_queue=self.memory_pool.available,
                log_queue=self.log_queue,
                config=load_config("configs/core/visual/visual.yaml")
            )
        ]
        
        return nodes

    def start_task(self):
        try:         
            all_processes = []
            
            # 为每个摄像头构建流水线
            for camera_id in self.cameras:
                camera_processes = self.build_pipeline(camera_id)
                all_processes.extend(camera_processes)
                self.main_logger.info(f"Built pipeline for camera {camera_id} with {len(camera_processes)} nodes")
            
            # 启动所有进程
            for p in all_processes:
                p.start()
                self.main_logger.info(f"Started process: {p.name}")
            
            # 监控系统运行状态
            self.monitor_system(all_processes)
            
        except Exception as e:
            self.main_logger.exception(f"System initialization failed: {str(e)}")
        finally:
            self.shutdown_system(all_processes)
    
    def monitor_system(self, processes: list):
        """监控系统运行状态"""
        try:
            last_report_time = time.time()
            
            while any(p.is_alive() for p in processes):
                time.sleep(1)
                
                # 每秒报告一次
                current_time = time.time()
                if current_time - last_report_time > 1.0:
                    self.report_system_status()   # 帧率 ，队列积压
                    last_report_time = current_time
                
        except KeyboardInterrupt:
            self.main_logger.info("Received keyboard interrupt, shutting down")
    
    def report_system_status(self):
        """报告系统状态"""
        # 计算帧率
        elapsed_time = time.time() - self.start_time
        fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
        
        # 报告基本信息
        self.main_logger.info(
            f"System status: FPS={fps:.2f}, "
            f"Frames={self.processed_frames}, "
            f"Uptime={elapsed_time:.1f}s"
        )
        
        # 报告每个摄像头的队列状态
        for camera_id in self.cameras:
            queues = self.pipeline_queues[camera_id]
            queue_status = " | ".join(
                f"{stage}:{q.qsize()}" 
                for stage, q in queues.items()
            )
            self.main_logger.debug(
                f"Camera {camera_id} queues: {queue_status}"
            )
    
    def shutdown_system(self, processes: list):
        """优雅关闭系统"""
        self.main_logger.info("Initiating system shutdown...")
        
        # 终止所有进程
        for p in processes:
            if p.is_alive():
                p.terminate()
        
        # 等待进程结束
        for p in processes:
            p.join(timeout=2.0)
            if p.is_alive():
                self.main_logger.warning(f"Process {p.name} did not terminate")
            else:
                self.main_logger.info(f"Process {p.name} terminated")
        
        # 关闭内存池
        self.memory_pool.cleanup()
        self.main_logger.info("Memory pool shutdown complete")
        
        # 关闭日志监听
        self.listener.stop()
        self.main_logger.info("Log listener stopped")
        
        # 最终报告
        total_time = time.time() - self.start_time
        self.main_logger.info(
            f"System shutdown complete. Total frames: {self.processed_frames}, "
            f"Average FPS: {self.processed_frames/total_time:.2f}"
        )

if __name__ == "__main__":
    mp.set_start_method("spawn")
    ana_system = AnaSystem()
    ana_system.start_task()