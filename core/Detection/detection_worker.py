import numpy as np

from .filter import process_detection_results
from core.base import BaseNode

# 检测节点
class DetectionPredictor(BaseNode):
    def __init__(self, cam_id, input_queue, output_queue, shm_queue, log_queue, config):
        super().__init__(
            node_type="detection",
            cam_id=cam_id,
            input_queue=input_queue,
            output_queue=output_queue,
            shm_queue=shm_queue,
            log_queue=log_queue,
            config=config
        )
        self.detection = None
        self.class_idxes = self.config['model']['class_idxes']
    
    def initialize(self):
        """初始化检测模型（在子进程中执行）"""
        from .detection import Detection
        self.detection = Detection(config=self.config)
        if self.config['slicer']['is_slicer']:
            self.detection.add_slicer_config()
        self.logger.info("Detection model initialized")
    
    def process(self, frame, metadata):
        """执行检测逻辑"""
        if self.config['slicer']['is_slicer']:
            detections = self.detection.slicer_predict(frame)
        else:
            detections = self.detection.predict(frame)
        
        # 处理检测结果
        result = np.concatenate((
            detections.xyxy, 
            detections.confidence[:, np.newaxis], 
            detections.class_id[:, np.newaxis]
        ), axis=1)
        
        result = process_detection_results(result)
        
        # 更新元数据
        for i in range(len(self.class_idxes)):
            class_result = result[result[:, -1] == self.class_idxes[i]].tolist()
            metadata[self.output_name[i]] = class_result
        
        return metadata