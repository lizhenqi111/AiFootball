import time

from core.base import BaseNode

class PitchPredictor(BaseNode):
    def __init__(self, cam_id, input_queue, output_queue, shm_queue, log_queue, config):
        super().__init__(
            node_type="pitch",
            cam_id=cam_id,
            input_queue=input_queue,
            output_queue=output_queue,
            shm_queue=shm_queue,
            log_queue=log_queue,
            config=config
        )
        self.model = None
        self.preprocess = None
        self.postprocess = None
    
    def initialize(self):
        """初始化场地关键点检测模型"""
        from .pitch import PipelinedTRTInference
        from .pitch import Preprocess
        from .pitch import HRNetPredictionTransform
        
        self.logger.info("Initializing PitchPredictor model...")
        
        # 初始化预处理和后处理
        self.preprocess = Preprocess(self.config['inference']['img_size'])
        self.postprocess = HRNetPredictionTransform(self.config["image"]["img_size"])
        
        # 初始化TRT推理模型
        self.model = PipelinedTRTInference(
            self.config["model"]["weights_path"],
        )
        
        self.logger.info("PitchPredictor model initialized")
    
    def process(self, frame, metadata):
        """执行场地关键点检测"""
        # 预处理图像
        input_tensor = self.preprocess.preprocess_image(frame)
        
        # 执行异步推理
        idx = self.model.start_async_inference(input_tensor)
        result = self.model.get_result(idx)
        
        # 后处理结果
        predictions = self.postprocess(result)[0]
        
        # 过滤低置信度的关键点
        valid_mask = predictions[:, 2] > self.config['inference']['conf']
        valid_points = predictions[valid_mask, :2].astype(int)
        
        # 更新元数据
        metadata[self.output_name] = valid_points.tolist()
        return metadata