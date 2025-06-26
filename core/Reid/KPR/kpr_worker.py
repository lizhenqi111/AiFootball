import numpy as np

from core.base import BaseNode

class KPRPredictor(BaseNode):
    def __init__(self, cam_id, input_queue, output_queue, shm_queue, log_queue, config):
        super().__init__(
            node_type="reid",
            cam_id=cam_id,
            input_queue=input_queue,
            output_queue=output_queue,
            shm_queue=shm_queue,
            log_queue=log_queue,
            config=config
        )
        self.model = None
    
    def initialize(self):
        """初始化KPR模型"""
        from .kpr import KPRTrtInference
        
        self.logger.info("Initializing KPRPredictor model...")
        
        # 初始化KPR模型
        self.model = KPRTrtInference(
            self.config["model"]["weights_path"],
            parts_num=self.config["model"]["parts_num"],
        )
        
        self.logger.info("KPRPredictor model initialized")
    
    def process(self, frame, metadata):
        """执行ReID特征提取"""
        persons = metadata.get('person', [])
        if not persons:
            # 如果没有检测到人，设置空输出
            metadata[self.output_name[0]] = []
            metadata[self.output_name[1]] = []
            return metadata
        
        embeddings = []
        visibility_scores = []
        
        for person in persons:
            x1, y1, x2, y2, conf, classid = person
            
            # 截取人物区域
            person_image = frame[int(y1):int(y2), int(x1):int(x2), :]
            
            # 跳过无效区域
            if person_image.size == 0:
                continue
                
            # 预处理
            person_image = self.model.preprocess_image(person_image)
            
            # 推理
            result, _ = self.model.infer(person_image)
            
            # 解析结果
            parts_embeddings = result["parts_embeddings"]
            bn_foreground_embeddings = result["bn_foreground_embeddings"]
            parts_visibility = result["parts_visibility"]
            foreground_visibility = result["foreground_visibility"]
            
            # 调整形状并拼接
            bn_foreground_embeddings = np.expand_dims(bn_foreground_embeddings, axis=1)
            embedding = np.concatenate((parts_embeddings, bn_foreground_embeddings), axis=1)[0]
            visibility_score = np.concatenate((parts_visibility, foreground_visibility), axis=1)[0]
            
            embeddings.append(embedding.tolist())
            visibility_scores.append(visibility_score.tolist())
        
        # 更新元数据
        metadata[self.output_name[0]] = embeddings
        metadata[self.output_name[1]] = visibility_scores
        
        return metadata