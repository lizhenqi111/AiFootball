import numpy as np
from .hrnet_preprocess import preprocess, KeypointMapper

from core.base import BaseNode

class HRNetPredictor(BaseNode):
    def __init__(self, cam_id, input_queue, output_queue, shm_queue, log_queue, config):
        super().__init__(
            node_type="keypoint",
            cam_id=cam_id,
            input_queue=input_queue,
            output_queue=output_queue,
            shm_queue=shm_queue,
            log_queue=log_queue,
            config=config
        )
        self.transform = None
        self.model = None
    
    def initialize(self):
        """初始化关键点模型（在子进程中执行）"""
        from .hrnet_preprocess import Compose
        from .hrnet import HRNetTRTPredictor

        self.transform = Compose([
            {'type': 'LoadImage'},
            {'type': 'GetBBoxCenterScale'},
            {'type': 'TopdownAffine', 'input_size': (192, 256)},
            {'type': 'PackPoseInputs'}
        ])
        
        # 初始化模型 - 在CUDA上下文已设置的环境中
        self.model = HRNetTRTPredictor(self.config["model"]["weights_path"])
        self.logger.info("HRNet model initialized")
    
    def process(self, frame, metadata):
        """执行关键点检测"""
        kps_xyc = []
        kps_conf = []
        
        persons = metadata.get('person', [])
        if not persons:
            metadata[self.output_name[0]] = []
            metadata[self.output_name[1]] = []
            return metadata
        
        for person in persons:
            x1, y1, x2, y2, conf, classid = person
            data = {'img': frame, 'bbox': [x1, y1, x2-x1, y2-y1]}
            person_data = self.transform(data)
            person_image = person_data['inputs']
            
            # 预处理
            person_image = preprocess(person_image)
            
            # 推理
            keypoints = self.model.postprocess(*(self.model.infer(person_image)))
            keypoints = KeypointMapper.map_keypoints_back(keypoints, person_data['data_samples']['trans'])
            
            # 过滤低置信度关键点
            keypoints[keypoints[:, 2] < 0.1][:, 2] = 0
            visible_kps = np.nonzero(keypoints[:, 2])[0]
            
            # 计算整体置信度
            if len(visible_kps) < 3:
                conf = 0
            else:
                conf = np.mean(keypoints[visible_kps][:, 2])
            
            kps_xyc.append(keypoints.tolist())
            kps_conf.append(conf)
        
        metadata[self.output_name[0]] = kps_xyc
        metadata[self.output_name[1]] = kps_conf
        
        return metadata
