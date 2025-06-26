import numpy as np

from core.base import BaseNode

class CAMELTrackWorker(BaseNode):
    def __init__(self, cam_id, input_queue, output_queue, shm_queue, log_queue, config):
        super().__init__(
            node_type="track",
            cam_id=cam_id,
            input_queue=input_queue,
            output_queue=output_queue,
            shm_queue=shm_queue,
            log_queue=log_queue,
            config=config
        )
        self.tracker = None
        self.frame_index = 0  # 用于替代全局索引
    
    def initialize(self):
        """初始化多目标跟踪器"""
        from core.Track.cameltrack.cameltrack import CAMELTrack
        
        self.logger.info("Initializing CAMELTrackWorker...")
        
        # 初始化跟踪器
        self.tracker = CAMELTrack(
            engine_path=self.config["model"]["weights_path"],
            image_size=self.config['image']["img_size"],
            **self.config['parameters']
        )
        
        self.logger.info("CAMELTrackWorker initialized")
    
    def process(self, frame, metadata):
        """执行多目标跟踪"""
        if not metadata.get('person'):
            # 如果没有检测到人，创建空字典
            data_dict = {}
        else:
            # 准备跟踪数据
            data_dict = {
                'bbox_tlwh': np.array(metadata['person'])[:, 0:4],
                'bbox_conf': np.array(metadata['person'])[:, 4],
                'bboxes_classidx': np.array(metadata['person'])[:, 5],
                'keypoints_xyc': np.array(metadata['kps_xyc']),
                'kps_conf': np.array(metadata["kps_conf"]),
                'embeddings': np.array(metadata["embeddings"]),
                'visibility_scores': np.array(metadata['visibility_scores']),
                'id_list': np.array([i for i in range(len(metadata['person']))]),
                'im_width': np.array([frame.shape[1]] * len(metadata['person'])),
                'im_height': np.array([frame.shape[0]] * len(metadata['person'])),
                'num': np.array([len(metadata['person'])]),
                'image_id': self.frame_index,  # 使用本地索引
                'pitch': metadata['pitch']
            }
        
        # 执行跟踪
        tracklets = self.tracker.process(data_dict)
        
        # 更新元数据
        metadata[self.output_name] = tracklets
        
        # 更新帧索引
        self.frame_index += 1
        
        return metadata