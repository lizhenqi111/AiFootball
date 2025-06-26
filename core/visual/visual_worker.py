import numpy as np

from .visualize import visualize_detections
from core.base import BaseNode

class Visualize(BaseNode):
    def __init__(self, cam_id, input_queue, shm_queue, log_queue, config):
        super().__init__(
            node_type="visual",
            cam_id=cam_id,
            input_queue=input_queue,
            shm_queue=shm_queue,
            log_queue=log_queue,
            config=config
        )
    
    def initialize(self):
        """可视化节点初始化"""
        # 可以在这里初始化可视化资源
        pass
    
    def process(self, frame, metadata):
        """执行可视化逻辑"""
        if not metadata.get('person'):
            data_dict = {}
        else:
            data_dict = {
                'bbox_tlwh': np.array(metadata['person'])[:, 0:4],
                'bbox_conf': np.array(metadata['person'])[:, 4],
                'bboxes_classidx': np.array(metadata['person'])[:, 5],
                'num': len(metadata['person']),
                'keypoints_xyc': np.array(metadata.get('kps_xyc', [])),
                'kps_conf': np.array(metadata.get("kps_conf", [])),
                'im_width': frame.shape[1],
                'im_height': frame.shape[0],
                'image_id': metadata.get('frameid', 0),
                'field_keypoint': np.array(metadata.get('pitch', [])),
                'tracklets': metadata.get('tracklets', [])
            }
        
        visualize_detections(frame, data_dict, metadata.get('frameid', 0), output_dir=f"./visualizations/{self.cam_id}")
        return metadata