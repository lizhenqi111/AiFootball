import numpy as np
import supervision as sv
from ultralytics import YOLO

class Detection:
    def __init__(self, config):
        self.model = YOLO(config['model']['weights_path'])  # 替换为你的模型路径
        self.device = config['model']['device']
        self.conf = config['inference']['conf']
        self.verbose = config['inference']['verbose']
        self.half = config['model']['fp16']

        self.config = config

        # self._warmup()

    def predict(self, image):
        results = self.model.predict(image, device=self.device, conf=self.conf, verbose=self.verbose, half=self.half, task="detect")
        return results

    def add_slicer_config(self):
        self.input_hight = self.config['slicer']['input_hight']
        self.input_width = self.config['slicer']['input_width']
        split_h_n = self.config['slicer']['split_h_n']
        split_w_n = self.config['slicer']['split_w_n']
        overlap_wh = self.config['slicer']['overlap_wh']
        iou_threshold = self.config['slicer']['slicer_iou_threshold']
        thread_workers = self.config['slicer']['thread_workers']

        def callback(patch: np.ndarray) -> sv.Detections: 
            result = self.model.predict(patch, device=self.device, conf=self.conf, verbose=self.verbose, half=self.half, task="detect")[0]
            return sv.Detections.from_ultralytics(result)
        
        self.slicer = sv.InferenceSlicer(
                            callback = callback,
                            overlap_filter = sv.OverlapFilter.NON_MAX_SUPPRESSION,
                            slice_wh = ( self.input_width // split_w_n if split_w_n==1 else self.input_width // split_w_n + 100, self.input_hight // split_h_n if split_h_n ==1 else self.input_hight // split_h_n + 100),
                            overlap_ratio_wh = None,
                            overlap_wh = (0 if split_w_n == 1 else overlap_wh, 0 if split_h_n == 1 else overlap_wh),
                            iou_threshold = iou_threshold,
                            thread_workers=thread_workers)


    
    def slicer_predict(self, image):
        return self.slicer(image)
    
    def _warmup(self, warm_up_times = 3):
        warmup_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        for i in range(warm_up_times):
            self.model.predict(warmup_image, device=self.device, conf=self.conf, verbose=self.verbose, half=self.half, task="detect")



