import time

from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from config import load_config
from memory.memory import write_to_shm_directly, read_from_shm_directly
from log_mp.app_log import config_formatter, setup_main_logger, setup_worker_logger
from utils.cuda import CUDAContextManager

from .rtmpose import RTMPose

class RTMPosePredictor(Process):
    def __init__(self, 
                 cam_id,
                 input_queue: Queue,
                 output_queue: Queue,
                 shm_queue: Queue,
                 log_queue: Queue):
        super().__init__()
        self.config = load_config("configs/core/Keypoint/rtmpose.yaml")
        self.log_queue = log_queue
        
        self.cam_id = cam_id
        self.name = f"keypoint_{self.cam_id}"
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shm_queue = shm_queue

        self.output_name = self.config['output_name']
        self.device = self.config['model']['device']

        self.daemon = True

    def run(self):
        logger = setup_worker_logger(self.log_queue)

        with CUDAContextManager(self.device) as ctx:
            model = RTMPose(self.config["model"]["weights_path"])
            
            while True:   
                frameid, shm_name, frame_shape, frame_dtype, out = self.input_queue.get()

                shm = SharedMemory(name=shm_name)

                debug_start = time.time()
                persons = out['person']
                if persons == []:
                    out['output_name'] = []
                else:
                    results = []
                    frame = read_from_shm_directly(shm, frame_shape, frame_dtype)
                    for person in persons:
                        x1, y1, x2, y2, conf, classid = person
                        person_image = frame[int(y1):int(y2), int(x1):int(x2), :]
                        person_image = model.preprocess(person_image)
                        result = model.postprocess(*(model.infer(person_image))) # [1, 14, 3]
                    results.append(result.tolist())
                    out[self.output_name] = results
                debug_end = time.time()

                # 如果满了，就跳过最旧的帧，不作处理了
                if self.output_queue.full():
                    frameid_, shm_name_, _, _, _ = self.output_queue.get()
                    self.shm_queue.put(shm_name_)
                # 发送元数据（不传递对象，仅传名称）
                self.output_queue.put((
                    frameid,
                    shm_name,
                    frame_shape,
                    frame_dtype,
                    out,
                ))

                shm.close()
                logger.info(f"keypoint frameid: {frameid}  | {(debug_end - debug_start)*1000:.2f}")
                time.sleep(0.001)