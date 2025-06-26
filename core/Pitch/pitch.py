import numpy as np
import tensorrt as trt
import cv2

import pycuda.driver as cuda

class HRNetPredictionTransform:
    def __init__(self, size):
        self.H, self.W = size

    def __call__(self, preds):
        preds = np.exp(preds)
        B, N, H, W = preds.shape
        
        # 向量化操作
        max_h = preds.max(axis=2)
        max_w = preds.max(axis=3)
        
        x = max_h.argmax(axis=2) * (self.W / W)
        y = max_w.argmax(axis=2) * (self.H / H)
        conf = np.minimum(max_h.max(axis=2), max_w.max(axis=2))
        
        predictions = np.stack([x, y, conf], axis=-1)
        return predictions[:, :-1, :]

class Preprocess:
    def __init__(self, target_size):
        self.target_size = target_size

    def preprocess_image(self, image):
        """图像预处理"""
        img = cv2.resize(image, self.target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32)
        return img / 255.0

class PipelinedTRTInference:
    def __init__(self, engine_path):
        # 加载TensorRT引擎
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出尺寸
        self.input_shape = tuple(self.engine.get_binding_shape(0))
        self.output_shape = tuple(self.engine.get_binding_shape(1))
        self.input_size = trt.volume(self.input_shape)
        self.output_size = trt.volume(self.output_shape)
        
        # 双缓冲设置
        self.streams = [cuda.Stream() for _ in range(2)]
        
        # 主机端固定内存
        self.h_inputs = [
            cuda.pagelocked_empty(self.input_size, dtype=np.float32),
            cuda.pagelocked_empty(self.input_size, dtype=np.float32)
        ]
        self.h_outputs = [
            cuda.pagelocked_empty(self.output_size, dtype=np.float32),
            cuda.pagelocked_empty(self.output_size, dtype=np.float32)
        ]
        
        # 设备端内存
        self.d_inputs = [
            cuda.mem_alloc(self.h_inputs[0].nbytes),
            cuda.mem_alloc(self.h_inputs[0].nbytes)
        ]
        self.d_outputs = [
            cuda.mem_alloc(self.h_outputs[0].nbytes),
            cuda.mem_alloc(self.h_outputs[0].nbytes)
        ]
        
        self.current_idx = 0
        self._warmup()

    def start_async_inference(self, input_tensor):
        """启动异步推理，返回缓冲区索引"""
        idx = self.current_idx
        stream = self.streams[idx]
        
        # 复制输入数据
        np.copyto(self.h_inputs[idx], input_tensor.ravel())
        
        # 异步操作
        cuda.memcpy_htod_async(self.d_inputs[idx], self.h_inputs[idx], stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_inputs[idx]), int(self.d_outputs[idx])],
            stream_handle=stream.handle
        )
        cuda.memcpy_dtoh_async(self.h_outputs[idx], self.d_outputs[idx], stream)
        
        # 更新索引
        self.current_idx = 1 - self.current_idx
        return idx

    def get_result(self, idx):
        """获取指定索引的推理结果"""
        self.streams[idx].synchronize()
        return self.h_outputs[idx].copy().reshape(self.output_shape)
    
    def _warmup(self, iterations=3):
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(iterations):
            idx = self.start_async_inference(dummy_input)
    
    def __del__(self):
        # 同步所有流以确保操作完成
        for stream in self.streams:
            if stream:
                stream.synchronize()
        
        # 释放设备端内存
        for d_input in self.d_inputs:
            if d_input:
                d_input.free()
        for d_output in self.d_outputs:
            if d_output:
                d_output.free()
        
        # 释放流资源
        for stream in self.streams:
            if stream:
                del stream
        
        # 释放TensorRT资源
        if self.context:
            del self.context
        if self.engine:
            del self.engine
        
        # 清除引用
        self.streams = None
        self.h_inputs = None
        self.h_outputs = None
        self.d_inputs = None
        self.d_outputs = None
        self.context = None
        self.engine = None
            
    


