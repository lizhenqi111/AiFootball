import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import time
import cv2

import torch


class HRNetTRTPredictor:
    """HRNet 模型的 TensorRT 推理预测器，包含预热功能"""
    
    def __init__(self, engine_path, warm_up_times=3, input_shape=None):
        """
        初始化 TensorRT 推理器
        
        Args:
            engine_path: TensorRT引擎文件路径
            warm_up_times: 预热次数
            input_shape: 输入形状，如未指定则从引擎获取
        """
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        self.engine_path = engine_path
        self.warm_up_times = warm_up_times
        
        # 加载引擎
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # 设置输入形状
        if input_shape is not None:
            self.input_shape = input_shape
            self.context.set_binding_shape(0, self.input_shape)
        else:
            self.input_shape = self.engine.get_binding_shape(0)
        
        # 分配缓冲区
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        # 执行预热
        self._warm_up()
        
    
    def _load_engine(self):
        """加载TensorRT引擎"""
        with open(self.engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        """分配输入输出缓冲区"""
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        for binding_idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding_idx)
            is_input = self.engine.binding_is_input(binding_idx)
            
            # 获取形状
            if is_input:
                shape = self.context.get_binding_shape(binding_idx)
            else:
                shape = self.engine.get_binding_shape(binding_idx)
            
            # 计算内存大小和数据类型
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if is_input:
                inputs.append({
                    "name": binding_name,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": shape,
                    "dtype": dtype
                })
            else:
                outputs.append({
                    "name": binding_name,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": shape,
                    "dtype": dtype
                })
        
        return inputs, outputs, bindings, stream
    
    def _warm_up(self):
        # 创建随机输入用于预热
        input_size = trt.volume(self.input_shape)
        warm_up_input = np.random.randn(input_size).astype(self.inputs[0]["dtype"])
        
        for i in range(self.warm_up_times):
            _ = self._infer_internal(warm_up_input)
    
    def _infer_internal(self, input_data):
        """内部推理函数，返回原始输出数据"""
        # 复制输入数据到设备
        np.copyto(self.inputs[0]["host"], input_data)
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings, 
            stream_handle=self.stream.handle
        )
        
        # 复制输出数据到主机
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        
        self.stream.synchronize()
        
        # 返回输出数据
        return [out["host"] for out in self.outputs]
    
    def infer(self, input_tensor):
        """
        执行推理并返回关键点和分数
        
        Args:
            input_tensor: 输入张量，形状需与初始化时的input_shape一致
        
        Returns:
            keypoints: 关键点坐标，形状 (17, 2)
            scores: 关键点分数，形状 (17,)
        """
        # 检查输入形状
        if input_tensor.shape != self.input_shape:
            raise ValueError(f"输入形状 {input_tensor.shape} 与引擎期望的 {self.input_shape} 不匹配")
        
        # 执行推理
        output_data = self._infer_internal(input_tensor.ravel())
        
        # 解析输出
        scores = output_data[0].reshape(1, 17)[0]
        keypoints = output_data[1].reshape(1, 17, 2)[0]
        
        return keypoints, scores
    
    def preprocess(self, img: np.array, target_size: tuple = (192, 256), 
                                        mean: list = [123.675, 116.28, 103.53], 
                                        std: list = [58.395, 57.12, 57.375]):

        # 1. 确保图像是3通道
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 灰度图转RGB
        
        # 2. 调整图像大小
        image = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 3. 转换为浮点数
        image = image.astype(np.float32)
        
        # 4. 标准化（直接使用像素值范围）
        image = (image - np.array(mean)) / np.array(std)
        
        # 5. 通道交换 (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # 6. 添加batch维度 (C, H, W) -> (1, C, H, W)
        tensor = np.expand_dims(image, axis=0)

        return tensor
    
    def postprocess(self, keypoints, scores):
        return np.concatenate([keypoints, np.expand_dims(scores, axis=1)], axis=1) 
    

    def __del__(self):
        try:
            # 释放CUDA流
            if hasattr(self, 'stream') and self.stream is not None:
                self.stream.synchronize()
                del self.stream
            
            # 释放设备和主机内存
            if hasattr(self, 'inputs') and self.inputs:
                for inp in self.inputs:
                    if 'device' in inp and inp['device'] is not None:
                        inp['device'].free()
            if hasattr(self, 'outputs') and self.outputs:
                for out in self.outputs:
                    if 'device' in out and out['device'] is not None:
                        out['device'].free()
            
            # 释放执行上下文和引擎
            if hasattr(self, 'context') and self.context is not None:
                del self.context
            if hasattr(self, 'engine') and self.engine is not None:
                del self.engine
            
        except Exception as e:
            pass

    
