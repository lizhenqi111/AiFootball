import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import time
from PIL import Image

class KPRTrtInference:
    def __init__(self, engine_path, parts_num=5):
        """
        初始化 TensorRT 推理引擎
        :param engine_path: TensorRT 引擎文件路径
        :param parts_num: 部分数量 (默认为8)
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.parts_num = parts_num
        
        # 获取输入输出绑定信息
        self.bindings = []
        self.input_shapes = {}
        self.output_shapes = {}
        
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            if self.engine.binding_is_input(binding):
                self.input_shapes[binding] = self.engine.get_binding_shape(binding_idx)
            else:
                self.output_shapes[binding] = self.engine.get_binding_shape(binding_idx)
        
        # 分配内存
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()
        
        # 预热推理
        self._warmup()
    
    def _load_engine(self, engine_path):
        """加载 TensorRT 引擎"""
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())
    
    def _allocate_buffers(self):
        """分配输入输出内存"""
        inputs = []
        outputs = []
        bindings = []
        
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            shape = self.engine.get_binding_shape(binding_idx)
            
            # 处理动态维度
            if -1 in shape:
                # 设置默认尺寸 (256x128)
                shape = [s if s != -1 else (256 if i == 2 else 128) for i, s in enumerate(shape)]
                if binding == "target_mask":
                    shape[1] = self.parts_num + 1
            
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({
                    "host": host_mem,
                    "device": device_mem,
                    "shape": shape,
                    "dtype": dtype,
                    "name": binding
                })
            else:
                outputs.append({
                    "host": host_mem,
                    "device": device_mem,
                    "shape": shape,
                    "dtype": dtype,
                    "name": binding
                })
        
        return inputs, outputs, bindings
    
    def _warmup(self, iterations=10):
        """预热推理引擎"""
        for _ in range(iterations):
            for inp in self.inputs:
                np.copyto(inp["host"], np.random.randn(*inp["shape"]).astype(inp["dtype"]).ravel())
                cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
            
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )
            
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            
            self.stream.synchronize()
    
    def preprocess_image(self, image, target_size=(128, 256)):
        """
        预处理输入图像
        :param image_path: 图像路径
        :param target_size: 目标尺寸 (width, height)
        :return: 预处理后的图像张量
        """
        # 加载图像
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            img = img.resize(target_size)
            img = np.array(img)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
        
        # 转换为numpy数组并归一化
        img = img.astype(np.float32) / 255.0
        
        # 标准化 (ImageNet均值和标准差)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # 转换为CHW格式
        img = img.transpose(2, 0, 1)
        
        # 添加批次维度
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def create_prompt_mask(self, height=256, width=128, channels=10):
        """
        创建目标掩码 (简单示例)
        :param height: 掩码高度
        :param width: 掩码宽度
        :return: 目标掩码张量
        """
        # 创建简单的水平条纹掩码 (实际应用中应使用真实掩码)
        mask = np.zeros((1, channels, height, width), dtype=np.float32)
        
        # 背景通道 (索引0)
        mask[:, 0, :, :] = 1.0
        
        # 部分通道 (索引1到K)
        stripe_height = height // self.parts_num
        for i in range(self.parts_num):
            start = i * stripe_height
            end = (i + 1) * stripe_height
            if i == self.parts_num - 1:  # 最后一个部分
                end = height
            mask[:, i + 1, start:end, :] = 1.0
        
        return mask
    
    def infer(self, image):
        """
        执行推理
        :param image: 预处理后的图像 (numpy数组)
        :param target_mask: 目标掩码 (numpy数组)
        :return: 推理结果字典
        """
        # 设置动态形状 (如果需要)
        if any(-1 in shape for shape in [self.context.get_binding_shape(i) for i in range(self.engine.num_bindings)]):
            self.context.set_binding_shape(0, image.shape)
            # self.context.set_binding_shape(1, prompt_mask.shape)
        
        # 复制输入数据到主机内存
        np.copyto(self.inputs[0]["host"], image.ravel())
        # np.copyto(self.inputs[1]["host"], prompt_mask.ravel())
        
        # 传输数据到设备
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        # cuda.memcpy_htod_async(self.inputs[1]["device"], self.inputs[1]["host"], self.stream)
        
        # 执行推理
        start_time = time.time()
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 回传结果到主机
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        
        self.stream.synchronize()
        inference_time = time.time() - start_time
        
        # 整理输出结果
        results = {}
        for out in self.outputs:
            # 获取原始形状
            if -1 in out["shape"]:
                shape = self.context.get_binding_shape(self.engine.get_binding_index(out["name"]))
            else:
                shape = out["shape"]
            
            # 重塑输出数组
            results[out["name"]] = out["host"].reshape(shape)
        
        return results, inference_time
    
    def __del__(self):
        # 同步流以确保所有操作完成
        if self.stream:
            self.stream.synchronize()
        
        # 释放设备端内存
        for inp in self.inputs:
            if inp and inp["device"]:
                inp["device"].free()
        for out in self.outputs:
            if out and out["device"]:
                out["device"].free()
        
        # 释放流资源
        if self.stream:
            del self.stream
        
        # 释放TensorRT资源
        if self.context:
            del self.context
        if self.engine:
            del self.engine
        
        # 清除引用
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.context = None
        self.engine = None
        self.stream = None