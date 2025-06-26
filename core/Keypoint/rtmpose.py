import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import cv2

# === 主类封装 ===
class RTMPose:
    def __init__(self, engine_path):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # 设置静态 input shape（可手动设定）
        self.input_shape = self.engine.get_binding_shape(0)
        self.context.set_binding_shape(0, self.input_shape)

        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine, self.context)

        self._warmup()

    def infer(self, input_tensor: np.ndarray):
        # 填充输入到 host 内存
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())

        # 推理
        output_data = self.do_inference(self.context, self.bindings, self.inputs, self.outputs, self.stream)

        # Output[0]: scores (1, 14)
        # Output[1]: keypoints (1, 14, 2)
        scores = output_data[0].reshape(1, 14)
        keypoints = output_data[1].reshape(1, 14, 2)

        return keypoints[0], scores[0]
    
    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
    def do_inference(self, context, bindings, inputs, outputs, stream):
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)

        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)

        stream.synchronize()
        return [out['host'] for out in outputs]
    
    def allocate_buffers(self, engine, context):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for binding_idx in range(engine.num_bindings):
            binding_name = engine.get_binding_name(binding_idx)

            if engine.binding_is_input(binding_idx):
                shape = context.get_binding_shape(binding_idx)
            else:
                shape = engine.get_binding_shape(binding_idx)

            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding_idx))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if engine.binding_is_input(binding_idx):
                inputs.append({'name': binding_name, 'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                outputs.append({'name': binding_name, 'host': host_mem, 'device': device_mem, 'shape': shape})

        return inputs, outputs, bindings, stream

    def preprocess(self, img):
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        img_input = img.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)[None]  # (1, 3, 256, 192)
        return img_input
    
    def postprocess(self, keypoints, scores):
        return np.concatenate([keypoints, np.expand_dims(scores, axis=1)], axis=1) 
    
    def _warmup(self, iterations=3):
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(iterations):
            _, _ = self.infer(dummy_input)
    


if __name__ == '__main__':
    engine_path = "rmtpose.engine"
    input_shape = (1, 3, 256, 192)  # (B, C, H, W)

    img = cv2.imread("demo.png")
    img = cv2.resize(img, (192, 256))  # 根据你的输入模型设定
    img_input = img.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)[None]  # (1, 3, 256, 192)

    # 加载模型 & 推理
    predictor = RTMPose("rmtpose.engine")
    keypoints, scores = predictor.infer(img_input)

    # 输出结果
    print("Keypoints:", keypoints.shape)
    print("Scores:", scores.shape)

    # 可视化（可选）
    for (x, y), s in zip(keypoints, scores):
        if s > 0.3:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imwrite("output.jpg", img)

