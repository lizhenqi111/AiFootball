import pycuda.driver as cuda

class CUDAContextManager:
    """优化的CUDA上下文管理器，正确管理GPU资源并处理设备ID"""
    
    def __init__(self, device=0):
        """
        初始化上下文管理器
        
        Args:
            device: 设备ID（整数）或设备标识字符串（如"cuda:0"）
        """
        self.device = device
        self.context = None
        self.device_id = self._parse_device_id(device)
        self._validate_device()
    
    def _parse_device_id(self, device):
        """解析设备ID，支持整数或字符串格式"""
        if isinstance(device, str):
            # 处理"cuda:0"或"0"格式
            return int(device.split(':')[-1]) if ':' in device else int(device)
        return int(device)
    
    def _validate_device(self):
        """验证设备是否存在"""
        try:
            cuda.init()
            device_count = cuda.Device.count()
            if self.device_id >= device_count:
                raise ValueError(f"设备ID {self.device_id} 不存在，系统中只有 {device_count} 个CUDA设备")
        except cuda.Error as e:
            raise RuntimeError(f"CUDA设备验证失败: {e}") from e
    
    def __enter__(self):
        """创建并激活CUDA上下文（上下文管理器入口）"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """释放CUDA上下文（上下文管理器出口）"""
        self.stop()
    
    def start(self):
        """创建并激活CUDA上下文"""
        try:
            cuda_device = cuda.Device(self.device_id)
            self.context = cuda_device.make_context()
            return self
        except cuda.Error as e:
            raise RuntimeError(f"创建CUDA上下文失败，设备ID: {self.device_id}") from e
    
    def stop(self):
        """释放CUDA上下文"""
        if self.context:
            try:
                self.context.pop()
            except cuda.Error as e:
                print(f"释放上下文时发生错误: {e}")
            finally:
                self.context = None
    
    @property
    def is_active(self):
        """检查上下文是否激活"""
        return self.context is not None and self.context is cuda.Context.current()


# 示例用法
if __name__ == "__main__":
    # 方法1: 使用上下文管理器
    try:
        with CUDAContextManager(device="cuda:0") as ctx:
            print(f"上下文已激活，设备: {cuda.Device(ctx.device_id).name()}")
            # 在此处执行CUDA操作
    except Exception as e:
        print(f"上下文管理错误: {e}")
    
    # 方法2: 手动调用start/stop
    ctx = CUDAContextManager(device=0)
    try:
        ctx.start()
        print(f"手动激活上下文，设备: {cuda.Device(ctx.device_id).name()}")
    except Exception as e:
        print(f"手动管理错误: {e}")
    finally:
        ctx.stop()