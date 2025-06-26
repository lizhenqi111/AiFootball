import numpy as np
import cv2

class Compose:
    """自定义实现的Compose类，用于构建数据处理流水线"""
    def __init__(self, transforms):
        self.transforms = []
        for transform in transforms:
            transform_type = transform.pop('type')
            if transform_type == 'LoadImage':
                self.transforms.append(LoadImage())
            elif transform_type == 'GetBBoxCenterScale':
                self.transforms.append(GetBBoxCenterScale())
            elif transform_type == 'TopdownAffine':
                self.transforms.append(TopdownAffine(**transform))
            elif transform_type == 'PackPoseInputs':
                self.transforms.append(PackPoseInputs())
            else:
                raise ValueError(f"不支持的变换类型: {transform_type}")
    
    def __call__(self, data: np.array):
        for transform in self.transforms:
            data = transform(data)
        return data

class LoadImage:
    """加载图像的变换类"""
    def __call__(self, results):
        img = results['img']
        img = img.transpose(2, 0, 1)  # 转换为(C, H, W)格式
        
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

class GetBBoxCenterScale:
    """从边界框计算中心点和缩放因子的变换类"""
    def __init__(self, padding=1.5):
        self.padding = padding  # 边距系数
    
    def __call__(self, results):
        bbox = results['bbox']  # 格式：[x, y, w, h]
        
        # 计算中心点
        center = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2], dtype=np.float32)
        
        # 使用更长边并增加边距，确保包含完整人体
        w, h = bbox[2], bbox[3]
        # aspect_ratio = 192 / 256  # 目标宽高比
        # if w > aspect_ratio * h:
        #     h = w / aspect_ratio
        # else:
        #     w = h * aspect_ratio
        
        # 计算缩放因子，调整边距系数
        scale = np.array([w, h], dtype=np.float32) * self.padding
        
        results['center'] = center
        results['scale'] = scale
        return results

class TopdownAffine:
    """基于中心点和缩放因子进行仿射变换的变换类"""
    def __init__(self, input_size):
        self.input_size = input_size  # (width, height)
    
    def _get_affine_transform(self, center, scale, rot, output_size):
        """获取仿射变换矩阵"""
        src_w, src_h = scale[0], scale[1]
        dst_w, dst_h = output_size
        
        # 旋转中心
        rot_rad = np.pi * rot / 180
        src_dir = self._get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)
        
        # 三个源点（使用图像坐标系）
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])
        
        # 三个目标点
        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])
        
        # 计算仿射变换矩阵
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans
    
    def _get_dir(self, src_point, rot_rad):
        """计算旋转后的方向"""
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return src_result
    
    def _get_3rd_point(self, a, b):
        """计算仿射变换所需的第三个点"""
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    
    def __call__(self, results):
        img = results['img'].transpose(1, 2, 0)  # 转回(H, W, C)格式
        center = results['center']
        scale = results['scale']
        
        # 获取仿射变换矩阵
        trans = self._get_affine_transform(center, scale, 0, self.input_size)
        
        # 应用仿射变换
        dst_img = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[0]), int(self.input_size[1])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,  # 使用复制边界代替常数填充
        )
        
        # 转回(C, H, W)格式
        dst_img = dst_img.transpose(2, 0, 1)
        
        results['img'] = dst_img
        results['img_shape'] = dst_img.shape
        results['trans'] = trans
        return results
    
class PackPoseInputs:
    """打包处理后的数据为模型输入格式的变换类"""
    def __call__(self, results):
        # 确保图像是float32类型并归一化
        img = results['img'].astype(np.uint8)
        
        # 添加批次维度 (C, H, W) -> (1, C, H, W)
        img = np.expand_dims(img, axis=0)
        
        # 创建数据样本字典
        data_samples = {
            'img': img,
            'center': results['center'],
            'scale': results['scale'],
            'img_shape': results['img_shape'],
            'ori_shape': results['ori_shape'],
            'trans': results['trans']
        }
        
        return {
            'inputs': img,
            'data_samples': data_samples
        }
    
class KeypointMapper:
    """关键点坐标映射工具类"""
    
    @staticmethod
    def get_inverse_transform(trans):
        """计算仿射变换矩阵的逆矩阵"""
        return cv2.invertAffineTransform(trans)
    
    @staticmethod
    def map_points(points, trans):
        """
        将点坐标通过仿射矩阵变换
        
        参数:
            points: 关键点坐标，形状为 (N, 2)，表示 N 个点的 (x, y)
            trans: 仿射变换矩阵，形状为 (2, 3)
        
        返回:
            变换后的点坐标，形状为 (N, 2)
        """
        # 确保输入是 numpy 数组
        points = np.array(points, dtype=np.float32)
        
        # 添加齐次坐标维度
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        
        # 应用变换矩阵 (N, 3) × (3, 2) → (N, 2)
        transformed_points = np.dot(points_homogeneous, trans.T)
        
        return transformed_points
    
    @staticmethod
    def map_keypoints_back(keypoints, trans):
        """
        将处理后的图像中的关键点映射回原始图像
        
        参数:
            keypoints: 处理后的关键点坐标，形状为 (N, 2) 或 (N, 3)（包含置信度）
            trans: 仿射变换矩阵，形状为 (2, 3)
        
        返回:
            原始图像中的关键点坐标
        """
        # 获取逆变换矩阵
        inv_trans = KeypointMapper.get_inverse_transform(trans)
        
        # 如果关键点包含置信度列，分离坐标和置信度
        has_confidence = keypoints.shape[1] == 3
        if has_confidence:
            coords = keypoints[:, :2]
            confidence = keypoints[:, 2:]
        else:
            coords = keypoints
        
        # 映射坐标回原始图像
        original_coords = KeypointMapper.map_points(coords, inv_trans)
        
        # 重新组合坐标和置信度（如果有）
        if has_confidence:
            return np.hstack([original_coords, confidence])
        else:
            return original_coords

def draw_keypoints_on_image(image_path, keypoints, skeleton=None, save_path="output.png"):
    """
    在图像上绘制关键点和骨骼连接
    
    参数:
        image_path: 原始图像路径
        keypoints: 关键点坐标，形状为 (N, 3)，包含 (x, y, confidence)
        skeleton: 骨骼连接列表，每个连接是一个二元组 (point_index1, point_index2)
        save_path: 保存图像的路径
    """
    # 加载原始图像
    image = cv2.imread(image_path)
    
    # 定义关键点颜色（BGR格式）
    colors = [
        (0, 255, 0),   # 绿色 - 头部
        (0, 0, 255),   # 红色 - 肩部
        (255, 0, 0),   # 蓝色 - 肘部
        (0, 255, 255), # 黄色 - 腕部
        (255, 0, 255), # 紫色 - 髋部
        (255, 255, 0), # 青色 - 膝部
        (0, 128, 255), # 橙色 - 踝部
    ]
    
    # 绘制关键点
    for i, kpt in enumerate(keypoints):
        x, y, conf = kpt
        if conf > 0.3:  # 只绘制置信度大于阈值的点
            color_idx = i % len(colors)
            cv2.circle(image, (int(x), int(y)), 5, colors[color_idx], -1)
            # 可选：在关键点旁边标注索引
            cv2.putText(image, str(i), (int(x)+5, int(y)+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[color_idx], 1)
    
    # 如果提供了骨骼连接信息，绘制骨骼
    if skeleton is not None:
        for connection in skeleton:
            idx1, idx2 = connection
            if keypoints[idx1, 2] > 0.3 and keypoints[idx2, 2] > 0.3:  # 两个点的置信度都要足够高
                cv2.line(image, 
                         (int(keypoints[idx1, 0]), int(keypoints[idx1, 1])),
                         (int(keypoints[idx2, 0]), int(keypoints[idx2, 1])),
                         (255, 255, 255), 2)  # 白色线条
    
    # 保存结果图像
    cv2.imwrite(save_path, image)
    print(f"已将关键点绘制在图像上并保存至: {save_path}")
    
    return image

# 预处理人体图像

def preprocess(img: np.array, target_size: tuple = (192, 256), 
                                    mean: list = [123.675, 116.28, 103.53], 
                                    std: list = [58.395, 57.12, 57.375]):
    
    # img :(b, n, h, w) (h,w) = target_size

    
    img = img[:, ::-1, :, :].astype(np.float32)  # 反转通道维度
    
    # 标准化（直接使用像素值范围）
    mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1) 
    std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
    img = (img - mean) / std

    # tensor = np.expand_dims(image, axis=0)

    return img

# 使用示例
if __name__ == "__main__":
    # 构建流水线
    transform = Compose([
        {'type': 'LoadImage'},
        {'type': 'GetBBoxCenterScale'},
        {'type': 'TopdownAffine', 'input_size': (192, 256)},
        {'type': 'PackPoseInputs'}
    ])
    
    # 准备输入数据
    data = {
        'img_path': 'test.png',
        'bbox': [457, 543, 65, 134]  # [x, y, w, h]
    }
    
    # 获取人体图像
    processed_data = transform(data)
    cv2.imwrite("exa.png", processed_data['inputs'].transpose(0,2,3,1)[0])

    # 检测
    keypoints = np.array([[0,0,0.87], [192, 256, 0.85]])

    # 使用KeypointMapper将关键点映射回原始图像
    original_keypoints = KeypointMapper.map_keypoints_back(keypoints, processed_data['data_samples']['trans'])
    
    # 将original_keypoints画在原始图像上
    draw_keypoints_on_image('test.png', keypoints=original_keypoints)

