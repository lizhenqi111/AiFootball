import cv2
import numpy as np
import time
import os


# PoseTrack18 关键点名称（供参考）
POSETRACK18_KEYPOINTS = [
    "nose", "head_bottom", "head_top",  # 头部
    "left_ear", "right_ear",            # 耳朵
    "left_shoulder", "right_shoulder",  # 肩部
    "left_elbow", "right_elbow",        # 肘部
    "left_wrist", "right_wrist",        # 腕部
    "left_hip", "right_hip",            # 髋部
    "left_knee", "right_knee",          # 膝部
    "left_ankle", "right_ankle"         # 踝部
]

# PoseTrack18 关键点连接关系（骨架结构）
POSETRACK18_CONNECTIONS = [
    (0, 1), (1, 2),    # 头部：鼻子 -> 头底部 -> 头顶
    (0, 3), (0, 4),    # 头部：鼻子 -> 左耳/右耳
    (3, 5), (4, 6),    # 上半身：左耳 -> 左肩，右耳 -> 右肩
    (5, 6),            # 肩部：左肩 -> 右肩
    (5, 7), (7, 9),    # 左臂：左肩 -> 左肘 -> 左手腕
    (6, 8), (8, 10),   # 右臂：右肩 -> 右肘 -> 右手腕
    (5, 11), (6, 12),  # 躯干：左/右肩 -> 左/右髋
    (11, 12),          # 髋部：左髋 -> 右髋
    (11, 13), (13, 15), # 左腿：左髋 -> 左膝 -> 左踝
    (12, 14), (14, 16)  # 右腿：右髋 -> 右膝 -> 右踝
]

# 关键点颜色映射（17个关键点示例）
KEYPOINT_COLORS = [
    (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255),    # 右上肢
    (255, 128, 0), (255, 170, 0), (255, 212, 0), (255, 255, 0),    # 左上肢
    (255, 0, 128), (255, 0, 170), (255, 0, 212), (255, 0, 255),    # 躯干和下肢
    (128, 0, 255), (170, 0, 255), (212, 0, 255), (255, 0, 255),    # 另一条下肢
    (77, 255, 222)                                         # 头部
]

def get_color_for_id(track_id,饱和度=0.9,亮度=0.9):
    """
    根据轨迹ID生成独特的RGB颜色
    
    Args:
        track_id: 轨迹ID
        饱和度: 颜色饱和度 (0-1)
        亮度: 颜色亮度 (0-1)
    
    Returns:
        BGR格式颜色元组 (b, g, r)
    """
    # 使用ID作为随机种子，确保相同ID颜色一致
    np.random.seed(track_id)
    
    # 从HSV空间生成颜色，确保颜色分散
    hue = np.random.rand()  # 随机色相 (0-1)
    saturation = 饱和度  # 高饱和度
    value = 亮度  # 高亮度
    
    # HSV转RGB
    c = value * saturation
    x = c * (1 - abs((hue * 6) % 2 - 1))
    m = value - c
    
    if 0 <= hue < 1/6:
        r, g, b = c, x, 0
    elif 1/6 <= hue < 2/6:
        r, g, b = x, c, 0
    elif 2/6 <= hue < 3/6:
        r, g, b = 0, c, x
    elif 3/6 <= hue < 4/6:
        r, g, b = 0, x, c
    elif 4/6 <= hue < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    # 转换为BGR格式 (OpenCV使用)
    b = int((r + m) * 255)
    g = int((g + m) * 255)
    r = int((b + m) * 255)
    
    return (b, g, r)

def visualize_detections(frame, data_dict, frame_id, output_dir="./visualizations/"):
    """
    在图像上可视化检测结果和关键点信息
    
    Args:
        frame: 输入图像数组
        data_dict: 包含检测和关键点数据的字典
        output_dir: 可视化结果保存目录
    """
    if not data_dict:
        print(f"Frame {frame_id}: 无检测结果，跳过可视化")
        return frame
    
    # 复制图像以避免修改原始数据
    vis_img = frame.copy()
    height, width = vis_img.shape[:2]
    
    # 获取检测数量
    num_detections = data_dict['num']
    
    for i in range(num_detections):
        # 1. 绘制边界框
        bbox = data_dict['bbox_tlwh'][i].astype(np.int32)
        x1, y1, x2, y2 = bbox
        conf = data_dict['bbox_conf'][i]
        class_idx = data_dict['bboxes_classidx'][i]
        
        # 边界框颜色（根据类别索引生成）
        color = (0, 255, 0) if class_idx == 0 else (255, 0, 0)  # 假设类别0为人体
        
        # 绘制矩形框
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # # 绘制置信度文本
        # conf_text = f"Conf: {conf:.2f}"
        # cv2.putText(vis_img, conf_text, (x1, y1 - 10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # # 2. 绘制关键点
        # keypoints = data_dict['keypoints_xyc'][i]
        # kps_conf = data_dict['kps_conf'][i]
        
        # for kp_idx, (kp_x, kp_y, visible) in enumerate(keypoints):
        #     if visible > 0.2 and kps_conf > 0.5:  # 仅绘制可见且置信度高的关键点
        #         kp_x, kp_y = int(kp_x), int(kp_y)
        #         # 绘制关键点圆圈
        #         cv2.circle(vis_img, (kp_x, kp_y), 3, KEYPOINT_COLORS[kp_idx], -1)
        
        # # 3. 连接关键点形成骨架
        # for (idx1, idx2) in POSETRACK18_CONNECTIONS:
        #     if keypoints[idx1, 2] > 0.2 and keypoints[idx2, 2] > 0.2:
        #         x_1, y_1 = int(keypoints[idx1, 0]), int(keypoints[idx1, 1])
        #         x_2, y_2 = int(keypoints[idx2, 0]), int(keypoints[idx2, 1])
        #         cv2.line(vis_img, (x_1, y_1), (x_2, y_2), KEYPOINT_COLORS[idx1], 2)

    # 绘制轨迹id
    for tracklet in data_dict['tracklets']:
        if tracklet['state'] == 'active':
            bbox_index = tracklet["index"]
            track_id = tracklet["track_id"]
            bbox = data_dict['bbox_tlwh'][bbox_index]
            x1, y1, x2, y2 = bbox.astype(np.int32)
            # 计算文本位置（边界框顶部或左上角）
            text_pos = (x1, y1 - 10) if y1 - 10 > 10 else (x, y2 + 20)
    
            # 绘制文本
            text = f"ID:{track_id}"
            cv2.putText(vis_img, text, text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color_for_id(track_id), 2)

        
    # 4. 绘制pitch
    for i, (x, y) in enumerate(data_dict['field_keypoint']):
        if x > 0 and y > 0:  # 过滤无效点
            cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)
            # 标注点序号
            cv2.putText(vis_img, str(i), (int(x)+5, int(y)+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 5. 绘制全局信息
    global_info = f"Frame ID: {frame_id}, Detections: {num_detections}"
    cv2.putText(vis_img, global_info, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 6. 保存可视化结果

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg")
    cv2.imwrite(save_path, vis_img)

    return vis_img