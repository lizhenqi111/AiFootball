import numpy as np

def parse_detection_results(boxes_array):
    """
    解析目标检测结果数组
    
    Args:
        boxes_array: 形状为(n, 6)的NumPy数组，每行格式为[x1,y1,x2,y2,conf,classid]
    
    Returns:
        boxes: 边界框坐标数组 (n,4)
        confidences: 置信度数组 (n,)
        class_ids: 类别ID数组 (n,)
    """
    if boxes_array.ndim != 2 or boxes_array.shape[1] != 6:
        raise ValueError("输入数组必须是形状为(n,6)的NumPy数组")
    
    boxes = boxes_array[:, :4]
    confidences = boxes_array[:, 4]
    class_ids = boxes_array[:, 5].astype(int)
    
    return boxes, confidences, class_ids

def filter_boxes(boxes, confidences, class_ids, 
                 iou_threshold=0.3, containment_threshold=0.7,
                 score_threshold=0.5, by_class=True):
    """
    过滤包含和交叠的边界框
    
    Args:
        boxes: 边界框坐标数组 (n,4)
        confidences: 置信度数组 (n,)
        class_ids: 类别ID数组 (n,)
        iou_threshold: 交叠IOU阈值
        containment_threshold: 包含比例阈值
        score_threshold: 置信度过滤阈值
        by_class: 是否按类别分别处理
    
    Returns:
        filtered_boxes: 过滤后的边界框
        filtered_scores: 过滤后的置信度
        filtered_class_ids: 过滤后的类别ID
    """
    # 置信度过滤
    mask = confidences >= score_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # 按类别分组处理
    if by_class:
        unique_classes = np.unique(class_ids)
        filtered_boxes_list, filtered_scores_list, filtered_class_ids_list = [], [], []
        
        for class_id in unique_classes:
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = confidences[class_mask]
            
            # 处理当前类别的框
            filtered_class_boxes, filtered_class_scores = _filter_class_boxes(
                class_boxes, class_scores, iou_threshold, containment_threshold
            )
            
            filtered_boxes_list.append(filtered_class_boxes)
            filtered_scores_list.append(filtered_class_scores)
            filtered_class_ids_list.extend([class_id] * len(filtered_class_boxes))
        
        # 合并所有类别结果
        if filtered_boxes_list:
            filtered_boxes = np.vstack(filtered_boxes_list)
            filtered_scores = np.hstack(filtered_scores_list)
            filtered_class_ids = np.array(filtered_class_ids_list)
        else:
            filtered_boxes = np.array([])
            filtered_scores = np.array([])
            filtered_class_ids = np.array([])
    
    # 不按类别处理，统一过滤
    else:
        filtered_boxes, filtered_scores = _filter_class_boxes(
            boxes, confidences, iou_threshold, containment_threshold
        )
        filtered_class_ids = class_ids[:len(filtered_boxes)]
    
    return filtered_boxes, filtered_scores, filtered_class_ids

def _filter_class_boxes(boxes, scores, iou_threshold, containment_threshold):
    """处理单个类别的边界框过滤"""
    if len(boxes) == 0:
        return np.array([]), np.array([])
    
    # 按面积从大到小排序，优先保留大框
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_indices = np.argsort(-areas)  # 降序排列
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    kept_indices = []
    
    for i in range(len(boxes)):
        keep = True
        current_box = boxes[i]
        current_area = areas[sorted_indices[i]]
        
        # 检查与已保留框的重叠情况
        for j in kept_indices:
            other_box = boxes[j]
            other_area = areas[sorted_indices[j]]
            
            # 计算交集
            x1 = max(current_box[0], other_box[0])
            y1 = max(current_box[1], other_box[1])
            x2 = min(current_box[2], other_box[2])
            y2 = min(current_box[3], other_box[3])
            
            # 无交集
            if x1 >= x2 or y1 >= y2:
                continue
            
            # 计算交集面积
            overlap_area = (x2 - x1) * (y2 - y1)
            
            # 判断包含关系：小框是否大部分被大框包含
            if current_area < other_area:
                # 当前框是小框，检查是否被大框包含
                ratio = overlap_area / current_area
                if ratio >= containment_threshold:
                    keep = False
                    break
            else:
                # 其他框是小框，检查当前大框是否与小框过度交叠
                ratio = overlap_area / other_area
                if ratio >= iou_threshold:
                    keep = False
                    break
        
        if keep:
            kept_indices.append(i)
    
    # 返回保留的框和分数
    return boxes[kept_indices], scores[kept_indices]

def process_detection_results(boxes_array, 
                            iou_threshold=0.3, 
                            containment_threshold=0.7,
                            score_threshold=0.5,
                            by_class=True):
    """处理目标检测结果，过滤包含和交叠的框"""
    # 解析输入数据
    boxes, confidences, class_ids = parse_detection_results(boxes_array)
    
    # 过滤框
    filtered_boxes, filtered_scores, filtered_class_ids = filter_boxes(
        boxes, confidences, class_ids,
        iou_threshold, containment_threshold,
        score_threshold, by_class
    )
    
    # 构建输出数组 (n,6)
    if len(filtered_boxes) > 0:
        output_array = np.hstack([
            filtered_boxes,
            filtered_scores.reshape(-1, 1),
            filtered_class_ids.reshape(-1, 1)
        ])
    else:
        output_array = np.empty((0, 6), dtype=boxes_array.dtype)
    
    return output_array