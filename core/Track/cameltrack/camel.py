import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorrt as trt
from onnxsim import simplify
import pycuda.driver as cuda
from collections import OrderedDict
import pickle

class CamelInferencer:
    def __init__(self, engine_path, logger=None):
        self.logger = logger or trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 存储绑定信息和最大形状
        self.bindings_info = []
        self.max_shapes = {}

        self.out_embedding_dim = 1024
        
        # 获取最大形状并分配内存
        self.initialize_memory()

        self._warm_up()
        
    def load_engine(self, engine_path):
        """加载TensorRT引擎"""
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())
    
    def initialize_memory(self):
        """初始化内存分配（使用最大形状）"""
        # 为每个绑定分配最大形状所需的内存
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            is_input = self.engine.binding_is_input(i)

            if is_input:
                max_shape = self.engine.get_profile_shape(0, name)[2]
                self.max_shapes[name] = max_shape
                
                # 计算最大形状下的元素数量
                max_elements = int(np.prod(max_shape))
                
                # 分配主机内存（分页锁定）
                host_mem = cuda.pagelocked_empty(max_elements, dtype)
                
                # 分配设备内存
                element_size = np.dtype(dtype).itemsize
                device_mem = cuda.mem_alloc(max_elements * element_size)
                
                self.bindings_info.append({
                    'name': name,
                    'dtype': dtype,
                    'is_input': is_input,
                    'device_mem': device_mem,
                    'host_mem': host_mem,
                    'max_shape': max_shape,
                    'max_elements': max_elements
                })
            else:
                max_shape = [*(self.bindings_info[0]["max_shape"][0:2]), self.out_embedding_dim]
                self.max_shapes[name] = max_shape

                # 计算最大形状下的元素数量
                max_elements = int(np.prod(max_shape))
                
                # 分配主机内存（分页锁定）
                host_mem = cuda.pagelocked_empty(max_elements, dtype)

                # 分配设备内存
                element_size = np.dtype(dtype).itemsize
                device_mem = cuda.mem_alloc(max_elements * element_size)

                self.bindings_info.append({
                    'name': name,
                    'dtype': dtype,
                    'is_input': is_input,
                    'device_mem': device_mem,
                    'host_mem': host_mem,
                    'max_shape': max_shape,
                    'max_elements': max_elements
                })
        
        # 准备绑定指针列表
        self.bindings = [int(b['device_mem']) for b in self.bindings_info]
    
    def infer(self, input_data, sim_threshold):
        """执行推理（使用预分配的最大内存）"""
        # 1. 为输入设置实际形状
        for name, tensor in input_data.items():
            shape = tuple(tensor.shape)
            
            # 验证形状是否在最大范围内
            max_shape = self.max_shapes[name]
            if any(s > m for s, m in zip(shape, max_shape)):
                raise ValueError(f"Shape {shape} exceeds max shape {max_shape} for {name}")
            
            # 确保dtype和申请的内存类型是一致的
            if tensor.dtype != self.get_binding(name)['dtype']:
                input_data[name] = tensor.astype(self.get_binding(name)['dtype'])
            
            # 设置执行上下文的绑定形状
            if not self.context.set_binding_shape(
                self.engine.get_binding_index(name), shape
            ):
                raise RuntimeError(f"Failed to set shape {shape} for {name}")
        
        # 2. 复制输入数据到主机内存
        for name, tensor in input_data.items():
            # print(name, ":",    tensor.shape)
            binding = self.get_binding(name)
            
            # 获取实际元素数量
            actual_elements = np.prod(tensor.shape)
            
            # 复制数据到预分配的主机内存
            np.copyto(binding['host_mem'][:actual_elements], tensor.ravel())
        
        # 3. 复制输入数据到设备
        for name in input_data.keys():
            binding = self.get_binding(name)
            actual_elements = np.prod(self.context.get_binding_shape(self.engine.get_binding_index(name)))
            cuda.memcpy_htod_async(
                binding['device_mem'], 
                binding['host_mem'],
                self.stream
            )
        
        # 4. 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 5. 复制输出数据回主机
        results = {}
        for binding in self.bindings_info:
            if not binding['is_input']:
                # 获取实际输出形状
                shape = tuple(self.context.get_binding_shape(self.engine.get_binding_index(binding['name'])))
                actual_elements = np.prod(shape)
                
                # 复制数据
                cuda.memcpy_dtoh_async(
                    binding['host_mem'], 
                    binding['device_mem'], 
                    self.stream
                )
                
                # 重塑为正确形状
                results[binding['name']] = binding['host_mem'][:actual_elements].reshape(shape)
        
        # 6. 同步流
        self.stream.synchronize()

        # 7 计算欧几里德相似矩阵
        embs_dets = results['embs_dets']
        embs_tracks = results['embs_tracks']
        masks_det = input_data['masks_det']
        masks_track = input_data['masks_track']
        td_sim_matrix = self.norm_euclidean_sim_matrix_np(embs_tracks, masks_track, embs_dets, masks_det)

        # 8.匈牙利指派 association

        association_matrix, association_result = self.hungarian_algorithm(td_sim_matrix, valid_tracks=masks_track, valid_dets=masks_det, sim_threshold=sim_threshold)

        return association_matrix, association_result, td_sim_matrix
    
    def get_binding(self, name):
        """获取绑定信息"""
        for binding in self.bindings_info:
            if binding['name'] == name:
                return binding
        raise ValueError(f"Binding {name} not found")

    def __del__(self):
        """清理资源"""
        self.destroy()
    
    def destroy(self):
        # 同步流以确保所有操作完成
        if self.stream:
            self.stream.synchronize()
        
        # 释放设备内存
        for binding in self.bindings_info:
            if binding['device_mem']:
                binding['device_mem'].free()
        
        # 释放流资源
        if self.stream:
            del self.stream
        
        # 释放TensorRT资源
        if self.context:
            del self.context
        if self.engine:
            del self.engine
        
        # 清除引用
        self.bindings_info = None
        self.bindings = None
        self.context = None
        self.engine = None
        self.stream = None
        self.max_shapes = None

    
    def euclidean_sim_matrix_np(self, track_embs, track_masks, det_embs, det_masks):
        """
        计算轨迹和检测之间的欧几里得相似度矩阵（NumPy版本）
        
        参数:
        track_embs: NumPy数组 [B, T(+P), E]
        track_masks: NumPy数组 [B, T(+P)] (布尔型)
        det_embs: NumPy数组 [B, D(+P), E]
        det_masks: NumPy数组 [B, D(+P)] (布尔型)
        
        返回:
        td_sim_matrix: NumPy数组 [B, T(+P), D(+P)]
            填充的位置设为负无穷
        """
        # 确保输入为NumPy数组
        track_embs = np.asarray(track_embs)
        track_masks = np.asarray(track_masks, dtype=bool)
        det_embs = np.asarray(det_embs)
        det_masks = np.asarray(det_masks, dtype=bool)
        
        # 验证输入形状
        B, T_plus_P, E = track_embs.shape
        B_det, D_plus_P, E_det = det_embs.shape
        assert B == B_det and E == E_det, "输入形状不匹配"
        
        # 计算欧几里得距离矩阵
        # 扩展维度以支持广播
        track_embs_3d = track_embs[:, :, np.newaxis, :]  # [B, T, 1, E]
        det_embs_3d = det_embs[:, np.newaxis, :, :]      # [B, 1, D, E]
        
        # 计算差值并平方
        diff = track_embs_3d - det_embs_3d
        squared_diff = np.square(diff)
        
        # 沿特征维度求和并开平方
        distances = np.sqrt(np.sum(squared_diff, axis=-1))  # [B, T, D]
        
        # 将距离转换为相似度（取负值）
        sim_matrix = -distances
        
        # 应用掩码：非有效位置设为负无穷
        # 扩展掩码以匹配相似度矩阵的维度
        track_masks_3d = track_masks[:, :, np.newaxis]    # [B, T, 1]
        det_masks_3d = det_masks[:, np.newaxis, :]        # [B, 1, D]
        valid_mask = track_masks_3d * det_masks_3d        # [B, T, D]
        
        # 使用掩码设置无效位置
        sim_matrix = np.where(valid_mask, sim_matrix, -np.inf)
        
        return sim_matrix
    
    def norm_euclidean_sim_matrix_np(self, track_embs, track_masks, det_embs, det_masks):
        """
        计算归一化欧几里得相似度矩阵（NumPy版本）
        
        参数:
        track_embs: NumPy数组 [B, T(+P), E]
        track_masks: NumPy数组 [B, T(+P)] (布尔型)
        det_embs: NumPy数组 [B, D(+P), E]
        det_masks: NumPy数组 [B, D(+P)] (布尔型)
        
        返回:
        td_sim_matrix: NumPy数组 [B, T(+P), D(+P)]
            填充的位置设为负无穷
        """
        # 确保输入为NumPy数组
        track_embs = np.asarray(track_embs, dtype=np.float32)
        track_masks = np.asarray(track_masks, dtype=bool)
        det_embs = np.asarray(det_embs, dtype=np.float32)
        det_masks = np.asarray(det_masks, dtype=bool)
        
        # 验证输入形状
        B, T_plus_P, E = track_embs.shape
        B_det, D_plus_P, E_det = det_embs.shape
        assert B == B_det and E == E_det, "输入形状不匹配"
        
        # 向量归一化（L2范数）
        track_embs_norm = track_embs / np.linalg.norm(track_embs, axis=-1, keepdims=True)
        det_embs_norm = det_embs / np.linalg.norm(det_embs, axis=-1, keepdims=True)
        
        # 处理零向量（避免除以零）
        track_embs_norm = np.nan_to_num(track_embs_norm)
        det_embs_norm = np.nan_to_num(det_embs_norm)
        
        # 计算欧几里得距离矩阵
        track_expanded = track_embs_norm[:, :, np.newaxis, :]  # [B, T, 1, E]
        det_expanded = det_embs_norm[:, np.newaxis, :, :]     # [B, 1, D, E]
        
        diff = track_expanded - det_expanded
        squared_diff = np.square(diff)
        distances = np.sqrt(np.sum(squared_diff, axis=-1))  # [B, T, D]
        
        # 转换为相似度：1 - 距离/2
        sim_matrix = 1 - distances / 2
        
        # 应用掩码：非有效位置设为负无穷
        track_masks_3d = track_masks[:, :, np.newaxis]    # [B, T, 1]
        det_masks_3d = det_masks[:, np.newaxis, :]        # [B, 1, D]
        valid_mask = track_masks_3d * det_masks_3d        # [B, T, D]
        
        sim_matrix = np.where(valid_mask, sim_matrix, -np.inf)
        
        return sim_matrix
    
    def hungarian_algorithm(self, td_sim_matrix, valid_tracks, valid_dets, sim_threshold=0.0, **kwargs):
        """
        apply hungarian algorithm on sim_matrix with the entries in valid_dets and valid_tracks

        td_sim_matrix: float32 numpy array [B, T, D]
        valid_tracks: bool numpy array [B, T]
            True is valid False otherwise
        valid_dets: bool numpy array [B, D]
            True is valid False otherwise
        :return: association_matrix: bool numpy array [B, T, D]
            True if the pair value is associated False otherwise
        """
        B, T, D = td_sim_matrix.shape
        association_matrix = np.zeros_like(td_sim_matrix, dtype=bool)
        association_result = []
        
        for b in range(B):
            if valid_tracks[b].sum() > 0 and valid_dets[b].sum() > 0:
                # 应用掩码并处理相似度阈值
                sim_matrix_masked = td_sim_matrix[b, valid_tracks[b], :][:, valid_dets[b]]
                sim_matrix_masked[sim_matrix_masked < sim_threshold] = sim_threshold - 1e-5
                
                # 应用匈牙利算法（注意这里使用负号，因为我们要最大化相似度）
                row_idx, col_idx = linear_sum_assignment(-sim_matrix_masked)
                
                # 获取有效索引
                valid_rows = np.nonzero(valid_tracks[b])[0]
                valid_cols = np.nonzero(valid_dets[b])[0]
                
                # 构建匹配对
                matched_td_indices = np.array(list(zip(valid_rows[row_idx], valid_cols[col_idx])))
                
                # 找出未匹配的跟踪器和检测
                matched_tracks = set(valid_rows[row_idx])
                matched_dets = set(valid_cols[col_idx])
                unmatched_trackers = [t for t in valid_rows if t not in matched_tracks]
                unmatched_detections = [d for d in valid_cols if d not in matched_dets]
                
                # 过滤掉相似度低于阈值的匹配
                matches = []
                for m in matched_td_indices:
                    if td_sim_matrix[b, m[0], m[1]] < sim_threshold:
                        unmatched_trackers.append(m[0])
                        unmatched_detections.append(m[1])
                    else:
                        association_matrix[b, m[0], m[1]] = True
                        matches.append(m)
                matched_td_indices = np.array(matches)
            else:
                # 处理没有有效匹配的情况
                matched_td_indices = np.empty((0, 2), dtype=int)
                unmatched_trackers = []
                unmatched_detections = []
                
                if valid_tracks[b].sum() > 0:
                    unmatched_trackers = np.nonzero(valid_tracks[b])[0].tolist()
                elif valid_dets[b].sum() > 0:
                    unmatched_detections = np.nonzero(valid_dets[b])[0].tolist()
            
            # 存储当前批次的匹配结果
            association_result.append({
                "matched_td_indices": matched_td_indices,
                "unmatched_trackers": unmatched_trackers,
                "unmatched_detections": unmatched_detections,
            })
        
        return association_matrix, association_result
    
    def _warm_up(self, iterations=3, input_shape=None):
        """
        执行热启动（预热）以初始化引擎和设备
        
        Args:
            iterations: 预热迭代次数
            input_shape: 用于预热的输入形状，若为None则使用最大形状
        """
        # 创建随机输入数据
        input_data = {}
        for binding in self.bindings_info:
            if binding['is_input']:
                if input_shape is not None:
                    shape = input_shape
                else:
                    shape = binding['max_shape']
                
                # 创建随机输入
                input_data[binding['name']] = np.random.randn(*shape).astype(binding['dtype'])
        
        # 执行预热迭代
        for i in range(iterations):
            self.infer(input_data, sim_threshold=0.5)
    


class Camel:
    index = 0
    def __init__(self, engine, sim_threshold=0, image_size = (1920, 1080)):
        self.inferencer = CamelInferencer(engine_path=engine)
        self.sim_threshold = sim_threshold

        self.image_size = image_size

    def predict_preprocess(self, batch):
        """
        :param batch:
            dict of tensors containing the inputs features and targets of detections and tracklets
        :return:
            dets: Detections - a dataclass wrapper containing batch infos for detections
            tracks: Tracklets - a dataclass wrapper containing batch infos for tracklets
        """

        batch = self.norm_coords(batch, self.image_size[0], self.image_size[1])
        batch['masks_det'] = np.any(batch['feats_masks_det'], axis=-1)
        batch['masks_track'] = np.any(batch['feats_masks_track'], axis=-1)
        return batch
    
    def norm_coords(self, batch, img_width, img_height):
        return self.positive_bboxes_keypoints(batch, img_width, img_height)

    def positive_bboxes_keypoints(self, batch, img_width, img_height):
        def normalize_bbox(bbox, W, H):
            bbox[..., 0] = bbox[..., 0] / W
            bbox[..., 1] = bbox[..., 1] / H
            bbox[..., 2] = bbox[..., 2] / W
            bbox[..., 3] = bbox[..., 3] / H
            return bbox

        def normalize_keypoints(kps, W, H):
            kps[..., 0] = kps[..., 0] / W
            kps[..., 1] = kps[..., 1] / H
            return kps


        if "bbox_tlwh" in batch:
            batch["bbox_tlwh_det"] = normalize_bbox(batch["bbox_ltwh"],
                                                    img_width,
                                                    img_height)
        if "keypoints_xyc_det" in batch:
            batch["keypoints_xyc_det"] = normalize_keypoints(batch["keypoints_xyc_det"],
                                                            img_width,
                                                            img_height)

        if "bbox_tlwh_track" in batch:
            batch["bbox_tlwh_track"] = normalize_bbox(batch["bbox_tlwh_track"],
                                                            img_width,
                                                            img_height)
        if "keypoints_xyc_track" in batch:
            batch["keypoints_xyc_track"] = normalize_keypoints(batch["keypoints_xyc_track"],
                                                                img_width,
                                                                img_height)
        return batch
    
    def predict(self, batch):

        batch = OrderedDict({
        "embeddings_det": batch['det_feats']['embeddings'],
        "visibility_scores_det": batch['det_feats']['visibility_scores'],
        "keypoints_xyc_det": batch['det_feats']['keypoints_xyc'],
        "bbox_tlwh_det": batch['det_feats']['bbox_tlwh'],
        "bbox_conf_det": batch['det_feats']['bbox_conf'],
        "feats_masks_det": batch['det_masks'],
        "ages_det": batch['det_feats']['age'],
        
        "embeddings_track": batch['track_feats']['embeddings'],
        "visibility_scores_track": batch['track_feats']['visibility_scores'],
        "keypoints_xyc_track": batch['track_feats']['keypoints_xyc'],
        "bbox_tlwh_track": batch['track_feats']['bbox_tlwh'],
        "bbox_conf_track": batch['track_feats']['bbox_conf'],
        "feats_masks_track": batch['track_masks'],
        "ages_track": batch['track_feats']['age'],
        })
        # print(batch['bbox_conf_track'])
        # print(batch['ages_track'])
        batch = self.predict_preprocess(batch)
        # with open(f'{Camel.index}_data.pkl', 'wb') as f:
        #     pickle.dump(batch, f)

        association_matrix, association_result, td_sim_matrix = self.inferencer.infer(batch, self.sim_threshold)

        return association_matrix, association_result, td_sim_matrix