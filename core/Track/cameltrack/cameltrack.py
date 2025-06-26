import logging
import numpy as np

from .camel import Camel

log = logging.getLogger(__name__)

def collate_fn(batch):  # FIXME collate_fn could handle a part of the preprocessing
    """
    :param batch: [(idxs,  [Detection, ...])]
    :return: ([idxs], [Detection, ...])
    """
    idxs, detections = batch[0]
    return ([idxs], detections)

class CAMELTrack:
    def __init__(
            self,
            engine_path,
            image_size = (1920, 1080),
            min_det_conf: float = 0.4,
            min_init_det_conf: float = 0.6,
            min_num_hits: int = 0,
            max_wo_hits: int = 150,
            max_track_gallery_size: int = 50,
            sim_threshold = 0.5,
            **kwargs,
    ):

        self.CAMEL = Camel(engine_path, sim_threshold = sim_threshold, image_size=image_size)

        self.min_det_conf = min_det_conf
        self.min_init_det_conf = min_init_det_conf
        self.min_num_hits = min_num_hits
        self.max_wo_hits = max_wo_hits
        self.max_track_gallery_size = max_track_gallery_size

        self.input_columns = ['embeddings', 'visibility_scores', 'keypoints_xyc', 'bbox_tlwh', 'bbox_conf']


        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.tracklets = []
        self.frame_count = 0

    def preprocess(self, data):
        if not data:
            return []
        
        tracklab_ids = data['id_list']
        image_id = data['image_id']

        features = {
            feature_name: np.expand_dims(data[feature_name], 0) # 加batch维度
            for feature_name in self.input_columns + ["im_width", "im_height"]
            if data[feature_name].size > 0
        }

        detections =  [
            Detection(image_id, {k: v[0, i] for k, v in features.items()}, tracklab_ids[i], frame_idx=self.frame_count) # image_id是真实的图像id， tracklab_ids是实例id, frame_idx是track处理的帧id
            for i in range(len(tracklab_ids)) if features["bbox_conf"][0, i] >= self.min_det_conf
        ] 

        return detections

    def process(self, data):
        detections = self.preprocess(data)

        # Update the states of the tracklets
        for track in self.tracklets:
            track.forward()

        # Associate detections to tracklets
        matched, unmatched_trks, unmatched_dets, td_sim_matrix = self.associate_dets_to_trks(self.tracklets, detections)

        #  Check that each track and detection index is present exactly once
        assert len(set([m[0] for m in matched.tolist()] + unmatched_trks)) == len(self.tracklets)
        assert len(set([m[1] for m in matched.tolist()] + unmatched_dets)) == len(detections)

        # Update matched tracklets with assigned detections
        for m in matched:
            tracklet = self.tracklets[m[0]]
            detection = detections[m[1]]
            tracklet.update(detection)
            detection.similarity_with_tracklet = td_sim_matrix[m[0], m[1]]
            detection.similarities = td_sim_matrix[:len(self.tracklets), m[1]]

        # Create and initialise new tracklets for unmatched detections
        for i in unmatched_dets:
            # Check that confidence is high enough
            if detections[i].bbox_conf >= self.min_init_det_conf:
                self.tracklets.append(Tracklet(detections[i], self.max_track_gallery_size))

        # Handle tracklets outputs and cleaning
        actives = []
        for trk in self.tracklets:
            # Get active tracklets and for first frames, also return tracklets in init state
            self.update_state(trk)
            if (trk.state == "active") or (trk.state == "init" and self.frame_count < self.min_num_hits):
                actives.append(
                    {
                        "index": trk.last_detection.index,
                        "track_id": trk.id,  # id is computed from a counter
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet) if trk.last_detection.similarity_with_tracklet is not None else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.tracklets[j].id: sim for j, sim in enumerate(trk.last_detection.similarities)} if trk.last_detection.similarities is not None else None,
                            "St": self.CAMEL.sim_threshold,
                    }
                })

        self.tracklets = [trk for trk in self.tracklets if trk.state != "dead"]
        self.frame_count += 1


        return actives

    def associate_dets_to_trks(self, tracklets, detections):
        if not tracklets:
            return np.empty((0, 2)), [], list(range(len(detections))), np.empty((0,))
        if not detections:
            return np.empty((0, 2)), list(range(len(tracklets))), [], np.empty((0,))

        # batch['det_feats'].keys: dict_keys(['embeddings', 'visibility_scores', 'keypoints_xyc', 'bbox_ltwh', 'bbox_conf', 'index', 'age', 'im_width', 'im_height'])
        # batch.keys: dict_keys(['image_id', 'det_feats', 'det_masks', 'track_feats', 'track_masks'])
        batch = self.build_camel_batch(tracklets, detections) 
        association_matrix, association_result, td_sim_matrix = self.CAMEL.predict(batch)
        matched = association_result[0]["matched_td_indices"]
        unmatched_trks = association_result[0]["unmatched_trackers"]
        unmatched_dets = association_result[0]["unmatched_detections"]
        return matched, unmatched_trks, unmatched_dets, td_sim_matrix.squeeze(0)


    def update_state(self, tracklet):
        s = tracklet.state
        if s == "init":
            new_state = "active" if tracklet.hit_streak >= self.min_num_hits else "dead" if tracklet.time_wo_hits >= 1 else "init"
        elif s == "active":
            new_state = "active" if tracklet.time_wo_hits == 0 else "lost" if tracklet.time_wo_hits < self.max_wo_hits else "dead"
        elif s == "lost":
            new_state = "active" if tracklet.time_wo_hits == 0 else "lost" if tracklet.time_wo_hits < self.max_wo_hits else "dead"
        elif s == "dead":
            new_state = "dead"
        else:
            raise ValueError(f"Tracklet {tracklet} is in undefined state {s}.")
        tracklet.state = new_state


    def build_camel_batch(self, tracklets, detections):
        T_max = max(len(t.detections) for t in tracklets)

        detection_features = self.build_detection_features(detections)
        tracklet_features = self.build_tracklet_features(tracklets, T_max)

        batch = {
            'image_id': detections[0].image_id,  # int
            'det_feats': detection_features,
            'det_masks': np.ones((1, len(detections), 1), dtype=bool),  # [1, N, 1] N 是det个数
            'track_feats': tracklet_features,
            'track_masks': np.expand_dims(
                                np.stack([
                                    np.concatenate([
                                        np.ones(len(t.detections), dtype=bool),
                                        np.zeros(T_max - len(t.detections), dtype=bool)
                                    ])
                                    for t in tracklets
                                ]),
                                axis=0
                            ) # [1, N, T] N是轨迹个数，T是轨迹中det个数
        }

        return batch

    def build_detection_features(self, detections):
        features = {}
        for feature in self.input_columns + ["index", "age", "im_width", "im_height"]:
            stacked_feature = np.stack([det[feature] for det in detections])
            features[feature] = np.expand_dims(np.expand_dims(stacked_feature, axis=1), axis=0)


        return features

    def build_tracklet_features(self, tracklets, T_max):
        features = {}
        for feature in self.input_columns + ["index", "age", "im_width", "im_height"]:
            stacked_feature = np.stack([t.padded_features(feature, T_max) for t in tracklets]) # 在n上堆叠stack
            features[feature] = np.expand_dims(stacked_feature, axis=0) # 添加batch

        return features


class Detection:
    def __init__(self, image_id, features, tracklab_id, frame_idx):
        # tracklab_id 是实例的id
        self.features = features
        for k, v in features.items():
            if len(v.shape) == 0:
                v = np.expand_dims(v, 0)
            setattr(self, k, v)
        self.index = tracklab_id
        self.image_id = image_id
        self.frame_idx = frame_idx
        self.similarity_with_tracklet = None
        self.similarities = None
        self.age = np.array([0], dtype=np.int32)

    def __getitem__(self, item):
        return getattr(self, item)

class Tracklet(object):
    # MOT benchmark requires positive:
    count = 1

    def __init__(self, detection, max_gallery_size):
        self.last_detection = detection
        self.detections = [detection]
        self.state = "init"
        self.id = Tracklet.count
        Tracklet.count += 1
        # Variables for tracklet management
        self.age = 0
        self.hits = 0
        self.hit_streak = 0   # 连续集中的次数
        self.time_wo_hits = 0 # 连续没有击中的次数

        self.max_gallery_size = max_gallery_size

    def forward(self):
        self.age += 1
        self.time_wo_hits += 1 
        # Update the age of all previous detections
        for detection in self.detections:
            detection.age += 1

        if self.time_wo_hits > 1:
            self.hit_streak = 0

    def update(self, detection):
        self.detections.append(detection)
        self.detections = self.detections[-self.max_gallery_size:]
        # Variables for tracklet management
        self.hits += 1
        self.hit_streak += 1
        self.time_wo_hits = 0
        self.last_detection = detection

    def padded_features(self, name, size):
        features = np.stack([getattr(det, name) for det in reversed(self.detections)])
        if features.shape[0] < size:
            if 'mask' not in name:
                features = np.concatenate(
                    [features, np.zeros((size - features.shape[0], *features.shape[1:]), features.dtype) + np.float32('nan')]
                )

        return features

    def __str__(self):
        return (f"Tracklet(id={self.id}, state={self.state}, age={self.age}, "
                f"hits={self.hits}, hit_streak={self.hit_streak}, "
                f"time_wo_hits={self.time_wo_hits}, "
                f"num_detections={len(self.detections)})")