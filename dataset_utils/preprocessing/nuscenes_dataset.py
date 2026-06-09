"""nuScenes CoT annotation dataset.

Mirrors NuplanCoTAnnotationDataset / WaymoE2ECoTAnnotationDataset so any
downstream entry point (e.g. cot_sample_generation.py,
symbolic_cot_sample_generation.py) can swap between datasets by changing the
`dataset_name` config field.

Helpers (`get_global_sensor_pose`, `get_ego_pose_future_his`,
`get_ego_velocity`, `get_planning_instruction`) live here and are re-exported
to `tools/preprocessing/nusc_sample_generation.py` to keep a single source of
truth.
"""

import os
import cv2
import base64
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

# nuscenes-devkit / pyquaternion are heavy and only installed in nuscenes-enabled
# environments. Defer the failure to first use so this module imports cleanly on
# any machine — nuplan/waymo workflows that never touch this code stay unaffected.
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits as _nusc_splits
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    _NUSC_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as _e:
    NuScenes = None        # type: ignore[assignment]
    _nusc_splits = None    # type: ignore[assignment]
    transform_matrix = None  # type: ignore[assignment]
    Quaternion = None      # type: ignore[assignment]
    _NUSC_IMPORT_ERROR = _e


def _require_nuscenes():
    if _NUSC_IMPORT_ERROR is not None:
        raise ImportError(
            "nuscenes-devkit (and pyquaternion) are required for nuScenes "
            "support but are not installed in this environment. "
            f"Original import error: {_NUSC_IMPORT_ERROR}"
        )


from dataset_utils.preprocessing.cot_prompts import get_cot_reasoning_prompt
from dataset_utils.preprocessing.nuplan_dataset import process_image_input

CAM_LIST = ['front', 'front_left', 'front_right',
            'back', 'back_left', 'back_right', 'left', 'right']

# (logical side, nuScenes camera channel) for the 4 cameras the symbolic CoT
# annotator consumes — forward triplet + back, matching the NuplanCoTAnnotationDataset
# 360° prompt set so cross-dataset CoT distributions stay aligned.
NUSC_CAMERA_TYPES = [
    ('front', 'CAM_FRONT'),
    ('front_left', 'CAM_FRONT_LEFT'),
    ('front_right', 'CAM_FRONT_RIGHT'),
    ('back', 'CAM_BACK'),
]


# ============================================================
# Pose / trajectory helpers (canonical source; nusc_sample_generation.py
# re-exports these for backward compatibility).
# ============================================================

def get_global_sensor_pose(rec, nusc, inverse=False):
    """Get the 4x4 transformation matrix for the sample's LIDAR_TOP pose."""
    _require_nuscenes()
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])

    if not inverse:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)

    return pose


def get_ego_pose_future_his(nusc, fut_ts, sample, his_ts):
    """Extract ego trajectory for history and future frames in LIDAR_TOP frame.

    Returns trajectories in the nuScenes-native frame where x is to the right
    and y is forward. Callers downstream rotate to a forward/left frame.
    """
    ego_his_trajs = np.zeros((his_ts + 1, 3))
    ego_his_trajs_diff = np.zeros((his_ts + 1, 3))
    ego_his_masks = np.zeros((his_ts + 1))
    sample_cur = sample

    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    # History: walk backwards
    for i in range(his_ts, -1, -1):
        if sample_cur is not None:
            pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
            ego_his_trajs[i] = pose_mat[:3, 3]
            ego_his_masks[i] = 1
            has_prev = sample_cur['prev'] != ''
            has_next = sample_cur['next'] != ''
            if has_next:
                sample_next = nusc.get('sample', sample_cur['next'])
                pose_mat_next = get_global_sensor_pose(sample_next, nusc, inverse=False)
                ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]
            sample_cur = nusc.get('sample', sample_cur['prev']) if has_prev else None
        else:
            ego_his_trajs[i] = ego_his_trajs[i + 1] - ego_his_trajs_diff[i + 1]
            ego_his_trajs_diff[i] = ego_his_trajs_diff[i + 1]

    # Global → ego (at current sample), then ego → lidar
    ego_his_trajs = ego_his_trajs - np.array(pose_record['translation'])
    rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
    ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
    ego_his_trajs = ego_his_trajs - np.array(cs_record['translation'])
    rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
    ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
    ego_his_diff = ego_his_trajs[1:] - ego_his_trajs[:-1]

    # Future: walk forwards
    ego_fut_trajs = np.zeros((fut_ts + 1, 3))
    ego_fut_masks = np.zeros((fut_ts + 1))
    sample_cur = sample
    for i in range(fut_ts + 1):
        pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
        ego_fut_trajs[i] = pose_mat[:3, 3]
        ego_fut_masks[i] = 1
        if sample_cur['next'] == '':
            ego_fut_trajs[i + 1:] = ego_fut_trajs[i]
            break
        else:
            sample_cur = nusc.get('sample', sample_cur['next'])

    ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
    rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
    ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
    rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T

    if ego_fut_trajs[-1][0] >= 2:
        command = 'Turn Right'
    elif ego_fut_trajs[-1][0] <= -2:
        command = 'Turn Left'
    else:
        command = 'Go Straight'

    ego_fut_diff = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

    return (
        ego_fut_diff[:, :2].astype(np.float32),
        ego_fut_trajs[:, :2].astype(np.float32),
        ego_his_trajs[:, :2].astype(np.float32),
        ego_his_diff[:, :2].astype(np.float32),
        ego_fut_masks[1:].astype(np.float32),
        ego_his_masks[:-1].astype(np.float32),
        command,
    )


def get_ego_velocity(sample, nusc):
    """Scalar ego speed (m/s) between this sample and its previous keyframe."""
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    if sample['prev'] != '':
        sample_prev = nusc.get('sample', sample['prev'])
        sd_rec_prev = nusc.get('sample_data', sample_prev['data']['LIDAR_TOP'])
        pose_record_prev = nusc.get('ego_pose', sd_rec_prev['ego_pose_token'])
    else:
        pose_record_prev = None

    assert pose_record_prev is not None, 'prev token is empty'

    ego_pos = np.array(pose_record['translation'])
    ego_pos_prev = np.array(pose_record_prev['translation'])
    return np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5


def get_planning_instruction(ego_fut_diff, ego_fut_trajs, ego_his_diff, ego_his_trajs):
    """Free-form action string ('move forward with a constant speed', 'STOP', ...).

    Used as `fut_ego_action` analogue of nuPlan's
    `get_action_instruction()` so the symbolic-CoT action hint stays
    consistent across datasets.
    """
    constant_eps = 0.5
    his_velos = np.linalg.norm(ego_his_diff, axis=1)
    fut_velos = np.linalg.norm(ego_fut_diff, axis=1)
    cur_velo = his_velos[-1]
    end_velo = fut_velos[-1]

    if cur_velo < constant_eps and end_velo < constant_eps:
        speed_meta = "stop"
    elif end_velo < constant_eps:
        speed_meta = "a deceleration to zero"
    elif np.abs(end_velo - cur_velo) < constant_eps:
        speed_meta = "a constant speed"
    elif cur_velo > end_velo:
        speed_meta = "a quick deceleration" if cur_velo > 2 * end_velo else "a deceleration"
    else:
        speed_meta = "a quick acceleration" if end_velo > 2 * cur_velo else "an acceleration"

    if speed_meta == "stop":
        return "STOP"

    forward_th = 2.0
    lane_changing_th = 4.0
    if (np.abs(ego_fut_trajs[:, 0]) < forward_th).all():
        behavior_meta = "move forward"
    elif ego_fut_trajs[-1, 0] < 0:
        behavior_meta = "turn left" if np.abs(ego_fut_trajs[-1, 0]) > lane_changing_th else "change lane to left"
    elif ego_fut_trajs[-1, 0] > 0:
        behavior_meta = "turn right" if np.abs(ego_fut_trajs[-1, 0]) > lane_changing_th else "change lane to right"
    else:
        behavior_meta = "move forward"

    return f"{behavior_meta} with {speed_meta}"


def _rotate_to_forward_left(xy_right_forward):
    """Convert (right, forward) → (forward, left) so trajectories match the nuPlan convention."""
    out = np.zeros_like(xy_right_forward)
    out[..., 0] = xy_right_forward[..., 1]
    out[..., 1] = -xy_right_forward[..., 0]
    return out


# ============================================================
# Dataset
# ============================================================

class NuscenesCoTAnnotationDataset(Dataset):
    """Iterates nuScenes samples and produces a per-sample dict whose schema
    matches `NuplanCoTAnnotationDataset.__getitem__`.

    Required config keys:
        dataset_path: nuScenes data root (passed to `NuScenes(dataroot=...)`).
        version:      e.g. 'v1.0-trainval'.
        split:        'train' | 'val'.

    Optional:
        num_history_frames: history frames including current (default 4).
        num_future_frames:  future frames extracted (default 10).
        path_prefix_maps:   list of (src, dst) tuples for cross-host path remap.
    """

    def __init__(self, config, processor=None):
        _require_nuscenes()
        nuscenes_path = config['dataset_path']
        version = config.get('version', 'v1.0-trainval')
        self.split = config.get('split', 'train')
        self.processor = processor
        # Path remap (e.g. /backup/... → /data/...). Optional, in-config.
        self.path_prefix_maps = config.get('path_prefix_maps') or []

        # Teacher camera-view count: 4 (front+front_left+front_right+back,
        # default — matches the original recipe) vs 3 (no back, matches the
        # 3-camera student inference path). Affects both NL CoT and symbolic
        # CoT generation since both share this dataset class.
        self.include_back_view = bool(config.get('include_back_view', True))

        # nuScenes counts history as N frames + current; nusc_sample_generation.py
        # uses his_ts=3 to get 4 frames total. Mirror that convention.
        total_history = int(config.get('num_history_frames', 4))
        self.his_ts = max(total_history - 1, 0)
        # Future: extracted long enough that the first 10 (gt_trajectory) are
        # well-defined; nusc_sample_generation.py uses 16.
        self.fut_ts = int(config.get('num_future_frames', 16))

        self.nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=False)

        # Resolve scene whitelist for this split.
        if self.split == 'train':
            split_names = _nusc_splits.train
        elif self.split == 'val':
            split_names = _nusc_splits.val
        else:
            raise ValueError(f"Invalid nuScenes split: {self.split!r} (expected 'train' or 'val')")

        available_scenes = self.nusc.scene
        available_scene_names = [s['name'] for s in available_scenes]
        split_names = [s for s in split_names if s in available_scene_names]
        scene_set = {
            available_scenes[available_scene_names.index(s)]['token']
            for s in split_names
        }

        # Pre-filter: keep samples in-split that have a prev keyframe (needed for
        # the velocity / history-trajectory computation downstream).
        self.sample_tokens: List[str] = []
        for sample in self.nusc.sample:
            if sample['scene_token'] not in scene_set:
                continue
            if sample['prev'] == '':
                continue
            self.sample_tokens.append(sample['token'])

        print(
            f"NuscenesCoTAnnotationDataset: {len(self.sample_tokens)} samples "
            f"in split={self.split} (from {len(self.nusc.sample)} total)"
        )

    def __len__(self):
        return len(self.sample_tokens)

    def _maybe_remap_path(self, path: str) -> str:
        for src, dst in self.path_prefix_maps:
            if path == src or path.startswith(src.rstrip('/') + "/"):
                return dst.rstrip('/') + path[len(src.rstrip('/')):]
        return path

    def __getitem__(self, idx):
        token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', token)

        # Pose & trajectory in LIDAR_TOP frame (x-right, y-forward).
        (
            gt_ego_fut_diff,
            gt_ego_fut_trajs,
            gt_ego_his_trajs,
            gt_ego_his_diff,
            _gt_ego_fut_masks,
            _gt_ego_his_masks,
            command,
        ) = get_ego_pose_future_his(self.nusc, self.fut_ts, sample, self.his_ts)

        # Velocity / acceleration in (forward, left) frame, m/s and m/s^2.
        # diff is per-frame displacement at 0.5s (nuScenes is 2 Hz).
        if len(gt_ego_his_diff) >= 1:
            last_diff = gt_ego_his_diff[-1]
            velocity = np.array(
                [float(last_diff[1] / 0.5), float(-last_diff[0] / 0.5)],
                dtype=np.float32,
            )
        else:
            velocity = np.zeros(2, dtype=np.float32)
        if len(gt_ego_his_diff) >= 2:
            prev_diff = gt_ego_his_diff[-2]
            v_now = last_diff / 0.5
            v_prev = prev_diff / 0.5
            accel_rf = (v_now - v_prev) / 0.5
            acceleration = np.array([float(accel_rf[1]), float(-accel_rf[0])], dtype=np.float32)
        else:
            acceleration = np.zeros(2, dtype=np.float32)

        # fut_ego_action is the GT-action label used as Hint in the symbolic prompt.
        fut_ego_action = get_planning_instruction(
            gt_ego_fut_diff, gt_ego_fut_trajs, gt_ego_his_diff, gt_ego_his_trajs
        )
        instruction = command.lower()

        # gt_trajectory: first 10 future frames in (forward, left, heading) frame.
        fut = gt_ego_fut_trajs[:11]
        gt_fl = np.zeros((fut.shape[0], 3), dtype=np.float32)
        gt_fl[:, :2] = _rotate_to_forward_left(fut)
        if gt_fl.shape[0] >= 2:
            heading = np.arctan2(
                gt_fl[1:, 1] - gt_fl[:-1, 1],
                gt_fl[1:, 0] - gt_fl[:-1, 0] + 1e-3,
            )
            gt_fl[1:, 2] = heading
        gt_trajectory = torch.tensor(gt_fl[1:], dtype=torch.float32)

        # history trajectory in (forward, left, heading) frame.
        his_fl = np.zeros((gt_ego_his_trajs.shape[0], 3), dtype=np.float32)
        his_fl[:, :2] = _rotate_to_forward_left(gt_ego_his_trajs)
        if his_fl.shape[0] >= 2:
            his_heading = np.arctan2(
                his_fl[1:, 1] - his_fl[:-1, 1],
                his_fl[1:, 0] - his_fl[:-1, 0] + 1e-3,
            )
            his_fl[1:, 2] = his_heading
        his_trajectory = torch.tensor(his_fl, dtype=torch.float32)

        # Camera paths: walk history backwards for 4 frames (his_ts + 1) per camera.
        camera_paths: Dict[str, List[str]] = {f"{side}_camera_paths": [] for side, _ in NUSC_CAMERA_TYPES}
        sample_cur = sample
        for i in range(self.his_ts, -1, -1):
            for side, cam_type in NUSC_CAMERA_TYPES:
                cam_token = sample_cur['data'][cam_type]
                cam_path, _, _ = self.nusc.get_sample_data(cam_token)
                cam_path = self._maybe_remap_path(cam_path)
                camera_paths[f"{side}_camera_paths"].insert(0, cam_path)
            if i != 0:
                if sample_cur['prev'] == '':
                    break
                sample_cur = self.nusc.get('sample', sample_cur['prev'])

        # Assemble the per-sample dict (matches NuplanCoTAnnotationDataset schema).
        inputs: Dict[str, Any] = {
            'token': token,
            'dataset_name': 'nuscenes',
            'velocity': [float(velocity[0]), float(velocity[1])],
            'acceleration': [float(acceleration[0]), float(acceleration[1])],
            'instruction': instruction,
            'fut_ego_action': fut_ego_action,
            'gt_trajectory': gt_trajectory,
            'his_trajectory': his_trajectory,
        }
        # Populate all 8 CAM_LIST slots so downstream `build_result` stays uniform.
        for side in CAM_LIST:
            key = f"{side}_camera_paths"
            inputs[key] = camera_paths.get(key, [])

        if self.processor is not None:
            inputs.update(self._build_vlm_inputs(
                camera_paths, velocity, acceleration, instruction, fut_ego_action,
            ))

        return inputs

    def _build_vlm_inputs(self, camera_paths, velocity, acceleration, instruction, fut_ego_action):
        """Decode 4 cameras × 4 frames and build the Qwen-style multimodal `messages`.

        Camera set (front + front_left + front_right + back) mirrors the
        NuplanCoTAnnotationDataset 360° prompt so cross-dataset CoT
        distributions stay aligned. Back view is included because the CoT
        annotator benefits from full-scene context when writing reasoning;
        SFTDataset will later subsample to a 3-camera training input on its
        own.
        """
        from qwen_vl_utils import process_vision_info

        def load_b64_video(paths: List[str]) -> List[str]:
            video = []
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    raise FileNotFoundError(f"Cannot read camera image: {p}")
                video.append("data:image/jpeg;base64," + process_image_input(img))
            return video

        front_video = load_b64_video(camera_paths["front_camera_paths"])
        front_left_video = load_b64_video(camera_paths["front_left_camera_paths"])
        front_right_video = load_b64_video(camera_paths["front_right_camera_paths"])
        if self.include_back_view:
            back_video = load_b64_video(camera_paths["back_camera_paths"])

        # Summary sentence — kept byte-identical to the original recipe when
        # include_back_view=True so existing CoT runs are reproducible.
        if self.include_back_view:
            summary_text = (
                "Four cameras are mounted on the vehicle to perceive the surrounding "
                "environment. These cameras provide the front, front-left, front-right, "
                "and back views. The multi-view multi-frame camera images are "
                "organized in a video format."
            )
        else:
            summary_text = (
                "Three cameras are mounted on the vehicle to perceive the surrounding "
                "environment. These cameras provide the front, front-left, and front-right "
                "views. The multi-view multi-frame camera images are organized in a "
                "video format."
            )

        user_content = [
            {"type": "text", "text": summary_text},
            {
                "type": "text",
                "text": (
                    "The video is from the front camera, capturing the history of the "
                    "vehicle's front view from the past two seconds at 2Hz."
                ),
            },
            {
                "type": "video",
                "min_pixels": 400 * 400,
                "max_pixels": 400 * 400,
                "video": front_video,
            },
            {
                "type": "text",
                "text": (
                    "The video is from the front-left camera, capturing the history of "
                    "the vehicle's front-left view from the past two seconds at 2Hz."
                ),
            },
            {
                "type": "video",
                "min_pixels": 400 * 400,
                "max_pixels": 400 * 400,
                "video": front_left_video,
            },
            {
                "type": "text",
                "text": (
                    "The video is from the front-right camera, capturing the history of "
                    "the vehicle's front-right view from the past two seconds at 2Hz."
                ),
            },
            {
                "type": "video",
                "min_pixels": 400 * 400,
                "max_pixels": 400 * 400,
                "video": front_right_video,
            },
        ]

        if self.include_back_view:
            user_content.extend([
                {
                    "type": "text",
                    "text": (
                        "The video is from the back camera, capturing the history of the "
                        "vehicle's back view from the past two seconds at 2Hz."
                    ),
                },
                {
                    "type": "video",
                    "min_pixels": 400 * 400,
                    "max_pixels": 400 * 400,
                    "video": back_video,
                },
            ])

        user_content.extend([
            {
                "type": "text",
                "text": (
                    f"The ego vehicle's current velocity is {velocity[0]:.3f} m/s at "
                    f"x-direction and {velocity[1]:.3f} m/s at y-direction. "
                    f"The ego vehicle's current acceleration is {acceleration[0]:.3f} m/s^2 "
                    f"at x-direction and {acceleration[1]:.3f} m/s^2 at y-direction. "
                    f"The current driving command instruction of ego vehicle is: "
                    f"{instruction}, indicating the intended route direction."
                ),
            },
            get_cot_reasoning_prompt(fut_ego_action),
        ])

        messages = [
            {
                "role": "system",
                "content": "As a professional driver, how do you drive in the following scenario.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        process_vision_kwargs = {"return_video_kwargs": True}
        patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", None)
        if patch_size is not None:
            process_vision_kwargs["image_patch_size"] = patch_size

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_metadata=True, **process_vision_kwargs,
            )
        except TypeError:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, **process_vision_kwargs,
            )
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True,
        )
        return {
            'messages': messages,
            'text': text,
            'image_inputs': image_inputs,
            'video_inputs': video_inputs,
            'mm_processor_kwargs': video_kwargs or {},
        }


@dataclass
class DataCollator:
    """List-merge collator mirroring nuplan_dataset.DataCollator."""

    processor: Any = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        if self.processor is not None:
            batch["text"] = [f["text"] for f in features]
            batch["image_inputs"] = [f["image_inputs"] for f in features]
            batch["video_inputs"] = [f["video_inputs"] for f in features]
            batch["mm_processor_kwargs"] = [f.get("mm_processor_kwargs", {}) for f in features]

        batch["token"] = [f["token"] for f in features]
        batch["dataset_name"] = [f.get("dataset_name", "nuscenes") for f in features]
        batch["velocity"] = [f["velocity"] for f in features]
        batch["acceleration"] = [f["acceleration"] for f in features]
        batch["instruction"] = [f["instruction"] for f in features]
        batch["fut_ego_action"] = [f.get("fut_ego_action", "") for f in features]
        batch["gt_trajectory"] = [f["gt_trajectory"] for f in features]
        batch["his_trajectory"] = [f["his_trajectory"] for f in features]

        for side in CAM_LIST:
            path_key = f"{side}_camera_paths"
            batch[path_key] = [f.get(path_key, []) for f in features]

        return batch
