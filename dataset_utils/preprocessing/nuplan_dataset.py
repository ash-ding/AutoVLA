import os
import cv2
import glob
import torch
import base64
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from dataclasses import dataclass, replace
from typing import Any, Dict, List
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
from navsim.common.dataloader import SceneLoader
from navsim.agents.vla_agent import VlaAgent
from navsim.common.dataclasses import SceneFilter
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from dataset_utils.preprocessing.cot_prompts import get_cot_reasoning_prompt

CAM_LIST = ['front', 'front_left', 'front_right', 
            'back', 'back_left', 'back_right', 'left', 'right']

class NuplanCoTAnnotationDataset(Dataset):
    def __init__(self, config, processor=None, *, build_messages=None):
        """
        Args:
            config: dataset config (paths, scene_filter, annotation_backend, ...).
            processor: HF AutoProcessor for vLLM tokenization. None for OpenAI / no-CoT.
            build_messages: explicit override for "do we decode images + build the
                Qwen messages list?". If None (default), auto-detect:
                - vLLM (processor != None) → True
                - OpenAI backend (config['annotation_backend'] == 'openai') → True
                - otherwise (e.g. no-CoT) → False
                Pass build_messages=False from no-CoT scripts to skip the per-frame
                JPEG decode entirely, even if a processor was loaded for some other
                reason.
        """
        self.data_path = config['dataset_path']
        self.data_folders = glob.glob(self.data_path + '/*/*')
        self.processor = processor

        if build_messages is None:
            backend = str(config.get('annotation_backend', '')).lower()
            self.needs_vision = (processor is not None) or (backend == 'openai')
        else:
            self.needs_vision = bool(build_messages)

        # Teacher camera-view count: 4 (front+back+front-left+front-right,
        # default) vs 3 (no back). Set `include_back_view: false` in the
        # dataset config to drop the back-view block.
        self.include_back_view = bool(config.get('include_back_view', True))

        # Front-left / front-right side camera angle. 90 (default) loads from
        # cam_l1/cam_r1 (90° pure side via images['left_camera']/['right_camera']);
        # 45 loads from cam_l0/cam_r0 (45° forward-canted via
        # images['front_left_camera']/['front_right_camera']). The prompt text
        # is identical either way — the physical source is the only variable.
        self.side_cam_yaw = int(config.get('side_cam_yaw', 90))
        assert self.side_cam_yaw in (45, 90), (
            f"side_cam_yaw must be 45 or 90, got {self.side_cam_yaw}"
        )

        if config['scene_filter'] is None:
            scene_filter = SceneFilter(
                num_history_frames=4, # number of past frames to be extracted, frames are at 2Hz
                num_future_frames=10, # number of future frames to be extracted, frames are at 2Hz
                frame_interval=4, # number of frames to skip between each scene, if null, extracted scenes are non-overlapping
            )
        else:
            scene_filter: SceneFilter = instantiate(OmegaConf.load(config['scene_filter']))

        self.interval_length = 0.5
        trajectory_sampling = TrajectorySampling(time_horizon=5, interval_length=self.interval_length)
        self._agent = VlaAgent(trajectory_sampling=trajectory_sampling)

        # Build a NARROW SensorConfig: enable only the cameras actually used
        # by the teacher prompt, AND only at history-frame iterations. The
        # `Union[bool, List[int]]` form lets a sensor opt out of future frames.
        # Rationale:
        #   1. nuPlan trainval workspace only carries F0+L1+R1+B0 (no L0/R0/L2/R2
        #      in Stage 1). All-True would make SceneLoader load missing jpgs.
        #   2. Even for the cameras we DO have, navsim hydrates frames at
        #      history+future indices (default 4+10=14 per camera). vessl
        #      hydrate only extracted the 4 history-frame paths per token, so
        #      requesting future-frame jpgs → FileNotFoundError.
        # Gating sensors at history-frame iterations only sidesteps both.
        history_idx = list(range(scene_filter.num_history_frames))   # e.g. [0,1,2,3]
        cam_side_90 = (self.side_cam_yaw == 90)
        cam_side_45 = (self.side_cam_yaw == 45)
        sensor_config = replace(
            self._agent.get_sensor_config(),
            cam_f0=history_idx,                                          # always (front)
            cam_b0=history_idx if self.include_back_view else False,     # back (gated)
            cam_l1=history_idx if cam_side_90 else False,                # side-left @ 90°
            cam_r1=history_idx if cam_side_90 else False,                # side-right @ 90°
            cam_l0=history_idx if cam_side_45 else False,                # front-left @ 45°
            cam_r0=history_idx if cam_side_45 else False,                # front-right @ 45°
            cam_l2=False,                                                # back-left (never used)
            cam_r2=False,                                                # back-right (never used)
            decode_images=self.needs_vision,
        )

        self._scene_loader = SceneLoader(
            data_path=Path(self.data_path.replace('placeholder', 'navsim_logs')),
            sensor_blobs_path=Path(self.data_path.replace('placeholder', 'sensor_blobs')),
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )

        print(f"Extracted {len(self._scene_loader.tokens)} scenarios")

    def __len__(self):
        return len(self._scene_loader.tokens)

    def __getitem__(self, idx):
        input_features: Dict[str, torch.Tensor] = {}

        scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
        agent_input = scene.get_agent_input()
        for builder in self._agent.get_feature_builders():
            input_features.update(builder.compute_features(agent_input))
        target_builder = self._agent.get_target_builders()[0]
        target_trajectory = target_builder.compute_targets(scene)

        # image sensor (pixel data only loaded when processor is provided)
        images = input_features['images']

        # vehicle state
        velocity = input_features["vehicle_velocity"]
        acceleration = input_features["vehicle_acceleration"]
        instruction = input_features["driving_command"].lower()
        ego_driving_state = "stationary" if velocity[0] < 0.1 else "moving"

        # his trajectory
        his_raw_trajectory = input_features["history_trajectory"]

        # gt trajectory
        gt_trajectory = target_trajectory

        # Extract positions and heading for action calculation
        gt_positions_np = gt_trajectory[:, :2].numpy()
        his_positions = his_raw_trajectory[:, :2].numpy()

        his_velocity = np.diff(his_positions, axis=0) / self.interval_length
        if len(his_velocity) > 0:
            his_velocity = np.concatenate([his_velocity, his_velocity[-1:]], axis=0)
        else:
            his_velocity = np.zeros((1, 2))
        his_ego_action = get_action_instruction(his_positions, his_velocity)

        # Calculate future ego action
        fut_velocity = np.diff(gt_positions_np, axis=0) / self.interval_length
        # Pad with last velocity to match trajectory length
        if len(fut_velocity) > 0:
            fut_velocity = np.concatenate([fut_velocity, fut_velocity[-1:]], axis=0)
        else:
            # Handle edge case with single frame
            fut_velocity = np.zeros((1, 2))
        fut_ego_action = get_action_instruction(gt_positions_np, fut_velocity)

        token = self._scene_loader.tokens[idx]
        inputs = {'token': token}

        # VLM-specific work (pixel decoding, base64 encoding, message templating).
        # `needs_vision` is True for both vLLM and OpenAI backends; False for the
        # no-CoT path that only needs camera paths/metadata.
        if self.needs_vision:
            front_camera_1 = images['front_camera'][0].image
            front_camera_2 = images['front_camera'][1].image
            front_camera_3 = images['front_camera'][2].image
            front_camera_4 = images['front_camera'][3].image

            if self.include_back_view:
                back_camera_1 = images['back_camera'][0].image
                back_camera_2 = images['back_camera'][1].image
                back_camera_3 = images['back_camera'][2].image
                back_camera_4 = images['back_camera'][3].image

            # Front-left / front-right physical source dispatched by side_cam_yaw.
            # Prompt labels stay "front-left" / "front-right" regardless of source.
            if self.side_cam_yaw == 90:
                fl_key, fr_key = 'left_camera', 'right_camera'         # CAM_L1, CAM_R1
            else:  # 45
                fl_key, fr_key = 'front_left_camera', 'front_right_camera'  # CAM_L0, CAM_R0
            front_left_camera_1 = images[fl_key][0].image
            front_left_camera_2 = images[fl_key][1].image
            front_left_camera_3 = images[fl_key][2].image
            front_left_camera_4 = images[fl_key][3].image

            front_right_camera_1 = images[fr_key][0].image
            front_right_camera_2 = images[fr_key][1].image
            front_right_camera_3 = images[fr_key][2].image
            front_right_camera_4 = images[fr_key][3].image

            # Summary sentence: unified naming across all teacher prompts.
            # Block order: front → front-left → front-right → back (gated),
            # matching nuscenes_dataset.py and the student SFT prompt.
            if self.include_back_view:
                summary_text = (
                    "Four cameras are mounted on the vehicle to perceive the surrounding environment. "
                    "These cameras provide the front, front-left, front-right, and back views. "
                    "The multi-view multi-frame camera images are organized in a video format. "
                )
            else:
                summary_text = (
                    "Three cameras are mounted on the vehicle to perceive the surrounding environment. "
                    "These cameras provide the front, front-left, and front-right views. "
                    "The multi-view multi-frame camera images are organized in a video format. "
                )

            user_content = [
                # sensor inputs information
                {"type": "text", "text": summary_text},

                # front camera (always present)
                {
                    "type": "text",
                    "text": "The video is from the front camera, capturing the history of the vehicle's front view from the past two seconds at 2Hz."
                },
                {
                    "type": "video",
                    "min_pixels": 400 * 400,
                    "max_pixels": 400 * 400,
                    "video": [
                        "data:image/jpeg;base64," + process_image_input(front_camera_1),
                        "data:image/jpeg;base64," + process_image_input(front_camera_2),
                        "data:image/jpeg;base64," + process_image_input(front_camera_3),
                        "data:image/jpeg;base64," + process_image_input(front_camera_4),
                    ],
                },
            ]

            user_content.extend([
                # front-left camera (always present)
                {
                    "type": "text",
                    "text": "The video is from the front-left camera, capturing history of the vehicle's front-left view from the past two seconds at 2Hz."
                },
                {
                    "type": "video",
                    "min_pixels": 400 * 400,
                    "max_pixels": 400 * 400,
                    "video": [
                        "data:image/jpeg;base64," + process_image_input(front_left_camera_1),
                        "data:image/jpeg;base64," + process_image_input(front_left_camera_2),
                        "data:image/jpeg;base64," + process_image_input(front_left_camera_3),
                        "data:image/jpeg;base64," + process_image_input(front_left_camera_4),
                    ],
                },

                # front-right camera (always present)
                {
                    "type": "text",
                    "text": "The video is from the front-right camera, capturing history of the vehicle's front-right view from the past two seconds at 2Hz."
                },
                {
                    "type": "video",
                    "min_pixels": 400 * 400,
                    "max_pixels": 400 * 400,
                    "video": [
                        "data:image/jpeg;base64," + process_image_input(front_right_camera_1),
                        "data:image/jpeg;base64," + process_image_input(front_right_camera_2),
                        "data:image/jpeg;base64," + process_image_input(front_right_camera_3),
                        "data:image/jpeg;base64," + process_image_input(front_right_camera_4),
                    ],
                },
            ])

            # back camera (gated; appended after front-left/front-right to
            # match nuscenes_dataset.py ordering)
            if self.include_back_view:
                user_content.extend([
                    {
                        "type": "text",
                        "text": "The video is from the back camera, capturing the history of the vehicle's back view from the past two seconds at 2Hz."
                    },
                    {
                        "type": "video",
                        "min_pixels": 400 * 400,
                        "max_pixels": 400 * 400,
                        "video": [
                            "data:image/jpeg;base64," + process_image_input(back_camera_1),
                            "data:image/jpeg;base64," + process_image_input(back_camera_2),
                            "data:image/jpeg;base64," + process_image_input(back_camera_3),
                            "data:image/jpeg;base64," + process_image_input(back_camera_4),
                        ],
                    },
                ])

            user_content.extend([
                # vehicle state and instruction and additional information
                {
                    "type": "text",
                    "text": f"The ego vehicle behavior in the past 4s is **{his_ego_action}**."
                            f"The ego vehicle's current velocity is {velocity[0]:.3f} m/s at x-direction and {velocity[1]:.3f} m/s at y-direction." + \
                            f"The ego vehicle's current acceleration is {acceleration[0]:.3f} m/s^2 at x-direction and {acceleration[1]:.3f} m/s^2 at y-direction. " + \
                            f"The current driving command instruction of ego vehicle is: {instruction}, indicating the intended route direction. Note that the left and right driving commands cover turns, lane changes and sharp curves driving behavior."
                },

                # CoT Reasoning
                get_cot_reasoning_prompt(fut_ego_action),
            ])

            messages = [
                {
                    "role": "system",
                    "content": "As a professional driver, how do you drive in the following scenario."
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
            inputs['messages'] = messages

            # Processor-driven tokenization is vLLM-only; OpenAI backend
            # consumes the raw Qwen-format messages directly.
            if self.processor is not None:
                process_vision_kwargs = {
                    "return_video_kwargs": True,
                }
                patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", None)
                if patch_size is not None:
                    process_vision_kwargs["image_patch_size"] = patch_size

                try:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        messages,
                        return_video_metadata=True,
                        **process_vision_kwargs,
                    )
                except TypeError:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        messages,
                        **process_vision_kwargs,
                    )
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
                )
                inputs['text'] = text
                inputs['image_inputs'] = image_inputs
                inputs['video_inputs'] = video_inputs
                inputs['mm_processor_kwargs'] = video_kwargs or {}

        for side in CAM_LIST:
            camera_key = f"{side}_camera"
            path_key = f"{side}_camera_paths"
            inputs[path_key] = [getattr(cam, 'camera_path', None) for cam in images[camera_key]]
        inputs.update({
            "velocity": [float(velocity[0]), float(velocity[1])],
            "acceleration": [float(acceleration[0]), float(acceleration[1])],
            "instruction": instruction,
            "gt_trajectory": gt_trajectory,
            "his_trajectory": his_raw_trajectory,
            "fut_ego_action": fut_ego_action,
        })

        return inputs
    

def process_image_input(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode()

    return base64_image

def get_action_instruction(ego_trj_trajs, ego_trj_diff):
        # speed meta
        constant_eps = 0.8
        stop_eps = 0.3
        velos = np.linalg.norm(ego_trj_diff, axis=1)
        cur_velo = velos[0]
        end_velo = velos[-1]

        if cur_velo < stop_eps and end_velo < stop_eps:
            speed_meta = "stop"
        elif end_velo < stop_eps:
            speed_meta = "a deceleration to zero"
        elif np.abs(end_velo - cur_velo) < constant_eps:
            speed_meta = "a constant speed"
        elif end_velo > cur_velo:
            speed_meta = "a quick acceleration" if end_velo > 2 * cur_velo else "an acceleration"
        else:
            speed_meta = "a quick deceleration" if cur_velo > 2 * end_velo else "a deceleration"

        # behavior meta
        if speed_meta == "stop":
            return "STOP"

        forward_th = 2.0
        lane_changing_th = 4.0
        final_lat = ego_trj_trajs[-1, 1]

        if np.all(np.abs(ego_trj_trajs[:, 1]) < forward_th):
            behavior_meta = "move forward"
        elif final_lat > 0:
            behavior_meta = "turn left" if abs(final_lat) > lane_changing_th else "change lane to left"
        elif final_lat < 0:
            behavior_meta = "turn right" if abs(final_lat) > lane_changing_th else "change lane to right"
        else:
            behavior_meta = "move forward"

        return f"{behavior_meta} with {speed_meta}"

@dataclass
class DataCollator:
    processor: Any = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        if self.processor is not None:
            batch["text"] = [feature["text"] for feature in features]
            batch["image_inputs"] = [feature["image_inputs"] for feature in features]
            batch["video_inputs"] = [feature["video_inputs"] for feature in features]
            batch["mm_processor_kwargs"] = [feature.get("mm_processor_kwargs", {}) for feature in features]

        batch["token"] = [feature["token"] for feature in features]
        batch["velocity"] = [feature["velocity"] for feature in features]
        batch["acceleration"] = [feature["acceleration"] for feature in features]
        batch["instruction"] = [feature["instruction"] for feature in features]

        batch["gt_trajectory"] = [feature["gt_trajectory"] for feature in features]
        batch["his_trajectory"] = [feature["his_trajectory"] for feature in features]

        for side in CAM_LIST:
            path_key = f"{side}_camera_paths"
            batch[path_key] = [feature[path_key] for feature in features]

        return batch
