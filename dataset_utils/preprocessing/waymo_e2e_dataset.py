import os
import cv2
import glob
import torch
import base64
import itertools
import numpy as np
import pickle
import lmdb 
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path
import tensorflow as tf

from datetime import datetime
import matplotlib.pyplot as plt

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
from dataset_utils.preprocessing.cot_prompts import get_cot_reasoning_prompt

CAM_LIST = ['front', 'front_left', 'front_right', 
            'back', 'back_left', 'back_right', 'left', 'right']
REQUIRED_CAM_LIST = CAM_LIST


class WaymoE2ECoTAnnotationDataset(Dataset):
    def __init__(self, config, processor=None):
        self.config = config
        self.processor = processor
        self.split = self.config.get('dataset_split')
        
        self._filenames = tf.io.matching_files(
            os.path.join(self.config['dataset_path'], self.split + '_*.tfrecord-*'))
        self._tfdata = tf.data.TFRecordDataset(self._filenames, compression_type='')
        self._lmdb_env = self.load_lmdb(
            os.path.join(self.config['dataset_path'], self.split + '_lmdb'))

        print("Loading Waymo Scenes.")
        self.scenes = self.scene_loader()
        print(f"Loaded {len(self.scenes)} Waymo Scenes.")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_token, images_path = self.scenes[idx]

        with self._lmdb_env.begin() as txn:
            raw_record = txn.get(scene_token.encode('utf-8'))
            if raw_record is None:
                raise ValueError(f"Token {scene_token} not found in LMDB database.")
        
        frame = wod_e2ed_pb2.E2EDFrame()
        frame.ParseFromString(raw_record)
        
        images = {}
        for side in REQUIRED_CAM_LIST:
            key = f"{side}_camera_paths"
            images[f"{side}_camera"] = []
            for path in images_path[key]:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"Could not read image from path: {path}")
                images[f"{side}_camera"].append(img)

        # vehicle current state
        past_states = frame.past_states
        velocity = [past_states.vel_x[-1], past_states.vel_y[-1]]
        acceleration = [past_states.accel_x[-1], past_states.accel_y[-1]]

        intent_map = {0: "unknown", 1: "go straight", 2: "go left", 3: "go right"}
        instruction = intent_map.get(frame.intent, "unknown")

        # trajectory
        traj_freq_ratio = int(self.config['raw_trajectory_freq'] / self.config['model_freq'])
        his_raw_trajectory = list(zip(past_states.pos_x[1::traj_freq_ratio], 
                                    past_states.pos_y[1::traj_freq_ratio]))
        gt_raw_trajectory = []
        future_states = frame.future_states
        
        if future_states:
            if self.split == 'test':
                # test split does not have future states
                with self._lmdb_env.begin() as txn:
                    base, num_str = scene_token.rsplit('-', 1)
                    num = int(num_str)
                    new_35 = num + 35  
                    new_50 = num + 50
                    token_35 = f"{base}-{new_35:03d}"
                    token_50 = f"{base}-{new_50:03d}"

                    raw_record_35 = txn.get(token_35.encode('utf-8'))
                    if raw_record_35 is None:
                        raise ValueError(f"Token {token_35} not found in LMDB database.")
                    raw_record_50 = txn.get(token_50.encode('utf-8'))
                    if raw_record_50 is None:
                        raise ValueError(f"Token {token_50} not found in LMDB database.")
                    
                    frame_35 = wod_e2ed_pb2.E2EDFrame()
                    frame_35.ParseFromString(raw_record_35)
                    past_states_35 = frame_35.past_states

                    frame_50 = wod_e2ed_pb2.E2EDFrame()
                    frame_50.ParseFromString(raw_record_50)
                    past_states_50 = frame_50.past_states

                    his_traj_50 = np.column_stack((past_states_50.pos_x, past_states_50.pos_y))
                    his_traj_35 = np.column_stack((past_states_35.pos_x, past_states_35.pos_y))
                    his_traj = np.column_stack((past_states.pos_x, past_states.pos_y))

                    # 5s to 3s to fill out the 5s future
                    his_traj_50_in_35 = transform_trajectory(his_traj_50[-8:-6, :], his_traj_35[-2:   , :], his_traj_50)
                    his_traj_35_full = np.vstack((his_traj_35, his_traj_50_in_35[-6:, :]))

                    # 3s to 0s
                    his_traj_35_in_base = transform_trajectory(
                        src_pts=his_traj_35_full[:2, :],
                        dst_pts=his_traj[-2:, :],
                        traj_B=his_traj_35_full
                    )

                    trans_x = his_traj_35_in_base[2:, 0]
                    trans_y = his_traj_35_in_base[2:, 1]
                    gt_head_raw = calculate_heading(trans_x, trans_y)
                    gt_raw_trajectory = list(
                        zip(
                            trans_x[1::traj_freq_ratio],
                            trans_y[1::traj_freq_ratio],
                            gt_head_raw[1::traj_freq_ratio]
                        )
                    )
            else:
                gt_head_raw = calculate_heading(future_states.pos_x, future_states.pos_y)
                gt_raw_trajectory = list(zip(future_states.pos_x[1::traj_freq_ratio], 
                                            future_states.pos_y[1::traj_freq_ratio],
                                            gt_head_raw[1::traj_freq_ratio]))
        
        preference_trajectories = []
        preference_scores = []
        if frame.preference_trajectories:
            for traj in frame.preference_trajectories:
                if traj.preference_score < 0:
                    continue
                xs = list(traj.pos_x)
                ys = list(traj.pos_y)
                preference_trajectories.append(list(zip(xs, ys)))
                preference_scores.append(traj.preference_score)

        # vehicle history and future action
        his_ego_action = get_action_instruction(np.stack([past_states.pos_x[1::traj_freq_ratio],
                                                          past_states.pos_y[1::traj_freq_ratio]
                                                        ], axis=1), 
                                                np.stack([past_states.vel_x[3::traj_freq_ratio], 
                                                          past_states.vel_y[3::traj_freq_ratio]], 
                                                          axis=1))
        
        if self.split != 'test':
            gt_raw_trajectory_np = np.stack([future_states.pos_x[1::traj_freq_ratio],
                                                future_states.pos_y[1::traj_freq_ratio]], axis=1)
            gt_velocity = np.diff(gt_raw_trajectory_np, axis=0) * self.config['raw_trajectory_freq']
            gt_velocity = np.concatenate([gt_velocity, gt_velocity[-1:]], axis=0)
            fut_ego_action = get_action_instruction(np.stack([future_states.pos_x[1::traj_freq_ratio],
                                                            future_states.pos_y[1::traj_freq_ratio]], axis=1), 
                                                            gt_velocity[1:13:traj_freq_ratio])
        else:
            fut_ego_action = "unknown"

        # image sensor
        image_freq_ratio = int(self.config['model_freq'] / self.config['inference_freq'])
        start_frame_index = self.config['model_his_frames'] -\
            (self.config['inference_his_frames'] - 1) * image_freq_ratio - 1
        end_frame_index = self.config['model_his_frames']
        front_video = []
        for i in range(start_frame_index, end_frame_index, image_freq_ratio):
            front_video.append("data:image/jpeg;base64," + process_image_input(images['front_camera'][i]))

        front_left_video = []
        for i in range(start_frame_index, end_frame_index, image_freq_ratio):
            front_left_video.append("data:image/jpeg;base64," + process_image_input(images['front_left_camera'][i]))

        front_right_video = []
        for i in range(start_frame_index, end_frame_index, image_freq_ratio):
            front_right_video.append("data:image/jpeg;base64," + process_image_input(images['front_right_camera'][i]))

        back_video = []
        for i in range(start_frame_index, end_frame_index, image_freq_ratio):
            back_video.append("data:image/jpeg;base64," + process_image_input(images['back_camera'][i]))

        # create messages
        messages = [
            {   
                "role": "system",
                "content": "As a professional driver, how do you drive in the following scenario."
            },

            {
                "role": "user",
                "content": [
                    # sensor inputs information
                    {
                        "type": "text", 
                        "text": "Four cameras are mounted on the vehicle to perceive the surrounding environment. " +
                                "These cameras provide the front, front left, front right, and back views. " +
                                "The multi-view multi-frame camera images are organized in a video format. "
                    },

                    # camera images
                    {
                        "type": "text", 
                        "text": "The video is from the front camera, capturing the history of the vehicle's front view from the past four seconds at 1Hz."
                    },
                    {
                        "type": "video",
                        "min_pixels": 400 * 400,
                        "max_pixels": 400 * 400,
                        "video": front_video,
                    },

                    {
                        "type": "text", 
                        "text": "The video is from the front left camera, capturing the history of the vehicle's front left view from the past four seconds at 1Hz."
                    },
                    {
                        "type": "video",
                        "min_pixels": 400 * 400,
                        "max_pixels": 400 * 400,
                        "video": front_left_video,
                    },

                    {
                        "type": "text", 
                        "text": "The video is from the front right camera, capturing the history of the vehicle's front right view from the past four seconds at 1Hz."
                    },
                    {
                        "type": "video",
                        "min_pixels": 400 * 400,
                        "max_pixels": 400 * 400,
                        "video": front_right_video,
                    },

                    {
                        "type": "text", 
                        "text": "The video is from the back camera, capturing the history of the vehicle's back view from the past four seconds at 1Hz."
                    },
                    {
                        "type": "video",
                        "min_pixels": 400 * 400,
                        "max_pixels": 400 * 400,
                        "video": back_video,
                    },
    
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
                ]
            },
        ]

        inputs = {'messages': messages, 'token': scene_token}

        # Qwen-specific processing (only when processor is available, i.e., vLLM backend)
        if self.processor is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )
            inputs['text'] = text
            inputs['image_inputs'] = image_inputs
            inputs['video_inputs'] = video_inputs

        for side in CAM_LIST:
            path_key = f"{side}_camera_paths"
            inputs[path_key] = images_path[path_key]
        inputs.update({
            "velocity": [float(velocity[0]), float(velocity[1])],
            "acceleration": [float(acceleration[0]), float(acceleration[1])],
            "instruction": instruction,
            "gt_trajectory": gt_raw_trajectory,
            "his_trajectory": his_raw_trajectory,
            "preference_scores": preference_scores,
            "preference_trajectories": preference_trajectories,
            "fut_ego_action": fut_ego_action,
        })

        return inputs
    
    def load_lmdb(self, lmdb_dir, map_size= int(1.5 * 1024 * 1024 * 1024 * 1024)):
        if not os.path.exists(lmdb_dir):
            os.makedirs(lmdb_dir, exist_ok=True)
            print(f"LMDB DB not found in {lmdb_dir}, building LMDB database...")

            env = lmdb.open(lmdb_dir, map_size=map_size)

            with env.begin(write=True) as txn:
                for raw_record in tqdm(self._tfdata, desc="Building LMDB DB"):
                    data = wod_e2ed_pb2.E2EDFrame()
                    data.ParseFromString(raw_record.numpy())
                    token = data.frame.context.name

                    txn.put(token.encode('utf-8'), raw_record.numpy())
            env.sync()
            env.close()
            print(f"LMDB DB built successfully at {lmdb_dir}")
        else:
            print(f"Using existing LMDB DB in {lmdb_dir}")

        lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False)
        return lmdb_env
    
    def _make_scene_entry(self, sequence_name, sequence_dir, frame_sequence):
        scene_entry = {}
        for cam in CAM_LIST:
            cam_folder = os.path.join(sequence_dir, cam)
            paths = []
            for f in frame_sequence:
                file_name = f"{sequence_name}-{f:03d}.jpg"
                file_path = os.path.join(cam_folder, file_name)
                if not os.path.isfile(file_path):
                    print(f"Missing file: {file_path} in scene {frame_sequence[-1]}")
                    return None
                paths.append(file_path)
            scene_entry[f'{cam}_camera_paths'] = paths
        return scene_entry
    
    def scene_loader(self):
        split = self.split
        allowed_splits = ('training', 'val', 'test')
        if split not in allowed_splits:
            raise ValueError(f"Invalid dataset_split '{split}'. Must be one of {allowed_splits}.")
        
        # Build the path to the images folder
        images_path = os.path.join(self.config['dataset_path'], split + '_images')

        frequency_ratio = int(self.config['raw_images_freq'] / (self.config['model_freq']))
        num_history_frames = frequency_ratio * (self.config['model_his_frames'] - 1) + 1 
        num_fut_frames = frequency_ratio * (self.config['model_fut_frames'])
        
        scenes = {}
        
        # Iterate over each sequence folder
        for sequence_name in os.listdir(images_path):
            sequence_dir = os.path.join(images_path, sequence_name)
            if not os.path.isdir(sequence_dir): continue
            
            # Check if the front camera folder exists
            front_cam_dir = os.path.join(sequence_dir, "front")
            if not os.path.isdir(front_cam_dir): continue

            # Extract all frame numbers from the front camera folder
            frame_nums = sorted(
                int(os.path.splitext(f)[0].split('-')[-1])
                for f in os.listdir(front_cam_dir) if f.endswith('.jpg')
            )
            if not frame_nums: continue
            min_frame, max_frame = frame_nums[0], frame_nums[-1]

            if split == 'test':
                # Loop over current frames that have enough historical frames
                for current_frame in range(min_frame + num_history_frames + self.config["frame_shift"] - 1, 
                                        max_frame + 1 - num_fut_frames, self.config['scene_frame_interval']):
                    frame_sequence = [current_frame - m for m in range(num_history_frames - 1, -1, -frequency_ratio)]
                    # Skip scene if any frame in the sequence is missing
                    if any(f not in frame_nums for f in frame_sequence): continue

                    entry = self._make_scene_entry(sequence_name, sequence_dir, frame_sequence)
                    if entry:
                        scenes[f"{sequence_name}-{current_frame:03d}"] = entry

            if split == 'training':
                # Loop over current frames that have enough historical frames
                for current_frame in range(min_frame + num_history_frames + self.config["frame_shift"] - 1, 
                                        max_frame + 1, self.config['scene_frame_interval']):
                    frame_sequence = [current_frame - m for m in range(num_history_frames - 1, -1, -frequency_ratio)]
                    # Skip scene if any frame in the sequence is missing
                    if any(f not in frame_nums for f in frame_sequence): continue

                    entry = self._make_scene_entry(sequence_name, sequence_dir, frame_sequence)
                    if entry:
                        scenes[f"{sequence_name}-{current_frame:03d}"] = entry

            if split == 'val':
                # Loop over current frames that have enough historical frames
                for current_frame in range(min_frame + num_history_frames + self.config["frame_shift"] - 1, 
                                        max_frame + 1 - num_fut_frames, self.config['scene_frame_interval']):
                    frame_sequence = [current_frame - m for m in range(num_history_frames - 1, -1, -frequency_ratio)]
                    # Skip scene if any frame in the sequence is missing
                    if any(f not in frame_nums for f in frame_sequence): continue

                    entry = self._make_scene_entry(sequence_name, sequence_dir, frame_sequence)
                    if entry:
                        scenes[f"{sequence_name}-{current_frame:03d}"] = entry


        # Convert the scenes dictionary to a list for indexed access
        return list(scenes.items())


def process_image_input(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode()

    return base64_image

def calculate_heading(pos_x, pos_y, static_thresh=0.07):
    pts_x = [0.0] + list(pos_x)
    pts_y = [0.0] + list(pos_y)
    n = len(pts_x) 

    gt_head_raw = []
    dists = []
    if n >= 2:
        for i in range(1, n):
            dx = pts_x[i] - pts_x[i - 1]
            dy = pts_y[i] - pts_y[i - 1]
            d = np.hypot(dx, dy)
            dists.append(d)
            head = float(np.arctan2(dy, dx+1e-4))
            gt_head_raw.append(head)
    else:
        gt_head_raw = [0.0] * (n - 1)
        dists= [0.0] * (n - 1)

    is_static = [d < static_thresh for d in dists]

    for i, static in enumerate(is_static):
        if not static:
            continue  
        # find the closed left point
        l = i - 1
        while l >= 0 and is_static[l]:
            l -= 1
        # find the closed right point
        r = i + 1
        while r < len(is_static) and is_static[r]:
            r += 1

        if l >= 0:
            fill_head = gt_head_raw[l]
        elif r < len(is_static):
            fill_head = gt_head_raw[r]
        else:
            fill_head = 0.0

        gt_head_raw[i] = fill_head
    return gt_head_raw

def transform_trajectory(src_pts: np.ndarray,
                         dst_pts: np.ndarray,
                         traj_B: np.ndarray) -> np.ndarray:
            mu_src = src_pts.mean(axis=0)
            mu_dst = dst_pts.mean(axis=0)
            X = src_pts - mu_src    # 2×2
            Y = dst_pts - mu_dst    # 2×2

            H = X.T.dot(Y)          # 2×2
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T.dot(U.T)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T.dot(U.T)

            t = mu_dst - R.dot(mu_src)

            return (R.dot(traj_B.T).T + t)

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
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch["text"] = [feature["text"] for feature in features]
        batch["image_inputs"] = [feature["image_inputs"] for feature in features]
        batch["video_inputs"] = [feature["video_inputs"] for feature in features]
        batch["token"] = [feature["token"] for feature in features]

        batch["velocity"] = [feature["velocity"] for feature in features]
        batch["acceleration"] = [feature["acceleration"] for feature in features]
        batch["instruction"] = [feature["instruction"] for feature in features]

        batch["gt_trajectory"] = [feature["gt_trajectory"] for feature in features]
        batch["his_trajectory"] = [feature["his_trajectory"] for feature in features]
        batch["preference_scores"] = [f["preference_scores"] for f in features]
        batch["preference_trajectories"] = [f["preference_trajectories"] for f in features]

        for side in CAM_LIST:
            path_key = f"{side}_camera_paths"
            batch[path_key] = [feature[path_key] for feature in features]

        return batch