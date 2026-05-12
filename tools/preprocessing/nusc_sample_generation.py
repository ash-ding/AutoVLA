"""
NuScenes Dataset Preprocessing Script for DriveVLA.

This script extracts trajectory data and camera paths from the NuScenes dataset
and saves them as JSON files for training and validation.

Usage:
    python tools/preprocessing/nusc_sample_generation.py \
        --nuscenes_path /path/to/nuscenes \
        --output_dir /path/to/output \
        --split train \
        --drivelm_path /path/to/drivelm/v1_1_train_nus.json

Arguments:
    --nuscenes_path: Path to NuScenes dataset root directory
    --output_dir: Output directory for preprocessed JSON files
    --split: Dataset split to process (train or val)
    --version: NuScenes dataset version (default: v1.0-trainval)
    --drivelm_path: Optional path to DriveLM annotations JSON file
"""

import json
import os
import argparse
import numpy as np
import torch
import math

from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils import splits

# Canonical source lives in dataset_utils/preprocessing/nuscenes_dataset.py so
# both this script and NuscenesCoTAnnotationDataset share the same logic.
from dataset_utils.preprocessing.nuscenes_dataset import (  # noqa: F401
    get_global_sensor_pose,
    get_ego_pose_future_his,
    get_ego_velocity,
    get_planning_instruction,
)


def convert_to_json_serializable(obj):
    """Convert numpy arrays and torch tensors to JSON serializable format."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)


def quart_to_rpy(qua):
    """Convert quaternion to roll, pitch, yaw."""
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def extract_nuscenes_data(nusc, save_path, split='train', drivelm_path=None,
                          cot_output_dir=None, nocot_output_dir=None):
    """
    Extract data from NuScenes dataset.

    Args:
        nusc: NuScenes instance
        save_path: Fallback output directory for JSON files. When cot/nocot
            override paths are both supplied, this is only used as a sentinel.
        split: 'train' or 'val'
        drivelm_path: Optional path to DriveLM annotations
        cot_output_dir: Optional override for DriveLM-derived CoT JSONs.
            Bypasses the default '<save_path>/nl_reasoning_samples/' subdir.
        nocot_output_dir: Optional override for no-CoT JSONs.
            Bypasses the default '<save_path>/action_only_samples/' subdir.
    """
    # Get scene splits
    train_scenes = splits.train
    val_scenes = splits.val

    available_scenes = nusc.scene
    available_scene_names = [s['name'] for s in available_scenes]

    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    
    train_scene_tokens = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scene_tokens = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    scene_set = train_scene_tokens if split == 'train' else val_scene_tokens

    # Output layout. Per stream, the precedence is:
    #   explicit override (--cot_output_dir / --nocot_output_dir)
    #     > '<save_path>/nl_reasoning_samples' or '.../action_only_samples' subdir, if drivelm_path is set
    #     > flat 'save_path' (single no-CoT stream).
    # Overrides let downstream callers land the two streams into independently-named
    # sibling dirs without the default subdir layer.
    if cot_output_dir:
        cot_save_path = cot_output_dir
    elif drivelm_path:
        cot_save_path = os.path.join(save_path, 'nl_reasoning_samples')
    else:
        cot_save_path = save_path

    if nocot_output_dir:
        nocot_save_path = nocot_output_dir
    elif drivelm_path:
        nocot_save_path = os.path.join(save_path, 'action_only_samples')
    else:
        nocot_save_path = save_path

    os.makedirs(cot_save_path, exist_ok=True)
    os.makedirs(nocot_save_path, exist_ok=True)
    if drivelm_path:
        print(f"CoT samples (DriveLM-derived) -> {cot_save_path}")
    print(f"No-CoT samples                 -> {nocot_save_path}")

    # Load DriveLM data if provided
    drivelm_samples = set()
    if drivelm_path and os.path.exists(drivelm_path):
        print(f"Loading DriveLM annotations from {drivelm_path}")
        with open(drivelm_path, 'r') as f:
            drivelm_data = json.load(f)
        drivelm_samples = process_drivelm_data(
            nusc, drivelm_data, scene_set, cot_save_path
        )
    
    # Process raw NuScenes samples
    camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
    fut_ts = 16
    his_ts = 3
    
    processed_count = 0
    skipped_count = 0
    
    for sample in tqdm(nusc.sample, desc=f"Processing NuScenes {split} samples"):
        # Skip if already processed from DriveLM
        if sample['token'] in drivelm_samples:
            continue
        
        # Filter by scene
        if sample['scene_token'] not in scene_set:
            continue
        
        # Skip if no previous frame (needed for velocity calculation)
        if sample['prev'] == '':
            skipped_count += 1
            continue
        
        frame_id = sample['token']
        sample_data = {'token': frame_id, 'dataset_name': 'nuscenes'}
        
        # Get trajectory data
        result = get_ego_pose_future_his(nusc, fut_ts, sample, his_ts)
        gt_ego_fut_diff, gt_ego_fut_trajs, gt_ego_his_trajs, gt_ego_his_diff, \
            gt_ego_fut_masks, gt_ego_his_masks, command = result
        
        # Validate trajectory data (only check future masks for training)
        if split == 'train':
            if np.sum(gt_ego_fut_masks) < 10:
                skipped_count += 1
                continue
        if np.sum(gt_ego_his_masks) < 3:
            skipped_count += 1
            continue

        # Get velocity and acceleration
        ego_v = get_ego_velocity(sample, nusc)
        sample_prev = nusc.get('sample', sample['prev'])
        ego_v_previous = get_ego_velocity(sample_prev, nusc)
        ego_acc = (ego_v - ego_v_previous) / 0.5

        # Get camera paths for history frames
        sample_data['front_camera_paths'] = []
        sample_data['front_right_camera_paths'] = []
        sample_data['front_left_camera_paths'] = []
        sample_data['back_camera_paths'] = []
        sample_data['back_right_camera_paths'] = []
        sample_data['back_left_camera_paths'] = []

        sample_cur = sample
        for i in range(his_ts, -1, -1):
            if sample_cur is not None:
                for camera_type in camera_types:
                    key_suffix = camera_type.lower().replace('cam_', '')
                    cam_token = sample_cur['data'][camera_type]
                    cam_path, _, _ = nusc.get_sample_data(cam_token)
                    sample_data[f'{key_suffix}_camera_paths'].insert(0, cam_path)
                if i != 0:
                    sample_cur = nusc.get('sample', sample_cur['prev'])
            else:
                skipped_count += 1
                break
        else:
            # Convert trajectory to x-forward, y-left coordinate frame
            gt_ego_fut_trajs_output = gt_ego_fut_trajs[:11]
            gt_ego_fut_trajs_output_f = np.zeros((gt_ego_fut_trajs_output.shape[0], 3))
            gt_ego_fut_trajs_output_f[:, 0] = gt_ego_fut_trajs_output[:, 1]
            gt_ego_fut_trajs_output_f[:, 1] = -gt_ego_fut_trajs_output[:, 0]

            # Calculate heading angles
            heading = np.arctan2(
                gt_ego_fut_trajs_output_f[1:, 1] - gt_ego_fut_trajs_output_f[:-1, 1],
                gt_ego_fut_trajs_output_f[1:, 0] - gt_ego_fut_trajs_output_f[:-1, 0] + 1e-3
            )
            gt_ego_fut_trajs_output_f[1:, 2] = heading

            sample_data['gt_trajectory'] = gt_ego_fut_trajs_output_f[1:]
            sample_data['cot_output'] = []
            sample_data['instruction'] = command
            sample_data['velocity'] = ego_v
            sample_data['acceleration'] = ego_acc
            
            # Include future_mask for evaluation data
            if split == 'val':
                sample_data['future_mask'] = gt_ego_fut_masks[:10]

            # Save to JSON
            json_data = convert_to_json_serializable(sample_data)
            output_path = os.path.join(nocot_save_path, f"{frame_id}.json")
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            processed_count += 1

    print(f"Processed {processed_count} samples, skipped {skipped_count} samples")
    return processed_count


def process_drivelm_data(nusc, drivelm_data, scene_set, save_path):
    """Process samples with DriveLM annotations."""
    camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
    fut_ts = 16
    his_ts = 3
    processed_samples = set()
    
    for scene_id in tqdm(drivelm_data.keys(), desc="Processing DriveLM data"):
        if scene_id not in scene_set:
            continue
            
        scene_data = drivelm_data[scene_id]['key_frames']
        
        for frame_id in scene_data.keys():
            processed_samples.add(frame_id)
            frame_data_infos = scene_data[frame_id]['key_object_infos']
            frame_data_qa = scene_data[frame_id]['QA']
            
            sample_data = {'token': frame_id, 'dataset_name': 'nuscenes'}
            
            # Extract QA data
            qa_data = {
                'perception': [],
                'prediction': [],
                'fov': [],
                'move_intent': []
            }

            perception = frame_data_qa.get("perception", [])
            prediction = frame_data_qa.get("prediction", [])

            # Process perception QA
            for qa in perception:
                question = qa['Q'].lower()
                answer = qa['A']
                
                fov_questions = [
                    'what are objects to the front of the ego car?',
                    'what are objects to the back of the ego car?',
                    'what are objects to the front right of the ego car?',
                    'what are objects to the front left of the ego car?',
                    'what are objects to the back left of the ego car?',
                    'what are objects to the back right of the ego car?'
                ]
                
                for fov_q in fov_questions:
                    if fov_q in question:
                        qa_data['fov'].append(answer)
                        break
                
                if 'what are the important objects in the current scene?' in question:
                    description = answer.split('.')[0]
                    qa_data['perception'].append(description)

            # Process prediction QA
            for qa in prediction:
                question = qa['Q']
                answer = qa['A']
                
                if "what object should the ego vehicle notice first" in question.lower():
                    description = answer
                    while '<' in description and '>' in description:
                        obj_id = description[description.find('<'):description.find('>') + 1]
                        obj_id_nospace = obj_id.replace(" ", "")
                        if obj_id_nospace in frame_data_infos:
                            obj_description = "**" + frame_data_infos[obj_id_nospace]['Visual_description'].split('.')[0] + "**"
                            description = description.replace(obj_id, obj_description)
                        else:
                            break
                    qa_data['prediction'].append(description)

                if "what is the future state" in question.lower():
                    move_intent = answer.split('.')[0].lower()
                    move_obj = question[question.find('<'):question.find('>') + 1]
                    move_obj_nospace = move_obj.replace(" ", "")
                    if move_obj_nospace in frame_data_infos:
                        move_obj_description = "**" + frame_data_infos[move_obj_nospace]['Visual_description'].split('.')[0] + "**"
                        move_status = f"The moving status of {move_obj_description} is {move_intent}."
                        qa_data['move_intent'].append(move_status)

            # Get trajectory data
            try:
                sample = nusc.get('sample', frame_id)
            except KeyError:
                continue
                
            result = get_ego_pose_future_his(nusc, fut_ts, sample, his_ts)
            gt_ego_fut_diff, gt_ego_fut_trajs, gt_ego_his_trajs, gt_ego_his_diff, \
                gt_ego_fut_masks, gt_ego_his_masks, command = result
            
            if np.sum(gt_ego_fut_masks) < 10 or np.sum(gt_ego_his_masks) < 3:
                continue

            ego_v = get_ego_velocity(sample, nusc)
            if sample['prev'] == '':
                continue
            sample_prev = nusc.get('sample', sample['prev'])
            ego_v_previous = get_ego_velocity(sample_prev, nusc)
            ego_acc = (ego_v - ego_v_previous) / 0.5

            # Get camera paths
            sample_data['front_camera_paths'] = []
            sample_data['front_right_camera_paths'] = []
            sample_data['front_left_camera_paths'] = []
            sample_data['back_camera_paths'] = []
            sample_data['back_right_camera_paths'] = []
            sample_data['back_left_camera_paths'] = []

            sample_cur = sample
            valid = True
            for i in range(his_ts, -1, -1):
                if sample_cur is not None:
                    for camera_type in camera_types:
                        key_suffix = camera_type.lower().replace('cam_', '')
                        cam_token = sample_cur['data'][camera_type]
                        cam_path, _, _ = nusc.get_sample_data(cam_token)
                        sample_data[f'{key_suffix}_camera_paths'].insert(0, cam_path)
                    if i != 0:
                        sample_cur = nusc.get('sample', sample_cur['prev'])
                else:
                    valid = False
                    break
            
            if not valid:
                continue

            # Convert trajectory
            gt_ego_fut_trajs_output = gt_ego_fut_trajs[:11]
            gt_ego_fut_trajs_output_f = np.zeros((gt_ego_fut_trajs_output.shape[0], 3))
            gt_ego_fut_trajs_output_f[:, 0] = gt_ego_fut_trajs_output[:, 1]
            gt_ego_fut_trajs_output_f[:, 1] = -gt_ego_fut_trajs_output[:, 0]

            heading = np.arctan2(
                gt_ego_fut_trajs_output_f[1:, 1] - gt_ego_fut_trajs_output_f[:-1, 1],
                gt_ego_fut_trajs_output_f[1:, 0] - gt_ego_fut_trajs_output_f[:-1, 0] + 1e-3
            )
            gt_ego_fut_trajs_output_f[1:, 2] = heading

            sample_data['gt_trajectory'] = gt_ego_fut_trajs_output_f[1:]
            sample_data['instruction'] = command
            sample_data['velocity'] = ego_v
            sample_data['acceleration'] = ego_acc

            # Get planning instruction
            planning_instruction = get_planning_instruction(
                gt_ego_fut_diff, gt_ego_fut_trajs, gt_ego_his_diff, gt_ego_his_trajs
            )

            # Build CoT output
            fov_text = ' '.join(qa_data['fov']) if qa_data['fov'] else ''
            perception_text = qa_data['perception'][0] if qa_data['perception'] else ''
            move_intent_text = ' '.join(qa_data['move_intent']) if qa_data['move_intent'] else ''
            prediction_text = qa_data['prediction'][0] if qa_data['prediction'] else ''
            
            sample_data['cot_output'] = [fov_text, perception_text, move_intent_text, prediction_text, planning_instruction]

            # Save to JSON
            json_data = convert_to_json_serializable(sample_data)
            output_path = os.path.join(save_path, f"{frame_id}.json")
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
    
    return processed_samples


def main():
    parser = argparse.ArgumentParser(
        description="NuScenes Dataset Preprocessing for DriveVLA"
    )
    parser.add_argument(
        "--nuscenes_path", type=str, required=True,
        help="Path to NuScenes dataset root directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for preprocessed JSON files. Required unless "
             "both --cot_output_dir and --nocot_output_dir are supplied."
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val"],
        help="Dataset split to process (train or val)"
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-trainval",
        help="NuScenes dataset version"
    )
    parser.add_argument(
        "--drivelm_path", type=str, default=None,
        help="Optional path to DriveLM annotations JSON file"
    )
    parser.add_argument(
        "--cot_output_dir", type=str, default=None,
        help="Override: write DriveLM-derived CoT JSONs here instead of "
             "<output_dir>/nl_reasoning_samples/. Only meaningful with --drivelm_path."
    )
    parser.add_argument(
        "--nocot_output_dir", type=str, default=None,
        help="Override: write no-CoT JSONs here instead of "
             "<output_dir>/action_only_samples/ (or <output_dir> in the flat case)."
    )
    args = parser.parse_args()

    if args.output_dir is None and not (args.cot_output_dir and args.nocot_output_dir):
        parser.error(
            "--output_dir is required unless BOTH --cot_output_dir and "
            "--nocot_output_dir are provided."
        )

    # Create output directory (parent placeholder; per-stream dirs are created inside extract_nuscenes_data)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize NuScenes
    print(f"Loading NuScenes {args.version} from {args.nuscenes_path}")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_path, verbose=True)

    # Process data
    print(f"Processing {args.split} split...")
    extract_nuscenes_data(
        nusc=nusc,
        save_path=args.output_dir or "",
        split=args.split,
        drivelm_path=args.drivelm_path,
        cot_output_dir=args.cot_output_dir,
        nocot_output_dir=args.nocot_output_dir,
    )

    if args.cot_output_dir or args.nocot_output_dir:
        if args.cot_output_dir:
            print(f"Done! CoT samples saved to {args.cot_output_dir}")
        if args.nocot_output_dir:
            print(f"Done! No-CoT samples saved to {args.nocot_output_dir}")
    else:
        print(f"Done! Preprocessed data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
