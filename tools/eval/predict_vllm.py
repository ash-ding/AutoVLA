"""
vLLM-backed replacement for AutoVLA.predict(). Drop-in compatible interface:

    predictor = VLLMAutoVLAPredictor(config, hf_dir="/backup/hf_ckpt/4v90")
    predictor.initialize()                         # loads vLLM (slow, once)
    trajectory, cot_text = predictor.predict(input_features)

The class wraps the vLLM engine so the heavy load happens once per process
(unlike a function-level call). For end-to-end eval, the caller instantiates
this once per worker and reuses it.

Compared to AutoVLA.predict():
  - Same input contract (input_features dict produced by AutoVLAAgent feature
    builder, with `images`, `sensor_data_path`, `vehicle_velocity`,
    `vehicle_acceleration`, `instruction`).
  - Same output shape: (trajectory tensor [num_poses, 3], cot_text str).
  - Sampling params come from the same config (gen_conf).
"""
import os
import pickle
from pathlib import Path

import numpy as np
import torch


class VLLMAutoVLAPredictor:
    def __init__(self, config: dict, hf_dir: str, dataset_name: str = "nuscenes",
                 nuplan_side_field: str = "left",
                 gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 8192):
        """
        :param config: full SFTAutoVLA config dict (model.tokens, model.video, training.sample, etc.)
        :param hf_dir: path to converted HF safetensors dir (e.g. /backup/hf_ckpt/4v90)
        :param dataset_name: 'nuscenes' or 'nuplan' — controls camera key mapping
        :param nuplan_side_field: 'left' (CAM_L1/R1, 90°) or 'front_left' (CAM_L0/R0, 45°)
        """
        self.cfg = config
        self.hf_dir = hf_dir
        self.dataset_name = dataset_name
        self.nuplan_side_field = nuplan_side_field
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        # Tokenization/decoding parameters (mirrors AutoVLA.__init__)
        self.action_start_id = config["model"]["tokens"]["action_start_id"]
        self.num_poses = config["model"]["trajectory"]["num_poses"]
        self.video_conf = config["model"]["video"]
        self.gen_conf = config["inference"]["sample"]  # matches AutoVLA.__init__
        self.codebook_path = config["model"]["codebook_cache_path"]

        # vLLM engine (lazy)
        self.llm = None
        self.processor = None
        self.tokenizer = None
        self.action_tokenizer = None

    def initialize(self):
        """Load vLLM + action tokenizer + HF processor."""
        from vllm import LLM
        from transformers import AutoProcessor

        print(f"[VLLMAutoVLAPredictor] Loading vLLM from {self.hf_dir}")
        self.llm = LLM(
            model=self.hf_dir,
            dtype="bfloat16",
            enforce_eager=True,
            disable_custom_all_reduce=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            limit_mm_per_prompt={"image": 0, "video": 4},
        )

        print(f"[VLLMAutoVLAPredictor] Loading processor")
        self.processor = AutoProcessor.from_pretrained(self.hf_dir)
        self.tokenizer = self.processor.tokenizer

        # Action tokenizer (matches AutoVLA.action_tokenizer)
        from models.action_tokenizer import ActionTokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer, self.cfg["model"])
        print(f"[VLLMAutoVLAPredictor] Ready (action_start_id={self.action_start_id}, "
              f"num_poses={self.num_poses})")

    def _build_messages(self, input_features: dict) -> list:
        """BYTE-IDENTICAL replica of AutoVLA.get_prompt() messages.

        Any deviation here causes semantic-level output divergence from the HF
        path (beyond what FlashAttention numerics alone explain). Production
        copy lives in models/autovla.py:get_prompt().
        """
        images = input_features["images"]
        sensor_data_path = input_features["sensor_data_path"]
        min_pixels = self.video_conf.get("min_pixels", 28 * 28 * 128)
        max_pixels = self.video_conf.get("max_pixels", 28 * 28 * 128)

        # Pick the right 3 camera keys (nuplan side field aware)
        if self.dataset_name == "nuplan":
            if self.nuplan_side_field == "left":
                fl_key, fr_key = "left_camera", "right_camera"
            else:
                fl_key, fr_key = "front_left_camera", "front_right_camera"
            cam_keys = ["front_camera", fl_key, fr_key]
        else:
            cam_keys = ["front_camera", "front_left_camera", "front_right_camera"]

        camera_images = {}
        for ck in cam_keys:
            camera_images[ck] = [
                os.path.join(sensor_data_path, images[ck][i]) for i in range(4)
            ]
        front_camera_1, front_camera_2, front_camera_3, front_camera_4 = camera_images[cam_keys[0]]
        front_left_camera_1, front_left_camera_2, front_left_camera_3, front_left_camera_4 = camera_images[cam_keys[1]]
        front_right_camera_1, front_right_camera_2, front_right_camera_3, front_right_camera_4 = camera_images[cam_keys[2]]

        # Vehicle state
        velocity = input_features["vehicle_velocity"]
        if isinstance(velocity, (list, np.ndarray)):
            velocity = float(np.sqrt(velocity[0] ** 2 + velocity[1] ** 2))
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, (list, np.ndarray)):
            acceleration = float(np.sqrt(acceleration[0] ** 2 + acceleration[1] ** 2))
        instruction = input_features["driving_command"].lower()

        user_content = [
            {"type": "text", "text":
                "The autonomous vehicle is equipped with three cameras mounted at the front, front-left, and front-right, enabling a comprehensive perception of the surrounding environment."},
            {"type": "text", "text":
                "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels,
             "video": [f"file://{front_camera_1}", f"file://{front_camera_2}",
                       f"file://{front_camera_3}", f"file://{front_camera_4}"]},
            {"type": "text", "text":
                "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels,
             "video": [f"file://{front_left_camera_1}", f"file://{front_left_camera_2}",
                       f"file://{front_left_camera_3}", f"file://{front_left_camera_4}"]},
            {"type": "text", "text":
                "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels,
             "video": [f"file://{front_right_camera_1}", f"file://{front_right_camera_2}",
                       f"file://{front_right_camera_3}", f"file://{front_right_camera_4}"]},
            {"type": "text", "text":
                f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
                f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."},
        ]

        use_cot = self.cfg["model"].get("use_cot", True)
        if use_cot:
            system_text = (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will receive visual observations from the ego vehicle’s cameras and dynamic information about the vehicle’s current state. "
                "Your task is to predict the optimal driving action for the next five seconds.\n\n"
                "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
                "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
                "Structure your reasoning as follows:\n"
                "1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n"
                "2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n"
                "3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n"
                "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n"
                "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
                "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
                "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
                "Present the final action clearly after your reasoning steps."
            )
        else:
            system_text = (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will be provided with video observations from the ego vehicle’s surrounding cameras, along with the vehicle’s current dynamic states. "
                "Your task is to predict the most appropriate driving action for the next five seconds."
            )

        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

    def predict(self, input_features: dict, greedy: bool = False) -> tuple:
        """vLLM-backed inference. Drop-in for AutoVLA.predict()."""
        from vllm import SamplingParams
        from qwen_vl_utils import process_vision_info

        assert self.llm is not None, "Call initialize() first"

        messages = self._build_messages(input_features)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        if greedy:
            sp = SamplingParams(temperature=0.0, max_tokens=self.gen_conf["max_length"], n=1)
        else:
            sp = SamplingParams(
                temperature=self.gen_conf.get("temperature", 0.7),
                top_p=self.gen_conf.get("top_p", 0.9),
                top_k=self.gen_conf.get("top_k", 40),
                max_tokens=self.gen_conf["max_length"],
                n=1,
            )

        prompt = {"prompt": text, "multi_modal_data": {"video": video_inputs}}
        outputs = self.llm.generate([prompt], sampling_params=sp)
        gen = outputs[0].outputs[0]

        # Extract action tokens (>= action_start_id) — same logic as AutoVLA.predict()
        token_ids = torch.tensor(list(gen.token_ids), dtype=torch.long)
        actions_tokens = token_ids[token_ids >= self.action_start_id]
        cot_text = gen.text  # vLLM already gives us decoded text

        # Pad/truncate to num_poses
        if len(actions_tokens) > self.num_poses:
            actions_tokens = actions_tokens[:self.num_poses]
        elif len(actions_tokens) < self.num_poses:
            pad = torch.zeros(self.num_poses - len(actions_tokens), dtype=torch.long)
            actions_tokens = torch.cat([actions_tokens, pad]).long()

        trajectory = self.action_tokenizer.decode_token_ids_to_trajectory(actions_tokens)
        if isinstance(trajectory, list):
            trajectory = torch.zeros((self.num_poses + 1, 3), dtype=torch.float32)
        trajectory = trajectory[0, 1:]

        return trajectory, cot_text

    def predict_batch(self, input_features_list: list, greedy: bool = False) -> list:
        """vLLM-native batch predict. Submits all prompts at once -> continuous batching."""
        from vllm import SamplingParams
        from qwen_vl_utils import process_vision_info

        assert self.llm is not None
        prompts = []
        for inp in input_features_list:
            messages = self._build_messages(inp)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            _, video_inputs = process_vision_info(messages)
            prompts.append({
                "prompt": text,
                "multi_modal_data": {"video": video_inputs},
            })

        if greedy:
            sp = SamplingParams(temperature=0.0, max_tokens=self.gen_conf["max_length"], n=1)
        else:
            sp = SamplingParams(
                temperature=self.gen_conf.get("temperature", 0.7),
                top_p=self.gen_conf.get("top_p", 0.9),
                top_k=self.gen_conf.get("top_k", 40),
                max_tokens=self.gen_conf["max_length"], n=1,
            )

        outputs = self.llm.generate(prompts, sampling_params=sp)

        results = []
        for out in outputs:
            gen = out.outputs[0]
            token_ids = torch.tensor(list(gen.token_ids), dtype=torch.long)
            actions_tokens = token_ids[token_ids >= self.action_start_id]
            cot_text = gen.text
            if len(actions_tokens) > self.num_poses:
                actions_tokens = actions_tokens[:self.num_poses]
            elif len(actions_tokens) < self.num_poses:
                pad = torch.zeros(self.num_poses - len(actions_tokens), dtype=torch.long)
                actions_tokens = torch.cat([actions_tokens, pad]).long()
            trajectory = self.action_tokenizer.decode_token_ids_to_trajectory(actions_tokens)
            if isinstance(trajectory, list):
                trajectory = torch.zeros((self.num_poses + 1, 3), dtype=torch.float32)
            trajectory = trajectory[0, 1:]
            results.append((trajectory, cot_text))
        return results
