import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from navsim.agents.autovla_agent import AutoVLAAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from typing import Dict

class RFTDataset(Dataset):
    def __init__(self, data_config, model_config):
        data_paths = data_config['json_dataset_path']
        self.sensor_data_path = data_config['sensor_data_path']

        if isinstance(data_paths, (str, Path)):
            self.data_paths = [Path(data_paths)]
        else:
            self.data_paths = [Path(path) for path in data_paths]
        
        # Load trajectory config from global config
        traj_conf = model_config['trajectory']
        time_horizon = traj_conf['time_horizon']
        interval_length = traj_conf['interval_length']

        trajectory_sampling = TrajectorySampling(time_horizon=time_horizon, 
                                                interval_length=interval_length)

        self._agent = AutoVLAAgent(trajectory_sampling=trajectory_sampling,
                                    sensor_data_path=self.sensor_data_path,
                                    codebook_cache_path=model_config['codebook_cache_path'],
                                    skip_model_load=True,
                                    nuplan_side_field=model_config.get('nuplan_side_field', 'left'))
        
        # Get all JSON files from all data paths
        self.scenes = []
        for data_path in self.data_paths:
            path_scenes = sorted(list(data_path.glob('*.json')))
            self.scenes.extend(path_scenes)
            
        if not self.scenes:
            raise ValueError(f"No JSON files found in any of the provided data paths: {self.data_paths}")


    def __len__(self):
        # return len(self._scene_loader.tokens)
        return len(self.scenes)

    def __getitem__(self, idx):
        # Load data from JSON file
        input_features: Dict[str, torch.Tensor] = {}
        target_trajectory: Dict[str, torch.Tensor] = {}
        
        scene_path = self.scenes[idx]
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
            
        for builder in self._agent.get_feature_builders():
            input_features.update(builder.compute_features(scene_data))
        for builder in self._agent.get_target_builders():
            target_trajectory.update(builder.compute_targets(scene_data))

        # integrate the sensor data path
        if self.sensor_data_path:
            input_features.update({"sensor_data_path": self.sensor_data_path})

        return {'input_features': input_features, 
                'target_trajectory': target_trajectory, 
                'token': scene_data['token']}
    
    def collate_fn(self, batch):
        # Only work for batch size = 1
        input_features = batch[0]['input_features']
        target_trajectory = batch[0]['target_trajectory']
        token = batch[0]['token']

        return {'input_features': input_features, 
                'target_trajectory': target_trajectory, 
                'token': token}
