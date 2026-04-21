<div align="center">

# <img src="images/AutoVLA-Logo.png" width="35" height="35" style="vertical-align: middle; margin-right: 10px;"> AutoVLA 
<!-- <br> <span style="font-size: 0.5em; font-weight: normal;">A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning</span> -->

[![website](https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome)](https://autovla.github.io/)
[![paper](https://img.shields.io/badge/arXiv-2506.13757-B31B1B.svg?style=flat&logo=arxiv)](https://arxiv.org/abs/2506.13757)
[![dataset](https://img.shields.io/badge/Dataset-Coming_Soon-gre.svg?style=flat&logo=huggingface)]()
[![License](https://img.shields.io/badge/License-Academic_Software_License-blue)]()

</div>

[NeurIPS 2025] This is the official implementation of the paper:

**AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning**

[Zewei Zhou](https://zewei-zhou.github.io/)\*</sup>, [Tianhui Cai](https://www.tianhui-vicky.com/)\*</sup>, [Seth Z. Zhao](https://sethzhao506.github.io/), [Yun Zhang](https://handsomeyun.github.io/), [Zhiyu Huang](https://mczhi.github.io/)<sup>†</sup>, [Bolei Zhou](https://boleizhou.github.io/), [Jiaqi Ma](https://mobility-lab.seas.ucla.edu/about/)

University of California, Los Angeles | <sup>*</sup> Equal contribution, <sup>†</sup> Project leader

![teaser](images/AutoVLA_framework.png)


- 🚗 AutoVLA integrates **chain-of-thought (CoT) reasoning** and **physical action tokenization** to directly generate planning trajectories through a unified autoregressive generative process, dynamically switching thinking modes.
- ⚙️ **Supervised fine-tuning (SFT)** is employed to enable the model with dual thinking modes: fast thinking (trajectory-only) and slow thinking (enhanced with CoT reasoning). 
- 🪜 **Reinforcement fine-tuning (RFT)** based on Group Relative Policy Optimization (GRPO) is adopted to enhance planning performance and runtime efficiency, reducing unnecessary reasoning in straightforward scenarios.
- 🔥 Extensive experiments across real-world and simulated datasets and benchmarks, including **nuPlan**, **nuScenes**, **Waymo**, and **CARLA**, demonstrate its competitive performance in both open-loop and closed-loop settings. 

## News
- **`2026/02`**: AutoVLA codebase is now released.
- **`2025/09`**: AutoVLA is accepted by [NeurIPS 2025](https://neurips.cc/) 👏👏.
- **`2025/06`**: AutoVLA paper release.
- **`2025/05`**: In the [Waymo Vision-based End-to-end Driving Challenge](https://waymo.com/open/challenges/2025/e2e-driving/), AutoVLA ranks highly in both RFS Overall and achieves the top RFS Spotlight score, which focuses on the most challenging scenarios.

<!-- ## Overview
- [Release Plan](#release-plan)
- [Dataset](#dataset)
- [Citation](#citation) -->

## Release Plan
- **`2025/06`**: ✅ AutoVLA paper.
- **`2026/02`**: ✅ AutoVLA annotation and training code.
- **`2026/03`**: AutoVLA checkpoints.
- **`TBD`** : Reasoning data (Pending approval from the data provider).

## Devkit Setup
### 1. Dataset Downloading
#### nuPlan Dataset
You can refer to [here](https://github.com/autonomousvision/navsim/blob/main/docs/install.md) to prepare the nuPlan dataset. Be careful with the dataset structure.
```bash
bash navsim/download/download_maps.sh
bash navsim/download/download_trainval.sh
bash navsim/download/download_test.sh
```
#### Waymo E2E Dataset
The Waymo end-to-end driving dataset can be downloaded at [here](https://waymo.com/open/download/). 

#### nuScenes Dataset
The nuScenes dataset can be downloaded from the official website: [https://www.nuscenes.org/](https://www.nuscenes.org/). You will need to register and download the `v1.0-trainval` split.

### 2. Conda Environment Setup
You can perform the following command to create a conda environment and install the required dependencies.
```bash
conda env create -f environment.yml
conda activate autovla
pip install -e . --no-warn-conflicts
bash install.sh
```

### 3. Navsim Setup
We have included the navsim code in this repo, and you can go to the `navsim` folder to install it. You can also refer to [here](https://github.com/autonomousvision/navsim/blob/v2.0/docs/install.md) to set up the navsim devkit, but please ensure version compatibility for the dependencies.
```bash
cd navsim
pip install -e . --no-warn-conflicts
```
Remember to set the navsim required environment variables:
```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

### 4. Pretrained Model Downloading
We use the `Qwen2.5-VL` model series as the pretrained VLM in the VLA model and CoT annotation model. You can run the command to download the pretrained model.
```bash
bash scripts/download_qwen.sh
```
Specifically, we use the 72B model in CoT annotation, and you can choose `Qwen2.5-VL-72B-Instruct` or `Qwen2.5-VL-72B-Instruct-AWQ` based on your device. We use the `Qwen2.5-VL-3B-Instruct` in the AutoVLA model.

<!-- Retrieve `Qwen2.5-VL-3B-Instruct` from Hugging Face. -->


## Getting Started
### 1. Dataset Preprocessing
#### nuPlan Dataset
You can perform the command to preprocess the nuPlan dataset. Please first revise your path and data split (refer to [here](https://github.com/autonomousvision/navsim/blob/v2.0/docs/splits.md)) in the config. The `INCLUDE_COT` setting in the bash determines whether to launch the CoT reasoning annotation.
```bash
bash scripts/run_nuplan_preprocessing.sh
```
To preprocess only a saved subset of nuPlan tokens and make the job resumable after interruption, pass your token JSON and `--resume`:
```bash
bash scripts/run_nuplan_preprocessing.sh \
    --sample-ids-json /data/nuplan_random_45600_ids.json \
    --resume
```
#### Waymo E2E Dataset
To organize the image data and support random access, we first cache the image data in the same format as the other dataset we used.
```bash
bash scripts/run_waymo_e2e_image_extraction.sh
```
You can perform the following command to preprocess the Waymo E2E dataset. Please also first revise your path and data split in the config and set the `INCLUDE_COT`.
```bash
bash scripts/run_waymo_e2e_preprocessing.sh
```
You can use `waymo_e2e_traj_project_visualization.py` and `waymo_e2e_visualization.py` in the `tools/visualization` folder to visualize the Waymo data after preprocessing.
#### nuScenes Dataset
You can download the DriveLM nuScenes annotations (`v1_1_train_nus.json`) from [https://github.com/OpenDriveLab/DriveLM/tree/main/challenge](https://github.com/OpenDriveLab/DriveLM/tree/main/challenge).

**Note:** nuScenes preprocessing requires `nuscenes-devkit`, which might have dependency conflicts with the main environment. We recommend using a separate conda environment:
```bash
# Create a separate environment for nuScenes preprocessing
conda env create -f environment_nusc_preprocess.yml
conda activate nusc_preprocess

# Run preprocessing
bash scripts/run_nuscenes_preprocessing.sh \
    --nuscenes_path /path/to/nuscenes \
    --output_dir /path/to/output \
    --drivelm_path /path/to/drivelm/v1_1_train_nus.json

# Switch back to the main environment when done
conda activate autovla
```

### 2. Action Codebook Creation
The action codebook discretizes continuous vehicle trajectories into a finite vocabulary for autoregressive prediction. To create the codebook from your preprocessed data:
```bash
python tools/action_token/action_token_cluster.py \
    --data_path /path/to/preprocessed/nuplan/data \
    --output codebook_cache/agent_vocab.pkl \
    --num_cluster 2048
```
This will generate a vocabulary file that maps trajectory segments to discrete tokens.

### 3. Supervised Fine-tuning (SFT)
First revise the dataset path and SFT parameters in the config file in `config/training`. You can customize:
- `data.train.json_dataset_path`: Dataset paths for training (supports multiple datasets as a list)
- `data.train.sensor_data_path`: Corresponding sensor data paths
- `training.train_sample_size`: Set to a number to train on a random subset, or `null` for the full dataset
- `model.use_cot`: Enable/disable chain-of-thought reasoning in training data

Then, launch the SFT training:
```bash
python tools/run_sft.py --config training/qwen2.5-vl-3B-mix-sft
```

### 4. Reinforcement Fine-tuning (RFT)
You can revise your dataset path and GRPO parameters in the config file in `config/training`. Then, execute the following command to run reinforcement finetuning.
```bash
bash scripts/run_rft.sh
```

### 5. Evaluation
#### nuPlan Evaluation (Navsim)
We leverage Navsim and its Predictive Driver Model Score (PDMS) to test and evaluate our model on nuPlan. You need to set up the dataset path and split in the evaluation bash, and run the command to launch the testing.
```bash
bash navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh
```

#### nuScenes Evaluation
To evaluate the AutoVLA model on nuScenes validation data, you need to prepare the segmentation data for collision evaluation. You can download the preprocessed segmentation data from [this link](https://drive.google.com/file/d/1YMiKqUi_rhdxdrNVXKFuwEg3xAGm-dvU/view?usp=sharing), which we preprocessed using code from [UniAD](https://github.com/OpenDriveLab/UniAD).

Then run:
```bash
python tools/eval/nusc_eval.py \
    --config config/training/qwen2.5-vl-3B-nusc-sft.yaml \
    --checkpoint /path/to/checkpoint.ckpt \
    --seg_data_path /path/to/nusc_eval_seg
```


## Citation
If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.
 ```bibtex
@article{zhou2025autovla,
  title={AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning},
  author={Zhou, Zewei and Cai, Tianhui and Zhao, Seth Z.and Zhang, Yun and Huang, Zhiyu and Zhou, Bolei and Ma, Jiaqi},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
