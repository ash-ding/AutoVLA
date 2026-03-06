import os
import sys
import yaml
import torch
import argparse
import functools
from peft import get_peft_model, LoraConfig, TaskType

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import FSDPStrategy

from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from models.autovla import GRPOAutoVLA
from dataset_utils.rft_dataset import RFTDataset
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
import datetime
import warnings

warnings.filterwarnings("ignore", message=".*weights_only=False.*")


torch.set_float32_matmul_precision('high')

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class GroupSampler(DistributedSampler):
    """
    A sampler for distributed training that returns the same indices on every device.

    This sampler differs from the default DistributedSampler in that every process
    gets the complete set of indices (optionally shuffled deterministically) rather than a subset of them. 

    If the distributed process group is not initialized, the sampler falls back to a
    single-process mode (num_replicas=1, rank=0) to avoid errors.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if not dist.is_initialized():
            num_replicas = 1
            rank = 0
        else:
            num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
            rank = rank if rank is not None else dist.get_rank()

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.num_samples = len(dataset)
        self.total_size = len(dataset)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        return iter(indices)

    def __len__(self):
        return self.total_size


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local-rank", type=int, default=0)

    args = parser.parse_args()
    seed_everything(args.seed)

    # Load configuration
    config = load_config(f"./config/{args.config}.yaml")

    # Dataset and dataloader
    train_dataset = RFTDataset(config['data']['train'], config['model'])
    val_dataset = RFTDataset(config['data']['val'], config['model'])

    train_data = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        sampler=GroupSampler(train_dataset, shuffle=True),
        collate_fn=train_dataset.collate_fn,
    )

    val_data = DataLoader(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        num_workers=config['inference']['num_workers'],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )    

    # Model
    model = GRPOAutoVLA(config)
    # model.load_state_dict(
    #     torch.load(config['sft_model_path'])['state_dict'], 
    #     strict=False
    # )

    # TODO: remove this hard coding
    print(f"Loading and remapping checkpoint from: {config['model']['sft_model_path']}")
    full_checkpoint = torch.load(config['model']['sft_model_path'], map_location="cpu")
    sd = full_checkpoint['state_dict']
    
    # Load the state dict
    msg = model.load_state_dict(sd, strict=False)

    # Create a LoRA configuration. Adjust the parameters (r, lora_alpha, lora_dropout) as needed.
    if config['model']['lora'].get("use", False):
        print("Using LoRA mode for GRPO training.")
        lora_conf = config['model']['lora']
        lora_config = LoraConfig(
            task_type=TaskType[lora_conf.get("task_type", "CAUSAL_LM")],
            target_modules=lora_conf.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            r=lora_conf.get("r", 8),
            lora_alpha=lora_conf.get("alpha", 8),
            lora_dropout=lora_conf.get("dropout", 0.1),
            bias=lora_conf.get("bias", "none")
        )
        model.autovla.vlm = get_peft_model(model.autovla.vlm, lora_config)
        print("LoRA-enabled model trainable parameters:",
              sum(p.numel() for p in model.autovla.vlm.parameters() if p.requires_grad))
    model = model.to(torch.bfloat16)

    # Training
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2_5_VLDecoderLayer
        },
    )

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"runs/grpo/{current_date}"
    
    trainer = Trainer(
        num_nodes=1,
        max_epochs=config['training']['epochs'],
        max_steps=config['training'].get('max_steps', -1),
        accelerator="gpu",
        devices=config['training']['devices'], 
        num_sanity_val_steps=0,
        strategy=FSDPStrategy(
            auto_wrap_policy=wrap_policy,
            cpu_offload=False,
            # Mixed precision training
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            ),
            # sharding strategy
            sharding_strategy='FULL_SHARD',
            # prefetching backward computation
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
            # save state dict type
            state_dict_type='full', # can be full or sharded
        ),
        callbacks=[
            ModelCheckpoint(
                monitor="avg_train_reward",
                mode="max",
                save_top_k=-1,
                dirpath=f"{save_dir}",
                filename="rft-step{step}-reward{avg_train_reward:.4f}",
                auto_insert_metric_name=False,
                save_weights_only=True,
                every_n_train_steps=500,
                save_on_train_epoch_end=False
            ),
            LearningRateMonitor(logging_interval="step")
        ],
        logger=[CSVLogger(save_dir="runs/"), TensorBoardLogger(save_dir="runs/")],
        enable_model_summary=True,
        log_every_n_steps=1,
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        limit_val_batches=0
    )

    trainer.fit(model, train_dataloaders=train_data)
