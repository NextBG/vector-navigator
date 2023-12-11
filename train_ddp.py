import os
import time
import numpy as np
import argparse
import wandb
import yaml

import torch

from torch.optim import AdamW
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from warmup_scheduler import GradualWarmupScheduler
from models import Vnav, VisionEncoder
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from train_eval_loop import train_eval_loop_vnav
from vnav_dataset import VnavDataset

from utils import *

'''
    Train the model
'''

# 2023-12-11: TODO: debug: DataLoader __del__ aborted due to timeout

def main(gpu_rank, config):
    # Initialize the process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=gpu_rank, world_size=config["num_gpus"])
    torch.cuda.set_device(gpu_rank)

    # wandb
    if config["use_wandb"] and gpu_rank == 0:
        wandb.login()
        wandb.init(
            project=config["project_name"],
        )
        wandb.run.name = config["run_name"]

        if wandb.run:
            wandb.config.update(config)

    # Seed
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True # Look for the optimal set of algorithms for that particular configuration

    # Normalization for image
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet
    ])
    transform = transforms.Compose(transform)

    train_datasets = []
    eval_datasets = []
    for dataset_name in config["datasets"]:
        for dataset_type in ["train", "eval"]:
            dataset = VnavDataset(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                datasets_folder=config["datasets_folder"],
                data_splits_folder=os.path.join(config["project_root_folder"], "data_splits"),
                image_size=config["image_size"],
                stride=config["stride"],
                pred_horizon=config["pred_horizon"],
                context_size=config["context_size"],
                min_goal_dist=config["min_goal_dist"],
                max_goal_dist=config["max_goal_dist"],
                max_traj_len=config["max_traj_len"],
            )
            # Train datasets
            if dataset_type == "train":
                train_datasets.append(dataset)
            # Test datasets
            elif dataset_type == "eval":
                eval_datasets.append(dataset)

    # Train dataloader
    train_dataset = ConcatDataset(train_datasets)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=DistributedSampler(train_dataset),
        num_workers=config["num_workers"],
        persistent_workers=True,
    )

    # Eval dataloader
    eval_dataset = ConcatDataset(eval_datasets)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["batch_size"],
        sampler=DistributedSampler(eval_dataset),
        num_workers=config["num_workers"],
        persistent_workers=True,
    )
    
    # Vision encoder
    visual_enc_net = VisionEncoder(
        context_size=config["context_size"],
        obs_encoding_size=config["encoding_size"],
    )

    # Noise prediction network
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        n_groups=config["pred_horizon"],
    )

    # Full Model
    model = Vnav(
        vision_encoder=visual_enc_net,
        noise_pred_net=noise_pred_net
    ).to(gpu_rank)

    # Distributed Data Parallel
    ddp_model = DDP(model, device_ids=[gpu_rank], find_unused_parameters=True)

    # Noise Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"], # Diffusion iterations
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Optimizer
    optimizer = AdamW(
        ddp_model.parameters(), 
        lr=float(config["lr"])
    )

    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config["epochs"],
    )

    # Warmup Scheduler
    lr_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=config["warmup_epochs"],
        after_scheduler=lr_scheduler
    )

    # Load the checkpoint if necessary
    if "load_checkpoint" in config:
        checkpoint_folder = os.path.join("logs", config["project_name"], config["load_checkpoint"]) 
        print(f"Loading checkpoint from {checkpoint_folder}")
        latest_path = os.path.join(checkpoint_folder, f"latest.pth")
        latest_checkpoint = torch.load(latest_path)
        
        # Load the state dict
        ddp_model.load_state_dict(latest_checkpoint)

    # Start training
    if gpu_rank == 0:
        print("Start training!")
    
    # Action stats
    action_stats = [torch.tensor(config["action_stats"]["min"], device=gpu_rank),
                    torch.tensor(config["action_stats"]["max"], device=gpu_rank)]
    
    train_eval_loop_vnav(
        model=ddp_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_dataloader, 
        eval_dataloader=eval_dataloader,
        transform=transform,
        action_stats=action_stats,
        epochs=config["epochs"],
        device=gpu_rank,
        project_folder=config["project_log_folder"],
        use_wandb=config["use_wandb"],
        current_epoch=0,
        eval_interval=config["eval_interval"],
    )

    # Clean up
    dist.destroy_process_group()

    if gpu_rank == 0:
        print("Training finished!")

if __name__ == "__main__":
    # Get path of current folder
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load the config
    with open(os.path.join(PROJECT_ROOT_DIR, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    config["project_root_folder"] = PROJECT_ROOT_DIR

    # Create project folder
    config["run_name"] = config["run_name"] + "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_log_folder"] = os.path.join("logs", config["project_name"], config["run_name"])
    os.makedirs(config["project_log_folder"])

    # Distributed training
    world_size = torch.cuda.device_count()
    config["num_gpus"] = world_size

    # Run the trainning code
    mp.spawn(
        main,
        args=(config, ),
        nprocs=world_size,
    )