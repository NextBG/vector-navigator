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

from warmup_scheduler import GradualWarmupScheduler
from models import Vnav, VisionEncoder
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from train_eval_loop import train_eval_loop_vnav
from vnav_dataset import VnavDataset

'''
    Train the model
'''

# Dataset
PROJECT_ROOT_DIR = "/home/caoruixiang/nomad_caorx/vector-navigator"
DATASETS_DIR = "/home/caoruixiang/vecnav_dataset/mount_point/datasets"

def main(config):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    test_datasets = {}
    for dataset_name in config["datasets"]:
        for dataset_type in ["train", "eval"]:
            dataset = VnavDataset(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                datasets_folder=f"{DATASETS_DIR}",
                data_splits_folder=f"{PROJECT_ROOT_DIR}/data_splits",
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
                test_dataset_key = f"{dataset_name}_eval"
                if test_dataset_key not in test_datasets:
                    test_datasets[test_dataset_key] = {}
                test_datasets[test_dataset_key] = dataset
    
    # Train dataloader
    train_dataset = ConcatDataset(train_datasets)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        persistent_workers=True,
    )

    # Eval dataloader
    eval_dataloaders = {}
    for eval_dataset_key, eval_dataset in test_datasets.items():
        eval_dataloaders[eval_dataset_key] = DataLoader(
            eval_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
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
    ).to(device)

    # Noise Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"], # Diffusion iterations
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(), 
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
        model.load_state_dict(latest_checkpoint)

    # Start training
    print("Start training!")
    train_eval_loop_vnav(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_dataloader, 
        eval_dataloaders=eval_dataloaders,
        transform=transform,
        action_stats=config["action_stats"],
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        use_wandb=config["use_wandb"],
        current_epoch=0,
        eval_interval=config["eval_interval"],
    )

    print("Training finished!")

if __name__ == "__main__":
    # Load the config
    with open(os.path.join(PROJECT_ROOT_DIR, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Create project folder
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join("logs", config["project_name"], config["run_name"])
    os.makedirs(config["project_folder"])

    # TODO: Wandb initialization
    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
        )
        wandb.run.name = config["run_name"]

        if wandb.run:
            wandb.config.update(config)

    # Run the trainning code
    main(config)