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
from models import VectorNavigator, VisionEncoder
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from train_eval_loop import train_eval_loop_vnav
from vnav_dataset import VnavDataset

'''
    Train the model
'''
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Dataset
    train_datasets = []
    test_datasets = []
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        dataset = VnavDataset(
            # TODO: Dataset parameters
        )
        # Train datasets
        if "train" in data_config:
            train_datasets.append(dataset)
        # Test datasets
        elif "test" in data_config:
            test_dataset_key = f"{dataset_name}_test"
            if test_dataset_key not in test_datasets:
                test_datasets[test_dataset_key] = {}
            test_datasets[test_dataset_key] = dataset
    
    # Train dataloader
    train_dataset = ConcatDataset(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        persistent_workers=True,
    )

    # Test dataloader
    test_dataloaders = {}
    for test_dataset_key, test_dataset in test_datasets.items():
        test_dataloaders[test_dataset_key] = DataLoader(
            test_dataset,
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
        input_dim=3,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
    )

    # Full Model
    model = VectorNavigator(
        vision_enc_net=visual_enc_net,
        noise_pred_net=noise_pred_net
    )

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
    
    # TODO: load the checkpoint if necessary

    # Start training
    current_epoch = 0
    train_eval_loop_vnav(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_dataloader, 
        test_dataloaders=test_dataloaders,
        transform=transform,
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        use_wandb=config["use_wandb"],
        current_epoch=current_epoch,
        eval_freq=config["eval_freq"],
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # Project config
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        type=str,
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create project folder
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join("logs", config["project_name"], config["run_name"])
    os.makedirs(config["project_folder"])

    # TODO: Wandb initialization

    # Run the trainning code
    main(config)