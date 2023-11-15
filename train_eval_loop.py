from tqdm import tqdm
import wandb
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from utils import *


# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])


# TODO: Implement train_eval_loop
def train_eval_loop_vnav(
    model: nn.Module,
    optimizer: AdamW,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    use_wandb: bool = True,
    current_epoch: int = 0,
    eval_freq: int = 1,
):
    # Prepare the EMA(Exponential Moving Average) model
    ema_model = EMAModel(model=model, power=0.75)

    # Train Loop
    for epoch in range(current_epoch, current_epoch+epochs):
        train_vnav(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            transform=transform,
            device=device,
            noise_scheduler=noise_scheduler,
            project_folder=project_folder,
            epoch=epoch,
            use_wandb=use_wandb,
        )
        lr_scheduler.step()
        
        # Model save paths
        ema_numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        latest_path = os.path.join(project_folder, f"latest.pth")
        optimizer_latest_path = os.path.join(project_folder, f"optimizer_latest.pth")
        scheduler_latest_path = os.path.join(project_folder, f"scheduler_latest.pth")

        # Save the EMA model
        torch.save(ema_model.averaged_model.state_dict(), ema_numbered_path)

        # Save the model
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)

        # Save the optimizer
        torch.save(optimizer.state_dict(), optimizer_latest_path)

        # Save the scheduler
        torch.save(lr_scheduler.state_dict(), scheduler_latest_path)

        # Evaluation
        if (epoch+1)%eval_freq == 0:
            for test_dataset_key in test_dataloaders:
                evaluate_vnav(
                    model=model,
                    ema_model=ema_model,
                    noise_scheduler=noise_scheduler,
                    dataloader=test_dataloaders[test_dataset_key],
                    transform=transform,
                    device=device,
                    epoch=epoch,
                    project_folder=project_folder,
                    use_wandb=use_wandb,
                )

        # TODO: Log to wandb

# TODO: write the dataset class and test "train_vnav" function
def train_vnav(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: AdamW,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    epoch: int,
    use_wandb: bool = True,
):
    model.train()
    num_batches = len(dataloader)

    with tqdm(dataloader, desc="Train batch") as t_epoch:
        for i, data in enumerate(t_epoch):
            (obs_image, goal_vec, actions) = data

            # Batch size
            BS = actions.shape[0]

            # Observation images
            obs_images = torch.split(obs_image, 3, dim=1) # Split the image pack into individual images
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)

            # Goal vectors
            batch_goal_vecs = goal_vec.to(device)

            # Inference the observation-goal context
            obsgoal_context = model(
                "vision_encoder", 
                obs_img=batch_obs_images,
                goal_vec=batch_goal_vecs,
                )
            
            deltas = get_delta(actions)
            n_deltas = normalize_data(deltas, ACTION_STATS)
            n_action = from_numpy(n_deltas).to(device)

            # Sample noise to add to actions
            noise = torch.randn(n_action.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (BS,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(n_action, noise, timesteps)
            
            # Predict the noise residual
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, context=obsgoal_context)

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                return unreduced_loss.mean() / 1e-2

            # L2 loss
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
            # Total loss
            loss = diffusion_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

def evaluate_vnav():
    raise NotImplementedError