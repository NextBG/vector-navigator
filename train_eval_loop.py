from tqdm import tqdm
import wandb
import os
from typing import Dict
import itertools
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from utils import *

def train_eval_loop_vnav(
    model: nn.Module,
    optimizer: AdamW,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_dataloader: DataLoader,
    eval_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    action_stats: np.ndarray,
    epochs: int,
    device,
    project_folder: str,
    current_epoch: int = 0,
    eval_interval: int = 1,
    use_wandb: bool = True,
):

    # Prepare the EMA(Exponential Moving Average) model
    ema_model = EMAModel(model=model, power=0.75)

    # Train Loop
    for epoch in range(current_epoch, current_epoch+epochs):
        print(f">>> Epoch {epoch}/{current_epoch+epochs} <<<")

        train_vnav(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            transform=transform,
            action_stats=action_stats,
            device=device,
            noise_scheduler=noise_scheduler,
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
        if (epoch+1)%eval_interval == 0:
            for eval_dataset_key in eval_dataloaders:
                evaluate_vnav(
                    ema_model=ema_model,
                    dataloader=eval_dataloaders[eval_dataset_key],
                    transform=transform,
                    action_stats=action_stats,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    project_folder=project_folder,
                    epoch=epoch,
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
    action_stats: np.ndarray,
    device,
    noise_scheduler: DDPMScheduler,
    epoch: int,
    use_wandb: bool = False,
):
    model.train()

    # check if using parallel training
    if type(device) == torch.device:
        use_ddp = False
    else:
        dataloader.sampler.set_epoch(epoch)
        use_ddp = True

    # TODO: 23/12/04: Improve the performance of the training loop, maybe the dataloader is the bottleneck
    for i, data in tqdm.tqdm(enumerate(dataloader), desc="Training batches", total=len(dataloader), disable=(use_ddp and device != 0)):
        # Print the load time
        (obs_image, goal_vec, actions) = data

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
        
        deltas = get_delta(actions) # Get delta between action and last action
        n_deltas = normalize_data(deltas, action_stats) # Normalize the deltas
        n_action = n_deltas.to(device)

        # Sample noise to add to actions
        noise = torch.randn(n_action.shape, device=device)

        # Sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],), device=device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each diffusion iteration
        noisy_action = noise_scheduler.add_noise(n_action, noise, timesteps)

        # Predict the noise residual
        noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_context)

        # Total loss = MSE between predicted noise and noise
        diffusion_loss = F.mse_loss(noise_pred, noise)

        # Optimize
        optimizer.zero_grad()
        diffusion_loss.backward()
        optimizer.step()

        # Update Exponential Moving Average of the model weights
        ema_model.step(model)

# Evaluate the model
def evaluate_vnav(
        ema_model: EMAModel,
        dataloader: DataLoader,
        transform: transforms,
        action_stats: np.ndarray,
        device,
        noise_scheduler: DDPMScheduler,
        epoch: int,
        project_folder: str,
        eval_fraction: float= 0.1,
        visualization_interval: int = 3,
        use_wandb: bool = False,
):
    # check if using parallel training
    if type(device) == torch.device:
        use_ddp = False
    else:
        dataloader.sampler.set_epoch(epoch)
        use_ddp = True

    ema_model = ema_model.averaged_model
    ema_model.eval()

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    with torch.no_grad():
        for i, data in tqdm.tqdm(
            enumerate(itertools.islice(dataloader, num_batches)), 
            desc="Evaluation batches", 
            total=num_batches,
            disable=(use_ddp and device != 0),
            ):

            (obs_image, goal_vec, actions) = data
            actions = actions.to(device)

            # Batch size
            BS = actions.shape[0]

            # Observation images
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)

            # Goal vectors
            batch_goal_vecs = goal_vec.to(device)

            # Inference the observation-goal context
            obsgoal_context = ema_model(
                "vision_encoder", 
                obs_img=batch_obs_images,
                goal_vec=batch_goal_vecs,
                )
            
            # Actions
            deltas = get_delta(actions)
            n_deltas = normalize_data(deltas, action_stats)

            # Sample noise to add to actions
            noise = torch.randn(n_deltas.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (BS,), device=device
            ).long()

            # Add noise to actions 
            noisy_deltas = noise_scheduler.add_noise(n_deltas, noise, timesteps)

            # Predict the noise residual
            noise_pred = ema_model("noise_pred_net", sample=noisy_deltas, timestep=timesteps, global_cond=obsgoal_context)
            
            # Total loss = MSE between predicted noise and noise
            diffusion_loss = F.mse_loss(noise_pred, noise)

            # log the loss to wandb
            if use_wandb:
                wandb.log({"Diffusion loss": diffusion_loss.item()})

            sampled_actions = sample_actions(
                ema_model,
                noise_scheduler,
                batch_obs_images[0],
                batch_goal_vecs[0],
                pred_horizon=len(actions[0]),
                action_dim=2,
                action_stats=action_stats,
                num_samples=10,
                device=device,
            )

            if i%visualization_interval == 0:
                visualize_obs_action( # TODO: add the ground truth to the plot
                    batch_idx=i,
                    obs_img=obs_images[0][-1],
                    sampled_actions=sampled_actions,
                    ground_truth_actions=actions[0],
                    goal_vec=goal_vec[0],
                    epoch=epoch,
                    project_folder=project_folder,
                    use_wandb=use_wandb,
                )
