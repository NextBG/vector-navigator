from tqdm import tqdm
import wandb
import os
import time
import pickle

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
    eval_dataloader: DataLoader,
    transform: transforms,
    action_stats: list,
    goal_mask_prob: float,
    epochs: int,
    device: int,
    log_folder: str,
    start_epoch: int = 0,
    eval_interval: int = 1,
    use_wandb: bool = True,
):


    # Prepare the EMA(Exponential Moving Average) model
    ema_model = EMAModel(model=model, power=0.75)

    # Train Loop
    for epoch in range(start_epoch, start_epoch+epochs):
        if device ==  torch.device("cuda") or device == 0:
            print(f">>> Epoch {epoch}/{start_epoch+epochs-1} <<<")

        # learning rate log
        if use_wandb:
            if device ==  torch.device("cuda") or device == 0:
                wandb.log({"Train/Learning rate": lr_scheduler.get_last_lr()[0]})

        train_vnav(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            transform=transform,
            action_stats=action_stats,
            goal_mask_prob=goal_mask_prob,
            device=device,
            noise_scheduler=noise_scheduler,
            epoch=epoch,
            use_wandb=use_wandb,
        )

        # learning rate scheduler step
        if epoch < start_epoch+epochs-1:
            lr_scheduler.step()

        # Model save paths
        ema_numbered_path = os.path.join(log_folder, f"ema_{epoch}.pth")
        numbered_path = os.path.join(log_folder, f"{epoch}.pth")
        latest_path = os.path.join(log_folder, f"latest.pth")
        optimizer_latest_path = os.path.join(log_folder, f"optimizer_latest.pth")
        scheduler_latest_path = os.path.join(log_folder, f"scheduler_latest.pth")
        
        # Save the EMA model
        torch.save(ema_model.averaged_model.state_dict(), ema_numbered_path)
        
        # Save the model
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        
        # Save the optimizer
        torch.save(optimizer.state_dict(), optimizer_latest_path)
        
        # Save the scheduler
        torch.save(lr_scheduler.state_dict(), scheduler_latest_path)

        # Save the metadata
        metadata = {
            "current_epoch": start_epoch+epoch+1,
        }
        metadata_path = os.path.join(log_folder, f"metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        # Evaluation
        if (epoch+1)%eval_interval == 0:
            evaluate_vnav(
                ema_model=ema_model,
                dataloader=eval_dataloader,
                transform=transform,
                action_stats=action_stats,
                goal_mask_prob=goal_mask_prob,
                device=device,
                noise_scheduler=noise_scheduler,
                epoch=epoch,
                use_wandb=use_wandb,
            )

def train_vnav(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: AdamW,
    dataloader: DataLoader,
    transform: transforms,
    action_stats: list,
    goal_mask_prob: float,
    device,
    noise_scheduler: DDPMScheduler,
    epoch: int,
    use_wandb: bool = False,
):
    model.train()

    # check if using parallel training
    if type(device) == int:
        dataloader.sampler.set_epoch(epoch)

    # Timer for the training loop
    last_time = time.time()

    # TODO: 23/12/04: Improve the performance of the training loop, maybe the dataloader is the bottleneck
    for i, data in tqdm(
        enumerate(dataloader), 
        desc="Training batches", 
        total=len(dataloader), 
        disable=(device !=  torch.device("cuda") and device != 0),
        bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        ):
        # Load data
        (obs_image, goal_vec, actions, yaws, dataset_name, metadata) = data
        obs_image = obs_image.to(device)
        goal_vec = goal_vec.to(device)
        actions = actions.to(device)

        # Observation images
        obs_images = torch.split(obs_image, 3, dim=1) # Split the image pack into individual images
        batch_obs_images = [transform(obs) for obs in obs_images]
        batch_obs_images = torch.cat(batch_obs_images, dim=1)

        # Generate goal mask
        BS = actions.shape[0]
        goal_mask = (torch.rand((BS,)) < goal_mask_prob).float().to(device)

        # Inference the observation-goal context
        obsgoal_context = model(
            "vision_encoder", 
            obs_img=batch_obs_images,
            goal_mask=goal_mask,
            goal_vec=goal_vec,
            )
        
        deltas = get_delta(actions) # Get delta between action and last action
        n_deltas = normalize_data(deltas, action_stats) # Normalize the deltas

        # Sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],), device=device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each diffusion iteration
        noise = torch.randn(n_deltas.shape, device=device)
        noisy_action = noise_scheduler.add_noise(n_deltas, noise, timesteps)

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

        # Wandb logging
        if use_wandb and (device == 0 or device == torch.device("cuda")):
            wandb.log({"Train/Diffusion loss": diffusion_loss.item()}) # Training loss
            wandb.log({"Train/Speed (iter per second)": 1/(time.time()-last_time)}) # Training speed
            last_time = time.time()


def evaluate_vnav(
        ema_model: EMAModel,
        dataloader: DataLoader,
        transform: transforms,
        action_stats: list,
        goal_mask_prob: float,
        device: int,
        noise_scheduler: DDPMScheduler,
        epoch: int,
        eval_fraction: float= 0.1,
        visualization_interval: int = 3,
        use_wandb: bool = False,
):
    # check if using parallel training
    if type(device) == int:
        dataloader.sampler.set_epoch(epoch)

    ema_model = ema_model.averaged_model
    ema_model.eval()

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    with torch.no_grad():
        for i, data in tqdm(
            enumerate(dataloader), 
            desc="Evaluation batches", 
            total= num_batches,
            disable=(device !=  torch.device("cuda") and device != 0),
            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ):

            # only evaluate on a fraction of the dataset
            if i >= num_batches:
                break

            (obs_imgs, goal_vec, actions, yaws, dataset_name, metadata) = data
            obs_imgs = obs_imgs.to(device) # [BS, context_size*3, H, W]
            goal_vec = goal_vec.to(device)
            actions = actions.to(device)
 
            # Apply the transform to the observation images
            splited_obs_imgs = torch.split(obs_imgs, 3, dim=1) # Split the image pack into individual images
            batch_obs_images = [transform(obs) for obs in splited_obs_imgs]
            batch_obs_images = torch.cat(batch_obs_images, dim=1) # [BS, context_size*3, H, W]

            # Generate goal mask
            BS = actions.shape[0]
            goal_mask = (torch.rand((BS,)) < goal_mask_prob).float().to(device)

            # Inference the observation-goal context
            obsgoal_context = ema_model(
                "vision_encoder", 
                obs_img=batch_obs_images,
                goal_mask=goal_mask,
                goal_vec=goal_vec,
                )
            
            # Actions
            deltas = get_delta(actions)
            n_deltas = normalize_data(deltas, action_stats)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (BS,), device=device
            ).long()

            # Add noise to actions 
            noise = torch.randn(n_deltas.shape, device=device)
            noisy_deltas = noise_scheduler.add_noise(n_deltas, noise, timesteps)

            # Predict the noise residual
            noise_pred = ema_model("noise_pred_net", sample=noisy_deltas, timestep=timesteps, global_cond=obsgoal_context)
            
            # Total loss = MSE between predicted noise and noise
            diffusion_loss = F.mse_loss(noise_pred, noise)

            # log the loss to wandb
            if use_wandb and (device == 0 or device == torch.device("cuda")):
                wandb.log({"Evaluation/Diffusion loss": diffusion_loss.item()})

            sampled_actions = sample_actions(
                ema_model,
                noise_scheduler,
                batch_obs_images[0],
                goal_vec[0],
                pred_horizon=len(actions[0]),
                goal_mask=goal_mask[0],
                action_dim=2,
                action_stats=action_stats,
                num_samples=10,
                device=device,
            )

            if i%visualization_interval == 0:
                visualize_obs_action( # TODO: add the ground truth to the plot
                    obs_imgs=obs_imgs[0],
                    sampled_actions=sampled_actions,
                    ground_truth_actions=actions[0],
                    ground_truth_yaws=yaws[0],
                    goal_vec=goal_vec[0],
                    goal_mask=goal_mask[0],
                    dataset_name=dataset_name[0],
                    metadata=metadata[0],
                    device=device,
                    use_wandb=use_wandb,
                )