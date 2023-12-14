import wandb
import numpy as np
from typing import Dict
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import io
from PIL import Image

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_delta(actions: torch.Tensor): # shape: [BS, pred_horizon, 2]
    padded_actions = F.pad(actions, (0, 0, 1, 0), mode="constant", value=0) # [BS, pred_horizon+1, 2]
    delta = padded_actions[:,1:] - padded_actions[:,:-1]

    return delta

def normalize_data(data: torch.Tensor, stats: list):
    data_min, data_max = stats

    # nomalize to [0,1]
    ndata = (data - data_min) / (data_max - data_min)

    # normalize to [-1, 1]
    ndata = ndata * 2 - 1

    return ndata

def unnormalize_data(ndata: torch.Tensor, stats: list):
    data_min, data_max = stats

    # unnormalize from [-1, 1]
    ndata = (ndata + 1) / 2

    # unnormalize from [0, 1]
    data = ndata * (data_max - data_min) + data_min

    return data

def count_parameters(model: nn.Module, print_table: bool = False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params = total_params + params
    if print_table:
        print(table)
    return total_params

# TODO: Test this function in train_eval_loop.py
def sample_actions(
        model: nn.Module,
        noise_scheduler: DDPMScheduler,
        obs_images: torch.Tensor,
        goal_vector: torch.Tensor,
        pred_horizon: int,
        goal_mask: torch.Tensor,
        action_dim: int,
        action_stats: Dict[str, np.ndarray],
        num_samples: int,
        device: torch.device,
):
    obs_cond = model(func_name="vision_encoder", obs_img=obs_images.unsqueeze(0), goal_mask=goal_mask.unsqueeze(0), goal_vec=goal_vector.unsqueeze(0)) # [1, enc_size]
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0) # [num_samples, enc_size]

    diffusion_output = torch.randn((num_samples, pred_horizon, action_dim), device=device)

    for i in noise_scheduler.timesteps[:]: # from num_train_timesteps-1 to 0
        # Predict the noise
        noise_pred = model(
            func_name="noise_pred_net",
            sample=diffusion_output,
            timestep=i.unsqueeze(-1).repeat(num_samples).to(device),
            global_cond=obs_cond,
        ) # [num_samples, pred_horizon, action_dim]

        # Inverse diffusion step (Remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=i,
            sample=diffusion_output,
        ).prev_sample # [num_samples, pred_horizon, action_dim]

    ndeltas = diffusion_output.reshape(num_samples, -1, 2)  # [num_samples, pred_horizon, 2]
    deltas = unnormalize_data(ndeltas, action_stats) # [num_samples, pred_horizon, 2]
    actions = torch.cumsum(deltas, dim=1) 

    return actions

def visualize_obs_action(
        obs_imgs: torch.Tensor, # [context_size*3, H, W]]
        sampled_actions: torch.Tensor,
        ground_truth_actions: torch.Tensor,
        ground_truth_yaws: torch.Tensor,
        goal_vec: torch.Tensor,
        goal_mask: torch.Tensor,
        dataset_name: str,
        metadata: Dict[str, str],
        device: int,
        use_wandb: bool = False,
        ):

    # Plot the ground truth actions
    ground_truth_actions = ground_truth_actions.detach().cpu().numpy() # [pred_horizon, 2]
    ground_truth_actions = np.concatenate([np.zeros((1,2)), ground_truth_actions], axis=0) # [pred_horizon+1, 2]
    gt_x = ground_truth_actions[:, 0]
    gt_y = ground_truth_actions[:, 1]

    # Plot the sampled actions
    sampled_actions = sampled_actions.detach().cpu().numpy() # [num_samples, pred_horizon, 2]
    sampled_actions = np.concatenate([np.zeros((sampled_actions.shape[0], 1, sampled_actions.shape[-1])), sampled_actions], axis=1) # [num_samples, pred_horizon+1, 2]

    # Plot the goal vector
    goal_vec = goal_vec.detach().cpu().numpy()

    # Goal mask
    goal_mask = goal_mask.detach().cpu().numpy()

    # Rotate the sampled actions and goal vector to local coordinates relative to the first observation
    ground_truth_yaws = ground_truth_yaws.detach().cpu().numpy() # [pred_horizon+1, ]

    # Calculate direction vectors from the yaw angles
    direction_vecs = np.stack([np.sin(ground_truth_yaws), np.cos(ground_truth_yaws)], axis=-1) # [pred_horizon+1, 2]


    # Plot the last observation image
    obs_imgs = torch.split(obs_imgs, 3, dim=0)
    obs_imgs = torch.stack(obs_imgs, dim=0) # [context_size, 3, H, W]
    obs_imgs = obs_imgs.permute(0, 2, 3, 1)
    obs_imgs = obs_imgs.detach().cpu().numpy()

    # Number of images
    num_imgs = obs_imgs.shape[0]

    # Plot
    fig = plt.figure(figsize=(10*num_imgs, 10))
    fig.suptitle(f"Dataset: {dataset_name}, Trajectory: {metadata[0]:06d}_{metadata[1]:06d}, Timestep: {metadata[2]:06d}")

    # Plot Observations
    for i in range(num_imgs):
        ax = fig.add_subplot(1, num_imgs, i+1)
        ax.set_title(f"Observation T-{num_imgs-1-i}")
        ax.imshow(obs_imgs[i]) 

    # Log the plot to W&B
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    if use_wandb and (device == torch.device("cuda") or device == 0):
        wandb.log({"Evaluation/Observations": wandb.Image(Image.open(buf))})

    # Actions plot
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Dataset: {dataset_name}, Trajectory: {metadata[0]:06d}_{metadata[1]:06d}, Timestep: {metadata[2]:06d}")

    # Plot the last observation image
    ax = fig.add_subplot(121)
    ax.set_title("Observations T")
    ax.imshow(obs_imgs[-1])
    
    # Plot the actions
    ax = fig.add_subplot(122)
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Actions")
    ax.axis("equal")

    # Plot all sampled trajectories
    for i in range(sampled_actions.shape[0]):
        x = sampled_actions[i, :, 0]
        y = sampled_actions[i, :, 1]
        ax.plot(x, y, "r-o", alpha=0.1, markersize=3)

    # Plot the ground truth actions
    ax.plot(gt_x, gt_y, "g-o", markersize=3)

    # plot the goal vector
    if goal_mask == 0:
        ax.plot(goal_vec[0], goal_vec[1], "b-x", markersize=10)
    else:
        ax.plot(goal_vec[0], goal_vec[1], "b-o", markersize=10)

    # Plot the yaw angles
    ax.quiver(
        ground_truth_actions[:,0], 
        ground_truth_actions[:,1], 
        direction_vecs[:,0],
        direction_vecs[:,1],
        color='g', 
        alpha=0.5,
        width=0.005,
        )

    # Log the plot to wandb
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    if use_wandb and (device == torch.device("cuda") or device == 0):
        wandb.log({"Evaluation/Actions": wandb.Image(Image.open(buf))})