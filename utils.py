import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.visualize_utils import to_numpy
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


# Get delta between action and last action
def get_delta(actions: torch.Tensor):
    device = actions.device
    actions = actions.detach().cpu().numpy()

    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]

    return torch.from_numpy(delta).float().to(device)

def normalize_data(data: torch.Tensor, stats: np.ndarray):
    device = data.device
    data = data.detach().cpu().numpy()

    x_min, x_max, y_min, y_max = stats

    # nomalize to [0,1]
    ndata = (data - np.array([x_min, y_min])) / (np.array([x_max, y_max]) - np.array([x_min, y_min]))
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1

    return torch.from_numpy(ndata).float().to(device)

def unnormalize_data(ndata: torch.Tensor, stats: np.ndarray):
    device = ndata.device
    ndata = ndata.detach().cpu().numpy()

    x_min, x_max, y_min, y_max = stats

    ndata = (ndata + 1) / 2
    data = ndata * (np.array([x_max, y_max]) - np.array([x_min, y_min])) + np.array([x_min, y_min])

    return torch.from_numpy(data).to(device)

def count_parameters(model: nn.Module):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params

# TODO: Test this function in train_eval_loop.py
def sample_actions(
        model: nn.Module,
        noise_scheduler: DDPMScheduler,
        obs_images: torch.Tensor,
        goal_vector: torch.Tensor,
        pred_horizon: int,
        action_dim: int,
        action_stats: Dict[str, np.ndarray],
        num_samples: int,
        device: torch.device,
):
    obs_cond = model(func_name="vision_encoder", obs_img=obs_images.unsqueeze(0), goal_vec=goal_vector.unsqueeze(0)) # [1, enc_size]
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
        batch_idx: int,
        obs_img: torch.Tensor,
        sampled_actions: torch.Tensor,
        ground_truth_actions: torch.Tensor,
        goal_vec: torch.Tensor,
        epoch:int,
        project_folder: str,
        use_wandb: bool = False,
        ):
    # Create a folder to save the visualizations
    visualize_path = os.path.join(
        project_folder,
        "visualize",
        f"epoch_{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    ground_truth_actions = ground_truth_actions.detach().cpu().numpy()
    ground_truth_actions = ground_truth_actions.reshape(-1, 2)
    ground_truth_actions = np.concatenate([np.zeros((1,2)), ground_truth_actions], axis=0) # [pred_horizon+1, 2]
    gt_x = ground_truth_actions[:, 0]
    gt_y = ground_truth_actions[:, 1]

    sampled_actions = sampled_actions.detach().cpu().numpy() # [num_samples, pred_horizon, 2]
    sampled_actions = np.concatenate([np.zeros((sampled_actions.shape[0], 1, sampled_actions.shape[-1])), sampled_actions], axis=1) # [num_samples, pred_horizon+1, 2]

    goal_vec = goal_vec.detach().cpu().numpy()

    fig = plt.figure()
    # Plot the first observation image
    obs_img = obs_img.detach().cpu().numpy()
    obs_img = np.transpose(obs_img, (1, 2, 0))
    ax = fig.add_subplot(121)
    ax.set_title("Observation image")
    ax.imshow(obs_img)

    # plot all sampled action trajectories in a single plot
    ax = fig.add_subplot(122)
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Actions")

    for i in range(sampled_actions.shape[0]):
            x = sampled_actions[i, :, 0]
            y = sampled_actions[i, :, 1]
            ax.plot(x, y, "r-o", alpha=0.1, markersize=3)

    # Plot the ground truth actions
    ax.plot(gt_x, gt_y, "g-o", markersize=5)

    # plot the goal vector
    ax.plot(goal_vec[0], goal_vec[1], "b-x", markersize=10)

    # Save the plot
    save_path = os.path.join(visualize_path, f"{batch_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)

    # Log the plot to W&B
    if use_wandb:
        wandb.log({"Eval actions": wandb.Image(save_path)})