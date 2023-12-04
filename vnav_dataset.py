import io
import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import time

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class VnavDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_type: str,
        datasets_folder: str,
        data_splits_folder: str,
        image_size: Tuple[int, int],
        stride: int,
        pred_horizon: int,
        context_size: int,
        max_goal_dist: int,
    ):
        """
        Main Vec Nav dataset class
        """
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(datasets_folder, dataset_name)
        traj_names_file = os.path.join(data_splits_folder, dataset_name, dataset_type, "traj_names.txt")

        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.stride = stride
        self.pred_horizon = pred_horizon
        self.context_size = context_size
        self.max_goal_dist = max_goal_dist
        self.index_to_data = self._build_index()

    def _build_index(self):
        samples_index = []

        for traj_name in self.traj_names:
            traj_data = self._get_trajectory(traj_name)
            # Skip if the trajectory doesn't exist
            if traj_data is None:
                continue
            traj_len = len(traj_data["positions"])

            begin_time = self.context_size * self.stride
            end_time = traj_len - self.pred_horizon * self.stride
            for curr_time in range(begin_time, end_time):
                max_goal_dist = min(self.max_goal_dist * self.stride, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_dist))

        return samples_index

    def _load_image(self, trajectory_name, time):
        image_path = os.path.join(self.dataset_folder, "trajectories", trajectory_name, "frames", f"{time:06d}.png")
        with open(image_path, "rb") as f:
            image_bytes = f.read() #<class 'bytes'>
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize(self.image_size) # Resize the image
        image_tensor = TF.to_tensor(image) # Tensor of shape (W, H, 3)
        return image_tensor # Tensor of shape (3, W, H)

    def _to_local_coords(self, points, origin, yaw0):
        """
        points: (N, 2) array
        origin: (2,) array
        yaw: scalar
        """
        R = np.array([
            [np.cos(-yaw0), -np.sin(-yaw0)],
            [np.sin(-yaw0), np.cos(-yaw0)],
        ])
        points = points - origin
        points = points @ R
        return points

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.pred_horizon * self.stride + 1

        positions = traj_data["positions"][start_index:end_index:self.stride]
        yaws = traj_data["yaws"][start_index:end_index:self.stride]
        goal_pos = traj_data["positions"][min(goal_time, len(traj_data["positions"]) - 1)]


        waypoints = self._to_local_coords(positions, positions[0], yaws[0])
        goal_pos = self._to_local_coords(goal_pos, positions[0], yaws[0])
        yaws = yaws - yaws[0]

        yaw_actions = yaws[1:]
        actions = waypoints[1:]
        
        # Add the yaw actions to the actions
        actions = np.concatenate([actions, yaw_actions[:, None]], axis=1)

        return actions, goal_pos
    
    def _get_trajectory(self, trajectory_name):
        # Return none if the trajectory doesn't exist
        if not os.path.exists(os.path.join(self.dataset_folder, "trajectories", trajectory_name, "traj_processed.pkl")):
            return None
        with open(os.path.join(self.dataset_folder, "trajectories", trajectory_name, "traj_processed.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        return traj_data

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        current_file, curr_time, max_goal_dist = self.index_to_data[i]

        # Sample goal
        goal_offset = np.random.randint(0, (max_goal_dist + 1)//self.stride)
        goal_time = curr_time + goal_offset * self.stride

        # Load context images
        context = []

        # sample the last self.context_size times from interval [0, curr_time)
        context_times = list(
            range(
                curr_time - self.context_size*self.stride,
                curr_time + 1,
                self.stride
                )
            )
        context = [(current_file, t) for t in context_times]
        obs_context = torch.cat([self._load_image(current_file, t) for current_file, t in context])

        # Get actions and goal vector
        curr_traj_data = self._get_trajectory(current_file)
        actions, goal_vec = self._compute_actions(curr_traj_data, curr_time, goal_time)
        
        return (
            torch.as_tensor(obs_context, dtype=torch.float32),
            torch.as_tensor(goal_vec, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.index_to_data)
    

# Test the dataset
if __name__ == "__main__":
    vnav_dataset = VnavDataset(
        dataset_name="nadawalk_tokyo",
        dataset_type="train",
        datasets_folder="/home/caoruixiang/vecnav_dataset/mount_point/datasets",
        data_splits_folder="/home/caoruixiang/nomad_caorx/vector-navigator/data_splits",
        image_size=(128, 72),
        stride=5,
        pred_horizon=20,
        context_size=5,
        min_goal_dist=5,
        max_goal_dist=20,
    )
    # print(len(vnav_dataset))

    for i in range(100):
        # Test: take a sample from the dataset
        sample = vnav_dataset[np.random.randint(0, len(vnav_dataset))]
        # sample = vnav_dataset[300]

        # slice the image list into individual images
        obs_images = sample[0]
        obs_images = torch.split(obs_images, 3, dim=0)
        sample_image = obs_images[0]

        import matplotlib.pyplot as plt
        # Visualize the image in subplot 121
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].imshow(sample_image.permute(1, 2, 0))

        # Visualze the goal vector
        goal_vec = sample[1]

        # Visualize the actions
        actions = sample[2]
        actions = np.concatenate([np.array([[0, 0, 0]]), actions], axis=0)

        axs[1].set_title('Trajectory')
        axs[1].set_xlabel('x(m)')
        axs[1].set_ylabel('z(m)')
        axs[1].axis('equal')

        # Visualize the trajectory
        axs[1].plot(actions[:, 0], actions[:, 1], 'ro', markersize=2)
        axs[1].plot(actions[:, 0], actions[:, 1], 'r-', markersize=0.5)
        
        # Plot end point
        axs[1].plot(actions[:, 0][-1], actions[:, 1][-1], 'rx', markersize=15)

        # Plot the goal vector
        axs[1].plot(goal_vec[0], goal_vec[1], 'bx', markersize=15)
        axs[1].plot(
            actions[0][0], 
            actions[0][1], 
            goal_vec[0],
            goal_vec[1],
            'b-'
            )

        # Plot the yaw angle
        vis_stride = 2
        axs[1].quiver(
            actions[:,0][::vis_stride], 
            actions[:,1][::vis_stride], 
            np.sin(actions[:,2][::vis_stride]), 
            np.cos(actions[:,2][::vis_stride]), 
            color='g', 
            width=0.005)

        plt.show()