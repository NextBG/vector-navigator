import io
import numpy as np
import os
import pickle
from typing import Tuple
from PIL import Image
import lmdb
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# TODO: 2023-12-11: visualize the timestamps of the sample

class VnavDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_type: str,
        datasets_folder: str,
        image_size: Tuple[int, int],
        stride: int,
        pred_horizon: int,
        context_size: int,
        min_goal_dist: int,
        max_goal_dist: int,
        max_traj_len: int = 200, # -1 means use all trajectories
    ):
        """
        Main Vec Nav dataset class
        """
        self.dataset_name = dataset_name
        self.dataset_folder = os.path.join(datasets_folder, dataset_name)
        traj_names_file = os.path.join(self.dataset_folder, "partitions", f"{dataset_type}.txt")

        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.stride = stride
        self.pred_horizon = pred_horizon
        self.context_size = context_size
        self.min_goal_dist = min_goal_dist
        self.max_goal_dist = max_goal_dist
        self.max_traj_len = max_traj_len

        self.index_to_data = self._build_index()

        self.cache_file = os.path.join(self.dataset_folder, f"cache_{image_size[0]}x{image_size[1]}.lmdb")
        self.lmdb_env = None

    def _build_index(self, yaw_th_deg=10):
        samples_index = []
        traj_len_to_use = min(self.max_traj_len, len(self.traj_names))
        for traj_name in self.traj_names[:traj_len_to_use]:
            traj_data = self._get_trajectory(traj_name)
            # Skip if the trajectory doesn't exist
            if traj_data is None:
                print(f"Trajectory {traj_name} doesn't exist, skipping...")
                continue

            traj_len = len(traj_data["positions"])

            begin_time = self.context_size * self.stride
            end_time = traj_len - max(self.pred_horizon, self.min_goal_dist) * self.stride

            for curr_time in range(begin_time, end_time):
                # Check if the camera direction is alone with the direction vector at T=0, If not, skip the trajectory
                # yaw at current time (T=0)
                yaw_t0 = traj_data["yaws"][curr_time] 
                # Convert to degrees
                yaw_t0_deg = yaw_t0 * 180 / np.pi

                # Get the direction vector by pos(T=1) - pos(T=0)
                dir_vec = traj_data["positions"][curr_time + self.stride] - traj_data["positions"][curr_time]
                # Convert to degrees
                dir_vec_deg = np.arctan2(dir_vec[0], dir_vec[1]) * 180 / np.pi

                # If the difference is larger than the threshold, skip the trajectory
                if np.abs(yaw_t0_deg - dir_vec_deg) > yaw_th_deg:
                    continue

                # min and max goal distance
                max_goal_dist = min(self.max_goal_dist * self.stride, traj_len - curr_time - 1)
                min_goal_dist = self.min_goal_dist * self.stride

                # Add to the index
                samples_index.append((traj_name, curr_time, min_goal_dist, max_goal_dist))

        return samples_index

    def _load_image(self, traj_name, time):
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(self.cache_file, map_size=2**40, readonly=True) # 1TB cache
        with self.lmdb_env.begin() as txn:
            image_buffer = txn.get(f"{traj_name}_{time:06d}".encode())
            image_bytes = io.BytesIO(bytes(image_buffer))

        image = Image.open(image_bytes)
        image_tensor = TF.to_tensor(image)

        return image_tensor

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

        # Get the actions, yaws, and goal vector
        positions = traj_data["positions"][start_index:end_index:self.stride] # [pred_horizon+1, 2]
        yaws = traj_data["yaws"][start_index:end_index:self.stride] # [pred_horizon+1, 1]
        goal_pos = traj_data["positions"][min(goal_time, len(traj_data["positions"]) - 1)]

        # Convert to local coordinates
        waypoints = self._to_local_coords(positions, positions[0], yaws[0])
        goal_pos = self._to_local_coords(goal_pos, positions[0], yaws[0])
        yaws = yaws - yaws[0]

        actions = waypoints[1:]

        return actions, yaws, goal_pos
    
    def _get_trajectory(self, trajectory_name):
        # Return none if the trajectory doesn't exist
        if not os.path.exists(os.path.join(self.dataset_folder, "trajectories", trajectory_name, "traj_processed.pkl")):
            return None
        with open(os.path.join(self.dataset_folder, "trajectories", trajectory_name, "traj_processed.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        return traj_data

    def _get_camera_intrinsics(self, trajectory_name):
        with open(os.path.join(self.dataset_folder, "trajectories", trajectory_name, "intr_est.pkl"), "rb") as f:
            intr_data = pickle.load(f)
        return intr_data

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        current_file, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]

        # Sample goal
        min_goal_dist_strided = min_goal_dist // self.stride
        max_goal_dist_strided = (max_goal_dist+1) // self.stride
        if min_goal_dist_strided == max_goal_dist_strided:
            goal_offset = min_goal_dist_strided
        else:
            goal_offset = np.random.randint(min_goal_dist_strided, max_goal_dist_strided)
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
        loaded_images = [self._load_image(current_file, t) for current_file, t in context]
        obs_context = torch.cat(loaded_images)

        # Get actions and goal vector
        curr_traj_data = self._get_trajectory(current_file)
        actions, yaws, goal_vec = self._compute_actions(curr_traj_data, curr_time, goal_time)

        # Pass the metadata as a tensor traj_name: xxxxxx_yyyyyy where xxxxxx is the video index and yyyyyy is the traj name
        dataset_name: str = self.dataset_name
        video_index = int(current_file.split("_")[0])
        traj_index = int(current_file.split("_")[1])
        metadata = torch.tensor([video_index, traj_index, curr_time], dtype=torch.int32)

        return (
            torch.as_tensor(obs_context, dtype=torch.float32),
            torch.as_tensor(goal_vec, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.float32),
            torch.as_tensor(yaws, dtype=torch.float32),
            dataset_name,
            metadata,
        )

    def __len__(self) -> int:
        return len(self.index_to_data)
    
    def __del__(self):
        if self.lmdb_env is not None:
            self.lmdb_env.close()
            self.lmdb_env = None

# Test the dataset
if __name__ == "__main__":
    
    vnav_dataset = VnavDataset(
        dataset_name="nadawalk_tokyo",
        dataset_type="train",
        datasets_folder="/home/caoruixiang/vecnav_dataset/mount_point/datasets",
        image_size=(128, 72),
        stride=5,
        pred_horizon=10,
        context_size=5,
        min_goal_dist=5,
        max_goal_dist=20
    )

    for i in range(100):
        # Test: take a sample from the dataset
        rand_index = np.random.randint(0, len(vnav_dataset))
        sample = vnav_dataset[rand_index]

        # slice the image list into individual images
        obs_images = sample[0]
        obs_images = torch.split(obs_images, 3, dim=0)
        sample_image = obs_images[-1]

        import matplotlib.pyplot as plt
        # Visualize the image in subplot 121
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].imshow(sample_image.permute(1, 2, 0))

        # Visualze the goal vector
        goal_vec = sample[1]

        # Visualize the actions
        actions = sample[2]
        if actions[0].shape == 3:
            actions = np.concatenate([np.array([[0, 0, 0]]), actions], axis=0)
        else:
            actions = np.concatenate([np.array([[0, 0]]), actions], axis=0)

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

        if int(actions[0].shape[0]) == 3:
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