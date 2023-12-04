import argparse
import os
import shutil
import random

DATASETS_PATH = "/home/caoruixiang/vecnav_dataset/mount_point/datasets/nadawalk_tokyo/trajectories"
DATA_SPLITS_PATH = "data_splits"

def main(args: argparse.Namespace):
    # Get the names of the folders in the data directory that contain the file 'traj_data.pkl'
    traj_folder_names = [
        f for f in sorted(os.listdir(DATASETS_PATH))
        if os.path.isdir(os.path.join(DATASETS_PATH, f))
        # and "traj_est_scaled.pkl" in os.listdir(os.path.join(DATASETS_PATH, f))
    ]

    # Randomly shuffle the names of the folders
    random.shuffle(traj_folder_names)

    # Split the names of the folders into train and evaluate sets
    split_index = int(args.split_ratio * len(traj_folder_names))
    train_traj_folder_names = traj_folder_names[:split_index]
    eval_traj_folder_names = traj_folder_names[split_index:]

    # Create directories for the train and evaluate setdatasets
    train_dir = os.path.join(DATA_SPLITS_PATH, args.dataset_name, "train")
    eval_dir = os.path.join(DATA_SPLITS_PATH, args.dataset_name, "eval")
    for dir_path in [train_dir, eval_dir]:
        if os.path.exists(dir_path):
            print(f"Clearing files from {dir_path} for new data split")
            shutil.rmtree(dir_path)
        else:
            print(f"Creating {dir_path}")
        os.makedirs(dir_path)

    # Write the names of the train and evaluate folders to files
    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        for folder_name in train_traj_folder_names:
            f.write(folder_name + "\n")

    with open(os.path.join(eval_dir, "traj_names.txt"), "w") as f:
        for folder_name in eval_traj_folder_names:
            f.write(folder_name + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        "-d",
        default="nadawalk_tokyo",
        type=str,
    )

    parser.add_argument(
        "--split_ratio",
        "-s",
        type=float,
        default=0.8,
        help="Train/test split (default: 0.8)",
    )
        
    args = parser.parse_args()
    main(args)
    print("Done")