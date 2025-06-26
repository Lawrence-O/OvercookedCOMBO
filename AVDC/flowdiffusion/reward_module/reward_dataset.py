import torch
import random
import numpy as np
import sys, os
sys.path.append("/home/law/Workspace/repos/COMBO/AVDC/flowdiffusion")
from hdf5_dataset import HDF5Dataset
from copy import deepcopy

class RewardPredictorDataset(torch.utils.data.Dataset):
    def __init__(self, args, split="train"):

        self.args = args
        dataset_path = args.dataset_path
        self.current_split = split
    
        # Load dataset from HDF5
        if dataset_path.endswith("hdf5"):
            self.dataset = HDF5Dataset(args, self.current_split)
            self.observations = np.array(self.dataset.dset["obs"]) # path_num * (path_length + 1) * num_agent * height * width * channels
            self.actions = np.array(self.dataset.dset["actions"]) # path_num * path_length * num_agent * action_dim (1)
            self.dones = np.array(self.dataset.dset["dones"]) # path_num * path_length * num_agent
            self.env_info = np.array(self.dataset.dset["env_info"])
            self.policy_id = np.array(self.dataset.dset["policy_id"]) # path_num * num_agent (agent1_policy_name, agent2_policy_name)
            self.rewards = np.array(self.dataset.dset["rewards"]) # path_num * path_length * num_agent * reward_dim (1)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
        
        
        *_, H, W, C = self.observations.shape
        self.observation_dim = self.obs_cond_dim = (H, W, C)

        self.reward_max = 25.0
        self.reward_min = 0.0

        print(f"Loaded {self.current_split} dataset with {len(self.observations)} paths")
        print("self.dataset length:", self.dataset.__len__())
        print("self.rewards shape:", self.rewards.shape)
        print("Reward stats before normalization:", np.min(self.rewards), np.max(self.rewards), np.mean(self.rewards))
        unique_rewards, counts = np.unique(self.rewards, return_counts=True)
        print("Reward distribution (before normalization):")
        for r, c in zip(unique_rewards, counts):
            print(f"Reward = {r:.2f}: {c} samples")

        print("Reward stats before normalization:")
        print(f"Min: {np.min(self.rewards):.2f}, Max: {np.max(self.rewards):.2f}, Mean: {np.mean(self.rewards):.4f}")

        
        normed = self.normalize_reward(deepcopy(self.rewards))
        print("Reward stats after normalization:", np.min(normed), np.max(normed), np.mean(normed))
    def actual_norm(self, obs):
        obs_max = obs.max(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
        obs_min = obs.min(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
        obs_norm = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-8) - 1
        return obs_norm.astype(np.float32)
    
    def normalize_reward(self, rewards):
        rewards = (rewards - self.reward_min) / (self.reward_max - self.reward_min)
        return rewards
    

    def __len__(self):
        return self.observations.__len__()
    
    def __getitem__(self, idx):
        obs = self.observations[idx]
        obs = self.actual_norm(to_np(obs))
        if obs.ndim == 4:
            obs = np.expand_dims(obs, axis=1)
        rewards = self.rewards[idx]
        # rewards = self.normalize_reward(to_np(rewards))

        T = obs.shape[0]
        # Sample A random time step
        t = random.randint(0, T - 2)

        obs_t = obs[t, 0]# (H, W, C)
        next_obs_t = obs[t + 1, 0]
        reward_t = rewards[t, 0]

        obs_t = torch.from_numpy(obs_t).float()
        next_obs_t = torch.from_numpy(next_obs_t).float()
        reward_t = torch.from_numpy(reward_t).float()

        assert obs.min() >= -1.0 and obs.max() <= 1.0
        assert next_obs_t.min() >= -1.0 and next_obs_t.max() <= 1.0
        # assert reward_t.min() >= 0.0 and reward_t.max() <= 1.0
        
        return obs_t, next_obs_t, reward_t
    
def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x
