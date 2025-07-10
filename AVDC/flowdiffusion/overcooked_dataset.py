from matplotlib import pyplot as plt
import torch
import random
import numpy as np
from hdf5_dataset import HDF5Dataset

class OvercookedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, args, split="train"):
        self.args = args
        self.current_split = split
        
        self.hdf5_dataset = HDF5Dataset(args, self.current_split)

        self._discover_all_policies_and_create_mappings()
        
        self.dummy_id = 0 # Explicitly reserve 0
        self.num_partner_policies = len(self.policy_name_to_id) + 1

        print(f"Dataset split '{self.current_split}' using a GLOBAL vocabulary of {self.num_partner_policies} classes.")

        self.horizon = args.horizon
        self.action_horizon = 8
        assert self.action_horizon <= self.horizon
        
        obs_sample, _, _ = self.hdf5_dataset[0]
        *_, H, W, C = obs_sample.shape
        self.observation_dim = (H, W, C)

    def _discover_all_policies_and_create_mappings(self):
        """
        Scans both train and test HDF5 splits to build one unified, global
        mapping from policy name to a unique integer ID.
        Reserves ID 0 for the dummy/unconditional case.
        """
        all_policy_names = set()
        
        # Scan the train split
        try:
            train_hdf5 = HDF5Dataset(self.args, "train")
            for key in train_hdf5.dset['policy_id'].attrs.keys():
                policy_name = key[key.find("[") + 1 : key.find("]")]
                all_policy_names.add(policy_name)
            del train_hdf5
        except Exception as e:
            print(f"Warning: Could not load or parse train split for vocab discovery: {e}")

        # Scan the test split
        try:
            test_hdf5 = HDF5Dataset(self.args, "test")
            for key in test_hdf5.dset['policy_id'].attrs.keys():
                policy_name = key[key.find("[") + 1 : key.find("]")]
                all_policy_names.add(policy_name)
            del test_hdf5
        except Exception as e:
            print(f"Warning: Could not load or parse test split for vocab discovery: {e}")

        # Create the single, sorted, global mapping
        sorted_names = sorted(list(all_policy_names))
        
        self.dummy_id = 0
        # Start real policy IDs from 1
        self.policy_name_to_id = {name: i + 1 for i, name in enumerate(sorted_names)}
        self.policy_id_to_name = {i + 1: name for i, name in enumerate(sorted_names)}

        print(f"Discovered {len(sorted_names)} unique policies across all splits.")
        print("Global Policy-ID Mapping:", self.policy_name_to_id)
        
        # We also need a way to get a policy name from its original ID in the HDF5 file
        self._original_id_to_name = {v: k[k.find("[") + 1 : k.find("]")] for k, v in self.hdf5_dataset.dset['policy_id'].attrs.items()}

    def __len__(self):
        return len(self.hdf5_dataset)

    def _normalize_obs(self, obs):
        """ Normalizes a numpy observation array to [-1, 1]. """
        # Scaled down by 255 since data is scaled by 255
        assert np.all(obs % 255 == 0)
        obs = obs.astype(np.float32) / 255.0

        HARDCODED_INDEX_TO_MAX_VAL = {
            16: 3.0,  # onions_in_pot
            17: 3.0,  # tomatoes_in_pot
            18: 3.0,  # onions_in_soup
            19: 3.0,  # tomatoes_in_soup
            20: 20.0, # soup_cook_time_remaining
        } 

        normalized_obs = np.zeros_like(obs, dtype=np.float32)

        for ch_idx in range(obs.shape[-1]):
            ch_data = obs[..., ch_idx]
            if ch_idx in HARDCODED_INDEX_TO_MAX_VAL:
                # Normalize from original game range [0, max_val] to [-1, 1]
                max_ch_val = HARDCODED_INDEX_TO_MAX_VAL[ch_idx]
                norm_ch = 2.0 * (ch_data / max_ch_val) - 1.0
            else:
                # Assume binary channel (original game values are 0 or 1).
                norm_ch = 2.0 * ch_data - 1.0
            normalized_obs[..., ch_idx] = norm_ch
        return normalized_obs

    def __getitem__(self, idx, start_t=None):
        
        obs, actions, policy_id = self.hdf5_dataset[idx]
        obs, actions, policy_id = obs.numpy(), actions.numpy(), policy_id.numpy()

        obs = self._normalize_obs(obs)

        T = obs.shape[0]
        if T <= self.horizon:
            raise ValueError(f"Episode length {T} is less than or equal to horizon {self.horizon}. Cannot sample valid trajectory.")
        if start_t is None:
            start_t = random.randint(1, T - self.horizon)
        else:
            assert 0 < start_t < T - self.horizon, f"start_t {start_t} must be in range (0, {T - self.horizon})"
        end_t = start_t + self.horizon

        # Extract data slices for the EGO agent (agent 0)
        if obs.ndim == 4:
            obs = np.expand_dims(obs, axis=1) 
        conditions_obs_np = obs[start_t - 1, 0] 
        trajectories_np = obs[start_t:end_t, 0]
        future_actions_np = actions[start_t - 1: start_t + self.action_horizon - 1, 0].squeeze(-1)

        # Get the partner policy ID for conditioning
        original_partner_id = policy_id[1]

        partner_name = self._original_id_to_name.get(original_partner_id)
        if partner_name is None:
             raise ValueError(f"Could not find name for original partner ID {original_partner_id}")

        # Use our clean mapping (which starts at 1) to get the final task embedding ID
        task_emb = self.policy_name_to_id[partner_name]

        # Convert to Tensors
        x = torch.from_numpy(trajectories_np)
        x_cond = torch.from_numpy(conditions_obs_np)
        actions = torch.from_numpy(future_actions_np).long()

        return x, x_cond, task_emb, actions
    
    
    
class SingleEpisodeOvercookedDataset(torch.utils.data.Dataset):    
    def __init__(self, base_dataset, policy_name, episode_idx=0):
        self.base_dataset = base_dataset
        self.policy_name = policy_name
        
        self.matching_episodes = self.base_dataset.get_path_indexes_episode(policy_name)
        if not self.matching_episodes:
            raise ValueError(f"No episodes found for policy name '{policy_name}'")
        
        if episode_idx >= len(self.matching_episodes):
            raise IndexError(f"Episode index {episode_idx} out of range for policy '{policy_name}'")
        
        self.path_idx = self.matching_episodes[episode_idx]
        self.observations = self.base_dataset.observations[self.path_idx]
        self.policy_id = self.base_dataset.policy_id[self.path_idx] 
        self.actions = self.base_dataset.actions[self.path_idx]
        self.episode_length = len(self.observations)
        self.horizon = self.base_dataset.horizon
        self.action_horizon = self.base_dataset.action_horizon

        print(f"Creating SingleEpisodeOvercookedDataset for policy '{policy_name}' at episode index {episode_idx} (path index {self.path_idx})")
        
    def __len__(self):
        return self.episode_length - self.horizon
    
    def __getitem__(self, idx, condition_single_input=True):
        # start = max(1, idx)
        start = 1
        end = start + self.horizon

        # Make sure we don't go out of bounds
        if end > self.episode_length:
            start = self.episode_length - self.horizon
            end = self.episode_length
        
        obs = self.observations[start:end] # Time x H x W x C
        cond_obs = self.observations[start-1]
        future_actions = self.actions[start:start + self.action_horizon, 0].squeeze(-1)
        
        # Normalize Observations
        obs = self.base_dataset.actual_norm(obs)
        cond_obs = self.base_dataset.actual_norm(cond_obs)

        # Condition on Partner Policy
        original_partner_id = self.policy_id[1]
        if original_partner_id not in self.train_id_mapping:
            raise ValueError(f"Original partner ID {original_partner_id} not found in train_id_mapping. Available IDs: {list(self.train_id_mapping.keys())}")
        conditions = self.train_id_mapping.get(original_partner_id)
         
        T, H, W, C = obs.shape # Time, Agent, Height, Width, Channel 

        x = torch.from_numpy(obs)
        x_cond = torch.from_numpy(cond_obs)
        task_emb = conditions
        actions = torch.from_numpy(future_actions).long()

        assert x.min() >= -1.0 and x.max() <= 1.0
        assert x_cond.min() >= -1.0 and x_cond.max() <= 1.0
        assert actions.min() >= 0 and actions.max() < 6

        return x, x_cond, task_emb, actions

class ActionOvercookedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, args, split="train"):

        self.args = args
        self.current_split = split
    
        # Load dataset from HDF5
        self.dataset = HDF5Dataset(args, self.current_split)
        self.observations = np.array(self.dataset.dset["obs"]) 
        self.actions = np.array(self.dataset.dset["actions"])
        self.rewards = np.array(self.dataset.dset["rewards"])
        
        self.horizon = args.horizon
        self.max_path_length = args.max_path_length
        self.use_padding = args.use_padding
        
        *_, H, W, C = self.observations.shape
        self.observation_dim = self.obs_cond_dim = (H, W, C)
        self.reward_threshold = 18.0
        self.filtered_indices = self._filter_high_reward_trajectories()
        # self._plot_reward_histogram()
          
    
    def _normalize_obs(self, obs):
        """ Normalizes a numpy observation array to [-1, 1]. """
        # Scaled down by 255 since data is scaled by 255
        assert np.all(obs % 255 == 0)
        obs = obs.astype(np.float32) / 255.0

        HARDCODED_INDEX_TO_MAX_VAL = {
            16: 3.0,  # onions_in_pot
            17: 3.0,  # tomatoes_in_pot
            18: 3.0,  # onions_in_soup
            19: 3.0,  # tomatoes_in_soup
            20: 20.0, # soup_cook_time_remaining
        } 

        normalized_obs = np.zeros_like(obs, dtype=np.float32)

        for ch_idx in range(obs.shape[-1]):
            ch_data = obs[..., ch_idx]
            if ch_idx in HARDCODED_INDEX_TO_MAX_VAL:
                # Normalize from original game range [0, max_val] to [-1, 1]
                max_ch_val = HARDCODED_INDEX_TO_MAX_VAL[ch_idx]
                norm_ch = 2.0 * (ch_data / max_ch_val) - 1.0
            else:
                # Assume binary channel (original game values are 0 or 1).
                norm_ch = 2.0 * ch_data - 1.0
            normalized_obs[..., ch_idx] = norm_ch
        return normalized_obs
    
    def _plot_reward_histogram(self):
        plt.figure(figsize=(16, 18))
        plt.hist(self._reward_sums_all, bins=30, color='teal', alpha=0.8)
        plt.axvline(self.reward_threshold, color='red', linestyle='--', label=f"Threshold = {self.reward_threshold}")
        plt.title("Reward Distribution of Sub-Trajectories")
        plt.xlabel("Sum of Rewards over Horizon")
        plt.ylabel("Count")
        plt.xticks(range(0,60,2), fontsize=10, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig("reward_histogram.png")
        plt.close()
    
    def _filter_high_reward_trajectories(self):
        filtered = []
        reward_sums = []
        for traj_idx in range(len(self.dataset)):
            rewards = self.rewards[traj_idx]  # [T, 2, 1]
            traj_len = rewards.shape[0]
            for start in range(1, traj_len - self.horizon):
                end = start + self.horizon
                reward_sum = rewards[start:end, 0].sum()  # agent 0
                reward_val = reward_sum.item() if hasattr(reward_sum, "item") else reward_sum
                reward_sums.append(reward_val)
                if reward_val >= self.reward_threshold:
                    filtered.append((traj_idx, start))
        # self._reward_sums_all = reward_sums
        print(f"Filtered {len(filtered)} sub-trajectories with rewards ≥ {self.reward_threshold}")
        return filtered
    

    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        traj_idx, start = self.filtered_indices[idx]
        obs = self.observations[traj_idx]  # shape: [T+1, 2, H, W, C]
        actions = self.actions[traj_idx]   # shape: [T, 2, 1]

        obs = self._normalize_obs(obs)

        if obs.ndim == 4:
            obs = np.expand_dims(obs, axis=1)

        conditions_obs = obs[start - 1, 0]  # ego agent
        future_actions = actions[start:start + self.horizon, 0]

        x = torch.from_numpy(future_actions)  # shape: [horizon, 1]
        x_cond = torch.from_numpy(conditions_obs)  # shape: [H, W, C]

        assert x.min() >= 0 and x.max() < 6, f"Actions must be in [0, 6), got range [{x.min()}, {x.max()}]"
        assert x_cond.min() >= -1.0 and x_cond.max() <= 1.0

        return x, x_cond

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x


    