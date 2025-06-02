import torch
import random
import numpy as np
import pickle
from hdf5_dataset import HDF5Dataset
class OvercookedSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, args, split="train"):

        self.args = args
        dataset_path = args.dataset_path
        self.current_split = split
    
        # Load dataset from HDF5
        if dataset_path.endswith("hdf5"):
            self.dataset = HDF5Dataset(args, self.current_split) # TODO: change to train later
            self.observations = np.array(self.dataset.dset["obs"]) # path_num * (path_length + 1) * num_agent * height * width * channels
            self.actions = np.array(self.dataset.dset["actions"]) # path_num * path_length * num_agent * action_dim (1)
            self.dones = np.array(self.dataset.dset["dones"]) # path_num * path_length * num_agent
            self.env_info = np.array(self.dataset.dset["env_info"])
            self.policy_id = np.array(self.dataset.dset["policy_id"]) # path_num * num_agent (agent1_policy_name, agent2_policy_name)
            self.rewards = np.array(self.dataset.dset["rewards"]) # path_num * path_length * num_agent * reward_dim (1)
        else:
            with open(dataset_path, "rb") as input_file:
                self.cond_init_mins, self.cond_init_maxs, self.cond_obs_imL_mean, self.cond_obs_imL_std, \
                    self.conditions, self.dummy_cond = pickle.load(input_file) #ee_pose+gripper, left image
        
        self.horizon = args.horizon
        self.max_path_length = args.max_path_length
        self.use_padding = args.use_padding
        
        *_, H, W, C = self.observations.shape
        self.observation_dim = self.obs_cond_dim = (H, W, C)
          
        self.n_episodes = len(self.observations)
        self.train_partner_policies, self.org_test_partner_policies, self.test_partner_policies, unique_num_ids = self.get_num_policy_ids()
        
        # unique_num_ids returns count of unique policy Ids (1, unique_policy_id) instead of (0, unique_policy_id)
        # dummy_id should just be the last one
        self.dummy_id = unique_num_ids
        self.num_partner_policies = unique_num_ids + 1 # +1 since num_classes is

        self.path_lengths = [obs.shape[0] for obs in self.observations]
        self.indices = self.make_indices(self.path_lengths, self.horizon)
    
    def get_num_policy_ids(self):
        train_policies = dict()
        test_policies = dict()
        try:
            train_dataset = HDF5Dataset(self.args, "train")
            if "policy_id" in train_dataset.dset:
                for key in train_dataset.dset['policy_id'].attrs.keys():
                    policy_name = key[key.find("[") + 1 : key.find("]")]
                    train_policies[policy_name] = train_dataset.dset['policy_id'].attrs[key]
        except Exception as e:
            print(f"Error loading train dataset: {e}")
            return {}, {}, 0
        try:
            test_dataset = HDF5Dataset(self.args, "test")
            if "policy_id" in test_dataset.dset:
                for key in test_dataset.dset['policy_id'].attrs.keys():
                    policy_name = key[key.find("[") + 1 : key.find("]")]
                    test_policies[policy_name] = test_dataset.dset['policy_id'].attrs[key]
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            return train_policies, {}, {}, max(train_policies.values())

        # Offset test policies by the maximum train policy ID
        max_train_id = max(train_policies.values()) if train_policies else -1
        offset = max_train_id + 1
        offset_test_policies = dict()
        for name, original_id in test_policies.items():
            offset_test_policies[name] = original_id + offset
        
        # Combine all unique policy names to count total unique policies
        all_unique_policy_names = set(train_policies.keys()).union(set(test_policies.keys()))
        num_total_unique_policies = len(all_unique_policy_names)

        # TODO
        # print("=== Policy IDs Summary ===")
        # print(f"Final Train Policies ({len(train_policies)}): {train_policies}")
        # print(f"Final Test Policies ({len(offset_test_policies)}): {offset_test_policies}")
        # print(f"Total unique policy names found across train and test: {num_total_unique_policies}")
        
        return train_policies, test_policies, offset_test_policies, num_total_unique_policies
    
    def actual_norm(self, obs):
        obs_max = obs.max(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
        obs_min = obs.min(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
        obs_norm = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-8) - 1
        return obs_norm.astype(np.float32)
    
    def consistent_norm(self, obs):
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

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices


    def __len__(self):
        return self.dataset.__len__()
    
    
    def __getitem__(self, idx, condition_single_input=True):
        # path_idx, start, end = self.indices[idx]
        # obs = self.observations[path_idx]
        # policy_id = self.policy_id[path_idx]

        obs, _, policy_id = self.dataset.__getitem__(idx)

        # obs, actions, policy_id = self.dataset.__getitem__(idx)
        # obs: horizon x agent_num (2) x H x W x C
        # actions: horizon x 2 x action dim (1) 
        # policy : 2 (tuple)
        
        obs = self.actual_norm(obs.numpy())

        # player_loc_orietnations = obs[:, :, :, :, :10]
        # dish_onions = obs[:, :, :, :, 22:24]
        # obs = np.concatenate([player_loc_orietnations, dish_onions], axis=-1)
        if obs.ndim == 4:
            obs = np.expand_dims(obs, axis=1)            
        T, _, H, W, C = obs.shape # Time, Agent, Height, Width, Channel 
        # Get Ego Agent Observation (Agent ID  = 0)
        start = random.randint(1, T - self.horizon)
        end = start + self.horizon
        trajectories = obs[start:end, 0]
        
        # Condition on Past Trajectory or Previous Start State
        conditions_obs = obs[start-1, 0] if condition_single_input else obs[:start, 0]

        # Condition on Partner (Agent ID = 1)
        conditions = policy_id[1]

        # Create a Mask for Valid Condition Observations
        valid_len = 1 if condition_single_input else start
        cond_inputs = np.zeros((self.horizon, H, W, C), dtype=np.float32)
        cond_masks = np.zeros((self.horizon), dtype=np.float32)
        cond_inputs[-valid_len:] = conditions_obs
        cond_masks[-valid_len:] = 1.0

        # Trajectory Shape: (Horizon, H, W, C)
        # Conditions Shape : (1)
        # Condition Inputs: (valid_len, H, W, C)
        # Condition Masks : (valid_len,)
        x = torch.from_numpy(trajectories)
        x_cond = torch.from_numpy(cond_inputs)
        task_emb = conditions

        assert x.min() >= -1.0 and x.max() <= 1.0
        assert x_cond.min() >= -1.0 and x_cond.max() <= 1.0

        return x, x_cond, task_emb
    
    def get_path_indexes_episode(self, policy_name):
        """
        Get a specific episode for a given policy name and episode index.
        """
        if policy_name in self.train_partner_policies:
            policy_id = self.train_partner_policies[policy_name]
        elif policy_name in self.test_partner_policies:
            # Use Original Test Policies
            policy_id = self.org_test_partner_policies[policy_name]
        else:
            raise ValueError(f"Policy name '{policy_name}' not found in dataset")
        

        matching_episodes = []
        for path_idx, policy_ids in enumerate(self.policy_id):
            if policy_ids[1] == policy_id: # Only check the partner policy ID
                matching_episodes.append(path_idx)
        
        print(f"Found {len(matching_episodes)} episodes for policy name '{policy_name}'")
        
        return matching_episodes
    
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

        print(f"Creating SingleEpisodeOvercookedDataset for policy '{policy_name}' at episode index {episode_idx} with path index {self.path_idx}")

        self.episode_length = len(self.observations)
        self.horizon = self.base_dataset.horizon
        self.observation_dim = self.base_dataset.observation_dim
        self.dummy_id = self.base_dataset.dummy_id

        print(f"Expected Reward: {self.base_dataset.rewards[self.path_idx].sum()}")
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx, condition_single_input=True):
        start = max(1, idx)
        end = start + self.horizon

        # Make sure we don't go out of bounds
        if end > self.episode_length:
            start = self.episode_length - self.horizon
            end = self.episode_length


        obs = self.observations[start:end]
        obs = self.base_dataset.actual_norm(obs)

        # Condition on Partner Policy
        policy_id = self.policy_id[1]

        if obs.ndim == 4:
            obs = np.expand_dims(obs, axis=1)            
        T, _, H, W, C = obs.shape # Time, Agent, Height, Width, Channel 

        trajectory = obs[start:end, 0]
        conditions_obs = obs[start-1, 0]


        # Create a Mask for Valid Condition Observations
        valid_len = 1 if condition_single_input else start
        cond_inputs = np.zeros((self.horizon, H, W, C), dtype=np.float32)
        cond_masks = np.zeros((self.horizon), dtype=np.float32)
        cond_inputs[-valid_len:] = conditions_obs
        cond_masks[-valid_len:] = 1.0

        # Trajectory Shape: (Horizon, H, W, C)
        # Conditions Shape : (1)
        # Condition Inputs: (valid_len, H, W, C)
        # Condition Masks : (valid_len,)
        x = torch.from_numpy(trajectory)
        x_cond = torch.from_numpy(cond_inputs)
        task_emb = policy_id

        assert x.min() >= -1.0 and x.max() <= 1.0
        assert x_cond.min() >= -1.0 and x_cond.max() <= 1.0

        return x, x_cond, task_emb






    