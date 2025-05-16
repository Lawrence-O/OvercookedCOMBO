import torch
import random
import numpy as np
import pickle
import copy
from hdf5_dataset import HDF5Dataset
class OvercookedSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, args, sample_rate=1):

        dataset_path = args.dataset_path

        if dataset_path.endswith("hdf5"):
            
            self.dataset = HDF5Dataset(args, "test") # TODO: change to train later
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

        self.dummy_cond = np.int64(0)
        
        self.cond_dim = 8 # input to model init, T5 self.conditions
        self.observation_dim = self.obs_cond_dim = (8,5,26)
        
        
        self.n_episodes = len(self.observations)
        self.num_partner_policies = 25
        
        # mins and max 0, 255
        self.mins = 0
        self.maxs = 255

        self.path_lengths = [obs.shape[0] for obs in self.observations]
        self.indices = self.make_indices(self.path_lengths, self.horizon)
        # self.normalize()
    

    def normalize(self):
        '''
            normalize fields that will be predicted by the diffusion model
            normalize from [0, 255] to [-1, 1]
        '''
        self.normed_observations = copy.deepcopy(self.observations)
        self.normed_observations = (self.normed_observations - self.mins) / (self.maxs - self.mins + 1e-5) # [ 0, 1 ]
        self.normed_observations = (self.normed_observations * 2) - 1 # [ -1, 1 ]
        # self.normed_observations = [self.normed_observations[np.sum(self.path_lengths[:i]) if i>0 else 0 : np.sum(self.path_lengths[:i+1])] for i in range(len(self.path_lengths))]


    def normalize_init(self, init_states):
        """normalize init state"""
        normed_init_states = (np.array(init_states) - self.mins) / (self.maxs - self.mins + 1e-5) # [0,1]
        normed_init_states = (normed_init_states * 2) - 1 # [-1,1]
        return normed_init_states.astype(np.float32)
    
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
        
        obs = self.consistent_norm(obs.numpy())

        # player_loc_orietnations = obs[:, :, :, :, :10]
        # dish_onions = obs[:, :, :, :, 22:24]
        # obs = np.concatenate([player_loc_orietnations, dish_onions], axis=-1)
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