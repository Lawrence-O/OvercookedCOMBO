import torch
import random
import numpy as np
import pickle
from hdf5_dataset import HDF5Dataset
class OvercookedSequenceDataset(torch.utils.data.Dataset):
    # policies = ["sp1_final", "sp2_final", "sp3_final", "sp4_final", "sp5_final", "sp9_final", "bc_train"]
    policies = ["sp1_final"]
    def __init__(self, args, split="train", allowed_policies=policies):

        self.args = args
        dataset_path = args.dataset_path
        self.current_split = split
        self.allowed_policies = allowed_policies
    
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
        
        self.horizon = args.horizon
        self.max_path_length = args.max_path_length
        self.use_padding = args.use_padding
        
        *_, H, W, C = self.observations.shape
        self.observation_dim = self.obs_cond_dim = (H, W, C)
          
        self.n_episodes = len(self.observations)
        self.train_partner_policies, self.org_test_partner_policies, self.test_partner_policies, unique_num_ids = \
            self.get_num_policy_ids(allowed_policies=self.allowed_policies)
        
        # unique_num_ids returns count of unique policy Ids (1, unique_policy_id) instead of (0, unique_policy_id)
        # dummy_id should just be the last one
        # self.dummy_id = unique_num_ids
        # self.num_partner_policies = unique_num_ids + 1 # +1 since num_classes is
        self.num_partner_policies = len(allowed_policies) + 1 

        self.allowed_indicies = None
        if self.allowed_policies:
            print(f"Dataset filtered to include only these policies: {self.allowed_policies}")
            print(f"Train policies after filtering: {self.train_partner_policies}")
            print(f"Original test policies: {self.org_test_partner_policies}")
            print(f"Test policies after filtering: {self.test_partner_policies}")

            # self.dataset_average_rewards = {}
            self.allowed_indicies = []
            for idx in range(len(self.dataset)):
                _, _, policy_id = self.dataset.__getitem__(idx)
                partner_id = policy_id[1]
                for pname, pid in self.train_partner_policies.items():
                    if pid == partner_id and pname in self.allowed_policies:
                        self.allowed_indicies.append(idx)
                        # self.dataset_average_rewards.setdefault(pname, []).append(self.rewards[idx])
                        break
            # print(f"Filtered dataset contains {len(self.allowed_indicies)} samples with allowed policies")
            # self.dataset_average_rewards = {k: np.mean(v) for k, v in self.dataset_average_rewards.items()}
            # print(f"Average rewards for allowed policies: {self.dataset_average_rewards}")
            # exit()

    
    def get_num_policy_ids(self, allowed_policies=None):
        train_policies_org = dict()
        test_policies_org = dict()
        try:
            train_dataset = HDF5Dataset(self.args, "train")
            if "policy_id" in train_dataset.dset:
                for key in train_dataset.dset['policy_id'].attrs.keys():
                    policy_name = key[key.find("[") + 1 : key.find("]")]
                    if allowed_policies is not None and policy_name not in allowed_policies:
                        continue
                    train_policies_org[policy_name] = train_dataset.dset['policy_id'].attrs[key]
        except Exception as e:
            print(f"Error loading train dataset: {e}")
            return {}, {}, 0
        try:
            test_dataset = HDF5Dataset(self.args, "test")
            if "policy_id" in test_dataset.dset:
                for key in test_dataset.dset['policy_id'].attrs.keys():
                    policy_name = key[key.find("[") + 1 : key.find("]")]
                    # if allowed_policies is not None and policy_name not in allowed_policies:
                    #     continue
                    test_policies_org[policy_name] = test_dataset.dset['policy_id'].attrs[key]
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            return train_policies_org, {}, {}, max(train_policies_org.values())
        
        self.original_ids = {**train_policies_org, **train_policies_org}
        all_policies = list(train_policies_org.keys()) + list(test_policies_org.keys())
        train_policies = dict()
        test_policies = dict()
        offset_test_policies = dict()

        for i, name in enumerate(sorted(train_policies_org.keys())):
            train_policies[name] = i
        self.dummy_id = len(train_policies) # TODO: REMOVE
        offset = len(train_policies) + 1 # TODO: REMOVE
        for i, name in enumerate(sorted(test_policies_org.keys())):
            test_policies[name] = test_policies_org[name]  
            offset_test_policies[name] = i + offset  

        return train_policies, test_policies, offset_test_policies, len(all_policies)
    
    def actual_norm(self, obs):
        obs_max = obs.max(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
        obs_min = obs.min(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
        obs_norm = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-8) - 1
        return obs_norm.astype(np.float32)
    

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
        if self.allowed_indicies is not None:
            return len(self.allowed_indicies)
        return self.dataset.__len__()
    
    
    def __getitem__(self, idx, condition_single_input=True):
        # path_idx, start, end = self.indices[idx]
        # obs = self.observations[path_idx]
        # policy_id = self.policy_id[path_idx]

        if self.allowed_indicies is not None:
            obs, policy_id = self.observations[self.allowed_indicies[idx]], \
                self.policy_id[self.allowed_indicies[idx]]
        else:
            obs, _, policy_id = self.dataset.__getitem__(idx)

        # obs: horizon x agent_num (2) x H x W x C
        # actions: horizon x 2 x action dim (1) 
        # policy : 2 (tuple)
        
        obs = self.actual_norm(to_np(obs))

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
        self.episode_length = len(self.observations)
        self.horizon = self.base_dataset.horizon

        print(f"Creating SingleEpisodeOvercookedDataset for policy '{policy_name}' at episode index {episode_idx} (path index {self.path_idx})")
        
    def __len__(self):
        return len(self.observations) - self.horizon + 1
    
    def __getitem__(self, idx, condition_single_input=True):
        start = max(1, idx)
        end = start + self.horizon

        # Make sure we don't go out of bounds
        if end > self.episode_length:
            start = self.episode_length - self.horizon
            end = self.episode_length
        
        obs = self.observations[start:end] # Time x H x W x C
        cond_obs = self.observations[start-1]
        obs = self.base_dataset.actual_norm(obs)
        cond_obs = self.base_dataset.actual_norm(cond_obs)

        # Condition on Partner Policy
        policy_id = self.policy_id[1]
         
        T, H, W, C = obs.shape # Time, Agent, Height, Width, Channel 

        trajectory = obs
        conditions_obs = cond_obs

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




def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x


    