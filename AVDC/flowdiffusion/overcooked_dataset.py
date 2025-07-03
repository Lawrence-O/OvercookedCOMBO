from matplotlib import pyplot as plt
import torch
import random
import numpy as np
import pickle
from hdf5_dataset import HDF5Dataset
# class OvercookedSequenceDataset(torch.utils.data.Dataset):
#     policies = ["sp1_final", "sp2_final", "sp3_final", "sp4_final", "sp5_final", "sp9_final", "bc_train"]
#     def __init__(self, args, split="train", allowed_policies=policies):

#         self.args = args
#         dataset_path = args.dataset_path
#         self.current_split = split
#         self.allowed_policies = allowed_policies
    
#         # Load dataset from HDF5
#         if dataset_path.endswith("hdf5"):
#             self.dataset = HDF5Dataset(args, self.current_split)
#             self.observations = np.array(self.dataset.dset["obs"]) # path_num * (path_length + 1) * num_agent * height * width * channels
#             self.actions = np.array(self.dataset.dset["actions"]) # path_num * path_length * num_agent * action_dim (1)
#             self.dones = np.array(self.dataset.dset["dones"]) # path_num * path_length * num_agent
#             self.env_info = np.array(self.dataset.dset["env_info"])
#             self.policy_id = np.array(self.dataset.dset["policy_id"]) # path_num * num_agent (agent1_policy_name, agent2_policy_name)
#             self.rewards = np.array(self.dataset.dset["rewards"]) # path_num * path_length * num_agent * reward_dim (1)
#         else:
#             raise ValueError(f"Unsupported dataset format: {dataset_path}")
        
#         self.horizon = args.horizon
#         self.max_path_length = args.max_path_length
#         self.use_padding = args.use_padding
#         self.action_horizon = 8
#         assert self.action_horizon <= self.horizon

        
#         *_, H, W, C = self.observations.shape
#         self.observation_dim = self.obs_cond_dim = (H, W, C)
        
          
#         self.n_episodes = len(self.observations)
#         self.train_partner_policies, self.org_test_partner_policies, self.test_partner_policies, unique_num_ids = \
#             self.get_num_policy_ids(allowed_policies=self.allowed_policies)
        
#         # unique_num_ids returns count of unique policy Ids (1, unique_policy_id) instead of (0, unique_policy_id)
#         # dummy_id should just be the last one
#         # self.dummy_id = unique_num_ids
#         self.num_partner_policies = unique_num_ids + 1 # +1 since num_classes is
#         # self.num_partner_policies = len(allowed_policies) + 1 

#         self.allowed_indicies = None
#         if self.allowed_policies:
#             print(f"Dataset filtered to include only these policies: {self.allowed_policies}")
#             print(f"Train policies after filtering: {self.train_partner_policies}")
#             print(f"Original test policies: {self.org_test_partner_policies}")
#             print(f"Test policies after filtering: {self.test_partner_policies}")

#             # self.dataset_average_rewards = {}
#             self.allowed_indicies = []
#             for idx in range(len(self.dataset)):
#                 _, _, policy_id = self.dataset.__getitem__(idx)
#                 partner_id = policy_id[1]
#                 for pname in self.allowed_policies:
#                     if pname in self.train_partner_policies:
#                         original_id = None
#                         for org_id, new_id in self.train_id_mapping.items():
#                             if new_id == self.train_partner_policies[pname]:
#                                 original_id = org_id
#                                 break
#                     if original_id == partner_id:
#                         self.allowed_indicies.append(idx)
#                         break
            

    
#     def get_num_policy_ids(self, allowed_policies=None):
#         # Reserve 0 for dummy_id
#         self.dummy_id = 0

#         train_policies_org = dict()
#         test_policies_org = dict()
#         try:
#             train_dataset = HDF5Dataset(self.args, "train")
#             if "policy_id" in train_dataset.dset:
#                 for key in train_dataset.dset['policy_id'].attrs.keys():
#                     policy_name = key[key.find("[") + 1 : key.find("]")]
#                     train_policies_org[policy_name] = train_dataset.dset['policy_id'].attrs[key]
#         except Exception as e:
#             print(f"Error loading train dataset: {e}")
#             return {}, {}, {}, 0
            
#         try:
#             test_dataset = HDF5Dataset(self.args, "test")
#             if "policy_id" in test_dataset.dset:
#                 for key in test_dataset.dset['policy_id'].attrs.keys():
#                     policy_name = key[key.find("[") + 1 : key.find("]")]
#                     test_policies_org[policy_name] = test_dataset.dset['policy_id'].attrs[key]
#         except Exception as e:
#             print(f"Error loading test dataset: {e}")
#             test_policies_org = {}
        
#         # Store ALL original IDs for reference
#         self.original_ids = {**train_policies_org, **test_policies_org}
        
#         # Create separate mappings for train and test policies
#         self.train_id_mapping = {}  # Original train ID -> new ID
#         self.test_id_mapping = {}   # Original test ID -> new ID
        
#         # Filter to only allowed policies
#         if allowed_policies is not None:
#             filtered_train_policies = {k: v for k, v in train_policies_org.items() if k in allowed_policies}
#             filtered_test_policies = test_policies_org  # Keep all test policies for now
#         else:
#             filtered_train_policies = train_policies_org
#             filtered_test_policies = test_policies_org
        
#         # Assign new IDs to train policies starting from 1
#         train_policies = dict()
#         next_id = 1
#         for name in sorted(filtered_train_policies.keys()):
#             train_policies[name] = next_id
#             original_id = filtered_train_policies[name]
#             self.train_id_mapping[original_id] = next_id
#             next_id += 1
        
#         # Assign new IDs to test policies
#         test_policies = dict()
#         offset_test_policies = dict()
#         for name in sorted(filtered_test_policies.keys()):
#             original_id = filtered_test_policies[name]
#             test_policies[name] = original_id  # Keep original for reference
#             offset_test_policies[name] = next_id
#             self.test_id_mapping[original_id] = next_id
#             next_id += 1
        
#         # Debug prints
#         print(f"Original train policies: {train_policies_org}")
#         print(f"Original test policies: {test_policies_org}")
#         print(f"Filtered train policies (new IDs): {train_policies}")
#         print(f"Test policies offset (new IDs): {offset_test_policies}")
#         print(f"Train ID mapping (orig->new): {self.train_id_mapping}")
#         print(f"Test ID mapping (orig->new): {self.test_id_mapping}")
        
#         return train_policies, test_policies, offset_test_policies, next_id - 1
    
#     def actual_norm(self, obs):
#         obs_max = obs.max(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
#         obs_min = obs.min(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
#         obs_norm = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-8) - 1
#         return obs_norm.astype(np.float32)
    

#     def make_indices(self, path_lengths, horizon):
#         '''
#             makes indices for sampling from dataset;
#             each index maps to a datapoint
#         '''
#         indices = []
#         for i, path_length in enumerate(path_lengths):
#             max_start = min(path_length - 1, self.max_path_length - horizon)
#             if not self.use_padding:
#                 max_start = min(max_start, path_length - horizon)
#             for start in range(max_start):
#                 end = start + horizon
#                 indices.append((i, start, end))
#         indices = np.array(indices)
#         return indices


#     def __len__(self):
#         if self.allowed_indicies is not None:
#             return len(self.allowed_indicies)
#         return self.dataset.__len__()
    
    
#     def __getitem__(self, idx):
#         # path_idx, start, end = self.indices[idx]
#         # obs = self.observations[path_idx]
#         # policy_id = self.policy_id[path_idx]

#         if self.allowed_indicies is not None:
#             obs, policy_id, actions = self.observations[self.allowed_indicies[idx]], \
#                 self.policy_id[self.allowed_indicies[idx]], \
#                 self.actions[self.allowed_indicies[idx]]
#         else:
#             obs, actions, policy_id = self.dataset.__getitem__(idx)

#         # obs: horizon x agent_num (2) x H x W x C
#         # actions: horizon x 2 x action dim (1) 
#         # policy : 2 (tuple)
        
#         obs = self.actual_norm(to_np(obs))

#         if obs.ndim == 4:
#             obs = np.expand_dims(obs, axis=1)            
#         T, _, H, W, C = obs.shape # Time, Agent, Height, Width, Channel 
        
#         # Get Ego Agent Observation (Agent ID  = 0)
#         start = random.randint(1, T - self.horizon)
#         end = start + self.horizon
#         trajectories = obs[start:end, 0]
#         future_actions = actions[start:start + self.action_horizon, 0].squeeze(-1)
        
#         # Condition on Past Trajectory or Previous Start State
#         conditions_obs = obs[start-1, 0]

#         # Condition on Partner (Agent ID = 1)
#         # TODO: Properly Handle Partner Policies
#         # original_partner_id = policy_id[1]
#         # if original_partner_id not in self.train_id_mapping:
#         #     raise ValueError(f"Original partner ID {original_partner_id} not found in train_id_mapping. Available IDs: {list(self.train_id_mapping.keys())}")
#         # conditions = self.train_id_mapping.get(original_partner_id)
#         conditions = policy_id[1]  # Use the original partner ID directly

#         x = torch.from_numpy(trajectories)
#         x_cond = torch.from_numpy(conditions_obs)
#         task_emb = conditions
#         actions = torch.from_numpy(to_np(future_actions)).long()

#         assert x.min() >= -1.0 and x.max() <= 1.0
#         assert x_cond.min() >= -1.0 and x_cond.max() <= 1.0
#         assert actions.min() >= 0 and actions.max() < 6, f"Actions must be in [0, 6), got range [{actions.min()}, {actions.max()}]"

#         return x, x_cond, task_emb, actions
        # def get_path_indexes_episode(self, policy_name):
        #         """
        #         Get a specific episode for a given policy name and episode index.
        #         """
        #         if policy_name in self.train_partner_policies:
        #             policy_id = self.train_partner_policies[policy_name]
        #         elif policy_name in self.test_partner_policies:
        #             # Use Original Test Policies
        #             policy_id = self.org_test_partner_policies[policy_name]
        #         else:
        #             raise ValueError(f"Policy name '{policy_name}' not found in dataset")

        #         matching_episodes = []
        #         for path_idx, policy_ids in enumerate(self.policy_id):
        #             if policy_ids[1] == policy_id: # Only check the partner policy ID
        #                 matching_episodes.append(path_idx)
                
        #         print(f"Found {len(matching_episodes)} episodes for policy name '{policy_name}'")
                
        #         return matching_episodes
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

    def __getitem__(self, idx):
        
        obs, actions, policy_id = self.hdf5_dataset[idx]
        obs, actions, policy_id = obs.numpy(), actions.numpy(), policy_id.numpy()

        obs = self._normalize_obs(obs)

        T = obs.shape[0]
        if T <= self.horizon:
            raise ValueError(f"Episode length {T} is less than or equal to horizon {self.horizon}. Cannot sample valid trajectory.")
            
        start_t = random.randint(1, T - self.horizon)
        end_t = start_t + self.horizon

        # Extract data slices for the EGO agent (agent 0)
        if obs.ndim == 4:
            obs = np.expand_dims(obs, axis=1) 
        conditions_obs_np = obs[start_t - 1, 0] 
        trajectories_np = obs[start_t:end_t, 0]
        future_actions_np = actions[start_t : start_t + self.action_horizon, 0].squeeze(-1)

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
        
        self.horizon = args.horizon
        self.max_path_length = args.max_path_length
        self.use_padding = args.use_padding
        
        *_, H, W, C = self.observations.shape
        self.observation_dim = self.obs_cond_dim = (H, W, C)
        self.reward_threshold = 18.0
        self.filtered_indices = self._filter_high_reward_trajectories()
        # self._plot_reward_histogram()
          
    
    def actual_norm(self, obs):
        obs_max = obs.max(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
        obs_min = obs.min(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
        obs_norm = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-8) - 1
        return obs_norm.astype(np.float32)
    
    # def _plot_reward_histogram(self):
    #     plt.figure(figsize=(16, 18))
    #     plt.hist(self._reward_sums_all, bins=30, color='teal', alpha=0.8)
    #     plt.axvline(self.reward_threshold, color='red', linestyle='--', label=f"Threshold = {self.reward_threshold}")
    #     plt.title("Reward Distribution of Sub-Trajectories")
    #     plt.xlabel("Sum of Rewards over Horizon")
    #     plt.ylabel("Count")
    #     plt.xticks(range(0,60,2), fontsize=10, rotation=45)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("reward_histogram.png")
    #     plt.close()
    
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

        obs = self.actual_norm(obs)

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


    