import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, args, split):
        self.hdf5_path = args.dataset_path
        self.dataset_name = self.hdf5_path.split("/")[-2]
        self.f = h5py.File(self.hdf5_path, "r")
        self.dset = self.f[split]
        self.length = self.dset["obs"].shape[0]

        self.episode_length = args.episode_length
        self.chunk_length = args.chunk_length or args.episode_length
        self.inp_outp_seq = "same"
        self.reset()
    
    def _reset_chunk_positions(self):
        if self.inp_outp_seq == "same":
            self.inp_seq_start = np.random.randint(0, self.episode_length - self.chunk_length + 1, self.length)
            self.outp_seq_start = self.inp_seq_start
            
    def reset(self):
        self._reset_chunk_positions()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obs = self.dset["obs"][idx]
        actions = self.dset["actions"][idx]
        policy_id = self.dset["policy_id"][idx]
        rewards = self.dset["rewards"][idx]

        chunk_start = self.inp_seq_start[idx]
        chunk_end = chunk_start + self.chunk_length
        
        obs = obs[chunk_start:chunk_end]
        actions = actions[chunk_start:chunk_end]
        return torch.FloatTensor(obs), torch.LongTensor(actions), torch.LongTensor(policy_id)

class New_HDF5Dataset(Dataset):
    def __init__(self, args, split):
        self.hdf5_path = args.dataset_path
        print("self.hdf5_path", self.hdf5_path)
        self.f = h5py.File(self.hdf5_path, "r")
        self.dset = self.f[split]
        self.length = self.dset["obs"].shape[0]

        self.episode_length = args.max_path_length
        self.chunk_length = args.max_path_length
        self.inp_outp_seq = "same"
        self.reset()
    
    def _reset_chunk_positions(self):
        if self.inp_outp_seq == "same":
            self.inp_seq_start = np.random.randint(0, self.episode_length - self.chunk_length + 1, self.length)
            self.outp_seq_start = self.inp_seq_start
            
    def reset(self):
        self._reset_chunk_positions()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return
            reference_observations: [L_in, *obs_shape]
            reference_actions: [L_in, 1] \in [0, num_actions]
            target_observations: [L_in, *obs_shape]
            targer_actions: [L_in, 1] \in [0, num_actions]
        ref_o + ref_a -> z
        tar_o + z -> tar_a
        """
        obs = self.dset["obs"][idx]
        actions = self.dset["actions"][idx]
        policy_id = self.dset["policy_id"][idx]

        chunk_start = self.inp_seq_start[idx]
        chunk_end = chunk_start + self.chunk_length
        
        obs = obs[chunk_start:chunk_end]
        actions = actions[chunk_start:chunk_end]
        return torch.FloatTensor(obs), torch.LongTensor(actions), torch.FloatTensor(obs), torch.LongTensor(actions), torch.LongTensor(policy_id)

    def get_start_matching_episode(self, start_obs):
        if not hasattr(self, "matched_episode_counter"):
            self.matched_episode_counter = 0
        self.matched_episode_counter = 0
        while True:
            obs, actions = self.dset["obs"][self.matched_episode_counter].astype(int), self.dset["actions"][self.matched_episode_counter]
            if False:
                print("start_obs", start_obs.shape)
                print("obs", obs.shape)
                for a in range(2):
                    for f in range(26):
                        start_f = start_obs[0, a, :, :, f]
                        obs_f = obs[0, a, :, :, f]
                        if not np.all(np.isclose(start_f, obs_f)):
                            print("a=", a, "f=", f)
                            print("start_f=", start_f)
                            print("obs_f=", obs_f)
                exit(0)
                print("start_obs", start_obs[0, 0, :, :, 0])
                print("obs[0]", obs[0, 0, :, :, 0])
                print("obs[0], agent 1", obs[0, 1, :, :, 0])
                print("np.all(np.isclose(obs[0], start_obs))", np.all(np.isclose(obs[0], start_obs)))
            if False:
                mmin = 100
                best_step = 0
                for step in range(401):
                    nn = np.logical_not(np.isclose(obs[step], start_obs)).sum()
                    if nn < mmin:
                        mmin = nn
                        best_step = step
                        best_obs = obs[step].copy()
                print("self.matched_episode_counter", self.matched_episode_counter)
                print("best_step", best_step)
                print("best_obs", best_obs[0, :, :, 0])
                print("wrong", np.logical_not(np.isclose(best_obs, start_obs)).sum())
            if np.all(np.isclose(obs[0], start_obs)):
                return obs, actions
            self.matched_episode_counter += 1
            if self.matched_episode_counter == 16:
                print("failed to find any")
                exit(0)
            self.matched_episode_counter %= self.__len__()
