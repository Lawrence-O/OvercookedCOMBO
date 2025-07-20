import argparse
from contextlib import contextmanager
import gc
from copy import deepcopy
import datetime
from pathlib import Path
import sys
mapbt_path = '/home/law/Workspace/repos/COMBO/mapbt_package/mapbt'
if mapbt_path not in sys.path:
    sys.path.append(mapbt_path)
overcooked_ai_py_src_path = '/home/law/Workspace/repos/COMBO/mapbt_package/mapbt/envs/overcooked/overcooked_berkeley/src/overcooked_ai_py'
if overcooked_ai_py_src_path not in sys.path:
    sys.path.append(overcooked_ai_py_src_path)
from overcooked_dataset import SingleEpisodeOvercookedDataset
import numpy as np
import torch as th
import os.path as osp
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")
from mapbt_package.mapbt.envs.overcooked.Overcooked_Env import Overcooked
from mapbt_package.mapbt.envs.env_wrappers import ChooseSubprocVecEnv
from einops.einops import rearrange
from mapbt_package.mapbt.algorithms.population.policy_pool import PolicyPool as Policy


def make_eval_env(all_args, run_dir, nenvs=3):
    def get_env_fn(rank):
        def init_env():
            env = Overcooked(all_args, run_dir, rank=rank)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(nenvs)])

def load_partner_policy(args, policy_name, device="cpu"):
    """Load a partner policy by name."""
    policy = Policy(None, None, None, None, device=device)
    featurize_type = policy.load_population(args.population_yaml_path, evaluation=True)
    policy = policy.policy_pool[policy_name]
    feat_type = featurize_type.get(policy_name, 'ppo')
    return policy, feat_type


def max_normalize_obs(obs): # TODO: Fix this
    if isinstance(obs, th.Tensor):
        # Assume obs shape is [T, H, W, C]
        obs_max = th.amax(obs, dim=(0, 1, 2), keepdim=True)
        obs_min = th.amin(obs, dim=(0, 1, 2), keepdim=True)
        obs_norm = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-8) - 1
        return obs_norm.float()
    elif isinstance(obs, np.ndarray):
        obs_max = obs.max(axis=(0, 1, 2), keepdims=True)
        obs_min = obs.min(axis=(0, 1, 2), keepdims=True)
        obs_norm = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-8) - 1
    else:
        raise ValueError("Unsupported observation type. Must be numpy array or PyTorch tensor.")
    return obs_norm.astype(np.float32)

def normalize_obs(obs, divide=True):
        """ Normalizes a numpy observation array to [-1, 1]. 
        If `divide` is True, it un-scales the observation by 255.0.
        """
        # Scaled down by 255 since data is scaled by 255
        if isinstance(obs, th.Tensor):
            obs = to_np(obs)
        
        if divide:
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

def normalize_obs_vectorized(obs, divide=True, num_channels=26):
    """
    Vectorized function to normalize a numpy observation array to [-1, 1].
    Args:
        obs (np.ndarray): The observation tensor. Can be of any shape as long as
                          the last dimension is the channel dimension.
        divide (bool): If True, un-scales the observation from [0, 255] to [0, 1].
        num_channels (int): The total number of channels in the observation.
    Returns:
        np.ndarray: The normalized observation array.
    """
    if isinstance(obs, th.Tensor):
        obs = to_np(obs)
    
    if divide:
        assert np.all(obs % 255 == 0)
        obs = obs.astype(np.float32) / 255.0


    # 1. Define the mapping from channel index to its original maximum value.
    #    Default to 1.0 for all binary channels.
    HARDCODED_INDEX_TO_MAX_VAL = {
        16: 3.0,  # onions_in_pot
        17: 3.0,  # tomatoes_in_pot
        18: 3.0,  # onions_in_soup
        19: 3.0,  # tomatoes_in_soup
        20: 20.0, # soup_cook_time_remaining
    }
    
    # This vector will contain the max value for each channel.
    max_vals = np.ones(num_channels, dtype=np.float32)
    for idx, max_val in HARDCODED_INDEX_TO_MAX_VAL.items():
        if idx < num_channels:
            max_vals[idx] = max_val

    scale = 2.0 / max_vals
    shift = -1.0
    normalized_obs = obs * scale + shift
    return normalized_obs.astype(np.float32)

def unnormalize_obs(obs, eps=5e-1):
    # Assume obs is in range [-1, 1]; just directly un-normalize back
    assert obs.min() >= -1.0 - eps and obs.max() <= 1.0 + eps, "Observation must be in range [-1, 1]" \
    " current min: {}, max: {}".format(obs.min(), obs.max())
    if isinstance(obs, th.Tensor):
        obs = obs.detach().cpu().numpy()

    FEATURE_CHANNEL_MAP = [
        "player_0_loc", "player_1_loc", 
        "player_0_orientation_0", "player_0_orientation_1", "player_0_orientation_2", "player_0_orientation_3",
        "player_1_orientation_0", "player_1_orientation_1", "player_1_orientation_2", "player_1_orientation_3",
        "pot_loc", "counter_loc", "onion_disp_loc", "tomato_disp_loc", "dish_disp_loc", "serve_loc",
        "onions_in_pot", "tomatoes_in_pot",
        "onions_in_soup", "tomatoes_in_soup", 
        "soup_cook_time_remaining", "soup_done",
        "dishes", "onions", "tomatoes",
        "urgency"
    ]
    CHANNEL_FEATURE_MAP = {name: i for i, name in enumerate(FEATURE_CHANNEL_MAP)}
    

    # obs = np.clip(obs, -1.0, 1.0)
    unnorm_obs = np.zeros_like(obs, dtype=np.float32)
    max_values = {
        "onions_in_pot": 3.0,
        "tomatoes_in_pot": 3.0,
        "onions_in_soup": 3.0,
        "tomatoes_in_soup": 3.0,
        "soup_cook_time_remaining": 20.0
    }

    idx_to_max = {}
    for ch_name, max_val in max_values.items():
        if ch_name in CHANNEL_FEATURE_MAP:
            ch_idx = CHANNEL_FEATURE_MAP[ch_name]
            idx_to_max[ch_idx] = max_val

    for ch_idx in range(obs.shape[-1]):
        channel_data = obs[:, :, ch_idx]
        if ch_idx in idx_to_max:
            max_val = idx_to_max[ch_idx]
            # Rescale [-1, 1] to [0, 1];  Rescale [0, 1] to [0, max_val]
            channel_data = ((channel_data + 1.0) / 2.0 )* max_val
        else:
            # Rescale from [-1, 1] to [0, 1]
            channel_data = (channel_data + 1.0) / 2.0
        
        channel_data = np.round(channel_data).astype(np.int32)
        unnorm_obs[...,ch_idx] = channel_data
    unnorm_obs = unnorm_obs.clip(0)
    
    return unnorm_obs

def convert_to_binary_obs(obs):
    if isinstance(obs, np.ndarray):
        # For numpy arrays
        preprocessed_obs = np.where(obs < 0, -1.0, 1.0)
    elif isinstance(obs, th.Tensor):
        # For PyTorch tensors
        preprocessed_obs = th.where(obs < 0, -1.0, 1.0)
        
    return preprocessed_obs

def to_np(x):
	if th.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def to_torch(x, dtype=None, device=None):
    DTYPE = th.float
    DEVICE = 'cuda:0'
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif th.is_tensor(x):
        return x.to(device).type(dtype)
    # elif x.dtype.type is np.str_:
    # 	return torch.tensor(x, device=device)
    return th.tensor(x, dtype=dtype, device=device)

def get_idm_action(current_obs, next_obs, idm_model):
    current_obs = convert_to_binary_obs(current_obs)
    next_obs = convert_to_binary_obs(next_obs)
    
    assert th.all((current_obs == -1) | (current_obs == 1)), "obs contains values other than -1 or 1"
    assert th.all((next_obs == -1) | (next_obs == 1)), "obs contains values other than -1 or 1"
    with th.no_grad():
        logits = idm_model(current_obs, next_obs)
        action = th.argmax(logits, dim=1)
    return action

@contextmanager
def managed_model_loading(runner, model_path, ema=True, num_classes=None):
    """Context manager for automatic model cleanup."""
    model = None
    try:
        model = runner.load_diffusion_model(model_path, ema, num_classes)
        yield model
    finally:
        if model is not None:
            del model
        gc.collect()
        th.cuda.empty_cache()
        

@contextmanager
def managed_environment(args, policy_name, n_envs):
    """Context manager for environment cleanup."""
    envs = None
    policy = None
    
    try:
        # Setup environment and policy
        policy, _ = load_partner_policy(args, policy_name, device="cpu")
        is_bc = policy_name in ["bc_train", "bc_test"]
        envs = make_eval_env(args, run_dir="", nenvs=n_envs)
        if is_bc:
            envs.reset_featurize_type([("ppo", "bc") for _ in range(n_envs)])
        
        yield envs, policy
        
    finally:
        if envs is not None:
            envs.close()
        del policy, envs
        gc.collect()

@contextmanager
def managed_concept_trainer(runner, cl_args):
    """Context manager for concept trainer cleanup with dataset reuse."""
    from experiments_classes import ConceptLearnExperiment
    trainer = None
    
    try:
        print(f"Creating concept trainer with pre-loaded datasets...")
        
        # Get or create initial embedding for this policy
        pid = cl_args.new_policy_id
        embedding_obj = runner.get_or_create_embedding(pid)
        init_emb = embedding_obj['embedding']
        init_w = embedding_obj['guidance_weight']

        # Get cached datasets first
        if cl_args.target_episode_idx is not None and cl_args.target_policy_name:
            # Load base dataset once
            base_dataset = runner._get_dataset(cl_args.dataset_path, split=cl_args.dataset_split)
            
            # Get cached single episode dataset
            single_ep_dataset = SingleEpisodeOvercookedDataset(
                base_dataset, 
                cl_args.target_policy_name, 
                cl_args.target_episode_idx
            )
            # Create trainer
            trainer = ConceptLearnExperiment(
                args=cl_args,
                observation_dim=base_dataset.observation_dim,
                embedding=init_emb,
                guidance_weight=init_w,
                num_concepts=runner.num_concepts,
                train_dataset=single_ep_dataset,
                device=runner.device,
            )
            embed, guidance_weight, metrics = trainer.train()
        else:
            raise NotImplementedError(
                "Managed concept trainer only supports single episode datasets for now"
            )
        
        yield embed, guidance_weight, metrics
        
    finally:
        if trainer is not None:
            # Don't delete datasets - keep them cached
            trainer.dataset = None
            trainer.train_dataset = None
            trainer.valid_dataset = None
            single_ep_dataset.base_dataset = None
            del trainer.trainer
            del trainer
            del single_ep_dataset
        gc.collect()
        th.cuda.empty_cache()
        print("Concept trainer cleanup complete (datasets preserved)")