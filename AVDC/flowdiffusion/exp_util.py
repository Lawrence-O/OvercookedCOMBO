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
from matplotlib import pyplot as plt
import pandas as pd
from overcooked_dataset import OvercookedSequenceDataset, SingleEpisodeOvercookedDataset

import numpy as np
import torch as th
import os
import os.path as osp
import warnings
import torch.nn.functional as F
import pickle
warnings.filterwarnings("ignore")
from idm.inverse_dynamics import InverseDynamicsModel
from mapbt_package.mapbt.envs.overcooked.Overcooked_Env import Overcooked
from mapbt_package.mapbt.envs.env_wrappers import ChooseSubprocVecEnv
from mapbt_package.mapbt.algorithms.population.policy_pool import PolicyPool as Policy
from overcooked_sample_renderer import OvercookedSampleRenderer
from einops.einops import rearrange
from train_overcooked import OvercookedTrainer
from mapbt_package.mapbt.config import get_config
from learn_concept import ConceptLearnOvercookedTrainer


def make_eval_env(all_args, run_dir, nenvs=3):
    def get_env_fn(rank):
        def init_env():
            env = Overcooked(all_args, run_dir, rank=rank)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(nenvs)])

def arg_max(obs):
    # Assume Obs -> [Batch, Horizon, H, W, C]
    if obs.dim() != 5:
        raise ValueError(f"Expected 5D input (B, T, H, W, C), got {obs.shape}")
    
    B, T, H, W, C = obs.shape

    # Flatten spatial dimensions [B, T, C, H*W]
    flat = rearrange(obs, "b t h w c -> b t c (h w)")

    # Get max indices along spatial dimension
    _, max_idxs = flat.max(dim=-1)  # [B, T, C]

    # Directly scatter 1.0 at max positions
    flat_mask = th.ones_like(flat)*-1
    flat_mask.scatter_(-1, max_idxs.unsqueeze(-1), 1.0)  
    # Reshape back to original dimensions
    peaks = rearrange(flat_mask, "b t c (h w) -> b t h w c", h=H, w=W)

    return peaks

def normalize_obs(obs):
    obs_max = obs.max(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
    obs_min = obs.min(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
    obs_norm = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-8) - 1
    return obs_norm.astype(np.float32)

def preprocess_for_idm(obs):
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
    current_obs = preprocess_for_idm(current_obs)
    next_obs = preprocess_for_idm(next_obs)
    
    assert th.all((current_obs == -1) | (current_obs == 1)), "obs contains values other than -1 or 1"
    assert th.all((next_obs == -1) | (next_obs == 1)), "obs contains values other than -1 or 1"

    with th.no_grad():
        logits = idm_model(current_obs, next_obs)
        probs = F.softmax(logits, dim=1)
        action = th.argmax(probs)
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
def managed_environment(runner, config, is_bc_fn):
    """Context manager for environment cleanup."""
    envs = None
    policy = None
    
    try:
        # Setup environment and policy
        policy, _ = runner.load_partner_policy(config["partner_policy_name"], device="cpu")
        
        is_bc = is_bc_fn(config["partner_policy_name"])
        if is_bc:
            runner.args.old_dynamics = True
            envs = make_eval_env(runner.args, run_dir=runner.args.run_dir, nenvs=runner.args.n_envs)
            envs.reset_featurize_type([("ppo", "bc") for _ in range(runner.args.n_envs)])
        else:
            envs = make_eval_env(runner.args, run_dir=runner.args.run_dir, nenvs=runner.args.n_envs)
        yield envs, policy
        
    finally:
        if envs is not None:
            envs.close()
        del policy, envs
        gc.collect()

@contextmanager
def managed_concept_trainer(runner, cl_args):
    """Context manager for concept trainer cleanup with dataset reuse."""
    trainer = None
    
    try:
        print(f"Creating concept trainer with pre-loaded datasets...")
        
        # Get or create initial embedding for this policy
        pid = cl_args.new_policy_id
        init_emb = runner.get_or_create_embedding(pid)
        cl_args.initial_embedding = init_emb
        
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

            # Create trainer with pre-loaded datasets 
            trainer = ConceptLearnOvercookedTrainer(
                cl_args, 
                runner.device,
                dataset=base_dataset,
            )
            trainer.train_dataset = single_ep_dataset
            trainer.valid_dataset = single_ep_dataset
            print(f"Created trainer with single episode dataset: {len(single_ep_dataset)} samples")
            
        else:
            base_dataset = runner._get_dataset(cl_args.dataset_path, split=cl_args.dataset_split)
            
            # Create trainer with pre-loaded dataset - NO dataset loading in __init__
            trainer = ConceptLearnOvercookedTrainer(
                cl_args,
                runner.device, 
                dataset=base_dataset,
                train_dataset=base_dataset,
                valid_dataset=base_dataset
            )
            print(f"Created trainer with base dataset: {len(base_dataset)} samples")
        
        yield trainer, init_emb
        
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