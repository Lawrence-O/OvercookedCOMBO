import argparse
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


class ExperimentRunner:
    def __init__(self, args, use_successive_models=True):
        """
        Initialize the experiment runner with configuration parameters.
        
        Args:
            args: Parsed command line arguments
            use_successive_models: Whether to use the model from the previous step
        """
        self.args = args

        # Disable WandB and Debugging
        self.args.wandb = False
        self.args.debug = False
        self.args.wandb_project = None
        self.args.wandb_run_name = None
        self.args.wandb_entity = None
        self.args.wandb_group = None
        
        self.use_successive_models = use_successive_models
        self.device = th.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")
        print(f"Using device: {self.device}")
        
        # Set up output directories
        self.base_output_dir = Path(args.basedir)
        self.exp_group_dir = self.base_output_dir / (args.exp_group_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.exp_group_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize common resources
        self.idm_model = self._load_idm_model()
        self.renderer = OvercookedSampleRenderer()
        
        # Track experiment state
        self.current_model_path = args.diffusion_model_path
        self.current_args = deepcopy(args)
        self.all_experiment_results = []
        
        # Cache for datasets to avoid reloading
        self.dataset_cache = {}

        
    def _load_idm_model(self):
        """Load the Inverse Dynamics Model."""
        print(f"Loading IDM model from {self.args.idm_path}")
        weights = th.load(self.args.idm_path)
        idm = InverseDynamicsModel(num_actions=6)
        idm.load_state_dict(weights['model'])
        idm.to(self.device)
        idm.eval()
        return idm
    
    def load_diffusion_model(self, model_path, ema=True):
        """Load the diffusion model for generation."""
        args_for_loading = deepcopy(self.current_args)
        args_for_loading.diffusion_model_path = model_path
        
        trainer = OvercookedTrainer(args_for_loading, self.device)
        print(f"Loading diffusion model from {model_path}")
        trainer_actual = trainer.trainer
        trainer_actual.load(model_path)
        
        if ema:
            print("Using EMA model for evaluation")
            return trainer_actual.ema.ema_model.to(self.device)
        else:
            print("Using actual model for evaluation")
            return trainer_actual.model.to(self.device)
    
    def _get_dataset(self, dataset_path, split="test"):
        """Get a dataset from cache or load it."""
        cache_key = f"{dataset_path}_{split}"
        if cache_key not in self.dataset_cache:
            print(f"Loading {split} dataset from {dataset_path}")
            dataset_args = argparse.Namespace(
                dataset_path=dataset_path,
                horizon=self.args.horizon,
                max_path_length=self.args.max_path_length,
                episode_length=self.args.episode_length,
                chunk_length=self.args.chunk_length,
                use_padding=self.args.use_padding,
            )
            self.dataset_cache[cache_key] = OvercookedSequenceDataset(
                args=dataset_args, split=split
            )
        return self.dataset_cache[cache_key]
    
    def _calculate_evaluation_summary(self, episode_rewards_list, all_metrics, 
                                 current_run_basedir, layout_name, agent_id):
        """Calculate summary statistics from episode rewards."""
        if episode_rewards_list:
            all_ep_rewards_np = np.array(episode_rewards_list)

            # Agent 0 statistics
            agent0_mean_rewards_per_episode = all_ep_rewards_np[:, :, 0].mean(axis=1)
            mean_reward_agent0 = agent0_mean_rewards_per_episode.mean()
            std_reward_agent0 = agent0_mean_rewards_per_episode.std()
            
            # Agent 1 statistics
            agent1_mean_rewards_per_episode = all_ep_rewards_np[:, :, 1].mean(axis=1)
            mean_reward_agent1 = agent1_mean_rewards_per_episode.mean()
            std_reward_agent1 = agent1_mean_rewards_per_episode.std()

            # Team statistics
            team_rewards_per_ep_env = all_ep_rewards_np.sum(axis=2)
            mean_team_reward = team_rewards_per_ep_env.mean()
            std_team_reward = team_rewards_per_ep_env.std()
        else:
            mean_reward_agent0 = std_reward_agent0 = mean_reward_agent1 = std_reward_agent1 = mean_team_reward = std_team_reward = 0.0

        # Log results
        print(f"Agent 0 (Diffusion) mean reward: {mean_reward_agent0:.2f} ± {std_reward_agent0:.2f}")
        print(f"Agent 1 (Partner) mean reward: {mean_reward_agent1:.2f} ± {std_reward_agent1:.2f}")
        print(f"Team mean reward: {mean_team_reward:.2f} ± {std_team_reward:.2f}")
        
        # Create summary dictionary
        return {
            'basedir': str(current_run_basedir),
            'agent0_mean_reward': mean_reward_agent0,
            'agent0_std_reward': std_reward_agent0,
            'agent1_mean_reward': mean_reward_agent1,
            'agent1_std_reward': std_reward_agent1,
            'team_mean_reward': mean_team_reward,
            'team_std_reward': std_team_reward,
            'layout_name': layout_name,
            'agent_id': agent_id,
            'total_episodes_evaluated': self.args.exp_eval_episodes * self.args.n_envs,
            'metrics_per_episode_reset': all_metrics,
            'raw_episode_rewards': [arr.tolist() for arr in episode_rewards_list], 
        }
    
    def _save_episode_videos(self, frames, samples_frames, grid, episode, video_dir):
        """Save videos for a completed episode."""
        print(f"Saving videos for episode {episode+1}...")
        for e in range(len(frames)):
            frames[e] = rearrange(frames[e], "f w h c -> f h w c")
            env_dir = video_dir / f"episode_{episode+1}_env_{e+1}"
            env_dir.mkdir(exist_ok=True)
            
            # Save actual trajectory
            saved_video = self.renderer.render_trajectory_video(
                frames[e], 
                grid, 
                output_dir=str(env_dir),
                video_path=str(env_dir / "actual_trajectory.mp4"),
                fps=1
            )
            
            # Save samples trajectory if needed
            if self.args.show_samples:
                self.renderer.render_trajectory_video(
                    samples_frames[e],
                    grid,
                    output_dir=str(env_dir),
                    video_path=str(env_dir / "samples_trajectory.mp4"),
                    fps=1
                )
                
        print(f"Videos saved to {video_dir}")
    
    def load_partner_policy(self, policy_name, device="cpu"):
        """Load a partner policy by name."""
        policy = Policy(None, None, None, None, device=device)
        featurize_type = policy.load_population(self.args.population_yaml_path, evaluation=True)
        policy = policy.policy_pool[policy_name]
        feat_type = featurize_type.get(policy_name, 'ppo')
        return policy, feat_type

    def create_concept_learning_args(self, concept_params):
        """
        Create a new set of arguments for concept learning based on the main_args and concept_params.
        """
        # Start out with a copy of the main args
        cl_args = deepcopy(self.args)
        
        # Override args for concept learning
        cl_args.dataset_path = concept_params['dataset_path']
        cl_args.pretrained_model_path = concept_params['pretrained_model_path']
        cl_args.dummy_policy_id = concept_params['dummy_policy_id']
        cl_args.new_policy_id = concept_params['new_policy_id']
        cl_args.results_dir = concept_params['results_dir_concept_learning']
        cl_args.milestone_name = concept_params['milestone_name']
        cl_args.max_train_steps = concept_params['train_steps']

        cl_args.horizon = concept_params.get('horizon', self.args.horizon) # Use the same horizon as main_args
        cl_args.guidance_weight = concept_params.get('guidance_weight', 1.0)

        # Training Args
        cl_args.train_batch_size = concept_params.get('train_batch_size', self.args.train_batch_size)
        cl_args.num_validation_samples = concept_params.get('num_validation_samples', self.args.num_validation_samples)
        cl_args.save_and_sample_every = concept_params.get('save_and_sample_every', 1000)
        cl_args.cond_drop_prob = concept_params.get('cond_drop_prob', self.args.cond_drop_prob)
        cl_args.split_batches = concept_params.get('split_batches', self.args.split_batches)
        cl_args.save_milestone = concept_params.get('save_milestone', self.args.save_milestone)

        # Dataset args for OvercookedSequenceDataset
        cl_args.max_path_length = self.args.max_path_length
        cl_args.episode_length = self.args.episode_length
        cl_args.chunk_length = self.args.chunk_length
        cl_args.use_padding = self.args.use_padding
        cl_args.dataset_split = concept_params["dataset_split"]

        # Single episode training args
        cl_args.target_episode_idx = concept_params.get('target_episode_idx', None)
        cl_args.target_policy_name = concept_params.get('target_policy_name', None)

        # Disable WandB and Debugging
        cl_args.wandb = False
        cl_args.debug = False
        cl_args.wandb_project = None
        cl_args.wandb_run_name = None
        cl_args.wandb_entity = None
        cl_args.wandb_group = None

        return cl_args
    
    def train_concept_learn(self, cl_args):
        """
        Train a concept learning model for Overcooked using the provided arguments.
        """
        gc.collect()
        th.cuda.empty_cache()

        # Check if we should use a single episode dataset
        if cl_args.target_episode_idx is not None and cl_args.target_policy_name:
            # Load the base dataset
            base_dataset = self._get_dataset(cl_args.dataset_path, split=cl_args.dataset_split)
            
            # Create base datatset
            print(f"Creating single episode dataset: policy '{cl_args.target_policy_name}', episode {cl_args.target_episode_idx}")
            single_ep_dataset = SingleEpisodeOvercookedDataset(
                base_dataset, 
                cl_args.target_policy_name, 
                cl_args.target_episode_idx
            )

            # Create trainer with this dataset
            concept_trainer = ConceptLearnOvercookedTrainer(cl_args, self.device)
            concept_trainer.dataset = single_ep_dataset
            concept_trainer.trainer.train_dataset = single_ep_dataset
            concept_trainer.trainer.valid_dataset = single_ep_dataset
            print(f"Single episode dataset created: {len(single_ep_dataset)} samples")
            
        else:
            # Standard training with OvercookedSequenceDataset   
            concept_trainer = ConceptLearnOvercookedTrainer(cl_args, self.device)
    
        # Run training
        concept_trainer.train()

        # Final Model Path
        model_path = os.path.join(cl_args.results_dir, f"modl-{cl_args.milestone_name}.pt")
        assert os.path.exists(model_path), f"Expected model at {model_path} but not found"
        
        # Clean up trainer explicitly
        del concept_trainer.trainer
        del concept_trainer
        
        gc.collect()
        th.cuda.empty_cache()
        
        return model_path
    
    def concept_learn_one_step(self, 
                           new_policy_id, 
                           milestone_name, 
                           pretrained_model_path,
                           dummy_policy_id,
                           train_steps,
                           eval_partner_policy=None,
                           eval_layout_name=None,
                           eval_horizon=None,
                           experiment_index=0,
                           is_test_split=False,
                           target_episode_idx=None):
        """Create a single concept learning experiment definition."""
        cl_params = {
            'dataset_path': self.args.dataset_path,
            'pretrained_model_path': pretrained_model_path,
            'train_steps': train_steps,
            'max_train_steps': train_steps,  # Ensure both are set
            'dummy_policy_id': dummy_policy_id,
            'new_policy_id': new_policy_id,
            'results_dir_concept_learning': str(self.base_output_dir / f"learned_concepts" / f"concept_{new_policy_id}_training_idx_{experiment_index}"),
            'milestone_name': milestone_name,
            'horizon': eval_horizon if eval_horizon is not None else self.args.horizon,
            'guidance_weight': getattr(self.args, 'guidance_weight', 1.0),
            'dataset_split': "test" if is_test_split else "train",
            'target_episode_idx': target_episode_idx,
            'target_policy_name': eval_partner_policy,
        }

        experiment_definition = {
            "name": f"concept_learn_experiment_id_{new_policy_id}_experiment_idx_{experiment_index}",
            "concept_learning_params": cl_params,
        }

        if eval_partner_policy:
            experiment_definition["evaluation_configs"] = [{
                "layout_name": eval_layout_name if eval_layout_name is not None else self.args.layout_name,
                "agent_id": new_policy_id,
                "horizon": eval_horizon if eval_horizon is not None else self.args.horizon,
                "partner_policy_name": eval_partner_policy,
            }]

        return experiment_definition
    
    def create_test_concept_learn_experiment(self, 
                                         num_concept_runs, 
                                         fine_tuning_steps, 
                                         target_episodes={}, 
                                         test_policies_to_use=None):
        """Create concept learning experiments for test policies."""
        experiment_configs = []
        
        
        # Get test dataset using caching mechanism
        dataset = self._get_dataset(self.args.dataset_path, "test")
        
        if not dataset.test_partner_policies:
            print("No test partner policies found in the dataset.")
            return experiment_configs
            
        test_policies = dataset.test_partner_policies
        print(f"Found {len(test_policies)} test policies: {list(test_policies.keys())}")
        
        # Track the actual policy counter (only for included policies)
        policy_counter = 0
        
        for policy_name in test_policies:
                
            if test_policies_to_use and policy_name not in test_policies_to_use:
                print(f"Skipping policy {policy_name} as it is not in specified test policies.")
                continue
                
            matching_episodes = dataset.get_path_indexes_episode(policy_name)
            if not matching_episodes:
                print(f"Skipping policy {policy_name} as no episodes were found.")
                continue
                
            # Handle target episodes index validation
            if policy_name in target_episodes:
                target_idx = target_episodes[policy_name]
                if target_idx >= len(matching_episodes):
                    target_idx = target_idx % len(matching_episodes)
                    print(f"Target index {target_idx} for policy {policy_name} out of bounds. Wrapping to {target_idx}.")
                    target_episodes[policy_name] = target_idx
                    
            print(f"Found {len(matching_episodes)} episodes for policy '{policy_name}'")
            
            for run_idx in range(num_concept_runs):
                base_episode_idx = target_episodes.get(policy_name, 0)
                episode_idx = (base_episode_idx + run_idx) % len(matching_episodes)
                
                # Use policy_counter for sequential experiment indices
                experiment_idx = policy_counter * num_concept_runs + run_idx
                print(f"Using episode {episode_idx} for {policy_name} run {run_idx} experiment {experiment_idx}.")
                
                milestone_name = f"test_concept_{policy_name}_run_{run_idx}_episode_{episode_idx}"
                
                # Create experiment config
                experiment_configs.append(
                    self.concept_learn_one_step(
                        new_policy_id=test_policies[policy_name],
                        milestone_name=milestone_name,
                        pretrained_model_path=self.args.diffusion_model_path,
                        dummy_policy_id=dataset.dummy_id,
                        train_steps=fine_tuning_steps,
                        eval_partner_policy=policy_name,
                        experiment_index=experiment_idx,
                        is_test_split=True,
                        target_episode_idx=episode_idx,
                    )
                )
                
            # Increment policy counter for successful processing
            policy_counter += 1
            
        print(f"Generated {len(experiment_configs)} test concept learning experiments.")
        return experiment_configs
    
    @th.no_grad()
    def full_horizon_eval(self, diffusion_model, policy, config):
        """
        Run full horizon evaluation with the given model and configuration.
        """
        agent_id = config["agent_id"]
        layout_name = config["layout_name"]
        horizon = config["horizon"]
        partner_policy_name = config["partner_policy_name"]
        
        # Create a unique basedir for this experiment
        current_experiment_name = f"{layout_name}_agent_id_{agent_id}_horizon_{horizon}_partner_policy_{partner_policy_name}"
        current_run_basedir = self.exp_group_dir / "evaluation" / current_experiment_name
        current_run_basedir.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluating: Layout={layout_name}, Agent ID={agent_id}, Horizon={horizon}, Partner={partner_policy_name}")
        
        # Setup directories using pathlib consistently
        video_dir = current_run_basedir / "videos"
        frames_dir = current_run_basedir / "frames" 
        metrics_dir = current_run_basedir / "metrics"
        video_dir.mkdir(exist_ok=True)
        frames_dir.mkdir(exist_ok=True)
        metrics_dir.mkdir(exist_ok=True)

        # Set environment variables
        os.environ["layout"] = layout_name 
        
        # Create environments
        is_bc = partner_policy_name in ["bc_train", "bc_test"]
        if is_bc:
            # Override old dynamics for bc evaluation specifically
            self.args.old_dynamics = True
            envs = make_eval_env(self.args, run_dir=self.args.run_dir, nenvs=self.args.n_envs)
            envs.reset_featurize_type([("ppo", "bc") for _ in range(self.args.n_envs)])
        else:
            envs = make_eval_env(self.args, run_dir=self.args.run_dir, nenvs=self.args.n_envs)
            
        all_metrics = []
        episode_rewards_list = []
        
        F = 32  # Number of frames in the diffusion horizon
        cond = th.full((self.args.n_envs,), agent_id, dtype=th.int64, device=self.device)
        for episode in range(self.args.exp_eval_episodes):
            print(f"Starting episode {episode+1}/{self.args.exp_eval_episodes}")

            # Reset the environment and policy
            policy.reset(num_envs=self.args.n_envs, num_agents=2)
            for e in range(self.args.n_envs):
                policy.register_control_agent(e=e, a=1)

            obs, _, _ = envs.reset([True] * self.args.n_envs)
            
            H, W, C = obs[0][0].shape[0], obs[0][0].shape[1], obs[0][0].shape[-1] 

            # Initialize variables for the episode
            steps = 0
            done = False
            episode_reward = np.zeros((self.args.n_envs, 2))


            frames = [[obs[i][0]] for i in range(self.args.n_envs)]
            samples_frames = [[] for _ in range(self.args.n_envs)]

            if self.args.save_videos:
                grid = np.transpose(obs[0][0], (1, 0, 2))  # "w h c" -> "h w c"
                grid = self.renderer.extract_grid_from_obs(grid)

            step_actions = np.zeros((self.args.n_envs, 2, 1), dtype=np.int64)
            
            while not done and steps <= self.args.max_steps:
                # Setup Condition Obs Based on Obs
                obs_stack = np.stack([normalize_obs(obs[e][0]) for e in range(self.args.n_envs)], axis=0) 
                condition_obs = th.tensor(obs_stack, device=self.device, dtype=th.float32)
                condition_obs = rearrange(condition_obs, "b h w c -> b c h w")

                samples = diffusion_model.sample(
                    x_cond=condition_obs,
                    task_embed=cond,
                    batch_size=self.args.n_envs,
                )
                    
                samples = rearrange(samples, "b (f c) h w -> b f h w c", c=C, f=F)
                 
                # Render out Samples
                if self.args.show_samples and episode == 0 and steps % 50 == 0:
                    self.renderer.visualize_all_channels(
                        obs=samples[0, 0].cpu().numpy(), 
                        output_dir=str(frames_dir / f"samples_channels_steps_{steps}_horizon_0_env_0.png")
                    )
                     
                # Now step through the environment using the plan
                plan_horizon = min(self.args.max_steps - steps, horizon)

                # We begin with the first ego obs (first obs of the environment)
                obs_t = to_torch(obs_stack, device=self.device)
                for t in range(plan_horizon): 
                    obs_tp1 = samples[:, t, ...]
                    
                    if self.args.save_videos:
                        for e in range(self.args.n_envs):
                            samples_frames[e].append(obs_tp1[e].cpu().numpy())

                    for env_i in range(self.args.n_envs):
                        # IDM takes in 26 channels
                        ego_action = get_idm_action(
                            to_torch(obs_t[env_i], device=self.device).unsqueeze(0), 
                            to_torch(obs_tp1[env_i], device=self.device).unsqueeze(0), 
                            self.idm_model
                        )
                        step_actions[env_i, 0, 0] = ego_action
      
                    partner_obs_lst = [obs[e][1] for e in range(self.args.n_envs)]
                    partner_obs = np.stack(partner_obs_lst, axis=0)
                    partner_action = policy.step(
                        partner_obs,
                        [(e, 1) for e in range(self.args.n_envs)],
                        deterministic=True,
                    )
                    step_actions[:, 1] = partner_action

                    # Take environment step
                    obs, _, reward, done, _, _ = envs.step(step_actions)
                    episode_reward += reward.squeeze(axis=2)
                    obs_t = obs_tp1
                    
                    if self.args.save_videos:
                        for e in range(min(self.args.n_envs, 3)):
                            frames[e].append(obs[e][0])

                    # Check for early termination
                    done = np.all(done)
                    steps += 1
                    if steps >= self.args.max_steps:
                        break

            # Process episode results
            mean_episode_reward_per_env = episode_reward.mean(axis=0)
            print(f"Episode {episode+1} complete: steps={steps}, mean rewards per agent={mean_episode_reward_per_env}")
            
            metrics = {
                'episode': episode,
                'steps': steps,
                'rewards_per_agent_total': episode_reward.tolist(), 
                'mean_reward_across_envs': mean_episode_reward_per_env.tolist(),
            }
            all_metrics.append(metrics)
            episode_rewards_list.append(episode_reward.copy()) # Store rewards for all envs for this episode
            
            # Save episode metrics
            with open(metrics_dir / f"episode_{episode+1}_metrics.pkl", 'wb') as f:
                pickle.dump(metrics, f)
            
            if self.args.save_videos:
                self._save_episode_videos(
                    frames, samples_frames, grid, episode, video_dir
                )

            # Clean temporary storage
            del frames, samples_frames
            th.cuda.empty_cache() 
        
        # Calculate final metrics
        summary = self._calculate_evaluation_summary(
            episode_rewards_list, all_metrics, current_run_basedir,
            layout_name, agent_id
        )
        
        # Save metrics
        metrics_path = current_run_basedir / "eval_summary.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(summary, f)
        print(f"Summary metrics saved to {metrics_path}")
                        
        envs.close()
        return summary
    
    def evaluate(self, config, diffusion_model):
        """Run evaluation based on the provided configuration."""
        partner_policy_obj, _ = self.load_partner_policy(
            config["partner_policy_name"],
            device="cpu"
        )
        
        summary = self.full_horizon_eval(
            diffusion_model=diffusion_model,
            policy=partner_policy_obj,
            config=config
        )
        
        # Add concept learning metadata if present
        if "concept_learning_params" in config and "target_episode_idx" in config["concept_learning_params"]:
            summary['training_episode_idx'] = config["concept_learning_params"]["target_episode_idx"]
            
        # Add experiment metadata
        summary['experiment_horizon_val'] = config["horizon"]
        summary['experiment_agent_id_val'] = config["agent_id"]
        summary['experiment_layout_name_val'] = config["layout_name"]
        summary['experiment_partner_policy_val'] = config["partner_policy_name"]
            
        return summary
    
    
    def run_experiments(self):
        """Run all experiments."""
        # Initialize experiment configs
        experiment_configs = self.create_test_concept_learn_experiment(
            num_concept_runs=self.args.num_concept_runs,
            fine_tuning_steps=self.args.test_concept_train_steps,
            test_policies_to_use=self.args.test_policies_to_use,
            target_episodes={}  # Start with empty, can be configured
        )
        
        assert experiment_configs, "No experiment configurations generated"
        
        for exp_idx, exp_config in enumerate(experiment_configs):
            # Clean up memory
            gc.collect()
            th.cuda.empty_cache()
            
            print(f"\n=== Running Experiment {exp_config['name']} ===")
            
            # Create experiment directory
            exp_dir = self.exp_group_dir / exp_config["name"]
            exp_dir.mkdir(exist_ok=True)
            
            # Initialize variables for this experiment
            use_ema = "concept_learning_params" not in exp_config
            model_path_for_eval = self.current_model_path
            experiment_results = []
            
            # Conduct concept learning if needed
            if "concept_learning_params" in exp_config:
                cl_params = exp_config["concept_learning_params"]
                
                # Update model path for successive models
                if self.use_successive_models and exp_idx > 0:
                    cl_params['pretrained_model_path'] = self.current_model_path
                    print(f"Using model from previous step: {self.current_model_path}")
                
                # Create args and set results dir
                cl_args = self.create_concept_learning_args(cl_params)
                cl_args.results_dir = str(exp_dir / "concept_learning_output")
                
                # Train concept model
                print(f"Starting concept learning for {exp_config['name']}...")
                concept_model_path = self.train_concept_learn(cl_args)
                print(f"Concept learning complete. Model saved to {concept_model_path}")
                
                # Update current model path for successive runs
                if self.use_successive_models:
                    self.current_model_path = concept_model_path
                    self.current_args = deepcopy(cl_args)
                
                model_path_for_eval = concept_model_path
            
            # Load diffusion model for evaluation
            print(f"Loading model for evaluation: {model_path_for_eval}")
            diffusion_model = self.load_diffusion_model(model_path_for_eval, ema=use_ema)
            
            # Run evaluations
            eval_configs = exp_config.get("evaluation_configs", [])
            if not isinstance(eval_configs, list):
                eval_configs = [eval_configs] if eval_configs else []
                
            for eval_config in eval_configs:
                if eval_config:
                    result = self.evaluate(eval_config, diffusion_model)
                    experiment_results.append(result)
            
            # Save results for this experiment
            self.all_experiment_results.append({
                "name": exp_config["name"],
                "config": exp_config,
                "results": experiment_results,
                "model_path": model_path_for_eval
            })
            
            # Clean up model to save memory
            del diffusion_model
            gc.collect()
            th.cuda.empty_cache()
            
            # Save intermediate results
            with open(self.exp_group_dir / f"results_through_exp_{exp_idx}.pkl", "wb") as f:
                pickle.dump(self.all_experiment_results, f)
        
        # Save final results
        results_file = self.exp_group_dir / "all_experiment_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(self.all_experiment_results, f)
            
        print(f"\nAll experiments complete. Results saved to {results_file}")
        return self.all_experiment_results
    
    def run_base_model_evaluation(self, policies_to_evaluate=None):
        """
        Run comprehensive evaluations of the base diffusion model on both training and test policies.
        This provides baseline metrics for comparison with concept learning results.
        """
        print("\n=== Running Base Model Evaluation ===")
        
        # Combined list of experiment configs
        experiment_configs = []
        
        included_policies = set()
        dataset = self._get_dataset(self.args.dataset_path, "train")
        if dataset.train_partner_policies:
            train_policies = dataset.train_partner_policies
            print(f"Found {len(train_policies)} train policies: {list(train_policies.keys())}")
            
            # Filter policies if requested
            for policy_name in train_policies:
                # Skip if not in the requested list (when a list is provided)
                if policies_to_evaluate is not None and policy_name not in policies_to_evaluate:
                    print(f"Skipping train policy {policy_name} (not in requested list)")
                    continue
                    
                included_policies.add(policy_name)
                experiment_configs.append({
                    "name": f"base_model_eval_train_policy_{policy_name}",
                    "evaluation_configs": [{
                        "layout_name": self.args.layout_name,
                        "agent_id": train_policies[policy_name],
                        "horizon": self.args.horizon,
                        "partner_policy_name": policy_name,
                    }]
                })
            print(f"Created {len(included_policies)} training policy evaluations")
        
        # Early exit if no policies to evaluate
        if not experiment_configs:
            print("No policies found for evaluation")
            return []
        
        # Load base diffusion model once for all evaluations
        print(f"Loading base diffusion model from {self.args.diffusion_model_path}")
        diffusion_model = self.load_diffusion_model(self.args.diffusion_model_path, ema=True)
        
        # Create a subdirectory for base model evaluations
        base_model_dir = self.exp_group_dir / "base_model_evaluations"
        base_model_dir.mkdir(exist_ok=True)
        
        # Run evaluations for each policy 
        for exp_idx, exp_config in enumerate(experiment_configs):
            print(f"\n=== Evaluating Base Model on {exp_config['name']} ===")
            
            # Create experiment directory
            exp_dir = base_model_dir / exp_config["name"]
            exp_dir.mkdir(exist_ok=True)
            
            # Run evaluations
            experiment_results = []
            eval_configs = exp_config.get("evaluation_configs", [])
            
            for eval_config in eval_configs:
                if eval_config:
                    result = self.evaluate(eval_config, diffusion_model)
                    # Mark as base model evaluation
                    result['is_base_model'] = True  
                    experiment_results.append(result)
            
            # Save results for this experiment
            self.all_experiment_results.append({
                "name": exp_config["name"],
                "config": exp_config,
                "results": experiment_results,
                "model_path": self.args.diffusion_model_path,
                "is_base_model": True
            })
            
            # Save intermediate results
            with open(base_model_dir / f"results_through_exp_{exp_idx}.pkl", "wb") as f:
                pickle.dump(self.all_experiment_results, f)
        
        # Clean up model
        del diffusion_model
        gc.collect()
        th.cuda.empty_cache()
        
        # Save overall results
        results_file = base_model_dir / "base_model_evaluation_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(self.all_experiment_results, f)
        
        # Generate plots
        self.plot_base_model_results()
        
        print(f"\nBase model evaluation complete. Results saved to {results_file}")
        return self.all_experiment_results
    
    def plot_horizon_evaluation_results(self, df, output_dir):
        """Plot results for horizon evaluations."""
        plt.figure(figsize=(12, 7))

        x_values = df['experiment_horizon_val'].to_numpy()
        agent0_mean = df['agent0_mean_reward'].to_numpy()
        agent0_std = df['agent0_std_reward'].to_numpy()
        agent1_mean = df['agent1_mean_reward'].to_numpy()
        agent1_std = df['agent1_std_reward'].to_numpy()
        team_mean = df['team_mean_reward'].to_numpy()
        team_std = df['team_std_reward'].to_numpy()

        plt.plot(x_values, agent0_mean, label='Agent 0 (Diffusion)', marker='o')
        plt.fill_between(x_values, 
                        agent0_mean - agent0_std, 
                        agent0_mean + agent0_std, 
                        alpha=0.2)
        
        plt.plot(x_values, agent1_mean, label='Agent 1 (Partner)', marker='o')
        plt.fill_between(x_values, 
                        agent1_mean - agent1_std, 
                        agent1_mean + agent1_std, 
                        alpha=0.2)
                        
        plt.plot(x_values, team_mean, label='Team Reward', marker='s', linestyle='--')
        plt.fill_between(x_values,
                            team_mean - team_std,
                            team_mean + team_std,
                            alpha=0.2)

        plt.xlabel('Planning Horizon Value')
        plt.ylabel('Mean Reward')
        plt.title('Horizon Evaluation Results')
        plt.legend()
        plt.grid(alpha=0.3)
        
        output_path = os.path.join(output_dir, "horizon_evaluation_results.png")
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        plt.close()

    def plot_episode_evaluation_results(self, df, output_dir):
        """Plot results for episode-based evaluations."""
        if "training_episode_idx" not in df.columns:
            print("DataFrame does not contain 'training_episode_idx' column. Skipping episode plots.")
            return
        
        for policy_name, group in df.groupby('experiment_partner_policy_val'):
            plt.figure(figsize=(12, 7))

            # Sort by episode index
            group = group.sort_values('training_episode_idx')

            # Convert pandas Series to numpy arrays
            x = group['training_episode_idx'].to_numpy()
            agent0_mean = group['agent0_mean_reward'].to_numpy()
            agent0_std = group['agent0_std_reward'].to_numpy()
            agent1_mean = group['agent1_mean_reward'].to_numpy()
            agent1_std = group['agent1_std_reward'].to_numpy()
            team_mean = group['team_mean_reward'].to_numpy()
            team_std = group['team_std_reward'].to_numpy()
            
            # Plot agent rewards
            plt.plot(x, agent0_mean, marker='o', label='Agent 0 (Diffusion)', linewidth=2)
            plt.fill_between(x, agent0_mean - agent0_std, agent0_mean + agent0_std, alpha=0.2)
            
            plt.plot(x, agent1_mean, marker='s', label='Agent 1 (Partner)', linewidth=2)
            plt.fill_between(x, agent1_mean - agent1_std, agent1_mean + agent1_std, alpha=0.2)
            
            # Plot team rewards
            plt.plot(x, team_mean, marker='^', label='Team Reward', linestyle='--', linewidth=2)
            plt.fill_between(x, team_mean - team_std, team_mean + team_std, alpha=0.2)
            
            plt.title(f'Performance vs. Training Episode for Policy: {policy_name}')
            plt.xlabel('Training Episode Index')
            plt.ylabel('Reward')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Save the policy-specific plot
            plt.savefig(os.path.join(output_dir, f"policy_{policy_name}_episodes.png"))
            plt.close()
        
        # Create comparative plot across policies
        plt.figure(figsize=(14, 8))
        markers = ['o', 's', '^', 'D', '*', 'x']
        colors = plt.cm.tab10.colors
        
        for i, (policy_name, group) in enumerate(df.groupby('experiment_partner_policy_val')):
            # Sort by episode
            group = group.sort_values('training_episode_idx')
            
            # Convert to numpy arrays
            x = group['training_episode_idx'].to_numpy()
            y = group['team_mean_reward'].to_numpy()
            std = group['team_std_reward'].to_numpy()
            
            plt.plot(x, y, 
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)],
                    label=f'Policy: {policy_name}',
                    linewidth=2)
            plt.fill_between(x, 
                            y - std,  
                            y + std,  
                            color=colors[i % len(colors)],
                            alpha=0.2)  
        
        plt.title('Team Reward vs. Training Episode Across Policies')
        plt.xlabel('Training Episode Index')
        plt.ylabel('Team Reward')
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Save overall comparison
        plt.savefig(os.path.join(output_dir, "all_policies_episode_comparison.png"))
        plt.close()
        
    def plot_results(self):
        """Generate plots from experiment results, focusing only on training policies."""
        # Create plot directory
        plot_dir = self.exp_group_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Extract results for plotting, filtering for training policies only
        plot_results = []
        for exp in self.all_experiment_results:

            if "results" in exp and exp["results"]:
                # Extract config metadata
                config = exp.get("config", {})
                cl_params = config.get("concept_learning_params", {})
                
                # Extract important metadata
                milestone_name = cl_params.get("milestone_name", "")
                target_episode_idx = cl_params.get("target_episode_idx", None)
                
                for res in exp["results"]:
                    # Create a copy to avoid modifying original
                    res_copy = res.copy()
                    
                    # Add metadata for plotting
                    res_copy["milestone_name"] = milestone_name
                    if "training_episode_idx" not in res_copy and target_episode_idx is not None:
                        res_copy["training_episode_idx"] = target_episode_idx
                    
                    # Add explicit marker that this is a training policy
                    res_copy["is_training_policy"] = True
                    
                    plot_results.append(res_copy)
        
        if plot_results:
            # Convert to DataFrame and determine plot type
            df = pd.DataFrame(plot_results)
            
            print(f"Plotting results for {len(df)} training policy evaluations")
            
            if "experiment_horizon_val" in df.columns and len(df["experiment_horizon_val"].unique()) > 1:
                # Horizon experiment
                self.plot_horizon_evaluation_results(df, plot_dir)
                print(f"Generated horizon plots for training policies in {plot_dir}")
                
            if "training_episode_idx" in df.columns:
                # Episode experiment 
                self.plot_episode_evaluation_results(df, plot_dir)
                print(f"Generated episode plots for training policies in {plot_dir}")

            
        else:
            print("No training policy evaluation results to plot")
    def plot_base_model_results(self):
        """Generate plots specifically for base model evaluation results."""
        # Create plot directory
        plot_dir = self.exp_group_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Extract base model evaluation results
        base_results = []
        for exp in self.all_experiment_results:
            if exp.get("is_base_model", False):
                for res in exp.get("results", []):
                    # Add the policy name from the experiment name for better readability
                    policy_name = exp["name"].replace("base_model_eval_train_policy_", "")
                    res_copy = res.copy()
                    res_copy["policy_name"] = policy_name
                    base_results.append(res_copy)
        
        if not base_results:
            print("No base model evaluation results to plot")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(base_results)
        
        # Create summary bar chart
        plt.figure(figsize=(14, 8))
        
        # Sort by team reward for better readability
        df = df.sort_values('team_mean_reward', ascending=False)
        
        # Extract policy names and metrics
        policies = df['experiment_partner_policy_val'].tolist()
        agent0_means = df['agent0_mean_reward'].tolist()
        agent0_stds = df['agent0_std_reward'].tolist()
        agent1_means = df['agent1_mean_reward'].tolist()
        agent1_stds = df['agent1_std_reward'].tolist()
        team_means = df['team_mean_reward'].tolist()
        team_stds = df['team_std_reward'].tolist()
        
        # Set up bar positions
        bar_width = 0.25
        r1 = np.arange(len(policies))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create grouped bar chart
        plt.bar(r1, agent0_means, yerr=agent0_stds, width=bar_width, label='Agent 0 (Diffusion)', capsize=7, color='skyblue')
        plt.bar(r2, agent1_means, yerr=agent1_stds, width=bar_width, label='Agent 1 (Partner)', capsize=7, color='lightgreen')
        plt.bar(r3, team_means, yerr=team_stds, width=bar_width, label='Team Total', capsize=7, color='coral')
        
        # Add labels and title
        plt.xlabel('Partner Policy')
        plt.ylabel('Mean Reward')
        plt.title('Base Model Performance Across Training Policies')
        plt.xticks([r + bar_width for r in range(len(policies))], policies, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(plot_dir, "base_model_performance.png")
        plt.savefig(output_path)
        print(f"Base model performance plot saved to {output_path}")
        plt.close()

def parse_args(args, parser):
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--results_dir', type=str, default='./overcooked_results', help='Directory to save results and checkpoints')
    parser.add_argument('--basedir', type=str, default='./experiment_results', help='Directory to save results and checkpoints')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (smaller dataset, faster training)')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Overcooked HDF5 dataset')
    parser.add_argument('--horizon', type=int, default=32, help='Sequence horizon for trajectories')
    parser.add_argument('--save_milestone', type=bool, default=True, help='Save milestones with step number in filename') # Or action='store_true'

    # For OvercookedSequenceDataset / HDF5Dataset
    parser.add_argument('--max_path_length', type=int, default=401, help='Maximum path length in episodes (for dataset indexing)')
    parser.add_argument('--chunk_length', type=int, default=None, help='Chunk length for HDF5Dataset (defaults to horizon if None, set via dataset_constructor_args)')
    parser.add_argument('--use_padding', type=bool, default=True, help='Whether to use padding for shorter sequences in dataset')


    # For GoalGaussianDiffusion (configurable ones)
    parser.add_argument('--timesteps', type=int, default=400, help='Number of diffusion timesteps for training (if not debug)')
    parser.add_argument('--sampling_timesteps', type=int, default=10, help='Number of timesteps for DDIM sampling (if not debug)')

    # For OvercookedEnvTrainer 
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size (if not debug)')
    parser.add_argument('--num_validation_samples', type=int, default=4, help='Number of samples to generate during validation step')
    parser.add_argument('--save_and_sample_every', type=int, default=1000, help='Frequency to save checkpoints and generate samples (if not debug)')
    parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Probability of dropping condition for CFG during training')
    parser.add_argument('--split_batches', type=bool, default=True, help='Whether to split batches for Accelerator')
    parser.add_argument('--resume_checkpoint_path', type=str, required=False, default=None, help='Path to a .pt checkpoint file to resume training from.')
    
    # overcooked evaluation
    parser.add_argument("--diffusion_model_path", type=str, required=True, help="Path to the diffusion model directory")
    parser.add_argument("--dataset", type=str, default="overcooked", help="Dataset name")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--agent_id", type=int, default=0, help="Agent ID for conditioning")
    parser.add_argument("--max_steps", type=int, default=400, help="Maximum steps per episode")
    parser.add_argument("--run_dir", type=str, default="eval_run", help="Directory for evaluation run")
    parser.add_argument("--idm_path", type=str, required=True, help="Path to the diffusion model directory")
    parser.add_argument("--exp_eval_episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--show_samples", default=False, action='store_true', help="Whether to visualize samples during evaluation")
    parser.add_argument("--save_videos", default=False, action='store_true', help="Whether to save videos of the evaluation")
    
    # Mapt Package Args  
    parser.add_argument("--old_dynamics", default=False, action='store_true', help="old_dynamics in mdp")
    parser.add_argument("--layout_name", type=str, default='counter_circuit_o_1order', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int, default=1, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")  
    parser.add_argument("--use_hsp", default=False, action='store_true')   
    parser.add_argument("--random_index", default=False, action='store_true')
    parser.add_argument("--use_agent_policy_id", default=False, action='store_true', help="Add policy id into share obs, default False")
    parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
    parser.add_argument("--use_detailed_rew_shaping", default=False, action='store_true')
    parser.add_argument("--random_start_prob", default=0., type=float)
    parser.add_argument("--store_traj", default=False, action='store_true')
    # population
    parser.add_argument("--population_yaml_path", type=str, help="Path to yaml file that stores the population info.")
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Fraction of data to use for validation (default: 0.1)')
    
    # Concept Learning Args
    parser.add_argument('--test_policies_to_use', nargs='+', type=str, default=None,
                   help='Specific test policies to use for concept learning')
    parser.add_argument('--num_concept_runs', type=int, required=True,
                    help='Number of runs (total times to iterate over data) for each concept learning experiment')
    parser.add_argument('--test_concept_train_steps', type=int, required=True,
                    help='Number of training steps to take for test concept learning')
    parser.add_argument('--exp_group_name', type=str, default=None, 
                    help='Name for experiment group folder')
    
    all_args = parser.parse_known_args(args)[0]

    

    return all_args

if __name__ == "__main__":
    parser = get_config()
    args = sys.argv[1:]
    args = parse_args(args, parser)

    #overrde episode len
    args.episode_length = 400
    runner = ExperimentRunner(args)
    runner.run_base_model_evaluation(policies_to_evaluate={
        "sp1_final",
        "sp2_final",
        "sp3_final",
        "sp4_final",
        "sp5_final",
    })

    