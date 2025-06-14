import argparse
import gc
from copy import deepcopy
import datetime
from pathlib import Path
import sys
from matplotlib import pyplot as plt
import pandas as pd
from overcooked_dataset import OvercookedSequenceDataset, SingleEpisodeOvercookedDataset
mapbt_path = '/home/law/Workspace/repos/COMBO/mapbt_package'
if mapbt_path not in sys.path:
    sys.path.append(mapbt_path)
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


@th.no_grad()
def full_horizon_eval(
    # Model components
    diffusion,
    idm,
    policy,
    renderer: OvercookedSampleRenderer,
    device,
    args, 
    # Configuration for this specific run (can be extracted from a loop)
    agent_id,
    layout_name,
    horzion=32,
    seed=0,
    n_envs=3,
    eval_episodes=3,
    max_steps=400,
    basedir=None,
    is_bc=False,
    show_samples=False,
    save_videos=False,
):
    """
    Run full horizon evaluation of a diffusion policy in Overcooked environment
    """
    print(f"Starting Overcooked Evaluation; BaseDir {basedir}, Layout: {layout_name}, Agent ID: {agent_id}, Episodes: {eval_episodes}, Max Steps: {max_steps}, Horizon: {horzion}, n_envs: {n_envs}, is_bc: {is_bc}")
    video_dir = osp.join(basedir, "videos")
    frames_dir = osp.join(basedir, "frames")
    metrics_dir = osp.join(basedir, "metrics")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Set environment variables
    os.environ["layout"] = layout_name # Important for Overcooked env loading
    
    # Create environments
    if is_bc:
        # Override old dynamics for bc evaluation specifically
        args.old_dynamics = True
        envs = make_eval_env(args, run_dir=args.run_dir, nenvs=n_envs)
        envs.reset_featurize_type([("ppo", "bc") for _ in range(n_envs)])
    else:
        envs = make_eval_env(args, run_dir=args.run_dir, nenvs=n_envs)
    all_metrics = []
    episode_rewards_list = []
    
    F = 32 # Number of Frames in the diffusion horizon
    for episode in range(eval_episodes):
        print(f"Starting episode {episode+1}/{eval_episodes}")

        policy.reset(num_envs=args.n_envs, num_agents=2)
        for e in range(args.n_envs):
            policy.register_control_agent(e=e, a=1)

        cond = np.full((args.n_envs,), agent_id, dtype=np.int64)
        cond = th.tensor(cond, device=device)

        obs, _, _ = envs.reset([True] * args.n_envs)
        
        H, W, C = obs[0][0].shape[0], obs[0][0].shape[1], obs[0][0].shape[-1] 

        steps = 0
        done = False
        episode_reward = np.zeros((args.n_envs, 2))
        frames = [[obs[i][0]] for i in range(args.n_envs)]
        samples_frames = [[] for _ in range(args.n_envs)]

        grid = renderer.extract_grid_from_obs(obs[0][0]) # For rendering
        
        while not done and steps <= max_steps:
            print(f"Steps: {steps}")

            # Setup Condition Obs Based on Obs
            obs_stack = np.stack([normalize_obs(obs[e][0]) for e in range(n_envs)], axis=0) 
            condition_obs = th.tensor(obs_stack, device=device, dtype=th.float32) # Shape: [n_envs, H, W, C]
            
            assert condition_obs.shape[-1] == 26 # Double Check

            condition_obs = rearrange(condition_obs, "b h w c -> b c h w")
            with th.no_grad():
                samples = diffusion.sample(
                x_cond=condition_obs,
                task_embed = cond,
                batch_size=n_envs,
            ) # Shape [n_envs, horizon * C, H, W]
                
            samples = rearrange(samples, "b (f c) h w -> b f h w c", c=C, f=F)
             
            # Render out Samples
            if show_samples:
                _ = [renderer.visualize_all_channels(
                    obs=to_np(samples[0, i]), 
                    output_dir=os.path.join(frames_dir, f"samples_channels_steps_{steps}_horizon_{i}_env_{0}.png")
                ) for i in range(F)]
                 
            # Now step through the environment using the 32-step plan
            plan_horizon = min(max_steps - steps, horzion)

            # We begin with the first ego obs (first obs of the environment)
            obs_t = to_torch(obs_stack) # 3,8,5,26
            for t in range(plan_horizon): 
                obs_tp1 = samples[:,t,...]
                for e in range(n_envs):
                    samples_frames[e].append(to_np(obs_tp1[e]))
                step_actions = np.zeros((n_envs, 2, 1), dtype=np.int64)

                for env_i in range(n_envs):
                    # IDM takes in 26 channels
                    ego_action = get_idm_action(to_torch(obs_t[env_i]).unsqueeze(0), to_torch(obs_tp1[env_i]).unsqueeze(0), idm)
                    step_actions[env_i, 0 ] = to_np(ego_action)
  
                partner_obs_lst = [obs[e][1] for e in range(n_envs)]
                partner_obs = np.stack(partner_obs_lst, axis=0)
                partner_action = policy.step(
                    partner_obs,
                    [(e, 1) for e in range(n_envs)],
                    deterministic=True,
                )
                step_actions[:, 1] = partner_action  # Fill partner action for step tWz

                obs, _, reward, done, _, _ = envs.step(step_actions)
                episode_reward += to_np(reward).squeeze(axis=2)
                obs_t = obs_tp1
                
                for e in range(n_envs):
                    frames[e].append(obs[e][0])

                # Check for early termination
                done = np.all(done)
                steps += 1
                if steps >= max_steps:
                    break

        # After episode completion
        mean_episode_reward_per_env = episode_reward.mean(axis=0)
        print(f"Episode {episode+1} complete: steps={steps}, mean rewards per agent={episode_reward.mean(axis=0)}")
        
        metrics = {
            'episode': episode,
            'steps': steps,
            'rewards_per_agent_total': episode_reward.tolist(), 
            'mean_reward_across_envs': mean_episode_reward_per_env.tolist(),
        }
        all_metrics.append(metrics)
        episode_rewards_list.append(episode_reward) # Store rewards for all envs for this episode
        
        with open(osp.join(metrics_dir, f"episode_{episode+1}_metrics.pkl"), 'wb') as f:
            pickle.dump(metrics, f)
        
        if save_videos:
            print(f"Saving videos for episode {episode+1}...")
            for e in range(n_envs):
                frames[e] = rearrange(frames[e], "f w h c -> f h w c")
                grid = renderer.extract_grid_from_obs(frames[e][0])
                env_dir = osp.join(video_dir, f"episode_{episode+1}_env_{e+1}")
                os.makedirs(env_dir, exist_ok=True)
                saved_video = renderer.render_trajectory_video(
                    frames[e], 
                    grid, 
                    output_dir=env_dir,
                    video_path=osp.join(env_dir, f"actual_trajectory.mp4"),
                    fps=1)
                if show_samples:
                    _ = renderer.render_trajectory_video(
                        samples_frames[e],
                        grid,
                        output_dir=env_dir,
                        video_path=osp.join(env_dir, f"samples_trajectory.mp4"),
                        fps=1
                    )
                print(f"Video saved to {saved_video}")
    
    # Aggregate results across all episodes for this run config
    if episode_rewards_list:
        all_ep_rewards_np = np.array(episode_rewards_list)

        # Calculate mean and std for Agent 0 based on per-episode average rewards
        agent0_mean_rewards_per_episode = all_ep_rewards_np[:, :, 0].mean(axis=1)
        mean_reward_agent0 = agent0_mean_rewards_per_episode.mean()
        std_reward_agent0 = agent0_mean_rewards_per_episode.std()
        
        # Calculate mean and std for Agent 0 based on per-episode average rewards
        agent1_mean_rewards_per_episode = all_ep_rewards_np[:, :, 1].mean(axis=1)
        mean_reward_agent1 = agent1_mean_rewards_per_episode.mean()
        std_reward_agent1 = agent1_mean_rewards_per_episode.std()

        team_rewards_per_ep_env = all_ep_rewards_np.sum(axis=2)
        mean_team_reward = team_rewards_per_ep_env.mean()
        std_team_reward = team_rewards_per_ep_env.std()
    else:
        mean_reward_agent0 = std_reward_agent0 = mean_reward_agent1 = std_reward_agent1 = mean_team_reward = 0.0

    print(f"Agent 0 (Diffusion) mean reward: {mean_reward_agent0:.2f} ± {std_reward_agent0:.2f}")
    print(f"Agent 1 (Partner) mean reward: {mean_reward_agent1:.2f} ± {std_reward_agent1:.2f}")
    print(f"Team mean reward: {mean_team_reward:.2f}")
    
    summary = {
        'basedir': basedir,
        'agent0_mean_reward': mean_reward_agent0,
        'agent0_std_reward': std_reward_agent0,
        'agent1_mean_reward': mean_reward_agent1,
        'agent1_std_reward': std_reward_agent1,
        'team_mean_reward': mean_team_reward,
        'team_std_reward': std_team_reward,
        'layout_name': layout_name,
        'agent_id': agent_id,
        'total_episodes_evaluated': eval_episodes * args.n_envs, # Total env-episodes
        'metrics_per_episode_reset': all_metrics, # List of dicts, one per episode reset
        'raw_episode_rewards': [arr.tolist() for arr in episode_rewards_list], 
    }
    
    metrics_path = osp.join(basedir, "eval_summary.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(summary, f)
    print(f"Summary metrics saved to {metrics_path}")
                    
    envs.close()
    return summary

def load_diffusion_model(args, device, ema=True):
    trainer = OvercookedTrainer(args, device)
    print(f"Loading diffusion model from {args.diffusion_model_path}")
    trainer_actual = trainer.trainer
    trainer_actual.load(args.diffusion_model_path)
    if ema:
        print("Using EMA model for evaluation")
        return trainer_actual.ema.ema_model.to(device)
    else:
        print("Using actual model for evaluation")
        return trainer_actual.model.to(device)

def load_idm_model(args, device): # args is the main parsed args
    print(f"Loading IDM model from {args.idm_path}")
    weights = th.load(args.idm_path)
    idm = InverseDynamicsModel(num_actions=6)
    idm.load_state_dict(weights['model'])
    idm.to(device)
    idm.eval()
    return idm

def load_partner_policy(population_yaml_path, policy_name, device="cpu"): # args is the main parsed args    
    policy = Policy(None, None, None, None, device=device)
    featurize_type = policy.load_population(population_yaml_path, evaluation=True)
    policy = policy.policy_pool[policy_name]
    feat_type = featurize_type.get(policy_name, 'ppo')
    return policy, feat_type

def plot_horizon_evaluation_results(df, output_dir):
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
    plt.grid()
    
    output_path = os.path.join(output_dir, "horizon_evaluation_results.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def plot_episode_evaluation_results(df, output_dir):
    if "training_episode_idx" not in df.columns:
        raise ValueError("DataFrame must contain 'training_episode_idx' column for episode evaluation.")
    
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
    
def plot_experiment_results(experiment_results, index, output_dir):
    df = pd.DataFrame(experiment_results)
    os.mkdir(output_dir, exist_ok=True)

    if index == "horizon_experiment":
        plot_horizon_evaluation_results(df, output_dir)
    else:
        plot_episode_evaluation_results(df, output_dir)

def horizon_experiment():
    """ Create an experiment definition for evaluating different planning horizons in Overcooked.
    """
    horizons_to_test = [2, 4, 8, 16, 24, 32]
    horizon_experiment = {
        "name": "horizon_experiment",
        "evaluation_configs": []
    }
    for horizon_val in horizons_to_test:
        layout = "cramped_room"
        agent_id = 0 
        partner_policy = "sp10_final"
        horizon_experiment['evaluation_configs'].append({
            "layout_name": layout,
            "agent_id": agent_id,
            "horizon": horizon_val,
            "partner_policy_name": partner_policy,
        })
    return horizon_experiment

def create_concept_learning_args(main_args, concept_params):
    """
    Create a new set of arguments for concept learning based on the main_args and concept_params.
    """
    # Start out with a copy of the main args
    cl_args = deepcopy(main_args)
    
    # Override args for concept learning
    cl_args.dataset_path = concept_params['dataset_path']
    cl_args.pretrained_model_path = concept_params['pretrained_model_path']
    cl_args.dummy_policy_id = concept_params['dummy_policy_id']
    cl_args.new_policy_id = concept_params['new_policy_id']
    cl_args.results_dir = concept_params['results_dir_concept_learning']
    cl_args.milestone_name = concept_params['milestone_name']
    cl_args.max_train_steps = concept_params['train_steps']

    cl_args.horizon = concept_params.get('horizon', main_args.horizon) # Use the same horizon as main_args
    cl_args.guidance_weight = concept_params.get('guidance_weight', 1.0)

    # Training Args
    cl_args.train_batch_size = concept_params.get('train_batch_size', main_args.train_batch_size)
    cl_args.num_validation_samples = concept_params.get('num_validation_samples', main_args.num_validation_samples)
    cl_args.save_and_sample_every = concept_params.get('save_and_sample_every', 1000)
    cl_args.cond_drop_prob = concept_params.get('cond_drop_prob', main_args.cond_drop_prob)
    cl_args.split_batches = concept_params.get('split_batches', main_args.split_batches)
    cl_args.save_milestone = concept_params.get('save_milestone', main_args.save_milestone)

    # Dataset args for OvercookedSequenceDataset
    cl_args.max_path_length = main_args.max_path_length
    cl_args.episode_length = main_args.episode_length
    cl_args.chunk_length = main_args.chunk_length
    cl_args.use_padding = main_args.use_padding
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

def train_concept_learn(cl_args, device):
    """
    Train a concept learning model for Overcooked using the provided arguments.
    """
    gc.collect()
    th.cuda.empty_cache()

    # Check if we should use a single episode dataset
    if cl_args.target_episode_idx is not None and cl_args.target_policy_name:
        
        # Create base datatset
        dataset_args = argparse.Namespace(
            dataset_path=cl_args.dataset_path,
            horizon=cl_args.horizon,
            max_path_length=cl_args.max_path_length, 
            episode_length=cl_args.episode_length, 
            chunk_length=cl_args.chunk_length, 
            use_padding=cl_args.use_padding,
        )

        base_dataset = OvercookedSequenceDataset(args=dataset_args, split=cl_args.dataset_split)

        # Create single episode dataset
        print(f"Creating single episode dataset for policy '{cl_args.target_policy_name}', episode {cl_args.target_episode_idx}")
        try:
            single_ep_dataset = SingleEpisodeOvercookedDataset(
                base_dataset, 
                cl_args.target_policy_name, 
                cl_args.target_episode_idx
            )
            
            # Create trainer with single episode dataset
            concept_trainer = ConceptLearnOvercookedTrainer(cl_args, device)
            
            # Replace the dataset with our single episode dataset
            concept_trainer.dataset = single_ep_dataset
            concept_trainer.trainer.train_dataset = single_ep_dataset
            concept_trainer.trainer.valid_dataset = single_ep_dataset

            del base_dataset  # Free memory
            gc.collect()
            th.cuda.empty_cache()
            
            print(f"Successfully created single episode dataset with {len(single_ep_dataset)} samples")
        except Exception as e:
            print(f"Error creating single episode dataset: {e}")
            raise e
    else:
        # Standard training with OvercookedSequenceDataset   
        concept_trainer = ConceptLearnOvercookedTrainer(args, device)
    
    concept_trainer.train() # Saves the model to cl_args.results_dir
    
    del concept_trainer.trainer # Free memory
    gc.collect()
    th.cuda.empty_cache()
    return os.path.join(cl_args.results_dir, f"modl-{cl_args.milestone_name}.pt")

def concept_learn_one_step(main_args,
                        dataset_path, 
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
    """ Create a single concept learning experiment definition.
    Args:
        main_args: Main arguments containing base configurations
        dataset_path: Path to the dataset for concept learning
        new_policy_id: ID for the new policy to be learned
        milestone_name: Name of the milestone for saving the model
        pretrained_model_path: Path to the pretrained model to fine-tune
        dummy_policy_id: ID of the unconditional policy (dummy)
        train_steps: Number of fine-tuning steps
        eval_partner_policy: Partner policy for evaluation (optional)
        eval_layout_name: Layout name for evaluation (optional)
        eval_horizon: Horizon value for evaluation (optional)
        experiment_index: Index for this specific experiment run
        is_test_split: Whether this is a test split (default False)
        target_episode_idx: Specific episode index for single episode training (optional)
    """
    cl_params = {
        'dataset_path': dataset_path,
        'pretrained_model_path': pretrained_model_path,
        'train_steps': train_steps, # Number of fine-tuning steps
        'dummy_policy_id': dummy_policy_id, # Unconditional Policy ID
        'new_policy_id': new_policy_id, # New Policy ID to be learned
        'results_dir_concept_learning': str(Path(main_args.basedir) / f"learned_concepts" / f"concept_{new_policy_id}_training_idx_{experiment_index}"),
        'milestone_name': milestone_name,
        'horizon': eval_horizon if eval_horizon is not None else main_args.horizon,
        'guidance_weight': main_args.guidance_weight if hasattr(main_args, 'guidance_weight') else 1.0,
        'dataset_split' : "test" if is_test_split else "train",
        'target_episode_idx': target_episode_idx, # For single episode training
        'target_policy_name': eval_partner_policy, # Used for single episode training
    }

    experiment_definition = {
        "name":f"concept_learn_experiment_id_{new_policy_id}_experiment_idx_{experiment_index}",
        "concept_learning_params": cl_params,
    }

    if eval_partner_policy:
        experiment_definition["evaluation_configs"] = [{
            "layout_name": eval_layout_name if eval_layout_name is not None else main_args.layout_name,
            "agent_id": new_policy_id, 
            "horizon": eval_horizon if eval_horizon is not None else main_args.horizon,
            "partner_policy_name": eval_partner_policy,
        }]

    return experiment_definition


def evaluate(config,
             args, 
             diffusion_model,
             idm_model,
             renderer,
             evaluation_dir,
             device):
    """
    Run evaluation based on the provided configuration.
    """
    curr_layout = config["layout_name"]
    curr_agent_id = config["agent_id"]
    curr_horizon = config["horizon"]
    curr_partner_policy = config["partner_policy_name"]
    print(f"Running Evaluation with layout: {curr_layout}, agent_id: {curr_agent_id}, horizon: {curr_horizon}, partner_policy: {curr_partner_policy}")

    # Create a unique basedir for this experiment
    current_experiment_name = f"{curr_layout}_agent_id_{curr_agent_id}_horizon_{curr_horizon}_partner_policy_{curr_partner_policy}"
    current_run_basedir = evaluation_dir / current_experiment_name
    current_run_basedir.mkdir(parents=True, exist_ok=True)

    partner_policy_obj, _ = load_partner_policy(
        args.population_yaml_path, 
        curr_partner_policy, 
        device="cpu"
    )

    summary = full_horizon_eval(
        diffusion=diffusion_model,
        idm=idm_model,
        policy=partner_policy_obj,
        renderer=renderer,
        device=device,
        args=args,
        
        agent_id=curr_agent_id,
        layout_name=curr_layout,
        horzion=curr_horizon,
        seed=args.seed,
        n_envs=args.n_envs,
        eval_episodes=args.exp_eval_episodes,
        max_steps=args.max_steps,
        basedir=current_run_basedir,
        is_bc=curr_partner_policy in ["bc_train", "bc_test"],
        show_samples=args.show_samples,
        save_videos=args.save_videos
    )

    if "concept_learning_params" in config and "target_episode_idx" in config["concept_learning_params"]:
        summary['training_episode_idx'] = config["concept_learning_params"]["target_episode_idx"]

    summary['experiment_horizon_val'] = curr_horizon
    summary['experiment_agent_id_val'] = curr_agent_id
    summary['experiment_layout_name_val'] = curr_layout
    summary['experiment_partner_policy_val'] = curr_partner_policy
    return summary


def bc_concept_learn_experiment(args):
    dataset_path = "/home/law/Workspace/repos/COMBO/data/bc_train_dataset.hdf5"
    partner_policy = "bc_train"
    return [
        concept_learn_one_step(
            main_args=args,
            dataset_path=dataset_path,
            new_policy_id=10,
            milestone_name="concept_run_id_10_run_1",
            pretrained_model_path=args.diffusion_model_path,
            dummy_policy_id=0,  # Unconditional policy ID
            train_steps=1,  # Number of fine-tuning steps
            eval_partner_policy=partner_policy,
            experiment_index=0,
        ),
        concept_learn_one_step(
            main_args=args,
            dataset_path=dataset_path,
            new_policy_id=10,
            milestone_name="concept_run_id_10_run_2",
            pretrained_model_path=args.diffusion_model_path,
            dummy_policy_id=0,  # Unconditional policy ID
            train_steps=1,  # Number of fine-tuning steps
            eval_partner_policy=partner_policy,
            experiment_index=1,
        ),
        concept_learn_one_step(
            main_args=args,
            dataset_path=dataset_path,
            new_policy_id=10,
            milestone_name="concept_run_id_10_run_3",
            pretrained_model_path=args.diffusion_model_path,
            dummy_policy_id=0,  # Unconditional policy ID
            train_steps=1,  # Number of fine-tuning steps
            eval_partner_policy=partner_policy,
            experiment_index=2,
        ),
    ]

def create_test_concept_learn_experiment(args, 
                                         num_concept_runs, 
                                         fine_tuning_steps, 
                                         target_episodes={}, 
                                         test_policies_to_use=None):
    """
    Creates concept learning experiments for test policies.
    
    Args:
        args: Main arguments
        num_concept_runs: Number of runs per policy
        fine_tuning_steps: Number of fine-tuning steps
        test_policies_to_use: Specific test policies to use (None means all)
        target_episodes: Dict mapping policy_name -> episode_idx for single episode training
                         (e.g., {"sp10_final": 0} trains only on the first episode of sp10_final)
    """
    experiment_configs = []
    dataset_args = argparse.Namespace(
            dataset_path = args.dataset_path,
            horizon=args.horizon,
            max_path_length=args.max_path_length,
            episode_length= args.episode_length,
            chunk_length=args.chunk_length,
            use_padding=args.use_padding,
        )
    try:
        dataset = OvercookedSequenceDataset(dataset_args, split="test")
        print(f"Successfully loaded test dataset from {args.dataset_path}")

        if not dataset.test_partner_policies:
            print("No test partner policies found in the dataset. Skipping test concept learning experiment.")
            return experiment_configs
    
        test_policies = dataset.test_partner_policies
        print(f"Found {len(test_policies)} test policies: {list(test_policies.keys())}")
        idx = 0
        for i, policy_name in enumerate(test_policies):
            if test_policies_to_use and policy_name not in test_policies_to_use:
                print(f"Skipping policy {policy_name} as it is not in the specified test policies to use.")
                continue
            
            
            
            matching_episodes = dataset.get_path_indexes_episode(policy_name)
            if not matching_episodes:
                print(f"Skipping policy {policy_name} as no episodes were found.")
                continue

            if policy_name in target_episodes:
                target_idx = target_episodes[policy_name]
                if target_idx >= len(matching_episodes):
                    target_idx = target_idx % len(matching_episodes)  # Wrap around if index is too high
                    print(f"Target index {target_idx} for policy {policy_name} is out of bounds. Wrapping around to {target_idx}.")
                    target_episodes[policy_name] = target_idx
            

            for run_idx in range(num_concept_runs):
                base_episode_idx = target_episodes.get(policy_name, 0)
                episode_idx = (base_episode_idx + run_idx) % len(matching_episodes)
                
                print(f"Using episode {episode_idx} for {policy_name} run {run_idx} experiment {idx * num_concept_runs + run_idx}.")
                
                milestone_name = f"test_concept_policy_name_{policy_name}_run_{run_idx}_episode_{episode_idx}"
                experiment_configs.append(
                    concept_learn_one_step(
                        main_args=args,
                        dataset_path=args.dataset_path,
                        new_policy_id=test_policies[policy_name], # Use Policy ID
                        milestone_name=milestone_name,
                        pretrained_model_path=args.diffusion_model_path,
                        dummy_policy_id=dataset.dummy_id,  # Unconditional policy ID
                        train_steps=fine_tuning_steps,  # Number of fine-tuning steps
                        eval_partner_policy=policy_name,  # Evaluate against the same policy
                        experiment_index=idx * num_concept_runs + run_idx,
                        is_test_split=True,  # Indicates this is a test split
                        target_episode_idx=episode_idx,  # Use the calculated episode index
                    )
                )
            idx += 1
        print(f"Generated {len(experiment_configs)} test concept learning experiments.")
    
    except Exception as e:
        print(f"Error loading test dataset or generating experiments: {e}")
        raise e
    
    return experiment_configs

def create_few_shot_experiment(args, policy_name, dummy_policy_id, episodes=5, runs=3):
    few_shot_configs = []
    policy_id = None
    
    dataset_args = argparse.Namespace(
        dataset_path=args.dataset_path,
        horizon=args.horizon,
        max_path_length=args.max_path_length,
        episode_length=args.episode_length,
        chunk_length=args.chunk_length,
        use_padding=args.use_padding,
    )
    
    try:
        dataset = OvercookedSequenceDataset(args=dataset_args, split="test")
        if policy_name in dataset.test_partner_policies:
            policy_id = dataset.test_partner_policies[policy_name]
    except Exception:
        pass
    
    if policy_id is None:
        raise ValueError(f"Could not determine policy ID for {policy_name}. Using default.")
        
    for run in range(runs):
        few_shot_configs.append(
            concept_learn_one_step(
                main_args=args,
                dataset_path=args.dataset_path,
                new_policy_id=policy_id,
                milestone_name=f"fewshot_{policy_name}_run{run}",
                pretrained_model_path=args.diffusion_model_path,
                dummy_policy_id=dummy_policy_id,
                train_steps=args.few_shot_train_steps,
                eval_partner_policy=policy_name,
                experiment_index=run,
                is_test_split=(policy_name in dataset.test_partner_policies)
            )
        )
    
    return few_shot_configs


def run_experiments(args, use_successive_models=True):
    """
    Main function to run all experiments based on the provided arguments.
    Args:
        args: Parsed command line arguments
        use_successive_models: Whether to use the model from the previous step for concept learning
    """
    device = th.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")
    print(f"Using device: {device}")

    # Load IDM and Renderer once, since they are static
    idm_model = load_idm_model(args, device)
    renderer = OvercookedSampleRenderer()

    # Edit this directly #TODO
    experiment_configs = []
    # experiment_configs.append(horizon_experiment())
    # experiment_configs.extend(bc_concept_learn_experiment(args))
    
    experiment_configs.extend(create_test_concept_learn_experiment(
        args, 
        num_concept_runs=args.num_concept_runs,
        fine_tuning_steps=args.test_concept_train_steps,
        test_policies_to_use=["sp10_final", "sp9_final"],
        target_episodes={} # Replace target_episodes with an empty dict since we'll be using sequential episodes
    ))

    base_output_dir = Path(args.basedir)
    exp_group_dir = base_output_dir / (args.exp_group_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    exp_group_dir.mkdir(parents=True, exist_ok=True)

    all_experiment_results = []

    # Path to the model that the concept learning should use as its pretrained input.
    current_model_path = args.diffusion_model_path
    current_args = deepcopy(args)

    for exp_idx, exp_config in enumerate(experiment_configs):
        gc.collect()
        th.cuda.empty_cache()

        print(f"\n=== Running Experiment {exp_config['name']} ===")
        experiment_results = []

        # Create directory for this specific experiment
        exp_dir = exp_group_dir / exp_config["name"]
        exp_dir.mkdir(exist_ok=True)

        # If concept learning, we use the actual model not the EMA model
        use_ema = "concept_learning_params" not in exp_config

        model_path_for_eval = current_model_path
        args_for_loading = deepcopy(current_args)

        # Conduct Concept Learning First If present
        if "concept_learning_params" in exp_config:
            cl_params = exp_config["concept_learning_params"]

            # Update the pretrained model path if using successive models
            if use_successive_models and exp_idx > 0:
                cl_params['pretrained_model_path'] = current_model_path
                assert cl_params['pretrained_model_path'] == current_model_path, \
+                    f"Experiment {exp_config['name']} did not pick up previous model {current_model_path}"
                print(f"Using model from previous step: {current_model_path}")
           
            
            # Prepare concept learning arguments
            cl_args = create_concept_learning_args(current_args, cl_params)
            cl_args.results_dir = str(exp_dir / "concept_learning_output")

            # Run concept learning
            print(f"Starting concept learning for {exp_config['name']}...")
            concept_model_path = train_concept_learn(cl_args, device)
            print(f"Concept learning complete. Model saved to {concept_model_path}")

            # Update current model path for successive experiments
            if use_successive_models:
                current_model_path = concept_model_path
                current_args = deepcopy(cl_args)
            
            # Load this concept model for evaluation
            model_path_for_eval = concept_model_path
            args_for_loading = deepcopy(cl_args)

            
        # Load the diffusion model
        args_for_loading.diffusion_model_path = model_path_for_eval
        diffusion_model = load_diffusion_model(args_for_loading, device, ema=use_ema)

        # Now Conduct Evaluation
        eval_configs = exp_config.get("evaluation_configs", [])
        if not isinstance(eval_configs, list):
            eval_configs = [eval_configs]
        
        experiment_results = []
        
        for eval_config in eval_configs:
            if eval_config:
                print(f"Running evaluation: {eval_config}")
                eval_dir = exp_dir / "evaluation"
                eval_dir.mkdir(exist_ok=True)
                result = evaluate(
                    eval_config, args, diffusion_model, idm_model, 
                    renderer, eval_dir, device
                )
                experiment_results.append(result)
            
        # Save results for this experiment
        exp_result = {
            "name": exp_config["name"],
            "config": exp_config,
            "results": experiment_results,
            "model_path": model_path_for_eval
        }
        all_experiment_results.append(exp_result)

        
        del diffusion_model  # Free memory
        gc.collect()
        th.cuda.empty_cache()

        with open(exp_group_dir / f"results_through_exp_{exp_idx}.pkl", "wb") as f:
            pickle.dump(all_experiment_results, f)

    
    # Save overall summary
    results_file = exp_group_dir / "all_experiment_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(all_experiment_results, f)

    # Generate plots for the experiment results
    plot_dir = exp_group_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    try:
        # Extract just the evaluation results for plotting with proper metadata
        plot_results = []
        for exp in all_experiment_results:
            if "results" in exp and exp["results"]:
                # Extract experiment config metadata
                config = exp.get("config", {})
                cl_params = config.get("concept_learning_params", {})
                
                # Extract important metadata for plotting
                milestone_name = cl_params.get("milestone_name", "")
                target_episode_idx = cl_params.get("target_episode_idx", None)
                
                for res in exp["results"]:
                    # Create a copy of the result to avoid modifying original
                    res_copy = res.copy()
                    
                    # Add the metadata needed for plotting
                    res_copy["milestone_name"] = milestone_name
                    res_copy["training_episode_idx"] = target_episode_idx
                    
                    plot_results.append(res_copy)
        
        # Generate the plots using os.makedirs instead of os.mkdir
        if plot_results:
            # Fix the mkdir issue by updating plot_experiment_results function
            def plot_experiment_results_fixed(experiment_results, index, output_dir):
                df = pd.DataFrame(experiment_results)
                os.makedirs(output_dir, exist_ok=True)  # Use makedirs instead of mkdir
                
                if index == "horizon_experiment":
                    plot_horizon_evaluation_results(df, output_dir)
                else:
                    plot_episode_evaluation_results(df, output_dir)
            
            # Use the fixed function
            plot_experiment_results_fixed(plot_results, "episode_experiment", plot_dir)
            print(f"Generated plots in {plot_dir}")
        else:
            print("No evaluation results to plot")
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    print(f"\nAll experiments complete. Results saved to {results_file}")
    
    return all_experiment_results


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
    run_experiments(args, use_successive_models=True)
    