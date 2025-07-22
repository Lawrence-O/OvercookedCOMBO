import argparse
import datetime
import json
import os
from pathlib import Path
import pickle
import numpy as np
import torch as th
import sys

from tqdm import tqdm

from einops.einops import rearrange
import numpy as np
import torch as th
import warnings
warnings.filterwarnings("ignore")
from overcooked.diffusion.goal_diffusion import GoalGaussianDiffusion
from overcooked.diffusion.unet import UnetOvercooked
from overcooked.utils.experiments_util import managed_environment, unnormalize_obs, normalize_obs, make_eval_env, to_torch, load_partner_policy, max_normalize_obs, convert_to_binary_obs
from overcooked.utils.overcooked_visualizer import OvercookedVisualizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp
from overcooked.agent.idm.ground_truth_idm import GroundTruthInverseDynamics
from overcooked.agent.reward.state_reward_model import RewardCalculator as GroundTruthRewardCalculator
from overcooked.agent.diffusion_agent import DiffusionPlannerAgent
from overcooked.dataset.overcooked_dataset import OvercookedSequenceDataset
from overcooked.diffusion.unet import UnetOvercooked, UnetOvercookedActionProposal
from ema_pytorch import EMA
import seaborn as sns
from contextlib import contextmanager
from overcooked.dataset.hdf5_dataset import HDF5Dataset
from overcooked_ai_py.mdp.overcooked_mdp import Action
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, mean_squared_error, mean_absolute_error


class BaseTester:
    """
    A base class for model testers, containing shared functionalities like
    argument handling, seeding, and directory management.
    """
    def __init__(self, args):
        self.args = args
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.renderer = OvercookedVisualizer()
        self.n_envs = args.n_envs
        self.max_steps = args.max_steps
        self.horizon = args.horizon
        self.H, self.W, self.C = 8, 5, 26  # Overcooked obs shape
        self.num_actions = 6
        self.value_min, self.value_max = 0.0, 25.0
        
        # Base directory for all outputs of this tester instance
        self.base_dir = Path("model_tester")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.set_seed()

    def set_seed(self, seed=42):
        """Set random seed for reproducibility."""
        th.manual_seed(seed)
        np.random.seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)
    
    def _make_json_serializable(self, data):
        """
        Recursively walks a dictionary or list and converts numpy types
        to native Python types.
        """
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._make_json_serializable(v) for v in data]
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def set_experiment_dir(self, experiment_name, subfolder_name=None):
        """Set the base_dir to an experiment-specific subfolder."""
        if subfolder_name is None:
            subfolder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = self.base_dir / experiment_name / subfolder_name
        self.base_dir = experiment_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment outputs will be saved to: {self.base_dir}")

    def load_data_from_pickle(self, filepath):
        """Load previously saved data from a pickle file."""
        print(f"Loading data from {filepath}...")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print("Data loaded successfully.")
        return data

    def _load_dataset(self, split="test"):
        """Loads the HDF5 dataset for offline evaluation."""
        print(f"Loading '{split}' split from dataset at {self.args.dataset_path}")
        dataset_args = argparse.Namespace(
            dataset_path=self.args.dataset_path,
            horizon=self.args.horizon,
            max_path_length=self.args.max_path_length,
            episode_length=self.args.episode_length,
            chunk_length=self.args.chunk_length,
            use_padding=self.args.use_padding,
        )
        return OvercookedSequenceDataset(args=dataset_args, split=split)
    def evaluate_in_env(self, state_action_fn, num_episodes=10, policy_name="bc_train"):
        """Test the value model by running episodes and collecting rewards.
        Supports both single-step and planning horizon action outputs.
        """
        all_data = []
        with managed_environment(self.args, policy_name, self.n_envs ) as (envs, partner_agent_policy):
            for episode in tqdm(range(num_episodes), desc="Collecting data"):
                # Reset Policy
                partner_agent_policy.reset(num_envs=self.n_envs, num_agents=1)
                for e in range(self.n_envs):
                    partner_agent_policy.register_control_agent(e=e, a=1)
                
                obs, _, _ = envs.reset([True] * self.n_envs)
                steps = 0
                done = False
                episode_reward = np.zeros((self.n_envs, 2))
                frames = [[obs[i][0]] for i in range(self.n_envs)]
                
                # Data containers
                obs_t_list = []
                obs_tp1_list = []
                actions_list = []
                rewards_list = []
                done_list = []
                step_list = []

                while not done and steps <= self.args.max_steps:
                    step_actions = np.zeros((self.n_envs, 2, 1), dtype=np.int64)
                    # Agent 0 actions using policy
                    agent0_obs_lst = [obs[e][0] for e in range(self.n_envs)]
                    agent0_obs = np.stack(agent0_obs_lst, axis=0)
                    agent0_actions = state_action_fn(agent0_obs, policy_name)

                    # If agent0_actions is (nenvs, horizon) or (nenvs, horizon, 1), plan ahead
                    if isinstance(agent0_actions, np.ndarray) and agent0_actions.ndim >= 2 and agent0_actions.shape[1] > 1:
                        print(f"Planning horizon detected: {agent0_actions.shape[1]} steps")
                        horizon = agent0_actions.shape[1]
                        # For each step in the planning horizon
                        for h in range(horizon):
                            if done or steps > self.args.max_steps:
                                break
                            # Prepare actions for this step
                            step_actions = np.zeros((self.n_envs, 2, 1), dtype=np.int64)
                            # Agent 0: take the h-th planned action
                            if agent0_actions.ndim == 3:
                                step_actions[:, 0, 0] = agent0_actions[:, h, 0]
                            else:
                                step_actions[:, 0, 0] = agent0_actions[:, h]
                            # Agent 1: always use bc_policy
                            agent1_obs_lst = [obs[e][1] for e in range(self.n_envs)]
                            agent1_obs = np.stack(agent1_obs_lst, axis=0)
                            agent1_actions = partner_agent_policy.step(
                                agent1_obs,
                                [(e, 1) for e in range(self.n_envs)],
                                deterministic=False,
                            )
                            step_actions[:, 1] = agent1_actions

                            # Step environment
                            obs, shared_obs, reward, done, info, aval_actions = envs.step(step_actions)
                            episode_reward += reward.squeeze(axis=2)
                            rewards_list.append(reward.squeeze(axis=2).copy())
                            done_list.append(done.copy())
                            step_list.append(steps)

                            # Save obs_tp1 for both agents
                            obs_t_list.append((agent0_obs.copy(), agent1_obs.copy()))
                            obs_tp1_list.append((
                                np.stack([obs[e][0] for e in range(self.n_envs)], axis=0),
                                np.stack([obs[e][1] for e in range(self.n_envs)], axis=0)
                            ))
                            actions_list.append((step_actions[:, 0, 0].copy(), agent1_actions.copy()))
                            for e in range(min(self.n_envs, 3)):
                                frames[e].append(obs[e][0])
                            done = np.all(done)
                            steps += 1
                    else:
                        # --- Single-step fallback ---
                        step_actions[:, 0] = agent0_actions
                        # Agent 1 actions using policy
                        agent1_obs_lst = [obs[e][1] for e in range(self.n_envs)]
                        agent1_obs = np.stack(agent1_obs_lst, axis=0)
                        agent1_actions = partner_agent_policy.step(
                            agent1_obs,
                            [(e, 1) for e in range(self.n_envs)],
                            deterministic=False,
                        )
                        step_actions[:, 1] = agent1_actions

                        # Step environment
                        obs, shared_obs, reward, done, info, aval_actions = envs.step(step_actions)
                        episode_reward += reward.squeeze(axis=2)
                        rewards_list.append(reward.squeeze(axis=2).copy())
                        done_list.append(done.copy())
                        step_list.append(steps)

                        # Save obs_tp1 for both agents
                        obs_t_list.append((agent0_obs.copy(), agent1_obs.copy()))
                        obs_tp1_list.append((
                            np.stack([obs[e][0] for e in range(self.n_envs)], axis=0),
                            np.stack([obs[e][1] for e in range(self.n_envs)], axis=0)
                        ))
                        actions_list.append((agent0_actions.copy(), agent1_actions.copy()))
                        for e in range(min(self.n_envs, 3)):
                            frames[e].append(obs[e][0])
                        done = np.all(done)
                        steps += 1

                    if steps >= self.args.max_steps:
                        break
                print(f"Episode {episode + 1}/{num_episodes} completed with total reward: {episode_reward.sum(axis=0) // self.n_envs}")
                all_data.append({
                    "rewards": rewards_list,
                    "steps": step_list,
                    "episode_reward": episode_reward,
                    "obs_t": obs_t_list,
                    "obs_tp1": obs_tp1_list,
                    "actions": actions_list,
                    "done": done_list,
                    "frames": frames,
                })
                if episode == num_episodes - 1:
                    video_dir = self.base_dir / "videos" / "evaluation"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    self._save_episode_videos(frames, episode, video_dir)
        return all_data
    def calculate_evaluation_summary(self, episode_rewards_list):
        """Calculate summary statistics from episode rewards."""
        all_ep_rewards_np = np.array(episode_rewards_list)
        
        print(f"Shape of all_ep_rewards_np: {all_ep_rewards_np.shape}")

        # Agent 0 statistics
        # Extract all total rewards for Agent 0 across all episodes and environments
        agent0_all_rewards = all_ep_rewards_np[:, :, 0] # Shape: (num_episodes, self.n_envs)
        mean_reward_agent0 = agent0_all_rewards.mean()
        std_reward_agent0 = agent0_all_rewards.std()
        
        # Agent 1 statistics
        # Extract all total rewards for Agent 1 across all episodes and environments
        agent1_all_rewards = all_ep_rewards_np[:, :, 1] # Shape: (num_episodes, self.n_envs)
        mean_reward_agent1 = agent1_all_rewards.mean()
        std_reward_agent1 = agent1_all_rewards.std()

        # Team statistics
        # Sum rewards of both agents for each (episode, environment) pair
        team_all_rewards = all_ep_rewards_np.sum(axis=2) # Shape: (num_episodes, self.n_envs)
        mean_team_reward = team_all_rewards.mean()
        std_team_reward = team_all_rewards.std()
        
        # Log results
        print(f"Agent 0 (Diffusion) mean reward: {mean_reward_agent0:.2f} ± {std_reward_agent0:.2f}")
        print(f"Agent 1 (Partner) mean reward: {mean_reward_agent1:.2f} ± {std_reward_agent1:.2f}")
        print(f"Team mean reward: {mean_team_reward:.2f} ± {std_team_reward:.2f}")
        
        # Create summary dictionary
        return {
            'agent0_mean_reward': mean_reward_agent0,
            'agent0_std_reward': std_reward_agent0,
            'agent1_mean_reward': mean_reward_agent1,
            'agent1_std_reward': std_reward_agent1,
            'team_mean_reward': mean_team_reward,
            'team_std_reward': std_team_reward,
        }
    def old_calculate_evaluation_summary(self, episode_rewards_list):
        """Calculate summary statistics from episode rewards."""
        all_ep_rewards_np = np.array(episode_rewards_list)
        
        print(all_ep_rewards_np.shape)

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
        

        # Log results
        print(f"Agent 0 (Diffusion) mean reward: {mean_reward_agent0:.2f} ± {std_reward_agent0:.2f}")
        print(f"Agent 1 (Partner) mean reward: {mean_reward_agent1:.2f} ± {std_reward_agent1:.2f}")
        print(f"Team mean reward: {mean_team_reward:.2f} ± {std_team_reward:.2f}")
        
        # Create summary dictionary
        return {
            'agent0_mean_reward': mean_reward_agent0,
            'agent0_std_reward': std_reward_agent0,
            'agent1_mean_reward': mean_reward_agent1,
            'agent1_std_reward': std_reward_agent1,
            'team_mean_reward': mean_team_reward,
            'team_std_reward': std_team_reward,
        }
    def plot_base_model_results(self, results, agent_label="bc_train"):
        """
        Generate a grouped bar plot for base model evaluation results.
        Assumes all results are for the same partner policy ("bc_train").
        """

        # Create plot directory
        plot_dir = self.base_dir / "plots" / "model_evaluation"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Convert results to DataFrame
        df = pd.DataFrame(results)
        if df.empty:
            print("No results to plot.")
            return

        # Always use "bc_train" as the label
        x = np.arange(len(df))  # the label locations

        # Bar width
        bar_width = 0.25

        # Metrics
        agent0_means = df['agent0_mean_reward'].values
        agent0_stds = df['agent0_std_reward'].values
        agent1_means = df['agent1_mean_reward'].values
        agent1_stds = df['agent1_std_reward'].values
        team_means = df['team_mean_reward'].values
        team_stds = df['team_std_reward'].values

        # Bar positions
        r1 = x
        r2 = x + bar_width
        r3 = x + 2 * bar_width

        plt.figure(figsize=(10, 6))
        plt.bar(r1, agent0_means, yerr=agent0_stds, width=bar_width, label='Agent 0 (Diffusion)', capsize=5, color='skyblue')
        plt.bar(r2, agent1_means, yerr=agent1_stds, width=bar_width, label=f'Agent 1 ({agent_label})', capsize=5, color='lightgreen')
        plt.bar(r3, team_means, yerr=team_stds, width=bar_width, label='Team Total', capsize=5, color='coral')

        plt.xlabel('Evaluation Run')
        plt.ylabel('Mean Reward')
        plt.title(f'Base Model Performance (Partner: {agent_label})')
        plt.xticks(x + bar_width, [f'Run {i+1}' for i in x])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        output_path = plot_dir / "base_model_performance.png"
        plt.savefig(output_path)
        print(f"Base model performance plot saved to {output_path}")
        plt.close()

        # Optionally: Save summary as CSV/JSON for further analysis
        df.to_csv(plot_dir / "base_model_performance.csv", index=False)
        df.to_json(plot_dir / "base_model_performance.json", orient='records', indent=2)
    def _save_episode_videos(self, frames, episode, video_dir):
        """Save videos for a completed episode."""
        print(f"Saving videos for episode {episode + 1}...")
        print(len(frames), len(frames[0]))
        frames = [
            [np.transpose(f, (1, 0, 2)) for f in env_frames]
            for env_frames in frames
        ]
        grid = self.renderer.extract_grid_from_obs(frames[0][0])
        for e in range(len(frames)):
            env_dir = video_dir / f"episode_{episode + 1}_env_{e + 1}"
            env_dir.mkdir(parents=True, exist_ok=True)
            print(frames[e][0].shape)
            
            # Save actual trajectory
            saved_video = self.renderer.render_trajectory_video(
                frames[e], 
                grid, 
                output_dir=str(env_dir),
                video_path=str(env_dir / "actual_trajectory.mp4"),
                fps=1
            )

class ActionProposalTester(BaseTester):
    """
    Tests the action proposal model, primarily through online evaluation in the environment.
    """
    def __init__(self, args):
        super().__init__(args)
        # self.action_proposal_model = self._load_action_proposal_model()
        # if hasattr(self.action_proposal_model, 'ema_model'):
        #     self.action_proposal_model = self.action_proposal_model.ema_model
        # self.action_proposal_model.to(self.device)

    def _load_action_proposal_model(self,ema=True):
        """Load the action proposal model."""
        if not osp.exists(self.args.action_proposal_model_path):
            raise FileNotFoundError(f"Action proposal model path {self.args.action_proposal_model_path} does not exist.")
        ckpt = th.load(self.args.action_proposal_model_path, map_location="cpu")
        unet = UnetOvercookedActionProposal(
            horizon=self.args.horizon,
            obs_dim=(self.H, self.W, self.C),
            num_actions=self.num_actions,
        ).to(self.device)
        diffusion = GoalGaussianDiffusion(
            model=unet,
            channels=self.num_actions * self.horizon,
            image_size=(1,1),
            timesteps=1000,
            sampling_timesteps=500,
            loss_type="l2",
            objective="pred_v",
            beta_schedule="cosine",
            min_snr_loss_weight=True,
            guidance_weight=getattr(self.args, 'guidance_weight', 1.0),
            auto_normalize=False,
        ).to(self.device)
        if ema:
            ema_wrap = EMA(diffusion,beta = 0.999,update_every=10)
            ema_wrap.load_state_dict(ckpt['ema'])
            return ema_wrap
        else:
            diffusion.load_state_dict(ckpt['model'])
            return diffusion
    
    @th.no_grad()
    def evaluate_action_proposal_in_env(self, num_episodes=10, planning_horizon=32, policy_name="bc_train"):
        # self.action_proposal_model.eval()
        def action_proposal_fn(obs, policy_name=None):
            """
            Given an observation, use the action proposal model to predict the next action.
            """
            # obs = normalize_obs(obs)
            # obs_tensor = to_torch(obs, device=self.device)
            # obs_tensor = obs_tensor.view(self.n_envs, self.C, self.H, self.W)
            # pred = self.action_proposal_model.sample(
            #     x_cond=obs_tensor,
            #     batch_size=self.n_envs,
            # )
            # pred = pred.view(self.n_envs, self.horizon, self.num_actions)
            # pred = th.argmax(pred, dim=-1)  # [N, L]
            # action = pred[:, :planning_horizon].cpu().numpy()  # Use only the first `planning_horizon` actions
            # return action
            action = th.randint(0, self.num_actions, (self.n_envs, 8), device=self.device).cpu().numpy()
            print(f"Action proposal model generated actions: {action}")
            return action
        # Evaluate in environment
        eval_data = self.evaluate_in_env(action_proposal_fn, num_episodes=num_episodes, policy_name=policy_name)
        return eval_data


    def run_online_evaluation(self, num_episodes=10, planning_horizon=8, partner_policy="bc_train", subfolder_name=None):
        """Orchestrates the online evaluation and reporting."""
        self.set_experiment_dir("action_proposal_evaluation", subfolder_name=subfolder_name)
        print(f"Running online evaluation for action proposal model...")
        eval_data = self.evaluate_action_proposal_in_env(
            num_episodes=num_episodes,
            planning_horizon=planning_horizon,
            policy_name=partner_policy
        )
        
        episode_rewards_list = [run["episode_reward"] for run in eval_data]
        summary = self.calculate_evaluation_summary(episode_rewards_list)
        self.plot_base_model_results([summary], agent_label=partner_policy)
        print("Action proposal model online evaluation completed and plots saved.")

class DatasetTester(BaseTester):
    """
    Tests dataset loading and provides a comprehensive suite of diagnostic plots.
    The plotting functions are designed to be robust and use seaborn best practices.
    """
    def __init__(self, args, split="train"):
        super().__init__(args)
        
        # Set a global, professional style for all plots generated by this class
        sns.set_theme(style="whitegrid", palette="deep")
        
        # Borrow policy mapping logic from the dataset wrapper
        print("Instantiating dataset wrapper to extract policy mappings...")
        dataset_wrapper = OvercookedSequenceDataset(args, split=split)
        self.original_id_to_name = dataset_wrapper._original_id_to_name
        self.horizon = dataset_wrapper.horizon # Grab horizon from dataset args
        del dataset_wrapper

        # Load raw data for efficient analysis
        print(f"Loading raw HDF5 data from '{split}' split for analysis...")
        self.hdf5_dataset = HDF5Dataset(args, split=split)
        self.actions = np.array(self.hdf5_dataset.dset["actions"])
        self.rewards = np.array(self.hdf5_dataset.dset["rewards"])
        self.policy_ids_raw = np.array(self.hdf5_dataset.dset["policy_id"])
        print("Raw data loaded successfully.")

    def plot_dataset_data(self, title_prefix=""):
        """
        Main orchestrator method to generate and save all diagnostic plots.
        """
        # Set the experiment directory for all outputs
        self.set_experiment_dir("dataset_analysis", "plots")
        save_dir = self.base_dir
        print(f"Diagnostic plots will be saved to: {save_dir}")

        # Prepare DataFrames once for all plots
        print("Preparing data for plotting...")
        ep_df, step_df, rtg_df = self._prepare_plot_data()
        n_agents = self.rewards.shape[-1]
        agent_palette = sns.color_palette("viridis", n_agents)

     
        print("\nGenerating general distribution plots...")
        self._plot_hist(step_df, 'reward', "Reward Distribution (All Steps)", "Reward", save_dir / "reward_hist_all_steps.png", title_prefix)
        
        non_zero_step_df = step_df[step_df['reward'] != 0].copy()
        self._plot_hist(non_zero_step_df, 'reward', "Non-Zero Reward Distribution (All Steps)", "Reward", save_dir / "reward_hist_all_steps_nonzero.png", title_prefix, force_ticks=[3])
        
        self._plot_hist(ep_df, 'mean_episode_reward', "Total Reward per Episode (Mean over Agents)", "Total Reward", save_dir / "reward_hist_per_episode.png", title_prefix)
        self._plot_hist(step_df, 'action', "Action Distribution (All Steps & Agents)", "Action", save_dir / "action_hist_overall.png", title_prefix, discrete=True)
        
 
        print("Generating per-agent and per-policy plots...")
        self._plot_boxplot_by_column(ep_df, [f'total_reward_agent_{i}' for i in range(n_agents)], "Total Reward per Episode by Agent", "Total Reward", save_dir / "reward_boxplot_per_agent.png", title_prefix, agent_palette)
        self._plot_action_dist_per_agent(step_df, n_agents, save_dir, title_prefix, agent_palette)
        self._plot_boxplot_by_policy(ep_df, 'mean_episode_reward', "Total Reward (Mean) by Partner Policy", "Total Reward", save_dir / "reward_boxplot_by_policy.png", title_prefix)
        self._plot_boxplot_by_policy(ep_df, 'total_reward_agent_0', "Ego Agent Reward by Partner Policy", "Ego Total Reward", save_dir / "ego_reward_boxplot_by_policy.png", title_prefix)

        print("Generating Reward-to-Go plots...")
        self._plot_rtg_curves(rtg_df, 'agent', "Mean Reward-to-Go by Agent", save_dir / "rtg_by_agent.png", title_prefix, palette=agent_palette)
        self._plot_rtg_curves(rtg_df, 'partner_policy', "Mean Reward-to-Go by Partner Policy", save_dir / "rtg_by_policy.png", title_prefix)
        
        print("\nGenerating sliding window reward distribution plots...")
        ego_sums, partner_sums, team_sums = self._calculate_sliding_window_rewards(horizon=self.horizon)
        self._plot_sliding_reward_histogram(ego_sums, f"Ego Agent Reward Distribution (Sliding Window H={self.horizon})", save_dir / "ego_sliding_reward_hist.png", title_prefix)
        self._plot_sliding_reward_histogram(partner_sums, f"Partner Agent Reward Distribution (Sliding Window H={self.horizon})", save_dir / "partner_sliding_reward_hist.png", title_prefix)
        self._plot_sliding_reward_histogram(team_sums, f"Team Reward Distribution (Sliding Window H={self.horizon})", save_dir / "team_sliding_reward_hist.png", title_prefix)
        
        print("\nGenerating time-to-reward analysis...")
        self._plot_time_to_reward_events(
            save_path=save_dir / "time_to_reward_hist.png",
            title_prefix=title_prefix
        )

        print(f"\nAll dataset diagnostic plots have been saved successfully.")

    def _prepare_plot_data(self):
        rewards = self.rewards.squeeze(-1)
        actions = self.actions.squeeze(-1)
        n_eps, ep_len, n_agents = rewards.shape
        raw_partner_ids = self.policy_ids_raw[:, 1]
        partner_policy_names = np.array([self.original_id_to_name.get(pid, f"unknown_id_{pid}") for pid in raw_partner_ids])
        ep_rewards = rewards.sum(axis=1)
        ep_data = {'episode_id': range(n_eps), 'partner_policy': partner_policy_names, 'mean_episode_reward': ep_rewards.mean(axis=1)}
        for i in range(n_agents):
            ep_data[f'total_reward_agent_{i}'] = ep_rewards[:, i]
        ep_df = pd.DataFrame(ep_data)
        step_data = {'reward': rewards.flatten(), 'action': actions.flatten(), 'agent': np.tile(np.arange(n_agents), n_eps * ep_len),
                     'timestep': np.tile(np.repeat(np.arange(ep_len), n_agents), n_eps), 'episode_id': np.repeat(np.arange(n_eps), ep_len * n_agents)}
        step_df = pd.DataFrame(step_data)
        rtg = np.flip(np.cumsum(np.flip(rewards, axis=1), axis=1), axis=1)
        rtg_data = {'rtg_value': rtg.flatten(), 'agent': np.tile(np.arange(n_agents), n_eps * ep_len),
                    'timestep': np.tile(np.repeat(np.arange(ep_len), n_agents), n_eps), 'episode_id': np.repeat(np.arange(n_eps), ep_len * n_agents)}
        rtg_df = pd.DataFrame(rtg_data).merge(ep_df[['episode_id', 'partner_policy']], on='episode_id')
        return ep_df, step_df, rtg_df

    def _calculate_sliding_window_rewards(self, horizon: int):
        print(f"Calculating sliding window rewards for horizon={horizon}...")
        rewards = self.rewards.squeeze(-1)
        ego_rewards, partner_rewards = rewards[..., 0], rewards[..., 1]
        team_rewards = rewards.sum(axis=2)
        padded_ego_cumsum = np.pad(np.cumsum(ego_rewards, axis=1), ((0, 0), (1, 0)))
        padded_partner_cumsum = np.pad(np.cumsum(partner_rewards, axis=1), ((0, 0), (1, 0)))
        padded_team_cumsum = np.pad(np.cumsum(team_rewards, axis=1), ((0, 0), (1, 0)))
        ego_window_sums = (padded_ego_cumsum[:, horizon:] - padded_ego_cumsum[:, :-horizon]).flatten()
        partner_window_sums = (padded_partner_cumsum[:, horizon:] - padded_partner_cumsum[:, :-horizon]).flatten()
        team_window_sums = (padded_team_cumsum[:, horizon:] - padded_team_cumsum[:, :-horizon]).flatten()
        print("...calculation complete.")
        return ego_window_sums, partner_window_sums, team_window_sums

    @contextmanager
    def _plot_context(self, save_path, title, figsize=(10, 6), title_prefix=""):
        fig, ax = plt.subplots(figsize=figsize)
        full_title = f"{title_prefix}{title}" if title_prefix else title
        ax.set_title(full_title, fontsize=16, pad=20)
        try:
            yield fig, ax
        finally:
            fig.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    def _plot_hist(self, df, col_name, title, xlabel, save_path, title_prefix, discrete=False, force_ticks=None):
        with self._plot_context(save_path, title, figsize=(8, 5), title_prefix=title_prefix) as (fig, ax):
            sns.histplot(x=df[col_name].to_numpy(), discrete=discrete, kde=(not discrete), ax=ax)
            if discrete:
                ax.set_xticks(sorted(df[col_name].unique()))
            if force_ticks is not None:
                current_ticks = ax.get_xticks()
                all_ticks = np.unique(np.concatenate((current_ticks, force_ticks))).astype(np.int64)
                ax.set_xticks(all_ticks)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Frequency")

    def _plot_boxplot_by_policy(self, df, value_col, title, ylabel, save_path, title_prefix):
        """Plots boxplots grouped by policy using a robust method for rotating labels."""
        with self._plot_context(save_path, title, figsize=(12, 7), title_prefix=title_prefix) as (fig, ax):
            sorted_policies = df.groupby('partner_policy')[value_col].median().sort_values(ascending=False).index
            sns.boxplot(data=df, x='partner_policy', y=value_col, order=sorted_policies, ax=ax)
            ax.set_xlabel("Partner Policy")
            ax.set_ylabel(ylabel)
            labels = ax.get_xticklabels()
            plt.setp(labels, rotation=45, ha='right')

    def _plot_boxplot_by_column(self, df, cols_to_plot, title, ylabel, save_path, title_prefix, palette):
        with self._plot_context(save_path, title, figsize=(8, 6), title_prefix=title_prefix) as (fig, ax):
            df_melted = df.melt(value_vars=cols_to_plot, var_name='Agent', value_name='Total Reward')
            df_melted['Agent'] = df_melted['Agent'].str.replace('total_reward_agent_', 'Agent ')
            sns.boxplot(data=df_melted, x='Agent', y='Total Reward', palette=palette, ax=ax)
            ax.set_xlabel(None)
            ax.set_ylabel(ylabel)

    def _plot_action_dist_per_agent(self, step_df, n_agents, save_dir, title_prefix, palette):
        g = sns.displot(data=step_df, x="action", col="agent", col_wrap=min(n_agents, 4), kind="hist",
                        discrete=True, hue="agent", palette=palette, legend=False, height=4, aspect=1.2)
        g.fig.suptitle(f"{title_prefix}Action Distribution per Agent", y=1.03, fontsize=16)
        g.set_axis_labels("Action", "Frequency")
        g.set_titles("Agent {col_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_dir / "action_hist_per_agent.png", dpi=150)
        plt.close()

    def _plot_rtg_curves(self, rtg_df, hue_col, title, save_path, title_prefix, palette=None):
        with self._plot_context(save_path, title, figsize=(12, 7), title_prefix=title_prefix) as (fig, ax):
            sns.lineplot(data=rtg_df, x='timestep', y='rtg_value', hue=hue_col, palette=palette, errorbar=('ci', 95), ax=ax)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Mean Reward-to-Go")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            if rtg_df[hue_col].nunique() > 6:
                ax.legend(title=hue_col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_sliding_reward_histogram(self, reward_sums, title, save_path, title_prefix, reward_threshold=None):
        with self._plot_context(save_path, title, figsize=(10, 8), title_prefix=title_prefix) as (fig, ax):
            sns.histplot(x=reward_sums, bins=30, color='teal', alpha=0.8, ax=ax)
            if reward_threshold is not None:
                ax.axvline(reward_threshold, color='red', linestyle='--', label=f"Threshold = {reward_threshold}")
                ax.legend()
            ax.set_xlabel("Sum of Rewards over Horizon")
            ax.set_ylabel("Count")
            min_val, max_val = int(np.min(reward_sums)), int(np.max(reward_sums))
            step = max(2, int(np.ceil((max_val - min_val) / 25))) if max_val > min_val else 2
            ax.set_xticks(np.arange(min_val, max_val + 1, step))
            ax.tick_params(axis='x', rotation=45, labelsize=10)
    def _plot_time_to_reward_events(self, save_path, title_prefix=""):
        """
        Analyzes and plots the time elapsed between significant reward events,
        ignoring outliers for a clearer view of the main distribution.
        """
        # Define the specific, significant reward values you care about.
        SIGNIFICANT_REWARDS = [3, 5, 20]
        
        print("Analyzing time between significant reward events...")
        
        inter_reward_intervals = {reward: [] for reward in SIGNIFICANT_REWARDS}
        rewards_squeezed = self.rewards.squeeze(-1)

        for i in range(rewards_squeezed.shape[0]):  # For each episode
            non_zero_indices = np.where(rewards_squeezed[i] != 0)
            events = sorted(
                [(t, rewards_squeezed[i, t, a]) for t, a in zip(*non_zero_indices)],
                key=lambda x: x[0]
            )

            last_event_time = 0
            for t, r_val in events:
                if r_val in SIGNIFICANT_REWARDS:
                    interval = t - last_event_time
                    inter_reward_intervals[r_val].append(interval)
                    last_event_time = t

        # Now, create the plot.
        with self._plot_context(save_path, "Time Between Significant Reward Events (Core Distribution)", figsize=(12, 7), title_prefix=title_prefix) as (fig, ax):
            
            data_to_plot = [np.array(inter_reward_intervals[r]) for r in SIGNIFICANT_REWARDS]
            labels = [f"Reward {r}" for r in SIGNIFICANT_REWARDS]

            valid_data = [(d, l) for d, l in zip(data_to_plot, labels) if len(d) > 0]
            if not valid_data:
                print("No significant reward events found to plot.")
                ax.text(0.5, 0.5, "No significant reward events found.", ha='center', va='center')
                return
            data_to_plot_filtered, labels_filtered = zip(*valid_data)
            sns.boxplot(data=list(data_to_plot_filtered), ax=ax, showfliers=False)
            ax.set_xticklabels(labels_filtered)
            ax.set_title("Time Between Significant Reward Events (Core Distribution)", fontsize=16)
            ax.set_xlabel("Reward Event Type")
            ax.set_ylabel("Number of Steps Since Last Event (Outliers Hidden)")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            print("Median time (in steps) to achieve reward:")
            # Note: We still calculate the median on the *full* data, which is correct.
            # We only hide the outliers for visualization.
            for r_val, data in zip(SIGNIFICANT_REWARDS, data_to_plot):
                if len(data) > 0:
                    print(f"  - Reward {r_val}: {np.median(data):.1f} steps")

class IverseDynamicsTester(BaseTester):
    """
    Tests the inverse dynamics model, primarily through online evaluation in the environment.
    """
    def __init__(self, args):
        super().__init__(args)

        self.ground_truth_model = GroundTruthInverseDynamics(args)
        self.learned_idm = self._load_idm_model()
        dataset_wrapper = self._load_dataset(split="train")
        hdf5_dset = dataset_wrapper.hdf5_dataset.dset
        self.observations = np.array(hdf5_dset['obs'])
        self.actions = np.array(hdf5_dset['actions'])
        self.y_true_0, self.y_true_1 = [], []
        self.y_pred_gt_0, self.y_pred_gt_1 = [], []
        self.failures_gt = 0
        self.y_pred_argmax_0 = []
        self.y_pred_categorical_0 = []
        self.num_actions = 6
        self.FULL_ACTION_NAMES = ["NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT"]
    
    def _load_idm_model(self):
        """Load the Inverse Dynamics Model."""
        if not osp.exists(self.args.idm_path):
            raise FileNotFoundError(f"Inverse Dynamics Model path {self.args.idm_path} does not exist.")
        weights = th.load(self.args.idm_path)
        idm = InverseDynamicsModel(num_actions=self.num_actions)
        idm.load_state_dict(weights['model'])
        idm.to(self.device)
        idm.eval()
        return idm
    
    @th.no_grad()
    def run_validation(self, num_samples=200):
        """
        Runs the validation loop over a random sample of transitions from the dataset.
        
        Args:
            num_samples (int): The number of random state transitions to test.
        """
        self.set_experiment_dir("inverse_dynamics_validation", "plots")
        print(f"Running validation on {num_samples} random transitions...")

        num_episodes, episode_len, _, _, _ = self.observations.shape

        for _ in tqdm(range(num_samples), desc="Validating IDM Models"):
            # Select a random episode and a random valid timestep t
            ep_idx = np.random.randint(0, num_episodes)
            t = np.random.randint(0, episode_len - 2)

            obs_t = self.observations[ep_idx, t]
            obs_t_plus_1 = self.observations[ep_idx, t + 1]

            gt_idx_0 = self.actions[ep_idx, t, 0, 0]
            gt_idx_1 = self.actions[ep_idx, t, 1, 0]

            self.y_true_0.append(gt_idx_0)
            self.y_true_1.append(gt_idx_1)

            obs_t = obs_t.astype(np.float32) / 255.0
            obs_t_plus_1 = obs_t_plus_1.astype(np.float32) / 255.0
            gt_pred_actions = self.ground_truth_model.find_actions_between_obs(obs_t, obs_t_plus_1)
            if gt_pred_actions is None:
                grid = self.renderer.extract_grid_from_obs(np.transpose(obs_t, (1, 0, 2)))
                validation_failure_path = self.base_dir / "validation_failures"
                validation_failure_path.mkdir(parents=True, exist_ok=True)
                self.renderer.save_obs_image(
                    np.transpose(obs_t, (1, 0, 2))*255.0,
                    grid,
                    file_path=os.path.join(validation_failure_path, f"failure_episode_{ep_idx}_timestep_{t}_id_{len(self.y_pred_gt_0)}_current.png")
                )
                self.renderer.visualize_all_channels(
                    np.transpose(obs_t, (1, 0, 2)),
                    output_dir=os.path.join(validation_failure_path, f"failure_episode_{ep_idx}_timestep_{t}_id_{len(self.y_pred_gt_0)}_current_channels.png")
                )
                self.renderer.save_obs_image(
                    np.transpose(obs_t_plus_1, (1, 0, 2))*255.0,
                    grid,
                    file_path=os.path.join(validation_failure_path, f"failure_episode_{ep_idx}_timestep_{t+1}_id_{len(self.y_pred_gt_0)}_next.png")
                )
                self.renderer.visualize_all_channels(
                    np.transpose(obs_t_plus_1, (1, 0, 2)),
                    output_dir=os.path.join(validation_failure_path, f"failure_episode_{ep_idx}_timestep_{t+1}_id_{len(self.y_pred_gt_0)}_next_channels.png")
                )
                self.failures_gt += 1
                self.y_pred_gt_0.append(-1) # Use -1 as FAIL index
                self.y_pred_gt_1.append(-1)
            else:
                self.y_pred_gt_0.append(Action.ACTION_TO_INDEX[gt_pred_actions[0]])
                self.y_pred_gt_1.append(Action.ACTION_TO_INDEX[gt_pred_actions[1]])
            obs_t_normed = normalize_obs(obs_t, divide=False)
            obs_t_plus_1_normed = normalize_obs(obs_t_plus_1, divide=False)

            predicted_logits = self.learned_idm(to_torch(obs_t_normed).unsqueeze(0), to_torch(obs_t_plus_1_normed).unsqueeze(0))
            pred_argmax_action = th.argmax(predicted_logits, dim=-1).cpu().item()
            self.y_pred_argmax_0.append(pred_argmax_action)
            
            # Categorical sampling
            probs = th.softmax(predicted_logits, dim=-1)
            dist = th.distributions.Categorical(probs=probs)
            pred_categorical_action = dist.sample().cpu().item()
            self.y_pred_categorical_0.append(pred_categorical_action)

        print("\n--- Validation Complete ---")
        self._save_results_and_reports()
    def _save_results_and_reports(self):
        """Calculates all metrics, prints them, saves them to JSON, and generates visual reports."""
        # Step 1: Calculate all the metrics and store them in a dictionary
        metrics = self.calculate_all_metrics()

        # Step 2: Print the report to the console
        self.print_metrics_report(metrics)

        # Step 3: Save the metrics dictionary to a JSON file
        save_path = self.base_dir / "validation_metrics.json"
        print(f"Saving aggregated metrics to {save_path}...")
        serializable_metrics = self._make_json_serializable(metrics)
        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        print("...metrics saved successfully.")

        # Step 4: Generate the visual plots
        self.generate_visual_reports()
        
    def calculate_all_metrics(self):
        """Computes all accuracy and per-class metrics and returns them in a structured dict."""
        total_samples = len(self.y_true_0)
        if total_samples == 0: return {}

        gt_metrics_0 = self._get_per_class_accuracy(self.y_true_0, self.y_pred_gt_0, is_gt=True)
        gt_metrics_1 = self._get_per_class_accuracy(self.y_true_1, self.y_pred_gt_1, is_gt=True)
        argmax_metrics_0 = self._get_per_class_accuracy(self.y_true_0, self.y_pred_argmax_0)
        categorical_metrics_0 = self._get_per_class_accuracy(self.y_true_0, self.y_pred_categorical_0)

        metrics = {
            "metadata": {
                "total_samples": total_samples,
                "gt_search_failures": self.failures_gt,
                "gt_search_failure_rate": self.failures_gt / total_samples
            },
            "ground_truth_search": {
                "agent_0": gt_metrics_0,
                "agent_1": gt_metrics_1
            },
            "learned_idm_argmax": {
                "agent_0": argmax_metrics_0
            },
            "learned_idm_categorical": {
                "agent_0": categorical_metrics_0
            }
        }
        return metrics
    def _get_per_class_accuracy(self, y_true, y_pred, is_gt=False):
        """Calculates overall and per-class accuracy for a given set of predictions."""
        labels = list(range(self.num_actions))
        
        # Filter out failed predictions for GT model before calculating recall
        true_filtered, pred_filtered = (y_true, y_pred)
        if is_gt:
            true_filtered, pred_filtered = [], []
            for t, p in zip(y_true, y_pred):
                if p != -1: # Only include successful predictions
                    true_filtered.append(t)
                    pred_filtered.append(p)
        
        # Calculate recall (per-class accuracy) and support (counts)
        _, r, _, s = precision_recall_fscore_support(true_filtered, pred_filtered, labels=labels, zero_division=0)

        # action_names = [Action.ACTION_TO_CHAR[Action.INDEX_TO_ACTION[i]] for i in labels]
        action_names = [self.FULL_ACTION_NAMES[i] for i in labels]
        if is_gt: action_names.append("FAIL")
        
        return {
            "overall_accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true),
            "per_class_accuracy": {name: recall for name, recall in zip(action_names, r)},
            "class_counts": {name: int(count) for name, count in zip(action_names, s)}
        }
    
    def print_metrics_report(self, metrics):
        """Prints a formatted report to the console from the metrics dictionary."""
        print("\n" + "="*60)
        print("--- ACCURACY REPORT ---")
        print(f"Total Transitions Tested: {metrics['metadata']['total_samples']}")
        print("="*60)

        gt_metrics = metrics['ground_truth_search']
        print("\n[1] Ground Truth (Search-Based) Model:")
        print(f"  - Failures (no action found): {metrics['metadata']['gt_search_failures']} ({metrics['metadata']['gt_search_failure_rate']:.2%})")
        print(f"  - Agent 0 Accuracy: {gt_metrics['agent_0']['overall_accuracy']:.2%}")
        print(f"  - Agent 1 Accuracy: {gt_metrics['agent_1']['overall_accuracy']:.2%}")

        argmax_metrics = metrics['learned_idm_argmax']
        print("\n[2] Learned IDM (Argmax Sampling):")
        print(f"  - Agent 0 Accuracy: {argmax_metrics['agent_0']['overall_accuracy']:.2%}")

        categorical_metrics = metrics['learned_idm_categorical']
        print("\n[3] Learned IDM (Categorical Sampling):")
        print(f"  - Agent 0 Accuracy: {categorical_metrics['agent_0']['overall_accuracy']:.2%}")
        print("="*60)

    def generate_visual_reports(self):
        """Orchestrates the creation of multiple, focused plot files."""
        # We still need the raw lists for plotting confusion matrices
        self._plot_gt_model_report()
        self._plot_gt_accuracy_report()
        self._plot_learned_idm_cm_report()
        self._plot_learned_idm_accuracy_report()
    
    def _plot_gt_accuracy_report(self):
        """Plots the per-class accuracy bar charts for the Ground Truth search model."""
        save_path = self.base_dir / "gt_accuracy_report.png"
        print(f"Generating Ground Truth model per-class accuracy report -> {save_path}")
        fig, axes = plt.subplots(1, 2, figsize=(22, 8), sharey=True)
        fig.suptitle("Ground Truth Model Per-Class Accuracy", fontsize=20)

        # Get the metrics dictionaries to plot
        gt_metrics_0 = self._get_per_class_accuracy(self.y_true_0, self.y_pred_gt_0, is_gt=True)
        gt_metrics_1 = self._get_per_class_accuracy(self.y_true_1, self.y_pred_gt_1, is_gt=True)

        self._plot_accuracy_bars(axes[0], gt_metrics_0, "Agent 0")
        self._plot_accuracy_bars(axes[1], gt_metrics_1, "Agent 1")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def _plot_gt_model_report(self):
        """Plots the confusion matrices for the Ground Truth search model."""
        save_path = self.base_dir / "gt_model_report.png"
        print(f"Generating Ground Truth model report -> {save_path}")
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle("Ground Truth (Search-Based) Model Confusion Matrix", fontsize=20)
        self._plot_cm(axes[0], self.y_true_0, self.y_pred_gt_0, "Agent 0", is_gt=True)
        self._plot_cm(axes[1], self.y_true_1, self.y_pred_gt_1, "Agent 1", is_gt=True)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def _plot_learned_idm_cm_report(self):
        """Plots the confusion matrices for the learned IDM's sampling methods."""
        save_path = self.base_dir / "learned_idm_cm_report.png"
        print(f"Generating Learned IDM confusion matrix report -> {save_path}")
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle("Learned IDM Confusion Matrix (Ego Agent)", fontsize=20)
        self._plot_cm(axes[0], self.y_true_0, self.y_pred_argmax_0, "Argmax Sampling")
        self._plot_cm(axes[1], self.y_true_0, self.y_pred_categorical_0, "Categorical Sampling")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def _plot_learned_idm_accuracy_report(self):
        """Plots the per-class accuracy bar charts for the learned IDM."""
        save_path = self.base_dir / "learned_idm_accuracy_report.png"
        print(f"Generating Learned IDM per-class accuracy report -> {save_path}")
        fig, axes = plt.subplots(1, 2, figsize=(22, 8), sharey=True)
        fig.suptitle("Learned IDM Per-Class Accuracy (Ego Agent)", fontsize=20)
        
        # Get the metrics dictionaries to plot
        argmax_metrics = self._get_per_class_accuracy(self.y_true_0, self.y_pred_argmax_0)
        categorical_metrics = self._get_per_class_accuracy(self.y_true_0, self.y_pred_categorical_0)
        
        self._plot_accuracy_bars(axes[0], argmax_metrics, "Argmax Sampling")
        self._plot_accuracy_bars(axes[1], categorical_metrics, "Categorical Sampling")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def _plot_accuracy_bars(self, ax, metrics, title):
        """Helper to plot a single per-class accuracy bar chart from a metrics dict."""
        accuracy_data = metrics['per_class_accuracy']
        class_names = list(accuracy_data.keys())
        accuracies = list(accuracy_data.values())
        
        sns.barplot(x=class_names, y=accuracies, ax=ax)
        ax.set_title(title, fontsize=18)
        ax.set_ylabel("Accuracy (Recall)", fontsize=14)
        ax.set_xlabel("Action Class", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=30)
        
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 0.02, f'{acc:.1%}', ha='center', color='black')

    def _plot_cm(self, ax, y_true, y_pred, title, is_gt=False):
        """Helper to plot a single confusion matrix with readable labels."""
        labels = list(range(self.num_actions))
        cm_labels = labels + [-1] if is_gt else labels
        action_name_map = {
            (0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST",
            (0, 0): "STAY", 'interact': "INTERACT"
        }
        display_labels = []
        for i in cm_labels:
            if i == -1:
                display_labels.append("FAIL")
            else:
                action_tuple = Action.INDEX_TO_ACTION[i]
                display_labels.append(action_name_map.get(action_tuple, "OTHER"))
        
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=display_labels, yticklabels=display_labels)
        ax.set_title(title, fontsize=18)
        ax.set_ylabel("True Action", fontsize=14)
        ax.set_xlabel("Predicted Action", fontsize=14)
        ax.tick_params(axis='x', rotation=45)

    def _save_results_to_json(self):
        """Saves the raw prediction and ground truth lists to a JSON file."""
        save_path = self.base_dir / "validation_results.json"
        print(f"Saving raw results to {save_path}...")

        results_data = {
            "ground_truth": {"agent_0": self.y_true_0, "agent_1": self.y_true_1},
            "predictions": {
                "ground_truth_search": {"agent_0": self.y_pred_gt_0, "agent_1": self.y_pred_gt_1},
                "learned_idm_argmax": {"agent_0": self.y_pred_argmax_0},
                "learned_idm_categorical": {"agent_0": self.y_pred_categorical_0}
            },
            "metadata": {"total_samples": len(self.y_true_0), "gt_search_failures": self.failures_gt}
        }
        serializable_results = self._make_json_serializable(results_data)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print("...results saved successfully.")
    
class RewardModelValidator(BaseTester):
    """
    Compares a learned reward model against a ground-truth calculator for EGO AGENT ONLY.
    Calculates regression and classification metrics, generates a truth vs. prediction
    scatter plot, and saves all results.
    """
    def __init__(self, args):
        super().__init__(args)
        print("--- Initializing Reward Model Validator ---")

        # Models to be tested
        self.gt_reward_calculator = GroundTruthRewardCalculator(args)
        self.learned_reward_model = self._load_reward_model()

        # Load data once
        dataset_wrapper = self._load_dataset(split="train")
        hdf5_dset = dataset_wrapper.hdf5_dataset.dset
        self.observations = np.array(hdf5_dset['obs'])
        self.actions = np.array(hdf5_dset['actions'])
        self.rewards = np.array(hdf5_dset['rewards'])

        # Data accumulators
        self.y_true_ego_rewards = []
        self.y_pred_learned = []
        self.y_pred_gt_ego = []

    def _load_reward_model(self):
        """Loads the learned RewardPredictor model."""
        if not Path(self.args.value_model_path).exists():
            raise FileNotFoundError(f"Reward model path {self.args.reward_model_path} does not exist.")
        weights = th.load(self.args.value_model_path, map_location=self.device)
        model = RewardPredictor()
        model.load_state_dict(weights)
        model.to(self.device)
        model.eval()
        return model

    @th.no_grad()
    def run_validation(self, num_samples=500):
        """
        Runs the validation loop, getting EGO AGENT reward predictions from both models.
        """
        self.set_experiment_dir("reward_model_validation", "plots")
        print(f"Running validation on {num_samples} random transitions...")

        num_episodes, episode_len, _, _, _ = self.observations.shape

        for _ in tqdm(range(num_samples), desc="Validating Reward Models"):
            ep_idx = np.random.randint(0, num_episodes)
            t = np.random.randint(0, episode_len - 1)

            obs_t = self.observations[ep_idx, t] # Ego obs
            obs_tp1 = self.observations[ep_idx, t + 1]
            
            gt_reward_ego = self.rewards[ep_idx, t, 0].sum() # Agent 0's reward
            self.y_true_ego_rewards.append(gt_reward_ego)
            
            gt_action_0 = Action.INDEX_TO_ACTION[self.actions[ep_idx, t, 0, 0]]
            gt_action_1 = Action.INDEX_TO_ACTION[self.actions[ep_idx, t, 1, 0]]
            joint_action = (gt_action_0, gt_action_1)
            
            gt_pred_ego, _ = self.gt_reward_calculator.calculate_reward(
                obs_t.astype(np.float32) / 255.0, obs_tp1.astype(np.float32) / 255.0, joint_action=joint_action, max_steps=self.max_steps
            )
            self.y_pred_gt_ego.append(gt_pred_ego)
            
            obs_t_torch = max_normalize_obs(obs_t)
            obs_tp1_torch = max_normalize_obs(obs_tp1)

            obs_t_torch = convert_to_binary_obs(obs_t_torch)
            obs_tp1_torch = convert_to_binary_obs(obs_tp1_torch)
            
            obs_t_torch = th.from_numpy(obs_t_torch).to(self.device).unsqueeze(0).float()
            obs_tp1_torch = th.from_numpy(obs_tp1_torch).to(self.device).unsqueeze(0).float()
            pred_reward = self.learned_reward_model(obs_t_torch, obs_tp1_torch).cpu().item()
            
            self.y_pred_learned.append(pred_reward)

        print("\n--- Validation Complete ---")
        self._save_and_report_metrics()
    
    def _save_and_report_metrics(self):
        """Calculates, prints, saves, and plots all metrics."""
        metrics = self.calculate_all_metrics()
        self.print_metrics_report(metrics)
        
        save_path = self.base_dir / "reward_validation_metrics.json"
        print(f"Saving aggregated metrics to {save_path}...")
        with open(save_path, 'w') as f:
            json.dump(self._make_json_serializable(metrics), f, indent=4)
        print("...metrics saved successfully.")

        self.generate_visual_reports()

    def calculate_all_metrics(self):
        """Computes regression and classification metrics for all models."""
        true = np.array(self.y_true_ego_rewards)
        pred_gt = np.array(self.y_pred_gt_ego)
        pred_learned = np.array(self.y_pred_learned)

        # --- MODIFIED: Calculate sparse accuracy for BOTH models ---
        gt_sparse_accuracy = self._calculate_sparse_reward_accuracy(true, pred_gt)
        learned_sparse_accuracy = self._calculate_sparse_reward_accuracy(true, pred_learned)

        metrics = {
            "metadata": {"total_samples": len(true)},
            "ground_truth_calculator": {
                "vs_dataset_ego_mse": mean_squared_error(true, pred_gt),
                "vs_dataset_ego_mae": mean_absolute_error(true, pred_gt),
                "sparse_reward_detection_accuracy": gt_sparse_accuracy
            },
            "learned_model": {
                "vs_dataset_ego_mse": mean_squared_error(true, pred_learned),
                "vs_dataset_ego_mae": mean_absolute_error(true, pred_learned),
                "sparse_reward_detection_accuracy": learned_sparse_accuracy
            }
        }
        return metrics

    def _calculate_sparse_reward_accuracy(self, y_true, y_pred):
        """Helper to calculate detection accuracy for key sparse rewards."""
        key_rewards = [3, 5, 20, 25]
        accuracy_dict = {}
        for r in key_rewards:
            # How often did the model predict a value close to r when the true reward was r?
            # We use rounding for robustness.
            true_is_r = (np.round(y_true) == r)
            pred_is_r = (np.round(y_pred) == r)
            correct_detections = np.sum(true_is_r & pred_is_r)
            total_instances = np.sum(true_is_r)
            accuracy = correct_detections / total_instances if total_instances > 0 else 0.0
            accuracy_dict[f'detect_{r}'] = accuracy
        return accuracy_dict

    def print_metrics_report(self, metrics):
        """Prints a formatted report to the console for all models."""
        print("\n" + "="*60)
        print("--- REWARD MODEL ACCURACY REPORT (EGO AGENT) ---")
        print(f"Total Transitions Tested: {metrics['metadata']['total_samples']}")
        print("="*60)
        
        # --- MODIFIED: Print sparse accuracy for GT model ---
        gt_metrics = metrics['ground_truth_calculator']
        print("\n[1] Ground Truth Calculator (vs. Dataset):")
        print(f"  - Ego Reward MSE: {gt_metrics['vs_dataset_ego_mse']:.4f}")
        print(f"  - Ego Reward MAE: {gt_metrics['vs_dataset_ego_mae']:.4f}")
        print("  - Sparse Reward Detection Accuracy:")
        for key, val in gt_metrics['sparse_reward_detection_accuracy'].items():
            reward_val = key.split('_')[-1]
            print(f"    - Accuracy for detecting reward '{reward_val}': {val:.2%}")

        learned_metrics = metrics['learned_model']
        print("\n[2] Learned Reward Model (vs. Dataset):")
        print(f"  - Ego Reward MSE: {learned_metrics['vs_dataset_ego_mse']:.4f}")
        print(f"  - Ego Reward MAE: {learned_metrics['vs_dataset_ego_mae']:.4f}")
        print("  - Sparse Reward Detection Accuracy:")
        for key, val in learned_metrics['sparse_reward_detection_accuracy'].items():
            reward_val = key.split('_')[-1]
            print(f"    - Accuracy for detecting reward '{reward_val}': {val:.2%}")
        print("="*60)

    def generate_visual_reports(self):
        """Generates and saves focused plot files for all models."""
        # --- MODIFIED: Create a bar chart for the GT model as well ---
        self._plot_truth_vs_pred_scatter(self.y_pred_gt_ego, "Ground Truth Calculator vs. Dataset Reward", "gt_truth_vs_pred_scatter.png")
        self._plot_truth_vs_pred_scatter(self.y_pred_learned, "Learned Model vs. Dataset Reward", "learned_truth_vs_pred_scatter.png")
        
        self._plot_sparse_reward_accuracy(
            metrics_dict=self.calculate_all_metrics()['ground_truth_calculator'],
            title="Ground Truth: Sparse Reward Accuracy",
            save_filename="gt_sparse_reward_accuracy.png"
        )
        self._plot_sparse_reward_accuracy(
            metrics_dict=self.calculate_all_metrics()['learned_model'],
            title="Learned Model: Sparse Reward Accuracy",
            save_filename="learned_sparse_reward_accuracy.png"
        )

    def _plot_truth_vs_pred_scatter(self, y_pred, title, save_filename):
        """
        Plots a scatter plot of true rewards vs. a given set of predicted rewards.
        """
        save_path = self.base_dir / save_filename
        print(f"Generating scatter plot -> {save_path}")

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(title, fontsize=20)
        
        sns.scatterplot(x=self.y_true_ego_rewards, y=y_pred, alpha=0.5, ax=ax, s=50, edgecolor=None)
        
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Prediction")
        
        ax.set_xlabel("True Ego Reward (from Dataset)", fontsize=14)
        ax.set_ylabel("Predicted Ego Reward", fontsize=14)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def _plot_sparse_reward_accuracy(self, metrics_dict, title, save_filename):
        """Plots a bar chart of the sparse reward detection accuracy from a metrics dict."""
        save_path = self.base_dir / save_filename
        print(f"Generating sparse reward accuracy bar chart -> {save_path}")

        accuracy_data = metrics_dict['sparse_reward_detection_accuracy']
        class_names = [f"Reward {k.split('_')[-1]}" for k in accuracy_data.keys()]
        accuracies = list(accuracy_data.values())

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        fig.suptitle(title, fontsize=18)
        
        sns.barplot(x=class_names, y=accuracies, ax=ax)
        ax.set_ylabel("Detection Accuracy", fontsize=14)
        ax.set_xlabel("Sparse Reward Value", fontsize=14)
        ax.set_ylim(0, 1.05)
        
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 0.02, f'{acc:.1%}', ha='center', color='black')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    

class WorldModelTester(BaseTester):
    """
    Systematically tests the coherency of a learned world model by comparing its
    predicted states against the ground-truth states from the dataset.
    """
    def __init__(self, args, num_classes=68, ema=True, guidance_weight=1.0):
        super().__init__(args)
        print("--- Initializing World Model Coherency Tester ---")

        # Load models and data
        self.action_horizon = 8
        self.no_op_action = 4 # Assuming STAY is index 4
        self.world_model = self._load_world_model(args.diffusion_model_path, ema, num_classes, guidance_weight)
        if hasattr(self.world_model, 'ema_model'):
            self.world_model = self.world_model.ema_model
        self.world_model.eval()
        self.idm = GroundTruthInverseDynamics(args)

        # Load dataset from HDF5
        dataset_wrapper = self._load_dataset(split="train")
        hdf5_dset = dataset_wrapper.hdf5_dataset.dset
        self.observations = np.array(hdf5_dset['obs'])
        self.actions = np.array(hdf5_dset['actions'])
        self.policy_ids_raw = np.array(hdf5_dset['policy_id'])

        self.matches_per_timestep = {i: [] for i in range(self.action_horizon + 1)}


    def _load_world_model(self, model_path, ema=True, num_classes=None, guidance_weight=1.0):
        """Load the diffusion model for evaluation only."""
        print(f"Loading diffusion model from {model_path}, ema={ema}, num_classes={num_classes}")
        if not osp.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        ckpt = th.load(model_path, map_location="cpu")
        unet = UnetOvercooked(
            horizon=self.args.horizon,
            obs_dim=(self.H, self.W, self.C),
            num_classes=num_classes,
            num_actions=self.num_actions,
            action_horizon=self.action_horizon,
        ).to(self.device)
        H,W,C = self.H, self.W, self.C
        diffusion = GoalGaussianDiffusion(
            model=unet,
            channels=C * self.horizon,
            image_size=(H,W),
            timesteps=1 if self.args.debug else 1000,
            sampling_timesteps=100,
            loss_type="l2",
            objective="pred_v",
            beta_schedule="cosine",
            min_snr_loss_weight=True,
            guidance_weight=guidance_weight,
            auto_normalize=False,
            num_actions=self.num_actions,
            no_op_action=self.no_op_action
        ).to(self.device)
        if ema:
            ema_wrap = EMA(diffusion,beta = 0.999,update_every=10)
            ema_wrap.load_state_dict(ckpt['ema'])
            return ema_wrap
        else:
            diffusion.load_state_dict(ckpt['model'])
            return diffusion

    @th.no_grad()
    def run_controllability_test(self, num_samples=500):
        """
        The main test function that samples random sequences from the dataset,
        generates predictions from the world model, and compares them against
        the ground truth observations.
        """
        self.set_experiment_dir("world_model_coherency", f"plots_{num_samples}")
        print(f"Running world model coherency test on {num_samples} random sequences.")

        num_episodes, episode_len, *_ = self.observations.shape

        for sample_idx in tqdm(range(num_samples), desc="Testing Coherency"):
            ep_idx = np.random.randint(0, num_episodes)
            t = np.random.randint(0, episode_len - self.action_horizon - 1)
            
            obs_t_raw = self.observations[ep_idx, t]
            action_seq_gt_indices = self.actions[ep_idx, t : t + self.action_horizon, 0, 0]
            gt_frames = self.observations[ep_idx, t : t + self.action_horizon + 1]
            policy_id = self.policy_ids_raw[ep_idx, 1]

            obs_t_th = th.from_numpy(normalize_obs(obs_t_raw)).to(self.device).unsqueeze(0).float()
            actions_th = th.from_numpy(action_seq_gt_indices).to(self.device).long().unsqueeze(0)
            policy_id_th = th.tensor([policy_id], device=self.device).long()

            pred_states_th = self.world_model.sample(
                action_embed=actions_th,
                x_cond=rearrange(obs_t_th, "b h w c -> b c h w"),
                task_embed=policy_id_th,
                batch_size=1,
            )
            pred_states = rearrange(pred_states_th, "b (f c) h w -> b f h w c", c=self.C, f=self.horizon)
            pred_states_np = pred_states.cpu().numpy()

            predicted_obs_sequence = [obs_t_raw.astype(np.float32) / 255.0] + \
                                     [unnormalize_obs(frame) for frame in pred_states_np[0][:self.action_horizon]]
            predicted_states = [self.idm.invert_obs_to_state(frame) for frame in predicted_obs_sequence]
            gt_states = [self.idm.invert_obs_to_state(frame.astype(np.float32) / 255.0) for frame in gt_frames]

            for i in range(self.action_horizon + 1):
                is_match = self.idm.fuzzy_state_equal_for_player(gt_states[i], predicted_states[i], player_idx=0, compare_world_objects=False)
                self.matches_per_timestep[i].append(is_match)
            if sample_idx % 250 == 0:
                accuracy = np.mean([np.mean(self.matches_per_timestep[i]) for i in range(self.action_horizon + 1)])
                accuracy_per_class = {f't+{i}': np.mean(self.matches_per_timestep[i]) for i in range(self.action_horizon + 1)}
                print(f"\n--- Sample {sample_idx + 1}/{num_samples} ---")
                print(f"Current accuracy: {accuracy:.2%}")
                for t, acc in accuracy_per_class.items():
                    print(f"  - Accuracy at {t}: {acc:.2%}")
        self._report_and_plot_results()
    @th.no_grad()
    def run_offline_divergence_test(self, num_samples=500, rollout_horizon=16, num_candidates=10):
        """
        Measures the distributional divergence of the World Model's predictions
        against the ground-truth future from the dataset.
        """
        experiment_name = f"offline_divergence_test_h{rollout_horizon}_c{num_candidates}"
        self.set_experiment_dir(experiment_name)
        
        if rollout_horizon > self.horizon:
            raise ValueError(f"Rollout horizon {rollout_horizon} exceeds model horizon {self.horizon}.")

        self.divergence_per_step = {t: [] for t in range(rollout_horizon)}
        
        num_episodes, episode_len, *_ = self.observations.shape

        for _ in tqdm(range(num_samples), desc="Running Offline Divergence Test"):
            ep_idx = np.random.randint(0, num_episodes)
            t = np.random.randint(0, episode_len - rollout_horizon - 1)
            
            # Sample Data from Dataset ---
            initial_obs_raw = self.observations[ep_idx, t]
            ego_action_seq = self.actions[ep_idx, t : t + self.action_horizon, 0, 0]
            ground_truth_frames = self.observations[ep_idx, t + 1 : t + rollout_horizon + 1]
            policy_id = self.policy_ids_raw[ep_idx, 1]

            # Generate N "Imagined" Trajectory Candidates ---
            initial_obs_norm = normalize_obs(initial_obs_raw)
            
            obs_t_th = th.from_numpy(initial_obs_norm).to(self.device).unsqueeze(0).float()
            actions_th = th.from_numpy(ego_action_seq).to(self.device).long().unsqueeze(0)
            policy_id_th = th.tensor([policy_id], device=self.device).long()
            
            # Repeat conditioning inputs to create a batch for sampling N candidates
            obs_t_th_batch = obs_t_th.repeat(num_candidates, 1, 1, 1)
            actions_th_batch = actions_th.repeat(num_candidates, 1)
            policy_id_th_batch = policy_id_th.repeat(num_candidates)
            
            pred_states_th = self.world_model.sample(
                action_embed=actions_th_batch,
                x_cond=rearrange(obs_t_th_batch, "b h w c -> b c h w"),
                task_embed=policy_id_th_batch,
                batch_size=num_candidates,
            )
            pred_states = rearrange(pred_states_th, "b (f c) h w -> b f h w c", c=self.C, f=self.horizon)
            
            # We process each of the N trajectories individually.
            imagined_trajectories_np = []
            for traj_idx in range(num_candidates):
                # Get one predicted trajectory (T, H, W, C)
                single_pred_traj = pred_states[traj_idx].cpu().numpy()
                # Unnormalize it frame by frame
                unnormalized_frames = [unnormalize_obs(frame) for frame in single_pred_traj]
                imagined_trajectories_np.append(np.array(unnormalized_frames))
            
            # The result is a list of N trajectories, let's stack them into one array
            imagined_trajectories_np = np.stack(imagined_trajectories_np, axis=0) # Shape: [N, rollout_horizon, H, W, C]


            # Compare Each Candidate to the Single Ground Truth ---
            real_states = self.idm.invert_obs_to_state_batch(ground_truth_frames.astype(np.float32) / 255.0)

            for i in range(rollout_horizon):
                real_state = real_states[i]
                distances_at_step_i = []
                
                # Get the i-th frame from all N imagined trajectories
                imagined_obs_at_step_i = imagined_trajectories_np[:, i, ...]
                imagined_states_at_step_i = self.idm.invert_obs_to_state_batch(imagined_obs_at_step_i)

                for imagined_state in imagined_states_at_step_i:
                    distance = self.calculate_state_mismatch_count(imagined_state, real_state)
                    distances_at_step_i.append(distance)
                
                self.divergence_per_step[i].append(distances_at_step_i)

        # --- 4. Report Results ---
        self._report_distributional_divergence_results()
    def calculate_state_mismatch_count(self, state1, state2):
        """
        Calculates an objective, unweighted distance between two states by
        counting the number of discrete differences. This is ideal for measuring
        divergence, as it doesn't use subjective planning weights.

        Args:
            state1 (OvercookedState): The first state.
            state2 (OvercookedState): The second state.

        Returns:
            int: The total count of differences between the two states.
        """
        mismatch_count = 0

        # Use a set to find players that don't have an exact match in the other state.
        # The `PlayerState` object's __eq__ method compares position, orientation, and held_object.
        players1 = set(state1.players)
        players2 = set(state2.players)

        # Count players in state1 not in state2, and vice-versa.
        # If they are identical, the symmetric difference will be empty.
        # If one player is different (e.g., wrong orientation), it will appear in the set.
        # If a player disappears, it will also appear.
        player_differences = players1.symmetric_difference(players2)
        mismatch_count += len(player_differences)
        
        # World objects are those not held by any player.
        # The `state.objects` dictionary already represents this correctly.
        
        # Find positions that are in one state but not the other
        object_positions1 = set(state1.objects.keys())
        object_positions2 = set(state2.objects.keys())
        
        added_or_removed_objects = object_positions1.symmetric_difference(object_positions2)
        mismatch_count += len(added_or_removed_objects)
        
        # For objects at the same position, check if they are identical
        common_positions = object_positions1.intersection(object_positions2)
        for pos in common_positions:
            obj1 = state1.objects[pos]
            obj2 = state2.objects[pos]
            
            # The default __eq__ for objects and soups should be sufficient here.
            # It will compare ingredients, readiness, etc.
            if obj1 != obj2:
                mismatch_count += 1
                
        return mismatch_count
    
    def _report_distributional_divergence_results(self):
        """Calculates and plots richer statistics for the distributional divergence test."""
        
        # We will now have mean, min, and std for our plots
        mean_divergence, min_divergence, std_divergence = [], [], []
        timesteps = sorted(self.divergence_per_step.keys())

        for t in timesteps:
            # data is now a list of lists: [[d1..dN], [d1..dN], ...]
            data_for_step = self.divergence_per_step.get(t, [])
            if data_for_step:
                # Flatten the list of lists into a single list of all distances for this step
                all_distances_at_t = np.array(data_for_step).flatten()
                mean_divergence.append(np.mean(all_distances_at_t))
                std_divergence.append(np.std(all_distances_at_t))
                
                # Calculate the mean of the minimums (best-case performance)
                min_distances_per_sample = [np.min(sample_distances) for sample_distances in data_for_step]
                min_divergence.append(np.mean(min_distances_per_sample))
            else:
                mean_divergence.append(0); std_divergence.append(0); min_divergence.append(0)

        print("\n" + "="*60)
        print("--- Distributional Divergence Test Report ---")
        print("="*60)
        for i, t in enumerate(timesteps):
            print(f"  - Timestep T+{t+1}: "
                f"Mean Dist = {mean_divergence[i]:.2f}, "
                f"Best-Case Dist = {min_divergence[i]:.2f}, "
                f"Std Dev = {std_divergence[i]:.2f}")
        print("="*60)

        # --- Plotting ---
        plot_timesteps = np.array(timesteps) + 1
        plt.figure(figsize=(12, 7))
        
        # Plot Mean Divergence
        plt.plot(plot_timesteps, mean_divergence, marker='o', linestyle='-', color='b', label='Mean Divergence')
        plt.fill_between(
            plot_timesteps,
            np.array(mean_divergence) - np.array(std_divergence),
            np.array(mean_divergence) + np.array(std_divergence),
            color='b', alpha=0.15, label='Mean ± 1 Std. Dev.'
        )
        
        # Plot Best-Case Divergence
        plt.plot(plot_timesteps, min_divergence, marker='*', linestyle='--', color='g', label='Best-Case Divergence (Mean of Mins)')
        
        plt.title("Distributional Divergence of World Model Over Time", fontsize=18)
        plt.xlabel("Prediction Timestep (T+k)", fontsize=14)
        plt.ylabel("State Distance", fontsize=14)
        plt.xticks(plot_timesteps)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        save_path = self.base_dir / "distributional_divergence.png"
        plt.savefig(save_path, dpi=150)
        print(f"Divergence plot saved to {save_path}")
        plt.close()
    
    def _report_and_plot_divergence_results(self):
        """
        Calculates, prints, and plots the summary statistics for a divergence test.
        This function processes the `self.divergence_per_step` dictionary.
        """
        # Check if any data was collected
        if not self.divergence_per_step or not any(self.divergence_per_step.values()):
            print("No divergence data collected. Skipping report.")
            return

        avg_divergence = []
        std_divergence = []
        timesteps = sorted(self.divergence_per_step.keys())

        # Calculate average and standard deviation for each step in the horizon
        for t in timesteps:
            distances = self.divergence_per_step.get(t, [])
            if distances:
                avg_divergence.append(np.mean(distances))
                std_divergence.append(np.std(distances))
            else:
                # Handle cases where a timestep might not have data (e.g., if all episodes ended early)
                avg_divergence.append(0)
                std_divergence.append(0)

        # --- Print a clear text report to the console ---
        print("\n\n" + "="*60)
        print("--- World Model Divergence Test Report ---")
        print("="*60)
        print(f"Evaluated over {len(self.divergence_per_step[0])} samples.")
        print("Average state distance between predicted and actual states over time:")
        
        for i, t in enumerate(timesteps):
            avg = avg_divergence[i]
            std = std_divergence[i]
            print(f"  - Timestep T+{t+1}: {avg:.3f} ± {std:.3f}")
            
        print("="*60)

        # --- Generate a plot visualizing the divergence ---
        avg_divergence = np.array(avg_divergence)
        std_divergence = np.array(std_divergence)
        # Create x-axis labels as T+1, T+2, ...
        plot_timesteps = np.array(timesteps) + 1 
        
        plt.figure(figsize=(12, 7))
        # Main line for the average distance
        plt.plot(plot_timesteps, avg_divergence, marker='o', linestyle='-', color='b', label='Mean State Distance')
        
        # Shaded region for the standard deviation
        plt.fill_between(
            plot_timesteps,
            avg_divergence - std_divergence,
            avg_divergence + std_divergence,
            color='b',
            alpha=0.15,
            label='Standard Deviation'
        )
        
        plt.title("World Model Divergence Over Prediction Horizon", fontsize=18, pad=20)
        plt.xlabel("Prediction Timestep (T+k)", fontsize=14)
        plt.ylabel("Average State Distance (Lower is Better)", fontsize=14)
        plt.xticks(plot_timesteps)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save the plot to the experiment directory
        save_path = self.base_dir / "divergence_over_time.png"
        plt.savefig(save_path, dpi=150)
        print(f"Divergence plot saved to {save_path}")
        plt.close()

    def render_debug_video(self, obs_t_raw, pred_states_np_traj, ep_idx, t, policy_id):
        """Helper to render a video for one trajectory."""
        obs_t_grid = obs_t_raw.transpose(1, 0, 2).astype(np.float32)
        grid = self.renderer.extract_grid_from_obs(obs_t_grid)
        
        pred_states_print = pred_states_np_traj.transpose(0, 2, 1, 3) # W, H -> H, W
        full_video = [obs_t_grid] + [frame for frame in pred_states_print]

        video_path = os.path.join(self.base_dir, f"trajectory_ep{ep_idx}_t{t}_id{policy_id}.mp4")
        self.renderer.render_trajectory_video(full_video, grid, self.base_dir, video_path, fps=2, normalize=True)
    
    def _report_and_plot_results(self):
        """Calculates accuracy and generates plots for the state coherency test."""
        print("\n" + "="*60)
        print("--- WORLD MODEL COHERENCY REPORT ---")
        print("="*60)

        total_matches = 0
        total_comparisons = 0
        
        accuracy_per_step = {}
        for i in range(self.action_horizon + 1):
            matches = np.sum(self.matches_per_timestep[i])
            count = len(self.matches_per_timestep[i])
            
            total_matches += matches
            total_comparisons += count
            
            accuracy = (matches / count) if count > 0 else 0
            accuracy_per_step[f't+{i}'] = accuracy
            print(f"  - Accuracy at T+{i}: {accuracy:.2%}")

        overall_accuracy = (total_matches / total_comparisons) if total_comparisons > 0 else 0
        print("-" * 60)
        print(f"  - Overall State Match Accuracy: {overall_accuracy:.2%}")
        print(f"  - Total States Compared: {total_comparisons}")
        
        self._plot_accuracy_over_time(accuracy_per_step)

    def _plot_accuracy_over_time(self, accuracy_data):
        """Plots a bar chart of state match accuracy over the prediction horizon."""
        save_path = self.base_dir / "coherency_accuracy_over_time.png"
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        timesteps = list(accuracy_data.keys())
        accuracies = list(accuracy_data.values())
        
        sns.barplot(x=timesteps, y=accuracies, ax=ax)
        
        ax.set_title("World Model Coherency: Accuracy vs. Timestep", fontsize=20)
        ax.set_ylabel("State Match Accuracy", fontsize=14)
        ax.set_xlabel("Prediction Timestep", fontsize=14)
        ax.set_ylim(0, 1.05)
        
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 0.02, f'{acc:.1%}', ha='center', color='black', weight='bold')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved accuracy plot to {save_path}")
        plt.close(fig)     

class DiffusionAgentTester(BaseTester):
    def __init__(self, args, run_name, num_classes=68, num_actions=6, guidance_weight=1.0, num_candidates=10, num_processes=1, planning_horizon=32, device=None):
        super().__init__(args)

        self.device = device if device else th.device("cuda" if th.cuda.is_available() else "cpu")
        self.args = args
        self.action_horizon = 8
        self.no_op_action = 4 
        self.num_classes = num_classes
        self.num_actions = num_actions
        self.guidance_weight = guidance_weight
        self.planning_horizon = planning_horizon
        self.run_name = run_name

        self.num_candidates = num_candidates
        self.num_processes = num_processes
        self.policy_name_to_id = {"actor_best_r_vs_bc_train": 1, "bc_train": 2, "comedi_best": 3, "comedi_mp0": 4, "comedi_mp1": 5, "comedi_mp2": 6, "comedi_mp3": 7, "comedi_mp4": 8, "comedi_mp5": 9, "comedi_mp6": 10, "comedi_mp7": 11, "mep1_final": 12, "mep1_init": 13, "mep1_mid": 14, "mep2_final": 15, "mep2_init": 16, "mep2_mid": 17, "mep3_final": 18, "mep3_init": 19, "mep3_mid": 20, "mep4_final": 21, "mep4_init": 22, "mep4_mid": 23, "mep5_final": 24, "mep5_init": 25, "mep5_mid": 26, "mep6_final": 27, "mep6_init": 28, "mep6_mid": 29, "mep7_final": 30, "mep7_init": 31, "mep7_mid": 32, "mep8_final": 33, "mep8_init": 34, "mep8_mid": 35, "mep_best": 36, "sp10_final": 37, "sp10_init": 38, "sp10_mid": 39, "sp1_final": 40, "sp1_init": 41, "sp1_mid": 42, "sp2_final": 43, "sp2_init": 44, "sp2_mid": 45, "sp3_final": 46, "sp3_init": 47, "sp3_mid": 48, "sp4_final": 49, "sp4_init": 50, "sp4_mid": 51, "sp5_final": 52, "sp5_init": 53, "sp5_mid": 54, "sp6_final": 55, "sp6_init": 56, "sp6_mid": 57, "sp7_final": 58, "sp7_init": 59, "sp7_mid": 60, "sp8_final": 61, "sp8_init": 62, "sp8_mid": 63, "sp9_final": 64, "sp9_init": 65, "sp9_mid": 66, "sp_best": 67}
        self.agent = None
        self.H, self.W, self.C = 8, 5, 26

        self._init_agent()

    def _load_world_model(self, model_path, ema=True, num_classes=None, guidance_weight=1.0):
        """Load the diffusion model for evaluation only."""
        print(f"Loading diffusion model from {model_path}, ema={ema}, num_classes={num_classes}")
        if not osp.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        ckpt = th.load(model_path, map_location="cpu")
        unet = UnetOvercooked(
            horizon=self.args.horizon,
            obs_dim=(self.H, self.W, self.C),
            num_classes=num_classes,
            num_actions=self.num_actions,
            action_horizon=self.action_horizon,
        ).to(self.device)
        H,W,C = self.H, self.W, self.C
        diffusion = GoalGaussianDiffusion(
            model=unet,
            channels=C * self.horizon,
            image_size=(H,W),
            timesteps=1 if self.args.debug else 1000,
            sampling_timesteps=100,
            loss_type="l2",
            objective="pred_v",
            beta_schedule="cosine",
            min_snr_loss_weight=True,
            guidance_weight=guidance_weight,
            auto_normalize=False,
            num_actions=self.num_actions,
            no_op_action=self.no_op_action
        ).to(self.device)
        if ema:
            ema_wrap = EMA(diffusion,beta = 0.999,update_every=10)
            ema_wrap.load_state_dict(ckpt['ema'])
            return ema_wrap
        else:
            diffusion.load_state_dict(ckpt['model'])
            return diffusion
    def _load_action_proposal_model(self,ema=True):
        """Load the action proposal model."""
        if not osp.exists(self.args.action_proposal_model_path):
            raise FileNotFoundError(f"Action proposal model path {self.args.action_proposal_model_path} does not exist.")
        ckpt = th.load(self.args.action_proposal_model_path, map_location="cpu")
        unet = UnetOvercookedActionProposal(
            horizon=self.args.horizon,
            obs_dim=(self.H, self.W, self.C),
            num_actions=self.num_actions,
        ).to(self.device)
        diffusion = GoalGaussianDiffusion(
            model=unet,
            channels=self.num_actions * self.horizon,
            image_size=(1,1),
            timesteps=1000,
            sampling_timesteps=100,
            loss_type="l2",
            objective="pred_v",
            beta_schedule="cosine",
            min_snr_loss_weight=True,
            guidance_weight=getattr(self.args, 'guidance_weight', 1.0),
            auto_normalize=False,
        ).to(self.device)
        if ema:
            ema_wrap = EMA(diffusion,beta = 0.999,update_every=10)
            ema_wrap.load_state_dict(ckpt['ema'])
            return ema_wrap
        else:
            diffusion.load_state_dict(ckpt['model'])
            return diffusion
    
    def _init_agent(self):
        self.world_model = self._load_world_model(
            self.args.diffusion_model_path, 
            ema=True, 
            num_classes=self.num_classes, 
            guidance_weight=self.guidance_weight
        )
        if hasattr(self.world_model, 'ema_model'):
            self.world_model = self.world_model.ema_model
        self.world_model.eval()
        self.action_proposal = self._load_action_proposal_model(ema=True)
        if hasattr(self.action_proposal, 'ema_model'):
            self.action_proposal = self.action_proposal.ema_model
        self.action_proposal.eval()

        self.agent = DiffusionPlannerAgent(
            args=self.args,
            world_model=self.world_model,
            action_proposal_model=self.action_proposal,
            planning_horizon=self.planning_horizon,
            action_horizon=self.action_horizon,
            num_candidates= self.num_candidates,
            num_processes=self.num_processes,
        )
    def run_evaluation(self, num_episodes, partner_policy_name="bc_train"):
        experiment_name = f"eval_{self.agent.__class__.__name__}_vs_{partner_policy_name}"
        self.set_experiment_dir(experiment_name, self.run_name)

        print(f"\n--- Starting Evaluation ---")
        print(f"Agent to Test: {self.agent.__class__.__name__}")
        print(f"Partner Policy: {partner_policy_name}")
        print(f"Number of Episodes: {num_episodes}")
        print("---------------------------")
        
        all_episode_data = self.evaluate_in_env(
            state_action_fn=self._agent_action_function,
            num_episodes=num_episodes,
            policy_name=partner_policy_name
        )

        # After all episodes, calculate and report the summary
        episode_rewards = [data['episode_reward'] for data in all_episode_data]
        summary = self.calculate_evaluation_summary(episode_rewards)
        
        # Generate the results plot using the inherited method
        self.plot_base_model_results([summary], agent_label=partner_policy_name)
        
        return summary
    def _agent_action_function(self, batched_obs_np, policy_name):
        """
        This is the "bridge" function that matches the state_action_fn signature
        required by the evaluate_in_env method in the BaseTester.

        It takes a batch of observations, normalizes them, and calls the agent's
        get_plan method to get a sequence of actions.

        Args:
            batched_obs_np (np.ndarray): A batch of ego-agent observations,
                                         shape [nenvs, H, W, C].

        Returns:
            np.ndarray: A plan of actions for the batch, shape [nenvs, horizon].
        """

        batch_size = batched_obs_np.shape[0]

        if policy_name not in self.policy_name_to_id:
            raise ValueError(f"Unknown policy name: {policy_name}. Available policies: {list(self.policy_name_to_id.keys())}")
        policy_id = self.policy_name_to_id.get(policy_name)
        policy_id_np = np.full((batch_size,), policy_id, dtype=np.int64)
        action_plan = self.agent.get_plan(batched_obs_np, policy_id_np)
        return action_plan
