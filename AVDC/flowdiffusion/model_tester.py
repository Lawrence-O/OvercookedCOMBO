import argparse
from collections import defaultdict
import datetime
import json
import os
from pathlib import Path
import pickle
import random
import sys

import pygame
from tqdm import tqdm

mapbt_path = '/home/law/Workspace/repos/COMBO/mapbt_package'
if mapbt_path not in sys.path:
    sys.path.append(mapbt_path)

from einops.einops import rearrange
import numpy as np
import torch as th
import warnings
warnings.filterwarnings("ignore")
from goal_diffusion import GoalGaussianDiffusion, ConceptTrainer
from unet import UnetOvercooked 
from experiments_util import to_np, normalize_obs, make_eval_env, to_torch, load_partner_policy, get_idm_action, convert_to_binary_obs
from overcooked_sample_renderer import OvercookedSampleRenderer
from mapbt_package.mapbt.config import get_config
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from reward_module.reward_model import RewardPredictor
import os.path as osp
from idm.inverse_dynamics import InverseDynamicsModel
from overcooked_dataset import OvercookedSequenceDataset
from unet import UnetOvercooked, UnetOvercookedActionProposal
from ema_pytorch import EMA
from model_test_util import analyze_ambiguity_direct

class ModelTester:
    def __init__(self, args):
        self.args = args

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.renderer = OvercookedSampleRenderer()
        self.n_envs = args.n_envs
        self.max_steps = args.max_steps
        self.horizon = args.horizon
        self.num_action_classes = 6  # Overcooked default
        self.H, self.W, self.C = 8, 5, 26  # Overcooked obs shape
        self.num_actions = 6
        self.action_horizon = 8
        self.no_op_action = 4
        self.n_envs = args.n_envs
        self.data = None
        self.value_min, self.value_max = 0.0, 25.0
        self.base_dir = Path("model_tester")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.world_model = None

        self.set_seed()

    def set_seed(self, seed=42):
        """Set random seed for reproducibility."""
        th.manual_seed(seed)
        np.random.seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)
    def collect_data(self, num_episodes=10):
        """Test the value model by running episodes and collecting rewards."""
        all_data = []
        bc_policy, _ = load_partner_policy(self.args, "bc_train", device="cpu")
        ego_poolicy, _ = load_partner_policy(self.args, "actor_best_r_vs_bc_train", device="cpu")
        envs = make_eval_env(args, run_dir=args.run_dir, nenvs=self.n_envs)
        envs.reset_featurize_type([("ppo", "bc") for _ in range(self.n_envs)])
        for episode in tqdm(range(num_episodes), desc="Collecting data"):
            
            # Reset Policy
            bc_policy.reset(num_envs=self.n_envs, num_agents=1)
            ego_poolicy.reset(num_envs=self.n_envs, num_agents=1)
            for e in range(self.n_envs):
                ego_poolicy.register_control_agent(e=e, a=0)
                bc_policy.register_control_agent(e=e, a=1)
            
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
                agent0_actions = ego_poolicy.step(
                    agent0_obs,
                    [(e, 0) for e in range(self.n_envs)],
                    deterministic=False,
                )
                step_actions[:, 0] = agent0_actions

                # Agent 1 actions using policy
                agent1_obs_lst = [obs[e][1] for e in range(self.n_envs)]
                agent1_obs = np.stack(agent1_obs_lst, axis=0)
                agent1_actions = bc_policy.step(
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
                actions_list.append((agent0_actions, agent1_actions))
                for e in range(min(self.args.n_envs, 3)):
                    frames[e].append(obs[e][0])
                done = np.all(done)
                steps += 1
                if steps >= self.args.max_steps:
                    break
            print(f"Episode {episode + 1}/{num_episodes} completed with total reward: {episode_reward.sum(axis=0) // self.n_envs}")
            all_data.append({
                "obs_t": obs_t_list,
                "obs_tp1": obs_tp1_list,
                "actions": actions_list,
                "rewards": rewards_list,
                "done": done_list,
                "steps": step_list,
                "frames": frames,
                "episode_reward": episode_reward,
            })
            if episode == num_episodes - 1:
                video_dir = self.base_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                self._save_episode_videos(frames, episode, video_dir)
        envs.close()
        self.data = all_data
        return all_data
    def evaluate_in_env(self, state_action_fn, num_episodes=10, policy_name="bc_train"):
        """Test the value model by running episodes and collecting rewards.
        Supports both single-step and planning horizon action outputs.
        """
        all_data = []
        bc_policy, _ = load_partner_policy(self.args, policy_name, device="cpu")
        envs = make_eval_env(self.args, run_dir=self.args.run_dir, nenvs=self.n_envs)
        if policy_name == "bc_train":
            envs.reset_featurize_type([("ppo", "bc") for _ in range(self.n_envs)])
        for episode in tqdm(range(num_episodes), desc="Collecting data"):
            # Reset Policy
            bc_policy.reset(num_envs=self.n_envs, num_agents=1)
            for e in range(self.n_envs):
                bc_policy.register_control_agent(e=e, a=1)
            
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
                agent0_actions = state_action_fn(agent0_obs)

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
                        agent1_actions = bc_policy.step(
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
                    # --- Single-step fallback (original logic) ---
                    step_actions[:, 0, 0] = agent0_actions
                    # Agent 1 actions using policy
                    agent1_obs_lst = [obs[e][1] for e in range(self.n_envs)]
                    agent1_obs = np.stack(agent1_obs_lst, axis=0)
                    agent1_actions = bc_policy.step(
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
        envs.close()
        return all_data
    def calculate_evaluation_summary(self, episode_rewards_list):
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
    def plot_base_model_results(self, results, agent_label="sp_best"):
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
    def save_data(self, filepath, data=None):
        """Save collected data to disk using pickle."""
        if data is None:
            data = self.data
        base_data_path = self.base_dir / "data" 
        base_data_path.mkdir(parents=True, exist_ok=True)
        filepath = base_data_path / filepath
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Data saved to {filepath}")
    def load_data(self, filepath):
        """Load previously saved data from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"Data loaded from {filepath}")
        self.data = data
        return data
    def _un_normalize_value(self, value):
        """Un-normalize the value predictions."""
        return value * (self.value_max - self.value_min) + self.value_min
    def _load_value_model(self):
        """Load the value predictor model."""
        if not osp.exists(self.args.value_model_path):
            raise FileNotFoundError(f"Value model path {self.args.value_model_path} does not exist.")
        weights = th.load(self.args.value_model_path)
        value_model = RewardPredictor()
        value_model.load_state_dict(weights)
        value_model.to(self.device)
        value_model.eval()
        return value_model
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
    def _load_dataset(self, split="train"):
        dataset_args = argparse.Namespace(
            dataset_path=self.args.dataset_path,
            horizon=self.args.horizon,
            max_path_length=self.args.max_path_length,
            episode_length=self.args.episode_length,
            chunk_length=self.args.chunk_length,
            use_padding=self.args.use_padding,
        )
        return OvercookedSequenceDataset(
            args=dataset_args, split=split, allowed_policies=None,
        )
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
    
    @th.no_grad()
    def get_value_model_data(self, data):
        """Evaluate the value model on collected data."""
        value_preds = []
        rewards = []
        self.value_model.eval()
        for episode, episode_data in enumerate(tqdm(data, desc="Evaluating Value Model")):
            obs_t = episode_data["obs_t"]
            obs_tp1 = episode_data["obs_tp1"]
            rewards_list = episode_data["rewards"]

            for i, (obs_t_i, obs_tp1_i, r) in enumerate(zip(obs_t, obs_tp1, rewards_list)):
                # Normalize observations
                
                obs_t_i = np.array(obs_t_i[0]) # ego obs
                obs_tp1_i = np.array(obs_tp1_i[0])

                obs_0_norm = normalize_obs(obs_t_i)
                obs_1_norm = normalize_obs(obs_tp1_i)

                # Convert to binary observations
                obs_0_bin = convert_to_binary_obs(obs_0_norm)
                obs_1_bin = convert_to_binary_obs(obs_1_norm)

                # Get value predictions
                obs_0_bin = to_torch(obs_0_bin, device=self.device)
                obs_1_bin = to_torch(obs_1_bin, device=self.device)

                value_pred = self.value_model(obs_0_bin, obs_1_bin).squeeze(-1).detach().cpu().numpy()
                value_preds.append(value_pred)
                rewards.append(r)
            
        value_preds = np.array(value_preds)
        value_preds = self._un_normalize_value(value_preds)
        rewards = np.array(rewards)
        rewards = rewards[..., 0] # Ego Reward only
        mse = np.mean((value_preds - rewards) ** 2)
        mae = np.mean(np.abs(value_preds - rewards))
        print("=" * 40)
        print("Value Model Evaluation Summary:")
        print(f"  Mean Reward: {np.mean(rewards):.4f}")
        print(f"  Mean Predicted Value: {np.mean(value_preds):.4f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print("=" * 40)  

        return {
            "mse": mse,
            "mae": mae,
            "value_preds": value_preds,
            "actual_rewards": rewards,
            "mean_reward": np.mean(rewards),
            "mean_value_pred": np.mean(value_preds),
        }
    def compute_input_gradients(self, obs, next_obs, target_action):
        obs = obs.clone().detach().requires_grad_(True)
        next_obs = next_obs.clone().detach().requires_grad_(True)

        logits = self.idm(obs, next_obs)
        selected_logits = logits[:, target_action]

        self.idm.zero_grad()
        # Backward pass to compute gradients
        selected_logits.sum().backward()

        grad_obs = obs.grad.abs().mean(dim=[0, 1, 2])  # avg over batch & spatial dims, shape: (C,)
        grad_next_obs = next_obs.grad.abs().mean(dim=[0, 1, 2])
        return grad_obs.detach().cpu().numpy(), grad_next_obs.detach().cpu().numpy()
    def get_idm_model_data(self, data):
        """
        Evaluate the IDM model on collected data.
        For each step, predict the ego agent's action given (obs_t, obs_tp1) and compare to ground truth.
        """
        all_preds = []
        all_targets = []
        input_grads = defaultdict(list)  # Store input gradients for analysis
        self.idm.eval()
        for episode, episode_data in enumerate(tqdm(data, desc="Evaluating IDM Model")):
            obs_t = episode_data["obs_t"]
            obs_tp1 = episode_data["obs_tp1"]
            actions_list = episode_data["actions"]

            for i, (obs_t_i, obs_tp1_i, actions) in enumerate(zip(obs_t, obs_tp1, actions_list)):
                # Ego agent (agent 0)
                obs_t_ego = np.array(obs_t_i[0])
                obs_tp1_ego = np.array(obs_tp1_i[0])
                action_ego = np.array(actions[0])  # shape: (n_envs, 1)

                # Normalize and convert to binary if needed (match IDM training)
                obs_t_ego_norm = normalize_obs(obs_t_ego)
                obs_tp1_ego_norm = normalize_obs(obs_tp1_ego)
                obs_t_ego_bin = convert_to_binary_obs(obs_t_ego_norm)
                obs_tp1_ego_bin = convert_to_binary_obs(obs_tp1_ego_norm)

                # Convert to torch tensors
                obs_t_tensor = to_torch(obs_t_ego_bin, device=self.device)
                obs_tp1_tensor = to_torch(obs_tp1_ego_bin, device=self.device)

                # if True:
                #     other_agent_channels = [1, 6, 7, 8, 9]  # player_1_loc and player_1_orientation_*
                #     obs_t_tensor[..., other_agent_channels] = 0.0
                #     obs_tp1_tensor[..., other_agent_channels] = 0.0

                # IDM predicts action given (obs_t, obs_tp1)
                pred_logits = self.idm(obs_t_tensor, obs_tp1_tensor)  # shape: (n_envs, num_actions)
                # TODO: Try out with softmax
                pred_actions = pred_logits.argmax(dim=-1).cpu().numpy()  # shape: (n_envs,)

                # Store input gradients for analysis
                for action in range(self.num_actions):
                    grad_obs, grad_next_obs = self.compute_input_gradients(obs_t_tensor, obs_tp1_tensor, action)
                    input_grads[action].append((grad_obs, grad_next_obs))

                # Flatten and store
                all_preds.append(pred_actions.flatten())
                all_targets.append(action_ego.flatten())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        accuracy = np.mean(all_preds == all_targets)
        print("=" * 40)
        print("IDM Model Evaluation Summary:")
        print(f"  Accuracy: {accuracy:.4f}")
        print("=" * 40)

        return {
            "accuracy": accuracy,
            "pred_actions": all_preds,
            "true_actions": all_targets,
            "input_gradients": input_grads,
        }
    
    @th.no_grad()
    def evaluate_action_proposal_in_env(self, num_episodes=10, planning_horizon=32, policy_name="sp_best"):
        self.action_proposal_model.eval()
        def action_proposal_fn(obs):
            """
            Given an observation, use the action proposal model to predict the next action.
            """
            obs = normalize_obs(obs)
            obs_tensor = to_torch(obs, device=self.device)
            obs_tensor = obs_tensor.view(self.n_envs, self.C, self.H, self.W)
            pred = self.action_proposal_model.sample(
                x_cond=obs_tensor,
                batch_size=self.n_envs,
            )
            pred = pred.view(self.n_envs, self.horizon, self.num_actions)
            pred = th.argmax(pred, dim=-1)  # [N, L]
            action = pred[:, :planning_horizon].cpu().numpy()  # Use only the first `planning_horizon` actions
            print(f"Action proposal shape: {action.shape}")
            return action
        # Evaluate in environment
        eval_data = self.evaluate_in_env(action_proposal_fn, num_episodes=num_episodes, policy_name=policy_name)
        return eval_data

    def extract_obs_action_tuples(self, dataset, ego_player_id=0):
        for ep in range(len(dataset)):
            obs = dataset.observations[ep]
            actions = dataset.actions[ep].squeeze(-1) # [T, num_players, 1] -> [T, num_players]
            ep_len = obs.shape[0] - 1  # Last observation is not used for action prediction
            for t in range(ep_len):
                obs_t = obs[t]
                obs_tp1 = obs[t+1]
                action_t = actions[t, ego_player_id]
                yield (obs_t, obs_tp1, action_t)
    def run_ambiguity_analysis(self, dataset, ego_player_id=0):
        tuples = self.extract_obs_action_tuples(dataset, ego_player_id=ego_player_id)
        results = analyze_ambiguity_direct(tuples, True)
        save_dir = self.base_dir / "ambiguity_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "ambiguity_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Ambiguity analysis results saved to {save_dir / 'ambiguity_analysis_results.json'}")

    def load_diffusion_model(self, model_path, ema=True, num_classes=None):
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
            sampling_timesteps=1 if self.args.debug else 100,
            loss_type="l2",
            objective="pred_v",
            beta_schedule="cosine",
            min_snr_loss_weight=True,
            guidance_weight=getattr(self.args, 'guidance_weight', 1.0),
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

    def evaluate_world_model(self, obs, policy_id, actions=None):
        predefined_actions = [
            # [0]*8, # Move Up
            # [1]*8, # Move Down
            # [2]*8, # Move Right
            # [3]*8, # Move Left
            # [4]*8, # No Op Action
            # [np.random.randint(0, self.num_actions) for _ in range(8)],
            actions,
        ]
        print(predefined_actions)
        if obs.ndim == 4:
            obs = obs.squeeze(0)
        obs = to_np(obs)
        obs_print = np.transpose(obs, (1, 0, 2))
        grid = self.renderer.extract_grid_from_obs(obs_print)
        eval_path = self.base_dir / "world_model_eval" / f"policy_{policy_id}"
        eval_path.mkdir(parents=True, exist_ok=True)
        self.renderer.save_obs_image(
            obs_print,
            grid,
            file_path=str(eval_path / "obs_image.png"),
            normalize=True,
        )
        self.renderer.visualize_all_channels(
                obs_print,  # First frame
                output_dir=str(eval_path / f"x_cond_policy_{policy_id}.png"),
            )
        self.renderer.visualize_all_channels(
            obs_print,  # First frame
            output_dir=str(eval_path / f"x_cond_channels_policy_{policy_id}_action_channels.png"),
        )
        obs = to_torch(obs, device=self.device).float()
        # obs = normalize_obs(obs)
        obs = obs.view(self.C, self.H, self.W)
        obs = obs.unsqueeze(0)  # Add batch dimension
        for i, action_condition in enumerate(predefined_actions):
            action_cond = th.tensor(action_condition, device=self.device).long()  # Add batch dimension
            policy_id = th.tensor([policy_id], device=self.device).long()
            trajectory = self.world_model.sample(
                    x_cond=obs,
                    action_embed=action_cond,
                    batch_size=1,
                    task_embed=policy_id)
            trajectory = rearrange(trajectory, "b (f c) h w -> b f w h c", c=self.C, f=self.horizon)
            self.renderer.render_trajectory_video(
                to_np(trajectory[0]),
                grid,
                output_dir=eval_path,
                video_path=str(eval_path / f"policy_{policy_id}_action_{i}.mp4"),
                fps=1,
                normalize=True,
            )
            first_frame = rearrange(trajectory, "b f w h c -> b f h w c")
            self.renderer.visualize_all_channels(
                to_np(first_frame[0, 0]),  # First frame
                output_dir=str(eval_path / f"policy_{policy_id}_action_{i}_channels.png"),
            )
            
    def plot_action_proposal_model_debug(self, metrics, title_prefix=""):
        summary = {}
        for L, data in metrics.items():
            pred_actions = data["pred_actions"][..., :L].cpu().numpy()
            true_actions = data["true_actions"]
            if true_actions.shape[-1] == 1:
                true_actions = np.squeeze(true_actions, axis=-1)

            assert pred_actions.shape == true_actions.shape, f"Shape mismatch: {pred_actions.shape} vs {true_actions.shape}"

            accuracy = (pred_actions == true_actions).mean()
            summary[L] = {
                "action_accuracy": float(accuracy)
            }

            stepwise_acc = (pred_actions == true_actions).astype(np.float32).mean(axis=0)  # [L]
            summary[L]["stepwise_accuracy"] = stepwise_acc.tolist()

            plots_dir = self.base_dir / "plots" / "action_proposal"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Plot stepwise accuracy
            plt.figure(figsize=(8, 4))
            plt.plot(stepwise_acc, marker='o', color='teal')
            plt.title(f"{title_prefix}Stepwise Accuracy (Len={L})")
            plt.xlabel("Step")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{title_prefix}stepwise_accuracy_L{L}.png")
            plt.close()

            # Plot confusion matrix heatmap
            conf_mat = self.manual_confusion_matrix(true_actions, pred_actions, self.num_actions)
            plt.figure(figsize=(8, 6))
            plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"{title_prefix}Confusion Matrix (Len={L})")
            plt.colorbar()
            tick_marks = np.arange(self.num_actions)
            plt.xticks(tick_marks, tick_marks)
            plt.yticks(tick_marks, tick_marks)
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')

            # Annotate counts on heatmap
            thresh = conf_mat.max() / 2.
            for i in range(self.num_actions):
                for j in range(self.num_actions):
                    plt.text(j, i, format(conf_mat[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if conf_mat[i, j] > thresh else "black")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{title_prefix}confusion_matrix_L{L}.png")
            plt.close()

            # 4. Diversity: Unique predicted sequences
            unique_seqs = len(set(tuple(seq) for seq in pred_actions))
            diversity_score = unique_seqs / len(pred_actions)
            summary[L]["sequence_diversity"] = float(diversity_score)

            print(f"  Accuracy: {accuracy:.3f} | Diversity: {diversity_score:.3f} | Unique Seqs: {unique_seqs}")
        with open(plots_dir / f"{title_prefix}action_proposal_eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def plot_idm_model_debug(self, metrics, bins=6, title_prefix=""):
        """
        Plots debugging charts for IDM model evaluation.
        """
        preds = np.array(metrics["pred_actions"])
        targets = np.array(metrics["true_actions"])

        plots_dir = self.base_dir / "plots" / "idm_model"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        num_classes = max(np.max(targets), np.max(preds)) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(targets, preds):
            cm[t, p] += 1
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f"{title_prefix}IDM Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")

        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)

        print(f"Confusion Matrix:\n{cm}")

        # Add counts in each cell
        thresh = cm.max() / 2
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(plots_dir / f"{title_prefix}idm_confusion_matrix.png")
        plt.close()

        # Accuracy per action class
        acc_per_class = [
        (preds[targets == i] == i).mean() if np.sum(targets == i) > 0 else 0
        for i in range(num_classes)
        ]

        plt.figure(figsize=(7, 4))
        plt.bar(range(num_classes), acc_per_class, color='skyblue', edgecolor='black')
        plt.title(f"{title_prefix}IDM Accuracy per Action Class")
        plt.xlabel("Action Class")
        plt.ylabel("Accuracy")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / f"{title_prefix}idm_accuracy_per_class.png")
        plt.close()

        input_grads = metrics["input_gradients"]
        num_actions = len(input_grads)
        grad_obs_means = []
        grad_next_obs_means = []

        for action in range(num_actions):
            grads = input_grads[action]
            if grads:
                grad_obs = np.stack([g[0] for g in grads], axis=0)  # [N, C]
                grad_next_obs = np.stack([g[1] for g in grads], axis=0)
                grad_obs_means.append(grad_obs.mean(axis=0))         # [C]
                grad_next_obs_means.append(grad_next_obs.mean(axis=0))
            else:
                # fallback zeros if no gradients recorded for this action
                grad_obs_means.append(np.zeros_like(grad_obs_means[0]) if grad_obs_means else np.array([]))
                grad_next_obs_means.append(np.zeros_like(grad_next_obs_means[0]) if grad_next_obs_means else np.array([]))

        grad_obs_means = np.stack(grad_obs_means, axis=0)         # [num_actions, C]
        grad_next_obs_means = np.stack(grad_next_obs_means, axis=0)

        channels = np.arange(grad_obs_means.shape[1])

        fig, axs = plt.subplots(2, num_actions, figsize=(4*num_actions, 8), sharey=True)

        for action in range(num_actions):
            axs[0, action].bar(channels, grad_obs_means[action])
            axs[0, action].set_title(f"Obs Grad | Action {action}")
            axs[0, action].set_xlabel("Channel")
            axs[0, action].set_ylabel("Avg |Grad|")

            axs[1, action].bar(channels, grad_next_obs_means[action])
            axs[1, action].set_title(f"Next Obs Grad | Action {action}")
            axs[1, action].set_xlabel("Channel")
            axs[1, action].set_ylabel("Avg |Grad|")

        plt.suptitle(f"{title_prefix}IDM Input Gradient Magnitude per Channel")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle

        plt.savefig(plots_dir / f"{title_prefix}idm_input_gradients.png")
        plt.close()

        print("\n=== IDM Model Evaluation Summary ===")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        summary_path = plots_dir / f"{title_prefix}idm_model_eval_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=== IDM Model Evaluation Summary ===\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
    def manual_confusion_matrix(self, true_actions, pred_actions, num_classes):
        true_flat = true_actions.flatten()
        pred_flat = pred_actions.flatten()
        conf_mat = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(true_flat, pred_flat):
            conf_mat[t, p] += 1
        return conf_mat
    
    def plot_value_model_debug(self, metrics, trajectory_len=32, bins=6, title_prefix=""):
        """
        Plots a suite of debugging charts based on value model evaluation.
        """
        preds = np.array(metrics["value_preds"])
        rewards = np.array(metrics["actual_rewards"])

        preds = np.asarray(preds).squeeze().flatten()
        rewards = np.asarray(rewards).squeeze().flatten()

        print(f"Preds shape: {preds.shape}, Rewards shape: {rewards.shape}")

        assert preds.shape == rewards.shape, "Shape mismatch between preds and rewards"
        plots_dir = self.base_dir / "plots" / "value_model"
        plots_dir.mkdir(parents=True, exist_ok=True)

        episode_len = 400
        num_episodes = (len(rewards) // self.n_envs) // episode_len
        print(f"Number of episodes: {num_episodes}, Episode length: {episode_len}")

        # Reshape and sum per episode
        episode_rewards = rewards[:num_episodes * episode_len].reshape(num_episodes, episode_len)
        episode_preds = preds[:num_episodes * episode_len].reshape(num_episodes, episode_len)

        # ---- 0. Plot total reward per episode ----
        print(f"Plotting total reward per episode")
        plt.figure(figsize=(12, 5))
        plt.plot(episode_rewards.sum(axis=1), marker='o', label='Actual Total Reward', color='blue')
        plt.plot(episode_preds.sum(axis=1), marker='x', label='Predicted Total Reward', color='orange')
        plt.title(f"{title_prefix}Total Reward per Episode (Actual vs Predicted)")
        plt.xlabel("Episode Index")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{title_prefix}reward_per_episode_actual_vs_pred.png")
        plt.close()

        # ---- 1. Scatter plot: Value vs Reward ----
        print(f"Plotting Value vs Reward scatter plot")
        plt.figure(figsize=(6, 5))
        plt.scatter(rewards, preds, alpha=0.4)
        plt.xlabel("Actual Reward")
        plt.ylabel("Predicted Value")
        plt.title(f"{title_prefix}Value Prediction vs Ground Truth")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{title_prefix}value_vs_reward.png")
        plt.close()

        # ---- 2. Residual histogram ----
        print(f"Plotting residuals histogram")
        residuals = preds - rewards
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=30)
        plt.title(f"{title_prefix}Residuals (Prediction - Reward)")
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{title_prefix}residual_histogram.png")
        plt.close()

        # ---- 3. Error vs Reward Magnitude ----
        print(f"Plotting Absolute Error vs Reward Magnitude")
        abs_error = np.abs(preds - rewards)
        plt.figure(figsize=(6, 4))
        plt.scatter(rewards, abs_error, alpha=0.4)
        plt.xlabel("Actual Reward")
        plt.ylabel("Absolute Error")
        plt.title(f"{title_prefix}Error vs Reward Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{title_prefix}error_vs_reward_magnitude.png")
        plt.close()

        # ---- 4. MAE per reward bin ----
        print(f"Plotting Mean Absolute Error per Reward Bin")
        print(type(rewards), type(abs_error))
        print(np.array(rewards).dtype, np.array(abs_error).dtype)
        print(np.array(rewards).shape, np.array(abs_error).shape)
        df = pd.DataFrame({'reward': rewards, 'error': abs_error})
        df['reward_bin'] = pd.cut(df['reward'], bins=np.linspace(min(rewards), max(rewards), bins + 1))
        bin_means = df.groupby('reward_bin')['error'].mean()
        plt.figure(figsize=(7, 4))
        bin_means.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f"{title_prefix}Mean Absolute Error per Reward Bin")
        plt.ylabel("MAE")
        plt.xlabel("Reward Bins")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / f"{title_prefix}mae_per_reward_bin.png")
        plt.close()
        
        # ---- 5. Value vs Reward-to-Go over time ----
        print(f"Plotting Value vs Reward-to-Go over time")
        reward_to_go_episodes = np.flip(np.cumsum(np.flip(episode_rewards, axis=1), axis=1), axis=1)
        preds_to_go_episodes = np.flip(np.cumsum(np.flip(episode_preds, axis=1), axis=1), axis=1)
        plt.figure(figsize=(18, 6))
        cmap_pred = plt.get_cmap('tab20', num_episodes)
        cmap_actual = plt.get_cmap('tab20b', num_episodes)
        for i in range(min(8, num_episodes)):
            plt.plot(preds_to_go_episodes[i], label=f"Predicted To-Go Ep {i+1}", color=cmap_pred(i), linewidth=2)
            plt.plot(reward_to_go_episodes[i], label=f"Actual To-Go Ep {i+1}", color=cmap_actual(i), linewidth=2)

        # Optional: plot average across episodes
        plt.plot(preds_to_go_episodes.mean(axis=0), label="Mean Predicted To-Go", color='red', linewidth=2.5)
        plt.plot(reward_to_go_episodes.mean(axis=0), label="Mean Actual To-Go", color='green', linewidth=2.5)

        plt.title(f"{title_prefix}Value vs Reward-to-Go per Episode")
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{title_prefix}value_vs_reward_to_go_per_episode.png")
        plt.close()

        # ---- 6. Segment-wise total value vs reward ----
        print(f"Plotting Segment-wise Total Value vs Reward")
        num_segments = len(rewards) // trajectory_len
        segment_preds = [np.sum(preds[i * trajectory_len: (i + 1) * trajectory_len]) for i in range(num_segments)]
        segment_actuals = [np.sum(rewards[i * trajectory_len: (i + 1) * trajectory_len]) for i in range(num_segments)]

        plt.figure(figsize=(8, 4))
        plt.plot(segment_preds, label="Predicted Value (per segment)", marker='o')
        plt.plot(segment_actuals, label="Actual Reward (per segment)", marker='x')
        plt.title(f"{title_prefix}Segment-wise Totals (every {trajectory_len} steps)")
        plt.xlabel("Segment Index")
        plt.ylabel("Total Value / Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{title_prefix}segment_value_vs_reward.png")
        plt.close()

        # ---- 7. Print summary stats ----
        print("\n=== Value Model Evaluation Summary ===")
        print(f"MSE:         {metrics['mse']:.4f}")
        print(f"MAE:         {metrics['mae']:.4f}")
        print(f"Mean Reward: {metrics['mean_reward']:.4f}")
        print(f"Mean Value:  {metrics['mean_value_pred']:.4f}")
        summary_path = plots_dir / f"{title_prefix}value_model_eval_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=== Value Model Evaluation Summary ===\n")
            f.write(f"MSE:         {metrics['mse']:.4f}\n")
            f.write(f"MAE:         {metrics['mae']:.4f}\n")
            f.write(f"Mean Reward: {metrics['mean_reward']:.4f}\n")
            f.write(f"Mean Value:  {metrics['mean_value_pred']:.4f}\n")
    def _plot_dataset_data(self, title_prefix=""):
        rewards = self.dataset.rewards
        actions = self.dataset.actions
        policy_ids = self.dataset.policy_id

        print(f"Rewards shape: {rewards.shape}, Actions shape: {actions.shape}, Policy IDs shape: {policy_ids.shape}")
        *_, n_agents = rewards.shape
        rewards = rewards.squeeze(-1)
        actions = actions.squeeze(-1)

        id_to_name = self.dataset.train_partner_policies
        save_dir = self.base_dir / "plots" / "dataset"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 1. Reward Distribution ---
        # Per step
        plt.figure(figsize=(7, 4))
        plt.hist(rewards.flatten(), bins=50, color='skyblue', edgecolor='black')
        plt.title("Reward Distribution (All Steps, All Agents)")
        plt.xlabel("Reward")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reward_hist_per_step.png")
        plt.close()

        # Per episode (sum over steps, mean over agents)
        ep_rewards = rewards.sum(axis=1)  # [n_eps, n_agents]
        ep_rewards_mean = ep_rewards.mean(axis=1)  # [n_eps]
        plt.figure(figsize=(7, 4))
        plt.hist(ep_rewards_mean, bins=40, color='orange', edgecolor='black')
        plt.title("Total Reward per Episode (Mean over Agents)")
        plt.xlabel("Total Reward")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reward_hist_per_episode.png")
        plt.close()

        # Per policy (partner agent, agent 1)
        partner_policy_ids = policy_ids[:, 1]
        partner_policy_names = [id_to_name.get(pid, str(pid)) for pid in partner_policy_ids]
        policy_to_ep_rewards = defaultdict(list)
        for i, pname in enumerate(partner_policy_names):
            policy_to_ep_rewards[pname].append(ep_rewards[i].mean())
        plt.figure(figsize=(10, 5))
        for pname, vals in policy_to_ep_rewards.items():
            plt.hist(vals, bins=30, alpha=0.5, label=pname)
        plt.title("Total Reward per Episode by Partner Policy")
        plt.xlabel("Total Reward")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reward_hist_per_policy.png")
        plt.close()

        # --- 3. Per-Agent Statistics ---
        # Reward per agent
        plt.figure(figsize=(7, 4))
        for agent in range(n_agents):
            plt.hist(ep_rewards[:, agent], bins=40, alpha=0.6, label=f"Agent {agent}")
        plt.title("Total Reward per Episode (Per Agent)")
        plt.xlabel("Total Reward")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reward_hist_per_agent.png")
        plt.close()

        # Action distribution per agent
        plt.figure(figsize=(7, 4))
        for agent in range(n_agents):
            plt.hist(actions[:, :, agent].flatten(), bins=np.arange(7)-0.5, alpha=0.6, label=f"Agent {agent}")
        plt.title("Action Distribution (Per Agent)")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.xticks(range(6))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/action_hist_per_agent.png")
        plt.close()

        # --- 4. Reward vs. Policy Scatter ---
        plt.figure(figsize=(10, 5))
        policy_names_unique = sorted(set(partner_policy_names))
        policy_to_idx = {name: i for i, name in enumerate(policy_names_unique)}
        x = [policy_to_idx[name] for name in partner_policy_names]
        y = ep_rewards_mean
        plt.scatter(x, y, alpha=0.5)
        plt.xticks(list(policy_to_idx.values()), list(policy_to_idx.keys()), rotation=45)
        plt.xlabel("Partner Policy")
        plt.ylabel("Total Reward per Episode")
        plt.title("Reward vs. Partner Policy (Scatter)")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reward_vs_policy_scatter.png")
        plt.close()

        # --- 6. Reward-to-Go Distribution ---
        # For each episode, for each agent, compute reward-to-go at each timestep
        reward_to_go = np.flip(np.cumsum(np.flip(rewards, axis=1), axis=1), axis=1)  # [n_eps, ep_len, n_agents]
        # Plot mean reward-to-go curve per agent
        plt.figure(figsize=(10, 5))
        for agent in range(n_agents):
            plt.plot(reward_to_go[:, :, agent].mean(axis=0), label=f"Agent {agent}")
        plt.title("Mean Reward-to-Go per Timestep (Per Agent)")
        plt.xlabel("Timestep")
        plt.ylabel("Reward-to-Go")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reward_to_go_per_agent.png")
        plt.close()
        # Per policy (mean over episodes for each policy)
        plt.figure(figsize=(10, 5))
        for pname in policy_to_ep_rewards.keys():
            idxs = [i for i, n in enumerate(partner_policy_names) if n == pname]
            if not idxs:
                continue
            mean_rtg = reward_to_go[idxs, :, :].mean(axis=(0, 2))  # mean over episodes and agents
            plt.plot(mean_rtg, label=pname)
        plt.title("Mean Reward-to-Go per Timestep (Per Partner Policy)")
        plt.xlabel("Timestep")
        plt.ylabel("Reward-to-Go")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reward_to_go_per_policy.png")
        plt.close()

        # --- 8. Action Distribution ---
        # Overall
        plt.figure(figsize=(7, 4))
        plt.hist(actions.flatten(), bins=np.arange(7)-0.5, color='purple', edgecolor='black')
        plt.title("Action Distribution (All Agents, All Steps)")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.xticks(range(6))
        plt.tight_layout()
        plt.savefig(f"{save_dir}/action_hist_overall.png")
        plt.close()
        # Per policy
        plt.figure(figsize=(10, 5))
        for pname in policy_to_ep_rewards.keys():
            idxs = [i for i, n in enumerate(partner_policy_names) if n == pname]
            if not idxs:
                continue
            acts = actions[idxs, :, :].flatten()
            plt.hist(acts, bins=np.arange(7)-0.5, alpha=0.5, label=pname)
        plt.title("Action Distribution by Partner Policy")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.xticks(range(6))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/action_hist_per_policy.png")
        plt.close()

        # --- Ego Agent Plots (Agent 0) ---

        ego_rewards = rewards[:, :, 0]  # [n_episodes, ep_len]
        ego_actions = actions[:, :, 0]  # [n_episodes, ep_len]

        # 1. Ego Reward Distribution (per step)
        plt.figure(figsize=(7, 4))
        plt.hist(ego_rewards.flatten(), bins=50, color='green', edgecolor='black')
        plt.title("Ego Agent Reward Distribution (All Steps)")
        plt.xlabel("Reward")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ego_reward_hist_per_step.png")
        plt.close()

        # 2. Ego Reward Distribution (per episode)
        ego_ep_rewards = ego_rewards.sum(axis=1)  # [n_episodes]
        plt.figure(figsize=(7, 4))
        plt.hist(ego_ep_rewards, bins=40, color='lime', edgecolor='black')
        plt.title("Ego Agent Total Reward per Episode")
        plt.xlabel("Total Reward")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ego_reward_hist_per_episode.png")
        plt.close()

        # 3. Ego Action Distribution (all steps)
        plt.figure(figsize=(7, 4))
        plt.hist(ego_actions.flatten(), bins=np.arange(7)-0.5, color='blue', edgecolor='black')
        plt.title("Ego Agent Action Distribution (All Steps)")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.xticks(range(6))
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ego_action_hist_overall.png")
        plt.close()

        # 4. Ego Reward vs. Partner Policy (scatter)
        plt.figure(figsize=(10, 5))
        x = [policy_to_idx[name] for name in partner_policy_names]
        y = ego_ep_rewards
        plt.scatter(x, y, alpha=0.5, color='green')
        plt.xticks(list(policy_to_idx.values()), list(policy_to_idx.keys()), rotation=45)
        plt.xlabel("Partner Policy")
        plt.ylabel("Ego Total Reward per Episode")
        plt.title("Ego Reward vs. Partner Policy (Scatter)")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ego_reward_vs_policy_scatter.png")
        plt.close()

        # 5. Ego Reward per Episode by Partner Policy (histograms)
        ego_policy_to_ep_rewards = defaultdict(list)
        for i, pname in enumerate(partner_policy_names):
            ego_policy_to_ep_rewards[pname].append(ego_ep_rewards[i])
        plt.figure(figsize=(10, 5))
        for pname, vals in ego_policy_to_ep_rewards.items():
            plt.hist(vals, bins=30, alpha=0.5, label=pname)
        plt.title("Ego Total Reward per Episode by Partner Policy")
        plt.xlabel("Total Reward")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ego_reward_hist_per_policy.png")
        plt.close()

        # 6. Ego Reward-to-Go Curve (mean over episodes)
        ego_reward_to_go = np.flip(np.cumsum(np.flip(ego_rewards, axis=1), axis=1), axis=1)  # [n_episodes, ep_len]
        plt.figure(figsize=(10, 5))
        plt.plot(ego_reward_to_go.mean(axis=0), label="Ego Agent")
        plt.title("Ego Agent Mean Reward-to-Go per Timestep")
        plt.xlabel("Timestep")
        plt.ylabel("Reward-to-Go")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ego_reward_to_go.png")
        plt.close()

        print(f"Saved Overcooked dataset diagnostics to {save_dir}")
    def plot_dataset_data(self, title_prefix=""):
        # Data preparation
        rewards = self.dataset.rewards.squeeze(-1)  # [n_ep, ep_len, n_agents]
        actions = self.dataset.actions.squeeze(-1)  # [n_ep, ep_len, n_agents]
        policy_ids = self.dataset.policy_id[:, 1]   # Partner IDs
        id_to_name = self.dataset.train_partner_policies
        partner_names = [id_to_name.get(pid, str(pid)) for pid in policy_ids]
        
        # Focus policies
        focus_policies = ["sp1_final", "sp2_final", "sp3_final", "sp4_final", "sp5_final"]
        
        # Action mapping
        action_names = ['North', 'South', 'East', 'West', 'Stay', 'Interact']
        
        # Setup
        save_dir = self.base_dir / "plots" / "dataset"
        save_dir.mkdir(parents=True, exist_ok=True)
        n_agents = rewards.shape[-1]
        
        # Key metrics
        ep_rewards = rewards.sum(axis=1)  # [n_ep, n_agents]
        ego_ep_rewards = ep_rewards[:, 0]  # Ego rewards
        team_ep_rewards = ep_rewards.mean(axis=1)  # Team rewards
        ego_rewards = rewards[:, :, 0]  # Per-step ego rewards

        # === 1. Ego Reward Overview ===
        plt.figure(figsize=(12, 6))
        
        # Overall distribution
        plt.subplot(121)
        plt.hist(ego_ep_rewards, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
        plt.title("Overall Ego Reward Distribution")
        plt.xlabel("Total Episode Reward")
        plt.ylabel("Count")
        plt.grid(alpha=0.2)
        
        # Focus policy comparison
        plt.subplot(122)
        for policy in focus_policies:
            mask = [name == policy for name in partner_names]
            plt.hist(ego_ep_rewards[mask], bins=30, alpha=0.6, label=policy)
        plt.title("Ego Reward by Partner Policy")
        plt.xlabel("Total Episode Reward")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ego_reward_overview.png", dpi=120)
        plt.close()

        # === 2. Action Analysis ===
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        
        # Overall action distribution
        all_actions = actions.flatten()
        action_counts = [np.sum(all_actions == i) for i in range(6)]
        ax[0].bar(action_names, action_counts, color='teal')
        ax[0].set_title("Overall Action Distribution")
        ax[0].set_ylabel("Count")
        ax[0].set_yscale('log')
        
        # Partner action distribution for focus policies
        partner_actions = actions[:, :, 1]  # Partner is agent 1
        for policy in focus_policies:
            mask = [name == policy for name in partner_names]
            p_actions = partner_actions[mask].flatten()
            action_probs = [np.mean(p_actions == i) for i in range(6)]
            ax[1].plot(action_names, action_probs, 'o-', label=policy, alpha=0.8, markersize=8)
        
        ax[1].set_title("Partner Action Distribution")
        ax[1].set_ylabel("Probability")
        ax[1].legend()
        ax[1].grid(alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/action_analysis.png", dpi=120)
        plt.close()

        # === 3. Reward-to-Go Analysis ===
        plt.figure(figsize=(12, 6))
        
        # Calculate RTG
        ego_rtg = np.flip(np.cumsum(np.flip(ego_rewards, axis=1), axis=1), axis=1)
        
        # Focus policies RTG
        for policy in focus_policies:
            mask = [name == policy for name in partner_names]
            mean_rtg = ego_rtg[mask].mean(axis=0)
            std_rtg = ego_rtg[mask].std(axis=0)
            plt.plot(mean_rtg, label=policy, lw=2)
            plt.fill_between(range(len(mean_rtg)), 
                            mean_rtg - 0.5*std_rtg, 
                            mean_rtg + 0.5*std_rtg, alpha=0.2)
        
        plt.title("Ego Reward-to-Go by Partner Policy")
        plt.xlabel("Timestep in Episode")
        plt.ylabel("Mean Reward-to-Go")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ego_rtg_by_policy.png", dpi=120)
        plt.close()

        # === 4. Team Performance Overview ===
        plt.figure(figsize=(10, 6))
        
        # Boxplot of team rewards by policy type
        policy_types = {}
        for name in set(partner_names):
            if "sp" in name: ptype = "SP"
            elif "mep" in name: ptype = "MEP"
            elif "comedi" in name: ptype = "Comedi"
            else: ptype = "Other"
            policy_types[name] = ptype
        
        # Group rewards by policy type
        type_rewards = {ptype: [] for ptype in set(policy_types.values())}
        for name, reward in zip(partner_names, team_ep_rewards):
            type_rewards[policy_types[name]].append(reward)
        
        # Plot boxplot
        plt.boxplot([type_rewards[ptype] for ptype in sorted(type_rewards.keys())],
                    labels=sorted(type_rewards.keys()))
        plt.title("Team Reward Distribution by Policy Type")
        plt.ylabel("Mean Episode Reward")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/team_reward_by_type.png", dpi=120)
        plt.close()

        print(f"Saved optimized dataset diagnostics to {save_dir}")
    
    def run_dataset_evaluation(self):
        self.dataset = self._load_dataset(split="train")
        # self.plot_dataset_data(title_prefix="sp10_final_bc_test_")
        self.set_experiment_dir("dataset_evaluation")
        self.run_ambiguity_analysis(self.dataset)
       
    def run_value_model_evaluation(self, title_prefix):
        self.value_model = self._load_value_model()
        value_data = self.get_value_model_data(self.data)
        print("Plotting value model evaluation results...")
        self.plot_value_model_debug(value_data, trajectory_len=32, title_prefix=title_prefix)
        print("Value model evaluation completed and plots saved.")
    
    def run_idm_model_evaluation(self,title_prefix):
        self.idm = self._load_idm_model()
        idm_data = self.get_idm_model_data(self.data)
        print("Plotting IDM model evaluation results...")
        self.plot_idm_model_debug(idm_data, title_prefix=title_prefix)
        print("IDM model evaluation completed and plots saved.")
    
    def run_action_proposal_evaluation(self, title_prefix):
        self.action_proposal_model = self._load_action_proposal_model()
        if hasattr(self.action_proposal_model, 'ema_model'):
            self.action_proposal_model = self.action_proposal_model.ema_model
        self.action_proposal_model.to(self.device)
        # self.plot_action_proposal_model_debug(eval_results, title_prefix=title_prefix)

        # Evaluation
        eval_data = self.evaluate_action_proposal_in_env(num_episodes=1, planning_horizon=2, policy_name="sp_best")
        episode_rewards_list = []
        for eval_run in eval_data:
            episode_reward = eval_run["episode_reward"]
            episode_rewards_list.append(episode_reward)
        summary = self.calculate_evaluation_summary(episode_rewards_list)
        self.plot_base_model_results([summary], agent_label="sp_best")
        print("Action proposal model evaluation completed and plots saved.")
    
    def run_world_model_evaluation_new_data(self, title_prefix):
        self.world_model = self.load_diffusion_model(self.args.diffusion_model_path, num_classes=60)
        if hasattr(self.world_model, 'ema_model'):
            self.world_model = self.world_model.ema_model
        num_samples = 1
        # policy_ids = [35, 38, 47, 44, 41]
        policy_ids = [0]

        # Sample random episodes and timesteps for diverse obs
        for sample_idx in range(num_samples):
            # Sample a random episode and timestep
            episode = random.choice(self.data)
            obs_seq = episode["obs_t"]
            action_seq = episode["actions"]

            # Pick a random timestep
            t = random.randint(0, len(obs_seq) - 1)
            obs = obs_seq[t]
            obs = np.array(obs[0])  # Convert to numpy array if needed
            
            # Evaluate for each policy_id
            for policy_id in policy_ids:
                print(f"Evaluating world model with sample {sample_idx}, timestep {t}, policy ID {policy_id}")
                self.evaluate_world_model(obs, policy_id)
    
    def run_world_model_evaluation_with_dataset(self, title_prefix):
        self.world_model = self.load_diffusion_model(self.args.diffusion_model_path, num_classes=60)
        if hasattr(self.world_model, 'ema_model'):
            self.world_model = self.world_model.ema_model
        self.world_model.to(self.device)
        dataset = self._load_dataset(split="train")
        loader = th.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        num_samples = 5

        # Sample random episodes and timesteps for diverse obs
        for i, (x_org, x_cond, task_embed, _) in enumerate(loader):
            actions = th.tensor([4]*8, dtype=th.long).unsqueeze(0)
            x_cond = rearrange(x_cond, "b h w c -> b c h w")
            x_cond = x_cond.to(self.device)
            task_embed = task_embed.to(self.device)
            actions = actions.to(self.device)
            pred_traj = self.world_model.sample(
                x_cond=x_cond,
                task_embed=task_embed,
                action_embed=actions,
                batch_size=1,
            )
            pred_traj = rearrange(pred_traj, "b (f c) h w -> b f h w c", c=26, f=32)
            pred_traj = rearrange(pred_traj, "b f h w c -> b f w h c")
            pred_traj = pred_traj.cpu().numpy()
            viz_output_path = self.base_dir / "world_model_eval"
            pred_traj_path = self.base_dir / "world_model_eval" / f"{title_prefix}_pred_traj_{i}.mp4"
            grid_obs = rearrange(x_org, "b f h w c -> b f w h c").cpu().numpy()[0, 0]
            grid = self.renderer.extract_grid_from_obs(grid_obs)
            self.renderer.render_trajectory_video(
                    pred_traj[0],
                    grid=grid,
                    output_dir=str(viz_output_path),
                    video_path=str(pred_traj_path),
                    fps=1,
                    normalize=True,
                )
            x_cond_print = rearrange(x_cond, "b c h w -> b w h c")
            x_cond_path = self.base_dir / "world_model_eval" / f"{title_prefix}_x_cond_{i}.png"
            self.renderer.save_obs_image(
                x_cond_print.cpu().numpy()[0],
                grid=grid,
                file_path=x_cond_path,
                normalize=True,
            )
            if i >= num_samples - 1:
                break
    def run_obs_roundtrip_test(self, title_prefix):
        # Sample random episodes and timesteps for diverse obs
        def consistent_norm(obs):
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
        num_samples = 1000000
        num_pass = 0
        num_fail = 0
        max_diff = 0.0
        for sample_idx in tqdm(range(num_samples), desc="Running Observation Roundtrip Test"):
            # Sample a random episode and timestep
            episode = random.choice(self.data)
            obs_seq = episode["obs_t"]

            # Pick a random timestep
            t = random.randint(0, len(obs_seq) - 1)
            obs = obs_seq[t]
            obs = np.array(obs[0])  # Convert to numpy array if needed
            obs = obs[0,...]
            norm_obs = consistent_norm(obs)
            assert norm_obs.min() >= -1.0 and norm_obs.max() <= 1.0
            
            denormalized_obs = self.renderer.unnormalize(norm_obs)
            unscaled_obs = obs / 255.0
            close = np.allclose(unscaled_obs, denormalized_obs, atol=1e-5)
            if close:
                num_pass += 1
            else:
                num_fail += 1
                diff = np.abs(unscaled_obs - denormalized_obs).max()
                max_diff = max(max_diff, diff)
                print(f"Sample {sample_idx}: Max diff {diff:.6f} at timestep {t}")
        print(f"[Obs Roundtrip Test] {num_pass}/{num_samples} passed, {num_fail} failed.")
        if num_fail > 0:
            print(f"Max difference in failed cases: {max_diff}")
    def visualize_attention(self, output_dir="attention_visualizations", sample_idx=42):
        from overcooked_sample_renderer import AttentionVisualizer
        self.world_model = self.load_diffusion_model(self.args.diffusion_model_path, num_classes=60)
        if hasattr(self.world_model, 'ema_model'):
            self.world_model = self.world_model.ema_model
        self.world_model.eval()
        self.world_model.to(self.device)
        dataset = self._load_dataset(split="train")
        
        x_org, x_cond, task_embed, actions = dataset[sample_idx]
        x_cond_standard = rearrange(x_cond, "h w c -> c h w")
        x_cond_dev = x_cond_standard.unsqueeze(0).to(self.device)
        task_embed_dev = th.tensor([task_embed], dtype=th.long, device=self.device)
        actions_dev = actions.unsqueeze(0).to(self.device)


        visualizer = AttentionVisualizer()
        print("Running model sampling to collect attention maps...")
        pred_traj = self.world_model.sample(
            x_cond=x_cond_dev,
            task_embed=task_embed_dev,
            action_embed=actions_dev,
            batch_size=1,
            vis=visualizer,
        )
        pred_traj = rearrange(pred_traj, "b (f c) h w -> b f h w c", c=26, f=32)
        pred_traj = pred_traj.cpu().numpy()[0]
        output_dir = self.base_dir / "attention_visualization"
        os.makedirs(output_dir, exist_ok=True)
        x_cond_for_render = x_cond_standard.permute(1, 2, 0).cpu().numpy()
        grid = self.renderer.extract_grid_from_obs(x_cond_for_render)

        self.renderer.save_obs_image(
            x_cond_for_render,
            grid=grid,
            file_path=output_dir / f"sample_{sample_idx}_x_cond.png",
            normalize=True,
        )
        self.renderer.render_trajectory_video(
            pred_traj,
            grid=grid,
            output_dir=str(output_dir / "sample_videos"),
            video_path=str(output_dir / f"sample_{sample_idx}_pred_traj.mp4"),
            fps=1,
            normalize=True,
        )
        background_surface = self.renderer.render_frame(x_cond_for_render, grid, normalize=False)
        background_img = pygame.surfarray.array3d(background_surface)
        background_img = np.swapaxes(background_img, 0, 1)
        action_plan_np = actions.cpu().numpy()
        num_action_tokens = action_plan_np.shape[0]
        context_split = (num_action_tokens, 8, 6)
        # visualizer.visualize_and_save(
        #     x_cond_img=background_img,
        #     action_plan=action_plan_np,
        #     output_dir=output_dir,
        #     step=f"sample_{sample_idx}", # Use a unique identifier for the filename
        #     T_context_split=context_split,
        # )
        # visualizer.visualize_and_save_blocks(
        #     x_cond_img=background_img,
        #     action_plan=action_plan_np,
        #     output_dir=output_dir,
        #     step=f"sample_{sample_idx}_blocks",
        # )
        # visualizer.visualize_and_save_temporal(
        #     action_plan=action_plan_np,
        #     output_dir=output_dir,
        #     step=f"sample_{sample_idx}_temporal",
        # )
        visualizer.visualize_self_attention_simple_and_save(
            output_dir=output_dir,
            step=f"sample_{sample_idx}_self_attention_simple",
        )
        visualizer.visualize_self_attention(
            grid=grid,
            renderer=self.renderer,
            pred_traj_np=pred_traj,
            output_dir=output_dir,
            step=f"sample_{sample_idx}_self_attention",
        )
        print(f"--- Attention visualization complete ---")
               
    def set_experiment_dir(self, experiment_name, subfolder_name=None):
        """
        Set the base_dir to the experiment-specific subfolder, only if not already set.
        """
        if subfolder_name is None:
            subfolder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = self.base_dir / experiment_name / subfolder_name
        # Prevent repeated nesting if already set
        if self.base_dir != experiment_dir:
            self.base_dir = experiment_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def run_sp10_ego(self, name=None):
        self.load_data("./model_tester/data/model_tester_sp10_bc_data_100.pkl")
        self.set_experiment_dir("sp10_bc", name)
        # self.run_idm_model_evaluation("sp10_bc_")
        # self.run_value_model_evaluation("sp10_bc_")
        # self.run_action_proposal_evaluation("sp10_bc_")
        # self.run_world_model_evaluation("world_model_eval_1")
        self.run_world_model_evaluation_with_dataset("world_model_eval_2")
        # self.run_obs_roundtrip_test("obs_roundtrip_test_1")
    
    def run_actor_best_bc(self, name=None):
        self.load_data("./model_tester/data/model_tester_actor_best_bc_train_100.pkl")
        self.set_experiment_dir("actor_best_bc", name)
        # self.run_idm_model_evaluation("actor_best_bc_")
        # self.run_value_model_evaluation("actor_best_bc_")
        # self.run_action_proposal_evaluation("actor_best_bc_")
        # self.run_obs_roundtrip_test("obs_roundtrip_test_1")
    
    def run_debug_tests(self):
        self.set_experiment_dir("debug_tests")
        self.visualize_attention(output_dir="attention_visualizations", sample_idx=42)
    
            

def parse_args(args, parser):
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--results_dir', type=str, default='./overcooked_results', help='Directory to save results and checkpoints')
    parser.add_argument('--basedir', type=str, default='./experiment_results', help='Directory to save results and checkpoints')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (smaller dataset, faster training)')
    parser.add_argument('--horizon', type=int, default=32, help='Sequence horizon for trajectories')
    parser.add_argument('--save_milestone', type=bool, default=True, help='Save milestones with step number in filename') # Or action='store_true'

    # For GoalGaussianDiffusion (configurable ones)
    parser.add_argument('--timesteps', type=int, default=400, help='Number of diffusion timesteps for training (if not debug)')
    parser.add_argument('--sampling_timesteps', type=int, default=10, help='Number of timesteps for DDIM sampling (if not debug)')
    
    # For OvercookedSequenceDataset / HDF5Dataset
    parser.add_argument('--max_path_length', type=int, default=401, help='Maximum path length in episodes (for dataset indexing)')
    parser.add_argument('--chunk_length', type=int, default=None, help='Chunk length for HDF5Dataset (defaults to horizon if None, set via dataset_constructor_args)')
    parser.add_argument('--use_padding', type=bool, default=True, help='Whether to use padding for shorter sequences in dataset')
    
    parser.add_argument("--diffusion_model_path", type=str, required=True, help="Path to the diffusion model directory")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--max_steps", type=int, default=400, help="Maximum steps per episode")
    parser.add_argument("--idm_path", type=str, required=True, help="Path to the diffusion model directory")
    parser.add_argument("--action_proposal_model_path", type=str, required=True, help="Path to the action proposal (diffusion) model checkpoint")
    parser.add_argument("--value_model_path", type=str, required=True, help="Path to the reward/value model checkpoint")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Overcooked HDF5 dataset')

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
    parser.add_argument("--run_dir", type=str, default="eval_run", help="Directory for evaluation run")
    all_args = parser.parse_known_args(args)[0]
    return all_args

if __name__ == "__main__":
    parser = get_config()
    args = sys.argv[1:]
    args = parse_args(args, parser)
    args.episode_length = 400
    tester = ModelTester(args=args)
    tester.run_debug_tests()
    # tester.run_sp10_ego()
    # tester.run_dataset_evaluation()
    # tester.run_actor_best_bc()
    # tester.collect_data(num_episodes=100)
    # tester.save_data("model_tester_actor_best_bc_train.pkl")
    # tester.load_data("./model_tester/data/model_tester_actor_best_bc_train_100.pkl")
    # tester.load_data("./model_tester/data/model_tester_sp10_bc_data_100.pkl")
    # tester.run_value_model_evaluation("actor_best_bc_train_")
    # tester.run_idm_model_evaluation("actor_best_bc_train_")
    # tester.run_dataset_evaluation()