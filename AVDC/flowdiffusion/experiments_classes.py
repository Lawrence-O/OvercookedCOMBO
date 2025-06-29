from pathlib import Path
import sys
mapbt_path = '/home/law/Workspace/repos/COMBO/mapbt_package/mapbt'
if mapbt_path not in sys.path:
    sys.path.append(mapbt_path)
overcooked_ai_py_src_path = '/home/law/Workspace/repos/COMBO/mapbt_package/mapbt/envs/overcooked/overcooked_berkeley/src/overcooked_ai_py'
if overcooked_ai_py_src_path not in sys.path:
    sys.path.append(overcooked_ai_py_src_path)

from einops.einops import rearrange
import numpy as np
import torch as th
import warnings
warnings.filterwarnings("ignore")
from goal_diffusion import GoalGaussianDiffusion, ConceptTrainer
from unet import UnetOvercooked 
from experiments_util import managed_environment, normalize_obs, to_torch, get_idm_action, convert_to_binary_obs
from overcooked_sample_renderer import OvercookedSampleRenderer

class ConceptLearnExperiment:
    """Base class for concept learning experiments in Overcooked."""
    def __init__(self, args, observation_dim, embedding, guidance_weight, num_concepts, train_dataset, device=None):
        self.args = args
        if device is None:
            self.device = th.device(f"cuda:{args.gpu_id}" if th.cuda.is_available() and args.gpu_id >= 0 else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        self.results_folder = Path(args.results_dir)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.horizon = args.horizon
        if train_dataset is None:
            raise ValueError("Must provide a dataset instance")
        
        if num_concepts is None:
            raise ValueError("Must provide number of concepts to learn")
        
        self.train_dataset = train_dataset
        self.valid_dataset = train_dataset  # For simplicity, using same dataset for validation
        self.observation_dim = observation_dim
        self.unet = None
        self.diffusion = None
        self.trainer = None
        self.new_policy_id = 0 # Assuming new policy ID starts at 0
        self.dummy_id = 1
        self.num_concepts = num_concepts
        self.unet_num_classes = num_concepts + 1
        print(f"Number of concepts to learn: {self.num_concepts}")
        self.embedding = embedding if embedding is not None else None
        self.guidance_weight = guidance_weight if guidance_weight is not None else 1.0
        print(f"Using guidance weight: {self.guidance_weight}")
    
    def setup_trainer(self):
        """Initialize the UNet and diffusion trainer."""
        H,W,C = self.observation_dim
        self.unet = UnetOvercooked(
            horizon=self.horizon,
            obs_dim=self.observation_dim,
            num_classes=self.unet_num_classes,
        ).to(self.device)
        
        self.diffusion = GoalGaussianDiffusion(
            model=self.unet,
            channels=C * 32, #TODO: Look into this Channels * Horizon
            image_size=(H,W),
            timesteps=1 if self.args.debug else 1000,
            sampling_timesteps=1 if self.args.debug else 100,
            loss_type="l2",
            objective="pred_v",
            beta_schedule="cosine",
            min_snr_loss_weight=True, 
            guidance_weight=1.0
        ).to(self.device)
        
        self.trainer = ConceptTrainer(
            diffusion_model=self.diffusion,
            channels=C,
            train_set=self.train_dataset,
            valid_set=self.valid_dataset,
            train_lr=1e-4,
            train_num_steps = self.args.max_train_steps if not self.args.debug else 1,
            save_and_sample_every = 2 if self.args.debug else 1000,
            ema_update_every = 10,
            ema_decay = 0.999,
            train_batch_size = 1 if self.args.debug else self.args.train_batch_size,
            valid_batch_size = 1 if self.args.debug else self.args.train_batch_size,
            gradient_accumulate_every = 1,
            num_samples=self.args.num_validation_samples, 
            results_folder = str(self.results_folder),
            fp16 =True,
            amp=True,
            save_milestone=self.args.save_milestone,
            cond_drop_chance=getattr(self.args, 'cond_drop_prob', 0),
            split_batches=getattr(self.args, 'split_batches', True),
            debug=self.args.debug,
            dummy_policy_id=self.dummy_id,
            new_policy_id=self.new_policy_id,
            embedding=self.embedding,
            guidance_w=self.guidance_weight,
        )
    def train(self):
        """Run the training process."""
        if self.trainer is None:
            self.setup_trainer()
        print(f"Starting training for {self.args.max_train_steps} steps...")
        self.trainer.load_concept_checkpoint(self.args.pretrained_model_path)
        self.trainer.train()
        print("Training completed.")
        self.trainer.save(milestone=self.args.milestone_name)
        print(f"Model saved to {self.results_folder / self.args.milestone_name}")
        embed, _ = self.trainer.get_embedding()
        metrics = self.trainer.get_metrics()
        guidance_weight = self.trainer.get_guidance_weight()
        return (embed, guidance_weight, metrics)

class EvaluationExperiment:
    """Base class for evaluation experiments in Overcooked."""
    def __init__(self, args, world_model, idm, action_proposal_model, value_model, policy_name, policy_id, num_episodes, results_dir, n_envs, max_steps, planning_horizon=32, action_horizon=8, device=None):
        self.args = args
        self.world_model = world_model
        self.idm = idm
        self.action_proposal_model = action_proposal_model
        self.value_model = value_model
        if hasattr(self.world_model, 'ema_model'):
            self.world_model = self.world_model.ema_model
        if hasattr(self.action_proposal_model, 'ema_model'):
            self.action_proposal_model = self.action_proposal_model.ema_model
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.policy_name = policy_name
        self.policy_id = policy_id
        self.planning_horizon = planning_horizon
        self.action_horizon = action_horizon
        self.num_episodes = num_episodes
        self.renderer = OvercookedSampleRenderer()
        self.results_dir = Path(results_dir)
        self.num_candidates = args.num_candidates if hasattr(args, 'num_candidates') else 10
        self.horizon = args.horizon
        self.num_action_classes = 6
        self.ranked_candidates_dicts = []
        self.H, self.W, self.C = (8, 5, 26)
        if device is None:
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
    def run(self, save_videos=False):
        """Run the evaluation process."""
        episode_metrics = []
        episode_rewards_list = []
        for episode in range(self.num_episodes):
            print(f"Starting episode {episode + 1}/{self.num_episodes}...")
            results = self._evaluate()
            frames = results["frames"]
            episode_reward = results["episode_reward"]
            steps = results["steps"]
            episode_metrics.append(results)
            episode_rewards_list.append(episode_reward)
            # Save videos for this episode
            video_dir = self.results_dir / f"episode_{episode + 1}"
            video_dir.mkdir(parents=True, exist_ok=True)
            if save_videos:
                self._save_episode_videos(frames, episode, video_dir)
            print(f"Episode {episode + 1} completed. Total steps: {steps}")
        return self.calculate_evaluation_summary(
            episode_rewards_list, 
            episode_metrics, 
            self.results_dir, 
            self.args.layout_name, 
            self.policy_id
        )
        
    def calculate_evaluation_summary(self, episode_rewards_list, episode_metrics, current_run_basedir, layout_name, policy_id):
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
            'agent_id': policy_id,
            'total_episodes_evaluated': self.num_episodes * self.n_envs,
            'metrics_per_episode_reset': episode_metrics,
            'raw_episode_rewards': [arr.tolist() for arr in episode_rewards_list], 
        }
    @th.no_grad()
    def _generate_trajectory_candidates(self, current_obs, policy_id):
        """Generate trajectory candidates using the world model and action proposal model."""
        # Generate initial action proposals
        proposals = []
        current_obs = to_torch(current_obs, device=self.device, dtype=th.float32)
        current_obs = current_obs.view(self.n_envs, self.C, self.H, self.W)
        for _ in range(self.num_candidates):
            # Sample a random action proposal
            action_proposal = self.action_proposal_model.sample(
                x_cond=current_obs,
                batch_size=self.n_envs,
            )
            # Reshape to [B, Horizon, Num_Action_Classes]
            action_proposal = action_proposal.view(self.n_envs, self.horizon, self.num_action_classes)
            action_condition = action_proposal[:, :self.action_horizon, :] 
            action_condition = th.argmax(action_condition, dim=-1)
            print(f"Sampled actions:", action_condition)
            # Sample a trajectory using the world model
            trajectory = self.world_model.sample(
                x_cond=current_obs,
                action_embed=action_condition,
                batch_size=self.n_envs,
                task_embed=policy_id)
            trajectory = rearrange(trajectory, "b (f c) h w -> b f h w c", c=self.C, f=self.horizon)
            proposals.append((action_proposal, trajectory))
        print(f"Generated {len(proposals)} trajectory candidates.")
        return proposals
    
    # def _rank_trajectories(self, trajectories):
    #     """Rank the generated candidates based on their expected rewards."""
    #     ranked_candidates = []
    #     for trajectory in trajectories: # [B, Horizon, H, W, C]
    #         trajectory = normalize_obs(trajectory)
    #         total_value = th.zeros((self.n_envs,), device=self.device)
    #         for i in range(trajectory.shape[0]-1):
    #             obs_t = trajectory[:, i, ...]  # [B, H, W, C]
    #             obs_tp1 = trajectory[:, i + 1, ...]
    #             value = self.value_model(obs_t, obs_tp1)
    #             total_value += value
    #         # Store the trajectory and its total value
    #         print(f"Total value for trajectory: {total_value}")
    #         print(total_value.shape)
    #         ranked_candidates.append((total_value, trajectory))
    #     # Sort candidates by their expected rewards (total value)
    #     for i in range(self.n_envs):
    #         ranked_candidates.sort(key=lambda x: x[0][i].item(), reverse=True)
    #     print(f"Ranked {len(ranked_candidates)} candidates based on expected rewards.")
    #     return ranked_candidates
    @th.no_grad()
    def _rank_trajectories(self, trajectories):
        """Rank the generated candidates based on their expected rewards."""
        N = len(trajectories)
        B, T, H, W, C = trajectories[0].shape
        traj_tensor = th.stack(trajectories, dim=0)  # [N, B, T, H, W, C]
        traj_tensor = traj_tensor.permute(1, 0, 2, 3, 4, 5)  # [B, N, T, H, W, C]
        all_trajs = traj_tensor.reshape(B*N, T, H, W, C)  # [B*N, T, H, W, C]
        print(f"All Trajectories Before Norm Min and Max: {all_trajs.min()}, {all_trajs.max()}")
        all_trajs = convert_to_binary_obs(all_trajs)  # Normalize the trajectories
        print(f"All Trajectories Post Norm Min and Max: {all_trajs.min()}, {all_trajs.max()}")
        print(f"All trajectories shape: {all_trajs.shape}")
        total_vals = th.zeros((B * N,), device=self.device)
        for t in range(T-1):
            obs_t = all_trajs[:, t, ...]
            obs_tp1 = all_trajs[:, t + 1, ...]
            vals = self.value_model(obs_t, obs_tp1)  # [B*N]
            total_vals += vals
        # Reshape total_vals to [B, N]
        total_vals = total_vals.view(B, N)  # [B, N]
        ranked_candidates = []
        print(f"Total values shape: {total_vals.shape}")
        print(f"Total values: {total_vals}")
        # Now we need to rank the candidates for each environment
        for i in range(B):
            # Sort candidates by their expected rewards (total value)
            sorted_indices = th.argsort(total_vals[i], descending=True)
            env_list = [
                (total_vals[i, idx].item(), trajectories[idx][i])  # now [T,H,W,C]
                for idx in sorted_indices
            ]
            ranked_candidates.append(env_list)
        # Sort candidates by their expected rewards (total value)
        ranked_candidates = sorted(ranked_candidates, key=lambda x: x[0], reverse=True)
        print(f"Ranked {len(ranked_candidates)} candidates based on expected rewards.")
        # Print the top 5 candidates for each environment
        for i in range(B):
            print(f"Top candidates for environment {i}:")
            for j in range(min(5, len(ranked_candidates[i]))):
                print(f"  Candidate {j+1}: Reward = {ranked_candidates[i][j][0]:.4f}, Trajectory shape = {ranked_candidates[i][j][1].shape}")
        return ranked_candidates
    
    def _rank_and_select_candidates(self, current_obs, policy_id):
        """Generate and rank trajectory candidates."""
        # Generate trajectory candidates
        proposals = self._generate_trajectory_candidates(current_obs, policy_id)
        # Rank the candidates based on their expected rewards
        action_proposals, trajectories = zip(*proposals)
        ranked_candidates = self._rank_trajectories(trajectories)
        # For each environment, select the top candidate
        reward_values = [env_list[0][0] for env_list in ranked_candidates]
        ranked_trajectories = [env_list[0][1] for env_list in ranked_candidates]
        ranked_trajectories = th.stack(ranked_trajectories, dim=0)  # [B, T, H, W, C]
        return {
            "action_candidates": action_proposals,
            "trajectory_candidates": trajectories,
            "ranked_candidates": ranked_candidates,
            "reward_values": reward_values,
            "ranked_trajectories": ranked_trajectories,
        }
    @th.no_grad()
    def _get_next_actions(self, current_obs, policy_id):
        """Get the next actions for the ego agent based on the ranked candidates."""
        ranked_candidates_dict = self._rank_and_select_candidates(current_obs, policy_id)
        # self.ranked_candidates_dicts.append(ranked_candidates_dict)
        best_trajectory = ranked_candidates_dict["ranked_trajectories"]
        B, T, H, W, C = best_trajectory.shape
        # Get Initial Actions
        current_obs = to_torch(current_obs, device=self.device, dtype=th.float32)
        init_states = best_trajectory[:, 0, ...]
        init_actions = get_idm_action(
            current_obs,
            init_states,
            self.idm
        )
        # Get all obs_t and obs_tp1 pairs
        obs_t = best_trajectory[:, :-1, ...]  # [B, T-1, H, W, C]
        obs_tp1 = best_trajectory[:, 1:, ...]  # [B, T-1, H, W, C]
        # Flatten batch and time dims for input:
        obs_t_flat = obs_t.reshape(B * (T - 1), H, W, C)
        obs_tp1_flat = obs_tp1.reshape(B * (T - 1), H, W, C)
        traj_actions = get_idm_action(
            obs_t_flat, 
            obs_tp1_flat,
            self.idm
        )
        ego_actions = th.cat([init_actions, traj_actions])
        ego_actions = ego_actions.reshape(B, T)
        print(f"Generated next actions shape: {ego_actions.shape}")
        print(f"Generated next actions: {ego_actions}")
        return ego_actions
    
    def _get_candidate_metrics(self, ranked_candidates_dict):
        pass
            

    def _evaluate(self):
        """Run the evaluation process with the given environments and policy."""
        with managed_environment(self.args, self.policy_name, self.n_envs) as (envs, policy):
            print(f"Running evaluation with {self.n_envs} environments...")
            cond = th.full((self.n_envs,), self.policy_id, dtype=th.int64, device=self.device)
            policy.reset(num_envs=self.args.n_envs, num_agents=1)
            for e in range(self.args.n_envs):
                policy.register_control_agent(e=e, a=1)
            obs, _, _ = envs.reset([True] * self.n_envs)
            steps = 0
            done = False
            episode_reward = np.zeros((self.n_envs, 2))
            frames = [[obs[i][0]] for i in range(self.n_envs)]
            step_actions = np.zeros((self.args.n_envs, 2, 1), dtype=np.int64)
            *_, C = obs[0][0].shape
            
            while not done and steps <= self.args.max_steps:
                # Setup Condition Obs Based on Obs
                obs_stack = np.stack([normalize_obs(obs[e][0]) for e in range(self.n_envs)], axis=0) 
                condition_obs = th.tensor(obs_stack, device=self.device, dtype=th.float32)
                assert condition_obs.shape[-1] == 26 # Double Check
                assert condition_obs.min() >= -1 and condition_obs.max() <= 1, \
                    f'condition_obs must be normalized to [-1, 1], got range [{condition_obs.min():.3f}, {condition_obs.max():.3f}]'
                assert cond.min() >= 0 and cond.max() < self.args.num_classes if hasattr(self.args, 'num_classes') else 10, \
                    f'cond (task_embed) contains invalid policy IDs: min={cond.min()}, max={cond.max()}'
                ego_actions = self._get_next_actions(condition_obs, cond)
                ego_actions = ego_actions.cpu().numpy()
                # Now step through the environment using the plan
                plan_horizon = min(self.max_steps - steps, self.planning_horizon, 8)
                # We begin with the first ego obs (first obs of the environment)
                for t in range(plan_horizon): 
                    step_actions[:, 0, 0] = ego_actions[:, t]
                    partner_obs_lst = [obs[e][1] for e in range(self.args.n_envs)]
                    partner_obs = np.stack(partner_obs_lst, axis=0)
                    partner_action = policy.step(
                        partner_obs,
                        [(e, 1) for e in range(self.args.n_envs)],
                        deterministic=False,
                    )
                    step_actions[:, 1] = partner_action
                    # Take environment step
                    obs, _, reward, done, _, _ = envs.step(step_actions)
                    episode_reward += reward.squeeze(axis=2)
                    if self.args.save_videos:
                        for e in range(min(self.args.n_envs, 3)):
                            frames[e].append(obs[e][0])
                    # Check for early termination
                    done = np.all(done)
                    steps += 1
                    if steps >= self.args.max_steps:
                        break
            return {
                "frames": frames,
                "episode_reward": episode_reward,
                "steps": steps,
            }

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
                
        print(f"Videos saved to {video_dir}")
    
    