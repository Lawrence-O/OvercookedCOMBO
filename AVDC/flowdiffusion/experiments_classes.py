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
from experiments_util import managed_environment, normalize_obs, to_torch, get_idm_action
from overcooked_sample_renderer import OvercookedSampleRenderer

class ConceptLearnExperiment:
    """Base class for concept learning experiments in Overcooked."""
    def __init__(self, args, observation_dim, embedding, num_concepts, train_dataset, device=None):
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
            guidance_weight= self.args.guidance_weight if self.args.guidance_weight else 1.0
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
        return embed, metrics

class EvaluationExperiment:
    """Base class for evaluation experiments in Overcooked."""
    def __init__(self, args, diffusion, idm, policy_name, policy_id, num_episodes, results_dir, n_envs, max_steps, planning_horizon=32, device=None):
        self.args = args
        self.diffusion = diffusion
        if hasattr(self.diffusion, 'ema_model'):
            self.diffusion = self.diffusion.ema_model
        self.idm = idm
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.policy_name = policy_name
        self.policy_id = policy_id
        self.planning_horizon = planning_horizon
        self.num_episodes = num_episodes
        self.renderer = OvercookedSampleRenderer()
        self.results_dir = Path(results_dir)
        self.horizon = args.horizon
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
            samples_frames = results["samples_frames"]
            episode_reward = results["episode_reward"]
            steps = results["steps"]
            episode_metrics.append(results)
            episode_rewards_list.append(episode_reward)
            # Save videos for this episode
            video_dir = self.results_dir / f"episode_{episode + 1}"
            video_dir.mkdir(parents=True, exist_ok=True)
            grid = self.renderer.extract_grid_from_obs(frames[0][0])
            if save_videos:
                self._save_episode_videos(frames, samples_frames, grid, episode, video_dir, show_samples=False)
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
    def _evaluate(self):
        """Run the evaluation process with the given environments and policy."""
        with managed_environment(self.args, self.policy_name, self.n_envs) as (envs, policy):
            print(f"Running evaluation with {self.n_envs} environments...")
            cond = th.full((self.n_envs,), self.policy_id, dtype=th.int64, device=self.device)
            policy.reset(num_envs=self.args.n_envs, num_agents=2)
            for e in range(self.args.n_envs):
                policy.register_control_agent(e=e, a=1)
            obs, _, _ = envs.reset([True] * self.n_envs)
            steps = 0
            done = False
            episode_reward = np.zeros((self.n_envs, 2))
            frames = [[obs[i][0]] for i in range(self.n_envs)]
            samples_frames = [[] for _ in range(self.n_envs)]
            step_actions = np.zeros((self.args.n_envs, 2, 1), dtype=np.int64)
            *_, C = obs[0][0].shape
            
            while not done and steps <= self.args.max_steps:
                # Setup Condition Obs Based on Obs
                obs_stack = np.stack([normalize_obs(obs[e][0]) for e in range(self.n_envs)], axis=0) 
                condition_obs = th.tensor(obs_stack, device=self.device, dtype=th.float32)
                assert condition_obs.shape[-1] == 26 # Double Check
                condition_obs = rearrange(condition_obs, "b h w c -> b c h w")
                samples = self.diffusion.sample(
                    x_cond=condition_obs,
                    task_embed=cond,
                    batch_size=self.n_envs,
                )
                samples = rearrange(samples, "b (f c) h w -> b f h w c", c=C, f=self.horizon)
                # Now step through the environment using the plan
                plan_horizon = min(self.max_steps - steps, self.planning_horizon)
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
                            self.idm
                        )
                        step_actions[env_i, 0, 0] = ego_action
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
                    obs_t = obs_tp1
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
                "samples_frames": samples_frames,
                "episode_reward": episode_reward,
                "steps": steps,
            }

    def _save_episode_videos(self, frames, samples_frames, grid, episode, video_dir, show_samples=False):
        """Save videos for a completed episode."""
        print(f"Saving videos for episode {episode + 1}...")
        for e in range(len(frames)):
            frames[e] = rearrange(frames[e], "f w h c -> f h w c")
            env_dir = video_dir / f"episode_{episode + 1}_env_{e + 1}"
            env_dir.mkdir(parents=True, exist_ok=True)
            
            # Save actual trajectory
            saved_video = self.renderer.render_trajectory_video(
                frames[e], 
                grid, 
                output_dir=str(env_dir),
                video_path=str(env_dir / "actual_trajectory.mp4"),
                fps=1
            )
            
            # Save samples trajectory if needed
            if show_samples:
                self.renderer.render_trajectory_video(
                    samples_frames[e],
                    grid,
                    output_dir=str(env_dir),
                    video_path=str(env_dir / "samples_trajectory.mp4"),
                    fps=1,
                    normalize=True,
                )
                
        print(f"Videos saved to {video_dir}")
        
    def _plan(self, trajectories):

        pass
    