from collections import deque
import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
import numpy as np
import torch 
import warnings
warnings.filterwarnings("ignore")
from overcooked.utils.utils import managed_environment
from overcooked.utils.overcooked_visualizer import OvercookedVisualizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn

class BaseTester:
    """
    A base class for model testers, containing shared functionalities like
    argument handling, seeding, and directory management.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderer = OvercookedVisualizer()
        self.n_envs = args.n_envs
        self.max_steps = args.max_steps
        self.horizon = args.horizon
        self.H, self.W, self.C = 8, 5, 26  # Overcooked obs shape
        self.num_actions = 6
        
        # Base directory for all outputs of this tester instance
        self.base_dir = Path("evaluation_results")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.set_seed()

    def set_seed(self, seed=42):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
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

    def evaluate_in_env(self, state_action_fn, observe_fn=None, num_episodes=10, policy_name="bc_train"):
        """Test the value model by running episodes and collecting rewards.
        Supports both single-step and planning horizon action outputs.
        """
        all_data = []
        with managed_environment(self.args, policy_name, self.n_envs ) as (envs, partner_agent_policy):
            for episode in tqdm(range(num_episodes), desc="Running Evaluation"):
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

                if observe_fn is not None:
                    initial_obs_batch = np.stack([obs[e][0] for e in range(self.n_envs)], axis=0)
                    observe_fn(initial_obs_batch)

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
                            
                            if steps % 4 == 0:
                                if observe_fn is not None:
                                    new_agent0_obs_batch = np.stack([obs[e][0] for e in range(self.n_envs)], axis=0)
                                    observe_fn(new_agent0_obs_batch)
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
                        if observe_fn is not None:
                            new_agent0_obs_batch = np.stack([obs[e][0] for e in range(self.n_envs)], axis=0)
                            observe_fn(new_agent0_obs_batch)
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
            self.renderer.render_trajectory_video(
                frames[e], 
                grid, 
                output_dir=str(env_dir),
                video_path=str(env_dir / "actual_trajectory.mp4"),
                fps=1
            )


class MultiPolicyTester(BaseTester):
    """
    A class for testing multiple policies in Overcooked environments.
    Inherits from BaseTester to utilize shared functionalities.
    """
    def __init__(self, args, agent, policies):
        super().__init__(args)
        self.args = args
        self.agent = agent 
        self.policies = policies 
        self.all_evaluation_summaries = []
        self.policy_name_to_id = {"best_r_vs_bc_train_held_out": 1, "comedi_mp0": 2, "comedi_mp1": 3, "comedi_mp2": 4, "comedi_mp3": 5, "mep1_final": 6, "mep1_init": 7, "mep1_mid": 8, "mep2_final": 9, "mep2_init": 10, "mep2_mid": 11, "mep3_final": 12, "mep3_init": 13, "mep3_mid": 14, "sp10_final": 15, "sp10_init": 16, "sp10_mid": 17, "sp1_final": 18, "sp1_init": 19, "sp1_mid": 20, "sp2_final": 21, "sp2_init": 22, "sp2_mid": 23, "sp3_final": 24, "sp3_init": 25, "sp3_mid": 26, "sp9_final": 27, "sp9_init": 28, "sp9_mid": 29}

    def evaluate_agent_against_partners(self, num_episodes=10):
        """
        Evaluates the self.agent against each partner policy in self.policies.
        
        Args:
            num_episodes (int): Number of episodes to run for each partner policy.
        """
        print(f"\n--- Starting Multi-Policy Evaluation for Agent: {type(self.agent).__name__} ---")
        
        # Set up a specific directory for multi-policy evaluation results
        self.set_experiment_dir(experiment_name="multi_policy_evaluation")

        for policy_name in self.policies:
            print(f"\nEvaluating against partner policy: {policy_name}")
            
            # Reset the agent's internal state (e.g., history buffers for DiffusionPlannerAgent)
            # This is crucial for history-dependent agents to start fresh for each new partner evaluation.
            if hasattr(self.agent, 'reset'):
                self.agent.reset()
            
            # Define the state_action_fn for Agent 0 (self.agent)
            # This function will be called by evaluate_in_env.
            # It needs to return actions for Agent 0.
            # The 'current_partner_policy_name_from_env' argument here is the 'policy_name' argument
            # that evaluate_in_env passes, which corresponds to the current partner policy being used.
            def agent0_planning_action_fn(obs_batch_np, current_partner_policy_name_from_env):
                # Get the policy ID for the current partner policy from args
                # This ID is used by DiffusionPlannerAgent as 'task_embed'
                if current_partner_policy_name_from_env not in self.policy_name_to_id:
                    raise ValueError(f"Policy name '{current_partner_policy_name_from_env}' not found in args.policy_name_to_id mapping.")
                
                partner_policy_id = self.policy_name_to_id[current_partner_policy_name_from_env]
                policy_id_batch_np = np.full((obs_batch_np.shape[0],), partner_policy_id, dtype=np.int64)
                
                # Get the action plan from self.agent
                # DiffusionPlannerAgent.get_plan is expected to return a numpy array of shape [B, horizon]
                agent0_actions_plan = self.agent.get_plan(obs_batch_np, policy_id_batch_np)
                return agent0_actions_plan

            # Run evaluation using the BaseTester's evaluate_in_env method.
            # The 'policy_name' argument passed here ensures that the 'managed_environment' context manager
            # sets up the correct partner policy for Agent 1.
            all_episode_data = self.evaluate_in_env(
                state_action_fn=agent0_planning_action_fn,
                observe_fn=self.agent.observe,
                num_episodes=num_episodes,
                policy_name=policy_name 
            )
            
            # Extract episode rewards for summary calculation
            episode_rewards_list = [d["episode_reward"] for d in all_episode_data]
            
            # Calculate summary statistics for the current partner policy
            summary = self.calculate_evaluation_summary(episode_rewards_list)
            
            # Add partner policy name to the summary dictionary for identification in plotting/reporting
            summary['partner_policy_name'] = policy_name
            self.all_evaluation_summaries.append(summary)

        print("\n--- Multi-Policy Evaluation Complete ---")
        
        # After evaluating against all partners, generate the consolidated plot
        self.plot_multi_policy_results()
        
        # Save all collected summaries to a single JSON/CSV file for comprehensive analysis
        output_path_csv = self.base_dir / "multi_policy_evaluation_summary.csv"
        output_path_json = self.base_dir / "multi_policy_evaluation_summary.json"
        pd.DataFrame(self.all_evaluation_summaries).to_csv(output_path_csv, index=False)
        pd.DataFrame(self.all_evaluation_summaries).to_json(output_path_json, orient='records', indent=2)
        print(f"Full evaluation summary saved to {output_path_csv} and {output_path_json}")

    def plot_multi_policy_results(self):
        """
        Generates a grouped bar plot comparing Agent 0's (self.agent) performance
        across different partner policies, including Agent 1's performance and Team Total.
        """
        plot_dir = self.base_dir / "plots" / "multi_policy_evaluation"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.all_evaluation_summaries)
        if df.empty:
            print("No multi-policy results to plot.")
            return

        partner_policies = df['partner_policy_name'].values
        x = np.arange(len(partner_policies))  # the label locations
        bar_width = 0.2

        # Metrics for plotting
        agent0_means = df['agent0_mean_reward'].values
        agent0_stds = df['agent0_std_reward'].values
        agent1_means = df['agent1_mean_reward'].values
        agent1_stds = df['agent1_std_reward'].values
        team_means = df['team_mean_reward'].values
        team_stds = df['team_std_reward'].values

        # Bar positions for grouping
        r1 = x - bar_width
        r2 = x
        r3 = x + bar_width

        plt.figure(figsize=(12, 7))
        plt.bar(r1, agent0_means, yerr=agent0_stds, width=bar_width, label=f'Agent 0 ({type(self.agent).__name__})', capsize=5, color='skyblue')
        plt.bar(r2, agent1_means, yerr=agent1_stds, width=bar_width, label='Agent 1 (Partner)', capsize=5, color='lightgreen')
        plt.bar(r3, team_means, yerr=team_stds, width=bar_width, label='Team Total', capsize=5, color='coral')

        plt.xlabel('Partner Policy')
        plt.ylabel('Mean Reward')
        plt.title(f'Agent 0 ({type(self.agent).__name__}) Performance Across Different Partner Policies')
        plt.xticks(x, partner_policies, rotation=45, ha='right') # Set partner names as x-axis labels
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        output_path = plot_dir / "multi_policy_performance.png"
        plt.savefig(output_path)
        print(f"Multi-policy performance plot saved to {output_path}")
        plt.close()

    
class ConceptLearningTester(BaseTester):
    """
    A class for testing concept learning agents in Overcooked environments.
    Inherits from BaseTester to utilize shared functionalities.
    """
    def __init__(self, args, agent, num_concepts):
        super().__init__(args)
        self.agent = agent
        self.all_evaluation_summaries = []
        self.buffer = deque(maxlen=self.args.buffer_size) # Initialize a replay buffer

    def evaluate_agent(self, num_episodes=10):
        """
        Evaluates the concept learning agent against a fixed partner policy.
        
        Args:
            num_episodes (int): Number of episodes to run.
        """
        print(f"\n--- Starting Concept Learning Evaluation for Agent: {type(self.agent).__name__} ---")
        
        # Set up a specific directory for concept learning evaluation results
        self.set_experiment_dir(experiment_name="concept_learning_evaluation")

        # Define the state_action_fn for Agent 0 (self.agent)
        def agent0_planning_action_fn(obs_batch_np, current_partner_policy_name_from_env):
            # Get the action plan from self.agent
            agent0_actions_plan = self.agent.get_plan(obs_batch_np)
            return agent0_actions_plan

        # Run evaluation using the BaseTester's evaluate_in_env method.
        all_episode_data = self.evaluate_in_env(
            state_action_fn=agent0_planning_action_fn,
            num_episodes=num_episodes,
            policy_name="bc_train"  # Fixed partner policy
        )
        
        # Extract episode rewards for summary calculation
        episode_rewards_list = [d["episode_reward"] for d in all_episode_data]
        
        # Calculate summary statistics
        summary = self.calculate_evaluation_summary(episode_rewards_list)
        
        # Add partner policy name to the summary dictionary for identification in plotting/reporting
        summary['partner_policy_name'] = "bc_train"
        self.all_evaluation_summaries.append(summary)

        print("\n--- Concept Learning Evaluation Complete ---")
        
        # Generate the consolidated plot
        self.plot_concept_learning_results()
        
        # Save all collected summaries to a single JSON/CSV file for comprehensive analysis
        output_path_csv = self.base_dir / "concept_learning_evaluation_summary.csv"
        output_path_json = self.base_dir / "concept_learning_evaluation_summary.json"
        pd.DataFrame(self.all_evaluation_summaries).to_csv(output_path_csv, index=False)
        pd.DataFrame(self.all_evaluation_summaries).to_json