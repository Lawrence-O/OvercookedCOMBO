from collections import deque
import datetime
from pathlib import Path
import numpy as np
import torch
from einops import rearrange, repeat
import numpy as np
import torch
import warnings

from tqdm import tqdm
warnings.filterwarnings("ignore")
from torch import nn
from overcooked.utils.utils import normalize_obs_vectorized
from overcooked.agent.idm.ground_truth_idm import GroundTruthInverseDynamics
from overcooked.agent.reward.state_reward_model import RewardCalculator as GroundTruthRewardCalculator
import multiprocessing
from overcooked.utils.overcooked_visualizer import OvercookedVisualizer
from overcooked.agent.action_proposal import ActionProposalUtil
from torch.optim import Adam
import torch.utils.checkpoint as cp
from torch.utils.data import Dataset
import torch.nn.functional as F

def _reward_calculation_worker(work_item):
    """Worker function for multiprocessing. Initializes its own RewardCalculator."""
    args, trajectory_np = work_item
    local_reward_calculator = GroundTruthRewardCalculator(args)
    obs_t_batch = trajectory_np[:-1]
    obs_tp1_batch = trajectory_np[1:]
    ego_rewards, partner_rewards = local_reward_calculator.calculate_reward_batch(obs_t_batch, obs_tp1_batch)
    return np.sum(ego_rewards) + np.sum(partner_rewards)



class ConceptLearningBuffer(Dataset):
    """
    A simple dataset to hold concept learning experiences.
    Each experience is a tuple of (initial_obs, trajectory, actions).
    """
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add_experience(self, initial_obs, trajectory, actions):
        self.buffer.append((initial_obs, trajectory, actions))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


class DiffusionPlannerAgent:
    """ A diffusion-based planning agent for Overcooked.
    This agent uses a diffusion model to generate action plans based on the current state of the game.
    It integrates with the world model, action proposal model, and inverse dynamics model to simulate and evaluate plans.
    
    Args:
        args (Namespace): Configuration parameters for the agent.
        world_model (nn.Module): The world model that simulates the game environment.
        action_proposal_model (nn.Module): The model that proposes action plans based on the current state.
        num_envs (int): Number of environments to run in parallel.
        horizon (int): The planning horizon for state sequences.
        target_reward (float): The target reward for the agent to achieve.
        history_horizon (int): The number of past observations to consider in planning.
        planning_horizon (int): The number of steps to plan ahead.
        action_horizon (int): The number of actions in each plan.
        num_action_candidates (int): Number of candidate action plans to generate.
        num_simulations_per_plan (int): Number of simulations to run for each candidate plan.
        num_concepts (int): Number of concepts to learn.
        model_embedding_dim (int): Dimensionality of the model's embedding space.
        use_checkpoint (bool): Whether to use checkpointing for memory efficiency.

    """
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
    def __init__(self, args, world_model, action_proposal_model, num_envs, horizon=32, target_reward=150,
                 history_horizon=16, planning_horizon=32, action_horizon=8, num_action_candidates=5, num_simulations_per_plan=10,
                 automatic_replan=True, max_replan_attempts=3,
                 num_concepts=10, model_embedding_dim=256*4, use_checkpoint=True, guidance_option="scalar", guidance_weight=1.0,
                 cl_buffer_size=256, num_processes=None, device=None, training_steps=10, batch_size=16):
        
        self.args = args
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent models operating on device: {self.device}")

        # Models 
        self.world_model = world_model.to(self.device).eval()
        self.action_proposer = ActionProposalUtil(action_proposal_model, history_horizon=history_horizon, device=self.device)
        self.idm = GroundTruthInverseDynamics(self.args) 
        self.reward_calculator = GroundTruthRewardCalculator(self.args)

        # Configuration
        self.horizon = horizon
        self.num_envs = num_envs
        self.planning_horizon = planning_horizon
        self.action_horizon = action_horizon
        self.num_action_candidates = num_action_candidates
        self.num_simulations_per_plan = num_simulations_per_plan
        self.target_reward = target_reward
        self.num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count() // 2
        self.automatic_replan = automatic_replan
        self.max_replan_attempts = max_replan_attempts

        # State Management
        self.history_buffers = [deque(maxlen=history_horizon) for _ in range(num_envs)]

        # Debugging
        self.renderer = OvercookedVisualizer()
        self.grid = None
        self.debug = True

        if self.debug:
            self.debug_dir = Path("DiffusionPlannerAgent_debug") / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"Debugging enabled. Output will be saved to {self.debug_dir}")

        # --- Concept Learning Components ---
        self.optimizer = None
        self.num_concepts = num_concepts
        self.model_embedding_dim = model_embedding_dim
        self.use_checkpoint = use_checkpoint
        self.dummy_id = 0
        self.guidance_option = guidance_option
        if guidance_option == "scalar":
            self.guidance_weight = torch.full((self.num_concepts,), guidance_weight, device=self.device)
        elif guidance_option == "learnable":
            self.guidance_logits = nn.Parameter(torch.full((self.num_concepts,), guidance_weight, device=self.device))
        self.initial_cl_buffer = deque(maxlen=horizon + 1)
        self.concept_learning_buffer = ConceptLearningBuffer(max_size=cl_buffer_size)
        self.training_steps = training_steps
        self.batch_size = batch_size  
        
        # Internal dimensions
        self.H, self.W, self.C = (8, 5, 26)

    def reset(self, env_indices=None):
        """Resets the history buffers for specified (or all) environments."""
        if env_indices is None:
            env_indices = range(self.num_envs)
        for i in env_indices:
            self.history_buffers[i].clear()
        print(f"Reset history for environments: {env_indices}")
    
    def add_concept_learning_experience(self, obs : np.ndarray, action: np.ndarray):
        """Adds a single trajectory experience to the concept learning buffer."""
        self.initial_cl_buffer.append((obs, action))
        if len(self.initial_cl_buffer) == self.horizon + 1:
            initial_obs = self.initial_cl_buffer.popleft()[0]
            obs_trajectory = np.array([x[0] for x in self.initial_cl_buffer])
            actions = np.array([x[1] for x in self.initial_cl_buffer])
            self.concept_learning_buffer.add_experience(initial_obs, obs_trajectory, actions)
    
    def sample_concept_learning_batch(self, batch_size=32):
        """
        Samples a batch of concept learning experiences.
        Returns a tuple of (initial_obs, obs_trajectory, actions).
        """
        if len(self.concept_learning_buffer) < batch_size:
            raise ValueError("Not enough samples in the concept learning buffer.")
        
        indices = np.random.choice(len(self.concept_learning_buffer), size=batch_size, replace=False)
        batch = [self.concept_learning_buffer[i] for i in indices]
        
        initial_obs = np.array([x[0] for x in batch])
        obs_trajectory = np.array([x[1] for x in batch])
        actions = np.array([x[2] for x in batch])
        
        return initial_obs, obs_trajectory, actions
    
    def observe(self, obs_batch_np: np.ndarray):
        """Updates the agent's history buffers with the latest observations."""
        assert obs_batch_np.shape == (self.num_envs, self.H, self.W, self.C), \
            f"Expected obs shape {(self.num_envs, self.H, self.W, self.C)}, got {obs_batch_np.shape}"
        
        for i, obs in enumerate(obs_batch_np):
            self.history_buffers[i].append(obs)
        
        if self.debug:
            print(f"History buffers updated for batch size {self.num_envs}. Current history lengths: {[len(h) for h in self.history_buffers]}")
    
    @torch.no_grad()
    def get_plan(self, obs_batch_np: np.ndarray, policy_id_batch_np: np.ndarray) -> np.ndarray:
        """
        The main public method. Simulates futures to find the best possible trajectory,
        then infers the actions needed to follow that trajectory using the IDM.
        """
        batch_size = obs_batch_np.shape[0]
        assert batch_size == self.num_envs, "Input batch size must match the number of environments."

        # Initialization for the planning loop
        final_trajectories = np.zeros((batch_size, self.horizon, self.H, self.W, self.C), dtype=np.int32)
        plan_is_finalized = np.zeros(batch_size, dtype=bool)
        indices_to_plan_for = np.arange(batch_size)

        # Main Replanning Loop
        for attempt in range(self.max_replan_attempts):
            if not self.automatic_replan and attempt > 0:
                break
            if np.all(plan_is_finalized):
                break

            sub_batch_obs = obs_batch_np[indices_to_plan_for]
            sub_batch_policy_ids = policy_id_batch_np[indices_to_plan_for]
            sub_batch_history = [self.history_buffers[i] for i in indices_to_plan_for]

            # Propose candidate plans for the sub-batch
            sub_candidate_plans_np = self.action_proposer.get_action_plans(
                current_obs_batch_np=sub_batch_obs,
                history_batch_list=sub_batch_history,
                target_reward=self.target_reward,
                num_candidates=self.num_action_candidates
            )
            
            # Simulate and select the BEST TRAJECTORIES for the sub-batch
            sub_best_trajectories, needs_replan_mask = self._simulate_and_select_best_plans(
                sub_batch_obs, sub_batch_policy_ids, sub_candidate_plans_np
            )
            
            # Store the successful TRAJECTORIES in our final result array
            successful_sub_indices = np.where(~needs_replan_mask)[0]
            if len(successful_sub_indices) > 0:
                original_successful_indices = indices_to_plan_for[successful_sub_indices]
                final_trajectories[original_successful_indices] = sub_best_trajectories[successful_sub_indices]
                plan_is_finalized[original_successful_indices] = True
            
            indices_to_plan_for = np.where(~plan_is_finalized)[0]

        if not np.all(plan_is_finalized):
            failed_env_indices = np.where(~plan_is_finalized)[0]
            raise RuntimeError(f"Planning failed after {self.max_replan_attempts} attempts. Could not find a feasible trajectory for environments: {failed_env_indices}.")

        # Convert Winning Trajectories to Actions
        normed_obs_batch_np = obs_batch_np / 255.0
        final_trajectories = np.concatenate([normed_obs_batch_np[:, None, ...], final_trajectories], axis=1)
        obs_t_batch = final_trajectories[:, :-1, ...]
        obs_tp1_batch = final_trajectories[:, 1:, ...]
        
        obs_t_flat = obs_t_batch.reshape(batch_size * self.horizon , self.H, self.W, self.C)
        obs_tp1_flat = obs_tp1_batch.reshape(batch_size * self.horizon , self.H, self.W, self.C)

        action_indices_flat = self._get_idm_actions(obs_t_flat, obs_tp1_flat)
        final_action_plans = action_indices_flat.reshape(batch_size, self.horizon)

        print(f"Final action plans shape: {final_action_plans.shape}, {final_action_plans}")

        return final_action_plans[:, :self.planning_horizon]
    
    
    def _simulate_and_select_best_plans(self, current_obs_batch_np: np.ndarray, policy_id_batch_np: np.ndarray, candidate_plans_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the single best simulated trajectory for each environment by looking for
        the highest positive reward across all plans and simulations.
        """
        batch_size = current_obs_batch_np.shape[0]
        num_plans = self.num_action_candidates
        num_sims = self.num_simulations_per_plan
        total_simulations = batch_size * num_plans * num_sims

        # Prepare Inputs for the GPU Batch 
        current_obs_norm = normalize_obs_vectorized(current_obs_batch_np, num_channels=self.C)
        current_obs_th = torch.from_numpy(current_obs_norm).to(self.device, dtype=torch.float32)
        current_obs_rearranged = rearrange(current_obs_th, 'b h w c -> b c h w')
        sim_obs_cond = repeat(current_obs_rearranged, 'b c h w -> (b n m) c h w', n=num_plans, m=num_sims)
        
        plans_th = torch.from_numpy(candidate_plans_np).to(self.device, dtype=torch.int64)
        sim_action_embed = repeat(plans_th, 'b n h -> (b n m) h', m=num_sims)[:, :self.action_horizon, ...]
        
        policy_id_th = torch.from_numpy(policy_id_batch_np).to(self.device, dtype=torch.int64)
        sim_policy_id = repeat(policy_id_th, 'b -> (b n m)', n=num_plans, m=num_sims)

        # Run All Simulations in a Single Batched GPU Call 
        simulated_trajectories_th = self.world_model.sample(
            x_cond=sim_obs_cond,
            action_embed=sim_action_embed,
            task_embed=sim_policy_id,
            batch_size=total_simulations
        )

        # Evaluate Trajectories
        trajectories_processed = rearrange(simulated_trajectories_th, 'b (f c) h w -> b f h w c', 
                                              c=self.C, f=self.horizon)
        trajectories_np = self._process_diffusion_trajectory(trajectories_processed.cpu().numpy())

        work_items = [(self.args, trajectories_np[i]) for i in range(total_simulations)]
        with multiprocessing.get_context("fork").Pool(processes=self.num_processes) as pool:
            trajectory_rewards = pool.map(_reward_calculation_worker, work_items)
        
        rewards_by_plan = np.array(trajectory_rewards, dtype=np.float32).reshape(batch_size, num_plans, num_sims)
        selectable_scores = np.where(rewards_by_plan >= 0, rewards_by_plan, -np.inf)
        scores_flat_per_env = selectable_scores.reshape(batch_size, num_plans * num_sims)
        best_sim_indices_flat = np.argmax(scores_flat_per_env, axis=1)
        max_scores = np.max(scores_flat_per_env, axis=1)
        needs_replan_mask = max_scores == -np.inf
        
        # Extract and Return the Best Trajectories
        env_offsets = np.arange(batch_size) * (num_plans * num_sims)
        global_best_indices = env_offsets + best_sim_indices_flat

        best_trajectories_np = trajectories_np[global_best_indices]
        
        if self.debug:
            for i in range(batch_size * self.num_action_candidates * self.num_simulations_per_plan):
                print(f"  Score: {trajectory_rewards[i]:.2f} for plan {i // self.num_simulations_per_plan} in env {i // (self.num_action_candidates * self.num_simulations_per_plan)}")
            for i in range(batch_size):
                if needs_replan_mask[i]:
                    print(f"Env {i}: NEEDS REPLAN. No simulation had a non-negative reward.")
                else:
                    print(f"Env {i}: Selected best trajectory with max score: {max_scores[i]:.2f}")

        return best_trajectories_np, needs_replan_mask
    
    
    def _get_idm_actions(self, obs_t_batch_np, obs_tp1_batch_np):
        """Helper to infer actions between two batches of numpy observations."""
        states_t_batch = self.idm.invert_obs_to_state_batch(obs_t_batch_np)
        states_tp1_batch = self.idm.invert_obs_to_state_batch(obs_tp1_batch_np)
        return np.array(self.idm.find_ego_actions_batch(states_t_batch, states_tp1_batch, return_indices=True), dtype=np.int64)

    
    def normalize_obs(self, obs, divide=True):
        return normalize_obs_vectorized(obs, divide=divide)
    
    def _process_diffusion_trajectory(self, trajectory_np):
        """
        Processes a trajectory from the diffusion model.
        Converts it to the expected format and checks for value ranges.
        """
        assert trajectory_np.ndim == 5, "Expected trajectory shape [B, T, H, W, C]"
        assert np.round(trajectory_np.min()) >= -1 and np.round(trajectory_np.max()) <= 1, \
            "Trajectory values should be in the range [-1, 1]"
        
        B, T, H, W, C = trajectory_np.shape
        assert H == self.H and W == self.W and C == self.C, \
            f"Expected trajectory shape [B, T, {self.H}, {self.W}, {self.C}], but got {trajectory_np.shape}"
        
        traj = np.clip(trajectory_np, -1.0, 1.0)

        max_values = {
            "onions_in_pot": 3.0,
            "tomatoes_in_pot": 3.0,
            "onions_in_soup": 3.0,
            "tomatoes_in_soup": 3.0,
            "soup_cook_time_remaining": 20.0
        }

        # This vector will have shape (C,) or (1, 1, 1, 1, C) to align with the trajectory.
        # Start with a default scaling factor of 1.0 for all channels (for binary 0-1 features).
        scale_factors = np.ones(C, dtype=np.float32)

        for ch_name, max_val in max_values.items():
            if ch_name in self.CHANNEL_FEATURE_MAP:
                ch_idx = self.CHANNEL_FEATURE_MAP[ch_name]
                scale_factors[ch_idx] = max_val
        
        # Rescale each channel based on its max value
        unnorm_traj = ((traj + 1.0) * 0.5) * scale_factors
        unnorm_traj = np.round(unnorm_traj).astype(np.int32)


        assert unnorm_traj.min() >= 0 and unnorm_traj.max() <= 20, \
            f"Unnormalized trajectory values should be in the range [0, 20], got min: {unnorm_traj.min()}, max: {unnorm_traj.max()}"

        return unnorm_traj

    
    def setup_concept_learning(self):
        """
        Prepares the agent to learn concepts by setting the embedding matrix,
        freezing old weights, and creating a new optimizer.
        """
        print("Setting up agent for concept learning...")
        
        # --- Embedding and Guidance Setup ---
        # The agent receives the full embedding matrix to work with
        W = torch.randn(self.num_concepts, self.model_embedding_dim, device=self.device)
        new_embedding = torch.nn.Embedding.from_pretrained(W, freeze=False).to(self.device)
        
        # Replace embedding layers
        goal_gaussian_model = self.accelerator.unwrap_model(self.world_model)
        goal_gaussian_model.model.unet.label_emb = new_embedding

        # --- Parameter Freezing ---
        # Freeze all parameters first
        for param in goal_gaussian_model.parameters():
            param.requires_grad = False
        # Then, unfreeze only the new embedding layer
        goal_gaussian_model.model.unet.label_emb.weight.requires_grad = True

        # --- Optimizer Setup ---
        params_to_optimize = [p for p in self.world_model.parameters() if p.requires_grad]
        expected_params = len(list(goal_gaussian_model.model.unet.label_emb.parameters()))

        assert len(params_to_optimize) == expected_params, \
            f"Expected {expected_params} trainable parameters, but found {len(params_to_optimize)}"
        
        self.optimizer = Adam(params_to_optimize, lr=self.args.train_lr)
        self.world_model_accel, self.optimizer, = self.accelerator.prepare(self.world_model, self.optimizer)
        self.world_model_accel.train()
        
        print(f"Concept learning setup complete with {self.num_concepts} concepts and embedding dim {self.model_embedding_dim}.")
    
    def concept_learn(self):
        """ Runs the concept learning training loop."""
        if self.optimizer is None:
            raise RuntimeError("Concept learning not set up. Call setup_concept_learning() first.")
        print("Starting concept learning training loop...")
        self.world_model.train()
        with tqdm(total=self.training_steps, desc="Concept Learning Training Steps") as pbar:
            for step in range(self.training_steps):
                # Sample a batch of concept learning experiences
                initial_obs, obs_trajectories_np, actions = self.sample_concept_learning_batch(self.batch_size)

                initial_obs = normalize_obs_vectorized(initial_obs, divide=True)
                obs_trajectories_np = normalize_obs_vectorized(obs_trajectories_np, divide=True)

                # Perform a single training step
                loss = self.learn_concept_one_step(obs_trajectories_np, initial_obs, actions)
                
                pbar.update(1)
                pbar.set_postfix({'loss': f"{loss:.4f}."})
        
        self.save_agent_state(self.debug_dir / f'concept_learning_state_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth')
                

    def learn_concept_one_step(self, obs_trajectories_np, initial_conditions_np, actions_np):
        """
        Performs a single training step to learn a concept from a batch of experience.
        
        Args:
            obs_trajectories_np (np.ndarray): Shape [B, T, H, W, C].
            initial_conditions_np (np.ndarray): Shape [B, H, W, C].
            new_concept_id (int): The class ID for the concept being learned.
            dummy_id (int): The class ID for the unconditional (dummy) case.

        Returns:
            float: The loss for this training step.
        """
        # Convert data to PyTorch tensors
        x = torch.from_numpy(obs_trajectories_np).to(self.device, dtype=torch.float32)
        x_cond = torch.from_numpy(initial_conditions_np).to(self.device, dtype=torch.float32)
        action_embed = torch.from_numpy(actions_np).to(self.device, dtype=torch.int64)
        
        total_loss = 0.0
        with self.accelerator.autocast():
            loss = self._calculate_concept_loss(x, x_cond, action_embed)
            total_loss += loss.item()
            
            self.accelerator.backward(loss)

        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return total_loss
    
    def _chkpt_forward(self, x, t, x_cond, policy_id, actions):
        return self.world_model.p_losses(
            x_start=x, t=t, x_cond=x_cond, task_embed=policy_id, action_embed=actions, 
            reward_embed=None, image_embed=None,history_embed=None,
        )

    def _calculate_concept_loss(self, x, x_cond, actions):
        """Internal loss calculation logic for concept learning."""
        batch_size = x.shape[0]

        # Reshape for diffusion model
        x = rearrange(x, "b f h w c -> b (f c) h w")
        x_cond  = rearrange(x_cond, "b h w c -> b c h w")
        
        t = torch.randint(0, self.world_model.num_timesteps, (batch_size,), device=self.device).long()
        
        # Calculate unconditional loss (conditioned on dummy_id)
        uncond_loss = self.world_model.p_losses(
            x, t, x_cond=x_cond, 
            task_embed=torch.full((batch_size,), self.dummy_id, device=self.device, dtype=torch.long), 
            action_embed=actions
        )

        if self.guidance_option == "learnable":
            guidance_weights = F.softmax(self.guidance_logits, dim=-1)
        else:
            guidance_weights = self.guidance_weight

        concept_ids = [c_id for c_id in range(1, self.num_concepts + 1)]
        weighted_cond_loss = 0.0
        for i, concept_id in enumerate(concept_ids):
            assert concept_id != self.dummy_id, "Dummy ID should not be in the concept IDs"
            cond_id = torch.full((batch_size,), concept_id, device=self.device, dtype=torch.long)
            cond_loss = cp.checkpoint(
                self._chkpt_forward, x, t, x_cond, cond_id, actions,
                use_reentrant=False if self.use_checkpoint else True
            )
            weighted_cond_loss += guidance_weights[i] * cond_loss

        total_loss = uncond_loss + weighted_cond_loss
        return total_loss

    def save_concept_learn_embedding(self, save_path):
        """
        Saves the learned concept embedding matrix to a file.
        
        Args:
            save_path (str): Path to save the embedding matrix.
        """
        goal_gaussian_model = self.accelerator.unwrap_model(self.world_model)
        embedding_matrix = goal_gaussian_model.model.unet.label_emb.weight.data.cpu().numpy()
        torch.save(embedding_matrix, save_path)
        print(f"Concept embedding matrix saved to {save_path}")
    
    def save_agent_state(self, save_path):
        """
        Saves the current state of the agent, including the world model and action proposer.
        
        Args:
            save_path (str): Path to save the agent state.
        """
        state = {
            'world_model': self.world_model.state_dict(),
            'action_proposer': self.action_proposer.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'history_buffers': [list(buf) for buf in self.history_buffers],
            'concept_learning_buffer': self.concept_learning_buffer.buffer,
            'guidance_weight': self.guidance_weight.cpu().numpy()
        }
        torch.save(state, save_path)
        print(f"Agent state saved to {save_path}")
    

