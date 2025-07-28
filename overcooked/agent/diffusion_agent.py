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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def _reward_calculation_worker(work_item):
    """Worker function for multiprocessing. Initializes its own RewardCalculator."""
    args, trajectory_np = work_item
    local_reward_calculator = GroundTruthRewardCalculator(args)
    obs_t_batch = trajectory_np[:-1]
    obs_tp1_batch = trajectory_np[1:]
    ego_rewards, partner_rewards = local_reward_calculator.calculate_reward_batch(obs_t_batch, obs_tp1_batch)
    return np.sum(ego_rewards) + np.sum(partner_rewards)

def _concept_learning_collate_fn(batch):
    """Custom collate function for concept learning data."""
    initial_obs = np.array([item['initial_obs'] for item in batch])
    trajectories = np.array([item['trajectory'] for item in batch])
    actions = np.array([item['actions'] for item in batch])
    
    # Pre-normalize here to avoid doing it in the training loop
    initial_obs_norm = normalize_obs_vectorized(initial_obs, divide=True)
    trajectories_norm = normalize_obs_vectorized(trajectories, divide=True)
    
    return {
        'initial_obs': initial_obs_norm,
        'trajectory': trajectories_norm,
        'actions': actions
    }


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
        initial_obs, trajectory, actions = self.buffer[idx]
        return {
            'initial_obs': initial_obs,
            'trajectory': trajectory, 
            'actions': actions
        }


class DiffusionPlannerAgent:
    """ A diffusion-based planning agent for Overcooked.
    This agent uses a diffusion model to generate action plans based on the current state of the game.
    It integrates with the world model, action proposal model, and inverse dynamics model to simulate and evaluate plans.
    
    Args:
        args: Command line arguments and configurations.
        world_model: The world model used for simulating future states.
        action_proposal_model: The model that proposes action plans.
        num_envs: Number of parallel environments to run.
        horizon: Sequence horizon for trajectories.
        target_reward: Target reward for the planning process.
        history_horizon: Length of the history to consider for planning.
        planning_horizon: Length of the plan to generate.
        action_horizon: Length of the action sequence to predict.
        num_action_candidates: Number of candidate action plans to consider.
        num_simulations_per_plan: Number of simulations per candidate plan.
        automatic_replan: Whether to automatically replan if initial plans fail.
        max_replan_attempts: Maximum number of attempts to replan if initial plans fail.
        num_concepts: Number of concepts for concept learning.
        model_embedding_dim: Dimensionality of the model embeddings.
        use_checkpoint: Whether to use checkpointing for memory efficiency during training.
        guidance_option: Type of guidance to use ("scalar" or "learnable").
        guidance_weight: Weight for the guidance loss (if applicable).
        cl_buffer_size: Size of the concept learning buffer.
        num_processes: Number of processes for parallel computation.
        device: Device to run the models on (CPU or GPU).

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
                 automatic_replan=True, max_replan_attempts=3, minimum_cl_steps=10,
                 num_concepts=10, model_embedding_dim=256*4, use_checkpoint=True, guidance_option="scalar", guidance_weight=1.0,
                 cl_buffer_size=256, num_processes=None, device=None, training_steps=10, batch_size=16, train_lr=1e-4):
        
        self.args = args
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent models operating on device: {self.device}")

        # Models 
        self.world_model = world_model.to(self.device)
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
        self.debug = False

        if self.debug:
            self.debug_dir = Path("DiffusionPlannerAgent_debug") / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"Debugging enabled. Output will be saved to {self.debug_dir}")

        # --- Concept Learning Components ---
        self.optimizer = None
        self.num_concepts = num_concepts
        self.model_embedding_dim = model_embedding_dim
        self.use_checkpoint = use_checkpoint
        self.guidance_option = guidance_option
        self.env_cl_buffers = [deque(maxlen=self.horizon + 1) for _ in range(self.num_envs)]
        self.concept_learning_buffer = ConceptLearningBuffer(max_size=cl_buffer_size)
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.train_lr = train_lr

        self.cl_steps = 0
        self.minimum_cl_steps = minimum_cl_steps

        if guidance_option == "scalar":
            if isinstance(guidance_weight, (list, np.ndarray)):
                assert len(guidance_weight) == self.num_concepts, f"Expected {self.num_concepts} weights, got {len(guidance_weight)}"
                self.guidance_weight = torch.tensor(guidance_weight, device=self.device, dtype=torch.float32)
                self.guidance_weight = self.guidance_weight / self.guidance_weight.sum()
            else:
                # Uniform distribution
                self.guidance_weight = torch.ones(self.num_concepts, device=self.device) / self.num_concepts
        elif guidance_option == "learnable":
            # Initialize with small random values
            self.guidance_logits = nn.Parameter(torch.randn(self.num_concepts, device=self.device) * 0.1)  
        
        # Internal dimensions
        self.H, self.W, self.C = (8, 5, 26)
        self.dummy_id = 0
        self.no_op_action = 4
        self.num_actions = 6

    def reset(self, env_indices=None):
        """Resets the history buffers for specified (or all) environments."""
        if env_indices is None:
            env_indices = range(self.num_envs)
        for i in env_indices:
            self.history_buffers[i].clear()
        print(f"Reset history for environments: {env_indices}")
    
    def add_concept_learning_experience(self, obs_batch: np.ndarray, action_batch: np.ndarray):
        """
        Adds trajectory experiences from all environments to the concept learning buffer.
        Uses a sliding window approach for more efficient data collection.
        
        Args:
            obs_batch: Observations from all environments [num_envs, H, W, C]
            action_batch: Actions from all environments [num_envs,]
        """
        assert obs_batch.shape[0] == self.num_envs, f"Expected {self.num_envs} envs, got {obs_batch.shape[0]}"
        
        # Process each environment's trajectory buffer
        for env_idx in range(self.num_envs):
            obs = obs_batch[env_idx]
            action = action_batch[env_idx]
            
            self.env_cl_buffers[env_idx].append((obs, action))
            
            # Use sliding window: once we have enough steps, extract trajectories
            if len(self.env_cl_buffers[env_idx]) >= self.horizon + 1:
                # Extract the trajectory
                trajectory_data = list(self.env_cl_buffers[env_idx])
                initial_obs = trajectory_data[0][0]
                obs_trajectory = np.array([x[0] for x in trajectory_data[1:]])  # Next horizon observations
                actions = np.array([x[1] for x in trajectory_data[:-1]])  # Actions that led to those observations
                
                self.concept_learning_buffer.add_experience(initial_obs, obs_trajectory, actions)
                self.env_cl_buffers[env_idx].popleft()  
        
        # # Log less frequently
        # if len(self.concept_learning_buffer) % 100 == 0:
        print(f"Concept learning buffer size: {len(self.concept_learning_buffer)}")
    
    def _sample_concept_learning_batch(self, batch_size=32):
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
    
    @torch.no_grad()
    def get_plan(self, obs_batch_np: np.ndarray, policy_id_batch_np: np.ndarray) -> np.ndarray:
        """
        The main public method. Simulates futures to find the best possible trajectory,
        then infers the actions needed to follow that trajectory using the IDM.
        """
        batch_size = obs_batch_np.shape[0]
        assert batch_size == self.num_envs, "Input batch size must match the number of environments."

        self.world_model.eval()

        # Initialization for the planning loop
        final_trajectories = np.zeros((batch_size, self.horizon, self.H, self.W, self.C), dtype=np.int32)
        plan_is_finalized = np.zeros(batch_size, dtype=bool)
        indices_to_plan_for = np.arange(batch_size)

        if self.debug and self.grid is None:
            grid_obs = obs_batch_np[0].astype(np.float32) / 255.0
            grid_obs = np.transpose(grid_obs, (1, 0, 2))
            self.grid = self.renderer.extract_grid_from_obs(grid_obs)


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

        # if self.debug and len(self.concept_learning_buffer) > self.batch_size:
        #     with torch.enable_grad():
        #         self.concept_learn()
        #     self.world_model.eval()

        if self.cl_steps > self.minimum_cl_steps:
            concept_ids = [c_id for c_id in range(1, self.num_concepts + 1)]
            if self.guidance_option == "learnable":
                concept_weights = F.softmax(self.guidance_logits, dim=-1)
            else:
                # Will use uniform weights
                concept_weights = None
            
            print(f"Using concept weights: {concept_weights} for {self.num_concepts} concepts.")
            simulated_trajectories_th = self.world_model.sample(
                x_cond=sim_obs_cond,
                task_embed=concept_ids,
                action_embed=sim_action_embed,
                batch_size=total_simulations,
                concept_weights=concept_weights,
            )

            if self.debug:
                print_trajs = rearrange(simulated_trajectories_th, 'b (f c) h w -> b f w h c', 
                                              c=self.C, f=self.horizon)
                for i in range(total_simulations):
                    video_path = self.debug_dir / "simulated_trajectories" /f'simulated_trajectories_attempt_batch_{i}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4'
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    self.renderer.render_trajectory_video(
                        print_trajs[i].cpu().numpy(),
                        self.grid,
                        output_dir=str(self.debug_dir),
                        video_path=str(video_path),
                        normalize=True,
                        fps=1
                    )
                    channel_path = self.debug_dir / "simulated_trajectories" / f'simulated_trajectory_channels_{i}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
                    self.renderer.visualize_all_channels(
                        print_trajs[i, 0].cpu().numpy(), 
                        output_dir=channel_path,
                    )


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
        
        # Initialize the embedding layer for concepts
        W = torch.randn(self.num_concepts + 1, self.model_embedding_dim, device=self.device)
        new_embedding = torch.nn.Embedding.from_pretrained(W, freeze=False).to(self.device)
        
        # Replace embedding layers
        self.world_model.model.unet.label_emb = new_embedding

        # Freeze all parameters
        for param in self.world_model.parameters():
            param.requires_grad = False
        
        # Then, unfreeze only the new embedding layer
        self.world_model.model.unet.label_emb.weight.requires_grad = True

        # --- Optimizer Setup ---
        params_to_optimize = [p for p in self.world_model.parameters() if p.requires_grad]
        expected_params = len(list(self.world_model.model.unet.label_emb.parameters()))

        assert len(params_to_optimize) == expected_params, \
            f"Expected {expected_params} trainable parameters, but found {len(params_to_optimize)}"
        
        self.optimizer = Adam(params_to_optimize, lr=self.train_lr)
        print(f"Concept learning setup complete with {self.num_concepts} concepts and embedding dim {self.model_embedding_dim}.")
    
    @torch.enable_grad()
    def concept_learn(self):
        """ Runs the concept learning training loop."""
        if self.optimizer is None:
            raise RuntimeError("Concept learning not set up. Call setup_concept_learning() first.")
        
        print("Starting concept learning training loop...")
        self.world_model.train()

        if len(self.concept_learning_buffer) < self.batch_size:
            print(f"Not enough samples in the concept learning buffer. Required: {self.batch_size}, Available: {len(self.concept_learning_buffer)}")
            return
        
        dataloader = DataLoader(
            self.concept_learning_buffer,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            collate_fn=_concept_learning_collate_fn 
        )

        num_epochs = max(1, self.training_steps // len(dataloader))
        total_steps = 0

        with tqdm(total=self.training_steps, desc="Concept Learning") as pbar:
            for epoch in range(num_epochs):
                for batch_idx, batch in enumerate(dataloader):
                    if total_steps >= self.training_steps:
                        break
                    
                    # Extract batch data
                    initial_obs = batch['initial_obs']
                    obs_trajectories = batch['trajectory']
                    actions = batch['actions']
                    
                    # Training step
                    loss = self._learn_concept_one_step(obs_trajectories, initial_obs, actions)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'epoch': epoch + 1,
                        'loss': f"{loss:.4f}",
                        'batch': f"{batch_idx + 1}/{len(dataloader)}"
                    })
                    
                    total_steps += 1
                    self.cl_steps += 1
        
        self.world_model.eval()
        # self.save_agent_state(self.debug_dir / f'concept_learning_state_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth')
                

    def _learn_concept_one_step(self, obs_trajectories_np, initial_conditions_np, actions_np):
        """
        Performs a single training step to learn a concept from a batch of experience.
        """
        # Convert data to PyTorch tensors
        x = torch.from_numpy(obs_trajectories_np).to(self.device, dtype=torch.float32)
        x_cond = torch.from_numpy(initial_conditions_np).to(self.device, dtype=torch.float32)
        action_embed = F.one_hot(torch.from_numpy(actions_np[:, :self.action_horizon]), num_classes=self.num_actions).to(self.device, dtype=torch.float32)

        loss = self._calculate_concept_loss(x, x_cond, action_embed)
        if not loss.requires_grad:
            # Additional debugging
            print("WARNING: Loss does not require gradients!")
            # Check if any intermediate tensors have gradients
            for name, param in self.world_model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.world_model.parameters() if p.requires_grad], max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _chkpt_forward(self, x, t, x_cond, policy_id, actions):
        return self.world_model.p_losses(
            x_start=x, t=t, x_cond=x_cond, task_embed=policy_id, action_embed=actions,
        )

    def _calculate_concept_loss(self, x, x_cond, actions):
        """Internal loss calculation logic for concept learning."""
        batch_size = x.shape[0]
        self.world_model.train()

        # Reshape for diffusion model
        x = rearrange(x, "b f h w c -> b (f c) h w")
        x_cond  = rearrange(x_cond, "b h w c -> b c h w")
        
        t = torch.randint(0, self.world_model.num_timesteps, (batch_size,), device=self.device).long()
        
        # Calculate unconditional loss
        uncond_id = torch.full((batch_size,), self.dummy_id, device=self.device, dtype=torch.long)
        uncond_actions = torch.ones_like(actions, device=self.device, dtype=torch.float32) * self.no_op_action
        if self.use_checkpoint:
            uncond_loss = cp.checkpoint(
                self._chkpt_forward, x, t, x_cond, uncond_id, uncond_actions,
                use_reentrant=False
            )
        else:
            uncond_loss = self.world_model.p_losses(
                x, t, x_cond=x_cond, 
                task_embed=uncond_id, 
                action_embed=uncond_actions
            )

        if self.guidance_option == "learnable":
            guidance_weights = F.softmax(self.guidance_logits, dim=-1)
        else:
            guidance_weights = self.guidance_weight

        concept_ids = [c_id for c_id in range(1, self.num_concepts + 1)]
        weighted_cond_loss = 0.0
        for i, concept_id in enumerate(concept_ids):
            cond_id = torch.full((batch_size,), concept_id, device=self.device, dtype=torch.long)
            if self.use_checkpoint:
                cond_loss = cp.checkpoint(
                    self._chkpt_forward, x, t, x_cond, cond_id, actions,
                    use_reentrant=False
                )
            else:
                cond_loss = self.world_model.p_losses(
                    x, t, x_cond=x_cond, 
                    task_embed=cond_id, 
                    action_embed=actions
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
        # Prepare guidance weights/logits
        if self.guidance_option == "learnable":
            guidance_data = {
                'type': 'learnable',
                'logits': self.guidance_logits.detach().cpu().numpy()
            }
        else:
            guidance_data = {
                'type': 'scalar',
                'weights': self.guidance_weight.detach().cpu().numpy() if isinstance(self.guidance_weight, torch.Tensor) else self.guidance_weight
            }
        
        # Convert concept learning buffer to serializable format
        cl_buffer_data = []
        for initial_obs, trajectory, actions in self.concept_learning_buffer.buffer:
            cl_buffer_data.append({
                'initial_obs': initial_obs,
                'trajectory': trajectory,
                'actions': actions
            })
        
        state = {
            # Model states
            'world_model': self.world_model.state_dict(),
            'action_proposer': self.action_proposer.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            
            # Agent configuration
            'num_concepts': self.num_concepts,
            'guidance_option': self.guidance_option,
            'guidance_data': guidance_data,
            
            # Buffers
            'history_buffers': [list(buf) for buf in self.history_buffers],
            'concept_learning_buffer': cl_buffer_data,
            
            # Training state
            'training_steps_completed': getattr(self, 'training_steps_completed', 0),
            
            # Metadata
            'save_timestamp': datetime.datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        # Create parent directory if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(state, save_path)
        print(f"Agent state saved to {save_path}")
    def load_agent_state(self, load_path):
        """
        Loads a previously saved agent state.
        
        Args:
            load_path (str): Path to load the agent state from.
        """
        print(f"Loading agent state from {load_path}")
        state = torch.load(load_path, map_location=self.device)
        
        # Load model states
        self.world_model.load_state_dict(state['world_model'])
        self.action_proposer.load_state_dict(state['action_proposer'])
        
        # Load optimizer if it exists
        if state['optimizer'] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state['optimizer'])
        
        # Load guidance weights/logits
        guidance_data = state['guidance_data']
        if guidance_data['type'] == 'learnable':
            self.guidance_logits = nn.Parameter(
                torch.tensor(guidance_data['logits'], device=self.device, dtype=torch.float32)
            )
        else:
            self.guidance_weight = torch.tensor(
                guidance_data['weights'], device=self.device, dtype=torch.float32
            )
        
        # Restore buffers
        self.history_buffers = [deque(buf, maxlen=len(buf)) for buf in state['history_buffers']]
        
        # Restore concept learning buffer
        self.concept_learning_buffer.buffer.clear()
        for item in state['concept_learning_buffer']:
            self.concept_learning_buffer.add_experience(
                item['initial_obs'], 
                item['trajectory'], 
                item['actions']
            )
        
        print(f"Agent state loaded successfully from {load_path}")
        print(f"Loaded {len(self.concept_learning_buffer)} concept learning experiences")
    

