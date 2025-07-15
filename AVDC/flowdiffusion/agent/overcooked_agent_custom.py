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

sys.path.append("/home/law/Workspace/repos/COMBO/AVDC/flowdiffusion")
mapbt_path = '/home/law/Workspace/repos/COMBO/mapbt_package'
if mapbt_path not in sys.path:
    sys.path.append(mapbt_path)
from einops.einops import rearrange
import numpy as np
import torch as th
import warnings
warnings.filterwarnings("ignore")
from goal_diffusion import GoalGaussianDiffusion
from unet import UnetOvercooked 
from experiments_util import to_torch
import numpy as np
import os.path as osp
from idm.ground_truth_idm import GroundTruthInverseDynamics
from reward_module.state_reward_model import RewardCalculator as GroundTruthRewardCalculator
from ema_pytorch import EMA
from contextlib import contextmanager
import multiprocessing
from overcooked_sample_renderer import OvercookedSampleRenderer
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.actions import Action

def _calculate_trajectory_reward_worker(work_item):
    """
    Worker function for multiprocessing. It now initializes its own
    RewardCalculator to avoid pickling issues.
    
    Args:
        work_item (tuple): A tuple containing (args, trajectory_obs_np).
    """
    args, trajectory_obs_np = work_item
    from reward_module.state_reward_model import RewardCalculator
    
    local_reward_calculator = RewardCalculator(args)
    obs_t_batch = trajectory_obs_np[:-1]
    obs_tp1_batch = trajectory_obs_np[1:]
    ego_rewards, partner_rewards = local_reward_calculator.calculate_reward_batch(obs_t_batch, obs_tp1_batch)
    print(f"Calculated rewards for trajectory with shape {trajectory_obs_np.shape}, "
            f"ego rewards: {np.sum(ego_rewards)}, partner rewards: {np.sum(partner_rewards)}")  
    # return np.sum(ego_rewards)
    return np.sum(ego_rewards) + np.sum(partner_rewards)  # Return the total reward for the trajectory


class DiffusionPlannerAgent:
    """
    An agent that uses diffusion models to plan a sequence of actions.
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

    def __init__(self, args, world_model, action_proposal_model, horizon=32,
                 planning_horizon=32, action_horizon=8, num_candidates=10, num_actions=6,
                 num_processes=None, device=None):
        
        self.args = args
        self.device = device if device else th.device("cuda" if th.cuda.is_available() else "cpu")
        print(f"Agent models operating on device: {self.device}")

        # Models 
        self.world_model = world_model
        # The action_proposal_model is replaced by the motion planner
        self.idm = GroundTruthInverseDynamics(self.args) 
        self.reward_calculator = GroundTruthRewardCalculator(self.args)

        # Initialize the motion planner
        self.mdp = OvercookedGridworld.from_layout_name(args.layout_name)
        self.mlam = MediumLevelActionManager(self.mdp, NO_COUNTERS_PARAMS)
        self.jmp = self.mlam.joint_motion_planner

        # Configuration
        self.horizon = horizon
        self.planning_horizon = planning_horizon
        self.action_horizon = action_horizon
        self.num_candidates = num_candidates
        self.num_actions = num_actions
        self.num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count() // 2

        # Debugging
        self.renderer = OvercookedSampleRenderer()
        self.grid = None
        self.debug = True

        if self.debug:
            self.debug_dir = Path("DiffusionPlannerAgent_debug") / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"Debugging enabled. Output will be saved to {self.debug_dir}")
        
        # Internal dimensions
        self.H, self.W, self.C = (8, 5, 26)

    @th.no_grad()
    def get_plan(self, current_obs_np, policy_id_np):
        """
        The main public method. Given the current observation, it returns a plan.
        Accepts and returns NumPy arrays.

        Args:
            current_obs_np (np.ndarray): Shape [B, H, W, C].
            policy_id_np (np.ndarray): Shape [B].

        Returns:
            np.ndarray: A plan of ego-agent actions, shape [B, T].
        """
        # Debug
        if not self.grid:
            grid_obs = current_obs_np[0].transpose(1, 0, 2)
            assert grid_obs.shape == (self.W, self.H, self.C)
            self.grid = self.renderer.extract_grid_from_obs(grid_obs)
        
        # Normalize the observation
        current_obs_norm = self.normalize_obs(current_obs_np)
        
        # Generate and rank candidate trajectories
        ranked_candidates_dict = self._rank_and_select_candidates(current_obs_norm, policy_id_np)
        best_trajectory_np = ranked_candidates_dict["ranked_trajectories"]
        print("best_trajectory_np: ",best_trajectory_np.min(), best_trajectory_np.max())

        B, T, H, W, C = best_trajectory_np.shape

        # Infer actions from the best trajectory using the IDM
        init_states_np = best_trajectory_np[:, 0, ...]
        init_actions = self._get_idm_actions(current_obs_np.astype(np.int32) / 255.0, init_states_np)
        
        # Subsequent actions for the rest of the trajectory
        obs_t_flat = best_trajectory_np[:, :-1, ...].reshape(B * (T - 1), H, W, C)
        obs_tp1_flat = best_trajectory_np[:, 1:, ...].reshape(B * (T - 1), H, W, C)
        traj_actions = self._get_idm_actions(obs_t_flat, obs_tp1_flat)
        
        # 3. Combine and return the full action plan
        full_plan = np.concatenate([init_actions, traj_actions])
        full_plan = full_plan.reshape(B, T)[:, :self.planning_horizon + 1]
        return full_plan

    def _get_idm_actions(self, obs_t_batch_np, obs_tp1_batch_np):
        """Helper to infer actions between two batches of numpy observations."""
        states_t_batch = self.idm.invert_obs_to_state_batch(obs_t_batch_np)
        states_tp1_batch = self.idm.invert_obs_to_state_batch(obs_tp1_batch_np)
        return np.array(self.idm.find_ego_actions_batch(states_t_batch, states_tp1_batch, return_indices=True), dtype=np.int64)

    def _rank_and_select_candidates(self, current_obs_np, policy_id_np):
        """Internal method to generate, rank, and select the best trajectories."""
        trajectories_np = self._generate_trajectory_candidates(current_obs_np, policy_id_np)
        ranked_candidates = self._rank_trajectories(trajectories_np)
        ranked_trajectories = np.stack([env_list[0][1] for env_list in ranked_candidates], axis=0)
        assert ranked_trajectories.min() >= 0 and ranked_trajectories.max() <= 20
        return {
            "ranked_candidates": ranked_candidates,
            "ranked_trajectories": ranked_trajectories,
        }
    
    def normalize_obs(self, obs, divide=True):
        """ Normalizes a numpy observation array to [-1, 1]. 
        If `divide` is True, it un-scales the observation by 255.0.
        """
        # Scaled down by 255 since data is scaled by 255
        if divide:
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
    
    def _after_generate_trajectory_candidates(self, current_obs_norm_np, policy_id_np):
        """
        Generates candidate trajectories using the "Full Delivery Cycle" fixed policy.
        """
        batch_size = current_obs_norm_np.shape[0]

        # --- Prepare inputs ---
        current_obs_th = th.from_numpy(current_obs_norm_np).to(self.device, dtype=th.float32)
        policy_id_th = th.from_numpy(policy_id_np).to(self.device, dtype=th.int64)
        current_obs_rearranged = rearrange(current_obs_th, 'b h w c -> b c h w')
        
        # 1. Convert observations to symbolic OvercookedState objects
        unnorm_obs_batch = self._process_diffusion_trajectory(current_obs_norm_np[:, np.newaxis, ...]).squeeze(1)
        states_t_batch = self.idm.invert_obs_to_state_batch(unnorm_obs_batch)
        
        candidate_trajectories = []
        for i in range(self.num_candidates):
            batch_action_plans = []
            # 2. For each state, generate a plan based on the policy
            for b_idx in range(batch_size):
                state_t = states_t_batch[b_idx]
                p0 = state_t.players[0]
                p0_start_state = p0.pos_and_or
                p0_goal_state = None
                
                # ===== START: "FULL DELIVERY CYCLE" POLICY LOGIC =====
                pot_states = self.mdp.get_pot_states(state_t)

                # Priority 1: Holding a soup? -> Serve it.
                if p0.has_object() and p0.get_object().name == 'soup':
                    target_locations = self.mdp.get_serving_locations()
                
                # Priority 2: Is a soup ready in a pot? -> Go get a dish.
                elif pot_states.get('ready'):
                    target_locations = self.mdp.get_dish_dispenser_locations()

                # Priority 3: Holding a dish? -> Go get the soup.
                elif p0.has_object() and p0.get_object().name == 'dish':
                    # We need to find the location of a ready pot
                    target_locations = pot_states['ready']
                
                # Priority 4: Holding an onion? -> Put it in a pot.
                elif p0.has_object() and p0.get_object().name == 'onion':
                    # Target any pot that is not full
                    partially_full_pots = self.mdp.get_partially_full_pots(pot_states)
                    target_locations = pot_states['empty'] + partially_full_pots
                
                # Priority 5: Is a pot full but not cooking? -> Start it.
                elif pot_states.get('full'):
                    target_locations = pot_states['full']

                # Priority 6 (Default): None of the above? -> Start a new cycle by getting an onion.
                else:
                    target_locations = self.mdp.get_onion_dispenser_locations()

                # Find a valid, reachable goal for Player 0 from the target locations
                if target_locations:
                    possible_goals = self.mlam._get_ml_actions_for_positions(target_locations)
                    for goal in possible_goals:
                        if self.jmp.motion_planner.is_valid_motion_start_goal_pair(p0_start_state, goal):
                            p0_goal_state = goal
                            break
                # ===== END: POLICY LOGIC =====

                # ===== DECOUPLED PLAN GENERATION =====
                is_p0_waiting = (p0_goal_state is None) or (p0_start_state == p0_goal_state)

                if is_p0_waiting:
                    p0_action_plan_strings = [Action.STAY]
                else:
                    p0_action_plan, _, _ = self.jmp.motion_planner.get_plan(p0_start_state, p0_goal_state)
                    p0_action_plan_strings = p0_action_plan
                
                p0_action_indices = [Action.ACTION_TO_INDEX[action] for action in p0_action_plan_strings]

                # Pad or truncate the plan
                if len(p0_action_indices) < self.action_horizon:
                    padding = [Action.ACTION_TO_INDEX[Action.STAY]] * (self.action_horizon - len(p0_action_indices))
                    p0_action_indices.extend(padding)
                else:
                    p0_action_indices = p0_action_indices[:self.action_horizon]
                
                batch_action_plans.append(p0_action_indices)

            # 3. Create the action conditioning tensor
            action_condition = th.tensor(batch_action_plans, dtype=th.long, device=self.device)

            # 4. Sample a trajectory from the world model
            trajectory_th = self.world_model.sample(
                x_cond=current_obs_rearranged,
                action_embed=action_condition,
                batch_size=batch_size,
                task_embed=policy_id_th
            )
            trajectory_th = rearrange(trajectory_th, "b (f c) h w -> b f h w c", c=self.C, f=self.horizon)
            
            # --- Debugging ---
            if self.debug:
                trajectory_print = rearrange(trajectory_th, "b f h w c -> b f w h c")
                output_dir = self.debug_dir / f"candidate_{i}"
                output_dir.mkdir(parents=True, exist_ok=True)
                for b_idx in range(batch_size):
                    self.renderer.render_trajectory_video(
                        trajectory_print[b_idx].cpu().numpy(),
                        self.grid,
                        video_path=osp.join(output_dir, f"trajectory_{b_idx}.mp4"),
                        normalize=True,
                    )

            trajectory_processed = self._process_diffusion_trajectory(trajectory_th.cpu().numpy())
            candidate_trajectories.append(trajectory_processed)
            
        return candidate_trajectories

    def _generate_trajectory_candidates(self, current_obs_norm_np, policy_id_np):
        """
        Generates candidate trajectories using a FIXED motion plan for PLAYER 0,
        and correctly decouples planning when one agent is waiting.
        """
        batch_size = current_obs_norm_np.shape[0]

        # --- Prepare inputs for models and planner ---
        current_obs_th = th.from_numpy(current_obs_norm_np).to(self.device, dtype=th.float32)
        policy_id_th = th.from_numpy(policy_id_np).to(self.device, dtype=th.int64)
        current_obs_rearranged = rearrange(current_obs_th, 'b h w c -> b c h w')
        
        # 1. Convert observations to symbolic OvercookedState objects
        unnorm_obs_batch = self._process_diffusion_trajectory(current_obs_norm_np[:, np.newaxis, ...]).squeeze(1)
        states_t_batch = self.idm.invert_obs_to_state_batch(unnorm_obs_batch)
        
        candidate_trajectories = []
        for i in range(self.num_candidates):
            batch_action_plans = []
            # 2. For each state, generate the fixed motion plan
            for b_idx in range(batch_size):
                state_t = states_t_batch[b_idx]
                
                # ===== START: HARDCODED POLICY FOR PLAYER 0 =====
                onion_dispenser_locations = self.mdp.get_onion_dispenser_locations()
                onion_goals = self.mlam._get_ml_actions_for_positions(onion_dispenser_locations)
                
                p0_start_state = state_t.players[0].pos_and_or
                p0_goal_state = None
                
                for goal in onion_goals:
                    if self.jmp.motion_planner.is_valid_motion_start_goal_pair(p0_start_state, goal):
                        p0_goal_state = goal
                        break
                # ===== END: HARDCODED POLICY FOR PLAYER 0 =====

                # ===== START: DECOUPLED PLAN GENERATION =====
                # Determine if Player 0 is already at its goal or has no goal.
                is_p0_waiting = (p0_goal_state is None) or (p0_start_state == p0_goal_state)

                if is_p0_waiting:
                    # Case 1: Player 0 is waiting. Player 1 also waits. The plan is simple.
                    p0_action_plan_strings = [Action.STAY]
                else:
                    # Case 2: Player 0 has a valid move. Plan for it using the single-agent planner.
                    p0_action_plan, _, _ = self.jmp.motion_planner.get_plan(p0_start_state, p0_goal_state)
                    p0_action_plan_strings = p0_action_plan

                # We only care about Player 0's actions for the conditioning
                p0_action_indices = [Action.ACTION_TO_INDEX[action] for action in p0_action_plan_strings]

                # Pad or truncate the plan to match the required action_horizon
                if len(p0_action_indices) < self.action_horizon:
                    padding = [Action.ACTION_TO_INDEX[Action.STAY]] * (self.action_horizon - len(p0_action_indices))
                    p0_action_indices.extend(padding)
                else:
                    p0_action_indices = p0_action_indices[:self.action_horizon]
                
                batch_action_plans.append(p0_action_indices)
                # ===== END: DECOUPLED PLAN GENERATION =====

            # 3. Create the action conditioning tensor
            action_condition = th.tensor(batch_action_plans, dtype=th.long, device=self.device)

            # 4. Sample a trajectory from the world model
            trajectory_th = self.world_model.sample(
                x_cond=current_obs_rearranged,
                action_embed=action_condition,
                batch_size=batch_size,
                task_embed=policy_id_th
            )
            trajectory_th = rearrange(trajectory_th, "b (f c) h w -> b f h w c", c=self.C, f=self.horizon)
            
            if self.debug:
                trajectory_print = rearrange(trajectory_th, "b f h w c -> b f w h c")
                output_dir = self.debug_dir / f"candidate_{i}"
                output_dir.mkdir(parents=True, exist_ok=True)
                for b_idx in range(batch_size):
                    self.renderer.render_trajectory_video(
                        trajectory_print[b_idx].cpu().numpy(),
                        self.grid,
                        video_path=osp.join(output_dir, f"trajectory_{b_idx}.mp4"),
                        normalize=True,
                    )

            trajectory_processed = self._process_diffusion_trajectory(trajectory_th.cpu().numpy())
            candidate_trajectories.append(trajectory_processed)
            
        return candidate_trajectories
    def _rank_trajectories(self, trajectories_np):
        """Ranks trajectories in parallel. Operates on a list of NumPy arrays."""
        N = len(trajectories_np)
        B, T, H, W, C = trajectories_np[0].shape
        
        work_items = [
            (self.args, trajectories_np[n_idx][b_idx])
            for n_idx in range(N) for b_idx in range(B)
        ]
        num_processes = min(B * N, self.num_processes)
        ctx = multiprocessing.get_context('fork') 
        with ctx.Pool(processes=num_processes) as pool:
            all_rewards = pool.map(_calculate_trajectory_reward_worker, work_items)
        
        total_rewards = np.array(all_rewards).reshape(N, B).T  # Reshape to [B, N]
        assert total_rewards.shape == (B, N), f"Bad reward shape {total_rewards.shape}"

        ranked_candidates = []
        for i in range(B):
            # Use np.argsort for sorting
            sorted_indices = np.argsort(total_rewards[i])[::-1]  # Descending order
            env_list = [
                (total_rewards[i, idx], trajectories_np[idx][i]) 
                for idx in sorted_indices
            ]
            ranked_candidates.append(env_list)
        assert len(ranked_candidates) == B

        print(f"Ranked candidates for {B} batches:")
        for i, env_list in enumerate(ranked_candidates):
            print(f"Batch {i}:")
            for j, (reward, traj) in enumerate(env_list):
                print(f"  Rank {j+1}: Reward = {reward:.4f}, Trajectory shape = {traj.shape}")
        
        return ranked_candidates
    # def setup_concept_learning(self, initial_embedding, guidance_weight):
    #     """
    #     Prepares the agent to learn concepts by setting the embedding matrix,
    #     freezing old weights, and creating a new optimizer.
    #     """
    #     print("Setting up agent for concept learning...")
        
    #     # --- Embedding and Guidance Setup ---
    #     # The agent receives the full embedding matrix to work with
    #     W = initial_embedding.clone().detach().to(self.device)
    #     new_embedding = th.nn.Embedding.from_pretrained(W, freeze=False).to(self.device)
        
    #     # Replace embedding layers
    #     unwrapped_model = self.accelerator.unwrap_model(self.world_model)
    #     unwrapped_ema_model = self.accelerator.unwrap_model(self.world_model_ema)
    #     unwrapped_model.model.unet.label_emb = new_embedding
    #     unwrapped_ema_model.model.unet.label_emb = new_embedding.clone()

    #     # --- Parameter Freezing ---
    #     # Freeze all parameters first
    #     for param in unwrapped_model.parameters():
    #         param.requires_grad = False
    #     # Then, unfreeze only the new embedding layer
    #     unwrapped_model.model.unet.label_emb.weight.requires_grad = True
        
    #     # --- Optimizer Setup ---
    #     params_to_optimize = [p for p in self.world_model.parameters() if p.requires_grad]
    #     self.optimizer = Adam(params_to_optimize, lr=self.args.train_lr)
    #     self.optimizer = self.accelerator.prepare(self.optimizer)
        
    #     print("Agent ready to learn concepts.")

    # def learn_concept_on_batch(self, obs_trajectories_np, initial_conditions_np, 
    #                           new_concept_id, dummy_id):
    #     """
    #     Performs a single training step to learn a concept from a batch of experience.
        
    #     Args:
    #         obs_trajectories_np (np.ndarray): Shape [B, T, H, W, C].
    #         initial_conditions_np (np.ndarray): Shape [B, H, W, C].
    #         new_concept_id (int): The class ID for the concept being learned.
    #         dummy_id (int): The class ID for the unconditional (dummy) case.

    #     Returns:
    #         float: The loss for this training step.
    #     """
    #     # Convert data to PyTorch tensors
    #     x_start = th.from_numpy(obs_trajectories_np).to(self.device, dtype=th.float32)
    #     x_cond = th.from_numpy(initial_conditions_np).to(self.device, dtype=th.float32)
        
    #     total_loss = 0.0
    #     # This loop is for gradient accumulation
    #     with self.accelerator.autocast():
    #         loss = self._calculate_concept_loss(x_start, x_cond, new_concept_id, dummy_id)
    #         # You might need gradient accumulation here, for simplicity I'll omit it
    #         # loss = loss / gradient_accumulate_every
    #         total_loss += loss.item()
            
    #         self.accelerator.backward(loss)

    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    #     self.world_model_ema.update() # Update the Exponential Moving Average model

    #     return total_loss

    # def _calculate_concept_loss(self, x_start, x_cond, concept_id, dummy_id):
    #     """Internal loss calculation logic for concept learning."""
    #     batch_size = x_start.shape[0]

    #     # Reshape for diffusion model
    #     x_start = rearrange(x_start, "b f h w c -> b (f c) h w")
    #     x_cond  = rearrange(x_cond, "b h w c -> b c h w")
        
    #     t = th.randint(0, self.world_model.num_timesteps, (batch_size,), device=self.device).long()
        
    #     # Use the unwrapped model for the forward pass
    #     unwrapped_model = self.accelerator.unwrap_model(self.world_model)
        
    #     # Calculate unconditional loss (conditioned on dummy_id)
    #     uncond_loss = unwrapped_model.p_losses(
    #         x_start, t, x_cond=x_cond, 
    #         cond_class=th.full((batch_size,), dummy_id, device=self.device, dtype=th.long)
    #     )
        
    #     # Calculate conditional loss (conditioned on the new concept)
    #     cond_loss = unwrapped_model.p_losses(
    #         x_start, t, x_cond=x_cond,
    #         cond_class=th.full((batch_size,), concept_id, device=self.device, dtype=th.long)
    #     )
        
    #     # Simple Classifier-Free Guidance style loss
    #     guidance_weight = self.args.guidance_weight # Assuming a fixed guidance for training
    #     total_loss = uncond_loss + guidance_weight * (cond_loss - uncond_loss)
        
    #     return total_loss

    # def get_learned_embedding(self):
    #     """Returns the trained embedding from the stable EMA model."""
    #     unwrapped_ema_model = self.accelerator.unwrap_model(self.world_model_ema)
    #     return unwrapped_ema_model.model.unet.label_emb.weight.clone().detach()