from collections import deque
from einops import rearrange, repeat
import numpy as np
import torch
from overcooked.utils.utils import normalize_obs_vectorized

class ActionProposalUtil:
    """
    A stateless, batched utility for proposing action sequences using a
    reward-conditioned diffusion model.

    This class is designed to be used by a master agent that manages the
    observation history for one or more parallel environments.
    """
    def __init__(
        self,
        model,
        horizon=32,
        num_actions=6,
        history_horizon=16,
        device=None
    ):
        """
        Initializes the action proposal utility.

        Args:
            model: The trained UnetOvercookedActionProposal nn.Module.
            args: A config object containing model and environment parameters.
            device (str, optional): The device to run the model on.
        """
        self.horizon = horizon
        self.num_actions = num_actions
        self.model = model.to(device).eval()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history_horizon = history_horizon
        print(f"ActionProposalUtil operating on device: {self.device}")

    def _prepare_inputs(self, current_obs_batch_np, history_batch_list):
        """
        Prepares and pads a batch of history lists and current observations.
        
        Args:
            current_obs_batch_np (np.ndarray): Shape [B, H, W, C].
            history_batch_list (list[list[np.ndarray]]): A list of B history lists.

        Returns:
            torch.Tensor: Batched current frames, shape [B, C, H, W].
            torch.Tensor: Batched histories, shape [B, C, F, H, W].
        """
        batch_size = current_obs_batch_np.shape[0]
        
        padded_histories = []
        for i in range(batch_size):
            history_list = history_batch_list[i]
            
            # Pad the history if it's not full yet
            num_missing_frames = self.history_horizon - len(history_list)
            if num_missing_frames > 0:
                pad_block = np.repeat(
                    np.expand_dims(history_list[0], axis=0),
                    repeats=num_missing_frames,
                    axis=0
                )
                obs_history_np = np.concatenate([pad_block, np.array(history_list)], axis=0)
            else:
                obs_history_np = np.array(history_list)
            
            padded_histories.append(obs_history_np)
        
        # Stack the processed histories into a single batch
        history_batch_np = np.stack(padded_histories, axis=0) # Shape: [B, F_hist, H, W, C]
        
        # Normalize both batches
        current_frame_norm = normalize_obs_vectorized(current_obs_batch_np)
        obs_history_norm = normalize_obs_vectorized(history_batch_np)

        # Convert to tensors
        current_frame_th = torch.from_numpy(current_frame_norm).to(self.device, dtype=torch.float32)
        obs_history_th = torch.from_numpy(obs_history_norm).to(self.device, dtype=torch.float32)

        # Permute to channels-first format for the model
        current_frame_th = rearrange(current_frame_th, 'b h w c -> b c h w')
        obs_history_th = rearrange(obs_history_th, 'b f h w c -> b c f h w')

        return current_frame_th, obs_history_th

    @torch.no_grad()
    def get_action_plans(self, current_obs_batch_np, history_batch_list, target_reward, num_candidates):
        """
        The main method. Generates a batch of candidate action plans.

        Args:
            current_obs_batch_np (np.ndarray): Batch of current observations, shape [B, H, W, C].
            history_batch_list (list[list[np.ndarray]]): List of B history lists.
            target_reward (float): The desired reward-to-go to steer the generation.
            num_candidates (int): The number of candidate plans (N) to generate for EACH observation.

        Returns:
            np.ndarray: A batch of proposed action plans, shape [B, N, action_horizon].
        """
        batch_size = current_obs_batch_np.shape[0]
        
        # Prepare all conditioning inputs in the correct batched format
        current_frame_embed, history_embed = self._prepare_inputs(current_obs_batch_np, history_batch_list)
        
        # We want to generate N candidates for each of the B environments.
        total_samples = batch_size * num_candidates
        
        # Repeat each of the B inputs N times
        image_embed = repeat(current_frame_embed, 'b c h w -> (b n) c h w', n=num_candidates)
        history_embed = repeat(history_embed, 'b c f h w -> (b n) c f h w', n=num_candidates)
        reward_embed = torch.full((total_samples, 1), target_reward, device=self.device, dtype=torch.float32)
        
        # Sample a batch of candidate action plans from the diffusion model
        # The model will run on a single large batch of size B*N
        action_proposals_logits = self.model.sample(
            image_embed=image_embed,
            history_embed=history_embed,
            reward_embed=reward_embed,
            batch_size=total_samples
        ) # Expected output shape: [B*N, horizon, 1, 1]

        action_proposals_logits = action_proposals_logits.view(total_samples, self.horizon, self.num_actions)

        # Convert logits to discrete actions
        best_plan_actions = torch.argmax(action_proposals_logits, dim=-1) # Shape: [B*N, horizon]
        
        # Reshape to separate the batch and candidate dimensions
        final_plans = rearrange(best_plan_actions, '(b n) h -> b n h', b=batch_size)
        print(f"Generated action proposals with shape: {final_plans.shape}")

        return final_plans.cpu().numpy()