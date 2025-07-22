import numpy as np
from overcooked.agent.idm.ground_truth_idm import GroundTruthInverseDynamics

class RewardCalculator:
    """
    A class to calculate rewards for state transitions using the true
    Overcooked simulator. It provides both single and batched interfaces.
    """
    def __init__(self, args):
        self.args = args
        self.idm_model = GroundTruthInverseDynamics(args)
        self.mdp = self.idm_model.base_mdp

    def calculate_reward_batch(self, obs_t_batch, obs_tp1_batch, joint_actions_batch=None):
        """
        Calculates rewards for a batch of transitions. This is the primary logic holder.

        Args:
            obs_t_batch (np.ndarray): Batch of observations at time t, shape [B, H, W, C].
            obs_tp1_batch (np.ndarray): Batch of observations at time t+1, shape [B, H, W, C].
            joint_actions_batch (list[tuple], optional): A list of joint actions. If None,
                                                         they will be inferred.

        Returns:
            np.ndarray: An array of ego-agent rewards, shape [B].
            np.ndarray: An array of partner-agent rewards, shape [B].
        """
        batch_size = obs_t_batch.shape[0]
        
        # Convert all observations to OvercookedState objects in a batch
        states_t_batch = self.idm_model.invert_obs_to_state_batch(obs_t_batch)
        
        # If actions are not provided, infer them for the whole batch
        if joint_actions_batch is None:
            states_tp1_batch = self.idm_model.invert_obs_to_state_batch(obs_tp1_batch)
            joint_actions_batch = self.idm_model.find_actions_between_states_batch(states_t_batch, states_tp1_batch)

        ego_rewards = np.zeros(batch_size)
        partner_rewards = np.zeros(batch_size)

        # Ensure joint_actions_batch is not None
        if joint_actions_batch is None:
            # If no actions were found, we cannot compute rewards.
            # Assign a large negative reward to heavily penalize this trajectory.
            ego_rewards.fill(-20.0)
            partner_rewards.fill(-20.0)
            print("WARNING: No joint actions found for the batch. Assigning large negative rewards.")
            return ego_rewards, partner_rewards

        for i in range(batch_size):
            state_t = states_t_batch[i]
            joint_action = joint_actions_batch[i]
            
            if joint_action is None:
                # This case indicates a failure in the IDM to find even a "best-fit" action.
                # Assign a large negative reward to heavily penalize this trajectory.
                ego_rewards[i] = -20.0
                partner_rewards[i] = -20.0
                continue

            # Get the reward info from the true MDP transition
            _, infos = self.mdp.get_state_transition(state_t, joint_action)
            sparse = infos.get('sparse_reward_by_agent', [0, 0])
            shaped = infos.get('shaped_reward_by_agent', [0, 0])
            
            ego_rewards[i] = sparse[0] + shaped[0]
            partner_rewards[i] = sparse[1] + shaped[1]
            
        return ego_rewards, partner_rewards

    def calculate_reward(self, obs_t, obs_t_plus_1, joint_action=None, max_steps=400):
        """
        Calculates the reward for a SINGLE observation transition.
        This is now a convenience wrapper around the batch method.
        """
        # Reshape single inputs into a batch of size 1
        obs_t_batch = np.expand_dims(obs_t, axis=0)
        obs_tp1_batch = np.expand_dims(obs_t_plus_1, axis=0)
        joint_actions_batch = [joint_action] if joint_action is not None else None
        
        # Call the powerful batch method
        ego_rewards_batch, partner_rewards_batch = self.calculate_reward_batch(
            obs_t_batch, obs_tp1_batch, joint_actions_batch
        )
        
        # Return the single result from the batch of size 1
        return ego_rewards_batch[0], partner_rewards_batch[0]