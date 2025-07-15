import numpy as np
from mapbt.envs.overcooked.Overcooked_Env import Overcooked
from collections import Counter
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedState, PlayerState, ObjectState, SoupState,
    Direction, Action )

class GroundTruthInverseDynamics:
    """
    Uses the true environment simulator to find the ground-truth actions
    that connect two consecutive states. This serves as a perfect validation
    tool for a learned world model.
    """
    def __init__(self, args):
        """
        Initializes the inverse dynamics model.
        Args:
            args: An argparse namespace containing at least 'layout_name' and
                  other parameters needed by the Overcooked environment.
        """
        self.args = args
        # We need a dummy environment to get access to the true MDP logic
        self.env = Overcooked(args, run_dir="inverse_dynamics_env")
        self.base_mdp = self.env.base_env.mdp
        
        # This will be used to create dummy states if needed
        self.start_orders = self.env.base_env.state.all_orders

        # Generate all 36 possible joint actions for the brute-force search
        self.all_joint_actions = [(a0, a1) for a0 in Action.ALL_ACTIONS for a1 in Action.ALL_ACTIONS]

    def invert_obs_to_state(self, obs_array, max_steps=400):
        """
        Converts a single observation numpy array back into a symbolic OvercookedState.
        This version corrects the mismatch between observation shape (H, W) and
        the MDP's expected geometry (W, H) by transposing the spatial axes first.
        """
        assert obs_array.min() >= 0 and obs_array.max() <= 20 \
            and obs_array.shape == (8, 5, 26), \
            f"Expected observation shape (8, 5, 26) with values in [0, 20], got {obs_array.shape} with min {obs_array.min()} and max {obs_array.max()}"
        transposed_obs = np.transpose(obs_array, (1, 0, 2))
        width, height, _ = transposed_obs.shape # Now width=5, height=8

        # --- Player States ---
        players = []
        for i in range(2): # For player 0 and player 1
            # Player location is a one-hot mask
            p_loc_layer = transposed_obs[:, :, i]
            if not np.any(p_loc_layer): continue

            p_pos_xy = np.unravel_index(np.argmax(p_loc_layer), p_loc_layer.shape)

            orientation_idx = np.argmax(transposed_obs[p_pos_xy[0], p_pos_xy[1], 2 + i*4 : 6 + i*4])
            p_orientation = Direction.INDEX_TO_DIRECTION[orientation_idx]

            p_held_obj = None
            held_item_layer = transposed_obs[p_pos_xy[0], p_pos_xy[1], 21:25]
            if np.any(held_item_layer):
                held_obj_idx = np.argmax(held_item_layer)
                obj_name_map = {0: 'soup', 1: 'dish', 2: 'onion', 3: 'tomato'}
                obj_name = obj_name_map.get(held_obj_idx)
                
                if obj_name == 'soup':
                    ing_dict = Counter()
                    ing_dict['onion'] = int(transposed_obs[p_pos_xy[0], p_pos_xy[1], 18])
                    ing_dict['tomato'] = int(transposed_obs[p_pos_xy[0], p_pos_xy[1], 19])
                    print(f"Player {i} is holding a soup with num onions: {ing_dict['onion']}, tomatoes: {ing_dict['tomato']}")
                    p_held_obj = SoupState.get_soup((p_pos_xy[1], p_pos_xy[0]), num_onions=ing_dict['onion'], num_tomatoes=ing_dict['tomato'], finished=True)
                elif obj_name:
                    p_held_obj = ObjectState(obj_name, (p_pos_xy[1], p_pos_xy[0]))
            players.append(PlayerState((p_pos_xy[1], p_pos_xy[0]), p_orientation, p_held_obj))
        
        # --- World Objects ---
        objects = {}
        
        # Iterate over every cell in the (now correctly shaped) observation grid.
        for x in range(width):
            for y in range(height):
                pos_xy = (y, x)
                
                # Skip locations occupied by players (held items are handled above)
                if any(p.position == pos_xy for p in players):
                    continue
                
                # Is there a pot at this location? (Channel 10)
                if transposed_obs[x, y, 10] > 0:
                    if transposed_obs[x, y, 21] > 0:  # soup_done
                        num_onions = int(transposed_obs[x, y, 18])
                        num_tomatoes = int(transposed_obs[x, y, 19])
                        objects[pos_xy] = SoupState.get_soup(pos_xy, num_onions=num_onions, num_tomatoes=num_tomatoes, finished=True)
                    elif transposed_obs[x, y, 20] > 0: # cooking
                        num_onions = int(transposed_obs[x, y, 18])
                        num_tomatoes = int(transposed_obs[x, y, 19])
                        if num_onions > 0 or num_tomatoes > 0:
                            cook_time_rem = int(transposed_obs[x, y, 20])
                            cooking_tick =  20 - cook_time_rem # TODO: Maybe do max(1, 20 - cook_time_rem) to avoid zero?
                            objects[pos_xy] = SoupState.get_soup(
                                pos_xy, 
                                num_onions=num_onions, 
                                num_tomatoes=num_tomatoes, 
                                cooking_tick=cooking_tick
                            )
                        else:
                            # The model predicted a cooking pot with no ingredients.
                            objects[pos_xy] = SoupState.get_soup(pos_xy, num_onions=0, num_tomatoes=0)
                    else:  # Idling pot
                        num_onions = int(transposed_obs[x, y, 16])
                        num_tomatoes = int(transposed_obs[x, y, 17])
                        if num_onions > 0 or num_tomatoes > 0:
                            objects[pos_xy] = SoupState.get_soup(pos_xy, num_onions=num_onions, num_tomatoes=num_tomatoes)
                    continue
                
                # Check for other items on this tile (e.g., on a counter)
                if transposed_obs[x, y, 23] > 0:      # onions layer
                    objects[pos_xy] = ObjectState('onion', pos_xy)
                elif transposed_obs[x, y, 24] > 0:    # tomatoes layer
                    objects[pos_xy] = ObjectState('tomato', pos_xy)
                elif transposed_obs[x, y, 22] > 0:    # dishes layer
                    objects[pos_xy] = ObjectState('dish', pos_xy)
                elif transposed_obs[x, y, 21] > 0:  # soup layer
                    ing_dict = Counter()
                    ing_dict['onion'] = int(transposed_obs[x, y, 18]) # onions_in_soup
                    ing_dict['tomato'] = int(transposed_obs[x, y, 19]) # tomatoes_in_soup
                    objects[pos_xy] = SoupState.get_soup(pos_xy, num_onions=ing_dict['onion'], num_tomatoes=ing_dict['tomato'], finished=True)
                
        # Final reconstructed state with positions that are valid for the MDP's geometry
        urgency_layer = transposed_obs[:, :, 25]
        is_urgent = np.any(urgency_layer > 0)
        estimated_timestep = int(np.where(is_urgent, max_steps - 20, max_steps // 2))
        
        # Get order information from the class's own environment instance
        bonus_orders = self.env.base_env.state.bonus_orders
        all_orders = self.env.base_env.state.all_orders
        
        # Create and return the final, complete OvercookedState object
        state =  OvercookedState(
            players=players,
            objects=objects,
            bonus_orders=[o.to_dict() for o in bonus_orders],
            all_orders=[o.to_dict() for o in all_orders],
            timestep=estimated_timestep
        )
        return state
    
    def invert_obs_to_state_batch(self, obs_array_batch, max_steps=400):
        """
        Converts a batch of observation numpy arrays back into symbolic OvercookedStates.
        This version corrects the mismatch between observation shape (H, W) and
        the MDP's expected geometry (W, H) by transposing the spatial axes first.
        """
        return [self.invert_obs_to_state(obs_array) for obs_array in obs_array_batch]
    
    def _is_state_valid_for_simulation(self, state):
        """
        A helper function to check if a state is physically valid before passing
        it to the core MDP simulator.
        """
        # Check for the correct number of players
        if len(state.players) != 2:
            print(f"WARNING: Invalid state detected ({len(state.players)} players).")
            return False
            
        # Check for illegally placed objects on empty floor space
        for obj in state.objects.values():
            terrain_type = self.base_mdp.get_terrain_type_at_pos(obj.position)
            if terrain_type == ' ':
                print(f"WARNING: Invalid state detected (object '{obj.name}' at floor space {obj.position}).")
                return False
        
        # Check for overlapping entities (players or objects)     
        all_pos = [p.position for p in state.players] + list(state.objects.keys())
        if len(all_pos) != len(set(all_pos)):
            # To find the duplicate for better logging (optional but helpful)
            from collections import Counter
            pos_counts = Counter(all_pos)
            duplicates = [pos for pos, count in pos_counts.items() if count > 1]
            print(f"VALIDATION FAIL: Overlapping entities detected at position(s): {duplicates}")
            return False
        
        return True
    
    def _find_best_actions_core(self, states_t_batch, states_tp1_batch, distance_fn):
        """
        Core private method to find the best action for a batch of transitions
        using a provided distance function.
        """
        best_actions = []
        for state_t, state_tp1 in zip(states_t_batch, states_tp1_batch):
            
            # This is the central loop: simulate all actions and calculate distance
            if not self._is_state_valid_for_simulation(state_t):
                return None  # Invalid state, cannot proceed
                

            all_distances = np.array([
                distance_fn(self.base_mdp.get_state_transition(state_t, ja)[0], state_tp1)
                for ja in self.all_joint_actions
            ])
            
            # Find the action(s) with the minimum distance
            min_distance = all_distances.min()
            best_action_indices = np.flatnonzero(all_distances == min_distance)
            
            # Randomly choose one of the best actions to break ties
            chosen_index = np.random.choice(best_action_indices)
            best_actions.append(self.all_joint_actions[chosen_index])
            
        return best_actions
    
    def find_actions_between_states_batch(self, states_t_batch, states_tp1_batch, return_indices=False):
        """Finds the best JOINT action for a batch, using the full state distance."""
        best_joint_actions_raw = self._find_best_actions_core(
            states_t_batch,
            states_tp1_batch,
            distance_fn=self._calculate_full_state_distance
        )
        if return_indices:
            action_indices_batch = [
                (Action.ACTION_TO_INDEX[joint_action[0]], Action.ACTION_TO_INDEX[joint_action[1]])
                for joint_action in best_joint_actions_raw
            ]
            return action_indices_batch
        return best_joint_actions_raw
    def find_ego_actions_batch(self, states_t_batch, states_tp1_batch, ego_player_idx=0, return_indices=False):
        """Finds the best EGO action for a batch, using an ego-centric distance."""
        # Note: The core search still finds a joint action. We extract the ego part.
        best_joint_actions = self._find_best_actions_core(
            states_t_batch,
            states_tp1_batch,
            distance_fn=lambda s1, s2: self._calculate_ego_state_distance(s1, s2, ego_player_idx)
        )
        ego_actions = [joint_action[ego_player_idx] for joint_action in best_joint_actions]
        if return_indices:
            action_indices = [Action.ACTION_TO_INDEX[action] for action in ego_actions]
            return action_indices
        return ego_actions
    def _calculate_full_state_distance(self, state1, state2):
        """Calculates distance based on all players and all objects."""
        # This reuses the ego-centric logic for both players
        distance = self._calculate_ego_state_distance(state1, state2, ego_player_idx=0)
        distance += self._calculate_ego_state_distance(state1, state2, ego_player_idx=1)
        return distance
    def _calculate_ego_state_distance(self, state1, state2, ego_player_idx=0):
        """Calculates distance from the perspective of a single agent."""
        distance = 0.0
        other_player_idx = 1 - ego_player_idx

        # Compare ego-player states
        p1_ego = state1.players[ego_player_idx] if len(state1.players) > ego_player_idx else None
        p2_ego = state2.players[ego_player_idx] if len(state2.players) > ego_player_idx else None

        if p1_ego is None and p2_ego is None:
            pass # No player to compare, so no distance is added.
        elif p1_ego is None or p2_ego is None:
            # A player appearing or disappearing is a huge error. Assign a large penalty.
            distance += 20.0
        else:
            # Both players exist, compare their attributes
            if p1_ego.position != p2_ego.position: distance += 10.0
            if p1_ego.orientation != p2_ego.orientation: distance += 2.0
            if (p1_ego.held_object is None) != (p2_ego.held_object is None): distance += 5.0
            elif p1_ego.held_object and p2_ego.held_object:
                if p1_ego.held_object.name != p2_ego.held_object.name: distance += 5.0
                elif isinstance(p1_ego.held_object, SoupState) and p1_ego.held_object.ingredients != p2_ego.held_object.ingredients: distance += 8.0

        # Compare world objects (ignoring objects at the *other* player's location)
        all_pos = set(state1.objects.keys()) | set(state2.objects.keys())
        other_player1_pos = state1.players[other_player_idx].position if len(state1.players) > other_player_idx else None
        other_player2_pos = state2.players[other_player_idx].position if len(state2.players) > other_player_idx else None
        other_player_pos = {pos for pos in [other_player1_pos, other_player2_pos] if pos is not None}

        for pos in all_pos:
            if pos in other_player_pos: continue

            obj1, obj2 = state1.objects.get(pos), state2.objects.get(pos)
            if (obj1 is None) != (obj2 is None) or type(obj1) != type(obj2):
                distance += 5.0
                continue
            if obj1 is None: continue

            if isinstance(obj1, SoupState):
                if obj1.ingredients != obj2.ingredients: distance += 5.0
                # TODO: We can check cook time here; seems like the model can predict it nicely
            elif obj1 != obj2:
                distance += 5.0
                
        return distance
    
    def fuzzy_state_equal_for_player(self, state1, state2, player_idx, compare_world_objects=True):
        """Checks for perfect equality from an ego-centric perspective."""
        # This can be simplified by leveraging the distance function!
        # If the distance is zero, they are equal.
        if not compare_world_objects:
            p1_ego = state1.players[player_idx]
            p2_ego = state2.players[player_idx]
            return p1_ego == p2_ego
        
        return self._calculate_ego_state_distance(state1, state2, player_idx) == 0
    
    def find_action_to_reach_next_state(self, current_state, target_next_state):
        """Convenience wrapper for finding a single joint action."""
        return self.find_actions_between_states_batch([current_state], [target_next_state])[0]
    
    def find_ego_action_to_reach_next_state(self, current_state, target_next_state, ego_player_idx=0):
        """Convenience wrapper for finding a single ego action."""
        return self.find_ego_actions_batch([current_state], [target_next_state], ego_player_idx)[0]