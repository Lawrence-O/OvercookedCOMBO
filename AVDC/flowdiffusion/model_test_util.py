import numpy as np
import xxhash
from collections import defaultdict
import warnings

def analyze_ambiguity_direct(dataset, verbose=False):
    """
    Analyzes a dataset of Overcooked trajectories to determine if the state-to-action
    mapping is one-to-one or one-to-many by directly hashing quantized observations.

    This method respects the model's perspective by not using external logic or
    handcrafted features like 'front_pos'. It processes the (obs_t, obs_tp1) tensors
    as a whole, making it a true diagnostic for the data the model was trained on.

    Args:
        dataset (iterable): An iterable of (obs_t, obs_tp1, action_t) tuples.
                            obs_t and obs_tp1 are numpy arrays of shape (H, W, C).
        verbose (bool): If True, prints details of every ambiguous transition found.

    Returns:
        dict: A dictionary containing analysis results:
              - 'total_transitions': Total number of transitions processed.
              - 'unique_signatures': Number of unique state transitions found.
              - 'ambiguous_signatures': Count of signatures mapping to >1 action.
              - 'ambiguity_percentage': Percentage of unique signatures that are ambiguous.
              - 'ambiguous_action_pairs': A frequency count of ambiguous action pairs.
              - 'transition_map': The full hash map from signature to set of actions.
    """
    # --- Start of self-contained helper logic ---

    # Feature map is needed to know which channels to process in which way
    FEATURE_CHANNEL_MAP = {
        name: i for i, name in enumerate([
            "player_0_loc", "player_1_loc",
            "player_0_orientation_0", "player_0_orientation_1", "player_0_orientation_2", "player_0_orientation_3",
            "player_1_orientation_0", "player_1_orientation_1", "player_1_orientation_2", "player_1_orientation_3",
            "pot_loc", "counter_loc", "onion_disp_loc", "tomato_disp_loc", "dish_disp_loc", "serve_loc",
            "onions_in_pot", "tomatoes_in_pot", "onions_in_soup", "tomatoes_in_soup",
            "soup_cook_time_remaining", "soup_done",
            "dishes", "onions", "tomatoes", "urgency"
        ])
    }
    
    # Define channel types for specialized processing
    COUNT_CHANNELS = {"onions_in_pot", "tomatoes_in_pot", "onions_in_soup", "tomatoes_in_soup"}
    TIMER_CHANNELS = {"soup_cook_time_remaining"}
    SCALING_FACTOR = 255.0

    def quantize_observation(obs: np.ndarray) -> np.ndarray:
        """
        Converts a raw observation tensor into a canonical integer form robust to numerical noise.
        """
        if obs.shape[-1] != len(FEATURE_CHANNEL_MAP):
            raise ValueError(f"Observation has {obs.shape[-1]} channels, but expected {len(FEATURE_CHANNEL_MAP)}")

        canonical_obs = np.zeros_like(obs, dtype=np.int32)
        H, W, C = obs.shape
        
        for c in range(C):
            # Find the channel name for the current index `c`
            channel_name = next((name for name, idx in FEATURE_CHANNEL_MAP.items() if idx == c), None)
            
            if channel_name is None:
                warnings.warn(f"No name found for channel index {c}. Using default rounding.")
                canonical_obs[:, :, c] = np.round(obs[:, :, c]).astype(np.int32)
                continue

            # Apply specialized processing based on channel type
            if channel_name in COUNT_CHANNELS:
                # Un-scale to get semantic integer counts
                canonical_obs[:, :, c] = np.round(obs[:, :, c] / SCALING_FACTOR).astype(np.int32)
            elif channel_name == "soup_cook_time_remaining":
                # Bin timers into discrete states {0: off, 1: cooking, 2: done}
                # This requires looking at the 'soup_done' channel for context.
                done_plane = obs[:, :, FEATURE_CHANNEL_MAP['soup_done']]
                cooking_plane = obs[:, :, c]
                
                binned_plane = np.zeros((H,W), dtype=np.int32)
                binned_plane[done_plane > 0] = 2  # State 2: Done
                binned_plane[(done_plane <= 0) & (cooking_plane > 0)] = 1  # State 1: Cooking
                canonical_obs[:, :, c] = binned_plane
            else:
                # For all other channels (location, orientation, binary flags, etc.),
                # simply round the value. This handles both {0, 1} and {0, 255} cases.
                canonical_obs[:, :, c] = np.round(obs[:, :, c] / SCALING_FACTOR ).astype(np.int32)
                
        return canonical_obs

    def hash_transition_direct(obs_t: np.ndarray, obs_tp1: np.ndarray) -> str:
        """Hashes a transition pair after applying robust quantization."""
        quant_t = quantize_observation(obs_t)
        quant_tp1 = quantize_observation(obs_tp1)
        
        # Concatenate the flattened canonical tensors and convert to bytes for hashing
        transition_bytes = np.concatenate([quant_t.flatten(), quant_tp1.flatten()]).tobytes()
        return xxhash.xxh64(transition_bytes).hexdigest()

    # --- Main analysis logic ---
    transition_map = defaultdict(set)
    total_transitions = 0

    print("Starting DIRECT ambiguity analysis (hashing full quantized state)...")
    for i, data_point in enumerate(dataset):
        try:
            obs_t, obs_tp1, action = data_point
        except (ValueError, TypeError):
            print(f"Error: Could not unpack data point at index {i}. Expected (obs_t, obs_tp1, action). Skipping.")
            continue
            
        key = hash_transition_direct(obs_t, obs_tp1)
        # Ensure action is hashable (e.g., an integer or string, not a list)
        action_val = action.item() if hasattr(action, 'item') else action
        transition_map[key].add(action_val)
        total_transitions += 1

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1} transitions...")
        if (i + 1) == 1000000:
            print("  Stopping early after 100,000 transitions for demonstration purposes.")
            break
    
    if total_transitions > 0:
        print(f"...Analysis complete. Processed a total of {total_transitions} transitions.")
    else:
        print("...Analysis complete. No transitions were processed.")
        return {}


    # --- Reporting ---
    ambiguous_signatures = 0
    ambiguous_action_pairs = defaultdict(int)

    for key, actions in transition_map.items():
        if len(actions) > 1:
            ambiguous_signatures += 1
            # To make the count order-independent, sort actions before creating the pair tuple
            sorted_actions = tuple(sorted(list(actions)))
            ambiguous_action_pairs[sorted_actions] += 1
            if verbose:
                print(f"  Ambiguous Signature Found! Hash: {key}, Actions: {actions}")

    unique_signatures = len(transition_map)
    ambiguity_percentage = (ambiguous_signatures / unique_signatures * 100) if unique_signatures > 0 else 0
    
    # Print Summary Report
    print("\n" + "="*60)
    print("      Direct Dataset Ambiguity Analysis Report")
    print("="*60)
    print(f"Total Transitions Processed:    {total_transitions:,}")
    print(f"Unique State Transitions Found: {unique_signatures:,}")
    print(f"Ambiguous (One-to-Many) Count:  {ambiguous_signatures:,}")
    print(f"Percentage of Ambiguity:        {ambiguity_percentage:.2f}%")
    print("-"*60)
    
    if ambiguous_signatures > 0:
        print("Most Common Ambiguous Action Sets:")
        # Sort the pairs by frequency for a clean report
        sorted_pairs = sorted(ambiguous_action_pairs.items(), key=lambda item: item[1], reverse=True)
        for i, (actions, count) in enumerate(sorted_pairs):
            print(f"  {i+1}. {actions}: {count:,} occurrences")
            if i >= 9: # Print top 10
                break
    else:
        print("No ambiguous action sets found. The mapping is one-to-one.")
    print("="*60 + "\n")

    return {
        'total_transitions': total_transitions,
        'unique_signatures': unique_signatures,
        'ambiguous_signatures': ambiguous_signatures,
        'ambiguity_percentage': ambiguity_percentage,
        'ambiguous_action_pairs': dict(ambiguous_action_pairs),
        'transition_map': dict(transition_map)
    }

# ===================================================================================
# ---                            EXAMPLE USAGE                                    ---
# ===================================================================================
if __name__ == '__main__':
    print("Running demonstration with a carefully crafted dummy dataset...\n")
    
    H, W, C = 8, 5, 26 # Standard dimensions
    
    # --- Define a believable ambiguous scenario ---
    # A pot is filled, but the state change is identical whether player 0 did it
    # or player 1 did it, making player 0's action ambiguous ("interact" vs "stay").

    # -- State t: Player 0 is near a pot, Player 1 is also near the same pot --
    obs_t = np.zeros((H, W, C), dtype=np.float32)
    # Player 0
    obs_t[3, 2, FEATURE_CHANNEL_MAP['player_0_loc']] = 255.0
    # Player 1
    obs_t[4, 3, FEATURE_CHANNEL_MAP['player_1_loc']] = 255.0
    # A single pot
    obs_t[3, 3, FEATURE_CHANNEL_MAP['pot_loc']] = 255.0

    # -- State t+1: The pot now contains one onion. Everything else is identical. --
    obs_tp1 = obs_t.copy()
    obs_tp1[3, 3, FEATURE_CHANNEL_MAP['onions_in_pot']] = 255.0 # pot has 1 onion

    # Define actions (as integers, which is common)
    ACTION_INTERACT = 5
    ACTION_STAY = 4
    ACTION_MOVE_NORTH = 0

    # -- Create the dataset --
    # Case 1: The (obs_t, obs_tp1) transition was caused by Player 0's "interact".
    # Case 2: The exact same (obs_t, obs_tp1) transition occurred, but Player 0's
    #         action was "stay" (implying Player 1 was the one who interacted).
    # Case 3: A totally different, unambiguous transition.
    dummy_dataset = [
        (obs_t, obs_tp1, ACTION_INTERACT),
        (obs_t, obs_tp1, ACTION_STAY),
        (np.random.rand(H,W,C) * 255, np.random.rand(H,W,C) * 255, ACTION_MOVE_NORTH)
    ]

    # --- Run the analysis ---
    # Set verbose=True to see the hash and actions for the ambiguous case.
    results = analyze_ambiguity_direct(dummy_dataset, verbose=True)

    # --- Verify the results ---
    print("\n--- Verifying Demo Results ---")
    try:
        assert results['ambiguous_signatures'] == 1
        assert results['unique_signatures'] == 2
        
        # The ambiguous pair should be (4, 5) sorted
        ambiguous_pair = tuple(sorted([ACTION_STAY, ACTION_INTERACT]))
        assert ambiguous_pair in results['ambiguous_action_pairs']
        assert results['ambiguous_action_pairs'][ambiguous_pair] == 1
        
        print("SUCCESS: Demo analysis correctly identified the ambiguity.")
    except (AssertionError, KeyError) as e:
        print(f"FAILURE: Demo analysis produced unexpected results. Error: {e}")
        print("Full results dictionary:")
        import json
        print(json.dumps(results, indent=2, default=str))