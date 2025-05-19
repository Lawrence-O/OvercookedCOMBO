import sys

mapbt_path = '/home/law/Workspace/repos/COMBO/mapbt_package'
if mapbt_path not in sys.path:
    sys.path.append(mapbt_path)

import numpy as np
import torch as th
import os
import os.path as osp
import pickle
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")
from idm.inverse_dynamics import InverseDynamicsModel
from mapbt_package.mapbt.envs.overcooked.Overcooked_Env import Overcooked
from mapbt_package.mapbt.envs.env_wrappers import ChooseSubprocVecEnv
from mapbt_package.mapbt.algorithms.population.policy_pool import PolicyPool as Policy
from mapbt_package.mapbt.config import get_config
from overcooked_sample_renderer import OvercookedSampleRenderer
from einops.einops import rearrange
from AVDC.flowdiffusion.goal_diffusion import GoalGaussianDiffusion, OvercookedEnvTrainer
from AVDC.flowdiffusion.unet import UnetOvercooked 
from AVDC.flowdiffusion.train_overcooked import OvercookedTrainer

def parse_args(args, parser):
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--results_dir', type=str, default='./overcooked_results', help='Directory to save results and checkpoints')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (smaller dataset, faster training)')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Overcooked HDF5 dataset')
    parser.add_argument('--horizon', type=int, default=32, help='Sequence horizon for trajectories')
    parser.add_argument('--save_milestone', type=bool, default=True, help='Save milestones with step number in filename') # Or action='store_true'

    # For OvercookedSequenceDataset / HDF5Dataset
    parser.add_argument('--max_path_length', type=int, default=401, help='Maximum path length in episodes (for dataset indexing)')
    parser.add_argument('--chunk_length', type=int, default=None, help='Chunk length for HDF5Dataset (defaults to horizon if None, set via dataset_constructor_args)')
    parser.add_argument('--use_padding', type=bool, default=True, help='Whether to use padding for shorter sequences in dataset')


    # For GoalGaussianDiffusion (configurable ones)
    parser.add_argument('--timesteps', type=int, default=400, help='Number of diffusion timesteps for training (if not debug)')
    parser.add_argument('--sampling_timesteps', type=int, default=10, help='Number of timesteps for DDIM sampling (if not debug)')

    # For OvercookedEnvTrainer 
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size (if not debug)')
    parser.add_argument('--num_validation_samples', type=int, default=4, help='Number of samples to generate during validation step')
    parser.add_argument('--save_and_sample_every', type=int, default=1000, help='Frequency to save checkpoints and generate samples (if not debug)')
    parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Probability of dropping condition for CFG during training')
    parser.add_argument('--split_batches', type=bool, default=True, help='Whether to split batches for Accelerator')
    
    # overcooked evaluation
    parser.add_argument("--agent0_policy_name", type=str, help="policy name of agent 0")
    parser.add_argument("--agent1_policy_name", type=str, help="policy name of agent 1")

    parser.add_argument("--diffusion_loadpath", type=str, required=True, 
                      help="Path to the diffusion model directory")
    parser.add_argument("--loadbase", type=str, default="logs",
                      help="Base directory for loading models")
    parser.add_argument("--dataset", type=str, default="overcooked",
                      help="Dataset name")
    parser.add_argument("--n_envs", type=int, default=3,
                      help="Number of parallel environments")
    parser.add_argument("--agent_id", type=int, default=5,
                      help="Agent ID for conditioning")
    parser.add_argument("--max_steps", type=int, default=400,
                      help="Maximum steps per episode")
    parser.add_argument("--run_dir", type=str, default="eval_run",
                      help="Directory for evaluation run")
    parser.add_argument("--idm_loadpath", type=str, required=True, 
                      help="Path to the diffusion model directory")
    parser.add_argument("--diffusion_milestone", type=str, required=True,
                        help="Diffusion Milestone")
    # From Eval    
    parser.add_argument("--old_dynamics", default=False, action='store_true', help="old_dynamics in mdp")
    parser.add_argument("--layout_name", type=str, default='counter_circuit_o_1order', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")  
    parser.add_argument("--use_hsp", default=False, action='store_true')   
    parser.add_argument("--random_index", default=False, action='store_true')
    parser.add_argument("--use_agent_policy_id", default=False, action='store_true', help="Add policy id into share obs, default False")
    parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
    parser.add_argument("--use_detailed_rew_shaping", default=False, action='store_true')
    parser.add_argument("--random_start_prob", default=0., type=float)
    parser.add_argument("--store_traj", default=False, action='store_true')
    # population
    parser.add_argument("--population_yaml_path", type=str, help="Path to yaml file that stores the population info.")
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Fraction of data to use for validation (default: 0.1)')
    all_args = parser.parse_known_args(args)[0]

    return all_args

def get_idm_action(current_obs, next_obs, idm_model):
    current_obs = preprocess_for_idm(current_obs)
    next_obs = preprocess_for_idm(next_obs)
    # render = OvercookedSampleRenderer()
    # render.visualize_all_channels(to_np(current_obs[0]), "./curr_obs.png")
    # render.visualize_all_channels(to_np(next_obs[0]), "./next_obs.png")

    assert th.all((current_obs == -1) | (current_obs == 1)), "obs contains values other than -1 or 1"
    assert th.all((next_obs == -1) | (next_obs == 1)), "obs contains values other than -1 or 1"

    with th.no_grad():
        logits = idm_model(current_obs, next_obs)
        probs = F.softmax(logits, dim=1)
        action = th.argmax(probs)
    return action

def make_eval_env(all_args, run_dir, nenvs=3):
    def get_env_fn(rank):
        def init_env():
            env = Overcooked(all_args, run_dir, rank=rank)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(nenvs)])

def get_agent(population_yaml_path, policy_name, device):
    policy = Policy(None, None, None, None, device=device)
    featurize_type = policy.load_population(population_yaml_path, evaluation=True)
    policy = policy.policy_pool[policy_name]
    feat_type = featurize_type.get(policy_name, 'ppo')
    return policy, feat_type

def to_np(x):
	if th.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def to_torch(x, dtype=None, device=None):
    DTYPE = th.float
    DEVICE = 'cuda:0'
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif th.is_tensor(x):
        return x.to(device).type(dtype)
    # elif x.dtype.type is np.str_:
    # 	return torch.tensor(x, device=device)
    return th.tensor(x, dtype=dtype, device=device)


def arg_max(obs):
    # Assume Obs -> [Batch, Horizon, H, W, C]
    if obs.dim() != 5:
        raise ValueError(f"Expected 5D input (B, T, H, W, C), got {obs.shape}")
    
    B, T, H, W, C = obs.shape

    # Flatten spatial dimensions [B, T, C, H*W]
    flat = rearrange(obs, "b t h w c -> b t c (h w)")

    # Get max indices along spatial dimension
    _, max_idxs = flat.max(dim=-1)  # [B, T, C]

    # Directly scatter 1.0 at max positions
    flat_mask = th.ones_like(flat)*-1
    flat_mask.scatter_(-1, max_idxs.unsqueeze(-1), 1.0)  
    # Reshape back to original dimensions
    peaks = rearrange(flat_mask, "b t c (h w) -> b t h w c", h=H, w=W)

    return peaks

def normalize_obs(obs):
    obs_max = obs.max(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
    obs_min = obs.min(axis=(0, 1, 2), keepdims=True)  # shape [1, 1, 1, C]
    obs_norm = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-8) - 1
    return obs_norm.astype(np.float32)

def preprocess_for_idm(obs):
    if isinstance(obs, np.ndarray):
        # For numpy arrays
        preprocessed_obs = np.where(obs < 0, -1.0, 1.0)
    elif isinstance(obs, th.Tensor):
        # For PyTorch tensors
        preprocessed_obs = th.where(obs < 0, -1.0, 1.0)
        
    return preprocessed_obs


@th.no_grad()
def full_horizon_eval(args, diffusion, idm, policy, device, max_horizon=None, show_samples=False, eval_episodes=3, basedir="./eval_folder"):
    print(f"Starting Overcooked Evaluation; BaseDir {basedir}")
    video_dir = osp.join(basedir, "videos")
    frames_dir = osp.join(basedir, "frames")
    metrics_dir = osp.join(basedir, "metrics")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    renderer = OvercookedSampleRenderer()

    n_envs = args.n_envs if hasattr(args, 'n_envs') else 3
    envs = make_eval_env(args, run_dir=args.run_dir, nenvs=n_envs)
    
    all_metrics = []
    episode_rewards = []
    # agent_id = args.agent_id if hasattr(args, 'agent_id') else 5
    agent_id = 0
    F,H,W,C = 32,8,5,26
    for episode in range(eval_episodes):
        print(f"Starting episode {episode+1}/{eval_episodes}")

        #Reset Policy
        policy.reset(num_envs=n_envs, num_agents=2)
        for e in range(n_envs):
            policy.register_control_agent(e=e, a=1)

        # Setup diffusion conditioning
        cond = np.full((n_envs,), agent_id, dtype=np.int64)
        cond = th.tensor(cond, device=device)

        # Reset environment
        obs, _, _ = envs.reset([True] * n_envs)

        steps = 0
        done = False
        episode_reward = np.zeros((n_envs, 2))
        max_steps = args.max_steps if hasattr(args, 'max_steps') else 400
        frames = [[obs[i][0]] for i in range(n_envs)]
        samples_frames = [[] for _ in range(n_envs)]

        # Store the previous observation for conditioning
        grid = renderer.extract_grid_from_obs(obs[0][0])
        while not done and steps <= max_steps:
            print(f"Steps: {steps}")

            # Setup Condition Obs Based on Obs
            obs_stack = np.stack([normalize_obs(obs[e][0]) for e in range(n_envs)], axis=0) 
            condition_obs = th.tensor(obs_stack, device=device, dtype=th.float32) # Shape: [n_envs, H, W, C]
            
            assert condition_obs.shape[-1] == 26 # Double Check

            condition_obs = rearrange(condition_obs, "b h w c -> b c h w")
            print(condition_obs.shape)
            print(cond.shape)
            print(n_envs)
            with th.no_grad():
                samples = diffusion.sample(
                x_cond=condition_obs,
                task_embed = cond,
                batch_size=n_envs,
            ) # Shape [n_envs, horizon * C, H, W]
                
            samples = rearrange(samples, "b (f c) h w -> b f h w c", c=C, f=F)
             
            # Render out Samples
            # samples_player_loc = samples[:, :, :, :, :10]
            # samples_dish_onions = samples[:, :, :, :, 10:12]
            # if show_samples:
            #     _ = [renderer.visualize_all_channels(
            #         obs=to_np(samples[0, i]), 
            #         output_dir=os.path.join(frames_dir, f"samples_channels_steps_{steps}_horizon_{i}_env_{0}.png")
            #     ) for i in range(F)]
            
            # I don't think we need to arg_max, the channels should be pretty deterministics
            # samples_player_loc = arg_max(samples_player_loc)
            # samples_12_final = th.cat([samples_player_loc, samples_dish_onions],dim=-1)

            # if show_samples:
            #     _ = [renderer.visualize_all_channels(
            #         obs=to_np(samples_12_final[0, i]), 
            #         output_dir=os.path.join(frames_dir, f"argmax_samples_channels_steps_{steps}_horizon_{i}_env_{0}.png")
            #     ) for i in range(dataset.horizon)]

            
            # Now step through the environment using the 32-step plan
            if max_horizon:
                plan_horizon = min(max_steps - steps, max_horizon)
            else:
                plan_horizon = min(F, max_steps - steps)

            # We begin with the first ego obs (first obs of the environment)
            obs_t = to_torch(obs_stack) # 3,8,5,26
            for t in range(plan_horizon): 
                
                # Unpack Samples_12_final
                # samples_final_player_loc = samples_12_final[:, t, :, :, :10]
                # samples_final_dishes_onions = samples_12_final[:, t, :, :, 10:12]

                # Build (obs_t+1) from the samples_12_final
                # current_obs = np.stack([normalize_obs(obs[e][0]) for e in range(n_envs)], axis=0)
                # curr_obs_11_22 = to_torch(current_obs[..., 10:22])
                # curr_obs_24_26 = to_torch(current_obs[..., 24:])

                # obs_tp1 = th.cat([
                #     samples_final_player_loc,
                #     curr_obs_11_22,
                #     samples_final_dishes_onions,
                #     curr_obs_24_26
                # ], dim=-1)

                obs_tp1 = samples[:,t,...]

                for e in range(n_envs):
                    samples_frames[e].append(to_np(obs_tp1[e]))

            
                step_actions = np.zeros((n_envs, 2, 1), dtype=np.int64)

                for env_i in range(n_envs):
                    # IDM takes in 26 channels
                    ego_action = get_idm_action(to_torch(obs_t[env_i]).unsqueeze(0), to_torch(obs_tp1[env_i]).unsqueeze(0), idm)
                    step_actions[env_i, 0 ] = to_np(ego_action)

                    
                partner_obs_lst = [obs[e][1] for e in range(n_envs)]
                partner_obs = np.stack(partner_obs_lst, axis=0)

                partner_action = policy.step(
                    partner_obs,
                    [(e, 1) for e in range(n_envs)],
                    deterministic=True,
                )

                step_actions[:, 1] = partner_action  # Fill partner action for step t

                print(step_actions)

                obs, shared_obs, reward, done, info, aval_actions = envs.step(step_actions)
                episode_reward += to_np(reward).squeeze(axis=2)
                obs_t = obs_tp1
                
                for e in range(n_envs):
                    frames[e].append(obs[e][0])

                # Check for early termination
                done = np.all(done)
                steps += 1
                if steps >= max_steps:
                    print("here")
                    print(done, steps)
                    break
        mean_episode_reward = episode_reward.mean(axis=0)
        print(f"Episode {episode+1} complete: steps={steps}, reward={mean_episode_reward}")
        metrics = {
            'episode': episode,
            'steps': steps,
            'rewards': episode_reward.tolist(),
            'mean_reward': mean_episode_reward.tolist(),
            'total_reward': episode_reward.sum().tolist()
        }
        all_metrics.append(metrics)
        episode_rewards.append(mean_episode_reward)
        with open(osp.join(metrics_dir, f"episode_{episode+1}_metrics.pkl"), 'wb') as f:
            pickle.dump(metrics, f)
        
        for e in range(n_envs):
            frames[e] = rearrange(frames[e], "f w h c -> f h w c")
            grid = renderer.extract_grid_from_obs(frames[e][0])
            env_dir = osp.join(video_dir, f"episode_{episode+1}_env_{e+1}")
            os.makedirs(env_dir, exist_ok=True)
            saved_video = renderer.render_trajectory_video(
                frames[e], 
                grid, 
                output_dir=env_dir,
                video_path=osp.join(env_dir, f"actual_trajectory.mp4"),
                fps=1)
            _ = renderer.render_trajectory_video(
                samples_frames[e],
                grid,
                output_dir=env_dir,
                video_path=osp.join(env_dir, f"samples_trajectory.mp4"),
                fps=1
            )
            print(f"Video saved to {saved_video}")
    
    if episode_rewards:
        mean_reward = np.mean([r[0] for r in episode_rewards])  # Agent 0 rewards
        std_reward = np.std([r[0] for r in episode_rewards])
        coop_mean_reward = np.mean([r[1] for r in episode_rewards])  # Agent 1 rewards
        coop_std_reward = np.std([r[1] for r in episode_rewards])
        total_mean = np.mean([np.sum(r) for r in episode_rewards])  # Total team rewards
    else:
        mean_reward = std_reward = coop_mean_reward = coop_std_reward = total_mean = 0.0

    print(f"Evaluation complete!")
    print(f"Agent 0 (Diffusion+IDM) mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Agent 1 (Partner) mean reward: {coop_mean_reward:.2f} ± {coop_std_reward:.2f}")
    print(f"Team total mean reward: {total_mean:.2f}")
    
    # Save all metrics to file
    summary = {
        'args': args,
        'metrics': all_metrics,
        'episode_rewards': episode_rewards,
        'agent0_mean_reward': mean_reward,
        'agent0_std_reward': std_reward,
        'agent1_mean_reward': coop_mean_reward,
        'agent1_std_reward': coop_std_reward,
        'team_mean_reward': total_mean,
        'environment': args.layout_name,
        'agent_id': agent_id,
        'episodes': eval_episodes,
        'steps_per_episode': steps / max(1, eval_episodes)
    }
    
    metrics_path = osp.join(basedir, "eval_summary.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(summary, f)
    print(f"Summary metrics saved to {metrics_path}")
    envs.close()
    return summary

def load_diffusion_model(args, device):
    
    trainer = OvercookedTrainer(args, device)
    # Load checkpoint
    checkpoint_path = osp.join(args.diffusion_loadpath, f"modl-{args.diffusion_milestone}.pt")
    print(f"Loading diffusion model from {checkpoint_path}")
    trainer_actual = trainer.trainer
    trainer_actual.load(checkpoint_path)
    return trainer_actual.ema.ema_model.to(device)

                

if __name__ == "__main__":
    parser = get_config()
    args = sys.argv[1:]
    args = parse_args(args, parser)
    device = th.device('cpu' if not th.cuda.is_available() else 'cuda')
    # device = th.device("cpu")

    # load diffusion model function from disk
    diffusion = load_diffusion_model(args, device)

    #overrde episode len
    args.episode_length = 400
    
    # MAPT Setup
    os.environ["layout"] = args.layout_name
    args.env_name = "Overcooked"
    population_yaml_path = args.population_yaml_path
    policy, featurize_type = get_agent(population_yaml_path, "sp10_final", "cpu")
    print("featurize_type: ", featurize_type)

    idm_path = args.idm_loadpath
    if os.path.exists(idm_path):
        print(f"Loading IDM model from {idm_path}")
        idm = th.load(idm_path)
        idm_model = InverseDynamicsModel(num_actions=6)
        idm_model.load_state_dict(idm['model'])
        idm_model = idm_model.to(device)
        idm_model.eval()
    else:
        print(f"IDM model not found at {idm_path}, please provide the correct path")
        sys.exit(1)

    results = full_horizon_eval(
        args=args,
        diffusion=diffusion,
        idm=idm_model,
        policy=policy,
        device=device,
        show_samples=True,
        eval_episodes=10)