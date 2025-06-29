# import sys
# import os

# # TODO: CHANGE ME
# mapbt_path = '/home/law/Workspace/repos/COMBO/AVDC/flowdiffusion/mapbt_package/mapbt'
# if mapbt_path not in sys.path:
#     sys.path.append(mapbt_path)
# overcooked_ai_py_src_path = '/home/law/Workspace/repos/COMBO/AVDC/flowdiffusion/mapbt_package/mapbt/envs/overcooked/overcooked_berkeley/src/overcooked_ai_py'
# if overcooked_ai_py_src_path not in sys.path:
#     sys.path.append(overcooked_ai_py_src_path)
# os.environ['POLICY_POOL'] = "/home/law/Workspace/repos/COMBO/AVDC/flowdiffusion/mapbt_package/mapbt/scripts/overcooked_population"
# import numpy as np
# import torch as th
# import os
# import os.path as osp
# import warnings
# import torch.nn.functional as F
# import pickle
# warnings.filterwarnings("ignore")
# from mapbt_package.mapbt.envs.overcooked.Overcooked_Env import Overcooked
# from mapbt_package.mapbt.envs.env_wrappers import ChooseSubprocVecEnv
# from mapbt_package.mapbt.algorithms.population.policy_pool import PolicyPool as Policy
# from mapbt_package.mapbt.config import get_config

# def parse_args(args, parser):
#     parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
#     parser.add_argument('--results_dir', type=str, default='./overcooked_results', help='Directory to save results and checkpoints')
#     parser.add_argument('--basedir', type=str, default='./experiment_results', help='Directory to save results and checkpoints')
#     parser.add_argument('--debug', action='store_true', help='Enable debug mode (smaller dataset, faster training)')
#     parser.add_argument('--dataset_path', type=str, required=False, help='Path to the Overcooked HDF5 dataset')
#     parser.add_argument('--horizon', type=int, default=32, help='Sequence horizon for trajectories')
#     parser.add_argument('--save_milestone', type=bool, default=True, help='Save milestones with step number in filename') # Or action='store_true'

#     # For OvercookedSequenceDataset / HDF5Dataset
#     parser.add_argument('--max_path_length', type=int, default=401, help='Maximum path length in episodes (for dataset indexing)')
#     parser.add_argument('--chunk_length', type=int, default=None, help='Chunk length for HDF5Dataset (defaults to horizon if None, set via dataset_constructor_args)')
#     parser.add_argument('--use_padding', type=bool, default=True, help='Whether to use padding for shorter sequences in dataset')


#     # For GoalGaussianDiffusion (configurable ones)
#     parser.add_argument('--timesteps', type=int, default=400, help='Number of diffusion timesteps for training (if not debug)')
#     parser.add_argument('--sampling_timesteps', type=int, default=10, help='Number of timesteps for DDIM sampling (if not debug)')

#     # For OvercookedEnvTrainer 
#     parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size (if not debug)')
#     parser.add_argument('--num_validation_samples', type=int, default=4, help='Number of samples to generate during validation step')
#     parser.add_argument('--save_and_sample_every', type=int, default=1000, help='Frequency to save checkpoints and generate samples (if not debug)')
#     parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Probability of dropping condition for CFG during training')
#     parser.add_argument('--split_batches', type=bool, default=True, help='Whether to split batches for Accelerator')
#     parser.add_argument('--resume_checkpoint_path', type=str, required=False, default=None, help='Path to a .pt checkpoint file to resume training from.')
    
#     # overcooked evaluation
#     parser.add_argument("--diffusion_model_path", type=str, required=False, help="Path to the diffusion model directory")
#     parser.add_argument("--dataset", type=str, default="overcooked", help="Dataset name")
#     parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
#     parser.add_argument("--agent_id", type=int, default=0, help="Agent ID for conditioning")
#     parser.add_argument("--max_steps", type=int, default=400, help="Maximum steps per episode")
#     parser.add_argument("--run_dir", type=str, default="eval_run", help="Directory for evaluation run")
#     parser.add_argument("--idm_path", type=str, required=False, help="Path to the diffusion model directory")
#     parser.add_argument("--exp_eval_episodes", type=int, default=3, help="Number of evaluation episodes")
#     parser.add_argument("--show_samples", default=False, action='store_true', help="Whether to visualize samples during evaluation")
#     parser.add_argument("--save_videos", default=False, action='store_true', help="Whether to save videos of the evaluation")
    
#     # Mapt Package Args  
#     parser.add_argument("--old_dynamics", default=False, action='store_true', help="old_dynamics in mdp")
#     parser.add_argument("--layout_name", type=str, default='counter_circuit_o_1order', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
#     parser.add_argument('--num_agents', type=int, default=1, help="number of players")
#     parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
#     parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
#     parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
#     parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")  
#     parser.add_argument("--use_hsp", default=False, action='store_true')   
#     parser.add_argument("--random_index", default=False, action='store_true')
#     parser.add_argument("--use_agent_policy_id", default=False, action='store_true', help="Add policy id into share obs, default False")
#     parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
#     parser.add_argument("--use_detailed_rew_shaping", default=False, action='store_true')
#     parser.add_argument("--random_start_prob", default=0., type=float)
#     parser.add_argument("--store_traj", default=False, action='store_true')
#     # population
#     parser.add_argument("--population_yaml_path", type=str, help="Path to yaml file that stores the population info.")
#     parser.add_argument('--valid_ratio', type=float, default=0.1, help='Fraction of data to use for validation (default: 0.1)')
    
#     # Concept Learning Args
#     parser.add_argument('--test_policies_to_use', nargs='+', type=str, default=None,
#                    help='Specific test policies to use for concept learning')
#     parser.add_argument('--num_concept_runs', type=int, required=False,
#                     help='Number of runs (total times to iterate over data) for each concept learning experiment')
#     parser.add_argument('--test_concept_train_steps', type=int, required=False,
#                     help='Number of training steps to take for test concept learning')
#     parser.add_argument('--exp_group_name', type=str, default=None, 
#                     help='Name for experiment group folder')
    
    
#     all_args = parser.parse_known_args(args)[0]

    

#     return all_args

# def make_eval_env(all_args, run_dir, nenvs=3):
#     def get_env_fn(rank):
#         def init_env():
#             env = Overcooked(all_args, run_dir, rank=rank)
#             env.seed(all_args.seed * 50000 + rank * 10000)
#             return env
#         return init_env
#     return ChooseSubprocVecEnv([get_env_fn(i) for i in range(nenvs)])

# def load_partner_policy(population_yaml_path, policy_name, device="cpu"):
#     """Load a partner policy by name."""
#     policy = Policy(None, None, None, None, device=device)
#     featurize_type = policy.load_population(population_yaml_path, evaluation=True)
#     policy = policy.policy_pool[policy_name]
#     feat_type = featurize_type.get(policy_name, 'ppo')
#     return policy, feat_type

# def to_np(x):
# 	if th.is_tensor(x):
# 		x = x.detach().cpu().numpy()
# 	return x

# def to_torch(x, dtype=None, device=None):
#     DTYPE = th.float
#     DEVICE = 'cuda:0'
#     dtype = dtype or DTYPE
#     device = device or DEVICE
#     if type(x) is dict:
#         return {k: to_torch(v, dtype, device) for k, v in x.items()}
#     elif th.is_tensor(x):
#         return x.to(device).type(dtype)
#     # elif x.dtype.type is np.str_:
#     # 	return torch.tensor(x, device=device)
#     return th.tensor(x, dtype=dtype, device=device)

# parser = get_config()
# args = sys.argv[1:]
# args = parse_args(args, parser)
# args.env_name = "Overcooked"
# args.num_agents = 2
# args.episode_length = 400
# population_yaml_path = args.population_yaml_path
# args.save_gifs = True
# args.cnn_layers_params='32,3,1,1 64,3,1,1 32,3,1,1'
# args.use_render = True
# args.use_wandb = False
# args.n_rollout_threads = 1
# args.seed = 42  
# args.old_dynamics = True
# args.n_training_threads = 1
# args.reward_shaping_horizon = 0 
# args.algorithm_name = "population"
# args.agent0_policy_name = "bc_train"
# args.agent1_policy_name = "bc_train" 
# args.algorithm_name = "population"
# args.use_centralized_V = False 
# args.use_policy_in_env = False 
# args.eval_stochastic = False   


# basedir = "./debug_results"
# eval_episodes = 1
# print(os.getcwd())
# bc_policy, _ = load_partner_policy(population_yaml_path, "bc_train", device="cpu")
# ego_poolicy, _ = load_partner_policy(population_yaml_path, "bc_train", device="cpu")
# device =  th.device("cuda:0" if th.cuda.is_available() else "cpu")
# n_envs = 3

# envs = make_eval_env(args, run_dir=args.run_dir, nenvs=n_envs)
# envs.reset_featurize_type([("bc", "bc") for _ in range(n_envs)])

# video_dir = osp.join(basedir, "videos")
# frames_dir = osp.join(basedir, "frames")
# metrics_dir = osp.join(basedir, "metrics")
# os.makedirs(video_dir, exist_ok=True)
# os.makedirs(frames_dir, exist_ok=True)
# os.makedirs(metrics_dir, exist_ok=True)

# # renderer = OvercookedSampleRenderer()
# all_metrics = []
# episode_rewards = []
# F,H,W,C = 32,8,5,26

# policy_pool = Policy(None, None, None, None, device="cpu")
# featurize_type_mapping = policy_pool.load_population(population_yaml_path, evaluation=True)

# # Create environment mapping
# map_ea2p = {}
# for e in range(n_envs):
#     for a in range(2):
#         map_ea2p[(e, a)] = "bc_train"

# # Set the mapping and reset policies
# policy_pool.set_map_ea2p(map_ea2p)

# for policy_name, policy in policy_pool.policy_pool.items():
#     if hasattr(policy, 'actor'):
#         print(f"Policy {policy_name} actor parameters:")
#         for name, param in policy.actor.named_parameters():
#             print(f"  {name}: {param.shape}, requires_grad: {param.requires_grad}")
#             if param.numel() < 10:
#                 print(f"    values: {param.data}")
        
#         # Check if the policy is in eval mode
#         print(f"Policy {policy_name} training mode: {policy.actor.training}")
#         policy.actor.eval()  # Set to eval mode

# # Set featurize types for environments
# featurize_type = [[policy_pool.featurize_type[map_ea2p[(e, a)]] for a in range(2)] for e in range(n_envs)]
# print("featurize_type", featurize_type)
# envs.reset_featurize_type(featurize_type)


# for episode in range(eval_episodes):
#     print(f"Starting episode {episode+1}/{eval_episodes}")

#     # Reset all policies
#     [policy.reset(n_envs, 2) for policy_name, policy in policy_pool.policy_pool.items()]
    
#     # Register control agents
#     for e in range(n_envs):
#         for agent_id in range(2):
#             if not map_ea2p[(e, agent_id)].startswith("script:"):
#                 policy_pool.policy_pool[map_ea2p[(e, agent_id)]].register_control_agent(e, agent_id)

    

#     # Reset environment properly
#     reset_choose = np.ones(n_envs) == 1
#     obs, _, _ = envs.reset(reset_choose)

#     steps = 0
#     done = False
#     episode_reward = np.zeros((n_envs, 2))
#     max_steps = 400

#     while not done and steps < max_steps:
#         # Initialize actions
#         actions = np.full((n_envs, 2, 1), fill_value=0).tolist()
        
#         # Get actions from each policy
#         for policy_name, policy in policy_pool.policy_pool.items():
#             if len(policy.control_agents) > 0:
#                 policy.prep_rollout()
#                 obs_lst = [obs[e][a] for (e, a) in policy.control_agents]
#                 agents = policy.control_agents
#                 step_actions = policy.step(np.stack(obs_lst, axis=0), agents, deterministic=False)
#                 for action, (e, a) in zip(step_actions, agents):
#                     actions[e][a] = action
#                     # print(f"Agent {a} in Env {e} takes action: {action}")

#         # Step environment
#         actions = np.array(actions)
#         obs, shared_obs, reward, done, info, aval_actions = envs.step(actions)
#         episode_reward += to_np(reward).squeeze(axis=2)
        
#         # Check for early termination
#         done = np.all(done)
#         steps += 1
#     mean_episode_reward = episode_reward.mean(axis=0)
#     print(f"Episode {episode+1} complete: steps={steps}, reward={episode_reward.mean(axis=0)}")
#     episode_rewards.append(mean_episode_reward)
    

# # Calculate final statistics
# if episode_rewards:
#     mean_reward = np.mean([r[0] for r in episode_rewards])
#     std_reward = np.std([r[0] for r in episode_rewards])
#     coop_mean_reward = np.mean([r[1] for r in episode_rewards])
#     coop_std_reward = np.std([r[1] for r in episode_rewards])
#     total_mean = np.mean([np.sum(r) for r in episode_rewards])
# else:
#     mean_reward = std_reward = coop_mean_reward = coop_std_reward = total_mean = 0.0

# print(f"Evaluation complete!")
# print(f"Agent 0 (BC Policy) mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
# print(f"Agent 1 (BC Policy) mean reward: {coop_mean_reward:.2f} ± {coop_std_reward:.2f}")
# print(f"Team total mean reward: {total_mean:.2f}")

# envs.close()

import sys
import os

# TODO: CHANGE ME
mapbt_path = '/home/law/Workspace/repos/COMBO/AVDC/flowdiffusion/mapbt_package/mapbt'
if mapbt_path not in sys.path:
    sys.path.append(mapbt_path)
overcooked_ai_py_src_path = '/home/law/Workspace/repos/COMBO/AVDC/flowdiffusion/mapbt_package/mapbt/envs/overcooked/overcooked_berkeley/src/overcooked_ai_py'
if overcooked_ai_py_src_path not in sys.path:
    sys.path.append(overcooked_ai_py_src_path)
os.environ['POLICY_POOL'] = "/home/law/Workspace/repos/COMBO/AVDC/flowdiffusion/mapbt_package/mapbt/scripts/overcooked_population"
import numpy as np
import torch as th
import os
import os.path as osp
import warnings
import torch.nn.functional as F
import pickle
warnings.filterwarnings("ignore")
from mapbt_package.mapbt.envs.overcooked.Overcooked_Env import Overcooked
from mapbt_package.mapbt.envs.env_wrappers import ChooseSubprocVecEnv
from mapbt_package.mapbt.algorithms.population.policy_pool import PolicyPool as Policy
from overcooked_sample_renderer import OvercookedSampleRenderer
from einops.einops import rearrange
from mapbt_package.mapbt.config import get_config

def parse_args(args, parser):
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--results_dir', type=str, default='./overcooked_results', help='Directory to save results and checkpoints')
    parser.add_argument('--basedir', type=str, default='./experiment_results', help='Directory to save results and checkpoints')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (smaller dataset, faster training)')
    parser.add_argument('--dataset_path', type=str, required=False, help='Path to the Overcooked HDF5 dataset')
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
    parser.add_argument('--resume_checkpoint_path', type=str, required=False, default=None, help='Path to a .pt checkpoint file to resume training from.')
    
    # overcooked evaluation
    parser.add_argument("--diffusion_model_path", type=str, required=False, help="Path to the diffusion model directory")
    parser.add_argument("--dataset", type=str, default="overcooked", help="Dataset name")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--agent_id", type=int, default=0, help="Agent ID for conditioning")
    parser.add_argument("--max_steps", type=int, default=400, help="Maximum steps per episode")
    parser.add_argument("--run_dir", type=str, default="eval_run", help="Directory for evaluation run")
    parser.add_argument("--idm_path", type=str, required=False, help="Path to the diffusion model directory")
    parser.add_argument("--exp_eval_episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--show_samples", default=False, action='store_true', help="Whether to visualize samples during evaluation")
    parser.add_argument("--save_videos", default=False, action='store_true', help="Whether to save videos of the evaluation")
    
    # Mapt Package Args  
    parser.add_argument("--old_dynamics", default=False, action='store_true', help="old_dynamics in mdp")
    parser.add_argument("--layout_name", type=str, default='counter_circuit_o_1order', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int, default=1, help="number of players")
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
    
    # Concept Learning Args
    parser.add_argument('--test_policies_to_use', nargs='+', type=str, default=None,
                   help='Specific test policies to use for concept learning')
    parser.add_argument('--num_concept_runs', type=int, required=False,
                    help='Number of runs (total times to iterate over data) for each concept learning experiment')
    parser.add_argument('--test_concept_train_steps', type=int, required=False,
                    help='Number of training steps to take for test concept learning')
    parser.add_argument('--exp_group_name', type=str, default=None, 
                    help='Name for experiment group folder')
    
    
    all_args = parser.parse_known_args(args)[0]

    

    return all_args

def make_eval_env(all_args, run_dir, nenvs=3):
    def get_env_fn(rank):
        def init_env():
            env = Overcooked(all_args, run_dir, rank=rank)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(nenvs)])

def load_partner_policy(population_yaml_path, policy_name, device="cpu"):
    """Load a partner policy by name."""
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

parser = get_config()
args = sys.argv[1:]
args = parse_args(args, parser)
args.env_name = "Overcooked"
args.num_agents = 2
args.episode_length = 400
population_yaml_path = args.population_yaml_path


basedir = "./debug_results"
eval_episodes = 10
print(os.getcwd())
bc_policy, _ = load_partner_policy(population_yaml_path, "bc_train", device="cpu")
ego_poolicy, _ = load_partner_policy(population_yaml_path, "actor_best_r_vs_bc_train", device="cpu")
device =  th.device("cuda:0" if th.cuda.is_available() else "cpu")
n_envs = 3

envs = make_eval_env(args, run_dir=args.run_dir, nenvs=n_envs)
envs.reset_featurize_type([("ppo", "bc") for _ in range(n_envs)])

video_dir = osp.join(basedir, "videos")
frames_dir = osp.join(basedir, "frames")
metrics_dir = osp.join(basedir, "metrics")
os.makedirs(video_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

renderer = OvercookedSampleRenderer()
all_metrics = []
episode_rewards = []
F,H,W,C = 32,8,5,26


for episode in range(eval_episodes):
    print(f"Starting episode {episode+1}/{eval_episodes}")

    # Reset Policy
    bc_policy.reset(num_envs=n_envs, num_agents=1)
    ego_poolicy.reset(num_envs=n_envs, num_agents=1)
    for e in range(n_envs):
        ego_poolicy.register_control_agent(e=e, a=0)
        bc_policy.register_control_agent(e=e, a=1)

    # Reset environment
    obs, _, _ = envs.reset([True] * n_envs)

    steps = 0
    done = False
    episode_reward = np.zeros((n_envs, 2))
    max_steps = 400
    frames = [[obs[i][0]] for i in range(n_envs)]

    while not done and steps < max_steps:
        step_actions = np.zeros((n_envs, 2, 1), dtype=np.int64)

        # Agent 0 actions using policy
        agent0_obs_lst = [obs[e][0] for e in range(n_envs)]
        agent0_obs = np.stack(agent0_obs_lst, axis=0)
        agent0_actions = ego_poolicy.step(
            agent0_obs,
            [(e, 0) for e in range(n_envs)],
            deterministic=False,
        )
        step_actions[:, 0] = agent0_actions
        
        # Agent 1 actions using policy
        agent1_obs_lst = [obs[e][1] for e in range(n_envs)]
        agent1_obs = np.stack(agent1_obs_lst, axis=0)
        agent1_actions = bc_policy.step(
            agent1_obs,
            [(e, 1) for e in range(n_envs)],
            deterministic=False,
        )
        # print(f"Agent 0 actions: {agent0_actions}, Agent 1 actions: {agent1_actions}")
        step_actions[:, 1] = agent1_actions

        # Step environment
        obs, shared_obs, reward, done, info, aval_actions = envs.step(step_actions)
        episode_reward += to_np(reward).squeeze(axis=2)
        
        for e in range(n_envs):
            frames[e].append(obs[e][0])

        # Check for early termination
        done = np.all(done)
        steps += 1

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
    
    # Render videos (removed the problematic samples_frames line)
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
            fps=1,
            normalize=False)
        print(f"Video saved to {saved_video}")

# Calculate final statistics
if episode_rewards:
    mean_reward = np.mean([r[0] for r in episode_rewards])
    std_reward = np.std([r[0] for r in episode_rewards])
    coop_mean_reward = np.mean([r[1] for r in episode_rewards])
    coop_std_reward = np.std([r[1] for r in episode_rewards])
    total_mean = np.mean([np.sum(r) for r in episode_rewards])
else:
    mean_reward = std_reward = coop_mean_reward = coop_std_reward = total_mean = 0.0

print(f"Evaluation complete!")
print(f"Agent 0 (BC Policy) mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"Agent 1 (BC Policy) mean reward: {coop_mean_reward:.2f} ± {coop_std_reward:.2f}")
print(f"Team total mean reward: {total_mean:.2f}")

