import sys
from overcooked.eval.evaluation import MultiPolicyTester
from overcooked.agent.diffusion_agent import DiffusionPlannerAgent as Agent
from mapbt.config import get_config
from overcooked.utils.utils import load_action_proposal_model, load_world_model


def main():
    parser = get_config()
    args = sys.argv[1:]
    args = parse_args(args, parser)
    args.episode_length = 400
    args.old_dynamics = True
    args.n_envs = 2
    args.population_yaml_path = "/home/law/Workspace/repos/COMBO/Overcooked_Population_Data/custom/sp_vs_best_r_sp_config.yml"
    
    args.action_proposal_model_path = "/mnt/linux_space/action_proposal_upgraded_run_1/modl-100.pt"
    args.diffusion_model_path = "/mnt/linux_space/upgraded_expert_run_1/modl-25.pt"

    world_model = load_world_model(args, args.diffusion_model_path, num_classes=68)
    action_proposal_model = load_action_proposal_model(args, args.action_proposal_model_path, sampling_timesteps=100)

    

    agent = Agent(args=args,
                  world_model=world_model,
                  action_proposal_model=action_proposal_model,
                  num_envs=args.n_envs,
                  target_reward=100,
                  planning_horizon=32,
                  num_action_candidates=4,
                  num_simulations_per_plan=2,
                  num_processes=12)
    tester = MultiPolicyTester(args, agent, policies=["sp1_final"])
    tester.evaluate_agent_against_partners(num_episodes=1)
    
    
    # dataset_tester = DatasetTester(args)
    # dataset_tester.plot_dataset_data()

    # args.idm_path = "AVDC/flowdiffusion/idm/models/idm_26_od.pt"
    # idm_tester = IverseDynamicsTester(args)
    # idm_tester.run_validation(num_samples=1000000)

    # args.value_model_path = "./reward_model_checkpoints_3/reward_predictor_step40000.pt"
    # reward_model_validator = RewardModelValidator(args)
    # reward_model_validator.run_validation(num_samples=1000000)

    # args.diffusion_model_path = "/mnt/linux_space/full_expert_run_3/modl-75.pt"
    # args.diffusion_model_path = "/mnt/linux_space/full_expert_run_1/full_expert_run_1/modl-75.pt"
    # args.diffusion_model_path = "/mnt/linux_space/full_expert_4/modl-73.pt"
    # world_model_tester = WorldModelTester(args)
    # world_model_tester.run_offline_divergence_test(num_samples=300, rollout_horizon=32, num_candidates=10)
    # world_model_tester.run_controllability_test(num_samples=100)

    # args.action_proposal_model_path = "/mnt/linux_space/action_proposal_cross_1/modl-100.pt"
    # args.diffusion_model_path = "/mnt/linux_space/full_expert_4/modl-73.pt"
    # agent_tester = DiffusionAgentTester(args,
    #                                     run_name="random_vs_sp1_final_test_1", 
    #                                     num_classes=68, 
    #                                     num_actions=6, 
    #                                     guidance_weight=1.0, 
    #                                     num_candidates=10, 
    #                                     num_processes=12,
    #                                     planning_horizon=32)
    # agent_tester.run_evaluation(num_episodes=3, partner_policy_name="sp1_final")



    


def parse_args(args, parser):
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--results_dir', type=str, default='./overcooked_results', help='Directory to save results and checkpoints')
    parser.add_argument('--basedir', type=str, default='./experiment_results', help='Directory to save results and checkpoints')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (smaller dataset, faster training)')
    parser.add_argument('--horizon', type=int, default=32, help='Sequence horizon for trajectories')
    parser.add_argument('--save_milestone', type=bool, default=True, help='Save milestones with step number in filename') # Or action='store_true'

    # For GoalGaussianDiffusion (configurable ones)
    parser.add_argument('--timesteps', type=int, default=400, help='Number of diffusion timesteps for training (if not debug)')
    parser.add_argument('--sampling_timesteps', type=int, default=10, help='Number of timesteps for DDIM sampling (if not debug)')
    
    # For OvercookedSequenceDataset / HDF5Dataset
    parser.add_argument('--max_path_length', type=int, default=401, help='Maximum path length in episodes (for dataset indexing)')
    parser.add_argument('--chunk_length', type=int, default=None, help='Chunk length for HDF5Dataset (defaults to horizon if None, set via dataset_constructor_args)')
    parser.add_argument('--use_padding', type=bool, default=True, help='Whether to use padding for shorter sequences in dataset')
    
    # parser.add_argument("--diffusion_model_path", type=str, required=True, help="Path to the diffusion model directory")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--max_steps", type=int, default=400, help="Maximum steps per episode")
    # parser.add_argument("--idm_path", type=str, required=True, help="Path to the diffusion model directory")
    # parser.add_argument("--action_proposal_model_path", type=str, required=True, help="Path to the action proposal (diffusion) model checkpoint")
    # parser.add_argument("--value_model_path", type=str, required=True, help="Path to the reward/value model checkpoint")
    # parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Overcooked HDF5 dataset')

    # Mapt Package Args  
    parser.add_argument("--old_dynamics", default=True, action='store_true', help="old_dynamics in mdp")
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
    parser.add_argument("--run_dir", type=str, default="eval_run", help="Directory for evaluation run")
    all_args = parser.parse_known_args(args)[0]
    return all_args

if __name__ == "__main__":
    main()
