import argparse
import gc
from copy import deepcopy
import datetime
from pathlib import Path
import sys
mapbt_path = '/home/law/Workspace/repos/COMBO/mapbt_package/mapbt'
if mapbt_path not in sys.path:
    sys.path.append(mapbt_path)
overcooked_ai_py_src_path = '/home/law/Workspace/repos/COMBO/mapbt_package/mapbt/envs/overcooked/overcooked_berkeley/src/overcooked_ai_py'
if overcooked_ai_py_src_path not in sys.path:
    sys.path.append(overcooked_ai_py_src_path)
from matplotlib import pyplot as plt
import pandas as pd
from overcooked_dataset import OvercookedSequenceDataset

import numpy as np
import torch as th
import os
import os.path as osp
import warnings
import torch.nn.functional as F
import pickle
warnings.filterwarnings("ignore")
from idm.inverse_dynamics import InverseDynamicsModel
from mapbt_package.mapbt.algorithms.population.policy_pool import PolicyPool as Policy
from overcooked_sample_renderer import OvercookedSampleRenderer
from mapbt_package.mapbt.config import get_config
from experiments_util import managed_model_loading, managed_concept_trainer
from experiments_classes import EvaluationExperiment
from goal_diffusion import GoalGaussianDiffusion
from unet import UnetOvercooked
from ema_pytorch import EMA



class ExperimentRunner:
    def __init__(self, args, num_concepts):
        """
        Initialize the experiment runner with configuration parameters.
        
        Args:
            args: Parsed command line arguments
            use_successive_models: Whether to use the model from the previous step
        """
        self.args = args

        # Disable WandB and Debugging
        self.args.wandb = False
        self.args.debug = False
        self.args.wandb_project = None
        self.args.wandb_run_name = None
        self.args.wandb_entity = None
        self.args.wandb_group = None
        
        self.device = th.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")
        print(f"Using device: {self.device}")
        
        # Set up output directories
        self.base_output_dir = Path(args.basedir)
        self.exp_group_dir = self.base_output_dir / (args.exp_group_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.exp_group_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize common resources
        self.idm_model = self._load_idm_model()
        self.renderer = OvercookedSampleRenderer()
        
        # Track experiment state
        self.current_model_path = args.diffusion_model_path
        self.current_args = deepcopy(args)
        self.all_experiment_results = []
        
        # Cache for datasets to avoid reloading
        self.dataset_cache = {}

        # Embeddings
        self.policy_embeddings = {}
        self.embedding_dim = 512 # Default embedding dimension
        self.num_concepts = num_concepts
        self.embedding_changes = []
        self.embedding_grad_norms = {}
        self.embedding_loss_history = {}
        self.embedding_history = {}

        assert self.num_concepts > 1, "num_concepts must be greater than 1 for concept learning due to dummy policy embedding"

        self.observation_dim = (8,5,26) #TODO: this should be set based on the dataset or model


        
    def _load_idm_model(self):
        """Load the Inverse Dynamics Model."""
        print(f"Loading IDM model from {self.args.idm_path}")
        weights = th.load(self.args.idm_path)
        idm = InverseDynamicsModel(num_actions=6)
        idm.load_state_dict(weights['model'])
        idm.to(self.device)
        idm.eval()
        return idm
    
    #TODO: REMOVE AUTO NORMALIZE FROM ALL DIFFUSION MODELS*****

    def load_diffusion_model(self, model_path, ema=True, num_classes=None):
        """Load the diffusion model for evaluation only."""
        print(f"Loading diffusion model from {model_path}, ema={ema}, num_classes={num_classes}")
        if not osp.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        ckpt = th.load(model_path, map_location="cpu")
        unet = UnetOvercooked(
            horizon=self.args.horizon,
            obs_dim=self.observation_dim,
            num_classes=num_classes,
        ).to(self.device)
        H,W,C = self.observation_dim
        diffusion = GoalGaussianDiffusion(
            model=unet,
            channels=C * 32,
            image_size=(H,W),
            timesteps=1 if self.args.debug else 1000,
            sampling_timesteps=1 if self.args.debug else 100,
            loss_type="l2",
            objective="pred_v",
            beta_schedule="cosine",
            min_snr_loss_weight=True,
            guidance_weight=getattr(self.args, 'guidance_weight', 1.0),
        ).to(self.device)
        if ema:
            ema_wrap = EMA(diffusion,beta = 0.999,update_every=10)
            ema_wrap.load_state_dict(ckpt['ema'])
            return ema_wrap
        else:
            diffusion.load_state_dict(ckpt['model'])
            return diffusion
    
    def _get_dataset(self, dataset_path, split="test"):
        """Get a dataset from cache or load it."""
        cache_key = f"{dataset_path}_{split}"
        if cache_key not in self.dataset_cache:
            print(f"Loading {split} dataset from {dataset_path}")
            dataset_args = argparse.Namespace(
                dataset_path=dataset_path,
                horizon=self.args.horizon,
                max_path_length=self.args.max_path_length,
                episode_length=self.args.episode_length,
                chunk_length=self.args.chunk_length,
                use_padding=self.args.use_padding,
            )
            self.dataset_cache[cache_key] = OvercookedSequenceDataset(
                args=dataset_args, split=split
            )
        return self.dataset_cache[cache_key]

    def get_or_create_embedding(self, policy_id):
        """Get or create an embedding for a given policy ID."""
        if policy_id in self.policy_embeddings:
            print(f"Using existing embedding for policy {policy_id} with shape {self.policy_embeddings[policy_id].shape}")
            return self.policy_embeddings[policy_id]
        
        random_embedding = th.randn(self.num_concepts, self.embedding_dim, device=self.device)
        self.policy_embeddings[policy_id] = random_embedding

        print(f"Created new embedding for policy {policy_id} with shape {random_embedding.shape}")
        print(f"Total embeddings now: {len(self.policy_embeddings)}")
        return self.policy_embeddings[policy_id]
    
    def save_embedding(self, path=None):
        if path is None:
            path = self.base_output_dir / "policy_embeddings.pt"
        th.save(self.policy_embeddings, path)
        print(f"Saved {len(self.policy_embeddings)} policy embeddings to {path}")
    
    def load_embedding(self, path):
        """Load policy embeddings from a file."""
        self.policy_embeddings = th.load(path, map_location=self.device)
        self.embedding_dim = next(iter(self.policy_embeddings.values())).shape[1]
        print(f"Loaded {len(self.policy_embeddings)} policy embeddings from {path}")

    
    def load_partner_policy(self, policy_name, device="cpu"):
        """Load a partner policy by name."""
        policy = Policy(None, None, None, None, device=device)
        featurize_type = policy.load_population(self.args.population_yaml_path, evaluation=True)
        policy = policy.policy_pool[policy_name]
        feat_type = featurize_type.get(policy_name, 'ppo')
        return policy, feat_type

    def create_concept_learning_args(self, concept_params):
        """
        Create a new set of arguments for concept learning based on the main_args and concept_params.
        """
        # Start out with a copy of the main args
        cl_args = deepcopy(self.args)
        
        # Override args for concept learning
        cl_args.dataset_path = concept_params['dataset_path']
        cl_args.pretrained_model_path = concept_params['pretrained_model_path']
        cl_args.dummy_policy_id = concept_params['dummy_policy_id']
        cl_args.new_policy_id = concept_params['new_policy_id']
        cl_args.results_dir = concept_params['results_dir_concept_learning']
        cl_args.milestone_name = concept_params['milestone_name']
        cl_args.max_train_steps = concept_params['train_steps']

        cl_args.horizon = concept_params.get('horizon', self.args.horizon) # Use the same horizon as main_args
        cl_args.guidance_weight = concept_params.get('guidance_weight', 1.0)

        # Training Args
        cl_args.train_batch_size = concept_params.get('train_batch_size', self.args.train_batch_size)
        cl_args.num_validation_samples = concept_params.get('num_validation_samples', self.args.num_validation_samples)
        cl_args.save_and_sample_every = concept_params.get('save_and_sample_every', 1000)
        cl_args.cond_drop_prob = concept_params.get('cond_drop_prob', self.args.cond_drop_prob)
        cl_args.split_batches = concept_params.get('split_batches', self.args.split_batches)
        cl_args.save_milestone = concept_params.get('save_milestone', self.args.save_milestone)

        # Dataset args for OvercookedSequenceDataset
        cl_args.max_path_length = self.args.max_path_length
        cl_args.episode_length = self.args.episode_length
        cl_args.chunk_length = self.args.chunk_length
        cl_args.use_padding = self.args.use_padding
        cl_args.dataset_split = concept_params["dataset_split"]

        # Single episode training args
        cl_args.target_episode_idx = concept_params.get('target_episode_idx', None)
        cl_args.target_policy_name = concept_params.get('target_policy_name', None)

        # Disable WandB and Debugging
        cl_args.wandb = False
        cl_args.debug = False
        cl_args.wandb_project = None
        cl_args.wandb_run_name = None
        cl_args.wandb_entity = None
        cl_args.wandb_group = None

        return cl_args
    
    def train_concept_learn(self, cl_args):
        """
        Train a concept learning model for Overcooked using the provided arguments.
        """
        gc.collect()
        th.cuda.empty_cache()

        # Get or create initial embedding for this policy
        pid = cl_args.new_policy_id

        with managed_concept_trainer(self, cl_args) as (trained_embed, metrics):
            self.policy_embeddings[pid] = trained_embed
            if pid not in self.embedding_loss_history:
                self.embedding_loss_history[pid] = []
            self.embedding_loss_history[pid].extend(metrics.get('loss', []))
            self.save_embedding()

            # Get model path before cleanup
            model_path = os.path.join(cl_args.results_dir, f"modl-{cl_args.milestone_name}.pt")
            assert os.path.exists(model_path), f"Expected model at {model_path} but not found"
            
            return model_path

    
    def concept_learn_one_step(self, 
                           new_policy_id, 
                           milestone_name, 
                           pretrained_model_path,
                           dummy_policy_id,
                           train_steps,
                           eval_partner_policy=None,
                           eval_layout_name=None,
                           eval_horizon=None,
                           experiment_index=0,
                           is_test_split=False,
                           target_episode_idx=None):
        """Create a single concept learning experiment definition."""
        cl_params = {
            'dataset_path': self.args.dataset_path,
            'pretrained_model_path': pretrained_model_path,
            'train_steps': train_steps,
            'max_train_steps': train_steps,  # Ensure both are set
            'dummy_policy_id': dummy_policy_id,
            'new_policy_id': new_policy_id,
            'results_dir_concept_learning': str(self.base_output_dir / f"learned_concepts" / f"concept_{new_policy_id}_training_idx_{experiment_index}"),
            'milestone_name': milestone_name,
            'horizon': eval_horizon if eval_horizon is not None else self.args.horizon,
            'guidance_weight': getattr(self.args, 'guidance_weight', 1.0),
            'dataset_split': "test" if is_test_split else "train",
            'target_episode_idx': target_episode_idx,
            'target_policy_name': eval_partner_policy,
        }

        experiment_definition = {
            "name": f"concept_learn_experiment_id_{new_policy_id}_experiment_idx_{experiment_index}",
            "concept_learning_params": cl_params,
        }

        # if eval_partner_policy:
        #     experiment_definition["evaluation_configs"] = [{
        #         "layout_name": eval_layout_name if eval_layout_name is not None else self.args.layout_name,
        #         "agent_id": new_policy_id,
        #         "horizon": eval_horizon if eval_horizon is not None else self.args.horizon,
        #         "partner_policy_name": eval_partner_policy,
        #     }]

        return experiment_definition
    
    def create_test_concept_learn_experiment(self, 
                                         num_concept_runs, 
                                         fine_tuning_steps, 
                                         target_episodes=dict, 
                                         test_policies_to_use=None):
        """Create concept learning experiments for test policies."""
        experiment_configs = []
        
        
        # Get test dataset using caching mechanism
        dataset = self._get_dataset(self.args.dataset_path, "test")
        
        if not dataset.test_partner_policies:
            print("No test partner policies found in the dataset.")
            return experiment_configs
            
        test_policies = dataset.test_partner_policies
        print(f"Found {len(test_policies)} test policies: {list(test_policies.keys())}")
        
        # Track the actual policy counter (only for included policies)
        policy_counter = 0
        
        for policy_name in test_policies:
                
            if test_policies_to_use and policy_name not in test_policies_to_use:
                print(f"Skipping policy {policy_name} as it is not in specified test policies.")
                continue
                
            matching_episodes = dataset.get_path_indexes_episode(policy_name)
            if not matching_episodes:
                print(f"Skipping policy {policy_name} as no episodes were found.")
                continue
                
            # Handle target episodes index validation
            if policy_name in target_episodes:
                target_idx = target_episodes[policy_name]
                if target_idx >= len(matching_episodes):
                    target_idx = target_idx % len(matching_episodes)
                    print(f"Target index {target_idx} for policy {policy_name} out of bounds. Wrapping to {target_idx}.")
                    target_episodes[policy_name] = target_idx
                    
            print(f"Found {len(matching_episodes)} episodes for policy '{policy_name}'")
            
            for run_idx in range(num_concept_runs):
                base_episode_idx = target_episodes.get(policy_name, 0)
                episode_idx = (base_episode_idx + run_idx) % len(matching_episodes)
                
                # Use policy_counter for sequential experiment indices
                experiment_idx = policy_counter * num_concept_runs + run_idx
                print(f"Using episode {episode_idx} for {policy_name} run {run_idx} experiment {experiment_idx}.")
                
                milestone_name = f"test_concept_{policy_name}_run_{run_idx}_episode_{episode_idx}"
                
                # Create experiment config
                experiment_configs.append(
                    self.concept_learn_one_step(
                        new_policy_id=test_policies[policy_name],
                        milestone_name=milestone_name,
                        pretrained_model_path=self.args.diffusion_model_path,
                        dummy_policy_id=dataset.dummy_id,
                        train_steps=fine_tuning_steps,
                        eval_partner_policy=policy_name,
                        experiment_index=experiment_idx,
                        is_test_split=True,
                        target_episode_idx=episode_idx,
                    )
                )
                
            # Increment policy counter for successful processing
            policy_counter += 1
            
        print(f"Generated {len(experiment_configs)} test concept learning experiments.")
        return experiment_configs
    
    def evaluate(self, config, diffusion_model):
        """Run evaluation based on the provided configuration."""
        agent_id = config["agent_id"]
        layout_name = config["layout_name"]
        horizon = config["horizon"]
        partner_policy_name = config["partner_policy_name"]

        # Create a unique basedir for this experiment
        current_experiment_name = f"{layout_name}_agent_id_{agent_id}_horizon_{horizon}_partner_policy_{partner_policy_name}"
        current_run_basedir = self.exp_group_dir / "evaluation" / current_experiment_name
        current_run_basedir.mkdir(parents=True, exist_ok=True)
        eval_experiment = EvaluationExperiment(
            args=self.args,
            diffusion=diffusion_model,
            idm=self.idm_model,
            policy_name=partner_policy_name,
            policy_id=agent_id,
            num_episodes=self.args.exp_eval_episodes,
            results_dir=current_run_basedir,
            n_envs=self.args.n_envs,
            max_steps=self.args.max_steps,
            planning_horizon=horizon,
            device=self.device
         )
        summary = eval_experiment.run(save_videos=self.args.save_videos)
        
        # Add concept learning metadata if present
        if "concept_learning_params" in config and "target_episode_idx" in config["concept_learning_params"]:
            summary['training_episode_idx'] = config["concept_learning_params"]["target_episode_idx"]
            
        # Add experiment metadata
        summary['experiment_horizon_val'] = config["horizon"]
        summary['experiment_agent_id_val'] = config["agent_id"]
        summary['experiment_layout_name_val'] = config["layout_name"]
        summary['experiment_partner_policy_val'] = config["partner_policy_name"]
            
        return summary
    
    def run_cl_experiments_with_context_managers(self):
        experiment_configs = self.create_test_concept_learn_experiment(
            num_concept_runs=self.args.num_concept_runs,
            fine_tuning_steps=self.args.test_concept_train_steps,
            test_policies_to_use=["sp9_final","sp10_final", "bc_train"],
            target_episodes={}
        )
        
        assert experiment_configs, "No experiment configurations generated"

        # Get dataset to determine num_classes for model architecture
        # dataset = self._get_dataset(self.args.dataset_path, "train")
        # num_classes = dataset.num_partner_policies  # This should be the total number of policies the model can handle
        for exp_idx, exp_config in enumerate(experiment_configs):
            print(f"\n=== Processing experiment {exp_idx+1}/{len(experiment_configs)}: {exp_config['name']} ===")
            
            # Initialize variables for this experimentwhat
            use_ema = "concept_learning_params" not in exp_config
            model_path_for_eval = self.current_model_path
            experiment_results = []
            
            # concept_learn
            if "concept_learning_params" in exp_config:
                cl_params = exp_config["concept_learning_params"]
                cl_params['pretrained_model_path'] = self.args.diffusion_model_path
                
                cl_args = self.create_concept_learning_args(cl_params)
                exp_dir = self.exp_group_dir / exp_config["name"]
                exp_dir.mkdir(parents=True, exist_ok=True)
                cl_args.results_dir = str(exp_dir / "concept_learning_output")
                
                print(f"Starting concept learning for {exp_config['name']}...")
                model_path_for_eval = self.train_concept_learn(cl_args)
                print(f"Concept learning complete. Model saved to {model_path_for_eval}")
            
            
            # Run evaluations with context managers
            with managed_model_loading(self, model_path_for_eval, use_ema, self.num_concepts) as diffusion_model:
                eval_configs = exp_config.get("evaluation_configs", [])
                if not isinstance(eval_configs, list):
                    eval_configs = [eval_configs] if eval_configs else []
                    
                for eval_config in eval_configs:
                    if not eval_config:
                        continue
                    result = self.evaluate(eval_config, diffusion_model)
                    experiment_results.append(result)
            
            # Save results for this experiment
            self.all_experiment_results.append({
                "name": exp_config["name"],
                "config": exp_config,
                "results": experiment_results,
                "model_path": model_path_for_eval
            })
            
            # Clear results from memory and force cleanup
            del experiment_results
            gc.collect()
            th.cuda.empty_cache()
        
        # Save final results (keep existing logic)
        results_file = self.exp_group_dir / "all_experiment_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(self.all_experiment_results, f)
            
        print(f"\nAll experiments complete. Results saved to {results_file}")
        return self.all_experiment_results
    
    def run_base_model_evaluation(self, policies_to_evaluate=None):
        """
        Run evaluations of the base diffusion model on both training and test policies.
        """
        print("\n=== Running Base Model Evaluation ===")
        
        # Combined list of experiment configs
        experiment_configs = []
        
        included_policies = set()
        dataset = self._get_dataset(self.args.dataset_path, "train")
        if dataset.train_partner_policies:
            train_policies = dataset.train_partner_policies
            print(f"Found {len(train_policies)} train policies: {list(train_policies.keys())}")
            
            # Filter policies if requested
            for policy_name in train_policies:
                # Skip if not in the requested list (when a list is provided)
                if policies_to_evaluate is not None and policy_name not in policies_to_evaluate:
                    print(f"Skipping train policy {policy_name} (not in requested list)")
                    continue
                    
                included_policies.add(policy_name)
                experiment_configs.append({
                    "name": f"base_model_eval_train_policy_{policy_name}",
                    "evaluation_configs": [{
                        "layout_name": self.args.layout_name,
                        "agent_id": train_policies[policy_name],
                        "horizon": self.args.horizon,
                        "partner_policy_name": policy_name,
                    }]
                })
            print(f"Created {len(included_policies)} training policy evaluations")
        
        # Early exit if no policies to evaluate
        if not experiment_configs:
            print("No policies found for evaluation")
            return []
        
        # Create a subdirectory for base model evaluations
        base_model_dir = self.exp_group_dir / "base_model_evaluations"
        base_model_dir.mkdir(exist_ok=True)

        # Get num_classes for model loading
        # num_classes = dataset.num_partner_policies if hasattr(dataset, 'num_partner_policies') else None
        num_classes = 8
        print(f"Number of classes for model: {num_classes}")
        if num_classes is None:
            raise ValueError("Dataset does not have num_partner_policies attribute. Cannot determine num_classes for model.")
        
        # Run evaluations for each policy 
        for exp_idx, exp_config in enumerate(experiment_configs):
            print(f"\n=== Evaluating Base Model on {exp_config['name']} ===")
            
            # Create experiment directory
            exp_dir = base_model_dir / exp_config["name"]
            exp_dir.mkdir(exist_ok=True)
            
            # Run evaluations
            experiment_results = []

            with managed_model_loading(self, self.args.diffusion_model_path, ema=True, num_classes=num_classes) as diffusion_model:
                eval_configs = exp_config.get("evaluation_configs", [])
                
                for eval_config in eval_configs:
                    if not eval_config:
                        continue
                    result = self.evaluate(eval_config, diffusion_model)
                    # Mark as base model evaluation
                    result['is_base_model'] = True  
                    experiment_results.append(result)
            
            
            # Save results for this experiment
            self.all_experiment_results.append({
                "name": exp_config["name"],
                "config": exp_config,
                "results": experiment_results,
                "model_path": self.args.diffusion_model_path,
                "is_base_model": True
            })
            
            # Save intermediate results
            with open(base_model_dir / f"results_through_exp_{exp_idx}.pkl", "wb") as f:
                pickle.dump(self.all_experiment_results, f)

        # Save overall results
        results_file = base_model_dir / "base_model_evaluation_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(self.all_experiment_results, f)
        
        # Generate plots
        self.plot_base_model_results()
        
        print(f"\nBase model evaluation complete. Results saved to {results_file}")
        return self.all_experiment_results
    
    def plot_horizon_evaluation_results(self, df, output_dir):
        """Plot results for horizon evaluations."""
        plt.figure(figsize=(12, 7))

        x_values = df['experiment_horizon_val'].to_numpy()
        agent0_mean = df['agent0_mean_reward'].to_numpy()
        agent0_std = df['agent0_std_reward'].to_numpy()
        agent1_mean = df['agent1_mean_reward'].to_numpy()
        agent1_std = df['agent1_std_reward'].to_numpy()
        team_mean = df['team_mean_reward'].to_numpy()
        team_std = df['team_std_reward'].to_numpy()

        plt.plot(x_values, agent0_mean, label='Agent 0 (Diffusion)', marker='o')
        plt.fill_between(x_values, 
                        agent0_mean - agent0_std, 
                        agent0_mean + agent0_std, 
                        alpha=0.2)
        
        plt.plot(x_values, agent1_mean, label='Agent 1 (Partner)', marker='o')
        plt.fill_between(x_values, 
                        agent1_mean - agent1_std, 
                        agent1_mean + agent1_std, 
                        alpha=0.2)
                        
        plt.plot(x_values, team_mean, label='Team Reward', marker='s', linestyle='--')
        plt.fill_between(x_values,
                            team_mean - team_std,
                            team_mean + team_std,
                            alpha=0.2)

        plt.xlabel('Planning Horizon Value')
        plt.ylabel('Mean Reward')
        plt.title('Horizon Evaluation Results')
        plt.legend()
        plt.grid(alpha=0.3)
        
        output_path = os.path.join(output_dir, "horizon_evaluation_results.png")
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        plt.close()

    def plot_episode_evaluation_results(self, df, output_dir):
        """Plot results for episode-based evaluations."""
        if "training_episode_idx" not in df.columns:
            print("DataFrame does not contain 'training_episode_idx' column. Skipping episode plots.")
            return
        
        for policy_name, group in df.groupby('experiment_partner_policy_val'):
            plt.figure(figsize=(12, 7))

            # Sort by episode index
            group = group.sort_values('training_episode_idx')

            # Convert pandas Series to numpy arrays
            x = group['training_episode_idx'].to_numpy()
            agent0_mean = group['agent0_mean_reward'].to_numpy()
            agent0_std = group['agent0_std_reward'].to_numpy()
            agent1_mean = group['agent1_mean_reward'].to_numpy()
            agent1_std = group['agent1_std_reward'].to_numpy()
            team_mean = group['team_mean_reward'].to_numpy()
            team_std = group['team_std_reward'].to_numpy()
            
            # Plot agent rewards
            plt.plot(x, agent0_mean, marker='o', label='Agent 0 (Diffusion)', linewidth=2)
            plt.fill_between(x, agent0_mean - agent0_std, agent0_mean + agent0_std, alpha=0.2)
            
            plt.plot(x, agent1_mean, marker='s', label='Agent 1 (Partner)', linewidth=2)
            plt.fill_between(x, agent1_mean - agent1_std, agent1_mean + agent1_std, alpha=0.2)
            
            # Plot team rewards
            plt.plot(x, team_mean, marker='^', label='Team Reward', linestyle='--', linewidth=2)
            plt.fill_between(x, team_mean - team_std, team_mean + team_std, alpha=0.2)
            
            plt.title(f'Performance vs. Training Episode for Policy: {policy_name}')
            plt.xlabel('Training Episode Index')
            plt.ylabel('Reward')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Save the policy-specific plot
            plt.savefig(os.path.join(output_dir, f"policy_{policy_name}_episodes.png"))
            plt.close()
        
        # Create comparative plot across policies
        plt.figure(figsize=(14, 8))
        markers = ['o', 's', '^', 'D', '*', 'x']
        colors = plt.cm.tab10.colors
        
        for i, (policy_name, group) in enumerate(df.groupby('experiment_partner_policy_val')):
            # Sort by episode
            group = group.sort_values('training_episode_idx')
            
            # Convert to numpy arrays
            x = group['training_episode_idx'].to_numpy()
            y = group['team_mean_reward'].to_numpy()
            std = group['team_std_reward'].to_numpy()
            
            plt.plot(x, y, 
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)],
                    label=f'Policy: {policy_name}',
                    linewidth=2)
            plt.fill_between(x, 
                            y - std,  
                            y + std,  
                            color=colors[i % len(colors)],
                            alpha=0.2)  
        
        plt.title('Team Reward vs. Training Episode Across Policies')
        plt.xlabel('Training Episode Index')
        plt.ylabel('Team Reward')
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Save overall comparison
        plt.savefig(os.path.join(output_dir, "all_policies_episode_comparison.png"))
        plt.close()
        
    def plot_results(self):
        """Generate plots from experiment results, focusing only on training policies."""
        # Create plot directory
        plot_dir = self.exp_group_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Extract results for plotting, filtering for training policies only
        plot_results = []
        for exp in self.all_experiment_results:

            if "results" in exp and exp["results"]:
                config = exp.get("config", {})
                cl_params = config.get("concept_learning_params", {})
                
                milestone_name = cl_params.get("milestone_name", "")
                target_episode_idx = cl_params.get("target_episode_idx", None)
                
                for res in exp["results"]:
                    # Create a copy to avoid modifying original
                    res_copy = res.copy()
                    
                    # Add metadata for plotting
                    res_copy["milestone_name"] = milestone_name
                    if "training_episode_idx" not in res_copy and target_episode_idx is not None:
                        res_copy["training_episode_idx"] = target_episode_idx
                    
                    res_copy["is_training_policy"] = True
                    
                    plot_results.append(res_copy)
        
        if plot_results:
            # Convert to DataFrame and determine plot type
            df = pd.DataFrame(plot_results)
            
            print(f"Plotting results for {len(df)} training policy evaluations")
            
            if "experiment_horizon_val" in df.columns and len(df["experiment_horizon_val"].unique()) > 1:
                # Horizon experiment
                self.plot_horizon_evaluation_results(df, plot_dir)
                print(f"Generated horizon plots for training policies in {plot_dir}")
                
            if "training_episode_idx" in df.columns:
                # Episode experiment 
                self.plot_episode_evaluation_results(df, plot_dir)
                print(f"Generated episode plots for training policies in {plot_dir}")

            
        else:
            print("No training policy evaluation results to plot")
    def plot_base_model_results(self):
        """Generate plots specifically for base model evaluation results."""
        # Create plot directory
        plot_dir = self.exp_group_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Extract base model evaluation results
        base_results = []
        for exp in self.all_experiment_results:
            if exp.get("is_base_model", False):
                for res in exp.get("results", []):
                    # Add the policy name from the experiment name for better readability
                    policy_name = exp["name"].replace("base_model_eval_train_policy_", "")
                    res_copy = res.copy()
                    res_copy["policy_name"] = policy_name
                    base_results.append(res_copy)
        
        if not base_results:
            print("No base model evaluation results to plot")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(base_results)
        
        # Create summary bar chart
        plt.figure(figsize=(14, 8))
        
        df = df.sort_values('team_mean_reward', ascending=False)
        
        # Extract policy names and metrics
        policies = df['experiment_partner_policy_val'].tolist()
        agent0_means = df['agent0_mean_reward'].tolist()
        agent0_stds = df['agent0_std_reward'].tolist()
        agent1_means = df['agent1_mean_reward'].tolist()
        agent1_stds = df['agent1_std_reward'].tolist()
        team_means = df['team_mean_reward'].tolist()
        team_stds = df['team_std_reward'].tolist()
        
        # Set up bar positions
        bar_width = 0.25
        r1 = np.arange(len(policies))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create grouped bar chart
        plt.bar(r1, agent0_means, yerr=agent0_stds, width=bar_width, label='Agent 0 (Diffusion)', capsize=7, color='skyblue')
        plt.bar(r2, agent1_means, yerr=agent1_stds, width=bar_width, label='Agent 1 (Partner)', capsize=7, color='lightgreen')
        plt.bar(r3, team_means, yerr=team_stds, width=bar_width, label='Team Total', capsize=7, color='coral')
        
        # Add labels and title
        plt.xlabel('Partner Policy')
        plt.ylabel('Mean Reward')
        plt.title('Base Model Performance Across Training Policies')
        plt.xticks([r + bar_width for r in range(len(policies))], policies, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(plot_dir, "base_model_performance.png")
        plt.savefig(output_path)
        print(f"Base model performance plot saved to {output_path}")
        plt.close()
    def plot_embedding_changes(self):
        plot_dir = self.exp_group_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Convert embedding changes to DataFrame for better plotting
        if self.embedding_changes:
            embedding_change_data = []
            for run_idx, (pid, delta) in enumerate(self.embedding_changes):
                embedding_change_data.append({
                    'policy_id': pid,
                    'run_index': run_idx,
                    'l2_shift': delta
                })
            
            if embedding_change_data:
                df_changes = pd.DataFrame(embedding_change_data)
                
                plt.figure(figsize=(14, 8))
                for pid in df_changes['policy_id'].unique():
                    policy_data = df_changes[df_changes['policy_id'] == pid]
                    # Convert to numpy arrays before plotting
                    plt.plot(policy_data['run_index'].values, policy_data['l2_shift'].values, 
                            marker='o', linestyle='-', label=f'Policy {pid}')
                
                plt.title("Embedding L2-shift per policy (multiple runs)")
                plt.xlabel("Run Index")
                plt.ylabel("L2 shift")
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir/"embedding_shifts.png")
                plt.close()
                print(f"Saved embedding shifts plot with {len(df_changes['policy_id'].unique())} policies")
            else:
                print("No embedding change data to plot")

        # Convert grad norms to DataFrame for better plotting
        if self.embedding_grad_norms:
            grad_norm_data = []
            for pid, norms in self.embedding_grad_norms.items():
                print(f"Processing grad norms for policy {pid} with {len(norms)} steps")
                for step_idx, norm_value in enumerate(norms):
                    grad_norm_data.append({
                        'policy_id': pid,
                        'training_step': step_idx,
                        'grad_norm': norm_value
                    })
            
            if grad_norm_data:
                df_grad = pd.DataFrame(grad_norm_data)
                
                plt.figure(figsize=(14, 8))
                for pid in df_grad['policy_id'].unique():
                    policy_data = df_grad[df_grad['policy_id'] == pid]
                    # Convert to numpy arrays before plotting
                    plt.plot(policy_data['training_step'].values, policy_data['grad_norm'].values, 
                            label=f"policy {pid}", marker='o', markersize=2)
                
                plt.xlabel("Training step")
                plt.ylabel("Grad-norm of embedding row")
                plt.title("Embedding-row grad norm vs step")
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir/"embedding_grad_norms.png")
                plt.close()
                print(f"Saved grad norms plot with {len(df_grad['policy_id'].unique())} policies")
            else:
                print("No grad norm data to plot")
        
        # Convert loss history to DataFrame for better plotting
        if self.embedding_loss_history:
            loss_data = []
            for pid, losses in self.embedding_loss_history.items():
                print(f"Processing loss history for policy {pid} with {len(losses)} steps")
                for step_idx, loss_value in enumerate(losses):
                    loss_data.append({
                        'policy_id': pid,
                        'training_step': step_idx,
                        'loss': loss_value
                    })
            
            if loss_data:
                df_loss = pd.DataFrame(loss_data)
                
                plt.figure(figsize=(14, 8))
                for pid in df_loss['policy_id'].unique():
                    policy_data = df_loss[df_loss['policy_id'] == pid]
                    # Convert to numpy arrays before plotting
                    plt.plot(policy_data['training_step'].values, policy_data['loss'].values, 
                            label=f"policy {pid}", marker='o', markersize=2)
                
                plt.xlabel("Training step")
                plt.ylabel("Loss")
                plt.title("Concept‐learning loss vs. step")
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir/"concept_loss_curves.png")
                plt.close()
                print(f"Saved loss curves plot with {len(df_loss['policy_id'].unique())} policies")
            else:
                print("No loss history data to plot")
        

def parse_args(args, parser):
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--results_dir', type=str, default='./overcooked_results', help='Directory to save results and checkpoints')
    parser.add_argument('--basedir', type=str, default='./experiment_results', help='Directory to save results and checkpoints')
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
    parser.add_argument('--resume_checkpoint_path', type=str, required=False, default=None, help='Path to a .pt checkpoint file to resume training from.')
    
    # overcooked evaluation
    parser.add_argument("--diffusion_model_path", type=str, required=True, help="Path to the diffusion model directory")
    parser.add_argument("--dataset", type=str, default="overcooked", help="Dataset name")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--agent_id", type=int, default=0, help="Agent ID for conditioning")
    parser.add_argument("--max_steps", type=int, default=400, help="Maximum steps per episode")
    parser.add_argument("--run_dir", type=str, default="eval_run", help="Directory for evaluation run")
    parser.add_argument("--idm_path", type=str, required=True, help="Path to the diffusion model directory")
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
    parser.add_argument('--num_concept_runs', type=int, required=True,
                    help='Number of runs (total times to iterate over data) for each concept learning experiment')
    parser.add_argument('--test_concept_train_steps', type=int, required=True,
                    help='Number of training steps to take for test concept learning')
    parser.add_argument('--exp_group_name', type=str, default=None, 
                    help='Name for experiment group folder')
    
    all_args = parser.parse_known_args(args)[0]

    

    return all_args

if __name__ == "__main__":
    parser = get_config()
    args = sys.argv[1:]
    args = parse_args(args, parser)

    #overrde episode len
    args.episode_length = 400
    runner = ExperimentRunner(args, num_concepts=2)
    # runner.run_cl_experiments_with_context_managers()

    runner.run_base_model_evaluation(policies_to_evaluate={
        "sp1_final",
        # "sp2_final",
        # "sp3_final",
        # "sp4_final",
        # "sp5_final",
    })
    runner.plot_results()
    runner.plot_embedding_changes()

    