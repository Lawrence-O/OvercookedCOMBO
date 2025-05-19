import torch
from torch.utils.data import Subset
from pathlib import Path
import argparse
import numpy as np

# Workspace imports
from AVDC.flowdiffusion.goal_diffusion import GoalGaussianDiffusion, OvercookedEnvTrainer
from AVDC.flowdiffusion.unet import UnetOvercooked 
from AVDC.flowdiffusion.overcooked_dataset import OvercookedSequenceDataset


class OvercookedTrainer:
    def __init__(self, args, device=None):
        self.args = args
        if device is None:
            self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu")
        else:
            self.device=device
        print(f"Using device: {self.device}")

        self.results_folder = Path(args.results_dir)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.tokenizer = None
        self.text_encoder = None
        self.text_embed_dim = None

        self.horizon  = args.horizon
        
        dataset_args = argparse.Namespace(
            dataset_path = args.dataset_path,
            horizon=self.horizon,
            max_path_length=getattr(args, 'max_path_length', 401), 
            episode_length=getattr(args, 'episode_length', 401), 
            chunk_length=getattr(args, 'chunk_length', self.horizon), 
            use_padding=getattr(args, 'use_padding', True),
        )

        self.dataset = OvercookedSequenceDataset(args=dataset_args)

        # Create train/validation split
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        
        split_idx = int(np.floor(args.valid_ratio * dataset_size))
        train_indices, valid_indices = indices[split_idx:], indices[:split_idx]
        
        self.train_dataset = Subset(self.dataset, train_indices)
        self.valid_dataset = Subset(self.dataset, valid_indices)


        self.observation_dim = self.dataset.observation_dim
        self.diffusion = None
        self.trainer = None
        self.unet = None

        self.init_diffusion_trainer()
    
    def split_dataset(self, dataset):
        raise NotImplementedError()
        
    def init_diffusion_trainer(self):
        H,W,C = self.observation_dim
        self.unet = UnetOvercooked(
            horizon=self.horizon,
            obs_dim=self.observation_dim,
        ).to(self.device)
        self.diffusion = GoalGaussianDiffusion(
            model=self.unet,
            channels=C * 32, #TODO: Look into this Channels * Batch Size
            image_size=(H,W),
            timesteps=1 if self.args.debug else 1000,
            sampling_timesteps=1 if self.args.debug else 100,
            loss_type="l2",
            objective="pred_v",
            beta_schedule="cosine",
            min_snr_loss_weight=True,
        ).to(self.device)
        self.trainer = OvercookedEnvTrainer(
                diffusion_model=self.diffusion,
                channels=C,
                train_set=self.train_dataset,
                valid_set=self.valid_dataset,
                train_lr=1e-4,
                train_num_steps = 200000,
                save_and_sample_every = 2 if self.args.debug else 1000,
                ema_update_every = 10,
                ema_decay = 0.999,
                train_batch_size = 1 if self.args.debug else self.args.train_batch_size,
                valid_batch_size = 32,
                gradient_accumulate_every = 1,
                num_samples=self.args.num_validation_samples, 
                results_folder = str(self.results_folder),
                fp16 =True,
                amp=True,
                save_milestone=self.args.save_milestone,
                cond_drop_chance=getattr(self.args, 'cond_drop_prob', 0),
                split_batches=getattr(self.args, 'split_batches', True),
                debug=self.args.debug,
            )
    def train(self):
        self.trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Overcooked Diffusion Model")

    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Fraction of data to use for validation (default: 0.1)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--results_dir', type=str, default='./overcooked_results', help='Directory to save results and checkpoints')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (smaller dataset, faster training)')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Overcooked HDF5 dataset')
    parser.add_argument('--horizon', type=int, default=32, help='Sequence horizon for trajectories')
    parser.add_argument('--save_milestone', type=bool, default=True, help='Save milestones with step number in filename') # Or action='store_true'

    # For OvercookedSequenceDataset / HDF5Dataset
    parser.add_argument('--max_path_length', type=int, default=401, help='Maximum path length in episodes (for dataset indexing)')
    parser.add_argument('--episode_length', type=int, default=401, help='Full episode length (for HDF5Dataset if used directly)')
    parser.add_argument('--chunk_length', type=int, default=None, help='Chunk length for HDF5Dataset (defaults to horizon if None, set via dataset_constructor_args)')
    parser.add_argument('--use_padding', type=bool, default=True, help='Whether to use padding for shorter sequences in dataset')


    # For GoalGaussianDiffusion
    parser.add_argument('--timesteps', type=int, default=400, help='Number of diffusion timesteps for training (if not debug)')
    parser.add_argument('--sampling_timesteps', type=int, default=10, help='Number of timesteps for DDIM sampling (if not debug)')

    # For OvercookedEnvTrainer 
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size (if not debug)')
    parser.add_argument('--num_validation_samples', type=int, default=4, help='Number of samples to generate during validation step')
    parser.add_argument('--save_and_sample_every', type=int, default=1000, help='Frequency to save checkpoints and generate samples (if not debug)')
    parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Probability of dropping condition for CFG during training')
    parser.add_argument('--split_batches', type=bool, default=True, help='Whether to split batches for Accelerator')

    config = parser.parse_args()

    if config.debug:
        print("--- RUNNING IN DEBUG MODE ---")
    overcooked_trainer_main = OvercookedTrainer(args=config)
    overcooked_trainer_main.train()         