from pathlib import Path
import torch
from torch.utils.data import DataLoader
from reward_dataset import RewardPredictorDataset
from reward_model import RewardPredictor 
import os
import wandb
import tqdm
from itertools import cycle, islice
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt
import datetime as datatime


class RewardTrainer:
    def __init__(
        self,
        args,
        run_name="reward_model_training" + datatime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        batch_size=32,
        lr=1e-4,
        log_wandb=False,
        save_dir="reward_model_checkpoints",
        train_steps=1000000,
        seed=42,
    ):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C = 26  # Number of channels in the observation
        self.model = RewardPredictor(input_channels=self.C*2, # Obs + Next Obs Channels 
                                     hidden_channels=128, 
                                     mlp_hidden=256).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._load_dataset()
        self.log_wandb = log_wandb
        if log_wandb:
            wandb.init(project="reward_model_training", entity="social-rl", config={
                "learning_rate": lr,
                "batch_size": batch_size,
                "input_channels": self.C*2, 
                "hidden_channels": 128,
                "mlp_hidden": 256,
            })
        self.steps = 0
        self.train_steps = train_steps
        self.validation_samples = 100
        self.save_every = 1000
        self.eval_every = 1000

        self._set_seed(seed)
    
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    
    def _load_dataset(self):
        train_dataset = RewardPredictorDataset(self.args, split="train")
        valid_dataset = RewardPredictorDataset(self.args, split="test") 
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.train_loader = cycle(self.train_loader)
        self.val_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.val_loader = cycle(self.val_loader)
        self.train_sample_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.train_sample_loader = cycle(self.train_sample_loader)

    def train_step(self):
        obs, next_obs, reward = next(self.train_loader)
        obs, next_obs, reward = obs.to(self.device), next_obs.to(self.device), reward.to(self.device)
        reward = reward.squeeze(-1)  # Ensure reward is of shape (B,)
        loss = self.model.loss(obs, next_obs, reward)
        if self.log_wandb:
            wandb.log({"train/loss": loss.item()}, step=self.steps)
        return loss
    
    def train(self):
        self.model.train()
        with tqdm.tqdm(initial=self.steps, total=self.train_steps) as pbar:
            while self.steps < self.train_steps:
                loss = self.train_step()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                self.opt.zero_grad()
                pbar.set_description(f"Loss: {loss.item():.4E}")
                pbar.update(1)
                
                if self.steps % self.save_every == 0:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_dir, f"reward_predictor_step{self.steps}.pt"),
                    )
                if self.steps % self.eval_every == 0 and self.val_loader is not None:
                    self.valid_step()
                    self.train_plot_step()
                self.steps += 1
    
    @torch.no_grad()
    def valid_step(self):
        total_loss = 0.0
        all_preds, all_targets = [], []
        for obs, next_obs, reward in islice(self.val_loader, self.validation_samples):
            obs, next_obs, reward = obs.to(self.device), next_obs.to(self.device), reward.to(self.device)
            reward = reward.squeeze(-1)
            # Evaluate the model
            pred = self.model(obs, next_obs).cpu().numpy()
            target = reward.cpu().numpy()
            all_preds.append(pred)
            all_targets.append(target)

            # Calculate loss
            loss = self.model.loss(obs, next_obs, reward)
            total_loss += loss.item()
        avg_loss = total_loss / self.validation_samples
        if self.log_wandb:
            wandb.log({"validation/loss": avg_loss}, step=self.steps)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        print("Validation step completed.")
        print("Pred mean/std:", np.mean(all_preds), np.std(all_preds))
        print("Target mean/std:", np.mean(all_targets), np.std(all_targets))
        self.plot_valid(all_targets, all_preds)
        print(f"Validation Loss: {avg_loss:.4f}")
    
    def plot_valid(self, all_targets, all_preds):
        # Plot Predicted vs True Rewards
        pred_vs_true_path = self.save_dir / f"pred_vs_true_{self.steps}.png"
        plt.figure(figsize=(10,5))
        plt.scatter(all_targets, all_preds, alpha=0.5)
        plt.scatter(all_targets, all_targets, color='blue', alpha=0.5)
        plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
        plt.xlabel("True Reward")
        plt.ylabel("Predicted Reward")
        plt.title("Predicted vs. True Reward")
        plt.tight_layout()
        plt.savefig(pred_vs_true_path)
        if self.log_wandb:
            wandb.log({"validation/pred_vs_true": wandb.Image(str(pred_vs_true_path), caption="Predicted vs True Rewards")}, step=self.steps)
        plt.close()

        # Errors vs True Rewards
        errors_vs_true_path = self.save_dir / f"errors_vs_true_{self.steps}.png"
        errors = all_preds - all_targets
        plt.figure(figsize=(10, 5))
        plt.scatter(all_targets, errors)
        plt.axhline(0, color='r', linestyle='--')  # zero-error line
        plt.xlabel('True Rewards')
        plt.ylabel('Prediction Error (Predicted - True)')
        plt.title('Prediction Errors vs True Rewards')
        plt.tight_layout()
        plt.savefig(errors_vs_true_path)
        if self.log_wandb:
            wandb.log({"validation/errors_vs_true": wandb.Image(str(errors_vs_true_path), 
                                                                caption="Errors vs True Rewards")},
                                                                step=self.steps)
        plt.close()

        print(f"Validation plots saved to {self.save_dir}")
    @torch.no_grad()
    def train_plot_step(self):
        # TODO: REMOVE THIS
        num_samples = self.validation_samples  
        all_preds, all_targets = [], []
        for obs, next_obs, reward in islice(self.train_loader, num_samples):
            obs, next_obs, reward = obs.to(self.device), next_obs.to(self.device), reward.to(self.device)
            reward = reward.squeeze(-1)
            pred = self.model(obs, next_obs).clone().cpu().numpy()
            target = reward.cpu().numpy()
            all_preds.append(pred)
            all_targets.append(target)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        self.plot_train(all_targets, all_preds)

    def plot_train(self, all_targets, all_preds):
        # Plot Predicted vs True Rewards for training set
        pred_vs_true_path = self.save_dir / f"train_pred_vs_true_{self.steps}.png"
        plt.figure(figsize=(10,5))
        plt.scatter(all_targets, all_preds, alpha=0.5)
        plt.scatter(all_targets, all_targets, color='blue', alpha=0.5)
        plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
        plt.xlabel("True Reward (Train)")
        plt.ylabel("Predicted Reward (Train)")
        plt.title("Train: Predicted vs. True Reward")
        plt.tight_layout()
        plt.savefig(pred_vs_true_path)
        if self.log_wandb:
            wandb.log({"train/pred_vs_true": wandb.Image(str(pred_vs_true_path), caption="Train Predicted vs True Rewards")}, step=self.steps)
        plt.close()

        # Errors vs True Rewards for training set
        errors_vs_true_path = self.save_dir / f"train_errors_vs_true_{self.steps}.png"
        errors = all_preds - all_targets
        plt.figure(figsize=(10, 5))
        plt.scatter(all_targets, errors)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel('True Rewards (Train)')
        plt.ylabel('Prediction Error (Predicted - True)')
        plt.title('Train: Prediction Errors vs True Rewards')
        plt.tight_layout()
        plt.savefig(errors_vs_true_path)
        if self.log_wandb:
            wandb.log({"train/errors_vs_true": wandb.Image(str(errors_vs_true_path), caption="Train Errors vs True Rewards")}, step=self.steps)
        plt.close()

        print(f"Train plots saved to {self.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Reward Predictor")
    # For OvercookedSequenceDataset / HDF5Dataset
    parser.add_argument('--max_path_length', type=int, default=401, help='Maximum path length in episodes (for dataset indexing)')
    parser.add_argument('--episode_length', type=int, default=401, help='Full episode length (for HDF5Dataset if used directly)')
    parser.add_argument('--chunk_length', type=int, default=None, help='Chunk length for HDF5Dataset (defaults to horizon if None, set via dataset_constructor_args)')
    parser.add_argument('--use_padding', type=bool, default=True, help='Whether to use padding for shorter sequences in dataset')
    
    # For RewardPredictorDataset
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file (HDF5 format)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    
    trainer = RewardTrainer(args, 
                            batch_size=args.batch_size, 
                            lr=1e-4, 
                            log_wandb=False, 
                            save_dir="reward_model_checkpoints", 
                            train_steps=50000, 
                            seed=42)
    trainer.train()
    
    