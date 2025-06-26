import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        return self.relu(out + residual)

class RewardPredictor(nn.Module):
    def __init__(self, input_channels=26, hidden_channels=64, mlp_hidden=128):
        super().__init__()
        # Initial 3x3 conv
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)

        # 3 residual 3x3 conv layers
        self.res_blocks = nn.Sequential(
            ResidualConvBlock(hidden_channels),
            ResidualConvBlock(hidden_channels),
            ResidualConvBlock(hidden_channels),
        )

        # MLP after global mean pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, 1) # Output a single reward value
        )
        self.criterion = nn.MSELoss()

    def forward(self, obs, next_obs):
        assert obs.min() >= -1.0 and obs.max() <= 1.0, "Observations must be normalized to [-1, 1]"
        assert next_obs.min() >= -1.0 and next_obs.max() <= 1.0, "Next observations must be normalized to [-1, 1]"
        # Input: obs and next_obs with shape (B, H, W, C)
        x = torch.cat([obs, next_obs], dim=-1)  # (B, H, W, 2C)
        x = x.permute(0, 3, 1, 2)  # (B, 2C, H, W)
        x = self.conv_in(x)
        x = self.res_blocks(x)
        # Global mean pooling across spatial dimensions
        x = x.mean(dim=[2, 3])  # (B, hidden_channels)
        return self.mlp(x).squeeze(-1) # (B,)
    
    def loss(self, obs, next_obs, reward):
        assert obs.min() >= -1.0 and obs.max() <= 1.0, "Observations must be normalized to [-1, 1]"
        assert next_obs.min() >= -1.0 and next_obs.max() <= 1.0, "Next observations must be normalized to [-1, 1]"  
        # assert reward.min() >= 0.0 and reward.max() <= 1.0, "Reward must be normalized to [0, 1]"
        
        pred_reward = self(obs, next_obs)
        loss = self.criterion(pred_reward, reward)
        return loss