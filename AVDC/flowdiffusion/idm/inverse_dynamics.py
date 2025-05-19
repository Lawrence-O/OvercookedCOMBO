import torch
import torch.nn as nn
import torch.nn.functional as F


PLAYER_OBS_CHANNELS = list(range(10)) + [22, 23, 24] # Channels for player observations, you can change it
INPUT_CHANNELS = len(PLAYER_OBS_CHANNELS) * 2 # Number of input channels for the model

def get_player_locations(obs):
    """
    Args:
        obs: Tensor of shape (Batch, width, height, channels)
             (assumes player location is at obs[..., 0] where one pixel = 1)
    
    Returns:
        player_locations: Tensor of shape (Batch, 2) where each row is (x, y)
    """
    # Extract the player channel (shape: Batch, width, height)
    player_channel = obs[..., 0]
    
    # Flatten spatial dims (Batch, width * height)
    flat_obs = player_channel.flatten(start_dim=1)
    
    # Get index of "1" in the flattened grid (shape: Batch,)
    player_indices = torch.argmax(flat_obs, dim=1)
    
    # Convert flat index back to (x, y) coordinates
    width = obs.shape[2]
    player_locations = torch.stack([
        player_indices // width,  # x-coordinate (row)
        player_indices % width    # y-coordinate (column)
    ], dim=1)
    
    return player_locations

class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)

class InverseDynamicsModel(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, hidden_channels=64, mlp_hidden=128, num_actions=1):
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
            nn.Linear(mlp_hidden, num_actions)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, obs, next_obs):
        # Input: obs and next_obs with shape (B, H, W, C)
        
        obs = obs[..., PLAYER_OBS_CHANNELS]
        next_obs = next_obs[..., PLAYER_OBS_CHANNELS]
        x = torch.cat([obs, next_obs], dim=-1)  # (B, H, W, 2C)
        x = x.permute(0, 3, 1, 2)  # (B, 2C, H, W)

        x = self.conv_in(x)
        x = self.res_blocks(x)

        # Global mean pooling across spatial dimensions
        x = x.mean(dim=[2, 3])  # (B, hidden_channels)

        return self.mlp(x)  # (B, num_actions)
    
    def loss(self, obs, next_obs, action):  
        target = action[:, 0]
        logits = self(obs, next_obs)
        # Use binary cross-entropy loss for action prediction
        loss = self.criterion(logits, target)  # Assuming action is of shape (B, 1)
        player_locations, next_player_locations = get_player_locations(obs), get_player_locations(next_obs)
        
        # Accuracy computation
        preds = logits.argmax(dim=1)  # (B,)
        correct = (preds == target)
        accuracy = correct.sum().item() / target.size(0)
        
        # --- NEW: Track incorrect predictions ---
        incorrect_mask = ~correct  # Mask where predictions are wrong
        incorrect_indices = incorrect_mask.nonzero(as_tuple=True)[0]  # Indices of incorrect predictions
        
        # Store (prediction, target) pairs for incorrect cases
        incorrect_cases = []
        if len(incorrect_indices) > 0:
            incorrect_cases = list(zip(
                preds[incorrect_indices].tolist(),   # Predicted class
                target[incorrect_indices].tolist(),  # True class
                player_locations[incorrect_indices].tolist(),  # Player location
                next_player_locations[incorrect_indices].tolist()  # Next player location
            ))

        
        return loss, {
            "accuracy": accuracy,
            "incorrect": incorrect_cases  # List of (pred, target) for wrong cases
        }