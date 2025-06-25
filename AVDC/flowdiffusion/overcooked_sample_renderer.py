
from overcooked_env.visualization.state_visualizer import StateVisualizer
import pygame
import os
import numpy as np
from overcooked_env.static import GRAPHICS_DIR, FONTS_DIR
from overcooked_env.visualization.pygame_utils import scale_surface_by_factor
from moviepy.editor import ImageSequenceClip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
import math
import torch


class OvercookedSampleRenderer:
    """ Assume Features are (height, width, channels) where channels = 26; Uses Normalized Images;
    Inspired by State Visualizor
    """

    # See lossless_state_encoding in overcooked_mdp.py
    FEATURE_CHANNEL_MAP = [
    "player_0_loc", "player_1_loc", 
    "player_0_orientation_0", "player_0_orientation_1", "player_0_orientation_2", "player_0_orientation_3",
    "player_1_orientation_0", "player_1_orientation_1", "player_1_orientation_2", "player_1_orientation_3",
    "pot_loc", "counter_loc", "onion_disp_loc", "tomato_disp_loc", "dish_disp_loc", "serve_loc",
    "onions_in_pot", "tomatoes_in_pot",
    "onions_in_soup", "tomatoes_in_soup", 
    "soup_cook_time_remaining", "soup_done",
    "dishes", "onions", "tomatoes",
    "urgency"
]
    CHANNEL_FEATURE_MAP = {name: i for i, name in enumerate(FEATURE_CHANNEL_MAP)}

    TERRAIN_IMG = StateVisualizer.TERRAINS_IMG
    OBJECTS_IMG = StateVisualizer.OBJECTS_IMG
    SOUPS_IMG = StateVisualizer.SOUPS_IMG
    CHEFS_IMG = StateVisualizer.CHEFS_IMG
    ARROW_IMG = pygame.image.load(os.path.join(GRAPHICS_DIR, "arrow.png"))
    INTERACT_IMG = pygame.image.load(os.path.join(GRAPHICS_DIR, "interact.png"))

    TILE_TO_FRAME_NAME = StateVisualizer.TILE_TO_FRAME_NAME
    UNSCALED_TILE_SIZE = StateVisualizer.UNSCALED_TILE_SIZE

    MAX_COOK_TIME = 20

  
    def __init__(self, tile_size=15):
        self.tile_size = tile_size
        pygame.init()
        self.cooking_timer_font_size = 10
        self.cooking_timer_font_color = (255, 0, 0)
        roboto_path = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")
        self.cooking_timer_font = pygame.font.Font(roboto_path, self.cooking_timer_font_size)
        
    def get_feature(self, feature_tensor, feature_name, y, x):
        """Gets the current cell value from the feature tensor"""
        if feature_name not in self.CHANNEL_FEATURE_MAP:
            return 0
        channel_idx = self.CHANNEL_FEATURE_MAP[feature_name]
        try:
            return feature_tensor[y,x,channel_idx]
        except IndexError:
            return 0
    def _position_in_unscaled_pixels(self, position):
        """
        get x and y coordinates in tiles, returns x and y coordinates in pixels
        """
        (x, y) = position
        return (self.UNSCALED_TILE_SIZE * x, self.UNSCALED_TILE_SIZE * y)

    def _position_in_scaled_pixels(self, position):
        """
        get x and y coordinates in tiles, returns x and y coordinates in pixels
        """
        (x, y) = position
        return (self.tile_size * x, self.tile_size * y)
    
    def _render_grid(self, surface, grid):
        for y_tile, row in enumerate(grid):
            for x_tile, tile in enumerate(row):
                frame_name = self.TILE_TO_FRAME_NAME.get(tile, "floor")
                self.TERRAIN_IMG.blit_on_surface(
                    surface,
                    self._position_in_unscaled_pixels((x_tile, y_tile)),
                    frame_name
                )
    @staticmethod
    def _soup_frame_name(num_onions, num_tomatoes, status):
        return "soup_%s_tomato_%i_onion_%i" % (
            status,
            num_tomatoes,
            num_onions,
        )
    def _is_player_location(self, feature_tensor, y, x, eps=1e-4):
        """Check if position contains a player"""
        return (self.get_feature(feature_tensor, "player_0_loc", y, x) >= (1 - eps) or 
                self.get_feature(feature_tensor, "player_1_loc", y, x) >= (1 - eps))
    
    def _render_cooking_timers(self, surface, obs, grid, eps=1e-4):
        grid_height, grid_width = len(grid), len(grid[0])
        for y in range(grid_height):
            for x in range(grid_width):
                if grid[y][x] == 'P':  # POT
                    soup_cook_time = self.get_feature(obs, "soup_cook_time_remaining", y, x)
                    soup_done = self.get_feature(obs, "soup_done", y, x) > 0
                    
                    if soup_cook_time > 0 or soup_done:
                        # Create timer text
                        # 20 - cook time
                        if soup_cook_time > 0:
                            timer_text = str(self.MAX_COOK_TIME - int(soup_cook_time))
                        else:
                            # When soup is done but timer is 0, still show "0"
                            timer_text = "20"
                        text_surface = self.cooking_timer_font.render(
                            timer_text, 
                            True, 
                            self.cooking_timer_font_color
                        )
                        
                        # Position the timer text centered on the pot
                        pos = self._position_in_unscaled_pixels((x, y))
                        text_pos = (
                            pos[0] + (self.UNSCALED_TILE_SIZE - text_surface.get_width()) // 2,
                            pos[1] + (self.UNSCALED_TILE_SIZE - text_surface.get_height()) // 2
                        )
                        
                        # Draw the timer text
                        surface.blit(text_surface, text_pos)
    def _render_objects(self, surface, obs, grid, eps=1e-4):
        grid_height, grid_width = len(grid), len(grid[0])
        for y in range(grid_height):
            for x in range(grid_width):
                pos = self._position_in_unscaled_pixels((x,y))

                if self._is_player_location(obs, y, x):
                    continue

                if grid[y][x] == 'P': # POT
                    onions_in_pot = self.get_feature(obs, "onions_in_pot", y, x)
                    tomatoes_in_pot = self.get_feature(obs, "tomatoes_in_pot", y, x)
                    onions_in_soup = self.get_feature(obs, "onions_in_soup", y, x)
                    tomatoes_in_soup = self.get_feature(obs, "tomatoes_in_soup", y, x)
                    soup_cook_time = self.get_feature(obs, "soup_cook_time_remaining", y, x)
                    soup_done = self.get_feature(obs, "soup_done", y, x) > 0

                    if onions_in_pot > 0 or tomatoes_in_pot > 0 or onions_in_soup > 0 or tomatoes_in_soup > 0:
                        status = "idle"
                        if soup_done or soup_cook_time > 0:
                            status = "cooked"
                    
                        num_onions = int(onions_in_pot + onions_in_soup)
                        num_tomatoes = int(tomatoes_in_pot + tomatoes_in_soup)
                        soup_frame = self._soup_frame_name(num_onions,num_tomatoes, status)
                        try:
                            self.SOUPS_IMG.blit_on_surface(surface, pos, soup_frame)
                        except KeyError:
                            print(f"Soup frame '{soup_frame}' not found")
                elif self.get_feature(obs, "dishes", y, x) >= (1 - eps):
                    self.OBJECTS_IMG.blit_on_surface(surface, pos, "dish")
                elif self.get_feature(obs, "onions", y, x) >= (1 - eps):
                    self.OBJECTS_IMG.blit_on_surface(surface, pos, "onion") 
                elif self.get_feature(obs, "tomatoes", y, x) >= (1 - eps):
                    self.OBJECTS_IMG.blit_on_surface(surface, pos, "tomato")
                    

    def _render_players(self, surface, obs, eps=1e-4):
        player_colors = ["blue", "green"]
        directions = ["NORTH", "SOUTH", "EAST", "WEST"]

        for p_idx in range(2): # only two players
            player_loc_ch = self.CHANNEL_FEATURE_MAP[f"player_{p_idx}_loc"]
            player_locs = np.argwhere(obs[:, :, player_loc_ch] >= (1-eps))

            if len(player_locs) > 0:
                player_y, player_x = player_locs[0]
                player_pos = self._position_in_unscaled_pixels((player_x, player_y))

                #Get orientation
                orientation_idx = 0
                for i in range(4):
                    if self.get_feature(obs, f"player_{p_idx}_orientation_{i}", player_y, player_x) >= (1-eps):
                        orientation_idx = i
                        break
                
                direction_name = directions[orientation_idx]
                player_color_name = player_colors[p_idx]

                held_obj_name = ""
                if self.get_feature(obs, "dishes", player_y, player_x) >= (1-eps):
                    held_obj_name = "dish"
                elif self.get_feature(obs, "onions", player_y, player_x) >= (1-eps):
                    held_obj_name = "onion"
                elif self.get_feature(obs, "tomatoes", player_y, player_x) >= (1-eps):
                    held_obj_name = "tomato"
                elif self.get_feature(obs, "soup_done", player_y, player_x) >= (1-eps):
                    onions = self.get_feature(obs, "onions_in_soup", player_y, player_x)
                    tomatoes = self.get_feature(obs, "tomatoes_in_soup", player_y, player_x)
                    if onions > 0 and tomatoes == 0:
                        held_obj_name = "soup-onion"
                    elif tomatoes > 0:
                        held_obj_name = "soup-tomato"
                
                # Render Chef
                chef_frame = direction_name
                if held_obj_name:
                    chef_frame = f"{direction_name}-{held_obj_name}"
                try:
                    self.CHEFS_IMG.blit_on_surface(surface, player_pos, chef_frame)
                except KeyError:
                    print(f"Chef frame '{chef_frame}' not found, using fallback")
                    fallbacks = list(self.CHEFS_IMG.frames_rectangles.keys())
                    if fallbacks:
                        self.CHEFS_IMG.blit_on_surface(surface, player_pos, fallbacks[0])
                
                # Render hat 
                hat_frame = f"{direction_name}-{player_color_name}hat"
                try:
                    self.CHEFS_IMG.blit_on_surface(surface, player_pos, hat_frame)
                except KeyError:
                    pass  # Skip hat if frame not found

    def render_frame(self, obs, grid, normalize=False, eps=1e-4):

        height = len(grid)
        width = len(grid[0])

        surface = pygame.Surface((width * self.UNSCALED_TILE_SIZE, height * self.UNSCALED_TILE_SIZE))
        surface.fill((155,101,0))

        if normalize:
            obs = self.normalize_obs(obs)
        else:
            obs = obs.astype(np.float32) / 255.0
        
        self._render_grid(surface, grid)
        self._render_objects(surface, obs, grid, eps)
        self._render_cooking_timers(surface, obs, grid, eps)
        self._render_players(surface,obs, eps)

        return surface
    
    def render_trajectory_frames(self, trajectory, grid, output_dir=None, normalize=False):
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "trajectory_viz_output")
        os.makedirs(output_dir, exist_ok=True)
        img_paths = []
        for i, obs in enumerate(trajectory):
            file_path = os.path.join(output_dir, f"viz_frame_{i:04d}.png")
            self.save_obs_image(obs, grid, file_path, normalize=normalize)
            img_paths.append(file_path)
        return img_paths
    
    def extract_grid_from_obs(self, obs, eps=1e-4):
        grid_height, grid_width = obs.shape[0], obs.shape[1]
        grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
        
        for y in range(grid_height):
            for x in range(grid_width):
                if self.get_feature(obs, "pot_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'P'
                elif self.get_feature(obs, "counter_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'X'
                elif self.get_feature(obs, "onion_disp_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'O'
                elif self.get_feature(obs, "tomato_disp_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'T'
                elif self.get_feature(obs, "dish_disp_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'D'
                elif self.get_feature(obs, "serve_loc", y, x) >= (1 - eps):
                    grid[y][x] = 'S'
        
        # Ensure grid has walls around the edges
        for y in range(grid_height):
            if grid[y][0] == ' ':
                grid[y][0] = 'X'
            if grid[y][grid_width-1] == ' ':
                grid[y][grid_width-1] = 'X'
                
        for x in range(grid_width):
            if grid[0][x] == ' ':
                grid[0][x] = 'X'
            if grid[grid_height-1][x] == ' ':
                grid[grid_height-1][x] = 'X'
        return grid
    
    def render_trajectory_video(self, trajectory, grid, output_dir=None, video_path=None, fps=30, scale=4, normalize=False):
        img_paths = self.render_trajectory_frames(trajectory, grid, output_dir, normalize=normalize)
        if video_path is None:
            video_path = os.path.join(output_dir, "trajectory_viz_video.mp4")
        clip = ImageSequenceClip(img_paths, fps=fps)
        clip.write_videofile(video_path, codec="libx264", fps=fps, verbose=False, logger=None)
        print(f"Video saved to {video_path}")
        return video_path
    
    def convert_flatten_map(self, flat_obs, height=8, width=5):
        """Converts a flattened feature vector to 3D Representation.
        Args:
            flat_obs : Numpy Array of Shape (1041,)
        """
        channels = len(self.FEATURE_CHANNEL_MAP)
        spatial_size = height * width * channels
        if len(flat_obs) < spatial_size:
            print(f"Warning: Flat feature vector length ({len(flat_obs)}) is smaller than " 
                f"expected spatial size ({spatial_size})")
            padded = np.zeros(spatial_size)
            padded[:len(flat_obs)] = flat_obs
            flat_obs = padded
        elif len(flat_obs) > spatial_size:
            # print(f"Note: Using first {spatial_size} elements of {len(flat_obs)}-length vector")
            flat_obs = flat_obs[:spatial_size]
    
        spatial_features = flat_obs.reshape(height, width, channels)
        return spatial_features
    
    def arg_max(self, obs):
        from einops.einops import rearrange

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)

        # Assume Obs -> [Batch, Horizon, H, W, C]
        if obs.dim() != 5:
            raise ValueError(f"Expected 5D input (B, T, H, W, C), got {obs.shape}")
        
        B, T, H, W, C = obs.shape

        # Flatten spatial dimensions [B, T, C, H*W]
        flat = rearrange(obs, "b t h w c -> b t c (h w)")

        # Get max indices along spatial dimension
        _, max_idxs = flat.max(dim=-1)  # [B, T, C]

        # Directly scatter 1.0 at max positions
        flat_mask = torch.ones_like(flat)*-1
        flat_mask.scatter_(-1, max_idxs.unsqueeze(-1), 1.0)  
        # Reshape back to original dimensions
        peaks = rearrange(flat_mask, "b t c (h w) -> b t h w c", h=H, w=W)

        return peaks.numpy()
    
    def arg_max_orientations(self, obs):
        # TODO: Could be interesting to see (assuming model doesn't learn to predict orientation with a high val)
        pass
    
    
    def normalize_obs(self, obs, threshold_value=-0.05):
        """Normalizes each channel to [0,1] and scales object channels to their max values."""
        normalized = np.zeros_like(obs)
        max_values = {
            "onions_in_pot": 3,
            "tomatoes_in_pot": 3,
            "onions_in_soup": 3, 
            "tomatoes_in_soup": 3,
            "soup_cook_time_remaining": 20
        }
        
        # Map object channels to their indices and max values
        idx_to_max = {}
        for ch_name, max_val in max_values.items():
            if ch_name in self.CHANNEL_FEATURE_MAP:
                ch_idx = self.CHANNEL_FEATURE_MAP[ch_name]
                idx_to_max[ch_idx] = max_val
        
        # Normalize each channel
        for ch_idx in range(obs.shape[-1]):
            channel_data = obs[:, :, ch_idx]
            # if ch_idx in idx_to_max:
            #     # Object channel: normalize from [0, max_val] to [0,1]
            #     max_val = idx_to_max[ch_idx]
            #     normalized_channel = channel_data / max_val
            # else:
                # Generic channel: normalize from its own [min, max] to [0,1]
            # if ch_idx not in idx_to_max:
            channel_data = np.clip(channel_data, threshold_value, None)
            input_min = channel_data.min()
            input_max = channel_data.max()
            input_range = input_max - input_min
            if np.isclose(input_range, 0):
                normalized_channel = np.zeros_like(channel_data)
            else:
                normalized_channel = (channel_data - input_min) / input_range
            normalized[:, :, ch_idx] = normalized_channel
        
        # Clip all channels to [0,1]
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Scale object channels back to [0, max_val] using threshold
        for ch_idx, max_val in idx_to_max.items():
            ch_values = normalized[:, :, ch_idx].copy()
            non_zero_mask = ch_values > 0.0
            if np.any(non_zero_mask):
                scaled_values = ch_values[non_zero_mask] * max_val
                ch_values[non_zero_mask] = np.round(scaled_values)
                # Ensure values do not exceed max_val due to floating point errors
                np.clip(ch_values, 0, max_val, out=ch_values)
                normalized[:, :, ch_idx] = ch_values
            else:
                normalized[:, :, ch_idx] = 0
        
        return normalized
    
    def unnormalize(self, obs):
        # Assume obs is in range [-1, 1]; just directly un-normalize back

        obs = np.clip(obs, -1.0, 1.0)
        unnorm_obs = np.zeros_like(obs, dtype=np.float32)
        max_values = {
            "onions_in_pot": 3.0,
            "tomatoes_in_pot": 3.0,
            "onions_in_soup": 3.0,
            "tomatoes_in_soup": 3.0,
            "soup_cook_time_remaining": 20.0
        }

        idx_to_max = {}
        for ch_name, max_val in max_values.items():
            if ch_name in self.CHANNEL_FEATURE_MAP:
                ch_idx = self.CHANNEL_FEATURE_MAP[ch_name]
                idx_to_max[ch_idx] = max_val

        for ch_idx in range(obs.shape[-1]):
            channel_data = obs[:, :, ch_idx]
            if ch_idx in idx_to_max:
                max_val = idx_to_max[ch_idx]
                # Rescale [-1, 1] to [0, 1];  Rescale [0, 1] to [0, max_val]
                channel_data = ((channel_data + 1.0) / 2.0 )* max_val
            else:
                # Rescale from [-1, 1] to [0, 1]
                channel_data = (channel_data + 1.0) / 2.0
            unnorm_obs[...,ch_idx] = channel_data
        
        return unnorm_obs

    def save_obs_image(self, obs, grid, file_path, scale=4, normalize=False):
        surface = self.render_frame(obs, grid, normalize=normalize)
        if scale != 1:
            surface = scale_surface_by_factor(surface, scale)
        pygame.image.save(surface, file_path)
        return file_path
    
    def visualize_all_channels(self, obs, output_dir=None, height=8, width=5):        
        if obs.ndim != 3:
            raise ValueError(f"Expected obs to be 3D (H, W, C), got shape {obs.shape}")
        
        H, W, C = obs.shape
        if C == 0:
            print("Warning: Observation has 0 channels, nothing to visualize.")
            return
        # Create a figure with subplots for each channel

        cols = math.ceil(math.sqrt(C))
        rows = math.ceil(C / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False) # Adjust figsize as needed
        axes_flat = axes.flatten()
        
        # Create a heatmap for each channel
        for idx in range(C):
            ax = axes_flat[idx]

            # Extract the channel data
            channel_data = obs[:, :, idx]

            # Create the heatmap
            im = ax.imshow(channel_data, cmap='viridis', interpolation='nearest', aspect='auto')

            # Get channel name if available, otherwise use index
            if idx < len(self.FEATURE_CHANNEL_MAP):
                channel_name = self.FEATURE_CHANNEL_MAP[idx]
            else:
                channel_name = f"Channel {idx}"

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{channel_name}", fontsize=9)

            # Add colorbar
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(C, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir, bbox_inches='tight', dpi=150)
            print(f"All channels heatmap saved to {output_dir}")
        plt.close(fig)
    def get_player_position(self, obs, player_idx=0, eps=1e-4):
        """Extract player position from observation."""
        player_loc_ch = self.CHANNEL_FEATURE_MAP[f"player_{player_idx}_loc"]
        player_locs = np.argwhere(obs[:, :, player_loc_ch] >= (1-eps))
        
        if len(player_locs) > 0:
            player_y, player_x = player_locs[0]
            return (player_x, player_y)
        return None
    def visualize_action_sequence_video(self, obs, actions, output_path, player_idx=0, fps=1, normalize=True):
        """
        Create a video showing action sequence, one frame per action.
        
        Args:
            obs: Initial observation [H, W, C] 
            actions: Action sequence [Horizon] with values 0-5
            output_path: Where to save the video
            player_idx: Which player to track (0 or 1)
            fps: Frames per second for video
        """
        
        # Extract player position
        player_pos = self.get_player_position(obs, player_idx)
        if player_pos is None:
            print(f"Warning: Could not find player {player_idx} in observation")
            return
            
        start_x, start_y = player_pos
        
        # Action mappings
        action_to_delta = {
            0: (0, -1),  # North
            1: (0, 1),   # South
            2: (1, 0),   # East
            3: (-1, 0),  # West
            4: (0, 0),   # Stay
            5: (0, 0)    # Interact
        }
        
        action_names = ['North', 'South', 'East', 'West', 'Stay', 'Interact']
        action_colors = ['blue', 'red', 'green', 'orange', 'gray', 'purple']
        impassable_tiles = {'P', 'X', 'O', 'T', 'D', 'S'}
        
        # Get grid and render initial frame
        grid = self.extract_grid_from_obs(obs)
        
        # Track position
        x, y = start_x, start_y
        frames = []
        tile_size = self.UNSCALED_TILE_SIZE
        
        # Track statistics
        action_counts = {i: 0 for i in range(6)}
        consecutive_stays = 0
        
        for t, action in enumerate(actions):
            action_idx = action.item() if hasattr(action, 'item') else action
            action_counts[action_idx] += 1
            
            # Track consecutive stays
            if action_idx == 4:  # Stay action
                consecutive_stays += 1
            else:
                consecutive_stays = 0
            
            # Render current state
            surface = self.render_frame(obs, grid, normalize=normalize)
 
            
            # Convert pygame surface to numpy array
            w, h = surface.get_size()
            surf_array = np.frombuffer(surface.get_buffer().raw, dtype=np.uint8)
            surf_array = pygame.surfarray.array3d(surface)
            surf_array = np.swapaxes(surf_array, 0, 1) 
            
            # Create figure for this frame
            fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(16, 8), 
                                                gridspec_kw={'width_ratios': [2, 1]})
            
            # Main game view
            ax_main.imshow(surf_array)
            
            # Convert grid coords to pixel coords
            px = x * tile_size + tile_size // 2
            py = y * tile_size + tile_size // 2
            
            # Check if move would be out of bounds
            dx, dy = action_to_delta[action_idx]
            new_x = x + dx
            new_y = y + dy
            is_out_of_bounds = not (0 <= new_x < obs.shape[1] and 0 <= new_y < obs.shape[0])
            is_blocked = True if not is_out_of_bounds and grid[new_y][new_x] in impassable_tiles else False
            # Draw current action
            if action_idx < 4:  # Movement action
                arrow_color = 'red' if is_out_of_bounds else action_colors[action_idx]
                ax_main.arrow(px, py, 
                        dx * tile_size * 0.7, dy * tile_size * 0.7,
                        head_width=15, head_length=15, 
                        fc=arrow_color, 
                        ec='black', 
                        linewidth=3,
                        alpha=0.9)
                
                if is_out_of_bounds or is_blocked:
                    # Add X marker for blocked move
                    blocked_px = px + dx * tile_size * 0.7
                    blocked_py = py + dy * tile_size * 0.7
                    ax_main.scatter(blocked_px, blocked_py, color='red', s=200, 
                                marker='x', linewidths=4)
                    
            elif action_idx == 4:  # Stay
                # Draw a circle at current position
                circle = patches.Circle((px, py), 20, 
                                    color='gray', 
                                    fill=True, 
                                    alpha=0.5)
                ax_main.add_patch(circle)
                ax_main.text(px, py, '●', ha='center', va='center', 
                        fontsize=20, color='white', weight='bold')
                
            elif action_idx == 5:  # Interact
                circle = patches.Circle((px, py), 25, 
                                    color='purple', 
                                    fill=False, 
                                    linewidth=4)
                ax_main.add_patch(circle)
                ax_main.text(px, py, 'I', ha='center', va='center', 
                        fontsize=25, color='purple', weight='bold')
            
            # Highlight current position
            current_rect = patches.Rectangle((x * tile_size, y * tile_size), 
                                        tile_size, tile_size,
                                        linewidth=3, edgecolor='yellow', 
                                        facecolor='none')
            ax_main.add_patch(current_rect)
            
            ax_main.set_title(f'Step {t+1}/{len(actions)} - Action: {action_names[action_idx]}', 
                            fontsize=16)
            ax_main.axis('off')
            
            # Info panel
            ax_info.axis('off')
            info_text = f"Current Step: {t+1}/{len(actions)}\n"
            info_text += f"Current Action: {action_names[action_idx]}\n"
            info_text += f"Position: ({x}, {y})\n\n"
            
            info_text += "Action Counts:\n"
            for i, name in enumerate(action_names):
                count = action_counts[i]
                percentage = (count / (t+1)) * 100
                info_text += f"{name}: {count} ({percentage:.1f}%)\n"
            
            info_text += f"\nConsecutive Stays: {consecutive_stays}\n"
            
            if is_out_of_bounds:
                info_text += "\n OUT OF BOUNDS!"
            
            ax_info.text(0.05, 0.85, info_text, transform=ax_info.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Action history (last 10 actions)
            history_start = max(0, t-16)
            history_text = "Recent Actions:\n"
            for i in range(history_start, t+1):
                hist_action = actions[i].item() if hasattr(actions[i], 'item') else actions[i]
                history_text += f"{i+1}: {action_names[hist_action]}\n"
            
            ax_info.text(0.65, 0.85, history_text, transform=ax_info.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            # Convert figure to frame
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            frame = np.asarray(buf)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))
            
            plt.close(fig)
            
            # Update position only if in bounds
            if not is_out_of_bounds and not is_blocked and action_idx < 4:
                x, y = new_x, new_y
        
        # Write video
        if frames:
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            clip = ImageSequenceClip(rgb_frames, fps=fps)
            clip.write_videofile(output_path, codec="libx264", fps=fps, verbose=False, logger=None)
            print(f"Action sequence video saved to {output_path}")
            
        return output_path




