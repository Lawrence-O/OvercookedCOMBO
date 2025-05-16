import sys
from pathlib import Path

# Get the current directory of file.py
current_dir = Path(__file__).resolve().parent

# Add the current directory to sys.path
sys.path.append(str(current_dir))

from guided_diffusion.unet import UNetModel, SuperResModel
from torch import nn
import torch
from einops import repeat, rearrange
import torch.nn.functional as F


class UnetBridge(nn.Module):
    def __init__(self):
        super(UnetBridge, self).__init__()

        self.unet = UNetModel(
            image_size=(48, 64),
            in_channels=6,
            model_channels=160,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=(4, 8),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.unet.convert_to_fp32()

    def forward(self, x, x_cond, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3
        x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')

class UnetMW(nn.Module):
    def __init__(self):
        super(UnetMW, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    def forward(self, x, x_cond, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3
        x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')

class UnetOvercooked(nn.Module):
    def __init__(self, horizon, obs_dim):
        super(UnetOvercooked, self).__init__()
        self.horizon = horizon
        self.H, self.W, self.C  = obs_dim
        self.unet = UNetModel(
            image_size=(8, 6),
            in_channels=26*2,
            model_channels=128,
            out_channels=26,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2),
            conv_resample=True,
            dims=3,
            num_classes=25,
            task_tokens=False,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    def pad_even(self, x):
        """Pads the Height and Width dimensions (dims 2 and 3 for 5D) to be even."""
        if x.dim() == 5: # Input: (B, F, H, W, C)
            B, _, H, W, C = x.shape
            pad_h = (H % 2)
            pad_w = (W % 2)
            # Pad tuple order: (pad_dim4_L, R, pad_dim3_L, R, pad_dim2_L, R)
            pad_tuple = (0, 0, 0, pad_w, 0, pad_h)
        elif x.dim() == 4: # Input: (B, H, W, C)
            B, H, W, C = x.shape
            pad_h = (H % 2)
            pad_w = (W % 2)
            # Pad tuple order: (pad_dim3_L, R, pad_dim2_L, R, pad_dim1_L, R)
            pad_tuple = (0, 0, 0, pad_w, 0, pad_h)
        else:
             raise ValueError(f"pad_even expects 4D or 5D tensor, got {x.dim()}D")

        x_padded = F.pad(x, pad_tuple, mode='constant', value=0)
        return x_padded
    def forward(self, x, x_cond, t, task_embed=None, vis=None, **kwargs):
        *_, H, W = x.shape
        x = rearrange(x, "b (f c) h w -> b f h w c", f=self.horizon, c=self.C)
        x = self.pad_even(x)
        x = rearrange(x, 'b f h w c -> b c f h w')
        x_cond = rearrange(x_cond, "b c h w -> b h w c")
        x_cond = self.pad_even(x_cond)
        x_cond = rearrange(x_cond, 'b h w c -> b c 1 h w')
        x_cond = repeat(x_cond, 'b c 1 h w -> b c f h w', f=self.horizon)

        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        out = rearrange(out, 'b c f h w -> b f h w c')
        out = out[:, :, :H, :W, :]
        out = rearrange(out, "b f h w c -> b (f c) h w")
        return out
      
class UnetMW_flow(nn.Module):
    def __init__(self):
        super(UnetMW_flow, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=5,
            model_channels=128,
            out_channels=2,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    def forward(self, x, x_cond, t, task_embed=None, **kwargs):
        f = x.shape[1] // 2
        x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
    
class UnetThor(nn.Module):
    def __init__(self):
        super(UnetThor, self).__init__()

        self.unet = UNetModel(
            image_size=(64, 64),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=(4, 8),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.unet.convert_to_fp32()

    def forward(self, x, x_cond, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3
        x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
    

class UnetMaco(nn.Module):
    def __init__(self, embed_dim=2048, num_frames=8, conds=1):
        super(UnetMaco, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=3*(conds+1),
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=embed_dim,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
            num_frames=num_frames
        )
    def forward(self, x, x_cond, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3
        x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
        x = rearrange(x, 'b (f c) h w -> b c f h w', f=f)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')

class UnetTDWMacoInpainting(nn.Module):
    def __init__(self, embed_dim=2048, conds=1):
        super().__init__()
        self.unet = UNetModel(
            image_size=(336, 336),
            in_channels=3*(conds+1),
            model_channels=64,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(16,),
            dropout=0,
            channel_mult=(1, 2, 4, 6, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            task_tokens=None,
            task_token_channels=embed_dim,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32
        )
    def forward(self, x, x_cond, t, *args, **kwargs):
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, *args, **kwargs)
        return out

class UnetSuperRes(nn.Module):
    def __init__(self, target_size=(512, 512)) -> None:
        super().__init__()
        self.unet = SuperResModel(
            image_size=target_size,
            in_channels=3,
            model_channels=64,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(16,),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=2,
            num_classes=None,
            task_tokens=None,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    
    def forward(self, x, x_cond, t, *args, **kwargs):
        # x: [b 3 nh nw], t: [b], low_res: [b 3 h w]
        return self.unet(x, t, x_cond)