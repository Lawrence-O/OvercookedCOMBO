from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from .imagen import CrossAttention



class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, vis=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, vis)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        cross=False,
        name="",
        cross_attention_heads=4,
        cross_attention_dim_head=32
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.cross=cross
        self.dims = dims
        self.name = name

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        if self.cross:
            self.cross_attn = CrossAttention(self.out_channels, context_dim=emb_channels, dim_head=cross_attention_dim_head, heads=cross_attention_heads)

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, vis):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, vis), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, vis=None):
        emb, latent, mask = emb
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        if self.cross:
            if self.dims == 3:
                _, _, frames, height, width = h.size()
                h = rearrange(h, 'b c f h w -> b (f h w) c')
                if vis is not None:
                    c, attn = self.cross_attn(h, latent, mask=mask, ret_score=True)
                    attn = attn.mean(dim=1) # b (f h w) t
                    vis.add_attn_map(rearrange(attn, 'b (f h w) t -> b f h w t', f=frames, h=height, w=width), self.name)
                else:
                    c = self.cross_attn(h, latent, mask=mask)
                h = h + c
                h = rearrange(h, 'b (f h w) c -> b c f h w', f=frames, h=height, w=width)
            elif self.dims == 2:
                _, _, height, width = h.size()
                h = rearrange(h, 'b c h w -> b (h w) c')
                h = self.cross_attn(h, latent) + h
                h = rearrange(h, 'b (h w) c -> b c h w', h=height, w=width)
            elif self.dims == 1:
                _, _, length = h.size()
                h = rearrange(h, 'b c t -> b t c')
                h = self.cross_attn(h, latent) + h
                h = rearrange(h, 'b t c -> b c t', t=length)
            else:
                raise ValueError(f"Unsupported dimension in Cross Attention : {self.dims}. Only 1D, 2D, and 3D are supported.")
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        dims=3,
        name="",
        enable_spatiotemporal_attention=False,
    ):
        super().__init__()
        self.channels = channels
        self.name = name
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = conv_nd(1, channels, channels, 1)
        self.dims = dims
        self.enable_spatiotemporal_attention = enable_spatiotemporal_attention

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        if self.dims == 1:
            assert len(spatial) == 1, "1D attention requires 1 spatial dimension"
            qkv = self.qkv(self.norm(x))
            h = self.attention(qkv)
            h = self.proj_out(h)
            return x + h
        elif self.dims == 2:
            assert len(spatial) == 2, "2D attention requires 2 spatial dimensions"
            x = rearrange(x, "b c x y -> b c (x y)")
            qkv = self.qkv(self.norm(x))
            h = self.attention(qkv)
            h = self.proj_out(h)
            return rearrange((x + h), "b c (x y) -> b c x y", c=c, x=spatial[0], y=spatial[1])
        elif self.dims == 3:
            assert len(spatial) == 3, "3D attention requires 3 spatial dimensions"
            if self.enable_spatiotemporal_attention:
                x = rearrange(x, "b c f x y -> b c (f x y)")
                qkv = self.qkv(self.norm(x))
                h = self.attention(qkv)
                h = self.proj_out(h)
                return rearrange((x + h), "b c (f x y) -> b c f x y", c=c, f=spatial[0], x=spatial[1], y=spatial[2])
            else:
                x = rearrange(x, "b c f x y -> (b f) c (x y)")
                qkv = self.qkv(self.norm(x))
                h = self.attention(qkv)
                h = self.proj_out(h)
                return rearrange((x + h), "(b f) c (x y) -> b c f x y", c=c, f=spatial[0], x=spatial[1], y=spatial[2])
        else:
            raise ValueError(f"Unsupported dimension: {self.dims}. Only 1D, 2D, and 3D are supported.")
def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, f, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c * f
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)



class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
    
class ResidualConvBlock(nn.Module):
    """ A residual block with an N-dimensional convolution, normalization, and activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dims=2):
        super().__init__()
        if isinstance(kernel_size, int):
            pad = kernel_size // 2
        else:
            pad = tuple(k // 2 for k in kernel_size)
        self.conv_block = nn.Sequential(
            conv_nd(dims, in_channels, out_channels, kernel_size, padding=pad),
            normalization(out_channels),
            nn.SiLU(),
            conv_nd(dims, out_channels, out_channels, kernel_size, padding=pad),
        )
        if in_channels != out_channels:
            self.skip_connection = conv_nd(dims, in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    def forward(self, x):
        residual = self.skip_connection(x)
        out = self.conv_block(x)
        return out + residual

class SpatialObservationEncoder(nn.Module):
    """
    Encodes an observation image into a sequence of spatial tokens
    to be used as context in a cross-attention layer.
    """
    def __init__(self, in_channels, context_dim):
        super().__init__()
        assert context_dim % 2 == 0, "context_dim must be even for ResidualConv2DBlock"
        self.feature_extractor = nn.Sequential(
            ResidualConvBlock(in_channels, context_dim // 2, dims=2),
            ResidualConvBlock(context_dim // 2, context_dim, dims=2),
            ResidualConvBlock(context_dim, context_dim, dims=2),
        )
    def forward(self, x):
        out = self.feature_extractor(x)
        return out

class VideoObservationEncoder(nn.Module):
    """
    Encodes a video observation into a sequence of spatial tokens
    to be used as context in a cross-attention layer.
    """
    def __init__(self, in_channels, context_dim):
        super().__init__()
        assert context_dim % 2 == 0, "context_dim must be even for ResidualConv3DBlock"
        self.feature_extractor = nn.Sequential(
            ResidualConvBlock(in_channels, context_dim // 2, dims=3),
            Downsample(context_dim // 2, True, dims=3),
            ResidualConvBlock(context_dim // 2, context_dim, dims=3),
            Downsample(context_dim, True, dims=3),
            ResidualConvBlock(context_dim, context_dim, dims=3),
        )
        
    def forward(self, x):
        out = self.feature_extractor(x)
        out = rearrange(out, 'b d f h w -> b (f h w) d')
        return out

class ActionConditionEncoder(nn.Module):
    """
    Encodes a sequence of actions into a context vector.
    This is used to condition the model on the actions taken in the environment.
    """
    def __init__(self, num_actions, context_dim):
        super().__init__()
        assert context_dim % 2 == 0, "context_dim must be even for ResidualConv1DBlock"
        self.feature_extractor = nn.Sequential(
            ResidualConvBlock(num_actions, context_dim // 2, dims=1),
            ResidualConvBlock(context_dim // 2, context_dim, dims=1),
            ResidualConvBlock(context_dim, context_dim, dims=1),
        )

    def forward(self, x):
        x = rearrange(x, 'b k a -> b a k')
        out = self.feature_extractor(x)
        out = rearrange(out, 'b c k -> b k c')
        return out
    


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        num_actions=None,
        image_cond_dim=None,
        video_cond_dim=None,
        action_horizon=None,
        reward_dim=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        cross=False,
        cross_attention_heads=4,
        cross_attention_dim_head=32,
        spatiotemporal_attention={},
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_actions = num_actions
        self.action_horizon = action_horizon
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.image_cond_dim = image_cond_dim
        self.video_cond_dim = video_cond_dim
        self.cross = cross
        self.use_scale_shift_norm = use_scale_shift_norm # Mostly here for wandb logging
        self.cross_attention_heads = cross_attention_heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.spatiotemporal_attention = spatiotemporal_attention
        self.reward_dim = reward_dim

        time_embed_dim = model_channels * 4
        self.context_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, self.context_dim)
        
        if self.reward_dim is not None:
            self.reward_emb = nn.Sequential(
                linear(self.context_dim, self.context_dim),
                nn.SiLU(),
                linear(self.context_dim, self.context_dim),
            )

        if self.num_actions is not None and self.action_horizon is not None:
            self.action_cond = ActionConditionEncoder(
                num_actions=self.num_actions,
                context_dim=self.context_dim,
            )
        
        if image_cond_dim is not None:
            C, _, _ = image_cond_dim
            self.obs_encoder = SpatialObservationEncoder(
                in_channels=C,
                context_dim=self.context_dim,
            )
        
        if video_cond_dim is not None:
            *_, C, _, _ = video_cond_dim
            self.video_encoder = VideoObservationEncoder(
                in_channels=C,
                context_dim=self.context_dim,
            )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        action_proposal = not (self.num_actions is not None and self.action_horizon is not None)
        for level, mult in enumerate(channel_mult):
            for i in range(num_res_blocks):
                if action_proposal:
                    use_cross = self.cross and (i == 0)
                else:
                    use_cross = self.cross
                enable_spatio_temporal = (level in self.spatiotemporal_attention)
                print(f"Input block {level} with mult {mult}, use_cross: {use_cross}, enable_spatio_temporal: {enable_spatio_temporal}")
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        cross=use_cross,
                        name=f'input_block_{level}_{i}',
                        cross_attention_heads=self.cross_attention_heads,
                        cross_attention_dim_head=self.cross_attention_dim_head
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            dims=dims,
                            name=f'input_block_{level}_attn',
                            enable_spatiotemporal_attention=enable_spatio_temporal,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            name=f'input_block_{level}_downsample'
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                cross=self.cross,
                name=f'middle_block_1',
                cross_attention_heads=self.cross_attention_heads,
                cross_attention_dim_head=self.cross_attention_dim_head
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                dims=dims,
                name=f'middle_block_self_attn',
                enable_spatiotemporal_attention=True,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                cross=self.cross,
                name=f'middle_block_2',
                cross_attention_heads=self.cross_attention_heads,
                cross_attention_dim_head=self.cross_attention_dim_head
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                if action_proposal:
                    use_cross = self.cross and (i == 0)
                else:
                    use_cross = self.cross
                enable_spatio_temporal = (level in self.spatiotemporal_attention)
                print(f"Output block {level} with mult {mult}, use_cross: {use_cross}, enable_spatio_temporal: {enable_spatio_temporal}")
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        cross=use_cross,
                        name=f'output_block_{level}_{i}',
                        cross_attention_heads=self.cross_attention_heads,
                        cross_attention_dim_head=self.cross_attention_dim_head
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            dims=dims,
                            name=f'output_block_{level}_self_attn',
                            enable_spatiotemporal_attention=enable_spatio_temporal,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            name=f'output_block_{level}_{i}_upsample',
                            cross=self.cross,
                            cross_attention_heads=self.cross_attention_heads,
                            cross_attention_dim_head=self.cross_attention_dim_head,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, action_embed=None, reward_embed=None, image_embed=None, history_embed=None, mask=None, vis=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        context_list = []

        # ---- World Model Context Embeddings ----

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            label_latent = self.label_emb(y).unsqueeze(1) # [B, 1, C]
            context_list.append(label_latent)
        
        if self.num_actions is not None and self.action_horizon is not None:
            B, K, A = action_embed.shape
            assert (B, K, A) == (x.shape[0], self.action_horizon, self.num_actions), \
                f"Expected action_embed shape {(x.shape[0], self.action_horizon, self.num_actions)}, got {action_embed.shape}"
            action_latent = self.action_cond(action_embed) # [B, C, K]
            context_list.append(action_latent)
        
        # ---- Action Proposal Embeddings ----
        if image_embed is not None and self.image_cond_dim is not None:
            assert image_embed.shape[0] == x.shape[0]
            assert image_embed.ndim == 4, f"Expected image_embed to be 4D, got {image_embed.ndim}D"
            image_latent = self.obs_encoder(image_embed)
            image_latent = rearrange(image_latent, 'b d h w -> b (h w) d')
            context_list.append(image_latent)
        
        if reward_embed is not None and self.reward_dim is not None:
            assert reward_embed.shape[0] == x.shape[0]
            assert reward_embed.ndim == 2 and reward_embed.shape[1] == self.reward_dim, \
                f"Expected reward_embed to be 2D of shape [B, {self.reward_dim}], got {reward_embed.shape}"
            rtg_embed = timestep_embedding(reward_embed.squeeze(-1), self.context_dim)
            reward_latent = self.reward_emb(rtg_embed)
            reward_latent = rearrange(reward_latent, 'b d -> b 1 d')
            context_list.append(reward_latent)
        
        if history_embed is not None and self.video_cond_dim is not None:
            assert history_embed.shape[0] == x.shape[0]
            assert history_embed.ndim == 5, f"Expected history_embed to be 5D, got {history_embed.ndim}D"
            history_latent = self.video_encoder(history_embed)
            context_list.append(history_latent)
        
        
        if len(context_list) > 0:
            latent = th.cat(context_list, dim=1) # [B, len(context_list), C]
            emb = emb + latent.mean(dim=1)
        else:
            latent = None

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, (emb, latent, mask), vis)
            hs.append(h)
        h = self.middle_block(h, (emb, latent, mask), vis)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, (emb, latent, mask), vis)
        h = h.type(x.dtype)
        return self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            dims=dims,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                dims=dims,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(dims, ch, out_channels, 1),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)
