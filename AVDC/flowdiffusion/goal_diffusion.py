import datetime
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt
import numpy as np
import json
import wandb

__version__ = "0.0"

import os
import shutil

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def to_device(data, device):
    if isinstance(data, list) or isinstance(data, tuple):
        return [to_device(i, device) for i in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

import tensorboard as tb

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def tensors2vectors(tensors):
    def tensor2vector(tensor):
        flo = (tensor.permute(1, 2, 0).numpy()-0.5)*1000
        r = 8
        plt.quiver(flo[::-r, ::r, 0], -flo[::-r, ::r, 1], color='r', scale=r*20)
        plt.savefig('temp.jpg')
        plt.clf()
        return plt.imread('temp.jpg').transpose(2, 0, 1)
    return torch.from_numpy(np.array([tensor2vector(tensor) for tensor in tensors])) / 255

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


   
class GoalGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        guidance_weight = 0,
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        vis = None,
        num_actions = None,
        no_op_action = None, 
    ):
        super().__init__()
        # assert not (type(self) == GoalGaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = channels
        self.image_size = image_size
        self.objective = objective
        self.vis = vis
        self.num_actions = num_actions
        if self.num_actions and no_op_action is None:
            raise ValueError('no_op_action must be provided if num_actions is specified')
        self.no_op_action = no_op_action

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.guidance_weight = guidance_weight

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond=None, task_embed=None, action_embed=None, reward_embed=None, image_embed=None, history_embed=None, mask=None, clip_x_start=False, rederive_pred_noise=False, **kwargs):
        batched_times = torch.full((x.size(0),), t, device = x.device, dtype = torch.long)
        model_output = self.model(x, x_cond, batched_times, task_embed, action_embed, reward_embed, image_embed, history_embed,  mask=mask, **kwargs)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, batched_times, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, batched_times, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, batched_times, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, batched_times, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, batched_times, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond=None, task_embed=None, action_embed=None, reward_embed=None, image_embed=None, history_embed=None, mask=None, clip_denoised=False):
        preds_cond = self.model_predictions(
            x=x, 
            t=t, x_cond=x_cond, 
            task_embed=task_embed, 
            action_embed=action_embed, 
            reward_embed=reward_embed,
            image_embed=image_embed, 
            history_embed=history_embed, mask=mask)
        x_start = preds_cond.pred_x_start * (1 + self.guidance_weight)
        if self.guidance_weight > 0:
            uncond_actions = torch.ones_like(action_embed) * self.no_op_action if action_embed is not None else None
            uncond_reward = torch.zeros_like(reward_embed) if reward_embed is not None else None
            uncond_history = torch.zeros_like(history_embed) if history_embed is not None else None
            uncond_image = torch.zeros_like(image_embed) if image_embed is not None else None
            uncond_x_cond = None if x_cond is None else x_cond
            preds_uncond = self.model_predictions(x=x, t=t, x_cond=uncond_x_cond, 
                                                task_embed=torch.zeros_like(task_embed),
                                                action_embed=uncond_actions, 
                                                reward_embed=uncond_reward,
                                                image_embed=uncond_image,
                                                history_embed=uncond_history, mask=None)
            x_start -= preds_uncond.pred_x_start * self.guidance_weight

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        batched_times = torch.full((x.size(0),), t, device = x.device, dtype = torch.long)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = batched_times)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_cond, task_embed=None, action_embed=None, reward_embed=None, image_embed=None, history_embed=None, mask=None):
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, 
                                                                          t=t, 
                                                                          x_cond=x_cond, 
                                                                          task_embed=task_embed, 
                                                                          action_embed=action_embed, 
                                                                          reward_embed=reward_embed, 
                                                                          image_embed=image_embed,
                                                                          history_embed=history_embed, 
                                                                          mask=mask, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, task_embed=None, action_embed=None, reward_embed=None, image_embed=None, history_embed=None, mask=None, return_all_timesteps=False):

        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(x=img, 
                                         t=t, 
                                         x_cond=x_cond, 
                                         task_embed=task_embed, 
                                         action_embed=action_embed, 
                                         reward_embed=reward_embed, 
                                         image_embed=image_embed, 
                                         history_embed=history_embed, 
                                         mask=mask)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
    
    @torch.no_grad()
    def ddim_sample(self, shape, x_cond=None, task_embed=None, action_embed=None, reward_embed=None, image_embed=None, history_embed=None, mask=None, comp_mask=None, return_all_timesteps=False, vis=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # Get Conditional Predictions
            cond_pred_noise, cond_x_start = self.model_predictions(
                x=img, t=time, x_cond=x_cond, 
                task_embed=task_embed, 
                action_embed=action_embed, 
                reward_embed=reward_embed, 
                image_embed=image_embed, 
                history_embed=history_embed, 
                mask=mask,
                clip_x_start=True
            )

            if self.guidance_weight == 0:
                x_start = cond_x_start
                pred_noise = cond_pred_noise
            else:
                # Get Unconditional Predictions
                uncond_task_embed = torch.zeros_like(task_embed) if task_embed is not None else None
                uncond_action_embed = torch.ones_like(action_embed) * self.no_op_action if action_embed is not None else None
                uncond_reward_embed = torch.zeros_like(reward_embed) if reward_embed is not None else None
                uncond_history_embed = torch.zeros_like(history_embed) if history_embed is not None else None
                uncond_image_embed = torch.zeros_like(image_embed) if image_embed is not None else None      

                uncond_pred_noise, uncond_x_start = self.model_predictions(
                    x=img, t=time, 
                    x_cond=x_cond, 
                    task_embed=uncond_task_embed, 
                    action_embed=uncond_action_embed, 
                    reward_embed=uncond_reward_embed,
                    image_embed=uncond_image_embed,
                    history_embed=uncond_history_embed, 
                    mask=None, clip_x_start = False, rederive_pred_noise = True)
                
                
                # Performance Classifier Free Guidance
                if self.guidance_weight == 0:
                    raise NotImplementedError("Haven't really thought this out yet, should we return the unconditional or conditional ? Math vs Practicality")
                
                x_start = uncond_x_start + self.guidance_weight * (cond_x_start - uncond_x_start)
                pred_noise = uncond_pred_noise + self.guidance_weight * (cond_pred_noise - uncond_pred_noise)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def deprecated_ddim_sample(self, shape, x_cond=None, task_embed=None, action_embed=None, reward_embed=None, image_embed=None, history_embed=None, mask=None, comp_mask=None, return_all_timesteps=False, vis=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # torch.manual_seed(0)

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None
        if task_embed is not None and not isinstance(task_embed, list):
            task_embed = [task_embed]
            mask = [mask]
        
        if task_embed is not None:
            if vis is not None and not isinstance(vis, list):
                vis = [vis]
            elif vis is None:
                vis = [None for _ in range(len(task_embed))]
        
        if action_embed is not None and not isinstance(action_embed, list):
            action_embed = [action_embed]
        
        if reward_embed is not None and not isinstance(reward_embed, list):
            reward_embed = [reward_embed]
        
        if image_embed is not None and not isinstance(image_embed, list):
            image_embed = [image_embed]
        
        if history_embed is not None and not isinstance(history_embed, list):
            history_embed = [history_embed]

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # self_cond = x_start if self.self_condition else None
            if task_embed is None and action_embed is None and reward_embed is None and image_embed is None and history_embed is None:
                # Literally no conditioning
                assert x_cond is not None, "x_cond must be provided when no task_embed, action_embed, reward_embed, image_embed, or history_embed is provided"
                pred_noise, x_start, *_ = self.model_predictions(img, time, x_cond, task_embed, action_embed, reward_embed, mask, clip_x_start = False, rederive_pred_noise = True, vis = vis)
            else:
                assert task_embed is not None and action_embed is not None, "task_embed and action_embed must be provided when using task embedding"
                uncond_task_embed = torch.zeros_like(task_embed[0]) if task_embed is not None else None
                uncond_action_embed = torch.ones_like(action_embed[0]) * self.no_op_action if action_embed is not None else None
                uncond_reward_embed = torch.zeros_like(reward_embed[0]) if reward_embed is not None else None
                uncond_history_embed = torch.zeros_like(history_embed[0]) if history_embed is not None else None
                uncond_image_embed = torch.zeros_like(image_embed[0]) if image_embed is not None else None      

                pred_uncond_noise, x_uncond, *_ = self.model_predictions(
                    x=img, t=time, 
                    x_cond=x_cond, 
                    task_embed=uncond_task_embed, 
                    action_embed=uncond_action_embed, 
                    reward_embed=uncond_reward_embed,
                    image_embed=uncond_image_embed,
                    history_embed=uncond_history_embed, 
                    mask=None, clip_x_start = False, rederive_pred_noise = True)
                pred_noise = pred_uncond_noise
                x_start = x_uncond
                
                for i in range(len(task_embed)):
                    cur_pred_noise, cur_x_start, *_ = self.model_predictions(
                        x=img, t=time, 
                        x_cond=x_cond, 
                        task_embed=task_embed[i],
                        action_embed=action_embed[i], 
                        reward_embed=reward_embed[i],
                        image_embed=image_embed[i],
                        history_embed=history_embed[i], 
                        mask=mask[i], clip_x_start = False, rederive_pred_noise = True, vis = vis[i])
                    c_mask = comp_mask[i].view(-1, 1, 1, 1) if comp_mask is not None else 1

                    x_start = (cur_x_start - x_uncond) * self.guidance_weight * c_mask + x_start
                    pred_noise = (cur_pred_noise - pred_uncond_noise) * self.guidance_weight * c_mask + pred_noise

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            # img.clamp_(-1., 1.)

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
#TODO: Clean up Assertions and add more checks
    @torch.no_grad()
    def sample(self, x_cond=None, task_embed=None, action_embed=None, reward_embed=None, image_embed=None, history_embed=None, mask=None, comp_mask=None, batch_size=16, return_all_timesteps=False, vis=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        if x_cond is not None:
            assert x_cond.min() >= -1 and x_cond.max() <= 1, \
                f'x_cond must be normalized to [-1, 1], got range [{x_cond.min():.3f}, {x_cond.max():.3f}]'
        
        if image_embed is not None:
            assert image_embed.min() >= -1 and image_embed.max() <= 1, \
                f'image_embed must be normalized to [-1, 1], got range [{image_embed.min():.3f}, {image_embed.max():.3f}]'
        
        if history_embed is not None:
            assert history_embed.min() >= -1 and history_embed.max() <= 1, \
                f'history_embed must be normalized to [-1, 1], got range [{history_embed.min():.3f}, {history_embed.max():.3f}]'
        
        if action_embed is not None and self.num_actions is not None:
            assert action_embed.min() >= 0 and action_embed.max() < self.num_actions, \
                f'action_embed values must be in range [0, {self.num_actions - 1}], got range [{action_embed.min().item()}, {action_embed.max().item()}]'
            action_embed = F.one_hot(action_embed, num_classes=self.num_actions).float()

        return sample_fn(
            shape=(batch_size, channels, image_size[0], image_size[1]), 
            x_cond=x_cond, 
            task_embed=task_embed, 
            action_embed=action_embed,
            reward_embed=reward_embed,
            image_embed=image_embed,
            history_embed=history_embed, 
            mask=mask, 
            comp_mask=comp_mask, 
            return_all_timesteps=return_all_timesteps, 
            vis=vis
        )
    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, x_cond=None, task_embed=None, action_embed=None, reward_embed=None, image_embed=None, history_embed=None, mask=None, comp_mask=None, loss_scale=None, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        composed = isinstance(task_embed, list)
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step

        if composed:
            model_out = [self.model(x, x_cond, t, embed, action_embed, reward_embed, image_embed, history_embed, mask=msk) for embed, msk in zip(task_embed, mask)]
        else:
            model_out = self.model(x, x_cond, t, task_embed, action_embed, reward_embed, image_embed, history_embed, mask=mask)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if composed:
            mo = torch.zeros_like(model_out[0])
            for i, m in enumerate(model_out):
                mo += m * comp_mask[i].view(-1, 1, 1, 1)
            mo /= torch.stack(comp_mask, dim=0).sum(dim=0).view(-1, 1, 1, 1)
            multiple_loss = reduce(self.loss_fn(mo, target, reduction = 'none'), 'b ... -> b (...)', 'mean')
            single_loss = [self.loss_fn(mo, target, reduction = 'none') for mo in model_out]
            if loss_scale is not None:
                single_loss = [l * s for l, s in zip(single_loss, loss_scale)]
            single_loss = [reduce(l, 'b ... -> b (...)', 'mean') * m.unsqueeze(-1) for l, m in zip(loss, comp_mask)]
            single_loss = torch.stack(loss, dim=0).sum(dim=0)
            loss = multiple_loss + single_loss
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none')
            if loss_scale is not None:
                loss = loss * loss_scale
            loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
# TODO: Really need to clean this up
    def forward(self, img, img_cond, task_embed=None, action_embed=None, reward_embed=None, image_embed=None, history_embed=None, mask=None, comp_mask=None, loss_scale=None):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}, got({h}, {w})'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        assert img.min() >= -1 and img.max() <= 1, f'Image must be normalized to [-1, 1], got range [{img.min():.3f}, {img.max():.3f}]'
        
        if img_cond is not None:
            assert img_cond.min() >= -1 and img_cond.max() <= 1, f'Conditional Image must be normalized to [-1, 1], got range [{img_cond.min():.3f}, {img_cond.max():.3f}]'
        
        if image_embed is not None:
            assert image_embed.min() >= -1 and image_embed.max() <= 1, f'Image must be normalized to [-1, 1], got range [{image_embed.min():.3f}, {image_embed.max():.3f}]'
        
        if self.num_actions is not None:
            #TODO: Move this to the trainer class
            assert action_embed is not None, 'action_embed must be provided when num_actions is set'
            assert action_embed.shape[0] == b, f'action_embed batch size {action_embed.shape[0]} must match image batch size {b}'
            assert action_embed.dtype == torch.long, f'action_embed must be of type long, got {action_embed.dtype}'
            assert action_embed.min() >= 0 and action_embed.max() < self.num_actions, f'action_embed values must be in range [0, {self.num_actions - 1}], got range [{action_embed.min().item()}, {action_embed.max().item()}]'
            # action_embed shape: ( B, K) where K is the number of actions
            action_embed = F.one_hot(action_embed, num_classes=self.num_actions).float()
            # Now shape: (B, K, num_actions)
        
        return self.p_losses(
            x_start=img, 
            t=t, 
            x_cond=img_cond, 
            task_embed=task_embed, 
            action_embed=action_embed, 
            reward_embed=reward_embed,
            image_embed=image_embed, 
            history_embed=history_embed,
            mask=mask, 
            comp_mask=comp_mask, 
            loss_scale=loss_scale)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        tokenizer, 
        text_encoder, 
        train_set,
        valid_set,
        channels = 3,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 3,
        results_folder = './results',
        amp = True,
        fp16 = True,
        split_batches = True,
        calculate_fid = True,
        inception_block_idx = 2048, 
        cond_drop_chance=0.1,
        save_milestone = True,
        embed_preprocessed = False,
        composed = False,
    ):
        super().__init__()

        self.cond_drop_chance = cond_drop_chance
        self.composed = composed

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model

        self.channels = channels

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        
        valid_ind = [i for i in range(len(valid_set))][:num_samples]

        train_set = train_set
        valid_set = Subset(valid_set, valid_ind)

        self.ds = train_set
        self.valid_ds = valid_set
        
        self.embed_preprocessed = embed_preprocessed
        collect_fn = partial(Trainer.embed_collect_fn, composed=composed) if embed_preprocessed else None
        
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 4, collate_fn=collect_fn)
        # dl = dataloader
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.valid_dl = DataLoader(self.valid_ds, batch_size = valid_batch_size, shuffle = False, pin_memory = True, num_workers = 4, collate_fn=collect_fn)


        # optimizer
        self.train_lr = train_lr
        self.adam_betas = adam_betas
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.text_encoder = \
            self.accelerator.prepare(self.model, self.opt, self.text_encoder)
        
        self.save_milestone = save_milestone

    @staticmethod
    def embed_collect_fn(samples, composed=False):
        x, x_cond, embed = [], [], []
        rem = [[] for i in range(len(samples[0]) - 3)]
        max_len = 0
        embed_len = 0
        if composed:
            max_comp = 0
            for s in samples:
                max_comp = max(max_comp, len(s[2]))
            embed = [[] for _ in range(max_comp)]

        for s in samples:
            x.append(s[0])
            x_cond.append(s[1])
            if not composed:
                embed.append(s[2])
                max_len = max(max_len, s[2].size(0))
                embed_len = s[2].size(1)
            else: #compose prompt
                for i in range(max_comp):
                    assert len(s[2]) > 0
                    if i < len(s[2]):
                        embed[i].append(s[2][i])
                        max_len = max(max_len, s[2][i].size(0))
                        embed_len = s[2][i].size(1)
                    else:
                        embed[i].append(None)
            for i in range(len(rem)):
                rem[i].append(s[i + 3]) 
        
        x = torch.stack(x)
        x_cond = torch.stack(x_cond)
        if not composed:
            batch_embed = torch.zeros(x.size(0), max_len, embed_len)
            attn_mask = torch.zeros(x.size(0), max_len, dtype=torch.bool)
            for i in range(x.size(0)):
                batch_embed[i, :embed[i].size(0)] = embed[i]
                attn_mask[i, :embed[i].size(0)] = True
            text_goal = (batch_embed, attn_mask)
        else:
            batch_embed = [torch.zeros(x.size(0), max_len, embed_len) for e in embed]
            attn_mask = [torch.zeros(x.size(0), max_len, dtype=torch.bool) for e in embed]
            comp_mask = [torch.zeros(x.size(0), dtype=torch.bool) for e in embed]
            for c in range(max_comp):
                for i in range(x.size(0)):
                    if embed[c][i] is not None:
                        batch_embed[c][i, :embed[c][i].size(0)] = embed[c][i]
                        attn_mask[c][i, :embed[c][i].size(0)] = True
                        comp_mask[c][i] = True
                    else:
                        attn_mask[c][i] = False
                        comp_mask[c][i] = False
            text_goal = (batch_embed, attn_mask, comp_mask)
        
        return x, x_cond, text_goal, *rem

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone=None):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }
        
        save_name = f'modl-{milestone}.pt' if milestone is not None else 'model_recent.pt'
        if (self.results_folder / save_name).exists():
            shutil.move(str(self.results_folder / save_name), str(self.results_folder / f'model_last_recent.pt'))
        torch.save(data, str(self.results_folder / save_name))

        wandb_is_enabled = getattr(self, 'wandb_enabled', False)
        is_debug_mode = getattr(self, 'debug', False)

        if wandb_is_enabled and not is_debug_mode:
            if wandb.run is not None: # Check if wandb run is active
                try:
                    save_path = self.results_folder / save_name
                    wandb.save(str(save_path), base_path=str(self.results_folder.parent), policy="live")
                    print(f"Saved {save_path} to wandb.")
                except Exception as e:
                    print(f"Warning: Failed to save {save_path} to wandb: {e}")
            else:
                print(f"Warning: Wandb run is not active. Cannot save {save_path} to wandb.")

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        if isinstance(milestone, int):
            load_name = f'model-{milestone}.pt' if milestone >= 0 else 'model_recent.pt'
            try:
                data = torch.load(str(self.results_folder / load_name), map_location=device)
            except Exception as e:
                if milestone >= 0:
                    raise e
                print("load model_rencent failed, try to load model_last_recent")
                data = torch.load(str(self.results_folder / 'model_last_recent.pt'), map_location=device)
        else:
            data = torch.load(str(milestone), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}, step {data['step']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    #     return fid_value (t5xxl)
    @torch.no_grad()
    def encode_batch_text(self, batch_text):
        if self.embed_preprocessed:
            return to_device(batch_text, self.device)
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        print(f"batch_text: {batch_text}\nbatch_text_ids: {batch_text_ids}\nbatch_text_embed: {batch_text_embed}")
        return batch_text_embed, batch_text_ids.attn_mask

    def sample(self, x_conds, batch_text, batch_size=1, return_all_timesteps=False, vis=None):
        device = self.device
        if not self.composed:
            task_embeds, attn_mask = self.encode_batch_text(batch_text)
            comp_mask = None
        else:
            task_embeds, attn_mask, comp_mask = self.encode_batch_text(batch_text)
        return self.ema.ema_model.sample(x_conds.to(device), to_device(task_embeds, device), batch_size=batch_size, mask=attn_mask, comp_mask=comp_mask, return_all_timesteps=return_all_timesteps, vis=vis)
    
    def train_one_step(self):
        total_loss = 0
        
        for _ in range(self.gradient_accumulate_every):
            x, x_cond, goal, *_ = next(self.dl)
            x, x_cond = x.to(self.device), x_cond.to(self.device)

            if not self.composed:
                goal_embed, attn_mask = self.encode_batch_text(goal)
                loss_scale = None
            else:
                goal_embed, attn_mask, comp_mask = self.encode_batch_text(goal)
                h, w = x.shape[2:]
                loss_scale = [torch.ones_like(x) for _ in range(len(goal_embed))]
                loss_scale[0][:, :, :h // 2, :] *= 2
                loss_scale[1][:, :, :, :w // 2] *= 2
                loss_scale[2][:, :, h // 2:, :] *= 2
                loss_scale[3][:, :, :, w // 2:] *= 2
            ### zero whole goal_embed if p < self.cond_drop_chance
            drop_goal = torch.rand(x_cond.shape[0], device = self.device) > self.cond_drop_chance
            if not self.composed:
                goal_embed[drop_goal] = 0
                attn_mask[drop_goal] = True # can attention on zero
                comp_mask = None
            else:
                for i in range(len(goal_embed)):
                    goal_embed[i][drop_goal] = 0
                    attn_mask[i][drop_goal] = True # can attention on zero
                    if i > 0:
                        comp_mask[i][drop_goal] = False # not composed
                    loss_scale[i][drop_goal] = 1


            with self.accelerator.autocast():
                loss = self.model(x, x_cond, goal_embed, mask=attn_mask, comp_mask=comp_mask, loss_scale=loss_scale)
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()

                self.accelerator.backward(loss)
        
        return total_loss
    
    def valid_one_step(self):
        with torch.no_grad():
            milestone = self.step // self.save_and_sample_every
            batches = num_to_groups(self.num_samples, self.valid_batch_size)
            ### get val_imgs from self.valid_dl
            x_conds = []
            xs = []
            task_embeds = []
            all_text = []
            for i, (x, x_cond, label, text, *_) in enumerate(self.valid_dl):
                xs.append(x.to(self.device))
                x_conds.append(x_cond.to(self.device))
                task_embeds.append(self.encode_batch_text(label))
                all_text.extend(text)
            
            with self.accelerator.autocast():
                all_xs_list = list(map(lambda n, c, e: self.sample(c, e, n), batches, x_conds, task_embeds))
                loss_list = [self.ema.ema_model(x, x_cond, goal[0], mask=goal[1], comp_mask=(goal[2] if self.composed else None)).item()
                                for x, x_cond, goal in zip(xs, x_conds, task_embeds)]
            
            valid_loss = np.mean(loss_list)
        
        print_gpu_utilization()
        
        gt_xs = torch.cat(xs, dim = 0).detach().cpu() # [batch_size, 3*n, 120, 160]
        # make it [batchsize*n, 3, 120, 160]
        n_rows = gt_xs.shape[1] // self.channels
        gt_xs = rearrange(gt_xs, 'b (n c) h w -> b n c h w', n=n_rows)
        ### save images
        x_conds = torch.cat(x_conds, dim = 0).detach().cpu()
        # x_conds = rearrange(x_conds, 'b (n c) h w -> b n c h w', n=1)
        all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu()
        all_xs = rearrange(all_xs, 'b (n c) h w -> b n c h w', n=n_rows)

        gt_first = rearrange(x_conds, 'b (k c) h w -> b k c h w', c=3)
        gt_last = gt_xs[:, -1:]

        if self.step == self.save_and_sample_every:
            os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
            gt_img = torch.cat([gt_first, gt_xs, gt_last], dim=1)
            n_rows = gt_img.shape[1]
            gt_img = rearrange(gt_img, 'b n c h w -> (b n) c h w', c=3)
            utils.save_image(gt_img, str(self.results_folder / f'imgs/gt_img.png'), nrow=n_rows)
            Path(self.results_folder / f'imgs/prompt.json').write_text(json.dumps(all_text, indent=4))

        os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
        pred_img = torch.cat([gt_first,  all_xs, gt_last], dim=1)
        n_rows = pred_img.shape[1]
        pred_img = rearrange(pred_img, 'b n (v c) h w -> (b v n) c h w', c=3)
        utils.save_image(pred_img, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow=n_rows)
        
        return valid_loss

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        if self.accelerator.is_main_process:
            tensorboard_folder = Path(self.results_folder, "tensorboard")
            if self.step == 0 and tensorboard_folder.exists():
                shutil.rmtree(tensorboard_folder)
            tensorboard_folder.mkdir(exist_ok=True)
            tensorboard_sw = SummaryWriter(tensorboard_folder)
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = self.train_one_step()

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                if exists(self.accelerator.scaler):
                    scale = self.accelerator.scaler.get_scale()
                else:
                    scale = 0.0
                
                pbar.set_description(f'loss: {total_loss:.4E}, loss scale: {scale:.1E}')
                
                if self.accelerator.is_main_process:
                    tensorboard_sw.add_scalar("train loss", total_loss, self.step)
                    tensorboard_sw.add_scalar("loss scale", scale, self.step)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        valid_loss = self.valid_one_step()
                        tensorboard_sw.add_scalar("valid loss", valid_loss, self.step)
                        
                        if self.save_milestone:
                            self.save(self.step // self.save_and_sample_every)
                        else:
                            self.save()

                pbar.update(1)

        accelerator.print('training complete')

class TDWMacoTopdownTrainer(Trainer):
    def __init__(self,
                 diffusion_model,
                 train_set,
                 valid_set,
                 *,
                 train_batch_size=1,
                 valid_batch_size=1,
                 gradient_accumulate_every=1,
                 train_lr=0.0001,
                 train_num_steps=100000,
                 ema_update_every=10,
                 ema_decay=0.995,
                 adam_betas=(0.9, 0.99),
                 save_and_sample_every=1000,
                 num_samples=3,
                 results_folder='./results',
                 amp=True,
                 fp16=True,
                 split_batches=True,
                 calculate_fid=True,
                 inception_block_idx=2048,
                 save_milestone=True,
                ):
        
        super().__init__(diffusion_model,
                         None,
                         None,
                         train_set,
                         valid_set,
                         3,
                         train_batch_size=train_batch_size,
                         valid_batch_size=valid_batch_size,
                         gradient_accumulate_every=gradient_accumulate_every,
                         train_lr=train_lr,
                         train_num_steps=train_num_steps,
                         ema_update_every=ema_update_every,
                         ema_decay=ema_decay,
                         adam_betas=adam_betas,
                         save_and_sample_every=save_and_sample_every,
                         num_samples=num_samples,
                         results_folder=results_folder,
                         amp=amp,
                         fp16=fp16,
                         split_batches=split_batches,
                         calculate_fid=calculate_fid,
                         inception_block_idx=inception_block_idx,
                         cond_drop_chance=0,
                         save_milestone=save_milestone,
                         embed_preprocessed=True)
    
    def train_one_step(self):
        total_loss = 0
        
        for _ in range(self.gradient_accumulate_every):
            x, x_cond, *_ = next(self.dl)
            x, x_cond = x.to(self.device), x_cond.to(self.device)

            with self.accelerator.autocast():
                loss = self.model(x, x_cond)
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()

                self.accelerator.backward(loss)
        
        return total_loss
    
    def valid_one_step(self):
        with torch.no_grad():
            milestone = self.step // self.save_and_sample_every
            batches = num_to_groups(self.num_samples, self.valid_batch_size)
            ### get val_imgs from self.valid_dl
            x_conds = []
            xs = []
            for i, (x, x_cond, *_) in enumerate(self.valid_dl):
                xs.append(x.to(self.device))
                x_conds.append(x_cond.to(self.device))
            
            with self.accelerator.autocast():
                all_xs_list = list(map(lambda n, c: self.ema.ema_model.sample(batch_size=n, x_cond=c), batches, x_conds))
                loss_list = [self.ema.ema_model(x, x_cond).item() for x, x_cond in zip(xs, x_conds)]
            
            valid_loss = np.mean(loss_list)
        
        print_gpu_utilization()
        
        gt_xs = torch.cat(xs, dim = 0).detach().cpu() # [batch_size, 3, 128, 128]
        ### save images
        x_conds = torch.cat(x_conds, dim = 0).detach().cpu() # [batch_size, 9, 128, 128]
        all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu() # [batch_size, 3, 128, 128]
        gt_xs = rearrange(gt_xs, 'b (k c) h w -> b k c h w', c=3)
        x_conds = rearrange(x_conds, 'b (k c) h w -> b k c h w', c=3)
        all_xs = rearrange(all_xs, 'b (k c) h w -> b k c h w', c=3)
        imgs = torch.cat([x_conds, all_xs, gt_xs], dim=1)

        os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
        utils.save_image(imgs.flatten(end_dim=1), str(self.results_folder / f'imgs/sample-{milestone}.png'), nrow=imgs.shape[1])
        
        return valid_loss

class SuperResTrainer(Trainer):
    def __init__(self,
                 diffusion_model,
                 train_set,
                 valid_set,
                 *,
                 train_batch_size=1,
                 valid_batch_size=1,
                 gradient_accumulate_every=1,
                 train_lr=0.0001,
                 train_num_steps=100000,
                 ema_update_every=10,
                 ema_decay=0.995,
                 adam_betas=(0.9, 0.99),
                 save_and_sample_every=1000,
                 num_samples=3,
                 results_folder='./results',
                 amp=True,
                 fp16=True,
                 split_batches=True,
                 calculate_fid=True,
                 inception_block_idx=2048,
                 save_milestone=True,
                ):
        
        super().__init__(diffusion_model,
                         None,
                         None,
                         train_set,
                         valid_set,
                         3,
                         train_batch_size=train_batch_size,
                         valid_batch_size=valid_batch_size,
                         gradient_accumulate_every=gradient_accumulate_every,
                         train_lr=train_lr,
                         train_num_steps=train_num_steps,
                         ema_update_every=ema_update_every,
                         ema_decay=ema_decay,
                         adam_betas=adam_betas,
                         save_and_sample_every=save_and_sample_every,
                         num_samples=num_samples,
                         results_folder=results_folder,
                         amp=amp,
                         fp16=fp16,
                         split_batches=split_batches,
                         calculate_fid=calculate_fid,
                         inception_block_idx=inception_block_idx,
                         cond_drop_chance=0,
                         save_milestone=save_milestone,
                         embed_preprocessed=False)
    
    def train_one_step(self):
        total_loss = 0
        
        for _ in range(self.gradient_accumulate_every):
            x, x_cond = next(self.dl)
            x, x_cond = x.to(self.device), x_cond.to(self.device)

            with self.accelerator.autocast():
                loss = self.model(x, x_cond)
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()

                self.accelerator.backward(loss)
        
        return total_loss
    
    def valid_one_step(self):
        with torch.no_grad():
            milestone = self.step // self.save_and_sample_every
            batches = num_to_groups(self.num_samples, self.valid_batch_size)
            ### get val_imgs from self.valid_dl
            x_conds = []
            xs = []
            for i, (x, x_cond) in enumerate(self.valid_dl):
                xs.append(x.to(self.device))
                x_conds.append(x_cond.to(self.device))
            
            with self.accelerator.autocast():
                all_xs_list = [self.ema.ema_model.sample(batch_size=n, x_cond=c) for n, c in zip(batches, x_conds)]
                loss_list = [self.ema.ema_model(x, x_cond).item() for x, x_cond in zip(xs, x_conds)]
            
            valid_loss = np.mean(loss_list)
        
        print_gpu_utilization()
        
        gt_xs = torch.cat(xs, dim = 0).detach().cpu() # [batch_size, 3, 512, 512]
        ### save images
        x_conds = torch.cat(x_conds, dim = 0).detach().cpu() # [batch_size, 3, 128, 128]
        all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu() # [batch_size, 3, 512, 512]

        if self.step == self.save_and_sample_every:
            os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
            utils.save_image(gt_xs, str(self.results_folder / f'imgs/gt_img.png'), nrow=2)
            utils.save_image(x_conds, str(self.results_folder / f'imgs/cond.png'), nrow=2)

        os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
        utils.save_image(all_xs, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow=2)
        
        return valid_loss
    
from overcooked_sample_renderer import OvercookedSampleRenderer
class OvercookedEnvTrainer(Trainer):
    def __init__(self,
                 diffusion_model,
                 train_set,
                 valid_set,
                 channels,
                 *,
                 train_batch_size=1,
                 valid_batch_size=1,
                 gradient_accumulate_every=1,
                 train_lr=0.0001,
                 train_num_steps=100000,
                 ema_update_every=10,
                 ema_decay=0.995,
                 adam_betas=(0.9, 0.99),
                 save_and_sample_every=5000,
                 num_samples=3,
                 results_folder='./results',
                 amp=True,
                 fp16=True,
                 split_batches=True,
                 cond_drop_chance=0.1,
                 save_milestone=True,
                 debug=False,
                 dummy_policy_id=None,
                 wandb_enabled=False,
                 wandb_project="combo_overcooked_diffuser",
                 wandb_entity="social-rl",
                 wandb_run_name=None,
                 no_op_action=None,
                ):
        
        super().__init__(diffusion_model,
                         None,
                         None,
                         train_set=train_set,
                         valid_set=valid_set,
                         channels=channels,
                         train_batch_size=train_batch_size,
                         valid_batch_size=valid_batch_size,
                         gradient_accumulate_every=gradient_accumulate_every,
                         train_lr=train_lr,
                         train_num_steps=train_num_steps,
                         ema_update_every=ema_update_every,
                         ema_decay=ema_decay,
                         adam_betas=adam_betas,
                         save_and_sample_every=save_and_sample_every,
                         num_samples=num_samples,
                         results_folder=results_folder,
                         amp=amp,
                         fp16=fp16,
                         split_batches=split_batches,
                         calculate_fid=False,
                         cond_drop_chance=cond_drop_chance,
                         save_milestone=save_milestone,
                         embed_preprocessed=True)
        
        # Dataset Setup
        dl = DataLoader(self.ds, batch_size = self.batch_size, shuffle = True, pin_memory = True, num_workers = 4, collate_fn=self.overcooked_collate_fn)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        if self.valid_ds:
            valid_dl = DataLoader(self.valid_ds, batch_size=self.valid_batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=self.overcooked_collate_fn)
            self.valid_dl = self.accelerator.prepare(valid_dl)
        else:
            self.valid_dl = None
        
        self.debug = debug
        self.renderer = OvercookedSampleRenderer()
        self.dummy_id = dummy_policy_id
        self.wandb_enabled = wandb_enabled
        if no_op_action is None:
            raise ValueError("no_op_action must be provided for OvercookedEnvTrainer")
        assert self.dummy_id == 0, "dummy_policy_id must be 0 for OvercookedEnvTrainer"
        self.no_op_action = no_op_action
        
        config_to_log = {
            # Trainer parameters
            "learning_rate": train_lr,
            "train_batch_size": train_batch_size,
            "valid_batch_size": valid_batch_size,
            "gradient_accumulate_every": gradient_accumulate_every,
            "train_num_steps": train_num_steps,
            "ema_update_every": ema_update_every,
            "ema_decay": ema_decay,
            "adam_beta1": adam_betas[0],
            "adam_beta2": adam_betas[1],
            "save_and_sample_every": save_and_sample_every,
            "num_validation_samples": num_samples,
            "amp_enabled": amp,
            "fp16_enabled": fp16,
            "accelerator_split_batches": split_batches,
            "cond_drop_chance": cond_drop_chance,
            "save_milestone_checkpoints": save_milestone,
            "debug_mode": self.debug,
            "dummy_policy_id_used": self.dummy_id,
            "no_op_action": self.no_op_action,
            
            # Dataset parameters
            "dataset_type": train_set.dataset.__class__.__name__,
            "dataset_horizon": getattr(train_set.dataset, 'horizon', 'N/A'),
            "dataset_max_path_length": getattr(train_set.dataset, 'max_path_length', 'N/A'),
            "dataset_use_padding": getattr(train_set.dataset, 'use_padding', 'N/A'),
            "dataset_action_horizon": getattr(train_set.dataset, 'action_horizon', 'N/A'),
            "dataset_observation_dim": getattr(train_set.dataset, 'observation_dim', 'N/A'),
            "dataset_num_actions": getattr(train_set.dataset, 'num_actions', 'N/A'),
            "dataset_no_op_action": getattr(train_set.dataset, 'no_op_action', 'N/A'),
            "dataset_dummy_id": getattr(train_set.dataset, 'dummy_id', 'N/A'),
            "dataset_num_partner_policies": getattr(train_set.dataset, 'num_partner_policies', 'N/A'),
            "dataset_current_split": getattr(train_set.dataset, 'current_split', 'N/A'),
            "dataset_valid_indices": len(getattr(train_set.dataset, 'valid_indices', [])) if hasattr(train_set, 'valid_indices') else 'N/A',
            "dataset_policy_name_to_id": json.dumps(getattr(train_set.dataset, "policy_name_to_id", {})),
            "dataset_policy_id_to_name": json.dumps(getattr(train_set.dataset, "policy_id_to_name", {})),

            # Diffusion model (GoalGaussianDiffusion) parameters
            "diffusion_model_type": diffusion_model.__class__.__name__,
            "diffusion_total_input_channels": diffusion_model.channels, # Total channels for the diffusion model
            "diffusion_image_height": diffusion_model.image_size[0],
            "diffusion_image_width": diffusion_model.image_size[1],
            "diffusion_objective": diffusion_model.objective,
            "diffusion_loss_type": diffusion_model.loss_type,
            "diffusion_timesteps_total": diffusion_model.num_timesteps,
            "diffusion_sampling_timesteps": diffusion_model.sampling_timesteps,
            "diffusion_guidance_weight": diffusion_model.guidance_weight,
            "diffusion_ddim_sampling_eta": diffusion_model.ddim_sampling_eta,
            "diffusion_num_actions": diffusion_model.num_actions if hasattr(diffusion_model, 'num_actions') else 'N/A',
            "diffusion_no_op_action": diffusion_model.no_op_action if hasattr(diffusion_model, 'no_op_action') else 'N/A',

            # Unet parameters
            "unet_model_channels": diffusion_model.model.unet.model_channels if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_res_blocks": diffusion_model.model.unet.num_res_blocks if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_attention_resolutions": str(diffusion_model.model.unet.attention_resolutions) if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_channel_mult": str(diffusion_model.model.unet.channel_mult) if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_classes": diffusion_model.model.unet.num_classes if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_attention_resolutions": str(diffusion_model.model.unet.attention_resolutions) if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_cross_attention_enabled": diffusion_model.model.unet.cross if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_heads": diffusion_model.model.unet.num_heads if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_head_channels": diffusion_model.model.unet.num_head_channels if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_classes": diffusion_model.model.unet.num_classes if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_use_scale_shift_norm": diffusion_model.model.unet.use_scale_shift_norm if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_dropout": diffusion_model.model.unet.dropout if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_use_checkpointing": diffusion_model.model.unet.use_checkpoint if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_cross_attention_heads": diffusion_model.model.unet.cross_attention_heads if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_cross_attention_dim_head": diffusion_model.model.unet.cross_attention_dim_head if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_spatiotemporal_attention": diffusion_model.model.unet.spatiotemporal_attention if hasattr(diffusion_model.model, 'unet') else 'N/A', 
            "unet_video_cond_dim":str(diffusion_model.model.unet.video_cond_dim) if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_image_cond_dim": str(diffusion_model.model.unet.image_cond_dim) if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_reward_dim": diffusion_model.model.unet.reward_dim if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_action_horizon": diffusion_model.model.unet.action_horizon if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_actions": diffusion_model.model.unet.num_actions if hasattr(diffusion_model.model, 'unet') else 'N/A',
        }

        
        if cond_drop_chance > 0 and dummy_policy_id is None:
            raise ValueError("Must provide dummy_policy_id when cond_drop_chance > 0")
        
        if not self.debug and self.wandb_enabled and self.accelerator.is_local_main_process:
            final_wandb_run_name = wandb_run_name if wandb_run_name else f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project=wandb_project, entity=wandb_entity, name=final_wandb_run_name, config=config_to_log)
        
        if self.accelerator.is_local_main_process:
            underlying_dataset = self.ds.dataset if isinstance(self.ds, Subset) else self.ds
            print("\n" + "="*30 + " TRAINING CONFIGURATION " + "="*30)
            print(f"  Results Folder: {self.results_folder}")
            print(f"  Device: {self.device}")
            print(f"  Debug Mode: {self.debug}")
            print("-" * (60 + len(" TRAINING CONFIGURATION ")))
            print("  Dataset Information:")
            print(f"    Full Dataset Size (from '{underlying_dataset.current_split}' split): {len(underlying_dataset)}")
            print(f"    Train Subset Size: {len(self.ds)}")
            if self.valid_ds:
                print(f"    Validation Subset Size: {len(self.valid_ds)}")
            else:
                print(f"    Validation Subset Size: 0 (No validation set)")
            print(f"    Observation Dimension (H, W, C): {underlying_dataset.observation_dim}")
            print(f"    Horizon: {underlying_dataset.horizon}")
            print(f"    Number of Unique Partner Policies (for UNet classes): {underlying_dataset.num_partner_policies}")
            print(f"    Dummy Policy ID (for CFG): {underlying_dataset.dummy_id}")
            if hasattr(underlying_dataset, 'train_partner_policies'):
                print(f"    Train Partner Policies ({len(underlying_dataset.train_partner_policies)}):")
                for name, id_val in underlying_dataset.train_partner_policies.items():
                    print(f"      '{name}': {id_val}")
            if hasattr(underlying_dataset, 'test_partner_policies'):
                print(f"    Test Partner Policies (Offset) ({len(underlying_dataset.test_partner_policies)}):")
                for name, id_val in underlying_dataset.test_partner_policies.items():
                    print(f"      '{name}': {id_val}")
            print("-" * (60 + len(" TRAINING CONFIGURATION ")))
        
            print(f"Trainer config: {json.dumps(config_to_log, indent=2)}")

    @staticmethod
    def overcooked_collate_fn(samples):
        # Unzip the samples into separate lists
        x, x_cond, policy_id, actions = zip(*samples)

        # Stack directly into tensors
        x = torch.stack(x) # (B F, H, W, C)
        x_cond = torch.stack(x_cond)  # (B, H, W, C)
        policy_id = torch.tensor(policy_id, dtype=torch.long)  # (B,)
        actions = torch.stack(actions)  # (B, K)
        assert actions.ndim == 2, f"Expected actions to be 2D tensor, got {actions.ndim}D tensor"
        return x, x_cond, policy_id, actions
    
    def train_one_step(self):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            x, x_cond, policy_id, actions = next(self.dl)
            x, x_cond, policy_id, actions = x.to(self.device), x_cond.to(self.device), policy_id.to(self.device), actions.to(self.device)

            x = rearrange(x, "b f h w c -> b (f c) h w")
            x_cond = rearrange(x_cond, "b h w c -> b c h w")

            
            if self.cond_drop_chance > 0: 
                batch_size = policy_id.shape[0]
                mask = torch.rand(batch_size, device=self.device) < self.cond_drop_chance
                policy_id = torch.where(mask, self.dummy_id*(torch.ones_like(policy_id)), policy_id)
                no_op_action = torch.ones_like(actions) * self.no_op_action
                actions = torch.where(mask.unsqueeze(-1), no_op_action, actions)

            with self.accelerator.autocast():
                loss = self.model(x, x_cond, task_embed=policy_id, action_embed=actions)
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()

                self.accelerator.backward(loss)
        
        if self.accelerator.is_local_main_process and not self.debug and self.wandb_enabled:
            wandb.log({
                "train/loss": total_loss,
                "train/step": self.step
            })
        
        return total_loss
    @torch.no_grad()
    def valid_one_step(self):
        if not self.valid_dl:
            return 0.0

        total_valid_loss = []
        total_unconditional_loss = []
        viz_output_dir = os.path.join(self.results_folder, "viz")
        os.makedirs(viz_output_dir, exist_ok=True)
        ch_output_dir = os.path.join(self.results_folder, "channels")
        os.makedirs(ch_output_dir, exist_ok=True)

        wandb_videos = {
            "predicted": None,
            "argmax": None,
            "reference": None,
            "unconditional": None,
            "no_op_test": None,
        }
        wandb_images = {
            "cond_image": None,
            "predicted_channels": None,
            "unconditional_channels": None,
        }
        wandb_actions = {
            "input_actions": None,
        }
        for x_org, x_cond, policy_id, actions in self.valid_dl:
            B, *_ = x_org.shape
            x = rearrange(x_org, "b f h w c -> b (f c) h w")
            x_cond = rearrange(x_cond, "b h w c -> b c h w")
            loss = self.model(x, x_cond, task_embed=policy_id, action_embed=actions)

            dummy_cond = torch.ones_like(policy_id)*self.dummy_id
            uncond_actions = torch.ones_like(actions) * self.no_op_action
            uncond_loss = self.model(x, x_cond, task_embed=dummy_cond, action_embed=uncond_actions)

            total_valid_loss.append(loss)
            total_unconditional_loss.append(uncond_loss)

            horizon = self.model.model.horizon
            C = self.model.model.C       
            
            pred_traj = self.ema.ema_model.sample(
                x_cond=x_cond,
                task_embed=policy_id,
                action_embed=actions,
                batch_size=B,
            )

            uncond_pred_traj = self.ema.ema_model.sample(
                x_cond=x_cond,
                task_embed=dummy_cond,
                action_embed=uncond_actions,
                batch_size=B,
            )

            no_op_traj = self.ema.ema_model.sample(
                x_cond=x_cond,
                task_embed=policy_id,
                action_embed=uncond_actions,
                batch_size=B,
            )
            # Process conditional trajectory
            pred_traj = rearrange(pred_traj, "b (f c) h w -> b f h w c", c=C, f=horizon)
            pred_traj = rearrange(pred_traj, "b f h w c -> b f w h c")
            pred_traj = pred_traj.cpu().numpy()
            argmax_pred = self.renderer.arg_max(pred_traj[..., :2])
            argmax_traj = np.concatenate([argmax_pred, pred_traj[..., 2:]], axis=-1)

            # Process unconditional trajectory
            uncond_pred_traj = rearrange(uncond_pred_traj, "b (f c) h w -> b f h w c", c=C, f=horizon)
            uncond_pred_traj = rearrange(uncond_pred_traj, "b f h w c -> b f w h c")
            uncond_pred_traj = uncond_pred_traj.cpu().numpy()

            #Process no-op trajectory
            no_op_traj = rearrange(no_op_traj, "b (f c) h w -> b f h w c", c=C, f=horizon)
            no_op_traj = rearrange(no_op_traj, "b f h w c -> b f w h c")
            no_op_traj = no_op_traj.cpu().numpy()

            grid_obs = rearrange(x_org, "b f h w c -> b f w h c").cpu().numpy()[0, 0]
            grid = self.renderer.extract_grid_from_obs(grid_obs)
            cond_img_path = os.path.join(viz_output_dir, f"cond_img_step_{self.step}.png")
            x_cond_print = rearrange(x_cond, "b c h w -> b w h c")
             
            for i in range(min(B, 4)):
                pred_traj_path = os.path.join(viz_output_dir, f"pred_traj_step_{self.step}_batch_{i}.mp4")
                self.renderer.render_trajectory_video(
                    pred_traj[i],
                    grid=grid,
                    output_dir=viz_output_dir,
                    video_path=pred_traj_path,
                    fps=1,
                    normalize=True,
                )

                arg_pred_traj_path = os.path.join(viz_output_dir, f"argmax_pred_traj_step_{self.step}_batch_{i}.mp4")
                self.renderer.render_trajectory_video(
                    argmax_traj[i],
                    grid=grid,
                    output_dir=viz_output_dir,
                    video_path=arg_pred_traj_path,
                    fps=1,
                    normalize=True,
                )
                ref_traj_path = os.path.join(viz_output_dir, f"reference_traj_step_{self.step}_batch_{i}.mp4")
                x_org_reshape = rearrange(x_org, "b f h w c -> b f w h c")
                self.renderer.render_trajectory_video(
                        x_org_reshape.cpu().numpy()[i], 
                        grid=grid, 
                        output_dir=viz_output_dir, 
                        video_path=ref_traj_path,
                        fps=1,
                        normalize=True,
                )

                uncond_traj_path = os.path.join(viz_output_dir, f"uncond_traj_step_{self.step}_batch_{i}.mp4")
                self.renderer.render_trajectory_video(
                    uncond_pred_traj[i],
                    grid=grid,
                    output_dir=viz_output_dir,
                    video_path=uncond_traj_path,
                    fps=1,
                    normalize=True,
                )

                no_op_traj_path = os.path.join(viz_output_dir, f"no_op_traj_step_{self.step}_batch_{i}.mp4")
                self.renderer.render_trajectory_video(
                    no_op_traj[i],
                    grid=grid,
                    output_dir=viz_output_dir,
                    video_path=no_op_traj_path,
                    fps=1,
                    normalize=True,
                )

                cond_img_path = os.path.join(viz_output_dir, f"cond_img_step_{self.step}_batch_{i}.png")
                self.renderer.save_obs_image(
                    x_cond_print.cpu().numpy()[i],
                    grid=grid,
                    file_path=cond_img_path,
                    normalize=True,
                )
                # Save channels visualization
                uncond_ch_path = os.path.join(ch_output_dir, f"uncond_channels_step_{self.step}_batch_{i}.png")
                self.renderer.visualize_all_channels(
                    obs=uncond_pred_traj[i, 0],
                    output_dir=uncond_ch_path
                )
                pred_ch_path = os.path.join(ch_output_dir, f"pred_channels_step_{self.step}_batch_{i}.png")
                self.renderer.visualize_all_channels(
                    obs=pred_traj[i, 0],
                    output_dir=pred_ch_path
                )  

                if i == 0:
                    wandb_videos["predicted"] = pred_traj_path
                    wandb_videos["argmax"] = arg_pred_traj_path
                    wandb_videos["reference"] = ref_traj_path
                    wandb_videos["unconditional"] = uncond_traj_path
                    wandb_videos["no_op_test"] = no_op_traj_path
                    wandb_images["cond_image"] = cond_img_path
                    wandb_images["predicted_channels"] = pred_ch_path
                    wandb_images["unconditional_channels"] = uncond_ch_path
                    wandb_actions["input_actions"] = actions.cpu().numpy()[i]

                if i + 1 == B:
                    ch_max = min(26, B)
                    for j in range(ch_max):
                        ch_path = os.path.join(ch_output_dir, f"pred_channels_step_{self.step}_batch_{i}_frame_{j}.png")
                        self.renderer.visualize_all_channels(
                            obs=pred_traj[i, j],
                            output_dir=ch_path
                        )
                        argmax_ch_path = os.path.join(ch_output_dir, f"arg_max_pred_channels_step_{self.step}_batch_{i}_frame_{j}.png")
                        self.renderer.visualize_all_channels(
                            obs=argmax_traj[i, j],
                            output_dir=argmax_ch_path
                        )
        valid_loss = sum(total_valid_loss) / len(total_valid_loss)
        uncond_loss = sum(total_unconditional_loss) / len(total_unconditional_loss)

        if self.wandb_enabled and self.accelerator.is_local_main_process and not self.debug:
            wandb_log_dict = {
                "evaluation/loss": valid_loss,
                "evaluation/unconditional_loss": uncond_loss,
            }
            for video_type, video_path in wandb_videos.items():
                if video_path and os.path.exists(video_path):
                    wandb_log_dict[f"evaluation/{video_type}_trajectory"] = wandb.Video(
                        video_path, 
                        fps=1, 
                        format="mp4",
                        caption=f"{video_type.capitalize()} trajectory at step {self.step}"
                    )
            for img_type, img_path in wandb_images.items():
                if img_path and os.path.exists(img_path):
                    wandb_log_dict[f"evaluation/{img_type}"] = wandb.Image(
                        img_path, 
                        caption=f"{img_type.replace('_', ' ').capitalize()} at step {self.step}"
                    )
            for action_type, actions_np in wandb_actions.items():
                if actions_np is not None:
                    columns = [f"Step {i+1}" for i in range(actions_np.shape[0])]
                    data = [actions_np.tolist()]
                    wandb_log_dict[f"evaluation/{action_type}"] = wandb.Table(columns=columns, data=data)
            wandb.log(wandb_log_dict, step=self.step)
        return valid_loss 

class ConceptTrainer(OvercookedEnvTrainer):
    def __init__(
            self,
            diffusion_model,
            train_set,
            valid_set,
            channels,
            dummy_policy_id,
            new_policy_id,
            guidance_w,
            embedding=None,
            **kwargs
    ):
        super().__init__(
            diffusion_model=diffusion_model,
            train_set=train_set,
            valid_set=valid_set,
            channels=channels,
            dummy_policy_id=dummy_policy_id,
            **kwargs
        )

        self.dummy_id = dummy_policy_id
        self.new_policy_id = new_policy_id
        self.embedding = embedding
        self.embedding_shape = embedding.shape if embedding is not None else None
        self.step = 0
        self.loss_history = []
        self.guidance_weight_history = []
        if guidance_w is None:
            self.guidance_weight = 1.0
            self.learnable_guidance = False
            print("Using fixed guidance weight of 1.0")
        else: # TODO: may want to support custom guidance weights instead of just ints / learnable parameters
            if isinstance(guidance_w, (int, float)):
                self.guidance_weight = guidance_w
                self.learnable_guidance = False
                print(f"Using fixed guidance weight of {guidance_w}")
            else:
                self.guidance_weight = nn.Parameter(torch.tensor(guidance_w, dtype=torch.float32, device=self.device), requires_grad=True)
                self.learnable_guidance = True
                print(f"Using learnable guidance weight initialized to {guidance_w}")
        
        
    def freeze_all_parameters(self):
        """Freeze all parameters in the model and EMA model."""
        for param in self.model.model.unet.parameters():
            param.requires_grad = False
        for param in self.ema.ema_model.model.unet.parameters():
            param.requires_grad = False
    def load_concept_checkpoint(self, ckpt_path):
        """
        Wrapper that first loads base weights (sans embedding),
        then replaces the embedding and rebuilds optimizer.
        """
        print(f"→ loading base weights from {ckpt_path}")
        self.load_model_except_embedding(ckpt_path)
        print("→ loading/reinitializing label_emb")
        self.load_embedding()
        print("✓ concept checkpoint loaded")

    def load_model_except_embedding(self, milestone):
        """Load model state except for the `label_emb` weights."""
        accelerator = self.accelerator
        device = accelerator.device

        if isinstance(milestone, int):
            load_name = f'model-{milestone}.pt' if milestone >= 0 else 'model_recent.pt'
            try:
                data = torch.load(str(self.results_folder / load_name), map_location=device)
            except Exception as e:
                if milestone >= 0:
                    raise e
                print("load model_rencent failed, try to load model_last_recent")
                data = torch.load(str(self.results_folder / 'model_last_recent.pt'), map_location=device)
        else:
            data = torch.load(str(milestone), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        filtered_model_state = {k: v for k, v in data['model'].items() if 'label_emb' not in k}
        missing, unexpected = model.load_state_dict(filtered_model_state, strict=False)
        assert any('label_emb.weight' in key for key in missing), \
            f"Expected a missing key containing 'label_emb.weight', got {missing}"
        assert len(unexpected) == 0, f"Unexpected keys found: {unexpected}"

        filtered_ema_state = {k: v for k, v in data['ema'].items() if 'label_emb' not in k}
        missing, unexpected = self.ema.load_state_dict(filtered_ema_state, strict=False)
        assert any('label_emb.weight' in key for key in missing), \
            f"Expected a missing key containing 'label_emb.weight', got {missing}"
        assert len(unexpected) == 0, f"Unexpected keys found: {unexpected}"

        if 'version' in data:
            print(f"loading from version {data['version']}, step {data['step']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    @torch.no_grad()
    def load_embedding(self):
        """Replace and unfreeze the `label_emb` weights, rebuild optimizer."""
        if self.embedding is not None:
            print(f"Loading provided embedding matrix of shape {tuple(self.embedding.shape)}")
            W = self.embedding.clone().detach().to(self.device)
        else:
            print("No embedding provided, initializing random embedding matrix")
            unet = self.model.model.unet
            D, C = unet.label_emb.weight.shape
            print(f"Initializing label_emb with shape ({D}, {C})")
            W = torch.randn(D, C, device=self.device)
            self.embedding_shape = W.shape

        new_unet_emb = nn.Embedding.from_pretrained(W, freeze=False).to(self.device)
        new_ema_emb  = nn.Embedding.from_pretrained(W, freeze=False).to(self.device)

        # Replace the existing label_emb with the new one
        unet = self.model.model.unet
        ema_unet = self.ema.ema_model.model.unet
        unet.label_emb = new_unet_emb
        ema_unet.label_emb = new_ema_emb

        # Unfreeze the label_emb parameters for training
        self.freeze_all_parameters()
        unet.label_emb.weight.requires_grad     = True
        ema_unet.label_emb.weight.requires_grad = True

        # Recreate the optimizer with only the label_emb parameters
        params_to_opt = [p for p in self.model.parameters() if p.requires_grad]
        if self.learnable_guidance:
            self.guidance_weight.requires_grad = True
            params_to_opt.append(self.guidance_weight)
            
        self.opt = Adam(params_to_opt, lr=self.train_lr, betas=self.adam_betas)

        expected_params = len(list(unet.label_emb.parameters()))
        if self.learnable_guidance:
            expected_params += 1
        
        actual_params = len(params_to_opt)
        assert actual_params == expected_params, \
            f"Expected {expected_params} parameters to optimize, got {actual_params}"

        self.model, self.opt, self.text_encoder = \
            self.accelerator.prepare(self.model, self.opt, self.text_encoder)

        print(f"Loaded embedding; now optimizing {sum(p.numel() for p in params_to_opt)} parameters "
              f"→ label_emb shape {unet.label_emb.weight.shape}")
    @torch.no_grad()
    def get_embedding(self):
        """Return the current `label_emb` weights from both UNet and EMA models."""
        unet, ema = self.model.model.unet, self.ema.ema_model.model.unet
        unet_embedding = unet.label_emb.weight.clone().detach().to(self.device)
        ema_embedding = ema.label_emb.weight.clone().detach().to(self.device)
        return unet_embedding, ema_embedding
    
    @torch.no_grad()
    def get_guidance_weight(self):
        return self.guidance_weight.clone().detach() if self.learnable_guidance else self.guidance_weight
    
    @torch.no_grad()
    def get_metrics(self):
        return {
            'loss': self.loss_history,
            'guidance_weight': self.guidance_weight_history,
        }
    
    @torch.no_grad()
    def print_embedding(self):
        pid = self.new_policy_id
        unet, ema = self.model.model.unet, self.ema.ema_model.model.unet
        print(f"Current embedding for policy ID {self.new_policy_id}:")
        unet_sum_embedding = unet.label_emb.weight.clone().sum().item()
        print(f"Unet Sum of embedding weights: {unet_sum_embedding}")
        ema_sum_embedding = ema.label_emb.weight.clone().sum().item()
        print(f"Ema Sum of embedding weights: {ema_sum_embedding}")
        row = unet.label_emb.weight[pid]
        print(f"Embedding sum for policy {pid}: {row.sum().item()}")
        

    def train_one_step(self):
        total_loss = 0.0
        for _ in range(self.gradient_accumulate_every):
            x, x_cond, _ = next(self.dl)
            x, x_cond = x.to(self.device), x_cond.to(self.device)

            if self.learnable_guidance:
                print(f"[DEBUG] Current guidance weight: {self.guidance_weight}")
                self.guidance_weight_history.append(self.guidance_weight.clone().detach().cpu().numpy())
            else:
                self.guidance_weight_history.append(self.guidance_weight)

            with self.accelerator.autocast():
                loss = self.other_loss(x, x_cond)
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()
                self.accelerator.backward(loss)
                self.loss_history.append(loss.item())
        return total_loss
    def _loss(self, x_start, x_cond):
        batch_size = x_start.shape[0]
        dummy_cond = torch.full((batch_size,), self.dummy_id, device=self.device, dtype=torch.long)

        x_start = rearrange(x_start, "b f h w c -> b (f c) h w")
        x_cond  = rearrange(x_cond,  "b h w c -> b c h w")
        x_start = self.model.normalize(x_start)  # TODO: REMOVE THIS
        if not x_start.requires_grad:
            x_start = x_start.requires_grad_(True)
        t = torch.randint(0, self.model.num_timesteps,(batch_size,), device=x_start.device).long()

        # unconditional loss
        uncond_loss = self.model.p_losses(x_start, t, x_cond, dummy_cond,None, None, None)

        # single-concept shortcut
        if self.embedding_shape[0] - 1 == 1:
            policy_id = torch.full((batch_size,), self.new_policy_id, device=self.device, dtype=torch.long)
            cond_loss = self.model.p_losses(x_start, t, x_cond, policy_id,None, None, None)
            total_loss = uncond_loss + self.guidance_weight * (cond_loss - uncond_loss)
            print(f"[DEBUG] Single concept loss: {total_loss.item()}")
            return total_loss

        # multi-concept case
        concept_ids = [c_id for c_id in range(self.embedding_shape[0]) if c_id != self.dummy_id]

        # pick your weights vector
        if self.learnable_guidance:
            weights = self.guidance_weight
        else:
            weights = torch.full((len(concept_ids),),self.guidance_weight,device=self.device)
        weighted_cond_loss = torch.tensor(0.0, device=self.device)
        for idx, c_id in enumerate(concept_ids):
            cond_id = torch.full((batch_size,), c_id,device=self.device, dtype=torch.long)
            cond_loss = self.model.p_losses(x_start, t, x_cond, cond_id,None, None, None)
            w_i = weights[idx]
            weighted_cond_loss = weighted_cond_loss + w_i * cond_loss

        print(f"[DEBUG] Weighted conditional loss: {weighted_cond_loss.item()}")
        print(f"[DEBUG] Unconditional loss: {uncond_loss.item()}")

        total_loss = uncond_loss + weighted_cond_loss
        return total_loss
    
    def other_loss(self, x_start, x_cond):
        import torch.utils.checkpoint as cp 
        batch_size = x_start.shape[0]
        dummy_cond = torch.full((batch_size,), self.dummy_id, device=self.device, dtype=torch.long)

        x_start = rearrange(x_start, "b f h w c -> b (f c) h w")
        x_cond  = rearrange(x_cond,  "b h w c -> b c h w")
        x_start = self.model.normalize(x_start)  # TODO: REMOVE THIS
        if not x_start.requires_grad:
            x_start = x_start.requires_grad_(True)
        t = torch.randint(0, self.model.num_timesteps,(batch_size,), device=x_start.device).long()

        def chkpt_fn(x_start, t, x_cond, policy_id):
            return self.model.p_losses(x_start, t, x_cond, policy_id, None, None, None)

        # unconditional loss
        uncond_loss = cp.checkpoint(chkpt_fn, x_start, t, x_cond, dummy_cond)
        # uncond_loss = self.model.p_losses(x_start, t, x_cond, dummy_cond,None, None, None)

        # single-concept shortcut
        if self.embedding_shape[0] - 1 == 1:
            policy_id = torch.full((batch_size,), self.new_policy_id, device=self.device, dtype=torch.long)
            # cond_loss = self.model.p_losses(x_start, t, x_cond, policy_id,None, None, None)
            cond_loss = cp.checkpoint(chkpt_fn, x_start, t, x_cond, policy_id)
            total_loss = uncond_loss + self.guidance_weight * (cond_loss - uncond_loss)
            print(f"[DEBUG] Single concept loss: {total_loss.item()}")
            return total_loss

        # multi-concept case
        concept_ids = [c_id for c_id in range(self.embedding_shape[0]) if c_id != self.dummy_id]

        # pick your weights vector
        if self.learnable_guidance:
            weights = self.guidance_weight
        else:
            weights = torch.full((len(concept_ids),),self.guidance_weight,device=self.device)
        weighted_cond_loss = torch.tensor(0.0, device=self.device)
        for idx, c_id in enumerate(concept_ids):
            cond_id = torch.full((batch_size,), c_id,device=self.device, dtype=torch.long)

            # cond_loss = self.model.p_losses(x_start, t, x_cond, cond_id,None, None, None)
            cond_loss = cp.checkpoint(chkpt_fn, x_start, t, x_cond, cond_id)

            w_i = weights[idx]
            weighted_cond_loss = weighted_cond_loss + w_i * cond_loss

        print(f"[DEBUG] Weighted conditional loss: {weighted_cond_loss.item()}")
        print(f"[DEBUG] Unconditional loss: {uncond_loss.item()}")

        total_loss = uncond_loss + weighted_cond_loss
        return total_loss
    
class OvercookedActionProposal(Trainer):
    def __init__(self,
                 diffusion_model,
                 train_set,
                 valid_set,
                 channels,
                 *,
                 train_batch_size=1,
                 valid_batch_size=1,
                 gradient_accumulate_every=1,
                 train_lr=0.0001,
                 train_num_steps=100000,
                 ema_update_every=10,
                 ema_decay=0.995,
                 adam_betas=(0.9, 0.99),
                 save_and_sample_every=5000,
                 num_samples=3,
                 results_folder='./results',
                 amp=True,
                 fp16=True,
                 split_batches=True,
                 cond_drop_chance=0.0,
                 save_milestone=True,
                 debug=False,
                 wandb_enabled=False,
                 wandb_project="action_proposal_diffusion",
                 wandb_entity="social-rl",
                 wandb_run_name=None
                ):
        
        super().__init__(diffusion_model,
                         None,
                         None,
                         train_set=train_set,
                         valid_set=valid_set,
                         channels=channels,
                         train_batch_size=train_batch_size,
                         valid_batch_size=valid_batch_size,
                         gradient_accumulate_every=gradient_accumulate_every,
                         train_lr=train_lr,
                         train_num_steps=train_num_steps,
                         ema_update_every=ema_update_every,
                         ema_decay=ema_decay,
                         adam_betas=adam_betas,
                         save_and_sample_every=save_and_sample_every,
                         num_samples=num_samples,
                         results_folder=results_folder,
                         amp=amp,
                         fp16=fp16,
                         split_batches=split_batches,
                         calculate_fid=False,
                         cond_drop_chance=cond_drop_chance,
                         save_milestone=save_milestone,
                         embed_preprocessed=True)
        
        # Dataset Setup
        dl = DataLoader(self.ds, batch_size = self.batch_size, shuffle = True, pin_memory = True, num_workers = 4, collate_fn=self.action_overcooked_collate_fn)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        if self.valid_ds:
            valid_dl = DataLoader(self.valid_ds, batch_size=self.valid_batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=self.action_overcooked_collate_fn)
            self.valid_dl = self.accelerator.prepare(valid_dl)
        else:
            self.valid_dl = None
        
        self.debug = debug
        self.renderer = OvercookedSampleRenderer()
        self.wandb_enabled = wandb_enabled
        
        config_to_log = {
            # Trainer parameters
            "learning_rate": train_lr,
            "train_batch_size": train_batch_size,
            "valid_batch_size": valid_batch_size,
            "gradient_accumulate_every": gradient_accumulate_every,
            "train_num_steps": train_num_steps,
            "ema_update_every": ema_update_every,
            "ema_decay": ema_decay,
            "adam_beta1": adam_betas[0],
            "adam_beta2": adam_betas[1],
            "save_and_sample_every": save_and_sample_every,
            "num_validation_samples": num_samples,
            "amp_enabled": amp,
            "fp16_enabled": fp16,
            "accelerator_split_batches": split_batches,
            "cond_drop_chance": cond_drop_chance,
            "save_milestone_checkpoints": save_milestone,
            "debug_mode": self.debug,
            
            # Dataset parameters
            "dataset_type": train_set.__class__.__name__,
            "dataset_horizon": train_set.horizon if hasattr(train_set, 'horizon') else 'N/A',
            "dataset_max_path_length": train_set.max_path_length if hasattr(train_set, 'max_path_length') else 'N/A',
            "dataset_use_padding": train_set.use_padding if hasattr(train_set, 'use_padding') else 'N/A',
            "dataset_max_rtg": train_set.max_rtg if hasattr(train_set, 'max_rtg') else 'N/A',
            "dataset_rtg_normalization_factor": train_set.rtg_normalization_factor if hasattr(train_set, 'rtg_normalization_factor') else 'N/A',
            "dataset_length": len(train_set),

            # Diffusion model (GoalGaussianDiffusion) parameters
            "diffusion_model_type": diffusion_model.__class__.__name__,
            "diffusion_total_input_channels": diffusion_model.channels, # Total channels for the diffusion model
            "diffusion_image_height": diffusion_model.image_size[0],
            "diffusion_image_width": diffusion_model.image_size[1],
            "diffusion_objective": diffusion_model.objective,
            "diffusion_loss_type": diffusion_model.loss_type,
            "diffusion_timesteps_total": diffusion_model.num_timesteps,
            "diffusion_sampling_timesteps": diffusion_model.sampling_timesteps,
            "diffusion_guidance_weight": diffusion_model.guidance_weight,
            "diffusion_ddim_sampling_eta": diffusion_model.ddim_sampling_eta,

            # Unet parameters
            "unet_model_channels": diffusion_model.model.unet.model_channels if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_res_blocks": diffusion_model.model.unet.num_res_blocks if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_attention_resolutions": str(diffusion_model.model.unet.attention_resolutions) if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_channel_mult": str(diffusion_model.model.unet.channel_mult) if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_classes": diffusion_model.model.unet.num_classes if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_attention_resolutions": str(diffusion_model.model.unet.attention_resolutions) if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_cross_attention_enabled": diffusion_model.model.unet.cross if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_heads": diffusion_model.model.unet.num_heads if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_head_channels": diffusion_model.model.unet.num_head_channels if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_classes": diffusion_model.model.unet.num_classes if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_use_scale_shift_norm": diffusion_model.model.unet.use_scale_shift_norm if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_dropout": diffusion_model.model.unet.dropout if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_use_checkpointing": diffusion_model.model.unet.use_checkpoint if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_cross_attention_heads": diffusion_model.model.unet.cross_attention_heads if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_cross_attention_dim_head": diffusion_model.model.unet.cross_attention_dim_head if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_spatiotemporal_attention": diffusion_model.model.unet.spatiotemporal_attention if hasattr(diffusion_model.model, 'unet') else 'N/A', 
            "unet_video_cond_dim":str(diffusion_model.model.unet.video_cond_dim) if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_image_cond_dim": str(diffusion_model.model.unet.image_cond_dim) if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_reward_dim": diffusion_model.model.unet.reward_dim if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_action_horizon": diffusion_model.model.unet.action_horizon if hasattr(diffusion_model.model, 'unet') else 'N/A',
            "unet_num_actions": diffusion_model.model.unet.num_actions if hasattr(diffusion_model.model, 'unet') else 'N/A',
        }
        
        
        if not self.debug and self.wandb_enabled and self.accelerator.is_local_main_process:
            final_wandb_run_name = wandb_run_name if wandb_run_name else f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project=wandb_project, entity=wandb_entity, name=final_wandb_run_name, config=config_to_log)
        self.num_classes = 6

    @staticmethod
    def action_overcooked_collate_fn(samples):
        # Unzip the samples into separate lists
        future_actions, current_obs, obs_history, reward_to_go = zip(*samples)

        # Stack directly into tensors
        future_actions = torch.stack(future_actions) #(B, Horizon, A)
        current_obs = torch.stack(current_obs)  # (B, H, W, C)
        obs_history = torch.stack(obs_history)  # (B, H, W, C)
        reward_to_go = torch.stack(reward_to_go)  # (B,)
        return future_actions, current_obs, obs_history, reward_to_go

    def train_one_step(self):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            future_actions, current_obs, obs_history, rtg = next(self.dl) 
            x, image_cond, history_cond, rtg_cond = future_actions.to(self.device), current_obs.to(self.device), obs_history.to(self.device), rtg.to(self.device)
            
            image_cond = rearrange(image_cond, "b h w c -> b c h w")
            history_cond = rearrange(history_cond, "b f h w c -> b c f h w")
            x = F.one_hot(x, num_classes=self.num_classes).float() #[B, Horizon, action dim, num_classes]
            x = rearrange(x, "b f a n -> b (f n) a 1") #[B, Horizon * num_classes, action dim, 1]

            if self.cond_drop_chance > 0: 
                batch_size = x.shape[0]
                mask = torch.rand(batch_size, device=self.device) < self.cond_drop_chance
                image_cond = torch.where(mask.view(-1, 1, 1, 1), torch.zeros_like(image_cond), image_cond)
                history_cond = torch.where(mask.view(-1, 1, 1, 1, 1), torch.zeros_like(history_cond), history_cond)
                rtg_cond = torch.where(mask.view(-1, 1), torch.zeros_like(rtg_cond), rtg_cond)

            with self.accelerator.autocast():
                loss = self.model(x, None, image_embed=image_cond, history_embed=history_cond, reward_embed=rtg_cond)
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()

                self.accelerator.backward(loss)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"WARNING: NaN found in gradient of {name}")
                        if torch.isinf(param.grad).any():
                            print(f"WARNING: Inf found in gradient of {name}")
        
        if self.accelerator.is_local_main_process and not self.debug and self.wandb_enabled:
            wandb.log({
                "train/loss": total_loss,
                "train/step": self.step
            })
        
        return total_loss
    
    
    @torch.no_grad()
    def valid_one_step(self):
        if not self.valid_dl:
            return 0.0

        total_valid_loss = []
        viz_output_dir = os.path.join(self.results_folder, "viz")
        os.makedirs(viz_output_dir, exist_ok=True)

        wandb_videos = {
            "recon_action_sequence": None,
            "low_reward_action_sequence": None,
            "high_reward_action_sequence": None,
            "medium_reward_action_sequence": None,
            "ref_action_sequence": None,
            "observation_history": None,
        }

        for j, (actions, current_frame, obs_history, rtg) in enumerate(self.valid_dl):
            B, horizon, *_ = actions.shape
            current_frame_rearranged = rearrange(current_frame, "b h w c -> b c h w")
            obs_history_rearranged = rearrange(obs_history, "b f h w c -> b c f h w")
            
            # Convert actions to one-hot for loss calculation
            actions_one_hot = F.one_hot(actions, num_classes=self.num_classes).float()
            actions_one_hot = rearrange(actions_one_hot, "b f a n -> b (f n) a 1")

            # Calculate loss
            loss = self.model(actions_one_hot, None, image_embed=current_frame_rearranged, history_embed=obs_history_rearranged, reward_embed=rtg)
            total_valid_loss.append(loss)

            recon_actions = self.ema.ema_model.sample(
                image_embed=current_frame_rearranged,
                history_embed=obs_history_rearranged,
                reward_embed=rtg,
                batch_size=B,
            )

            low_reward_actions = self.ema.ema_model.sample(
                image_embed=current_frame_rearranged,
                history_embed=obs_history_rearranged,
                reward_embed=torch.ones_like(rtg)*50,
                batch_size=B,
            )

            high_reward_actions = self.ema.ema_model.sample(
                image_embed=current_frame_rearranged,
                history_embed=obs_history_rearranged,
                reward_embed=torch.ones_like(rtg)*100,
                batch_size=B,
            )

            medium_reward_actions = self.ema.ema_model.sample(
                image_embed=current_frame_rearranged,
                history_embed=obs_history_rearranged,
                reward_embed=torch.ones_like(rtg)*75,
                batch_size=B,
            )
            
            # Convert predictions back to discrete actions
            # pred_actions shape: [B, (horizon * num_classes), 1, 1]
            recon_actions = recon_actions.view(B, horizon, self.num_classes)
            low_reward_actions = low_reward_actions.view(B, horizon, self.num_classes)
            high_reward_actions = high_reward_actions.view(B, horizon, self.num_classes)
            medium_reward_actions = medium_reward_actions.view(B, horizon, self.num_classes)


            recon_actions_path = os.path.join(viz_output_dir, f"recon_actions_step_{self.step}.png")
            low_reward_actions_path = os.path.join(viz_output_dir, f"low_reward_actions_step_{self.step}.png")
            high_reward_actions_path = os.path.join(viz_output_dir, f"high_reward_actions_step_{self.step}.png")
            medium_reward_actions_path = os.path.join(viz_output_dir, f"medium_reward_actions_step_{self.step}.png")
            
            
            self.renderer.visualize_action_logits(recon_actions, recon_actions_path)
            self.renderer.visualize_action_logits(low_reward_actions, low_reward_actions_path)
            self.renderer.visualize_action_logits(high_reward_actions, high_reward_actions_path)
            self.renderer.visualize_action_logits(medium_reward_actions, medium_reward_actions_path)
            
            if self.wandb_enabled and self.accelerator.is_local_main_process and j == 0:
                wandb.log({
                    "evaluation/recon_action_logits": wandb.Image(recon_actions_path, caption=f"Reconstructed Action Logits at step {self.step}"),
                    "evaluation/low_reward_action_logits": wandb.Image(low_reward_actions_path, caption=f"Low Reward Action Logits at step {self.step}"),
                    "evaluation/high_reward_action_logits": wandb.Image(high_reward_actions_path, caption=f"High Reward Action Logits at step {self.step}"),
                    "evaluation/medium_reward_action_logits": wandb.Image(medium_reward_actions_path, caption=f"Medium Reward Action Logits at step {self.step}"),
                }, step=self.step)
            
            recon_actions = torch.argmax(recon_actions, dim=-1)  # [B, horizon]
            low_reward_actions = torch.argmax(low_reward_actions, dim=-1)
            high_reward_actions = torch.argmax(high_reward_actions, dim=-1)
            medium_reward_actions = torch.argmax(medium_reward_actions, dim=-1)

            # Visualize action sequences for each batch
            current_frame = rearrange(current_frame, "b h w c -> b w h c")
            for i in range(min(B, 4)):  # Limit to 4 samples
                # Create action sequence video
                recon_action_video_path = os.path.join(viz_output_dir, 
                                            f"recon_action_seq_step_{self.step}_batch_{i}.mp4")
                self.renderer.visualize_action_sequence_video(
                    current_frame[i].cpu().numpy(),  
                    recon_actions[i, :].cpu(),      
                    recon_action_video_path,
                    player_idx=0,            
                    fps=1                    
                )

                low_reward_action_video_path = os.path.join(
                    viz_output_dir, f"low_reward_action_seq_step_{self.step}_batch_{i}.mp4"
                )
                self.renderer.visualize_action_sequence_video(
                    current_frame[i].cpu().numpy(),
                    low_reward_actions[i, :].cpu(),
                    low_reward_action_video_path,
                    player_idx=0,
                    fps=1
                )

                high_reward_action_video_path = os.path.join(
                    viz_output_dir, f"high_reward_action_seq_step_{self.step}_batch_{i}.mp4"
                )
                self.renderer.visualize_action_sequence_video(
                    current_frame[i].cpu().numpy(),
                    high_reward_actions[i, :].cpu(),
                    high_reward_action_video_path,
                    player_idx=0,
                    fps=1
                )

                medium_reward_action_video_path = os.path.join(
                    viz_output_dir, f"medium_reward_action_seq_step_{self.step}_batch_{i}.mp4"
                )
                self.renderer.visualize_action_sequence_video(
                    current_frame[i].cpu().numpy(),
                    medium_reward_actions[i, :].cpu(),
                    medium_reward_action_video_path,
                    player_idx=0,
                    fps=1
                )

                ref_action_video_path = os.path.join(
                    viz_output_dir, f"ref_action_seq_step_{self.step}_batch_{i}.mp4"
                )
                self.renderer.visualize_action_sequence_video(
                    current_frame[i].cpu().numpy(),
                    actions[i, :].cpu().numpy(),
                    ref_action_video_path,
                    player_idx=0,
                    fps=1
                )

                obs_history_video_path = os.path.join(
                    viz_output_dir, f"obs_history_step_{self.step}_batch_{i}.mp4"
                )
                self.renderer.render_trajectory_video(
                    obs_history[i].cpu().numpy(),
                    grid=self.renderer.extract_grid_from_obs(obs_history[i, 0]),
                    video_path=obs_history_video_path,
                    output_dir=viz_output_dir,
                    normalize=True,
                    fps=1,
                )
                
                if i == 0:
                    wandb_videos["recon_action_sequence"] = recon_action_video_path
                    wandb_videos["low_reward_action_sequence"] = low_reward_action_video_path
                    wandb_videos["high_reward_action_sequence"] = high_reward_action_video_path
                    wandb_videos["medium_reward_action_sequence"] = medium_reward_action_video_path
                    wandb_videos["ref_action_sequence"] = ref_action_video_path
                    wandb_videos["observation_history"] = obs_history_video_path
                
        valid_loss = sum(total_valid_loss) / len(total_valid_loss) if total_valid_loss else 0.0

        if self.wandb_enabled and self.accelerator.is_local_main_process and not self.debug:
            wandb_log_dict = {
                "evaluation/loss": valid_loss,
            }
            for video_type, video_path in wandb_videos.items():
                if video_path and os.path.exists(video_path):
                    wandb_log_dict[f"evaluation/{video_type}"] = wandb.Video(
                        video_path, 
                        fps=2, 
                        format="mp4",
                        caption=f"{video_type.replace('_', ' ').capitalize()} at step {self.step}"
                    )
            
            wandb.log(wandb_log_dict, step=self.step)
        
        return valid_loss
    
        
