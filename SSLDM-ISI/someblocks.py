import math
import operator
import functools

import einops
from tqdm.auto import tqdm
from functools import partial, wraps
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange



# helper functions

def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def divisible_by(numer, denom):
    return (numer % denom) == 0


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(val, length=None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output


def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255


def module_device(module):
    return next(module.parameters()).device


def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def pad_tuple_to_length(t, length, fillvalue=None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))


# helper classes

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


# tensor helpers

def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, dim=-1)


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def masked_mean(t, *, dim, mask=None):
    if not exists(mask):
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


def resize_video_to(
        video,
        target_image_size,
        target_frames=None,
        clamp_range=None,
        mode='nearest'
):
    orig_video_size = video.shape[-1]

    frames = video.shape[2]
    target_frames = default(target_frames, frames)

    target_shape = (target_frames, target_image_size, target_image_size)

    if tuple(video.shape[-3:]) == target_shape:
        return video

    out = F.interpolate(video, target_shape, mode=mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out


def scale_video_time(
        video,
        downsample_scale=1,
        mode='nearest'
):
    if downsample_scale == 1:
        return video

    image_size, frames = video.shape[-1], video.shape[-3]
    assert divisible_by(frames,
                        downsample_scale), f'trying to temporally downsample a conditioning video frames of length {frames} by {downsample_scale}, however it is not neatly divisible'

    target_frames = frames // downsample_scale

    resized_video = resize_video_to(
        video,
        image_size,
        target_frames=target_frames,
        mode=mode
    )

    return resized_video


# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# norms and residuals

class LayerNorm(nn.Module):
    def __init__(self, dim, stable=False):
        super().__init__()
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, stable=False):
        super().__init__()
        self.stable = stable
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)


# rearranging

class RearrangeTimeCentric(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = rearrange(x, 'b c f ... -> b ... f c')
        x, ps = pack([x], '* f c')

        x = self.fn(x)

        x, = unpack(x, ps, '* f c')
        x = rearrange(x, 'b ... f c -> b c f ...')
        return x


# attention pooling

class PerceiverAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            scale=8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            num_latents=64,
            num_latents_mean_pooled=4,  # number of latents derived from mean pooled representation of the sequence
            max_seq_len=512,
            ff_mult=4
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n=num_latents_mean_pooled)
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device=device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        return latents


# main contribution from make-a-video - pseudo conv3d
# axial space-time convolutions, but made causal to keep in line with the design decisions of imagen-video paper

class Conv3d(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            kernel_size=3,
            *,
            temporal_kernel_size=None,
            **kwargs
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)

        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size=temporal_kernel_size) if kernel_size > 1 else None
        self.kernel_size = kernel_size

        if exists(self.temporal_conv):
            nn.init.dirac_(self.temporal_conv.weight.data)  # initialized to be identity
            nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(
            self,
            x,
            ignore_time=False
    ):
        b, c, *_, h, w = x.shape

        is_video = x.ndim == 5
        ignore_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) c h w')

        x = self.spatial_conv(x)

        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', b=b)

        if ignore_time or not exists(self.temporal_conv):
            return x

        x = rearrange(x, 'b c f h w -> (b h w) c f')

        # causal temporal convolution - time is causal in imagen-video

        if self.kernel_size > 1:
            x = F.pad(x, (self.kernel_size - 1, 0))

        x = self.temporal_conv(x)

        x = rearrange(x, '(b h w) c f -> b c f h w', h=h, w=w)

        return x


# attention

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            dim_head=64,
            heads=8,
            causal=False,
            context_dim=None,
            rel_pos_bias=False,
            rel_pos_bias_mlp_depth=2,
            init_zero=False,
            scale=8
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal

        self.rel_pos_bias = DynamicPositionBias(dim=dim, heads=heads,
                                                depth=rel_pos_bias_mlp_depth) if rel_pos_bias else None

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_attn_bias = nn.Parameter(torch.randn(heads))

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(
            context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

        if init_zero:
            nn.init.zeros_(self.to_out[-1].g)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            attn_bias=None
    ):

        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b=b), self.null_kv.unbind(dim=-2))
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim=-1)
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        # relative positional encoding (T5 style)

        if not exists(attn_bias) and exists(self.rel_pos_bias):
            attn_bias = self.rel_pos_bias(n, device=device, dtype=q.dtype)

        if exists(attn_bias):
            null_attn_bias = repeat(self.null_attn_bias, 'h -> h n 1', n=n)
            attn_bias = torch.cat((null_attn_bias, attn_bias), dim=-1)
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


# pseudo conv2d that uses conv3d but with kernel size of 1 across frames dimension

def Conv2d(dim_in, dim_out, kernel, stride=1, padding=0, **kwargs):
    kernel = cast_tuple(kernel, 2)
    stride = cast_tuple(stride, 2)
    padding = cast_tuple(padding, 2)

    if len(kernel) == 2:
        kernel = (1, *kernel)

    if len(stride) == 2:
        stride = (1, *stride)

    if len(padding) == 2:
        padding = (0, *padding)

    return nn.Conv3d(dim_in, dim_out, kernel, stride=stride, padding=padding, **kwargs)


class Pad(nn.Module):
    def __init__(self, padding, value=0.):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, x):
        return F.pad(x, self.padding, value=self.value)


# decoder

def Upsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        Conv2d(dim, dim_out, 3, padding=1)
    )


class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU()
        )

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, f, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, f, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        out = self.net(x)
        frames = x.shape[2]
        out = rearrange(out, 'b c f h w -> (b f) c h w')
        out = self.pixel_shuffle(out)
        return rearrange(out, '(b f) c h w -> b c f h w', f=frames)


def Downsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c f (h p1) (w p2) -> b (c p1 p2) f h w', p1=2, p2=2),
        Conv2d(dim * 4, dim_out, 1)
    )


# temporal up and downsamples

class TemporalPixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out=None, stride=2):
        super().__init__()
        self.stride = stride
        dim_out = default(dim_out, dim)
        conv = nn.Conv1d(dim, dim_out * stride, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU()
        )

        self.pixel_shuffle = Rearrange('b (c r) n -> b c (n r)', r=stride)

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, f = conv.weight.shape
        conv_weight = torch.empty(o // self.stride, i, f)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.stride)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b h w) c f')
        out = self.net(x)
        out = self.pixel_shuffle(out)
        return rearrange(out, '(b h w) c f -> b c f h w', h=h, w=w)


def TemporalDownsample(dim, dim_out=None, stride=2):
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (f p) h w -> b (c p) f h w', p=stride),
        Conv2d(dim * stride, dim_out, 1)
    )


# positional embedding

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            groups=8,
            norm=True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = Conv3d(dim, dim_out, 3, padding=1)

    def forward(
            self,
            x,
            scale_shift=None,
            ignore_time=False
    ):
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x, ignore_time=ignore_time)


class ResnetBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            *,
            cond_dim=None,
            time_cond_dim=None,
            groups=8,
            linear_attn=False,
            use_gca=False,
            squeeze_excite=False,
            **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = attn_klass(
                dim=dim_out,
                context_dim=cond_dim,
                **attn_kwargs
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.gca = GlobalContext(dim_in=dim_out, dim_out=dim_out) if use_gca else Always(1)

        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()

    def forward(
            self,
            x,
            time_emb=None,
            cond=None,
            ignore_time=False
    ):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, ignore_time=ignore_time)

        if exists(self.cross_attn):
            assert exists(cond)
            h = rearrange(h, 'b c ... -> b ... c')
            h, ps = pack([h], 'b * c')

            h = self.cross_attn(h, context=cond) + h

            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b ... c -> b c ...')

        h = self.block2(h, scale_shift=scale_shift, ignore_time=ignore_time)

        h = h * self.gca(h)

        return h + self.res_conv(x)


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            context_dim=None,
            dim_head=64,
            heads=8,
            norm_context=False,
            scale=8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads).contiguous(), (q, k, v))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b h 1 d', h=self.heads, b=b), self.null_kv.unbind(dim=-2))

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
        return self.to_out(out)


class LinearCrossAttention(CrossAttention):
    def forward(self, x, context, mask=None):
        x = rearrange(x,"(b t) c w h -> b c t w h",t =16).contiguous()
        h = rearrange(x, 'b c ... -> b ... c').contiguous()

        x, ps = pack([h], 'b * c')
        b, n, device = *x.shape[:2], x.device
        # print(x.shape)
        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads).contiguous(), (q, k, v))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> (b h) 1 d', h=self.heads, b=b).contiguous(), self.null_kv.unbind(dim=-2))

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # masking

        max_neg_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b n -> b n 1').contiguous()
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads).contiguous()
        out = self.to_out(out)

        out, = unpack(out, ps, 'b * c')
        out = rearrange(out, 'b ... c -> b c ...').contiguous()

        return rearrange(out,"b c t w h -> (b t) c w h").contiguous()

class LinearAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            heads=8,
            dropout=0.05,
            context_dim=None,
            **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            Conv2d(dim, inner_dim, 1, bias=False),
            Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            Conv2d(dim, inner_dim, 1, bias=False),
            Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            Conv2d(dim, inner_dim, 1, bias=False),
            Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim),
                                        nn.Linear(context_dim, inner_dim * 2, bias=False)) if exists(
            context_dim) else None

        self.to_out = nn.Sequential(
            Conv2d(inner_dim, dim, 1, bias=False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context=None):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=h), (q, k, v))

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim=-1)
            ck, cv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (ck, cv))
            k = torch.cat((k, ck), dim=-2)
            v = torch.cat((v, cv), dim=-2)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)

class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
            self,
            *,
            dim_in,
            dim_out
    ):
        super().__init__()
        self.to_k = Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = map(lambda t: rearrange(t, 'b n ... -> b n (...)'), (x, context))
        out = einsum('b i n, b c n -> b c i', context.softmax(dim=-1), x)
        out = rearrange(out, '... -> ... 1 1')
        return self.net(out)


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False)
    )


class TimeTokenShift(nn.Module):
    def forward(self, x):
        if x.ndim != 5:
            return x

        x, x_shift = x.chunk(2, dim=1)
        x_shift = F.pad(x_shift, (0, 0, 0, 0, 1, -1), value=0.)
        return torch.cat((x, x_shift), dim=1)


def ChanFeedForward(dim, mult=2,
                    time_token_shift=True):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return Sequential(
        ChanLayerNorm(dim),
        Conv2d(dim, hidden_dim, 1, bias=False),
        nn.GELU(),
        TimeTokenShift() if time_token_shift else None,
        ChanLayerNorm(hidden_dim),
        Conv2d(hidden_dim, dim, 1, bias=False)
    )


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth=1,
            heads=8,
            dim_head=32,
            ff_mult=2,
            ff_time_token_shift=True,
            context_dim=None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, context_dim=context_dim),
                ChanFeedForward(dim=dim, mult=ff_mult, time_token_shift=ff_time_token_shift)
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = rearrange(x, 'b c ... -> b ... c')
            x, ps = pack([x], 'b * c')

            x = attn(x, context=context) + x

            x, = unpack(x, ps, 'b * c')
            x = rearrange(x, 'b ... c -> b c ...')

            x = ff(x) + x
        return x


class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth=1,
            heads=8,
            dim_head=32,
            ff_mult=2,
            ff_time_token_shift=True,
            context_dim=None,
            **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim=dim, heads=heads, dim_head=dim_head, context_dim=context_dim),
                ChanFeedForward(dim=dim, mult=ff_mult, time_token_shift=ff_time_token_shift)
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x


class CrossEmbedLayer(nn.Module):
    def __init__(
            self,
            dim_in,
            kernel_sizes,
            dim_out=None,
            stride=2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)

class DynamicPositionBias(nn.Module):
    def __init__(
            self,
            dim,
            *,
            heads,
            depth
    ):
        super().__init__()
        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            LayerNorm(dim),
            nn.SiLU()
        ))

        for _ in range(max(depth - 1, 0)):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                LayerNorm(dim),
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    def forward(self, n, device, dtype):
        i = torch.arange(n, device=device)
        j = torch.arange(n, device=device)

        indices = rearrange(i, 'i -> i 1') - rearrange(j, 'j -> 1 j')
        indices += (n - 1)

        pos = torch.arange(-n + 1, n, device=device, dtype=dtype)
        pos = rearrange(pos, '... -> ... 1')

        for layer in self.mlp:
            pos = layer(pos)

        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias
    



class TemporalShift(nn.Module):
    def __init__(self, net=None, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        if net==None:
            net = nn.Identity()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
            print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)




class ResnetBlock_3d(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        x = rearrange(x,"(b t) c w h -> b c t w h",t=16)
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        h + self.res_conv(x)
        return rearrange(h,"b c t w h -> (b t) c w h")



