"""Randomly initialized pixel-space conditional diffusion architecture."""
import math
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    resolution: int = 64
    text_dim: int = 128
    text_layers: int = 2
    heads: int = 4
    base_channels: int = 32
    multipliers: tuple = (1, 2, 3, 4)
    blocks: int = 2
    max_length: int = 64
    timesteps: int = 1000
    prediction_type: str = 'v'
    schedule: str = 'cosine'
    # Missing in legacy checkpoints: keep their architecture exactly loadable.
    spatial_self_attention: bool = False

    def dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, dim, context_dim, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(context_dim, dim)
        self.v = nn.Linear(context_dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context, mask=None):
        b, n, d = x.shape
        def split(z):
            return z.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(context)), split(self.v(context))
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(.1), nn.Linear(dim * 4, dim))

    def forward(self, x, mask):
        normalized = self.norm1(x)
        x = x + self.attn(normalized, normalized, mask)
        return x + self.ff(self.norm2(x))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        d = config.text_dim
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, config.max_length, d) * .02)
        self.layers = nn.ModuleList([TransformerBlock(d, config.heads) for _ in range(config.text_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, ids):
        mask = ids != 0
        x = self.embedding(ids) + self.position[:, :ids.shape[1]]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return x, pooled, mask


def norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.norm1, self.norm2 = norm(in_channels), norm(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(cond_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(condition)[:, :, None, None].chunk(2, 1)
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale) + shift))
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels, text_dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = Attention(channels, text_dim, heads)

    def forward(self, x, text, mask):
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = seq + self.attn(self.norm(seq), text, mask)
        return seq.transpose(1, 2).reshape(b, c, h, w)


class ImageSelfAttention(nn.Module):
    """Let low-resolution image locations exchange information directly."""
    def __init__(self, channels, heads):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = Attention(channels, channels, heads)
        nn.init.zeros_(self.attn.out.weight)
        nn.init.zeros_(self.attn.out.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        normalized = self.norm(seq)
        seq = seq + self.attn(normalized, normalized)
        return seq.transpose(1, 2).reshape(b, c, h, w)


class ConditionalUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        base = config.base_channels
        cond = base * 4
        self.time = nn.Sequential(nn.Linear(base, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.global_text = nn.Linear(config.text_dim, cond)
        channels = [base * m for m in config.multipliers]
        self.input = nn.Conv2d(3, channels[0], 3, padding=1)
        self.down_blocks, self.down_attn, self.downsample = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, c in enumerate(channels):
            self.down_blocks.append(nn.ModuleList([ResidualBlock(c, c, cond) for _ in range(config.blocks)]))
            self.down_attn.append(SpatialAttention(c, config.text_dim, config.heads) if config.resolution // 2**i <= 16 else nn.Identity())
            if i < len(channels) - 1:
                self.downsample.append(nn.Conv2d(c, channels[i + 1], 3, stride=2, padding=1))
        self.middle = ResidualBlock(channels[-1], channels[-1], cond)
        self.middle_attention = ImageSelfAttention(channels[-1], config.heads) if config.spatial_self_attention else nn.Identity()
        self.up_blocks, self.up_attn, self.upsample = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(channels))):
            c = channels[i]
            self.up_blocks.append(nn.ModuleList([ResidualBlock(c * 2, c, cond)] + [ResidualBlock(c, c, cond) for _ in range(config.blocks - 1)]))
            self.up_attn.append(SpatialAttention(c, config.text_dim, config.heads) if config.resolution // 2**i <= 16 else nn.Identity())
            if i > 0:
                self.upsample.append(nn.Conv2d(c, channels[i - 1], 3, padding=1))
        self.output = nn.Sequential(norm(base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))

    def forward(self, x, t, text, pooled, mask):
        half = self.config.base_channels // 2
        frequency = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / max(half - 1, 1))
        phase = t.float()[:, None] * frequency[None]
        cond = self.time(torch.cat([phase.sin(), phase.cos()], -1)) + self.global_text(pooled)
        x, skips = self.input(x), []
        for i, (blocks, attention) in enumerate(zip(self.down_blocks, self.down_attn)):
            for block in blocks:
                x = block(x, cond)
            if isinstance(attention, SpatialAttention):
                x = attention(x, text, mask)
            skips.append(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)
        x = self.middle_attention(self.middle(x, cond))
        for i, (blocks, attention) in enumerate(zip(self.up_blocks, self.up_attn)):
            x = torch.cat([x, skips.pop()], 1)
            for block in blocks:
                x = block(x, cond)
            if isinstance(attention, SpatialAttention):
                x = attention(x, text, mask)
            if i < len(self.upsample):
                x = self.upsample[i](F.interpolate(x, scale_factor=2, mode='nearest'))
        return self.output(x)


class NatureModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.text = TextEncoder(vocab_size, config)
        self.unet = ConditionalUNet(config)

    def forward(self, x, t, ids):
        text, pooled, mask = self.text(ids)
        return self.unet(x, t, text, pooled, mask)
