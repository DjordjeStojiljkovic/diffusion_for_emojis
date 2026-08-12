"""Simple UNet noise predictor for 64x64 diffusion."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, add time embedding, then the same again."""

    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Single-head self-attention over the H*W positions."""

    def __init__(self, ch, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w)
        q, k, v = (z.transpose(1, 2) for z in qkv.unbind(1))  # each (B, HW, C)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class ResAttnBlock(nn.Module):
    """A residual block, optionally followed by attention."""

    def __init__(self, in_ch, out_ch, time_dim, use_attn, groups=8):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_dim, groups)
        self.attn = AttentionBlock(out_ch, groups) if use_attn else nn.Identity()

    def forward(self, x, t_emb):
        return self.attn(self.res(x, t_emb))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x, t_emb=None):
        return self.op(x)


class UNet(nn.Module):
    """Predicts the noise in a 64x64 image given the timestep."""

    def __init__(
        self,
        in_ch=3,
        base=64,
        ch_mults=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resolution=64,
        time_dim=None,
        groups=8,
    ):
        super().__init__()
        time_dim = time_dim or base * 4

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base),
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        # Down path. Every tensor pushed here is concatenated back in the up path.
        ch, res = base, resolution
        skip_chs = [ch]
        self.down = nn.ModuleList()
        for i, mult in enumerate(ch_mults):
            out_ch = base * mult
            for _ in range(num_res_blocks):
                self.down.append(
                    ResAttnBlock(ch, out_ch, time_dim, res in attn_resolutions, groups)
                )
                ch = out_ch
                skip_chs.append(ch)
            if i != len(ch_mults) - 1:
                self.down.append(Downsample(ch))
                skip_chs.append(ch)
                res //= 2

        self.mid_res1 = ResidualBlock(ch, ch, time_dim, groups)
        self.mid_attn = AttentionBlock(ch, groups)
        self.mid_res2 = ResidualBlock(ch, ch, time_dim, groups)

        # Up path: mirrors the down path, one extra block per level for the skips.
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(
                    ResAttnBlock(
                        ch + skip_chs.pop(),
                        out_ch,
                        time_dim,
                        res in attn_resolutions,
                        groups,
                    )
                )
                ch = out_ch
            if i != 0:
                self.up.append(Upsample(ch))
                res *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        h = self.stem(x)
        skips = [h]
        for block in self.down:
            h = block(h, t_emb)
            skips.append(h)

        h = self.mid_res2(self.mid_attn(self.mid_res1(h, t_emb)), t_emb)

        for block in self.up:
            if isinstance(block, Upsample):
                h = block(h)
            else:
                h = block(torch.cat([h, skips.pop()], dim=1), t_emb)

        return self.out(h)


if __name__ == "__main__":
    model = UNet()
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)

    print(f"out:    {tuple(out.shape)}")
    assert out.shape == x.shape
