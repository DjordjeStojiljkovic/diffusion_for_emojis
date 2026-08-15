"""Linear-beta DDPM: forward noising, training loss, and ancestral sampling.

This is the reusable twin of the inline cells in train.ipynb — keep them in sync.

Works with either model: pass `y=None` for the unconditional UNet, or class
indices for ConditionedUNet.
"""

import torch
import torch.nn.functional as F


def _eps(model, x, t, y, guidance_scale=1.0):
    """Predict the noise, optionally with classifier-free guidance.

    Guidance extrapolates away from the unconditional prediction:
    eps = eps_uncond + s * (eps_cond - eps_uncond). s=1 is plain conditional,
    s>1 trades diversity for a stronger class signal. Both predictions go
    through the model in one batched call.
    """
    if y is None:
        return model(x, t)
    if guidance_scale == 1.0:
        return model(x, t, y)

    # Each conditional model defines its own "unconditional" — a null class
    # index, or the embedding of the empty prompt for the text-conditioned one.
    null = (
        model.null_like(y)
        if hasattr(model, "null_like")
        else torch.full_like(y, model.null_class)
    )
    both = model(
        torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null])
    )
    eps_cond, eps_uncond = both.chunk(2)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


def _as_labels(y, n, device):
    """Accept an int class ("all of this class") or a per-image tensor."""
    if y is None:
        return None
    if isinstance(y, int):
        return torch.full((n,), y, device=device, dtype=torch.long)
    return y.to(device)


class Diffusion:
    """Holds the beta schedule and the forward/reverse processes."""

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        """Forward process: x_t = sqrt(acp_t) * x0 + sqrt(1 - acp_t) * noise."""
        acp = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return acp.sqrt() * x0 + (1 - acp).sqrt() * noise

    def loss(self, model, x0, y=None, t=None, noise=None):
        """MSE between the noise we added and the noise the model predicts.

        `t` and `noise` are drawn fresh per call during training. Pass them
        explicitly to make the loss deterministic: the validation loss has to
        hold them fixed across evaluations, or the curve measures which
        timesteps were sampled rather than how the model improved.
        """
        if t is None:
            t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        return F.mse_loss(_eps(model, self.q_sample(x0, t, noise), t, y), noise)

    @torch.no_grad()
    def sample(
        self, model, n=1, shape=(3, 64, 64), start_t=None, x0=None, y=None,
        guidance_scale=1.0,
    ):
        """Reverse diffusion from `start_t` down to 0.

        With `x0=None` it starts from pure noise. Pass `x0` to instead noise a
        real image up to `start_t` and denoise from there (`n` is then ignored).

        `y` is the class to draw: an int for "all of this class", or a tensor of
        one index per image.
        """
        start_t = start_t or self.timesteps

        if x0 is None:
            x = torch.randn(n, *shape, device=self.device)
        else:
            t = torch.full((x0.shape[0],), start_t - 1, device=self.device)
            x = self.q_sample(x0, t, torch.randn_like(x0))

        y = _as_labels(y, x.shape[0], self.device)

        for i in reversed(range(start_t)):
            t = torch.full((x.shape[0],), i, device=self.device)
            eps = _eps(model, x, t, y, guidance_scale)

            mean = (
                x - self.betas[i] / (1 - self.alphas_cumprod[i]).sqrt() * eps
            ) / self.alphas[i].sqrt()

            if i > 0:
                # sigma_t^2 = beta_t, the "fixedlarge" choice from the DDPM paper.
                x = mean + self.betas[i].sqrt() * torch.randn_like(x)
            else:
                x = mean

        return x

    @torch.no_grad()
    def ddim_sample(
        self, model, n=1, shape=(3, 64, 64), steps=100, eta=0.0, y=None,
        guidance_scale=1.0,
    ):
        """Deterministic (eta=0) DDIM sampling on a strided subsequence.

        Same trained model, ~10x fewer network calls than `sample`. Used for
        previews during training and for the metric batches, where hundreds of
        images at 1000 ancestral steps each would dominate the runtime.
        """
        # Descending subsequence of timesteps, e.g. 999, 989, ..., 0.
        ts = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()

        x = torch.randn(n, *shape, device=self.device)
        y = _as_labels(y, n, self.device)

        for i, t_cur in enumerate(ts):
            t = torch.full((n,), t_cur, device=self.device)
            eps = _eps(model, x, t, y, guidance_scale)

            acp = self.alphas_cumprod[t_cur]
            # alphas_cumprod at the *next* (earlier) step; 1.0 past the end.
            acp_prev = (
                self.alphas_cumprod[ts[i + 1]]
                if i + 1 < len(ts)
                else torch.ones((), device=self.device)
            )

            # The clean image implied by the current noise prediction.
            x0_hat = ((x - (1 - acp).sqrt() * eps) / acp.sqrt()).clamp(-1, 1)

            # eta=0 makes this deterministic; eta=1 recovers DDPM-like noise.
            sigma = (
                eta
                * ((1 - acp_prev) / (1 - acp)).sqrt()
                * (1 - acp / acp_prev).clamp(min=0).sqrt()
            )
            direction = (1 - acp_prev - sigma**2).clamp(min=0).sqrt() * eps

            x = acp_prev.sqrt() * x0_hat + direction
            if eta > 0 and i + 1 < len(ts):
                x = x + sigma * torch.randn_like(x)

        return x
