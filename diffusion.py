"""Linear-beta DDPM: forward noising, training loss, and ancestral sampling.

This is the reusable twin of the inline cells in train.ipynb — keep them in sync.
"""

import torch
import torch.nn.functional as F


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

    def loss(self, model, x0):
        """MSE between the noise we added and the noise the model predicts."""
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, model, n=1, shape=(3, 64, 64), start_t=None, x0=None):
        """Reverse diffusion from `start_t` down to 0.

        With `x0=None` it starts from pure noise. Pass `x0` to instead noise a
        real image up to `start_t` and denoise from there (`n` is then ignored).
        """
        start_t = start_t or self.timesteps

        if x0 is None:
            x = torch.randn(n, *shape, device=self.device)
        else:
            t = torch.full((x0.shape[0],), start_t - 1, device=self.device)
            x = self.q_sample(x0, t, torch.randn_like(x0))

        for i in reversed(range(start_t)):
            t = torch.full((x.shape[0],), i, device=self.device)
            eps = model(x, t)

            mean = (
                x - self.betas[i] / (1 - self.alphas_cumprod[i]).sqrt() * eps
            ) / self.alphas[i].sqrt()

            if i > 0:
                # sigma_t^2 = beta_t, the "fixedlarge" choice from the DDPM paper.
                x = mean + self.betas[i].sqrt() * torch.randn_like(x)
            else:
                x = mean

        return x
