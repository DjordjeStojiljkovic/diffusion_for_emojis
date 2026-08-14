"""Text-conditioned UNet: the emoji is chosen by a CLIP text embedding.

The difference from `ConditionedUNet` is what the conditioning vector is made
of. There, a learned `nn.Embedding` row per class — which with one image per
emoji name can only memorise, since the classes share nothing. Here, a frozen
CLIP embedding of the name, projected into the timestep-embedding width:

    emb = time_mlp(t) + text_proj(text_bank[y])

Because CLIP places "grinning face" near "winking face", the model learns
face-ness from every smiley at once instead of treating each name as an
unrelated class. And since the conditioning space is continuous and shared with
CLIP, sampling accepts any prompt at all, not only names seen in training.

CLIP is never trained. `text_proj` (~200k params) is the only new thing that
learns, and it trains jointly with the UNet at no extra cost — the bank is
precomputed, so the text encoder is not even loaded during training.
"""

import torch
from torch import nn

from unet import UNet, Upsample


class TextConditionedUNet(UNet):
    """UNet conditioned on precomputed text embeddings.

    `text_bank` is (N + 1, text_dim): one row per emoji name, plus a trailing
    null row (the embedding of the empty prompt) used for classifier-free
    guidance. Build it with `emoji_text.build_bank`.

    `forward` accepts either form of `y`:
      * int64 indices into the bank — what training uses
      * float (B, text_dim) embeddings — an arbitrary prompt at sampling time
    """

    def __init__(self, text_bank, text_dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        if text_bank.dim() != 2 or len(text_bank) < 2:
            raise ValueError(f"text_bank should be (N + 1, dim), got {tuple(text_bank.shape)}")

        # A buffer, not a parameter: it moves with .to(device) and is saved in
        # the state dict, but no gradient ever reaches it.
        #
        # The clone is load-bearing. `.float()` on an already-float32 tensor
        # returns the *same* tensor, so without it the buffer aliases the bank
        # the caller passed in — and `load_state_dict` copies in place, which
        # would silently overwrite the caller's bank with the checkpoint's and
        # make any bank-mismatch check impossible.
        self.register_buffer("text_bank", text_bank.detach().clone().float())
        self.text_dim = text_bank.shape[1]
        self.num_classes = len(text_bank) - 1
        self.null_class = self.num_classes  # the last row is the empty prompt
        self.text_dropout = text_dropout

        time_dim = self.time_mlp[-1].out_features
        self.text_proj = nn.Sequential(
            # CLIP embeddings are L2-normalized and so quite small in
            # magnitude; the norm lets the projection set its own scale.
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def maybe_drop(self, y):
        """Replace some indices with the null row, but only while training."""
        if not self.training or self.text_dropout <= 0:
            return y
        drop = torch.rand(y.shape, device=y.device) < self.text_dropout
        return torch.where(drop, torch.full_like(y, self.null_class), y)

    def null_like(self, y):
        """The unconditional counterpart of `y`, for classifier-free guidance."""
        if y.dtype.is_floating_point:
            return self.text_bank[self.null_class].expand(y.shape[0], -1)
        return torch.full_like(y, self.null_class)

    def conditioning(self, y):
        """(B,) indices or (B, text_dim) embeddings -> (B, time_dim)."""
        vectors = y if y.dtype.is_floating_point else self.text_bank[self.maybe_drop(y)]
        return self.text_proj(vectors.to(self.text_bank.dtype))

    def forward(self, x, t, y):
        emb = self.time_mlp(t) + self.conditioning(y)

        h = self.stem(x)
        skips = [h]
        for block in self.down:
            h = block(h, emb)
            skips.append(h)

        h = self.mid_res2(self.mid_attn(self.mid_res1(h, emb)), emb)

        for block in self.up:
            if isinstance(block, Upsample):
                h = block(h)
            else:
                h = block(torch.cat([h, skips.pop()], dim=1), emb)

        return self.out(h)


if __name__ == "__main__":
    torch.manual_seed(0)
    num_names, text_dim = 500, 512

    bank = torch.randn(num_names + 1, text_dim)
    model = TextConditionedUNet(bank)

    params = sum(p.numel() for p in model.parameters())
    proj = sum(p.numel() for p in model.text_proj.parameters())
    print(f"params: {params / 1e6:.2f}M  (text_proj: {proj:,} = {100 * proj / params:.2f}%)")
    print(f"names: {model.num_classes}  null index: {model.null_class}")

    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))

    # Indices — the training path.
    out = model(x, t, torch.randint(0, num_names, (2,)))
    assert out.shape == x.shape

    # Raw embeddings — an arbitrary prompt at sampling time.
    out = model(x, t, torch.randn(2, text_dim))
    assert out.shape == x.shape

    # The frozen bank must stay frozen.
    assert not model.text_bank.requires_grad
    assert "text_bank" in dict(model.named_buffers())

    # Loading a checkpoint must not reach back and mutate the caller's bank.
    other_bank = torch.randn(num_names + 1, text_dim)
    reloaded = TextConditionedUNet(other_bank)
    reloaded.load_state_dict(model.state_dict())
    assert torch.equal(reloaded.text_bank, bank), "checkpoint bank should win inside the model"
    assert not torch.equal(other_bank, bank), "the caller's bank must not be overwritten"
    model(x, t, torch.randint(0, num_names, (2,))).sum().backward()
    assert model.text_bank.grad is None, "no gradient may reach the CLIP bank"

    # Guidance needs a null counterpart for both forms of y.
    assert model.null_like(torch.zeros(4, dtype=torch.long)).shape == (4,)
    assert model.null_like(torch.randn(4, text_dim)).shape == (4, text_dim)

    # Dropout on in train mode, off in eval.
    model.train()
    many = torch.zeros(4000, dtype=torch.long)
    fraction = (model.maybe_drop(many) == model.null_class).float().mean()
    print(f"text dropout: {fraction:.3f} (expected ~0.1)")
    assert 0.07 < fraction < 0.13
    model.eval()
    assert (model.maybe_drop(many) == many).all()

    print("ok")
