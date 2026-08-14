"""Class-conditional UNet: the same noise predictor, told which class to draw.

Conditioning is a plain nn.Embedding lookup summed into the time embedding.
Every ResidualBlock already adds that embedding as a per-channel bias, so the
class signal reaches the whole trunk without touching any block.
"""

import torch
from torch import nn

from unet import UNet, Upsample


class ConditionedUNet(UNet):
    """UNet whose conditioning embedding is time + class.

    The embedding table has one extra row, `null_class`, which stands for "no
    class given". Training drops labels onto it with probability
    `label_dropout` so the same weights can later be used for classifier-free
    guidance; sampling here is plain conditional.
    """

    def __init__(self, num_classes, label_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.null_class = num_classes  # last row of the table
        self.label_dropout = label_dropout

        time_dim = self.time_mlp[-1].out_features
        self.label_emb = nn.Embedding(num_classes + 1, time_dim)

    def null_like(self, y):
        """The unconditional counterpart of `y`, for classifier-free guidance."""
        return torch.full_like(y, self.null_class)

    def maybe_drop(self, y):
        """Replace some labels with the null class, but only while training."""
        if not self.training or self.label_dropout <= 0:
            return y
        drop = torch.rand(y.shape, device=y.device) < self.label_dropout
        return torch.where(drop, torch.full_like(y, self.null_class), y)

    def forward(self, x, t, y):
        emb = self.time_mlp(t) + self.label_emb(self.maybe_drop(y))

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
    num_classes = 11

    model = ConditionedUNet(num_classes)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, num_classes, (2,))
    out = model(x, t, y)

    print(f"out:    {tuple(out.shape)}")
    assert out.shape == x.shape

    # The null class is a valid index too — that is what guidance will pass.
    model.eval()
    null = torch.full((2,), model.null_class)
    assert model(x, t, null).shape == x.shape
