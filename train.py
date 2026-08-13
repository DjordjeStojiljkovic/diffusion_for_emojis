"""Headless version of train.ipynb: train the UNet to predict diffusion noise."""

import argparse

import torch

from dataloader import EmojiDataset
from diffusion import Diffusion
from unet import UNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ckpt.pt")
    p.add_argument(
        "--all", action="store_true", help="train on the whole dataset (default: one image)"
    )
    p.add_argument("--image-index", type=int, default=0, help="which image to overfit")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def batch_source(args):
    """Returns a get_batch() callable yielding (B, 3, 64, 64) tensors on device."""
    dataset = EmojiDataset()

    if args.all:
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print(f"training on all {len(dataset)} images")

        def get_batch(_iter=[iter(loader)]):
            try:
                batch = next(_iter[0])
            except StopIteration:
                _iter[0] = iter(loader)
                batch = next(_iter[0])
            return batch.to(args.device)

    else:
        x0 = dataset[args.image_index].unsqueeze(0).to(args.device)
        print(f"overfitting a single image: {dataset.paths[args.image_index].name}")

        def get_batch():
            return x0.repeat(args.batch_size, 1, 1, 1)

    return get_batch


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    get_batch = batch_source(args)
    diffusion = Diffusion(timesteps=args.timesteps, device=args.device)

    model = UNet().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M on {args.device}")

    window = []
    for step in range(1, args.steps + 1):
        loss = diffusion.loss(model, get_batch())

        opt.zero_grad()
        loss.backward()
        opt.step()

        window.append(loss.item())
        if step % args.log_every == 0:
            print(f"step {step:5d} | loss {sum(window) / len(window):.4f}")
            window.clear()

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
