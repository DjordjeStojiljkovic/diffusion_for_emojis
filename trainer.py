"""Training loop and sample generation for the class-conditional model.

Kept out of the notebook so the 10-image and 200-image sections are two calls
with different configs rather than two copies of the same 60 lines.
"""

import time

import torch
from torch.utils.data import DataLoader

from diffusion import Diffusion


@torch.no_grad()
def ema_update(ema, model, decay):
    """ema <- decay * ema + (1 - decay) * model."""
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.lerp_(p, 1 - decay)
    for b_ema, b in zip(ema.buffers(), model.buffers()):
        b_ema.copy_(b)


def make_ema(model):
    """A frozen copy of `model` to hold the weight average we sample from."""
    import copy

    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    ema.eval()
    return ema


@torch.no_grad()
def generate(
    model,
    diffusion,
    labels,
    shape=(3, 64, 64),
    guidance_scale=1.0,
    ddim_steps=100,
    batch_size=64,
    device=None,
):
    """Generate one image per entry of `labels`, in batches.

    `ddim_steps=None` falls back to full ancestral sampling — better quality,
    ~10x slower. DDIM is the default because the metric batches need hundreds
    of images.
    """
    device = device or diffusion.device
    labels = labels.to(device)

    was_training = model.training
    model.eval()

    out = []
    for i in range(0, len(labels), batch_size):
        chunk = labels[i : i + batch_size]
        if ddim_steps:
            images = diffusion.ddim_sample(
                model, n=len(chunk), shape=shape, steps=ddim_steps,
                y=chunk, guidance_scale=guidance_scale,
            )
        else:
            images = diffusion.sample(
                model, n=len(chunk), shape=shape, y=chunk, guidance_scale=guidance_scale,
            )
        out.append(images.cpu())

    model.train(was_training)
    return torch.cat(out)


def class_labels(num_classes, per_class, device="cpu"):
    """[0]*per_class + [1]*per_class + ... — the label vector for a class grid."""
    return torch.arange(num_classes, device=device).repeat_interleave(per_class)


def train(
    model,
    dataset,
    run=None,
    steps=20_000,
    batch_size=32,
    lr=2e-4,
    timesteps=1000,
    ema_decay=0.999,
    log_every=250,
    checkpoint_every=2_000,
    preview_every=2_000,
    preview_per_class=4,
    preview_ddim_steps=100,
    guidance_scale=1.0,
    num_workers=2,
    device="cpu",
    seed=0,
    on_preview=None,
):
    """Train `model` on `dataset` and return (model, ema, losses, diffusion).

    The EMA copy is what gets sampled from and evaluated; the raw weights are
    kept too so the two can be compared.

    Losing a multi-hour run to a dead kernel is the expensive failure here, so
    weights are written to runs/<name>/ckpt.pt every `checkpoint_every` steps as
    well as at the end, and interrupting with Ctrl-C exits cleanly and keeps
    everything logged so far.
    """
    torch.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=len(dataset) > batch_size,
        num_workers=num_workers,
        pin_memory=(str(device) == "cuda"),
        # These datasets are small enough that an epoch can be a handful of
        # steps; without this the workers are torn down and respawned that
        # often, which costs more than the loading itself.
        persistent_workers=num_workers > 0,
    )
    steps_per_epoch = max(1, len(loader))

    model = model.to(device)
    ema = make_ema(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    diffusion = Diffusion(timesteps=timesteps, device=device)

    losses, window = [], []
    step = 0
    started = time.time()

    try:
        while step < steps:
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                loss = diffusion.loss(model, images, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                ema_update(ema, model, ema_decay)

                step += 1
                losses.append(loss.item())
                window.append(loss.item())

                if step % log_every == 0:
                    print(
                        f"step {step:6d} | epoch {step / steps_per_epoch:7.1f}"
                        f" | loss {sum(window) / len(window):.4f}"
                        f" | {(time.time() - started) / 60:5.1f} min"
                    )
                    window.clear()

                if run is not None and checkpoint_every and step % checkpoint_every == 0:
                    run.save_checkpoint(model, ema=ema, step=step)

                if preview_every and step % preview_every == 0:
                    labels_preview = class_labels(
                        model.num_classes, preview_per_class, device
                    )
                    images_preview = generate(
                        ema, diffusion, labels_preview,
                        guidance_scale=guidance_scale,
                        ddim_steps=preview_ddim_steps, device=device,
                    )
                    if run is not None:
                        run.save_grid(
                            images_preview, f"preview_{step:06d}.png",
                            ncols=preview_per_class,
                        )
                    if on_preview is not None:
                        on_preview(step, images_preview, labels_preview.cpu())

                if step >= steps:
                    break
    except KeyboardInterrupt:
        print(f"interrupted at step {step}")

    minutes = (time.time() - started) / 60
    print(f"done: {step} steps in {minutes:.1f} min, final loss {losses[-1]:.4f}")

    if run is not None:
        run.save_losses(losses)
        print(f"saved weights -> {run.save_checkpoint(model, ema=ema, step=step)}")
        run.save_config(
            steps_completed=step,
            train_minutes=round(minutes, 2),
            final_loss=losses[-1],
            steps_per_epoch=steps_per_epoch,
        )

    return model, ema, losses, diffusion


if __name__ == "__main__":
    # A tiny end-to-end run: 3 steps on the 10-image subset with a small model.
    import tempfile

    from dataloader import OPENMOJI_10, ConditionedEmojiDataset
    from runlog import Run
    from unet_cond import ConditionedUNet

    dataset = ConditionedEmojiDataset.from_dir(OPENMOJI_10)
    model = ConditionedUNet(
        dataset.num_classes, base=16, ch_mults=(1, 2), attn_resolutions=()
    )

    with tempfile.TemporaryDirectory() as tmp:
        run = Run("smoke", root=tmp, config={"dataset": OPENMOJI_10})
        model, ema, losses, diffusion = train(
            model, dataset, run,
            steps=3, batch_size=4, timesteps=20,
            log_every=1, checkpoint_every=2, preview_every=3, preview_per_class=2,
            preview_ddim_steps=4, num_workers=0,
        )

        images = generate(
            ema, diffusion, class_labels(dataset.num_classes, 2), ddim_steps=4
        )
        assert images.shape == (4, 3, 64, 64)
        assert len(losses) == 3
        assert (run.samples_dir / "preview_000003.png").exists()
        assert run.read_json("config.json")["steps_completed"] == 3
        print("\nfiles:", sorted(p.name for p in run.dir.rglob("*")))
        print("ok")
