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
    labels=None,
    shape=(3, 64, 64),
    guidance_scale=1.0,
    ddim_steps=100,
    batch_size=64,
    device=None,
    n=None,
):
    """Generate one image per entry of `labels`, in batches.

    Pass `labels=None` with `n=` for an unconditional model, which has no class
    to be told about.

    `ddim_steps=None` falls back to full ancestral sampling — better quality,
    ~10x slower. DDIM is the default because the metric batches need hundreds
    of images.
    """
    device = device or diffusion.device
    if labels is None:
        if n is None:
            raise ValueError("pass `n` when generating without labels")
    else:
        labels = labels.to(device)
        n = len(labels)

    was_training = model.training
    model.eval()

    out = []
    for i in range(0, n, batch_size):
        count = min(batch_size, n - i)
        chunk = None if labels is None else labels[i : i + count]
        if ddim_steps:
            images = diffusion.ddim_sample(
                model, n=count, shape=shape, steps=ddim_steps,
                y=chunk, guidance_scale=guidance_scale,
            )
        else:
            images = diffusion.sample(
                model, n=count, shape=shape, y=chunk, guidance_scale=guidance_scale,
            )
        out.append(images.cpu())

    model.train(was_training)
    return torch.cat(out)


def unpack(batch):
    """(images, labels) from a loader batch, with labels None when unlabelled.

    EmojiDataset yields bare tensors and ConditionedEmojiDataset yields pairs,
    so the training loop has to accept either.
    """
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    return batch, None


def fixed_val_batch(dataset, max_images=512, seed=1234, timesteps=1000, unconditional=False):
    """Materialise the validation set with one *fixed* (t, noise) draw per image.

    The diffusion loss is an expectation over timesteps and noise. Re-drawing
    them at every evaluation swamps the real signal: consecutive validation
    losses would differ mostly by which timesteps happened to come up. Freezing
    them turns the val curve into something you can actually read a trend off,
    at the cost of it being an estimate over one fixed draw rather than the full
    expectation — fine, since we only ever compare it against itself.

    Capped at `max_images` because this runs every `val_every` steps.
    """
    generator = torch.Generator().manual_seed(seed)
    count = min(max_images, len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:count]

    rows = [unpack(dataset[int(i)]) for i in indices]
    images = torch.stack([image for image, _ in rows])
    labels = None
    if not unconditional and rows[0][1] is not None:
        labels = torch.tensor([label for _, label in rows], dtype=torch.long)

    t = torch.randint(0, timesteps, (count,), generator=generator)
    noise = torch.randn(images.shape, generator=generator)
    return images, labels, t, noise


@torch.no_grad()
def evaluate_loss(model, diffusion, batch, batch_size=64, device="cpu"):
    """Mean diffusion loss over a `fixed_val_batch`, in eval mode."""
    images, labels, t, noise = batch
    was_training = model.training
    model.eval()

    total = 0.0
    for i in range(0, len(images), batch_size):
        chunk = slice(i, i + batch_size)
        loss = diffusion.loss(
            model,
            images[chunk].to(device),
            None if labels is None else labels[chunk].to(device),
            t=t[chunk].to(device),
            noise=noise[chunk].to(device),
        )
        total += float(loss) * len(images[chunk])

    model.train(was_training)
    return total / len(images)


def class_labels(num_classes, per_class, device="cpu"):
    """[0]*per_class + [1]*per_class + ... — the label vector for a class grid."""
    return torch.arange(num_classes, device=device).repeat_interleave(per_class)


def train(
    model,
    dataset,
    run=None,
    val_dataset=None,
    steps=20_000,
    batch_size=32,
    lr=2e-4,
    timesteps=1000,
    ema_decay=0.999,
    log_every=250,
    val_every=None,
    val_max_images=512,
    val_seed=1234,
    checkpoint_every=2_000,
    preview_every=2_000,
    preview_per_class=4,
    preview_n=8,
    preview_labels=None,
    preview_ddim_steps=100,
    guidance_scale=1.0,
    num_workers=2,
    device="cpu",
    seed=0,
    on_preview=None,
    ema=None,
    start_step=0,
    optimizer_state=None,
    losses=None,
    val_losses=None,
    unconditional=False,
):
    """Train `model` on `dataset` and return (model, ema, losses, diffusion).

    The EMA copy is what gets sampled from and evaluated; the raw weights are
    kept too so the two can be compared.

    Pass `val_dataset` (see dataloader.train_val_split) to log a validation loss
    every `val_every` steps alongside the training loss. It is measured on the
    *raw* weights, not the EMA copy, so it is directly comparable to the
    training loss it is plotted against — a gap opening between the two is the
    overfitting signal.

    `unconditional=True` trains the plain UNet, ignoring any labels the dataset
    yields. That lets an unconditional run reuse a labelled dataset purely for
    its train/val split machinery.

    Losing a multi-hour run to a dead kernel is the expensive failure here, so
    weights are written to runs/<name>/ckpt.pt every `checkpoint_every` steps as
    well as at the end, and interrupting with Ctrl-C exits cleanly and keeps
    everything logged so far.

    `steps` is always how many steps *this call* runs. To continue an earlier
    run, pass its `ema`, its `start_step`, its `optimizer_state` and its
    `losses`, and the logs, loss history and step numbering carry on rather than
    restarting at zero.
    """
    # Offset by start_step so a resumed run does not replay the same batches.
    torch.manual_seed(seed + start_step)

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
    ema = (make_ema(model) if ema is None else ema).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state is not None:
        opt.load_state_dict(optimizer_state)

    diffusion = Diffusion(timesteps=timesteps, device=device)

    losses = list(losses) if losses is not None else []
    val_losses = dict(val_losses) if val_losses is not None else {}
    val_every = val_every or log_every
    val_batch = None
    if val_dataset is not None:
        val_batch = fixed_val_batch(
            val_dataset, val_max_images, val_seed, timesteps, unconditional
        )
        print(f"validation: {len(val_batch[0])} of {len(val_dataset)} images, "
              f"fixed noise, every {val_every} steps")

    window = []
    pending = []
    step = start_step
    target = start_step + steps
    started = time.time()

    try:
        while step < target:
            for batch in loader:
                images, labels = unpack(batch)
                images = images.to(device, non_blocking=True)
                if labels is not None:
                    labels = labels.to(device, non_blocking=True)
                if unconditional:
                    labels = None

                loss = diffusion.loss(model, images, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                ema_update(ema, model, ema_decay)

                step += 1
                losses.append(loss.item())
                window.append(loss.item())

                val_loss = None
                if val_batch is not None and step % val_every == 0:
                    val_loss = evaluate_loss(model, diffusion, val_batch, device=device)
                    val_losses[step] = val_loss
                pending.append((step, loss.item(), val_loss))

                if step % log_every == 0:
                    val_note = f" | val {val_losses[step]:.4f}" if step in val_losses else ""
                    print(
                        f"step {step:6d} | epoch {step / steps_per_epoch:7.1f}"
                        f" | loss {sum(window) / len(window):.4f}{val_note}"
                        f" | {(time.time() - started) / 60:5.1f} min"
                    )
                    window.clear()

                    # Flush to disk as we go: the loss history used to be
                    # written only at the end, so a dead kernel lost hours of it
                    # even though the checkpoints had survived.
                    if run is not None and pending:
                        run.append_losses(pending)
                        pending.clear()

                if run is not None and checkpoint_every and step % checkpoint_every == 0:
                    run.save_checkpoint(model, ema=ema, step=step)

                if preview_every and step % preview_every == 0:
                    # With hundreds of classes (one per emoji name) a preview of
                    # every class is thousands of images, so callers pass an
                    # explicit handful instead.
                    if unconditional:
                        labels_preview = None
                    elif preview_labels is None:
                        labels_preview = class_labels(
                            model.num_classes, preview_per_class, device
                        )
                    else:
                        labels_preview = preview_labels.to(device)

                    images_preview = generate(
                        ema, diffusion, labels_preview,
                        guidance_scale=guidance_scale,
                        ddim_steps=preview_ddim_steps, device=device,
                        n=preview_n,
                    )
                    if run is not None:
                        run.save_grid(
                            images_preview, f"preview_{step:06d}.png",
                            ncols=preview_per_class,
                        )
                    if on_preview is not None:
                        on_preview(
                            step, images_preview,
                            None if labels_preview is None else labels_preview.cpu(),
                        )

                if step >= target:
                    break
    except KeyboardInterrupt:
        print(f"interrupted at step {step}")

    minutes = (time.time() - started) / 60
    ran = step - start_step
    print(f"done: {ran} steps in {minutes:.1f} min (now at {step}), final loss {losses[-1]:.4f}")

    if run is not None:
        # Rewrites the file, superseding the incremental appends above.
        run.save_losses(losses, val_losses)
        # The rolling checkpoint carries the optimiser so training can resume.
        print(f"saved weights -> {run.save_checkpoint(model, ema=ema, step=step, opt=opt)}")
        run.save_config(
            steps_completed=step,
            train_minutes=round(run.config.get("train_minutes", 0.0) + minutes, 2),
            last_round_minutes=round(minutes, 2),
            final_loss=losses[-1],
            steps_per_epoch=steps_per_epoch,
            **({"final_val_loss": val_losses[max(val_losses)],
                "val_every": val_every,
                "n_val_images": len(val_batch[0])} if val_losses else {}),
        )

    # val_losses stays out of the return tuple so existing 4-way unpacking in
    # the notebooks keeps working; read it back with run.load_loss_history().
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

        # Resume: two more steps continue the numbering and the loss history.
        checkpoint = run.load_checkpoint()
        assert "opt" in checkpoint, "the rolling checkpoint must carry the optimiser"

        print()
        model, ema, losses, _ = train(
            model, dataset, run,
            steps=2, batch_size=4, timesteps=20, log_every=1,
            checkpoint_every=0, preview_every=0, num_workers=0,
            ema=ema, start_step=checkpoint["step"],
            optimizer_state=checkpoint["opt"], losses=run.load_losses(),
        )
        assert len(losses) == 5, f"history should extend to 5, got {len(losses)}"
        assert run.read_json("config.json")["steps_completed"] == 5
        assert run.load_checkpoint()["step"] == 5
        # The merged config must not have dropped what the first round wrote.
        assert run.read_json("config.json")["dataset"] == OPENMOJI_10

        print("\nfiles:", sorted(p.name for p in run.dir.rglob("*")))
        print("ok")
