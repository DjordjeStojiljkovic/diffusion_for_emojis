"""Run bookkeeping: one directory per training run, plus the figures for it.

Everything a run produces lands under runs/<name>/ so results survive the
notebook kernel and can be pulled straight into a presentation:

    runs/<name>/config.json      hyperparameters and dataset description
    runs/<name>/losses.csv       per-step training loss + sparse val_loss
    runs/<name>/loss.png         loss curve
    runs/<name>/metrics.json     everything metrics.evaluate returned
    runs/<name>/ckpt.pt          model + EMA weights
    runs/<name>/samples/*.png    preview grids and the final per-class figure

Grids are built with PIL rather than torchvision.utils, so nothing here needs a
dependency beyond torch, PIL and matplotlib.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

RUNS_DIR = "runs"

RAW_COLOR = "#c9ccd1"
LINE_COLOR = "#3b6fd4"
VAL_COLOR = "#d4703b"
TEXT_COLOR = "#555"


def to_uint8(images):
    """(B, 3, H, W) in [-1, 1] -> (B, H, W, 3) uint8."""
    x = (images.detach().cpu().clamp(-1, 1) + 1) / 2
    return (x * 255).round().byte().permute(0, 2, 3, 1).numpy()


def make_grid(images, ncols=8, pad=2, background=255):
    """Tile a batch into a single PIL image."""
    tiles = to_uint8(images)
    n, h, w, _ = tiles.shape
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    canvas = np.full(
        (nrows * h + (nrows + 1) * pad, ncols * w + (ncols + 1) * pad, 3),
        background,
        dtype=np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, ncols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = tile
    return Image.fromarray(canvas)


def show_grid(images, ncols=8, title=None, path=None, scale=1.4):
    """Draw a batch as a grid; optionally save it."""
    grid = make_grid(images, ncols)
    if path is not None:
        grid.save(path)

    ncols = min(ncols, len(images))
    nrows = (len(images) + ncols - 1) // ncols
    fig, ax = plt.subplots(figsize=(scale * ncols, scale * nrows + 0.3))
    ax.imshow(np.asarray(grid))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, color=TEXT_COLOR)
    plt.tight_layout()
    plt.show()
    return grid


def show_class_grid(images, labels, class_names, title=None, path=None, scale=1.0):
    """One row per class, labelled — the figure that shows conditioning working.

    Samples are grouped by their *requested* class, so a row that is visually
    incoherent is a conditioning failure, not a rendering artifact.
    """
    labels = labels.cpu()
    present = sorted(set(labels.tolist()))
    per_row = max(int((labels == c).sum()) for c in present)

    fig, axes = plt.subplots(
        len(present), per_row,
        figsize=(scale * per_row, scale * len(present) + 0.4),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")

    for row, c in enumerate(present):
        tiles = to_uint8(images[labels == c])
        for col, tile in enumerate(tiles):
            axes[row][col].imshow(tile)
        # axis("off") hides a normal ylabel, so write the class in the margin.
        axes[row][0].text(
            -0.15, 0.5, class_names[c], transform=axes[row][0].transAxes,
            ha="right", va="center", fontsize=8, color=TEXT_COLOR,
        )

    if title:
        fig.suptitle(title, fontsize=11, color=TEXT_COLOR)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def smooth(values, k=25):
    """Running mean over the last k values."""
    out, window = [], []
    for v in values:
        window.append(v)
        if len(window) > k:
            window.pop(0)
        out.append(sum(window) / len(window))
    return out


def plot_losses(losses, k=100, title=None, path=None, val_losses=None):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(losses, color=RAW_COLOR, linewidth=0.8, label="train (raw)")
    ax.plot(smooth(losses, k), color=LINE_COLOR, linewidth=1.6, label="train")
    if val_losses:
        steps = sorted(val_losses)
        # Validation is evaluated on fixed (t, noise), so it is already smooth
        # and is plotted unsmoothed — a gap opening against the train curve is
        # the overfitting signal.
        ax.plot(steps, [val_losses[s] for s in steps],
                color=VAL_COLOR, linewidth=1.6, label="validation")
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR)
    ax.set_yscale("log")
    ax.set_xlabel("step", color=TEXT_COLOR)
    ax.set_ylabel("loss", color=TEXT_COLOR)
    if title:
        ax.set_title(title, fontsize=10, color=TEXT_COLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


class Run:
    """A directory of results for one training run."""

    def __init__(self, name, root=RUNS_DIR, config=None):
        self.name = name
        self.dir = Path(root) / name
        self.samples_dir = self.dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        # Merge rather than replace: re-running the cell that constructs a Run
        # must not erase what training recorded afterwards (steps_completed,
        # wall-clock, final loss), and a resumed run needs its history intact.
        existing = {}
        if self.path("config.json").exists():
            existing = self.read_json("config.json")

        self.config = {**existing, **(config or {})}
        if self.config:
            self.save_config()

    def __repr__(self):
        return f"Run({self.name!r} -> {self.dir})"

    def path(self, *parts):
        return self.dir.joinpath(*parts)

    def save_config(self, **updates):
        self.config.update(updates)
        self.write_json("config.json", self.config)
        return self.config

    def write_json(self, filename, payload):
        with open(self.path(filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return self.path(filename)

    def read_json(self, filename):
        with open(self.path(filename), encoding="utf-8") as f:
            return json.load(f)

    LOSS_COLUMNS = ["step", "loss", "val_loss"]

    def save_losses(self, losses, val_losses=None):
        """Rewrite the whole history. `val_losses` is {step: loss}, sparse.

        Called at the end of training, where it supersedes whatever the
        incremental appends left behind.
        """
        val_losses = val_losses or {}
        with open(self.path("losses.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.LOSS_COLUMNS)
            for step, loss in enumerate(losses, start=1):
                writer.writerow([step, loss, val_losses.get(step, "")])
        return self.path("losses.csv")

    def append_losses(self, rows):
        """Append (step, loss, val_loss_or_None) rows mid-training.

        Insurance against losing hours of history to a dead kernel: the final
        save_losses rewrites the file anyway, so a partial append is never the
        authoritative copy, only the surviving one.
        """
        path = self.path("losses.csv")
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(self.LOSS_COLUMNS)
            for step, loss, val in rows:
                writer.writerow([step, loss, "" if val is None else val])
        return path

    def load_losses(self):
        """The training-loss history so far, so a resumed run extends it
        instead of overwriting it with only the new steps."""
        return self.load_loss_history()[0]

    def load_loss_history(self):
        """(losses, {step: val_loss}). Reads the pre-val_loss 2-column format too."""
        path = self.path("losses.csv")
        if not path.exists():
            return [], {}

        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]

        losses, val_losses = [], {}
        for row in rows:
            losses.append(float(row[1]))
            if len(row) > 2 and row[2] != "":
                val_losses[int(row[0])] = float(row[2])
        return losses, val_losses

    def save_loss_curve(self, losses, k=100, val_losses=None):
        plot_losses(losses, k, title=f"{self.name} — training loss",
                    path=self.path("loss.png"), val_losses=val_losses)
        return self.path("loss.png")

    def save_grid(self, images, filename, ncols=8):
        path = self.samples_dir / filename
        make_grid(images, ncols).save(path)
        return path

    def save_checkpoint(self, model, ema=None, step=None, filename="ckpt.pt", opt=None, **extra):
        """Write model + EMA weights, replacing the previous checkpoint.

        Saved through a temporary file: mid-training checkpoints overwrite the
        only copy, and a crash during a 120 MB write would otherwise leave a
        truncated file where the last good one used to be.

        Pass `opt` to include the optimiser state, which roughly doubles the
        file but lets training resume without Adam's moments restarting from
        zero. Archival snapshots normally omit it.
        """
        payload = {"model": model.state_dict(), "step": step, **extra}
        if ema is not None:
            payload["ema"] = ema.state_dict()
        if opt is not None:
            payload["opt"] = opt.state_dict()

        path = self.path(filename)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def load_checkpoint(self, filename="ckpt.pt", map_location="cpu"):
        return torch.load(self.path(filename), map_location=map_location, weights_only=False)

    def save_metrics(self, metrics):
        return self.write_json("metrics.json", metrics)


def load_metrics(names, root=RUNS_DIR):
    """Collect metrics.json from several runs into {run_name: metrics}."""
    out = {}
    for name in names:
        path = Path(root) / name / "metrics.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                out[name] = json.load(f)
    return out


def comparison_table(metrics_by_run, keys=None, digits=3):
    """Render {run: metrics} as a plain-text table for the writeup."""
    keys = keys or [
        "fid", "kid_mean", "cmmd", "precision", "recall",
        "clip_score", "clip_accuracy", "clip_accuracy_real",
        "nn_similarity_mean", "lpips_nn_mean", "perceptual_copy_rate",
        "pixel_nn_rmse_mean", "diversity_fake", "diversity_real",
    ]
    runs = list(metrics_by_run)
    width = max([len(k) for k in keys] + [6])

    lines = ["metric".ljust(width) + "".join(f"{r:>16}" for r in runs)]
    lines.append("-" * len(lines[0]))
    for key in keys:
        row = key.ljust(width)
        for run in runs:
            value = metrics_by_run[run].get(key)
            row += f"{value:>16.{digits}f}" if isinstance(value, (int, float)) else f"{'—':>16}"
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        run = Run("smoke", root=tmp, config={"steps": 10, "note": "self-test"})
        images = torch.randn(8, 3, 64, 64).clamp(-1, 1)

        run.save_losses([1.0, 0.5, 0.25], {2: 0.6})
        losses, val_losses = run.load_loss_history()
        assert losses == [1.0, 0.5, 0.25], losses
        assert val_losses == {2: 0.6}, val_losses

        # Incremental appends must round-trip through the same reader.
        run2 = Run("smoke_append", root=tmp)
        run2.append_losses([(1, 1.0, None), (2, 0.5, 0.6)])
        run2.append_losses([(3, 0.25, None)])
        assert run2.load_loss_history() == ([1.0, 0.5, 0.25], {2: 0.6})

        # The pre-val_loss two-column format still reads.
        with open(run2.path("losses.csv"), "w", encoding="utf-8") as f:
            f.write("step,loss\n1,1.0\n2,0.5\n")
        assert run2.load_loss_history() == ([1.0, 0.5], {})

        run.save_grid(images, "grid.png", ncols=4)
        run.save_metrics({"fid": 12.5, "clip_accuracy": 0.75})

        model = torch.nn.Linear(2, 2)
        run.save_checkpoint(model, step=10)

        assert run.read_json("config.json")["steps"] == 10
        assert run.load_checkpoint()["step"] == 10
        assert (run.samples_dir / "grid.png").exists()
        assert make_grid(images, ncols=4).size == (4 * 64 + 5 * 2, 2 * 64 + 3 * 2)

        table = comparison_table({"a": {"fid": 1.0}, "b": {"fid": 2.0}})
        print(table)
        print("\nrun dir:", sorted(p.name for p in run.dir.rglob("*")))
        print("ok")
