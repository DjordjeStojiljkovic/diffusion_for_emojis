"""Score a trained run's checkpoint, without the notebook that produced it.

The three notebooks each compute metrics inline, which ties evaluation to a live
kernel with the right variables in scope. This does the same job from the
command line, so a checkpoint archived weeks ago can still be scored, and a
series of snapshots can be turned into a metrics-vs-steps curve:

    python evaluate_run.py cond_aug_full
    python evaluate_run.py text_aug_full --checkpoint ckpt_step070000.pt
    python evaluate_run.py cond_aug_full --snapshots     # every ckpt_step*.pt
    python evaluate_run.py --all                         # every run in runs/

Nothing about the architecture is passed in: the checkpoint carries enough to
rebuild the model that wrote it.

    text_bank in the state dict   -> TextConditionedUNet (the bank is in there too)
    label_emb.weight              -> ConditionedUNet, num_classes from its shape
    neither                       -> plain UNet, unconditional

Everything else — which dataset, which label column, how the split was drawn —
comes from runs/<name>/eval.json when the training cell wrote one, and is
inferred from config.json plus the dataset's own CSV when it did not.

Results land in runs/<name>/metrics.json (for the rolling ckpt.pt) or
metrics_step<N>.json (for a snapshot), and every evaluation appends a row to
runs/<name>/metrics_history.csv.
"""

import argparse
import csv
import json
from pathlib import Path

import torch

import metrics as metrics_lib
from dataloader import ConditionedEmojiDataset, train_val_split
from diffusion import Diffusion
from trainer import class_labels, generate
from unet import UNet
from unet_cond import ConditionedUNet
from unet_text import TextConditionedUNet

PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "runs"

# Defaults chosen to match what the notebooks used, so a script-produced
# metrics.json is comparable with one a notebook wrote.
DEFAULTS = {
    "real_sample": 1000,
    "n_fake": 1000,
    "guidance": 1.0,
    "ddim_steps": 100,
    "seed": 0,
    "val_variants": 2,
}

HISTORY_COLUMNS = [
    "step", "checkpoint", "n_real", "n_fake", "fid", "kid_mean", "cmmd",
    "precision", "recall", "diversity_fake", "nn_similarity_mean",
    "lpips_nn_mean", "perceptual_copy_rate", "pixel_copy_rate",
    "clip_accuracy", "name_top1", "name_top5",
]


# --------------------------------------------------------------------------
# rebuilding the model from the checkpoint alone
# --------------------------------------------------------------------------

def build_model(state_dict):
    """(model, kind) for the architecture that produced `state_dict`."""
    if "text_bank" in state_dict:
        # The bank is a buffer, so it travelled with the weights: no need for
        # open_clip, and no risk of rebuilding a bank the model never saw.
        return TextConditionedUNet(state_dict["text_bank"]), "text"
    if "label_emb.weight" in state_dict:
        # One row per class plus the null class used for guidance.
        return ConditionedUNet(state_dict["label_emb.weight"].shape[0] - 1), "class"
    return UNet(), "uncond"


def load_ema(checkpoint_path, device="cpu"):
    """The EMA weights, rebuilt into the right model class.

    EMA rather than the raw weights because that is what every sample and every
    existing metric in this project came from.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("ema") or payload["model"]

    model, kind = build_model(state_dict)
    model.load_state_dict(state_dict)   # strict: a mismatch is a real problem
    model.requires_grad_(False)
    model.eval().to(device)
    return model, kind, payload.get("step")


# --------------------------------------------------------------------------
# working out how the run was set up
# --------------------------------------------------------------------------

def csv_columns(dataset_dir):
    with open(Path(dataset_dir) / "metadata.csv", newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def infer_label_col(dataset_dir, num_classes):
    """Which CSV column has exactly `num_classes` distinct values."""
    with open(Path(dataset_dir) / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for column in ("group", "name", "subgroup"):
        if column in rows[0] and len({r[column] for r in rows}) == num_classes:
            return column
    raise SystemExit(
        f"no column in {dataset_dir}/metadata.csv has {num_classes} distinct values; "
        "write an eval.json manifest with an explicit label_col"
    )


def resolve_setup(run_dir, config, kind, num_classes, overrides):
    """Merge the manifest, the run config, inference and CLI overrides."""
    manifest_path = run_dir / "eval.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    setup = {**DEFAULTS, **manifest}
    setup["dataset"] = manifest.get("dataset") or config["dataset"]
    setup["manifest"] = manifest_path.name if manifest else None

    if kind == "uncond":
        setup["label_col"] = None
    else:
        setup["label_col"] = manifest.get("label_col") or (
            "name" if kind == "text" else infer_label_col(setup["dataset"], num_classes)
        )

    if "holdout" not in setup:
        # An augmented dataset has aug_index and was trained on a variant split;
        # the older curated subsets were trained on everything.
        setup["holdout"] = (
            "variant" if "aug_index" in csv_columns(setup["dataset"]) else None
        )

    setup.update({k: v for k, v in overrides.items() if v is not None})
    return setup


def load_real(setup, kind):
    """The training split the samples should be compared against.

    Held-out variants are excluded because every notebook metric compared
    against training data — including the copy detection, where comparing
    against unseen images would measure the wrong thing entirely.
    """
    label_col = setup["label_col"] or "group"
    if setup["holdout"]:
        dataset, _ = train_val_split(
            setup["dataset"], label_col=label_col,
            holdout=setup["holdout"], val_variants=setup["val_variants"],
        )
    else:
        dataset = ConditionedEmojiDataset.from_dir(setup["dataset"], label_col=label_col)

    # A fixed subsample: if this moved between snapshots, a metric drifting
    # across rounds could just be a different reference draw.
    generator = torch.Generator().manual_seed(setup["seed"])
    count = min(setup["real_sample"], len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:count]

    images = torch.stack([dataset[int(i)][0] for i in indices])
    labels = torch.tensor([dataset.labels[int(i)] for i in indices])
    return dataset, images, labels


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def sample_labels(kind, dataset, setup):
    """What to ask the model to draw, and how many.

    `n_fake` may be null in a manifest, which for a text run means "one sample
    per name" — the text notebook's behaviour, and the only setting under which
    the retrieval numbers mean what they did there. The other two need a number,
    so they fall back to the default.
    """
    n_fake = setup["n_fake"] or DEFAULTS["n_fake"]

    if kind == "uncond":
        return None, n_fake

    if kind == "text":
        n_names = dataset.num_classes
        if setup["n_fake"] and setup["n_fake"] < n_names:
            generator = torch.Generator().manual_seed(setup["seed"])
            labels = torch.randperm(n_names, generator=generator)[: setup["n_fake"]].sort().values
        else:
            labels = torch.arange(n_names)
        return labels, len(labels)

    per_class = max(1, n_fake // dataset.num_classes)
    labels = class_labels(dataset.num_classes, per_class)
    return labels, len(labels)


def evaluate_checkpoint(run_dir, checkpoint, overrides, device="cpu", extractors=None):
    """Score one checkpoint. Returns the metrics dict."""
    config = json.loads((run_dir / "config.json").read_text())
    model, kind, step = load_ema(run_dir / checkpoint, device)
    num_classes = getattr(model, "num_classes", 0)

    setup = resolve_setup(run_dir, config, kind, num_classes, overrides)
    dataset, real_images, real_labels = load_real(setup, kind)

    print(f"  arch: {type(model).__name__}"
          + (f", {num_classes} classes" if num_classes else "")
          + f"  |  step {step}")
    print(f"  setup: {setup['dataset']} label_col={setup['label_col']} "
          f"holdout={setup['holdout']} "
          f"(manifest: {setup['manifest'] or 'none — inferred'})")

    labels, n_fake = sample_labels(kind, dataset, setup)
    diffusion = Diffusion(timesteps=config.get("timesteps", 1000), device=device)

    torch.manual_seed(setup["seed"])   # so a re-run reproduces the same samples
    print(f"  generating {n_fake} samples (ddim {setup['ddim_steps']}, "
          f"guidance {setup['guidance']})...")
    fake_images = generate(
        model, diffusion, labels, n=n_fake, guidance_scale=setup["guidance"],
        ddim_steps=setup["ddim_steps"], device=device,
    )

    extractors = extractors or {}
    if kind == "text":
        # Class-level metrics use the emoji's group, matching the text notebook:
        # 1321 names would leave one real image per class, which no
        # distribution metric can say anything about.
        by_name = {row["name"]: row for row in dataset.meta}
        groups = sorted({row["group"] for row in dataset.meta})
        group_index = {g: i for i, g in enumerate(groups)}
        group_of = torch.tensor([group_index[by_name[n]["group"]] for n in dataset.classes])
        eval_real_labels, eval_fake_labels, class_names = (
            group_of[real_labels], group_of[labels], groups
        )
    elif kind == "class":
        eval_real_labels, eval_fake_labels, class_names = (
            real_labels, labels, dataset.classes
        )
    else:
        eval_real_labels = eval_fake_labels = class_names = None

    scores = metrics_lib.evaluate(
        real_images, eval_real_labels, fake_images, eval_fake_labels, class_names,
        device=device, **extractors,
    )

    if kind == "text":
        clip = extractors.get("clip") or metrics_lib.ClipFeatures(device=device)
        fake_feats = clip.encode_images(fake_images).cpu()
        scores.update(metrics_lib.name_retrieval(
            fake_feats, model.text_bank.cpu(), labels, n_names=dataset.num_classes
        ))

    scores.update({
        "run": run_dir.name, "checkpoint": checkpoint, "step": step,
        "conditioning": kind,
        **{k: setup[k] for k in
           ("dataset", "label_col", "holdout", "guidance", "ddim_steps",
            "real_sample", "n_fake", "seed")},
    })
    return scores


def write_results(run_dir, checkpoint, scores):
    name = ("metrics.json" if checkpoint == "ckpt.pt"
            else f"metrics_{Path(checkpoint).stem.replace('ckpt_', '')}.json")
    (run_dir / name).write_text(json.dumps(scores, indent=2, default=str), encoding="utf-8")

    # One row per evaluated checkpoint, so the metrics-vs-steps curve is a
    # single read_csv rather than a glob over JSON files.
    history = run_dir / "metrics_history.csv"
    rows = []
    if history.exists():
        with open(history, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["checkpoint"] != checkpoint]
    rows.append({c: scores.get(c, "") for c in HISTORY_COLUMNS})
    rows.sort(key=lambda r: (int(r["step"] or 0), r["checkpoint"]))

    with open(history, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return run_dir / name, history


def checkpoints_for(run_dir, requested, snapshots):
    if requested:
        return [requested]
    found = sorted(p.name for p in run_dir.glob("ckpt_step*.pt")) if snapshots else []
    if (run_dir / "ckpt.pt").exists():
        found.append("ckpt.pt")
    if not found:
        raise SystemExit(f"no checkpoints in {run_dir}")
    return found


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run names under runs/")
    parser.add_argument("--all", action="store_true", help="every run in runs/")
    parser.add_argument("--checkpoint", help="a specific file, e.g. ckpt_step070000.pt")
    parser.add_argument("--snapshots", action="store_true",
                        help="also score every archived ckpt_step*.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-fake", type=int)
    parser.add_argument("--real-sample", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--ddim-steps", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    names = sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir()) if args.all else args.runs
    if not names:
        raise SystemExit("name at least one run, or pass --all")

    overrides = {
        "n_fake": args.n_fake, "real_sample": args.real_sample,
        "guidance": args.guidance, "ddim_steps": args.ddim_steps, "seed": args.seed,
    }

    for name in names:
        run_dir = RUNS_DIR / name
        if not (run_dir / "config.json").exists():
            print(f"\n{name}: no config.json, skipping")
            continue

        for checkpoint in checkpoints_for(run_dir, args.checkpoint, args.snapshots):
            print(f"\n=== {name} / {checkpoint} ===")
            scores = evaluate_checkpoint(run_dir, checkpoint, overrides, args.device)
            path, history = write_results(run_dir, checkpoint, scores)
            headline = {k: scores[k] for k in
                        ("fid", "precision", "recall", "perceptual_copy_rate")
                        if k in scores}
            print("  " + "  ".join(f"{k}={v:.4f}" for k, v in headline.items()))
            print(f"  wrote {path.name} and {history.name}")


if __name__ == "__main__":
    main()
