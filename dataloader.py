"""Torch Datasets for the emoji images: unconditional and class-conditional."""

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# The original unconditional run trained on this one.
IMAGES_DIR = "datasets/valhalla_emoji/images"

# Curated openmoji subsets for the conditional runs — see make_subsets.py.
OPENMOJI_10 = "datasets/openmoji_10"
OPENMOJI_200 = "datasets/openmoji_200"
OPENMOJI_500 = "datasets/openmoji_500"


def default_transform():
    """uint8 PIL image -> float32 CHW tensor in [-1, 1]."""
    return transforms.Compose([
        transforms.ToTensor(),                       # uint8 HWC -> float32 CHW in [0, 1]
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
    ])


class EmojiDataset(Dataset):
    """Loads the 64x64 emoji PNGs as float tensors normalized to [-1, 1]."""

    def __init__(self, root=IMAGES_DIR, transform=None):
        self.paths = sorted(Path(root).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"no .png files found in {Path(root).resolve()}")

        self.transform = transform or default_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image)


class ConditionedEmojiDataset(Dataset):
    """Same images, paired with an integer class index read from a metadata CSV.

    The CSV columns differ per dataset, so `filename_col` / `label_col` are
    arguments: point them at whatever the new dataset calls them.
    """

    @classmethod
    def from_dir(cls, root=OPENMOJI_200, **kwargs):
        """Load a dataset laid out as <root>/images + <root>/metadata.csv."""
        root = Path(root)
        return cls(root / "images", root / "metadata.csv", **kwargs)

    def __init__(
        self,
        root=f"{OPENMOJI_200}/images",
        csv_path=f"{OPENMOJI_200}/metadata.csv",
        filename_col="filename",
        label_col="group",
        classes=None,
        transform=None,
    ):
        self.paths = sorted(Path(root).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"no .png files found in {Path(root).resolve()}")

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"no rows in {Path(csv_path).resolve()}")

        missing_cols = [c for c in (filename_col, label_col) if c not in rows[0]]
        if missing_cols:
            raise ValueError(
                f"{csv_path} has no column(s) {missing_cols}; "
                f"available: {sorted(rows[0])}"
            )

        rows_by_file = {row[filename_col]: row for row in rows}

        unlabelled = [p.name for p in self.paths if p.name not in rows_by_file]
        if unlabelled:
            raise ValueError(
                f"{len(unlabelled)} image(s) have no row in {csv_path}, e.g. "
                f"{unlabelled[:5]}"
            )

        # The full CSV row per image, aligned with self.paths: the notebook uses
        # `name` for CLIP prompts and for labelling samples.
        self.meta = [rows_by_file[p.name] for p in self.paths]
        names = [row[label_col] for row in self.meta]

        # `classes` pins the label space. Two subsets of the same dataset would
        # otherwise disagree on what index 0 means, which silently invalidates
        # any comparison between models trained on them.
        self.classes = sorted(set(names)) if classes is None else list(classes)
        unknown = sorted(set(names) - set(self.classes))
        if unknown:
            raise ValueError(f"labels not in the given `classes`: {unknown}")

        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.labels = [self.class_to_idx[name] for name in names]

        self.transform = transform or default_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        # Default collation turns the ints into an int64 (B,) tensor — what
        # nn.Embedding wants.
        return self.transform(image), self.labels[i]


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = EmojiDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    batch = next(iter(loader))
    print(f"images: {len(dataset)}")
    print(f"batch:  {batch.shape} {batch.dtype}")
    print(f"range:  [{batch.min():.2f}, {batch.max():.2f}]")

    for subset in (OPENMOJI_10, OPENMOJI_200):
        cond = ConditionedEmojiDataset.from_dir(subset)
        images, labels = next(iter(DataLoader(cond, batch_size=8, shuffle=True)))

        print(f"\n{subset}: {len(cond)} images, {cond.num_classes} classes")
        print(f"classes: {cond.classes}")
        print(f"images:  {images.shape} {images.dtype}")
        print(f"labels:  {labels.shape} {labels.dtype} -> {labels.tolist()}")
        print(f"example: {cond.meta[0]['name']!r} -> {cond.classes[cond.labels[0]]}")
