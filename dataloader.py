"""Minimal torch Dataset for the valhalla emoji images."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGES_DIR = "datasets/valhalla_emoji/images"


class EmojiDataset(Dataset):
    """Loads the 64x64 emoji PNGs as float tensors normalized to [-1, 1]."""

    def __init__(self, root=IMAGES_DIR, transform=None):
        self.paths = sorted(Path(root).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"no .png files found in {Path(root).resolve()}")

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),                       # uint8 HWC -> float32 CHW in [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = EmojiDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    batch = next(iter(loader))
    print(f"images: {len(dataset)}")
    print(f"batch:  {batch.shape} {batch.dtype}")
    print(f"range:  [{batch.min():.2f}, {batch.max():.2f}]")
