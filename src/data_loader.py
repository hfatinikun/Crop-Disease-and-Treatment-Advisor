"""
dataloader.py
=============
Augmentation pipelines and DataLoader factory for the tomato disease classifier.

What this provides
------------------
  get_transforms(split, img_size)   → torchvision transform pipeline
  TomatoDataset                     → thin ImageFolder wrapper with source tracking
  get_dataloaders(cfg)              → returns {train, val, test} DataLoader dict
  compute_class_weights(dataset)    → inverse-frequency weights for weighted loss
  show_batch(loader, n)             → quick visual sanity-check (saves a grid PNG)

Augmentation strategy
---------------------
  Train  : random horizontal/vertical flip, rotation, colour jitter,
           random resized crop, random erasing (occlusion robustness).
           All applied ONLY to training images — never val/test.
  Val/Test: deterministic centre-crop + normalise.

Normalisation uses ImageNet statistics (mean/std), which is correct when
fine-tuning a pretrained ResNet or EfficientNet backbone.

Usage (standalone test)
-----------------------
  python dataloader.py --data data/processed --batch 32 --workers 4

Usage (from training script)
----------------------------
  from dataloader import get_dataloaders, compute_class_weights

  cfg = {
      "data_root":  "data/processed",
      "img_size":   224,          # match backbone input (224 for ResNet/EfficientNet)
      "batch_size": 32,
      "num_workers": 4,
      "pin_memory": True,
  }
  loaders = get_dataloaders(cfg)
  class_weights = compute_class_weights(loaders["train"].dataset)
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

log = logging.getLogger(__name__)

# ── ImageNet statistics (used when loading pretrained backbones) ───────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 224) -> T.Compose:
    """
    Return the appropriate transform pipeline for a given split.

    Parameters
    ----------
    split    : "train" | "val" | "test"
    img_size : target square size (should match backbone input, e.g. 224)
    """
    if split == "train":
        return T.Compose([
            # ── geometric augmentations ──────────────────────────────────────
            # RandomResizedCrop simulates varying distances / zoom levels.
            # scale=(0.6, 1.0) means we never crop away more than 40% of area.
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),      # less common but valid for leaves
            T.RandomRotation(degrees=30),

            # ── colour augmentations ─────────────────────────────────────────
            # Simulate different lighting conditions and camera sensors.
            # brightness/contrast/saturation stay moderate to preserve disease
            # colour cues (yellowing, browning) — don't distort too aggressively.
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,          # hue shift kept very small — disease colours matter
            ),
            T.RandomGrayscale(p=0.05),         # rare: forces texture-based learning

            # ── tensor conversion & normalise ────────────────────────────────
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),

            # ── occlusion robustness ─────────────────────────────────────────
            # RandomErasing after ToTensor. Simulates partial occlusion by
            # other leaves, insects, or motion blur patches.
            T.RandomErasing(
                p=0.2,
                scale=(0.02, 0.15),   # erase at most 15% of the image area
                ratio=(0.3, 3.0),
                value=0,
            ),
        ])

    else:   # val or test — fully deterministic, no augmentation
        return T.Compose([
            # Slightly larger resize then centre-crop gives a clean evaluation
            # frame without arbitrary edge effects from direct resize.
            T.Resize(int(img_size * 1.14)),    # e.g. 256 for img_size=224
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ── dataset ────────────────────────────────────────────────────────────────────

class TomatoDataset(ImageFolder):
    """
    Thin wrapper around ImageFolder.

    ImageFolder already handles the train/val/test subdirectory structure
    produced by filter_datasets.py. We add:
      - a human-readable class list property
      - optional transform override (useful for test-time augmentation later)
    """

    def __init__(self, root: str | Path, split: str, img_size: int = 224, transform=None):
        """
        Parameters
        ----------
        root     : path to data/processed  (parent of train/ val/ test/)
        split    : "train" | "val" | "test"
        img_size : fed to get_transforms if transform is None
        transform: override — pass a custom pipeline or None to use the default
        """
        split_dir = Path(root) / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Run filter_datasets.py first."
            )
        super().__init__(
            root      = str(split_dir),
            transform = transform if transform is not None else get_transforms(split, img_size),
        )
        self.split    = split
        self.img_size = img_size

    @property
    def class_names(self) -> list[str]:
        """Sorted list of class folder names (same order as model output logits)."""
        return self.classes

    def class_counts(self) -> dict[str, int]:
        """Number of images per class."""
        from collections import Counter
        label_counts = Counter(label for _, label in self.samples)
        return {self.classes[idx]: count for idx, count in sorted(label_counts.items())}

    def __repr__(self) -> str:
        return (
            f"TomatoDataset(split={self.split!r}, "
            f"classes={len(self.classes)}, "
            f"images={len(self.samples)})"
        )


# ── class weights ──────────────────────────────────────────────────────────────

def compute_class_weights(dataset: TomatoDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for use with
    torch.nn.CrossEntropyLoss(weight=...).

    Formula: w_c = N / (C * n_c)
      where N = total images, C = number of classes, n_c = images in class c.

    Returns a float tensor of shape (C,) on CPU.
    """
    from collections import Counter
    label_list = [label for _, label in dataset.samples]
    counts = Counter(label_list)
    n_total = len(label_list)
    n_classes = len(dataset.classes)

    weights = torch.zeros(n_classes, dtype=torch.float32)
    for class_idx, count in counts.items():
        weights[class_idx] = n_total / (n_classes * count)

    log.info("Class weights (inverse frequency):")
    for i, cls in enumerate(dataset.classes):
        log.info("  [%d] %-38s  w=%.3f  n=%d", i, cls, weights[i].item(), counts[i])

    return weights


def compute_sampler_weights(dataset: TomatoDataset) -> WeightedRandomSampler:
    """
    Alternative to class-weighted loss: oversample minority classes
    so every class appears equally often within each epoch.

    Use this OR compute_class_weights — not both simultaneously.
    Weighted loss is usually preferred (more stable); this is provided
    as an option if your minority classes are very small (<100 images).
    """
    from collections import Counter
    label_list   = [label for _, label in dataset.samples]
    counts       = Counter(label_list)
    class_weight = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = torch.tensor([class_weight[label] for label in label_list])
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )


# ── dataloader factory ─────────────────────────────────────────────────────────

def get_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    """
    Build and return a dict of DataLoaders for train, val, and test splits.

    Expected cfg keys
    -----------------
    data_root   : str | Path  — path to data/processed
    img_size    : int         — input image size (default 224)
    batch_size  : int         — images per batch (default 32)
    num_workers : int         — parallel loading workers (default 4)
    pin_memory  : bool        — True speeds up GPU transfer (default True)
    use_sampler : bool        — oversample minority classes (default False)
                               if False, use compute_class_weights instead

    Returns
    -------
    {
        "train": DataLoader,
        "val":   DataLoader,
        "test":  DataLoader,
        "class_names":    list[str],
        "class_weights":  torch.Tensor,   # for weighted loss (if use_sampler=False)
        "num_classes":    int,
    }
    """
    root        = Path(cfg["data_root"])
    img_size    = cfg.get("img_size",    224)
    batch_size  = cfg.get("batch_size",  32)
    num_workers = cfg.get("num_workers", 4)
    pin_memory  = cfg.get("pin_memory",  True)
    use_sampler = cfg.get("use_sampler", False)

    datasets: dict[str, TomatoDataset] = {
        split: TomatoDataset(root, split, img_size)
        for split in ("train", "val", "test")
    }

    # Verify all splits have the same class set
    train_classes = datasets["train"].classes
    for split in ("val", "test"):
        if datasets[split].classes != train_classes:
            raise RuntimeError(
                f"Class mismatch between train and {split} splits.\n"
                f"This usually means filter_datasets.py left some classes empty.\n"
                f"Train classes : {train_classes}\n"
                f"{split} classes: {datasets[split].classes}"
            )

    train_ds = datasets["train"]
    class_weights = compute_class_weights(train_ds)

    # Sampler (oversampling) vs shuffle — mutually exclusive
    if use_sampler:
        sampler = compute_sampler_weights(train_ds)
        train_loader_kwargs = dict(sampler=sampler, shuffle=False)
    else:
        train_loader_kwargs = dict(shuffle=True)

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            batch_size  = batch_size,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = True,    # avoids batch-norm issues on tiny last batch
            **train_loader_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        ),
    }

    # Attach metadata so the training script doesn't need to re-open files
    loaders["class_names"]   = train_classes
    loaders["class_weights"] = class_weights
    loaders["num_classes"]   = len(train_classes)

    # Log a summary
    log.info("DataLoaders ready:")
    for split in ("train", "val", "test"):
        ds = datasets[split]
        log.info(
            "  %-5s  %5d images  |  %d batches of %d  |  %d classes",
            split, len(ds), len(loaders[split]), batch_size, len(ds.classes),
        )

    return loaders


# ── visual sanity check ────────────────────────────────────────────────────────

def show_batch(
    loader: DataLoader,
    n: int = 16,
    save_path: str | Path = "batch_preview.png",
) -> None:
    """
    Grab one batch, un-normalise, and save a grid image to disk.
    Useful for confirming augmentations look correct before a long training run.
    """
    import torchvision.utils as vutils
    import numpy as np
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping batch preview.")
        return

    imgs, labels = next(iter(loader))
    imgs = imgs[:n]
    labels = labels[:n]

    # Un-normalise: x = x * std + mean
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    imgs = (imgs * std + mean).clamp(0, 1)

    grid = vutils.make_grid(imgs, nrow=int(n**0.5), padding=2)
    np_grid = grid.permute(1, 2, 0).numpy()

    class_names = loader.dataset.classes
    label_str   = "  ".join(class_names[l] for l in labels.tolist())

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(np_grid)
    ax.set_title(f"Batch preview ({loader.dataset.split})\n{label_str}", fontsize=7)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info("Batch preview saved to %s", save_path)


# ── standalone test ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test dataloader and augmentation pipeline.")
    p.add_argument("--data",    type=str, default="data/processed")
    p.add_argument("--size",    type=int, default=224)
    p.add_argument("--batch",   type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--preview", action="store_true",
                   help="Save a batch preview image to disk.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
                        datefmt="%H:%M:%S")

    args = parse_args()
    cfg = {
        "data_root":   args.data,
        "img_size":    args.size,
        "batch_size":  args.batch,
        "num_workers": args.workers,
        "pin_memory":  torch.cuda.is_available(),
    }

    loaders = get_dataloaders(cfg)

    print(f"\nClass index mapping ({loaders['num_classes']} classes):")
    for i, name in enumerate(loaders["class_names"]):
        print(f"  {i:2d}  {name}")

    print(f"\nClass weights (for weighted loss):")
    for i, (name, w) in enumerate(zip(loaders["class_names"], loaders["class_weights"])):
        print(f"  {i:2d}  {name:<40}  {w:.4f}")

    # Iterate one batch and report shape
    imgs, labels = next(iter(loaders["train"]))
    print(f"\nSample batch — images: {tuple(imgs.shape)}, labels: {tuple(labels.shape)}")
    print(f"Image dtype: {imgs.dtype}, min: {imgs.min():.3f}, max: {imgs.max():.3f}")

    if args.preview:
        show_batch(loaders["train"], n=16, save_path="batch_preview.png")
        show_batch(loaders["val"],   n=16, save_path="batch_preview_val.png")
