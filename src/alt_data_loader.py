"""
STEP 2
data_loader.py

Dataset classes and DataLoader factories for tomato disease segmentation.

Two dataset classes:
    PlantSegDataset     — images with real segmentation masks (PlantSeg only)
    PseudoMaskDataset   — images with pseudo-masks (PlantDoc + PlantVillage, Stage 3)

Two loader factories:
    get_stage1_loaders  — PlantSeg real masks only (train + val)
    get_stage3_loaders  — PlantSeg + pseudo-mask datasets combined (train + val)

Image size: 256x256 (safe for memory on HPC)
Augmentation: applied to train split only, always to image+mask simultaneously
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT  = Path("data/tomato")
META_CSV   = DATA_ROOT / "metadata.csv"
IMAGE_SIZE = 256   # resize both sides to this

# Loss weight applied to pseudo-mask samples in Stage 3
# (handled in training loop, stored here for reference)
PSEUDO_MASK_LOSS_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms() -> A.Compose:
    """
    Geometric augmentations for training.
    Applied to image AND mask simultaneously so pixel labels stay aligned.
    No colour jitter on the mask — albumentations handles this automatically
    when you pass the mask via the 'mask' keyword.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),   # ImageNet stats
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """
    Validation/test transforms — resize and normalise only, no augmentation.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset: PlantSeg (real masks)
# ---------------------------------------------------------------------------

class PlantSegDataset(Dataset):
    """
    Loads PlantSeg tomato images paired with real segmentation masks.

    Each item returns:
        image   : FloatTensor [3, H, W]   normalised RGB
        mask    : LongTensor  [H, W]      integer class indices (0 = background)
        weight  : float                   always 1.0 (real mask, full loss weight)
    """

    def __init__(self, split: str, transform: A.Compose = None):
        """
        Args:
            split     : 'train', 'val', or 'test'
            transform : albumentations Compose pipeline
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        df = pd.read_csv(META_CSV)
        self.df = df[
            (df["source"] == "plantseg") &
            (df["split"]  == split) &
            (df["has_real_mask"] == True)
        ].reset_index(drop=True)

        self.transform = transform
        self.split     = split

        print(f"[PlantSegDataset] split={split}  images={len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Load image (RGB)
        img_path = DATA_ROOT / row["image_path"]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load mask (grayscale PNG → integer array)
        mask_path = DATA_ROOT / row["mask_path"]
        mask = np.array(Image.open(mask_path).convert("L"))  # [H, W] uint8

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # FloatTensor [3, H, W]
            mask  = augmented["mask"]    # Tensor [H, W]

        mask = mask.long()   # CrossEntropyLoss expects LongTensor

        return image, mask, 1.0   # weight=1.0 for real masks


# ---------------------------------------------------------------------------
# Dataset: PseudoMask (PlantDoc + PlantVillage, Stage 3)
# ---------------------------------------------------------------------------

class PseudoMaskDataset(Dataset):
    """
    Loads PlantDoc / PlantVillage images paired with pseudo-masks generated
    in Stage 2.  Used only in Stage 3 retraining.

    Pseudo-masks are stored in the same grayscale PNG format as real masks,
    with low-confidence pixels already set to 0 (background) during generation.

    Each item returns:
        image   : FloatTensor [3, H, W]
        mask    : LongTensor  [H, W]
        weight  : float                   PSEUDO_MASK_LOSS_WEIGHT (0.5)
    """

    def __init__(
        self,
        split: str,
        sources: tuple[str, ...] = ("plantdoc", "plantvillage"),
        pseudo_mask_dir: str | Path = "pseudo_masks",
        transform: A.Compose = None,
    ):
        """
        Args:
            split           : 'train', 'val', or 'test'
            sources         : which datasets to include
            pseudo_mask_dir : folder (relative to DATA_ROOT) where Stage 2
                              saved the pseudo-mask PNGs.  Expected layout:
                                data/tomato/pseudo_masks/{plantdoc,plantvillage}/
                                    {train,val,test}/filename.png
            transform       : albumentations Compose pipeline
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        df = pd.read_csv(META_CSV)
        self.df = df[
            (df["source"].isin(sources)) &
            (df["split"] == split)
        ].reset_index(drop=True)

        self.transform       = transform
        self.split           = split
        self.pseudo_mask_dir = DATA_ROOT / pseudo_mask_dir

        print(
            f"[PseudoMaskDataset] split={split}  "
            f"sources={sources}  images={len(self.df)}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _pseudo_mask_path(self, row) -> Path:
        """
        Derive pseudo-mask path from the image's source + filename.
        Pseudo-masks are saved by Stage 2 using the same filename as the image.
        """
        img_path = Path(row["image_path"])   # e.g. plantdoc/images/train/leaf.jpg
        filename  = img_path.stem + ".png"   # always PNG regardless of image ext
        source    = row["source"]
        split     = row["split"]
        return self.pseudo_mask_dir / source / split / filename

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Load image
        img_path = DATA_ROOT / row["image_path"]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load pseudo-mask (generated by Stage 2)
        pmask_path = self._pseudo_mask_path(row)
        if not pmask_path.exists():
            raise FileNotFoundError(
                f"Pseudo-mask not found: {pmask_path}\n"
                f"Run Stage 2 (pseudo-mask generation) before using PseudoMaskDataset."
            )
        mask = np.array(Image.open(pmask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        mask = mask.long()

        return image, mask, PSEUDO_MASK_LOSS_WEIGHT


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def get_stage1_loaders(
    batch_size: int = 8,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Stage 1: train and val loaders using PlantSeg real masks only.

    Returns:
        train_loader, val_loader
    """
    train_ds = PlantSegDataset(split="train", transform=get_train_transforms())
    val_ds   = PlantSegDataset(split="val",   transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n[Stage 1 Loaders] train={len(train_ds)}  val={len(val_ds)}")
    return train_loader, val_loader


def get_stage3_loaders(
    batch_size: int = 8,
    num_workers: int = 2,
    pseudo_mask_dir: str | Path = "pseudo_masks",
) -> tuple[DataLoader, DataLoader]:
    """
    Stage 3: train and val loaders combining PlantSeg real masks
    with PlantDoc + PlantVillage pseudo-masks.

    Each sample carries a 'weight' field (1.0 or 0.5) that the
    training loop uses to scale the loss.

    Returns:
        train_loader, val_loader
    """
    # --- train ---
    plantseg_train  = PlantSegDataset(
        split="train", transform=get_train_transforms()
    )
    pseudo_train    = PseudoMaskDataset(
        split="train",
        pseudo_mask_dir=pseudo_mask_dir,
        transform=get_train_transforms(),
    )
    train_ds = ConcatDataset([plantseg_train, pseudo_train])

    # --- val (PlantSeg only — pseudo-mask val is not ground truth) ---
    val_ds = PlantSegDataset(split="val", transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"\n[Stage 3 Loaders] "
        f"train={len(train_ds)} (plantseg={len(plantseg_train)} + pseudo={len(pseudo_train)})  "
        f"val={len(val_ds)}"
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Stage 1 DataLoader sanity check ===\n")
    train_loader, val_loader = get_stage1_loaders(batch_size=4, num_workers=0)

    images, masks, weights = next(iter(train_loader))
    print(f"  image batch : {images.shape}   dtype={images.dtype}")
    print(f"  mask batch  : {masks.shape}    dtype={masks.dtype}")
    print(f"  weights     : {weights}")
    print(f"  mask unique values: {masks.unique().tolist()}")
    print("\n[✓] DataLoader OK")