"""
STEP 4
pseudo_mask_gen.py

Stage 2: Generate pseudo-masks for PlantDoc and PlantVillage images
using the Stage 1 trained model (best_model.pth).

For each image in PlantDoc and PlantVillage:
    1. Run the trained UNet through the image
    2. Apply softmax to get per-pixel class probabilities
    3. Take argmax to get predicted class per pixel
    4. Set any pixel whose max probability < CONFIDENCE_THRESHOLD to 0 (background)
    5. Save the resulting mask as a grayscale PNG

Output layout:
    data/tomato/pseudo_masks/
        plantdoc/
            train/
            val/
            test/
        plantvillage/
            train/
            val/
            test/

Usage:
    python pseudo_mask_gen.py
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from alt_data_loader_2 import IMAGE_SIZE, NUM_CLASSES, DATA_ROOT, META_CSV
from train_3 import UNetResNet50

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH      = Path("./outputs/checkpoints/20260508_002604/best_model.pth")  # update to best run
PSEUDO_MASK_DIR      = DATA_ROOT / "pseudo_masks"
CONFIDENCE_THRESHOLD = 0.7
BATCH_SIZE           = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Image transforms (no augmentation, just resize + normalise)
# ---------------------------------------------------------------------------

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_inference_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Single image inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_pseudo_mask(
    model: nn.Module,
    image: np.ndarray,
    transform: A.Compose,
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Run model on a single image and return a pseudo-mask.

    Args:
        model         : trained UNet
        image         : HxWx3 uint8 numpy array (RGB)
        transform     : inference transforms
        original_size : (H, W) of original image to resize mask back to

    Returns:
        pseudo_mask   : HxW uint8 numpy array with class indices
                        (low-confidence pixels set to 0)
    """
    # Apply transforms
    augmented = transform(image=image)
    tensor    = augmented["image"].unsqueeze(0).to(DEVICE)   # [1, 3, H, W]

    # Forward pass
    logits = model(tensor)                                    # [1, C, H, W]
    probs  = torch.softmax(logits, dim=1)                    # [1, C, H, W]

    # Per-pixel confidence and predicted class
    confidence, pred_class = probs.max(dim=1)                # [1, H, W] each

    confidence  = confidence.squeeze(0).cpu().numpy()        # [H, W]
    pred_class  = pred_class.squeeze(0).cpu().numpy().astype(np.uint8)  # [H, W]

    # Zero out low-confidence pixels
    pred_class[confidence < CONFIDENCE_THRESHOLD] = 0

    # Resize back to original image size using nearest neighbour
    # (preserve integer class indices, no interpolation)
    pseudo_mask = np.array(
        Image.fromarray(pred_class).resize(
            (original_size[1], original_size[0]),  # PIL uses (W, H)
            Image.NEAREST
        )
    )

    return pseudo_mask


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Stage 2 — Pseudo-Mask Generation")
    print(f"  Device     : {DEVICE}")
    print(f"  Checkpoint : {CHECKPOINT_PATH}")
    print(f"  Confidence : {CONFIDENCE_THRESHOLD}")
    print("=" * 60)

    # --- Load model ---
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            f"Update CHECKPOINT_PATH to point to your best_model.pth"
        )

    model = UNetResNet50(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print(f"\n[✓] Model loaded from {CHECKPOINT_PATH}")

    # --- Load metadata ---
    df = pd.read_csv(META_CSV)
    target_df = df[df["source"].isin(["plantdoc", "plantvillage"])].reset_index(drop=True)
    print(f"[✓] {len(target_df)} images to process (plantdoc + plantvillage)\n")

    transform = get_inference_transforms()

    # Track stats
    total_processed = 0
    total_skipped   = 0
    all_zero_masks  = 0

    # --- Process each image ---
    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Generating pseudo-masks"):
        img_path = DATA_ROOT / row["image_path"]

        if not img_path.exists():
            total_skipped += 1
            continue

        # Load image
        pil_img = Image.open(img_path).convert("RGB")
        original_size = (pil_img.height, pil_img.width)   # (H, W)
        image = np.array(pil_img)

        # Generate pseudo-mask
        pseudo_mask = generate_pseudo_mask(model, image, transform, original_size)

        # Track all-zero masks (healthy images or no confident predictions)
        if pseudo_mask.max() == 0:
            all_zero_masks += 1

        # Save pseudo-mask
        source = row["source"]
        split  = row["split"]
        fname  = Path(row["image_path"]).stem + ".png"

        save_dir  = PSEUDO_MASK_DIR / source / split
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / fname

        Image.fromarray(pseudo_mask).save(save_path)
        total_processed += 1

    # --- Update metadata.csv with pseudo-mask paths ---
    print("\n[Updating metadata.csv with pseudo-mask paths...]")

    def get_pseudo_mask_path(row):
        if row["source"] not in ("plantdoc", "plantvillage"):
            return row["mask_path"]
        fname = Path(row["image_path"]).stem + ".png"
        return str(
            Path("pseudo_masks") / row["source"] / row["split"] / fname
        )

    df["mask_path"] = df.apply(get_pseudo_mask_path, axis=1)
    df.to_csv(META_CSV, index=False)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Pseudo-Mask Generation Complete")
    print(f"  Processed  : {total_processed}")
    print(f"  Skipped    : {total_skipped} (missing images)")
    print(f"  All-zero   : {all_zero_masks} (no confident predictions / healthy)")
    print(f"  Saved to   : {PSEUDO_MASK_DIR}")
    print(f"  metadata.csv updated with pseudo-mask paths")
    print("=" * 60)


if __name__ == "__main__":
    main()