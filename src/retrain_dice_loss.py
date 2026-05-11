"""
STEP 5
retrain_dice_loss.py

Stage 3: Retrain UNet on combined dataset:
    - PlantSeg real masks (loss weight = 1.0)
    - PlantDoc + PlantVillage pseudo-masks (loss weight = 0.5)

Evaluates on the same PlantSeg test set as Stage 1 for fair comparison.

Usage:
    python retrain_dice_loss.py

Outputs (saved to outputs/checkpoints/stage3_<timestamp>/):
    best_model.pth      — best model weights by val mIoU
    last_model.pth      — final epoch weights
    training_log.csv    — per-epoch metrics
    config.txt          — run configuration
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(str(Path(__file__).parent))
from alt_data_loader_3 import (
    DATA_ROOT, META_CSV, IMAGE_SIZE, NUM_CLASSES,
    get_stage3_loaders, PlantSegDataset, get_val_transforms,
)
from train_3 import UNetResNet50, compute_miou

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to Stage 1 best model — used as starting weights for Stage 3
STAGE1_CHECKPOINT = Path("./outputs/checkpoints/train/20260508_120557/best_model.pth")  # update to your run

RUN_ID         = "stage3_" + datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_DIR = Path("outputs/checkpoints/retrain_dice") / RUN_ID
LOG_FILE       = CHECKPOINT_DIR / "training_log.csv"

NUM_EPOCHS          = 80    #changed from 50 to 80
BATCH_SIZE          = 8
LEARNING_RATE       = 0.0005   # lower than Stage 1 — fine-tuning, not training from scratch
MOMENTUM            = 0.9
WEIGHT_DECAY        = 0.0005
EARLY_STOP_PATIENCE = 20    #changed to from 10 to 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class weights — same as Stage 1
CLASS_WEIGHTS = torch.tensor(
    [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=torch.float32
)

# ---------------------------------------------------------------------------
# Weighted loss — applies per-sample weight (1.0 real, 0.5 pseudo)
# ---------------------------------------------------------------------------

class WeightedCrossEntropyDiceLoss(nn.Module):
    """
    Combined CrossEntropy + Dice loss.

    CrossEntropy helps with pixel-wise class prediction.
    Dice loss helps with segmentation overlap, especially when disease pixels
    are much smaller than background pixels.

    Pseudo-mask samples are still down-weighted using sample_weights.
    """

    def __init__(self, class_weights: torch.Tensor, dice_weight: float = 0.5):
        super().__init__()
        self.class_weights = class_weights
        self.dice_weight = dice_weight

    def dice_loss(self, logits, targets, sample_weights, smooth=1e-6):
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]

        # one-hot targets: [B, H, W] -> [B, C, H, W]
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=logits.shape[1]
        ).permute(0, 3, 1, 2).float()

        # ignore background class 0
        probs = probs[:, 1:, :, :]
        targets_one_hot = targets_one_hot[:, 1:, :, :]

        dims = (2, 3)

        intersection = (probs * targets_one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        loss = 1.0 - dice  # [B, C-1]

        # average over disease classes
        loss = loss.mean(dim=1)  # [B]

        # apply real/pseudo sample weighting
        sample_weights = sample_weights.to(logits.device).float()
        loss = loss * sample_weights

        return loss.mean()

    def forward(self, logits, targets, sample_weights):
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.class_weights.to(logits.device),
            reduction="none",
        )

        sample_weights = sample_weights.to(logits.device).float()
        ce = ce * sample_weights.view(-1, 1, 1)
        ce = ce.mean()

        dice = self.dice_loss(logits, targets, sample_weights)

        return ce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_miou = 0.0

    for images, masks, weights in loader:
        images  = images.to(device)
        masks   = masks.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks, weights)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item()
        total_miou += compute_miou(preds.cpu(), masks.cpu(), NUM_CLASSES)

    n = len(loader)
    return total_loss / n, total_miou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0

    for images, masks, weights in loader:
        images  = images.to(device)
        masks   = masks.to(device)
        weights = weights.to(device)

        outputs = model(images)
        loss    = criterion(outputs, masks, weights)

        preds = outputs.argmax(dim=1)
        total_loss += loss.item()
        total_miou += compute_miou(preds.cpu(), masks.cpu(), NUM_CLASSES)

    n = len(loader)
    return total_loss / n, total_miou / n


# ---------------------------------------------------------------------------
# Test set evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test_set(model, device) -> Tuple[float, float]:
    """
    Evaluate on PlantSeg test set — same split used for Stage 1 baseline.
    Returns (test_miou, test_macc).
    """
    model.eval()

    test_ds = PlantSegDataset(split="test", transform=get_val_transforms())
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_preds   = []
    all_targets = []

    for images, masks, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(masks)

    all_preds   = torch.cat(all_preds,   dim=0).view(-1)
    all_targets = torch.cat(all_targets, dim=0).view(-1)

    # mIoU (excluding background)
    iou_list = []
    acc_list = []
    for cls in range(1, NUM_CLASSES):
        pred_mask   = (all_preds == cls)
        target_mask = (all_targets == cls)

        intersection = (pred_mask & target_mask).sum().item()
        union        = (pred_mask | target_mask).sum().item()

        if union > 0:
            iou_list.append(intersection / union)

        if target_mask.sum() > 0:
            acc_list.append(
                (pred_mask & target_mask).sum().item() / target_mask.sum().item()
            )

    test_miou = float(np.mean(iou_list)) if iou_list else 0.0
    test_macc = float(np.mean(acc_list)) if acc_list else 0.0

    return test_miou, test_macc


# ---------------------------------------------------------------------------
# Config saving
# ---------------------------------------------------------------------------

def save_config(stage1_miou: float = 0.1003):
    with open(CHECKPOINT_DIR / "config.txt", "w") as f:
        f.write(f"RUN_ID             : {RUN_ID}\n")
        f.write(f"Stage              : 3 (retrain with pseudo-masks)\n")
        f.write(f"Stage1 checkpoint  : {STAGE1_CHECKPOINT}\n")
        f.write(f"Stage1 baseline    : val mIoU = {stage1_miou}\n")
        f.write(f"NUM_EPOCHS         : {NUM_EPOCHS}\n")
        f.write(f"BATCH_SIZE         : {BATCH_SIZE}\n")
        f.write(f"LEARNING_RATE      : {LEARNING_RATE}\n")
        f.write(f"EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}\n")
        f.write(f"NUM_CLASSES        : {NUM_CLASSES}\n")
        f.write(f"IMAGE_SIZE         : {IMAGE_SIZE}\n")
        f.write(f"CONFIDENCE_THRESH  : 0.7 (used in pseudo-mask generation)\n")
        f.write(f"PSEUDO_LOSS_WEIGHT : 0.5\n")
        f.write("LOSS_FUNCTION      : CrossEntropy + 0.5 DiceLoss\n")   #added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Stage 3 Retraining — UNet + pseudo-masks")
    print(f"  Device : {DEVICE}")
    print(f"  Run ID : {RUN_ID}")
    print("=" * 60)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_config()

    # --- Data ---
    train_loader, val_loader = get_stage3_loaders(
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    # --- Model: initialise from Stage 1 weights ---
    model = UNetResNet50(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)

    if STAGE1_CHECKPOINT.exists():
        model.load_state_dict(torch.load(STAGE1_CHECKPOINT, map_location=DEVICE))
        print(f"\n[✓] Loaded Stage 1 weights from {STAGE1_CHECKPOINT}")
    else:
        print(f"\n[WARN] Stage 1 checkpoint not found — training from scratch")

    # --- Loss, optimiser, scheduler ---
    criterion = WeightedCrossEntropyDiceLoss(
    class_weights=CLASS_WEIGHTS,
    dice_weight=0.5
    )   
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=5, verbose=True,
    )

    # --- Logging ---
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_miou", "val_loss", "val_miou", "lr"])

    # --- Training loop ---
    best_val_miou     = 0.0
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_miou = validate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_miou)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f}  mIoU: {train_miou:.4f} | "
            f"Val loss: {val_loss:.4f}  mIoU: {val_miou:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )

        # Log
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_miou, val_loss, val_miou, lr])

        # Save best
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"  [✓] New best val mIoU: {best_val_miou:.4f} — model saved")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n[Early Stop] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping.")
                break

    # Save final model
    torch.save(model.state_dict(), CHECKPOINT_DIR / "last_model.pth")

    # --- Final test set evaluation ---
    print("\n--- Final Evaluation on PlantSeg Test Set ---")
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=DEVICE))
    test_miou, test_macc = evaluate_test_set(model, DEVICE)

    print(f"\n  Stage 1 baseline val mIoU : 0.1003")
    print(f"  Stage 3 best val mIoU     : {best_val_miou:.4f}")
    print(f"  Stage 3 test mIoU         : {test_miou:.4f}")
    print(f"  Stage 3 test mAcc         : {test_macc:.4f}")

    # Save results
    with open(CHECKPOINT_DIR / "results.txt", "w") as f:
        f.write(f"Stage 1 baseline val mIoU : 0.1003\n")
        f.write(f"Stage 3 best val mIoU     : {best_val_miou:.4f}\n")
        f.write(f"Stage 3 test mIoU         : {test_miou:.4f}\n")
        f.write(f"Stage 3 test mAcc         : {test_macc:.4f}\n")

    print(f"\n[✓] Results saved to {CHECKPOINT_DIR}/results.txt")
    print(f"[✓] Checkpoints saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()