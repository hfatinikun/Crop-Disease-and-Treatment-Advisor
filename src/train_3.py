"""
STEP 3
train.py

Stage 1 training script for tomato disease segmentation.
Architecture: UNet with pretrained ResNet50 encoder (transfer learning)
Trains on PlantSeg real masks only.

Usage:
    python train.py

Outputs (saved to checkpoints/):
    best_model.pth      — best model weights by val mIoU
    last_model.pth      — final epoch weights
    training_log.csv    — per-epoch metrics
"""

import os
import csv
import time
from pathlib import Path

# make sure this is at the top of data_loader.py
from typing import Optional, Tuple

#to assist with logging runs, checkpoints, etc
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

# Import our DataLoader factory
import sys
sys.path.append(str(Path(__file__).parent))
from alt_data_loader_2 import get_stage1_loaders, NUM_CLASSES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN_ID         = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_DIR = Path("outputs/checkpoints") / RUN_ID
LOG_FILE       = CHECKPOINT_DIR / "training_log.csv"

NUM_EPOCHS     = 50
BATCH_SIZE     = 8
LEARNING_RATE  = 0.001
MOMENTUM       = 0.9
WEIGHT_DECAY   = 0.0005
EARLY_STOP_PATIENCE = 10   # stop if val mIoU doesn't improve for this many epochs
#UNFREEZE_EPOCH = 10  # unfreeze encoder after this many epochs

#weighted loss incorporated to handle class imbalanace -> background dominating training -> penalizes disease class errors more heavily
CLASS_WEIGHTS = torch.tensor(
    [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # low weight for background, equal for diseases
    dtype=torch.float32
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# UNet with pretrained ResNet50 encoder
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two consecutive Conv → BN → ReLU layers."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetResNet50(nn.Module):
    """
    UNet with a pretrained ResNet50 encoder.

    Encoder skip connections come from ResNet50 intermediate layers:
        layer0  (stem):  [B, 64,  H/2,  W/2]
        layer1:          [B, 256, H/4,  W/4]
        layer2:          [B, 512, H/8,  W/8]
        layer3:          [B, 1024,H/16, W/16]
        layer4 (bridge): [B, 2048,H/32, W/32]

    Decoder upsamples back to original resolution.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # --- Encoder (pretrained ResNet50) ---
        backbone = models.resnet50(pretrained=pretrained)

        # Split ResNet50 into stages for skip connections
        self.encoder0 = nn.Sequential(
            backbone.conv1,   # 7x7 conv, stride 2
            backbone.bn1,
            backbone.relu,
        )                     # output: [B, 64, H/2, W/2]

        self.pool     = backbone.maxpool   # [B, 64, H/4, W/4]
        self.encoder1 = backbone.layer1   # [B, 256, H/4, W/4]
        self.encoder2 = backbone.layer2   # [B, 512, H/8, W/8]
        self.encoder3 = backbone.layer3   # [B, 1024, H/16, W/16]
        self.encoder4 = backbone.layer4   # [B, 2048, H/32, W/32]  ← bridge

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024 + 1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512 + 512, 256)

        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256 + 256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)

        # Final upsample back to original resolution (undo stride-2 in encoder0)
        self.up0    = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0   = ConvBlock(32, 32)

        # Classification head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encode ---
        e0 = self.encoder0(x)        # [B, 64,   H/2,  W/2]
        p  = self.pool(e0)           # [B, 64,   H/4,  W/4]
        e1 = self.encoder1(p)        # [B, 256,  H/4,  W/4]
        e2 = self.encoder2(e1)       # [B, 512,  H/8,  W/8]
        e3 = self.encoder3(e2)       # [B, 1024, H/16, W/16]
        e4 = self.encoder4(e3)       # [B, 2048, H/32, W/32]

        # --- Decode with skip connections ---
        d4 = self.up4(e4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e0], dim=1))

        d0 = self.up0(d1)
        d0 = self.dec0(d0)

        return self.head(d0)   # [B, num_classes, H, W]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Compute mean Intersection over Union (mIoU) over all classes.
    Ignores classes not present in both pred and target for that batch.
    """
    iou_list = []
    preds   = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(1, num_classes): #start at 1 -> skip background=0
        pred_mask   = (preds == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().item()
        union        = (pred_mask | target_mask).sum().item()

        if union == 0:
            continue   # class not present in this batch, skip
        iou_list.append(intersection / union)

    return np.mean(iou_list) if iou_list else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_miou = 0.0

    for images, masks, weights in loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)                    # [B, C, H, W]
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)              # [B, H, W]
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
        images = images.to(device)
        masks  = masks.to(device)

        outputs = model(images)
        loss    = criterion(outputs, masks)

        preds = outputs.argmax(dim=1)
        total_loss += loss.item()
        total_miou += compute_miou(preds.cpu(), masks.cpu(), NUM_CLASSES)

    n = len(loader)
    return total_loss / n, total_miou / n

## --- Freeze encoder for first N epochs ---
# def freeze_encoder(model):
#     for name, param in model.named_parameters():
#         if name.startswith("encoder"):
#             param.requires_grad = False
#     print("[✓] Encoder frozen")

# def unfreeze_encoder(model):
#     for name, param in model.named_parameters():
#         if name.startswith("encoder"):
#             param.requires_grad = True
#     print("[✓] Encoder unfrozen")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Stage 1 Training — UNet + ResNet50 encoder")
    print(f"  Device : {DEVICE}")
    print(f"  Classes: {NUM_CLASSES}")
    print("=" * 60)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    train_loader, val_loader = get_stage1_loaders(
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    # --- Model ---
    model = UNetResNet50(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    print(f"\n[Model] UNet-ResNet50 | params: {sum(p.numel() for p in model.parameters()):,}")

    # #freeze model
    # freeze_encoder(model)

    # --- Loss, optimiser, scheduler ---
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
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
    best_val_miou    = 0.0
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):

        # # unfreeze encoder after UNFREEZE_EPOCH epochs
        # if epoch == UNFREEZE_EPOCH + 1:
        #     unfreeze_encoder(model)  

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

        # Log to CSV
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_miou, val_loss, val_miou, lr])

        # Save best model
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
    print(f"\nTraining complete. Best val mIoU: {best_val_miou:.4f}")
    print(f"    Checkpoints saved to: {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
