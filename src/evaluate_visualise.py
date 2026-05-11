"""
evaluate_visualize.py

Evaluate and visualise Stage 1 vs Stage 3 segmentation models.

Outputs:
    outputs/evaluation/
        metrics_summary.csv
        per_image_metrics.csv
        visual_panels/
            sample_000.png
            sample_001.png
            ...

Run:
    python src/7_evaluate_visualise.py \
        --stage1 outputs/checkpoints/train/TIMESTAMP/best_model.pth \
        --stage3 outputs/checkpoints/retrain/TIMESTAMP/best_model.pth \
        --n-vis 20
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Allow imports from src/
sys.path.append(str(Path(__file__).parent))

from train_3 import UNetResNet50
from alt_data_loader_2 import (
    PlantSegDataset,
    get_val_transforms,
    NUM_CLASSES,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CLASS_NAMES = {
    0: "background",
    1: "bacterial spot",
    2: "early blight",
    3: "late blight",
    4: "leaf mold",
    5: "mosaic virus",
    6: "septoria leaf spot",
    7: "yellow leaf curl virus",
}


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    """Load UNetResNet50 model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = UNetResNet50(num_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model


def compute_iou_and_acc(pred: torch.Tensor, target: torch.Tensor):
    """
    Compute mean IoU and mean accuracy, excluding background class 0.

    pred, target: [H, W]
    """
    ious = []
    accs = []

    per_class = {}

    for cls in range(1, NUM_CLASSES):
        pred_mask = pred == cls
        target_mask = target == cls

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        target_pixels = target_mask.sum().item()

        if union > 0:
            iou = intersection / union
            ious.append(iou)
        else:
            iou = np.nan

        if target_pixels > 0:
            acc = intersection / target_pixels
            accs.append(acc)
        else:
            acc = np.nan

        per_class[cls] = {
            "iou": iou,
            "acc": acc,
            "target_pixels": target_pixels,
            "pred_pixels": pred_mask.sum().item(),
        }

    miou = float(np.nanmean(ious)) if len(ious) > 0 else 0.0
    macc = float(np.nanmean(accs)) if len(accs) > 0 else 0.0

    return miou, macc, per_class


def unnormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalised tensor [3, H, W] back to displayable image [H, W, 3].
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()

    return img


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert class-index mask [H, W] into RGB image for visualisation.
    """
    colours = np.array([
        [0, 0, 0],          # 0 background - black
        [220, 20, 60],      # 1
        [255, 140, 0],      # 2
        [255, 215, 0],      # 3
        [34, 139, 34],      # 4
        [30, 144, 255],     # 5
        [138, 43, 226],     # 6
        [255, 105, 180],    # 7
    ], dtype=np.uint8)

    return colours[mask]


def save_visual_panel(
    image: np.ndarray,
    gt_mask: np.ndarray,
    stage1_pred: np.ndarray,
    stage3_pred: np.ndarray,
    save_path: Path,
    title: str,
):
    """Save one visual comparison panel."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(mask_to_rgb(gt_mask))
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(mask_to_rgb(stage1_pred))
    axes[2].set_title("Stage 1 prediction")
    axes[2].axis("off")

    axes[3].imshow(mask_to_rgb(stage3_pred))
    axes[3].set_title("Stage 3 prediction")
    axes[3].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


@torch.no_grad()
def evaluate_models(stage1_model, stage3_model, loader, output_dir: Path, n_vis: int):
    """Evaluate both models on the same PlantSeg test set."""
    visual_dir = output_dir / "visual_panels"
    visual_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows = []

    stage1_mious = []
    stage1_maccs = []
    stage3_mious = []
    stage3_maccs = []

    sample_idx = 0

    for batch_idx, (images, masks, _) in enumerate(loader):
        images = images.to(DEVICE)
        masks = masks.cpu()

        stage1_logits = stage1_model(images)
        stage3_logits = stage3_model(images)

        stage1_preds = stage1_logits.argmax(dim=1).cpu()
        stage3_preds = stage3_logits.argmax(dim=1).cpu()

        batch_size = images.shape[0]

        for i in range(batch_size):
            gt = masks[i]
            pred1 = stage1_preds[i]
            pred3 = stage3_preds[i]

            stage1_miou, stage1_macc, _ = compute_iou_and_acc(pred1, gt)
            stage3_miou, stage3_macc, _ = compute_iou_and_acc(pred3, gt)

            stage1_mious.append(stage1_miou)
            stage1_maccs.append(stage1_macc)
            stage3_mious.append(stage3_miou)
            stage3_maccs.append(stage3_macc)

            improvement = stage3_miou - stage1_miou

            per_image_rows.append({
                "sample_index": sample_idx,
                "stage1_miou": stage1_miou,
                "stage1_macc": stage1_macc,
                "stage3_miou": stage3_miou,
                "stage3_macc": stage3_macc,
                "miou_difference_stage3_minus_stage1": improvement,
            })

            if sample_idx < n_vis:
                image_np = unnormalize_image(images[i])
                gt_np = gt.numpy().astype(np.uint8)
                pred1_np = pred1.numpy().astype(np.uint8)
                pred3_np = pred3.numpy().astype(np.uint8)

                save_path = visual_dir / f"sample_{sample_idx:03d}.png"
                title = (
                    f"Sample {sample_idx} | "
                    f"Stage1 mIoU={stage1_miou:.3f} | "
                    f"Stage3 mIoU={stage3_miou:.3f} | "
                    f"Diff={improvement:.3f}"
                )

                save_visual_panel(
                    image=image_np,
                    gt_mask=gt_np,
                    stage1_pred=pred1_np,
                    stage3_pred=pred3_np,
                    save_path=save_path,
                    title=title,
                )

            sample_idx += 1

    summary = {
        "stage1_mean_miou": float(np.mean(stage1_mious)),
        "stage1_mean_macc": float(np.mean(stage1_maccs)),
        "stage3_mean_miou": float(np.mean(stage3_mious)),
        "stage3_mean_macc": float(np.mean(stage3_maccs)),
        "mean_miou_difference_stage3_minus_stage1": float(np.mean(stage3_mious) - np.mean(stage1_mious)),
        "num_test_images": sample_idx,
    }

    return summary, per_image_rows


def save_csvs(summary, per_image_rows, output_dir: Path):
    """Save summary and per-image metrics."""
    summary_path = output_dir / "metrics_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    per_image_path = output_dir / "per_image_metrics.csv"
    with open(per_image_path, "w", newline="") as f:
        fieldnames = list(per_image_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(f"[✓] Saved summary metrics: {summary_path}")
    print(f"[✓] Saved per-image metrics: {per_image_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", required=True, help="Path to Stage 1 best_model.pth")
    parser.add_argument("--stage3", required=True, help="Path to Stage 3 best_model.pth")
    parser.add_argument("--output-dir", default="outputs/evaluation_dice")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-vis", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stage 1 vs Stage 3 Evaluation + Visualisation")
    print(f"Device       : {DEVICE}")
    print(f"Stage 1 model: {args.stage1}")
    print(f"Stage 3 model: {args.stage3}")
    print(f"Output dir   : {output_dir}")
    print("=" * 70)

    stage1_model = load_model(Path(args.stage1))
    stage3_model = load_model(Path(args.stage3))

    test_dataset = PlantSegDataset(split="test", transform=get_val_transforms())
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    summary, per_image_rows = evaluate_models(
        stage1_model=stage1_model,
        stage3_model=stage3_model,
        loader=test_loader,
        output_dir=output_dir,
        n_vis=args.n_vis,
    )

    save_csvs(summary, per_image_rows, output_dir)

    print("\nFinal summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\n[✓] Visual panels saved to: {output_dir / 'visual_panels'}")


if __name__ == "__main__":
    main()