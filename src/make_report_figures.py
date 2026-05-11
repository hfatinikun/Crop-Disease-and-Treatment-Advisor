import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_training_log(log_path, output_dir, label):
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)

    # Loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and validation loss: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{label}_loss_curve.png", dpi=200)
    plt.close()

    # mIoU curve
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_miou"], label="Train mIoU")
    plt.plot(df["epoch"], df["val_miou"], label="Val mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title(f"Training and validation mIoU: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{label}_miou_curve.png", dpi=200)
    plt.close()

    summary = {
        "run": label,
        "best_val_miou": df["val_miou"].max(),
        "best_epoch": int(df.loc[df["val_miou"].idxmax(), "epoch"]),
        "final_train_loss": df["train_loss"].iloc[-1],
        "final_val_loss": df["val_loss"].iloc[-1],
        "final_train_miou": df["train_miou"].iloc[-1],
        "final_val_miou": df["val_miou"].iloc[-1],
        "epochs_completed": len(df),
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-log", required=True)
    parser.add_argument("--normal-log", required=True)
    parser.add_argument("--dice-log", required=False)
    parser.add_argument("--output-dir", default="outputs/report_figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    summaries.append(
        plot_training_log(args.stage1_log, output_dir, "stage1_baseline")
    )

    summaries.append(
        plot_training_log(args.normal_log, output_dir, "stage3_normal")
    )

    if args.dice_log:
        summaries.append(
            plot_training_log(args.dice_log, output_dir, "stage3_dice")
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "training_summary_table.csv", index=False)

    print("\nSaved figures to:", output_dir)
    print(summary_df)


if __name__ == "__main__":
    main()