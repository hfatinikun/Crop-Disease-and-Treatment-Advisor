"""
filter_datasets.py

Filters tomato-related images from PlantSeg, PlantDoc, and PlantVillage into a unified folder structure and builds a master metadata CSV.

Expected raw dataset layout (set via ROOT_* constants below):
    RAW_PLANTSEG/ *plantseg*
        images/          <- all plant images
        annotations/     <- grayscale PNG masks (same filename as image)
        Metadata.csv     <- columns include: image_filename, plant_host, split, disease_class
        *Name,Index,Plant,Disease,Resolution,Label file,Mask ratio,URL,License,Split*
    RAW_PLANTDOC/ *PlantDoc-Dataset-master*
        train/
            Tomato Early blight leaf/
            Tomato Septoria leaf spot/
            ...
        test/
            ...
    RAW_PLANTVILLAGE/ *PlantVillage
        Tomato_Early_blight/
        Tomato_healthy/
        ...

Output layout:
    data/tomato/
        plantseg/
            images/{train,val,test}/
            annotations/{train,val,test}/   <- real masks (grayscale PNGs)
        plantdoc/
            images/{train,val,test}/
        plantvillage/
            images/{train,val,test}/
        metadata.csv
"""

import os
import shutil
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config — change these paths to match your file path setup -> will change to more relative later
# ---------------------------------------------------------------------------
RAW_PLANTSEG    = Path("/data/users/aikiror/deepLearning/project/plantseg")
RAW_PLANTDOC    = Path("/data/users/aikiror/deepLearning/project/PlantDoc-Dataset-master")
RAW_PLANTVILLAGE = Path("/data/users/aikiror/deepLearning/project/PlantVillage")

#saving the output dataset outside of where git is - in case it overloads the git memory
OUTPUT_ROOT = Path("./tomato")

#set random seed for replication down the line
RANDOM_SEED = 42

#
PLANTDOC_VAL_FRACTION = 0.10   #fraction of train images held out for val
PLANTVILLAGE_SPLITS = (0.70, 0.10, 0.20)  # train / val / test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_dirs():
    """Create the full output directory tree."""
    for dataset in ("plantseg", "plantdoc", "plantvillage"):
        for split in ("train", "val", "test"):
            (OUTPUT_ROOT / dataset / "images" / split).mkdir(parents=True, exist_ok=True)
            if dataset == "plantseg":
                (OUTPUT_ROOT / dataset / "annotations" / split).mkdir(parents=True, exist_ok=True)
    print(f"Output directory tree created under '{OUTPUT_ROOT}/'")


def safe_copy(src: Path, dst: Path):
    """Copy source to destination, skipping if source does not exist."""
    if not src.exists():
        print(f"  [WARN] Missing file, skipped: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


# ---------------------------------------------------------------------------
# PlantSeg
# ---------------------------------------------------------------------------

def process_plantseg() -> list[dict]:
    """
    Filter Metadata.csv for tomato images, copy images + annotation masks,
    respect the existing train/val/test split column.
    Returns a list of metadata dicts.
    """
    meta_path = RAW_PLANTSEG / "Metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"PlantSeg Metadata.csv not found at {meta_path}")

    df = pd.read_csv(meta_path)

    print("Columns:", df.columns.tolist())
    print(df.head(2))

    # Normalise column names first
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Now use normalised names everywhere
    tomato_df = df[df["plant"].str.lower() == "tomato"].copy()

    records =[]
    for _, row in tomato_df.iterrows():
        filename  = row["name"]
        split     = row["split"].lower().replace("training", "train").replace("validation", "val").replace("testing", "test")
        disease   = row.get("disease", "unknown")
        mask_file = row["label_file"]  # was "Label file", now "label_file"

        src_img  = RAW_PLANTSEG / "images" / split / filename
        src_mask = RAW_PLANTSEG / "annotations" / split / mask_file  # same name, .png

        # Destination
        dst_img  = OUTPUT_ROOT / "plantseg" / "images" / split / filename
        dst_mask = OUTPUT_ROOT / "plantseg" / "annotations" / split / mask_file

        img_ok  = safe_copy(src_img,  dst_img)
        mask_ok = safe_copy(src_mask, dst_mask)

        records.append({
            "image_path":    str(dst_img.relative_to(OUTPUT_ROOT)) if img_ok else "",
            "source":        "plantseg",
            "disease_class": str(disease),
            "split":         split,
            "has_real_mask": True,
            "mask_path":     str(dst_mask.relative_to(OUTPUT_ROOT)) if mask_ok else "",
        })

    print(f"[PlantSeg] Done. {len(records)} records processed.")
    return records


# ---------------------------------------------------------------------------
# PlantDoc
# ---------------------------------------------------------------------------

def process_plantdoc() -> list[dict]:
    """
    Grab all Tomato* folders from train/ and test/.
    Carve a val set by holding back PLANTDOC_VAL_FRACTION of each
    train class's images (stratified by class folder).
    Returns metadata dicts.
    """
    records = []

    for official_split in ("train", "test"):
        split_dir = RAW_PLANTDOC / official_split
        if not split_dir.exists():
            print(f"  [WARN] PlantDoc {official_split}/ not found, skipping.")
            continue

        tomato_folders = sorted(
            f for f in split_dir.iterdir()
            if f.is_dir() and f.name.startswith("Tomato")
        )

        for class_folder in tomato_folders:
            disease_class = class_folder.name
            images = sorted([
                p for p in class_folder.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            ])

            if official_split == "train" and len(images) > 0:
                # Carve val out of train
                n_val = max(1, int(len(images) * PLANTDOC_VAL_FRACTION))
                random.seed(RANDOM_SEED)
                random.shuffle(images)
                val_images   = images[:n_val]
                train_images = images[n_val:]
                split_map = (
                    [(p, "train") for p in train_images] +
                    [(p, "val")   for p in val_images]
                )
            else:
                split_map = [(p, "test") for p in images]

            for src_img, split in split_map:
                dst_img = OUTPUT_ROOT / "plantdoc" / "images" / split / src_img.name
                img_ok = safe_copy(src_img, dst_img)
                records.append({
                    "image_path":    str(dst_img.relative_to(OUTPUT_ROOT)) if img_ok else "",
                    "source":        "plantdoc",
                    "disease_class": disease_class,
                    "split":         split,
                    "has_real_mask": False,
                    "mask_path":     "",
                })

    print(f"[PlantDoc] Done. {len(records)} records processed.")
    return records


# ---------------------------------------------------------------------------
# PlantVillage
# ---------------------------------------------------------------------------

def process_plantvillage() -> list[dict]:
    """
    Grab all Tomato* folders.  Apply a fresh stratified 70/10/20 split
    per class (since PlantVillage has no existing split).
    Returns metadata dicts.
    """
    if not RAW_PLANTVILLAGE.exists():
        raise FileNotFoundError(f"PlantVillage root not found at {RAW_PLANTVILLAGE}")

    tomato_folders = sorted(
        f for f in RAW_PLANTVILLAGE.iterdir()
        if f.is_dir() and f.name.startswith("Tomato")
    )
    print(f"[PlantVillage] Found {len(tomato_folders)} Tomato* folders.")

    train_frac, val_frac, test_frac = PLANTVILLAGE_SPLITS
    records = []

    for class_folder in tomato_folders:
        disease_class = class_folder.name
        images = sorted([
            p for p in class_folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ])

        if len(images) == 0:
            continue

        # Split: first train+val vs test, then train vs val
        train_val, test_imgs = train_test_split(
            images,
            test_size=test_frac,
            random_state=RANDOM_SEED,
        )
        # val fraction relative to the train_val pool
        val_frac_adjusted = val_frac / (train_frac + val_frac)
        train_imgs, val_imgs = train_test_split(
            train_val,
            test_size=val_frac_adjusted,
            random_state=RANDOM_SEED,
        )

        split_map = (
            [(p, "train") for p in train_imgs] +
            [(p, "val")   for p in val_imgs]   +
            [(p, "test")  for p in test_imgs]
        )

        for src_img, split in split_map:
            dst_img = OUTPUT_ROOT / "plantvillage" / "images" / split / src_img.name
            img_ok = safe_copy(src_img, dst_img)
            records.append({
                "image_path":    str(dst_img.relative_to(OUTPUT_ROOT)) if img_ok else "",
                "source":        "plantvillage",
                "disease_class": disease_class,
                "split":         split,
                "has_real_mask": False,
                "mask_path":     "",
            })

    print(f"[PlantVillage] Done. {len(records)} records processed.")
    return records


# ---------------------------------------------------------------------------
# Metadata CSV
# ---------------------------------------------------------------------------

def build_metadata(all_records: list[dict]):
    """Write all_records to data/tomato/metadata.csv."""
    df = pd.DataFrame(all_records, columns=[
        "image_path", "source", "disease_class",
        "split", "has_real_mask", "mask_path",
    ])
    out_path = OUTPUT_ROOT / "metadata.csv"
    df.to_csv(out_path, index=False)

    # Quick summary
    print("\n[metadata.csv] Summary")
    print(f"  Total images : {len(df)}")
    print(df.groupby(["source", "split"]).size().to_string())
    print(f"\n  Saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  Tomato Disease Dataset Organiser")
    print("=" * 60)

    make_output_dirs()

    all_records = []

    print("\n--- PlantSeg ---")
    all_records.extend(process_plantseg())

    print("\n--- PlantDoc ---")
    all_records.extend(process_plantdoc())

    print("\n--- PlantVillage ---")
    all_records.extend(process_plantvillage())

    print("\n--- Building metadata.csv ---")
    build_metadata(all_records)

    print("\n[✓] All done.")


if __name__ == "__main__":
    main()