"""
filter_datasets.py
==================
Collects tomato images from three datasets, deduplicates, resizes,
and writes a unified train/val/test split ready for ImageFolder.

Dataset structures expected
----------------------------
PlantVillage  (already extracted):
    data/PlantVillage/
        Tomato_Bacterial_spot/   ← class folder, images inside
        Tomato_Early_blight/
        ...

PlantDoc  (already extracted):
    data/PlantDoc/
        train/
            'Tomato leaf bacterial spot'/   ← class folder with spaces
            'Tomato leaf late blight'/
            'Tomato leaf'/                  ← = healthy
            ...
        test/                               ← same structure; we re-split ourselves

PlantSeg  (already extracted):
    data/plantseg/
        images/
            train/  val/  test/
                tomato_early_blight_001.jpg ← class encoded in filename stem
        annotations/
            train/  val/  test/
                tomato_early_blight_001.png ← matching segmentation mask

Output
------
data/processed/
    train/  val/  test/
        Tomato_Bacterial_spot/
        Tomato_Early_blight/
        ...
        Tomato_healthy/
    masks/
        train/  val/  test/
            Tomato_Early_blight/
                00000_plantseg.png
                ...
    dataset_stats.json

Usage
-----
python filter_datasets.py \
    --plantvillage data/PlantVillage \
    --plantdoc     data/PlantDoc \
    --plantseg     data/plantseg \
    --output       data/processed \
    --size         256 \
    --split        0.70 0.15 0.15 \
    --seed         42

Omit any source flag to skip that dataset, e.g. --plantdoc is optional.
"""

import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── canonical class labels ─────────────────────────────────────────────────────
# These are the 10 output classes. Every source maps into this set.
CANONICAL_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_spot",
    "Tomato_Yellow_leaf_curl",
    "Tomato_Mosaic_virus",
    "Tomato_healthy",
]

# ── PlantVillage: exact folder name → canonical ────────────────────────────────
PV_FOLDER_MAP: dict[str, str] = {
    "Tomato_Bacterial_spot":                       "Tomato_Bacterial_spot",
    "Tomato_Early_blight":                         "Tomato_Early_blight",
    "Tomato_Late_blight":                          "Tomato_Late_blight",
    "Tomato_Leaf_Mold":                            "Tomato_Leaf_mold",
    "Tomato_Septoria_leaf_spot":                   "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato_Spider_mites",
    "Tomato__Target_Spot":                         "Tomato_Target_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       "Tomato_Yellow_leaf_curl",
    "Tomato__Tomato_mosaic_virus":                 "Tomato_Mosaic_virus",
    "Tomato_healthy":                              "Tomato_healthy",
}

# ── PlantDoc: exact folder name → canonical ────────────────────────────────────
# "Tomato leaf" (no disease) = disease-free leaf = healthy
PD_FOLDER_MAP: dict[str, str] = {
    "Tomato leaf bacterial spot":          "Tomato_Bacterial_spot",
    "Tomato Early blight leaf":            "Tomato_Early_blight",
    "Tomato leaf late blight":             "Tomato_Late_blight",
    "Tomato mold leaf":                    "Tomato_Leaf_mold",
    "Tomato Septoria leaf spot":           "Tomato_Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf":"Tomato_Spider_mites",
    # Target spot not present in PlantDoc listing — omit; will be PV/PS only
    "Tomato leaf yellow virus":            "Tomato_Yellow_leaf_curl",
    "Tomato leaf mosaic virus":            "Tomato_Mosaic_virus",
    "Tomato leaf":                         "Tomato_healthy",   # disease-free
}

# ── PlantSeg: filename stem keyword → canonical ────────────────────────────────
# Stems look like: tomato_early_blight_001, tomato_healthy_042, etc.
PS_STEM_MAP: list[tuple[str, str]] = [
    ("bacterial_spot",  "Tomato_Bacterial_spot"),
    ("early_blight",    "Tomato_Early_blight"),
    ("late_blight",     "Tomato_Late_blight"),
    ("leaf_mold",       "Tomato_Leaf_mold"),
    ("septoria",        "Tomato_Septoria_leaf_spot"),
    ("spider_mite",     "Tomato_Spider_mites"),
    ("target_spot",     "Tomato_Target_spot"),
    ("yellow",          "Tomato_Yellow_leaf_curl"),
    ("mosaic",          "Tomato_Mosaic_virus"),
    ("healthy",         "Tomato_healthy"),
]

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VALID_MASK_EXT = {".png", ".tif", ".tiff"}


# ── helpers ────────────────────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def stem_to_class(stem: str) -> str | None:
    """Infer canonical class from a PlantSeg filename stem."""
    stem_lower = stem.lower()
    if "tomato" not in stem_lower:
        return None
    for keyword, canonical in PS_STEM_MAP:
        if keyword in stem_lower:
            return canonical
    return None


def resize_and_save(src: Path, dst: Path, size: int) -> bool:
    img = cv2.imread(str(src))
    if img is None:
        log.warning("Unreadable, skipped: %s", src.name)
        return False
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)
    return True


def copy_mask(src: Path, dst: Path) -> None:
    """Copy mask as-is (no resize — masks are binary/categorical)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(src, dst)


def stratified_split(
    files: list[Path],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    files = list(files)
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return (
        files[:n_train],
        files[n_train : n_train + n_val],
        files[n_train + n_val :],
    )


# ── per-source collectors ──────────────────────────────────────────────────────

def collect_plantvillage(root: Path) -> dict[str, list[Path]]:
    log.info("[PlantVillage] Scanning %s", root)
    collected: dict[str, list[Path]] = defaultdict(list)
    skipped = 0
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        canonical = PV_FOLDER_MAP.get(folder.name)
        if canonical is None:
            skipped += 1
            continue
        imgs = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in VALID_IMG_EXT]
        collected[canonical].extend(imgs)
        log.info("  %-50s %4d images", folder.name, len(imgs))
    log.info("  Skipped %d non-tomato folders.", skipped)
    return dict(collected)


def collect_plantdoc(root: Path) -> dict[str, list[Path]]:
    log.info("[PlantDoc] Scanning %s", root)
    collected: dict[str, list[Path]] = defaultdict(list)
    # PlantDoc has pre-split train/ and test/ — we pool them and re-split
    for split_dir in ["train", "test"]:
        split_path = root / split_dir
        if not split_path.exists():
            log.warning("  %s not found, skipping.", split_path)
            continue
        for folder in sorted(split_path.iterdir()):
            if not folder.is_dir():
                continue
            canonical = PD_FOLDER_MAP.get(folder.name)
            if canonical is None:
                continue  # silently skip non-tomato (apple, corn, etc.)
            imgs = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in VALID_IMG_EXT]
            collected[canonical].extend(imgs)

    for cls, imgs in sorted(collected.items()):
        log.info("  %-40s %4d images", cls, len(imgs))
    return dict(collected)


def collect_plantseg(
    root: Path,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """
    Returns (images_by_class, masks_by_class).
    Masks are optional — if the annotations folder is missing we return {}.
    """
    log.info("[PlantSeg] Scanning %s", root)
    img_by_class:  dict[str, list[Path]] = defaultdict(list)
    mask_by_class: dict[str, list[Path]] = defaultdict(list)

    images_root = root / "images"
    annots_root = root / "annotations"

    if not images_root.exists():
        log.error("  images/ folder not found under %s. Check --plantseg path.", root)
        return {}, {}

    # Walk train/val/test sub-splits
    for split_dir in ["train", "val", "test"]:
        img_split = images_root / split_dir
        ann_split = annots_root / split_dir if annots_root.exists() else None

        if not img_split.exists():
            continue

        for img_path in sorted(img_split.iterdir()):
            if img_path.suffix.lower() not in VALID_IMG_EXT:
                continue
            canonical = stem_to_class(img_path.stem)
            if canonical is None:
                continue
            img_by_class[canonical].append(img_path)

            # Match mask by same stem (may be .png)
            if ann_split:
                for ext in VALID_MASK_EXT:
                    mask_path = ann_split / (img_path.stem + ext)
                    if mask_path.exists():
                        mask_by_class[canonical].append(mask_path)
                        break

    for cls, imgs in sorted(img_by_class.items()):
        n_masks = len(mask_by_class.get(cls, []))
        log.info("  %-40s %4d images, %3d masks", cls, len(imgs), n_masks)

    return dict(img_by_class), dict(mask_by_class)


# ── deduplication ──────────────────────────────────────────────────────────────

def deduplicate(
    by_class: dict[str, list[Path]],
) -> dict[str, list[Path]]:
    """Remove byte-identical duplicates globally across all classes."""
    seen: set[str] = set()
    clean: dict[str, list[Path]] = {}
    total_removed = 0
    for label, files in by_class.items():
        unique = []
        for f in files:
            h = file_hash(f)
            if h not in seen:
                seen.add(h)
                unique.append(f)
            else:
                total_removed += 1
        clean[label] = unique
    log.info("Deduplication: removed %d duplicate images across all sources.", total_removed)
    return clean


# ── merge ──────────────────────────────────────────────────────────────────────

def merge(
    *sources: dict[str, list[Path]],
) -> dict[str, list[Path]]:
    merged: dict[str, list[Path]] = defaultdict(list)
    for src in sources:
        for label, files in src.items():
            merged[label].extend(files)
    return dict(merged)


# ── reporting ──────────────────────────────────────────────────────────────────

def print_distribution(by_class: dict[str, list[Path]], title: str) -> None:
    total = sum(len(v) for v in by_class.values())
    max_n = max((len(v) for v in by_class.values()), default=1)
    log.info("")
    log.info("── %s (%d total) ─────────────────────────", title, total)
    for cls in CANONICAL_CLASSES:
        imgs = by_class.get(cls, [])
        bar = "█" * (len(imgs) * 28 // max_n) if max_n else ""
        log.info("  %-38s %5d  %s", cls, len(imgs), bar)
    log.info("")


# ── main pipeline ──────────────────────────────────────────────────────────────

def build_dataset(
    pv_root:  Path | None,
    pd_root:  Path | None,
    ps_root:  Path | None,
    output:   Path,
    img_size: int,
    ratios:   tuple[float, float, float],
    seed:     int,
) -> None:

    # 1. Collect from each source
    sources_img  = []
    sources_mask: dict[str, list[Path]] = {}

    if pv_root:
        sources_img.append(collect_plantvillage(pv_root))
    if pd_root:
        sources_img.append(collect_plantdoc(pd_root))
    if ps_root:
        ps_imgs, ps_masks = collect_plantseg(ps_root)
        sources_img.append(ps_imgs)
        sources_mask = ps_masks  # masks only come from PlantSeg

    if not sources_img:
        log.error("No source datasets provided. Pass at least one of --plantvillage / --plantdoc / --plantseg.")
        return

    # 2. Merge all image sources
    all_imgs = merge(*sources_img)
    print_distribution(all_imgs, "Before deduplication")

    # 3. Deduplicate
    all_imgs = deduplicate(all_imgs)
    print_distribution(all_imgs, "After deduplication")

    # 4. Warn about class imbalance
    counts = {c: len(all_imgs.get(c, [])) for c in CANONICAL_CLASSES}
    max_c, min_c = max(counts.values()), min(v for v in counts.values() if v > 0)
    if max_c / min_c > 5:
        log.warning(
            "Imbalance ratio %.1fx (max=%d, min=%d). "
            "Consider weighted loss — see dataset_stats.json for per-class counts.",
            max_c / min_c, max_c, min_c,
        )

    # 5. Split, resize, and write
    splits = ("train", "val", "test")
    stats: dict = {
        "img_size": img_size,
        "sources": {
            "plantvillage": pv_root is not None,
            "plantdoc":     pd_root is not None,
            "plantseg":     ps_root is not None,
        },
        "split_ratios": dict(zip(splits, ratios)),
        "classes": {},
        "totals": {s: 0 for s in splits},
    }

    # Build a stem→mask lookup for fast matching
    mask_by_stem: dict[str, Path] = {}
    for masks in sources_mask.values():
        for m in masks:
            mask_by_stem[m.stem] = m

    for cls in tqdm(CANONICAL_CLASSES, desc="Processing classes"):
        imgs = all_imgs.get(cls, [])
        if not imgs:
            log.warning("No images found for class: %s", cls)
            stats["classes"][cls] = {s: 0 for s in splits}
            continue

        train_f, val_f, test_f = stratified_split(imgs, ratios, seed)
        split_map = {"train": train_f, "val": val_f, "test": test_f}
        stats["classes"][cls] = {s: len(f) for s, f in split_map.items()}

        for split_name, split_imgs in split_map.items():
            dst_dir = output / split_name / cls
            for i, src in enumerate(tqdm(split_imgs, desc=f"  {split_name}", leave=False)):
                dst = dst_dir / f"{i:05d}.jpg"
                if not dst.exists() and resize_and_save(src, dst, img_size):
                    stats["totals"][split_name] += 1
                elif dst.exists():
                    stats["totals"][split_name] += 1

                # Copy PlantSeg mask if available (train split only)
                if split_name == "train" and src.stem in mask_by_stem:
                    mask_src = mask_by_stem[src.stem]
                    mask_dst = output / "masks" / "train" / cls / f"{i:05d}_mask{mask_src.suffix}"
                    if not mask_dst.exists():
                        copy_mask(mask_src, mask_dst)

    # 6. Save stats
    stats_path = output / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("── Done ─────────────────────────────────────────────────────")
    log.info("  Output  : %s", output.resolve())
    log.info("  train   : %d images", stats["totals"]["train"])
    log.info("  val     : %d images", stats["totals"]["val"])
    log.info("  test    : %d images", stats["totals"]["test"])
    log.info("  Stats   : %s", stats_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter PlantVillage + PlantDoc + PlantSeg to tomato classes, resize, and split."
    )
    p.add_argument("--plantvillage", type=Path, default=None,
                   help="Path to data/PlantVillage (optional)")
    p.add_argument("--plantdoc",     type=Path, default=None,
                   help="Path to data/PlantDoc (optional)")
    p.add_argument("--plantseg",     type=Path, default=None,
                   help="Path to data/plantseg (optional)")
    p.add_argument("--output",  type=Path, default=Path("data/processed"))
    p.add_argument("--size",    type=int,  default=256)
    p.add_argument("--split",   type=float, nargs=3, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed",    type=int,  default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratios = tuple(args.split)
    if abs(sum(ratios) - 1.0) > 1e-6:
        log.error("Split ratios must sum to 1.0 (got %.4f)", sum(ratios))
        return

    if not any([args.plantvillage, args.plantdoc, args.plantseg]):
        log.error("Provide at least one source: --plantvillage, --plantdoc, or --plantseg.")
        return

    build_dataset(
        pv_root  = args.plantvillage,
        pd_root  = args.plantdoc,
        ps_root  = args.plantseg,
        output   = args.output,
        img_size = args.size,
        ratios   = ratios,
        seed     = args.seed,
    )


if __name__ == "__main__":
    main()