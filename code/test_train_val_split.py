"""
Train/Val/Test Splitter
=======================
Splits a flat YOLO dataset (images + labels) into train/val/test directories
and generates a data.yaml file.

Input structure (any of):
  dataset/
    images/   ← image files
    labels/   ← matching .txt label files

Output structure:
  output/
    train/images/  train/labels/
    val/images/    val/labels/
    test/images/   test/labels/
    data.yaml

Usage:
  python test_train_val_split.py --src /path/to/dataset --out /path/to/output
  python test_train_val_split.py --src /path/to/dataset --out /path/to/output --classes person vehicle
  python test_train_val_split.py --src /path/to/dataset --out /path/to/output --workers 16

Requirements:
  pip install pyyaml tqdm
"""

import argparse
import logging
import os
import random
import shutil
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# ── constants ─────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ── logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("splitter")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

# ── discovery ─────────────────────────────────────────────────────────────────

def find_images_dir(src: Path, logger: logging.Logger) -> Path | None:
    for name in ("images", "image", "imgs", "data"):
        candidate = src / name
        if candidate.is_dir():
            logger.debug(f"Found images dir: {candidate}")
            return candidate
    has_images = any(
        e.is_file() and os.path.splitext(e.name)[1].lower() in IMG_EXTS
        for e in os.scandir(src)
    )
    if has_images:
        logger.debug(f"Using src itself as images dir: {src}")
        return src
    return None


def find_labels_dir(images_dir: Path, src: Path, logger: logging.Logger) -> Path | None:
    for name in ("labels", "labels_yolo", "label", "annotations"):
        for base in (images_dir.parent, src):
            candidate = base / name
            if candidate.is_dir():
                logger.debug(f"Found labels dir: {candidate}")
                return candidate
    return None


def _scan_images(images_dir: Path) -> list[Path]:
    """Fast image scan using os.scandir; falls back to rglob for nested dirs."""
    results = []
    with os.scandir(images_dir) as it:
        for entry in it:
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMG_EXTS:
                results.append(Path(entry.path))
            elif entry.is_dir():
                # nested — use rglob only when necessary
                results.extend(
                    p for p in Path(entry.path).rglob("*")
                    if p.is_file() and p.suffix.lower() in IMG_EXTS
                )
    return results


def collect_pairs(
    images_dir: Path,
    labels_dir: Path | None,
    logger: logging.Logger,
) -> list[tuple[Path, Path | None]]:
    images = _scan_images(images_dir)

    if labels_dir is None:
        pairs = [(img, None) for img in images]
        logger.info(f"Found {len(images)} images (no labels dir)")
        return pairs

    # Build stem→path index of all labels in one pass — avoids N exists() calls
    logger.debug("Indexing label files...")
    label_index: dict[str, Path] = {}
    for root, _, files in os.walk(labels_dir):
        for fname in files:
            if fname.endswith(".txt"):
                stem = os.path.splitext(fname)[0]
                label_index[stem] = Path(root) / fname

    pairs = []
    missing = 0
    for img in images:
        lbl = label_index.get(img.stem)
        if lbl is None:
            logger.debug(f"No label for: {img.name}")
            missing += 1
        pairs.append((img, lbl))

    labeled = len(pairs) - missing
    logger.info(f"Found {len(pairs)} images ({labeled} with labels, {missing} without)")
    return pairs

# ── split ─────────────────────────────────────────────────────────────────────

def split_pairs(
    pairs: list,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )

# ── parallel copy ─────────────────────────────────────────────────────────────

def _copy_one(img_path: Path, lbl_path: Path | None, img_dst: Path, lbl_dst: Path):
    """Copy a single image+label pair. Returns (ok: bool, error_msg: str|None)."""
    try:
        shutil.copy(img_path, img_dst / img_path.name)
    except Exception as e:
        return False, f"image {img_path}: {e}"

    if lbl_path is not None:
        try:
            shutil.copy(lbl_path, lbl_dst / lbl_path.name)
        except Exception as e:
            return False, f"label {lbl_path}: {e}"

    return True, None


def copy_split(
    pairs: list[tuple[Path, Path | None]],
    split_name: str,
    out_root: Path,
    logger: logging.Logger,
    workers: int,
) -> tuple[int, int]:
    img_dst = out_root / split_name / "images"
    lbl_dst = out_root / split_name / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    ok = errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_copy_one, img, lbl, img_dst, lbl_dst): img
            for img, lbl in pairs
        }
        with tqdm(total=len(futures), desc=f"  {split_name}", unit="img", miniters=50) as bar:
            for fut in as_completed(futures):
                success, err_msg = fut.result()
                if success:
                    ok += 1
                else:
                    errors += 1
                    logger.error(f"Failed to copy {err_msg}")
                bar.update(1)

    return ok, errors

# ── yaml ──────────────────────────────────────────────────────────────────────

def write_yaml(out_root: Path, class_names: list[str], logger: logging.Logger):
    data = {
        "path":  str(out_root).replace("\\", "/"),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info(f"data.yaml written → {yaml_path}")

# ── class discovery ───────────────────────────────────────────────────────────

def discover_classes(src: Path, logger: logging.Logger) -> list[str] | None:
    for name in ("classes.txt", "obj.names", "data.yaml", "data.yml", "dataset.yaml"):
        p = src / name
        if not p.exists():
            continue
        if p.suffix in (".yaml", ".yml"):
            try:
                with open(p, encoding="utf-8") as f:
                    d = yaml.safe_load(f)
                if isinstance(d, dict) and "names" in d:
                    names = list(d["names"])
                    logger.info(f"Classes from {p.name}: {names}")
                    return names
            except Exception as e:
                logger.warning(f"Could not parse {p}: {e}")
        else:
            try:
                names = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                if names:
                    logger.info(f"Classes from {p.name}: {names}")
                    return names
            except Exception as e:
                logger.warning(f"Could not read {p}: {e}")
    return None


def _infer_classes_from_labels(pairs: list) -> list[str]:
    """Scan label files to find max class id and build placeholder names."""
    max_id = 0
    for _, lbl in pairs:
        if lbl is None:
            continue
        try:
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    max_id = max(max_id, int(parts[0]))
        except Exception:
            pass
    return [f"class{i}" for i in range(max_id + 1)]

# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test")
    p.add_argument("--src",     required=True, help="Source dataset directory")
    p.add_argument("--out",     required=True, help="Output directory")
    p.add_argument("--split",   nargs=3, type=float, default=[70, 20, 10],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Split percentages, must sum to 100 (default: 70 20 10)")
    p.add_argument("--classes", nargs="+", default=None,
                   help="Class names in order. Auto-detected from classes.txt/data.yaml if omitted.")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel copy threads (default: 8; increase for network drives)")
    p.add_argument("--copy-unlabeled", action="store_true",
                   help="Include images with no matching label file")
    return p.parse_args()


def main():
    args = parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        print(f"[ERROR] Source directory does not exist: {src}")
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out / "split.log")
    logger.info(f"Source  : {src}")
    logger.info(f"Output  : {out}")
    logger.info(f"Workers : {args.workers}")

    total_pct = sum(args.split)
    if abs(total_pct - 100) > 0.01:
        logger.error(f"Split ratios must sum to 100, got {total_pct}")
        sys.exit(1)
    ratios = tuple(r / 100 for r in args.split)

    images_dir = find_images_dir(src, logger)
    if images_dir is None:
        logger.error(f"No images directory found inside: {src}")
        sys.exit(1)

    labels_dir = find_labels_dir(images_dir, src, logger)
    if labels_dir is None:
        logger.warning("No labels directory found — images will be copied without labels")

    pairs = collect_pairs(images_dir, labels_dir, logger)

    if not args.copy_unlabeled:
        before = len(pairs)
        pairs = [(img, lbl) for img, lbl in pairs if lbl is not None]
        skipped = before - len(pairs)
        if skipped:
            logger.info(f"Skipped {skipped} unlabeled images (use --copy-unlabeled to include)")

    if not pairs:
        logger.error("No valid image-label pairs found. Exiting.")
        sys.exit(1)

    class_names = args.classes or discover_classes(src, logger)
    if class_names is None:
        logger.warning("Could not auto-detect class names — using placeholders.")
        class_names = _infer_classes_from_labels(pairs)

    logger.info(f"Classes ({len(class_names)}): {class_names}")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, ratios, args.seed)
    logger.info(f"Split → train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    total_ok = total_err = 0
    for split_name, split_list in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        ok, err = copy_split(split_list, split_name, out, logger, args.workers)
        logger.info(f"{split_name}: {ok} copied, {err} errors")
        total_ok  += ok
        total_err += err

    write_yaml(out, class_names, logger)
    logger.info(f"Done. Total copied: {total_ok}, errors: {total_err}")
    if total_err:
        logger.warning(f"Check split.log for details on {total_err} error(s)")


if __name__ == "__main__":
    main()
