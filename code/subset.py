#!/usr/bin/env python3
"""
Create a subset of a YOLO dataset by copying (or symlinking) N image+label pairs.

Input supports:
  SRC/images/train + SRC/labels/train  (with --split train)
  SRC/images + SRC/labels              (no --split)

Output mirrors the same structure under DST:
  DST/images/<split>/... and DST/labels/<split>/...

By default:
- selects randomly (set --mode first to take first N)
- requires a matching .txt label for each image (use --allow-missing-labels to keep images without labels)
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def has_any_images(folder: Path) -> bool:
    if not folder.exists():
        return False
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def resolve_dirs(
    src_root: Path,
    images_subdir: str,
    labels_subdir: str,
    split: str | None,
) -> Tuple[Path, Path, str | None]:
    """
    Resolve images_dir and labels_dir. If split is None but images/train exists and
    images/ has no images, auto-assume split=train.
    """
    if split is None:
        cand_images = src_root / images_subdir
        cand_labels = src_root / labels_subdir

        # Auto-detect typical YOLO structure: images/train, labels/train
        if (cand_images / "train").is_dir() and (cand_labels / "train").is_dir():
            if (not has_any_images(cand_images)) and has_any_images(cand_images / "train"):
                split = "train"

    images_dir = src_root / images_subdir / split if split else src_root / images_subdir
    labels_dir = src_root / labels_subdir / split if split else src_root / labels_subdir
    return images_dir, labels_dir, split


def iter_image_files(images_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p
    else:
        for p in images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def paired_label_path(images_dir: Path, labels_dir: Path, img_path: Path) -> Path:
    """
    Map an image path to its label path using the standard YOLO mirroring rule:
      labels_dir / (relative_path_from_images_dir).with_suffix(".txt")
    """
    rel = img_path.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, symlink: bool, overwrite: bool) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if symlink:
        # Relative symlink is nicer if dst is moved with its dataset folder
        try:
            rel_src = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel_src)
        except Exception:
            # Fallback to absolute symlink if relpath fails
            dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="Source dataset root folder")
    ap.add_argument("--dst", required=True, type=Path, help="Destination subset dataset root folder")
    ap.add_argument("--num", type=int, default=20000, help="Number of samples to copy (default: 20000)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--mode", choices=["random", "first"], default="random",
                    help="Selection mode: random sample or first N (default: random)")

    ap.add_argument("--split", type=str, default=None,
                    help="Optional split folder name, e.g. train or val. "
                         "If omitted and images/train exists, it may auto-use split=train.")

    ap.add_argument("--images-subdir", type=str, default="images", help="Images subdir name (default: images)")
    ap.add_argument("--labels-subdir", type=str, default="labels", help="Labels subdir name (default: labels)")

    ap.add_argument("--recursive", action="store_true", help="Search images recursively under images dir")
    ap.add_argument("--allow-missing-labels", action="store_true",
                    help="Include images even if label .txt is missing (creates empty .txt)")
    ap.add_argument("--symlink", action="store_true", help="Symlink files instead of copying (fast, saves space)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    ap.add_argument("--manifest", type=str, default="manifest.txt",
                    help="Write a manifest file listing selected items (default: manifest.txt)")
    args = ap.parse_args()

    src_root: Path = args.src
    dst_root: Path = args.dst

    images_dir, labels_dir, resolved_split = resolve_dirs(
        src_root, args.images_subdir, args.labels_subdir, args.split
    )

    if not images_dir.exists():
        raise SystemExit(f"[ERROR] Images dir not found: {images_dir}")
    if not labels_dir.exists() and not args.allow_missing_labels:
        raise SystemExit(f"[ERROR] Labels dir not found: {labels_dir} (use --allow-missing-labels to ignore)")

    # Destination dirs
    dst_images_dir = dst_root / args.images_subdir / resolved_split if resolved_split else dst_root / args.images_subdir
    dst_labels_dir = dst_root / args.labels_subdir / resolved_split if resolved_split else dst_root / args.labels_subdir
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    # Build pairs
    pairs: List[Tuple[Path, Path]] = []
    missing_labels = 0

    for img in iter_image_files(images_dir, recursive=args.recursive):
        lbl = paired_label_path(images_dir, labels_dir, img)

        if lbl.exists():
            pairs.append((img, lbl))
        else:
            if args.allow_missing_labels:
                pairs.append((img, lbl))  # we'll create empty label file later
                missing_labels += 1

    if not pairs:
        raise SystemExit("[ERROR] No images found (or no image+label pairs). Check paths/split.")

    if args.num <= 0:
        raise SystemExit("[ERROR] --num must be > 0")

    if args.num > len(pairs):
        raise SystemExit(f"[ERROR] Requested {args.num} samples, but only {len(pairs)} available.")

    # Select
    if args.mode == "first":
        selected = pairs[: args.num]
    else:
        rng = random.Random(args.seed)
        selected = rng.sample(pairs, args.num)

    # Copy/link
    manifest_lines: List[str] = []
    created_empty = 0

    for img_src, lbl_src in selected:
        rel = img_src.relative_to(images_dir)
        img_dst = dst_images_dir / rel
        lbl_dst = (dst_labels_dir / rel).with_suffix(".txt")

        link_or_copy(img_src, img_dst, symlink=args.symlink, overwrite=args.overwrite)

        if lbl_src.exists():
            link_or_copy(lbl_src, lbl_dst, symlink=args.symlink, overwrite=args.overwrite)
        else:
            # create empty label file
            ensure_parent(lbl_dst)
            if args.overwrite or not lbl_dst.exists():
                lbl_dst.write_text("", encoding="utf-8")
            created_empty += 1

        manifest_lines.append(str(rel))

    # Write manifest
    if args.manifest:
        manifest_path = dst_root / args.manifest
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print("✅ Subset created")
    print(f"Source images: {images_dir}")
    print(f"Source labels: {labels_dir}")
    print(f"Destination:   {dst_root}")
    print(f"Split:         {resolved_split if resolved_split else '(none)'}")
    print(f"Selected:      {len(selected)}")
    if missing_labels:
        print(f"Missing labels encountered (allowed): {missing_labels}")
    if created_empty:
        print(f"Empty label files created: {created_empty}")
    print(f"Manifest:      {dst_root / args.manifest}")


if __name__ == "__main__":
    main()
