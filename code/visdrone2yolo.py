"""
Convert VisDrone annotation format to YOLO label .txt files.

VisDrone format (per line):
  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

  score     : 0 = ignored region (skip), 1 = valid
  category  : 0 = ignored, 1-10 = valid classes, 11 = others (skip)

VisDrone categories -> YOLO class id (category - 1):
  1  pedestrian       -> 0
  2  people           -> 1
  3  bicycle          -> 2
  4  car              -> 3
  5  van              -> 4
  6  bus              -> 5
  7  tricycle         -> 6
  8  awning-tricycle  -> 7
  9  truck            -> 8
  10 motor            -> 9

YOLO output: <class> <cx> <cy> <w> <h>  (normalized)

Usage:
  python visdrone2yolo.py --ann <annotations_dir> --img <images_dir> --out <output_labels_dir>

Example:
  python visdrone2yolo.py \
    --ann "VisDrone/train/annotations" \
    --img "VisDrone/train/images" \
    --out "VisDrone/train/labels"
"""

import argparse
import cv2
import os
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = [
    "pedestrian",       # 0  (VisDrone cat 1)
    "people",           # 1  (VisDrone cat 2)
    "bicycle",          # 2  (VisDrone cat 3)
    "car",              # 3  (VisDrone cat 4)
    "van",              # 4  (VisDrone cat 5)
    "bus",              # 5  (VisDrone cat 6)
    "tricycle",         # 6  (VisDrone cat 7)
    "awning-tricycle",  # 7  (VisDrone cat 8)
    "truck",            # 8  (VisDrone cat 9)
    "motor",            # 9  (VisDrone cat 10)
]


def get_image_size(img_dir: Path, stem: str) -> tuple[int, int] | None:
    """Return (W, H) for the image matching stem, or None if not found."""
    for ext in IMG_EXTS:
        p = img_dir / (stem + ext)
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    return None


def convert(ann_dir: Path, img_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    if not ann_files:
        print(f"[Warn] No .txt files found in {ann_dir}")
        return

    no_img = 0
    total_boxes = 0
    skipped_boxes = 0

    for ann_path in ann_files:
        stem = ann_path.stem
        size = get_image_size(img_dir, stem)

        if size is None:
            print(f"[Warn] Image not found for {stem}, skipping.")
            no_img += 1
            continue

        W, H = size
        lines_out = []

        for raw in ann_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue

            parts = raw.split(",")
            if len(parts) < 6:
                continue

            x, y, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score    = int(parts[4])
            category = int(parts[5])

            # Skip ignored regions and invalid/other categories
            if score == 0 or category == 0 or category == 11:
                skipped_boxes += 1
                continue

            cls = category - 1  # map 1-10 -> 0-9

            cx = (x + bw / 2) / W
            cy = (y + bh / 2) / H
            nw = bw / W
            nh = bh / H

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            total_boxes += 1

        (out_dir / f"{stem}.txt").write_text(
            "\n".join(lines_out) + ("\n" if lines_out else ""),
            encoding="utf-8"
        )

    converted = len(ann_files) - no_img
    print(f"Done. {converted}/{len(ann_files)} label files written to: {out_dir}")
    if no_img:
        print(f"  Skipped (no image found): {no_img}")
    print(f"  Valid boxes  : {total_boxes:,}")
    print(f"  Skipped boxes: {skipped_boxes:,}  (score=0 / cat=0 / cat=11)")

    # Write classes.txt
    classes_file = out_dir / "classes.txt"
    classes_file.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    print(f"classes.txt written -> {classes_file}")


def main():
    ap = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO label files.")
    ap.add_argument("--ann", required=True, help="Input directory of VisDrone .txt annotation files")
    ap.add_argument("--img", required=True, help="Directory of corresponding images (for W/H)")
    ap.add_argument("--out", required=True, help="Output directory for YOLO .txt label files")
    args = ap.parse_args()

    convert(
        ann_dir = Path(args.ann),
        img_dir = Path(args.img),
        out_dir = Path(args.out),
    )


if __name__ == "__main__":
    main()
