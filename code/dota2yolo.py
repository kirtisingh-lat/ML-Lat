"""
Convert DOTA format labels to standard YOLO axis-aligned bbox format.

DOTA format (per line):
  x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
  (absolute pixel OBB corners; first 2 lines may be metadata headers)

Output YOLO format:
  <class_id> <cx> <cy> <w> <h>  (normalized to image dimensions)

OBB -> AABB: take min/max of the 4 corner coordinates.

DOTA class -> master mapping:
  vehicle (1): plane, ship, small-vehicle, large-vehicle, helicopter
  ignore (-1): everything else (courts, fields, storage-tank, harbor, etc.)

Usage:
  python dota2yolo.py --dota <labels_dir> --images <images_dir> --out <output_labels_dir>

  # All splits:
  python dota2yolo.py --dota dota/train/labels --images dota/train/images --out dota/train/labels_yolo
  python dota2yolo.py --dota dota/val/labels   --images dota/val/images   --out dota/val/labels_yolo
"""

import argparse
import cv2
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# DOTA class name -> YOLO master class id (0=person, 1=vehicle, -1=ignore)
DOTA_MAPPING = {
    "plane":            1,
    "ship":             1,
    "small-vehicle":    1,
    "large-vehicle":    1,
    "helicopter":       1,
    "baseball-diamond": -1,
    "basketball-court": -1,
    "bridge":           -1,
    "container-crane":  -1,
    "ground-track-field": -1,
    "harbor":           -1,
    "roundabout":       -1,
    "soccer-ball-field":-1,
    "storage-tank":     -1,
    "swimming-pool":    -1,
    "tennis-court":     -1,
    "airport":          -1,
    "helipad":          -1,
}

CLASSES = ["person", "vehicle"]


def get_image_size(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = images_dir / (stem + ext)
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    return None


def convert(dota_dir: Path, images_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(dota_dir.glob("*.txt"))
    if not label_files:
        print(f"[Warn] No .txt files in {dota_dir}")
        return

    written = skipped_no_img = skipped_boxes = total_boxes = 0

    for lbl_path in label_files:
        stem = lbl_path.stem
        size = get_image_size(images_dir, stem)
        if size is None:
            skipped_no_img += 1
            continue

        W, H = size
        lines_out = []

        for raw in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = raw.strip()
            # Skip DOTA metadata header lines
            if raw.startswith("imagesource") or raw.startswith("gsd"):
                continue
            parts = raw.split()
            if len(parts) < 9:
                continue

            try:
                coords = list(map(float, parts[:8]))
                cls_name = parts[8].lower()
            except ValueError:
                continue

            master = DOTA_MAPPING.get(cls_name, -1)
            if master == -1:
                skipped_boxes += 1
                continue

            xs = coords[0::2]  # x1,x2,x3,x4
            ys = coords[1::2]  # y1,y2,y3,y4

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            cx = max(0.0, min(1.0, ((xmin + xmax) / 2) / W))
            cy = max(0.0, min(1.0, ((ymin + ymax) / 2) / H))
            nw = max(0.0, min(1.0, (xmax - xmin) / W))
            nh = max(0.0, min(1.0, (ymax - ymin) / H))

            lines_out.append(f"{master} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            total_boxes += 1

        (out_dir / f"{stem}.txt").write_text(
            "\n".join(lines_out) + ("\n" if lines_out else ""),
            encoding="utf-8"
        )
        written += 1

    # Write classes.txt so unified_dataset.py can discover classes
    (out_dir / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    print(f"Done. {written}/{len(label_files)} label files -> {out_dir}")
    if skipped_no_img:
        print(f"  Skipped (no image): {skipped_no_img}")
    print(f"  Valid boxes  : {total_boxes:,}")
    print(f"  Ignored boxes: {skipped_boxes:,}  (non-vehicle/person classes)")


def main():
    ap = argparse.ArgumentParser(description="Convert DOTA labels to YOLO axis-aligned bbox format.")
    ap.add_argument("--dota",   required=True, help="Directory of DOTA .txt label files")
    ap.add_argument("--images", required=True, help="Directory of corresponding images")
    ap.add_argument("--out",    required=True, help="Output directory for YOLO .txt files")
    args = ap.parse_args()

    convert(Path(args.dota), Path(args.images), Path(args.out))


if __name__ == "__main__":
    main()
