"""
Convert AU-AIR annotations.json to YOLO format.

Bounding boxes in the dataset are axis-aligned (top, left, width, height),
so standard YOLO format is used: <class> <cx> <cy> <w> <h>  (all normalized).

Usage:
    python auair2yolo.py --json annotations.json --out labels/
    python auair2yolo.py --json annotations.json --out labels/ --yaml dataset.yaml --images path/to/images
"""

import argparse
import json
from pathlib import Path


def convert(json_path: Path, out_dir: Path, yaml_path: Path | None, images_dir: Path | None):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    categories: list[str] = data["categories"]
    annotations: list[dict] = data["annotations"]

    # Group bbox entries by image
    image_records: dict[str, dict] = {}
    for rec in annotations:
        name = rec["image_name"]
        if name not in image_records:
            image_records[name] = rec
        else:
            # merge bboxes (shouldn't happen, but be safe)
            image_records[name]["bbox"].extend(rec["bbox"])

    skipped = 0
    for name, rec in image_records.items():
        # NOTE: the dataset has a typo in the key — "image_width:" with a colon
        W = rec.get("image_width:", rec.get("image_width"))
        H = rec.get("image_height")

        if not W or not H:
            print(f"[Warn] Missing image dimensions for {name}, skipping.")
            skipped += 1
            continue

        W, H = float(W), float(H)
        lines = []

        for box in rec.get("bbox", []):
            cls   = int(box["class"])
            left  = float(box["left"])
            top   = float(box["top"])
            bw    = float(box["width"])
            bh    = float(box["height"])

            # YOLO: center x/y, width, height — all normalized [0,1]
            cx = (left + bw / 2) / W
            cy = (top  + bh / 2) / H
            nw = bw / W
            nh = bh / H

            # Clamp to [0, 1] to handle any boundary overflow
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stem = Path(name).stem
        label_file = out_dir / f"{stem}.txt"
        label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    total = len(image_records)
    print(f"Done. {total - skipped}/{total} label files written to: {out_dir}")
    if skipped:
        print(f"  Skipped (missing dimensions): {skipped}")

    # Optional: write a YOLO dataset.yaml
    if yaml_path:
        img_path_str = str(images_dir.resolve()) if images_dir else "path/to/images"
        names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(categories))
        yaml_content = (
            f"path: {img_path_str}\n"
            f"train: .\n"
            f"val: .\n\n"
            f"nc: {len(categories)}\n"
            f"names:\n{names_block}\n"
        )
        yaml_path.write_text(yaml_content, encoding="utf-8")
        print(f"dataset.yaml written to: {yaml_path}")
        print(f"Classes ({len(categories)}): {categories}")


def main():
    ap = argparse.ArgumentParser(description="Convert AU-AIR annotations.json to YOLO label files.")
    ap.add_argument("--json",   required=True, help="Path to annotations.json")
    ap.add_argument("--out",    required=True, help="Output directory for .txt label files")
    ap.add_argument("--yaml",   default=None,  help="(Optional) Write a dataset.yaml to this path")
    ap.add_argument("--images", default=None,  help="(Optional) Absolute path to images folder, written into yaml")
    args = ap.parse_args()

    convert(
        json_path  = Path(args.json),
        out_dir    = Path(args.out),
        yaml_path  = Path(args.yaml) if args.yaml else None,
        images_dir = Path(args.images) if args.images else None,
    )


if __name__ == "__main__":
    main()
