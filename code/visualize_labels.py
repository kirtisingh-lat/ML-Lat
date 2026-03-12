"""
visualize_labels.py

Overlay YOLO OBB bounding boxes on images, processing entire folders.

Usage:
    python visualize_labels.py <images_dir> <labels_dir> <output_dir> [--thickness N]

Example:
    python visualize_labels.py dataset/images dataset/labels output/visualized
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def draw_boxes(img_path: Path, lbl_path: Path, out_path: Path, thickness: int = 2):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[Warn] Could not read image: {img_path}")
        return

    h, w = img.shape[:2]
    font_scale = max(0.5, w / 3000.0)
    font_th = max(1, int(w / 2000.0))

    if not lbl_path.exists():
        print(f"[Warn] No label file: {lbl_path}")
    else:
        text = lbl_path.read_text(encoding="utf-8").strip()
        for line in text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 9:
                # YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4
                class_id = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
                pts = []
                for i in range(0, 8, 2):
                    x = int(round(max(0, min(w - 1, coords[i] * w))))
                    y = int(round(max(0, min(h - 1, coords[i + 1] * h))))
                    pts.append([x, y])
                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts_np], isClosed=True, color=(0, 255, 0), thickness=thickness)
                x0, y0 = pts[0]
                cv2.putText(
                    img, f"cls {class_id}",
                    (x0 + 5, max(font_th + 5, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_th, cv2.LINE_AA,
                )

            elif len(parts) == 5:
                # YOLO standard bbox: class cx cy w h (normalized)
                class_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                cv2.putText(
                    img, f"cls {class_id}",
                    (x1, max(font_th + 5, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_th, cv2.LINE_AA,
                )

            else:
                print(f"[Warn] Unexpected label format ({len(parts)} tokens): {line.strip()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")


def process_folder(images_dir: str, labels_dir: str, output_dir: str, thickness: int = 2):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not image_paths:
        print(f"[Error] No images found in: {images_dir}")
        return

    print(f"Processing {len(image_paths)} image(s) from '{images_dir}' -> '{output_dir}'")

    for img_path in image_paths:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        out_path = output_dir / img_path.name
        draw_boxes(img_path, lbl_path, out_path, thickness)

    print(f"\nDone. Results in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay YOLO labels on images (folder to folder).")
    parser.add_argument("images_dir", help="Input folder with images")
    parser.add_argument("labels_dir", help="Folder with YOLO .txt label files")
    parser.add_argument("output_dir", help="Output folder for annotated images")
    parser.add_argument("--thickness", type=int, default=2, help="Box line thickness (default: 1)")
    args = parser.parse_args()

    process_folder(args.images_dir, args.labels_dir, args.output_dir, args.thickness)
