import os
import math
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
DOTA_LABEL_DIR = "/home/ss/Kirti/lat/dota_dataset/labels/val"
IMAGE_DIR = "/home/ss/Kirti/lat/dota_dataset/images/val"
OUTPUT_DIR = "/home/ss/Kirti/lat/yolo_labels/val"
FORMAT = "xywha"  # "xywha" or "8pts"

MISSING_LOG = "missing_images.txt"

DOTA_CLASSES = [
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field",
    "roundabout", "harbor", "swimming-pool", "helicopter",
    "container-crane"
]

CLASS_TO_ID = {c: i for i, c in enumerate(DOTA_CLASSES)}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def polygon_to_xywha(pts):
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle_deg = rect

    if w < h:
        w, h = h, w
        angle_deg += 90

    angle_rad = math.radians(angle_deg)
    return cx, cy, w, h, angle_rad


# =========================
# MAIN
# =========================
missing_images = []

for label_file in os.listdir(DOTA_LABEL_DIR):
    if not label_file.endswith(".txt"):
        continue

    base = os.path.splitext(label_file)[0]
    img_path = os.path.join(IMAGE_DIR, base + ".png")

    # ---- Missing image handling ----
    if not os.path.exists(img_path):
        missing_images.append(base + ".png")
        continue

    img = cv2.imread(img_path)
    if img is None:
        missing_images.append(base + ".png")
        continue

    H, W = img.shape[:2]

    in_path = os.path.join(DOTA_LABEL_DIR, label_file)
    out_path = os.path.join(OUTPUT_DIR, label_file)

    with open(in_path, "r") as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            coords = list(map(float, parts[:8]))
            class_name = parts[8]

            if class_name not in CLASS_TO_ID:
                continue

            cls_id = CLASS_TO_ID[class_name]
            pts = np.array(
                [(coords[i], coords[i + 1]) for i in range(0, 8, 2)],
                dtype=np.float32
            )

            if FORMAT == "8pts":
                norm = []
                for x, y in pts:
                    norm.append(x / W)
                    norm.append(y / H)

                line_out = f"{cls_id} " + " ".join(f"{v:.6f}" for v in norm)

            elif FORMAT == "xywha":
                cx, cy, w, h, angle = polygon_to_xywha(pts)

                cx /= W
                cy /= H
                w /= W
                h /= H

                line_out = (
                    f"{cls_id} "
                    f"{cx:.6f} {cy:.6f} "
                    f"{w:.6f} {h:.6f} "
                    f"{angle:.6f}"
                )
            else:
                raise ValueError("Unknown FORMAT")

            f_out.write(line_out + "\n")

# =========================
# WRITE MISSING IMAGES LOG
# =========================
if missing_images:
    with open(MISSING_LOG, "w") as f:
        for name in sorted(set(missing_images)):
            f.write(name + "\n")

    print(f"[WARN] {len(set(missing_images))} missing images logged to {MISSING_LOG}")
else:
    print("[OK] No missing images found")