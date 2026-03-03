# this code crop the images and labels into tiles, avoiding chopped objects. It uses Sutherland–Hodgman polygon clipping to handle partial objects at tile borders, and fits 4-point OBBs for YOLO-OBB format. Usage example:
# python tile_yolo_obb.py \
#   --images data/images \
#   --labels data/labels \
#   --out data_tiled \
#   --tile 1024 1024 \
#   --overlap 0.25 \
#   --keep_vis 0.60 \
#   --save_negatives

import argparse
import cv2
import numpy as np
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------- geometry helpers ----------------
def poly_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def clip_poly_to_rect(poly, x0, y0, x1, y1):
    """
    Sutherland–Hodgman polygon clipping to axis-aligned rectangle.
    poly: (N,2) absolute pixels
    returns (M,2) or None
    """
    poly = poly.astype(np.float32)

    def inside(p, edge):
        if edge == "L": return p[0] >= x0
        if edge == "R": return p[0] <= x1
        if edge == "T": return p[1] >= y0
        if edge == "B": return p[1] <= y1
        raise ValueError(edge)

    def intersect(a, b, edge):
        xA, yA = a
        xB, yB = b
        if edge in ("L", "R"):
            xE = x0 if edge == "L" else x1
            if abs(xB - xA) < 1e-12:
                return np.array([xE, yA], dtype=np.float32)
            t = (xE - xA) / (xB - xA)
            yE = yA + t * (yB - yA)
            return np.array([xE, yE], dtype=np.float32)
        else:
            yE = y0 if edge == "T" else y1
            if abs(yB - yA) < 1e-12:
                return np.array([xA, yE], dtype=np.float32)
            t = (yE - yA) / (yB - yA)
            xE = xA + t * (xB - xA)
            return np.array([xE, yE], dtype=np.float32)

    def clip_edge(inp, edge):
        if inp is None or len(inp) < 3:
            return None
        out = []
        prev = inp[-1]
        prev_in = inside(prev, edge)
        for cur in inp:
            cur_in = inside(cur, edge)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur, edge))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur, edge))
            prev, prev_in = cur, cur_in
        return np.array(out, dtype=np.float32) if len(out) >= 3 else None

    for e in ["L", "R", "T", "B"]:
        poly = clip_edge(poly, e)
        if poly is None:
            return None
    return poly

def min_area_rect_4pts(poly_tile_xy):
    rect = cv2.minAreaRect(poly_tile_xy.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)  # (4,2)
    return box

# ---------------- core tiler ----------------
def tile_dataset(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    tile_w: int,
    tile_h: int,
    overlap: float,
    keep_vis: float,
    save_negatives: bool,
    draw_overlays: bool,
    thickness: int,
):
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_ovr = out_dir / "overlays"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    if draw_overlays:
        out_ovr.mkdir(parents=True, exist_ok=True)

    stride_w = max(1, int(tile_w * (1.0 - overlap)))
    stride_h = max(1, int(tile_h * (1.0 - overlap)))

    img_files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print("[Warn] Could not read:", img_path)
            continue

        H, W = img.shape[:2]
        lbl_path = labels_dir / (img_path.stem + ".txt")

        # Parse labels: cls x1 y1 x2 y2 x3 y3 x4 y4  (normalized to full image)
        objects = []
        if lbl_path.exists():
            lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for ln in lines:
                parts = ln.split()
                if len(parts) != 9:
                    continue
                cls = int(float(parts[0]))
                pts = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(4, 2)
                pts[:, 0] *= W
                pts[:, 1] *= H
                a0 = poly_area(pts)
                if a0 > 1.0:
                    objects.append((cls, pts, a0))

        tile_id = 0
        # iterate tiles for this image (works for variable image sizes)
        for y0 in range(0, H, stride_h):
            for x0 in range(0, W, stride_w):
                x1 = min(x0 + tile_w, W)
                y1 = min(y0 + tile_h, H)

                tile = img[y0:y1, x0:x1].copy()
                th, tw = tile.shape[:2]

                kept_lines = []
                overlay = tile.copy() if draw_overlays else None

                for cls, poly_abs, a0 in objects:
                    # fast reject via bbox
                    minxy = poly_abs.min(axis=0)
                    maxxy = poly_abs.max(axis=0)
                    if maxxy[0] < x0 or maxxy[1] < y0 or minxy[0] > x1 or minxy[1] > y1:
                        continue

                    clipped = clip_poly_to_rect(poly_abs, x0, y0, x1, y1)
                    if clipped is None:
                        continue

                    a1 = poly_area(clipped)
                    if (a1 / a0) < keep_vis:
                        # too chopped -> drop to avoid half objects
                        continue

                    # shift to tile coords
                    clipped_tile = clipped.copy()
                    clipped_tile[:, 0] -= x0
                    clipped_tile[:, 1] -= y0

                    # fit 4-point OBB (YOLO expects 4 pts)
                    box4 = min_area_rect_4pts(clipped_tile)

                    # normalize to tile
                    boxn = box4.copy()
                    boxn[:, 0] /= tw
                    boxn[:, 1] /= th
                    boxn = np.clip(boxn, 0.0, 1.0)

                    coords = boxn.reshape(-1).tolist()
                    kept_lines.append(str(cls) + " " + " ".join(f"{c:.6f}" for c in coords))

                    if draw_overlays:
                        cv2.polylines(overlay, [box4.astype(np.int32)], True, (0, 255, 0), thickness)

                # save tile?
                if (not kept_lines) and (not save_negatives):
                    tile_id += 1
                    continue

                out_name = f"{img_path.stem}_t{tile_id:05d}.jpg"
                cv2.imwrite(str(out_img / out_name), tile)
                (out_lbl / (Path(out_name).stem + ".txt")).write_text(
                    "\n".join(kept_lines) + ("\n" if kept_lines else ""),
                    encoding="utf-8"
                )
                if draw_overlays:
                    cv2.imwrite(str(out_ovr / out_name), overlay)

                tile_id += 1

        print(f"[OK] {img_path.name}: {tile_id} tiles")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Tile variable-size images + YOLO-OBB 8pt labels, avoiding chopped objects.")
    ap.add_argument("--images", required=True, help="Path to images directory")
    ap.add_argument("--labels", required=True, help="Path to labels directory")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--tile", nargs=2, type=int, default=[1024, 1024], metavar=("W", "H"),
                    help="Tile size, e.g. --tile 1024 1024")
    ap.add_argument("--overlap", type=float, default=0.25, help="Overlap fraction, e.g. 0.25")
    ap.add_argument("--keep_vis", type=float, default=0.60, help="Keep if clipped area/original area >= this")
    ap.add_argument("--save_negatives", action="store_true", help="Also save tiles with no objects")
    ap.add_argument("--no_overlay", action="store_true", help="Do not save overlay images")
    ap.add_argument("--thickness", type=int, default=2, help="Overlay polygon thickness")
    args = ap.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_dir = Path(args.out)

    tile_w, tile_h = args.tile
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        out_dir=out_dir,
        tile_w=tile_w,
        tile_h=tile_h,
        overlap=args.overlap,
        keep_vis=args.keep_vis,
        save_negatives=args.save_negatives,
        draw_overlays=(not args.no_overlay),
        thickness=args.thickness,
    )

if __name__ == "__main__":
    main()
