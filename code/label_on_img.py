import cv2
import numpy as np
from pathlib import Path


def draw_yolo_obb_on_image(
    image_path,
    label_path,
    out_path=None,
    thickness=2,
    show_corner_ids=True,
    show_class_text=True,
):
    image_path = Path(image_path)
    label_path = Path(label_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    print(f"IMG size: {w} x {h}  |  {image_path.name}")

    # Resolution-aware text scaling (good for 5K images)
    font_scale = max(0.6, w / 3000.0)          # ~1.8 when w=5616
    font_th = max(1, int(w / 2000.0))          # ~2 when w=5616
    r = max(3, int(w / 1200.0))                # circle radius

    if not label_path.exists():
        print(f"[Warn] Label file not found: {label_path}")
        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), img)
            print(f"Saved (image only): {out_path}")
        return img

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        print(f"[Info] Empty label file (no objects): {label_path.name}")
        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), img)
            print(f"Saved (no boxes): {out_path}")
        return img

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for ln in lines:
        parts = ln.split()
        if len(parts) != 9:
            print(f"[Warn] Bad label line (expected 9 tokens): {ln}")
            continue

        class_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        # Convert normalized corners -> pixel corners
        pts = []
        for i in range(0, 8, 2):
            x = coords[i] * w
            y = coords[i + 1] * h

            # Clamp for safety
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))

            pts.append([int(round(x)), int(round(y))])

        pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        # Draw polygon
        cv2.polylines(img, [pts_np], isClosed=True, color=(0, 255, 0), thickness=thickness)

        # Optional: corner ids
        # if show_corner_ids:
        #     for k in range(4):
        #         xk, yk = pts_np[k, 0]
        #         cv2.circle(img, (xk, yk), r, (0, 0, 255), -1)
        #         cv2.putText(
        #             img,
        #             str(k + 1),
        #             (xk + 8, yk + 8),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             font_scale,
        #             (0, 0, 255),
        #             font_th,
        #             cv2.LINE_AA,
        #         )

        # Optional: class text near first point
        # if show_class_text:
        #     x0, y0 = pts_np[0, 0]
        #     cv2.putText(
        #         img,
        #         f"class {class_id}",
        #         (x0 + 10, max(0, y0 - 10)),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         font_scale,
        #         (0, 255, 0),
        #         font_th,
        #         cv2.LINE_AA,
        #     )

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path}")

    return img


if __name__ == "__main__":
    image_path = "/home/ss/Kirti/lat/EAGLE_Dataset_public/Val/images/2006-05-03-Allianz-links-yr7e0006.jpg"
    label_path = "debug_yolo_obb.txt"
    out_path = "/home/ss/Kirti/lat/debug_yeah.jpg"

    draw_yolo_obb_on_image(
        image_path=image_path,
        label_path=label_path,
        out_path=out_path,
        thickness=2,
        show_corner_ids=True,
        show_class_text=True,
    )
