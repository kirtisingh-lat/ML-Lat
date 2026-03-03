import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# ============================================================
# CONFIG: directories
# ============================================================
IMAGES_DIR = Path("/home/ss/Kirti/lat/EAGLE_Dataset_public/Test/images")
XML_DIR    = Path("/home/ss/Kirti/lat/EAGLE_Dataset_public/Test/label_xmls")
OUT_DIR    = Path("/home/ss/Kirti/lat/EAGLE_Dataset_public/Test/label_yoloobb")  # output .txt here
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Conventions (match your verified visualization) ----
ANGLE_AXIS = "w"       # "w": angle is along width/heading
CLOCKWISE  = True      # dataset uses clockwise angles
ANGLE_DEG  = True
ANGLE_OFFSET_DEG = 0

# ---- Class mapping (EDIT if your XML uses different names) ----
NAME_TO_ID = {
    "small-vehicle": 0,
    "large-vehicle": 1,
}
DEFAULT_CLASS_ID = 0  # fallback if name missing / unknown

# Optional: skip writing label file if no boxes found
SKIP_EMPTY = False

# ============================================================
# GEOMETRY: (cx,cy,w,h,a) -> 4 corners
# ============================================================
def corners_from_cxcywha(cx, cy, w, h, a):
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)

    theta = float(a)
    if ANGLE_DEG:
        theta = np.deg2rad(theta + ANGLE_OFFSET_DEG)
    if CLOCKWISE:
        theta = -theta

    if ANGLE_AXIS.lower() == "w":
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)  # width direction
        v = np.array([-u[1], u[0]], dtype=np.float32)                  # height direction
    else:
        v = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        u = np.array([-v[1], v[0]], dtype=np.float32)

    c  = np.array([cx, cy], dtype=np.float32)
    du = (w / 2.0) * u
    dv = (h / 2.0) * v

    p1 = c - du - dv
    p2 = c + du - dv
    p3 = c + du + dv
    p4 = c - du + dv
    return np.stack([p1, p2, p3, p4], axis=0)  # (4,2)


def normalize_pts(pts, W, H):
    pts = pts.astype(np.float32).copy()
    pts[:, 0] /= float(W)
    pts[:, 1] /= float(H)
    return pts


# ============================================================
# XML parsing
# ============================================================
def parse_xml_objects(xml_file: Path):
    """
    Extract objects with cx,cy,w,h,a and optional class name from EAGLE-style XML.
    Returns list of tuples: (name_or_none, cx,cy,w,h,a)
    """
    tree = ET.parse(str(xml_file))
    root = tree.getroot()

    def find_text(node, tags):
        tags = {t.lower() for t in tags}
        for ch in node.iter():
            if ch.tag.lower() in tags and ch.text:
                t = ch.text.strip()
                if t:
                    return t
        return None

    objects = []
    for node in root.iter():
        cx = find_text(node, ["cx", "centerx", "xcenter"])
        cy = find_text(node, ["cy", "centery", "ycenter"])
        w  = find_text(node, ["w", "width"])
        h  = find_text(node, ["h", "height"])
        a  = find_text(node, ["a", "angle", "theta", "rotation"])
        if cx and cy and w and h and a:
            name = find_text(node, ["name", "class", "label", "type"])
            objects.append((name, float(cx), float(cy), float(w), float(h), float(a)))

    return objects


# ============================================================
# Main conversion
# ============================================================
def main():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts]

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {IMAGES_DIR}")

    n_imgs = 0
    n_written = 0
    n_missing_xml = 0
    n_empty = 0

    for img_path in sorted(image_paths):
        n_imgs += 1

        xml_path = XML_DIR / f"{img_path.stem}.xml"
        if not xml_path.exists():
            n_missing_xml += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[Warn] Could not read image: {img_path}")
            continue
        H, W = img.shape[:2]

        objs = parse_xml_objects(xml_path)

        if not objs:
            n_empty += 1
            if SKIP_EMPTY:
                continue
            # Write an empty label file (YOLO expects empty .txt for no objects)
            (OUT_DIR / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
            n_written += 1
            continue

        lines_out = []
        for name, cx, cy, w, h, a in objs:
            pts = corners_from_cxcywha(cx, cy, w, h, a)   # pixel
            pts_n = normalize_pts(pts, W, H)              # normalized 0..1

            cls_name = (name or "").strip().lower()
            cls_id = NAME_TO_ID.get(cls_name, DEFAULT_CLASS_ID)

            flat = pts_n.reshape(-1).tolist()
            line = f"{cls_id} " + " ".join(f"{v:.6f}" for v in flat)
            lines_out.append(line)

        out_txt = OUT_DIR / f"{img_path.stem}.txt"
        out_txt.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        n_written += 1

    print("=== Done ===")
    print("Images scanned        :", n_imgs)
    print("Label files written   :", n_written)
    print("Missing XML files     :", n_missing_xml)
    print("Images with 0 objects :", n_empty)
    print("Output dir            :", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
