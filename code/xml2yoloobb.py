import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------
# INPUTS
# ---------------------------
image_path = "/home/ss/Kirti/lat/EAGLE_Dataset_public/Val/images/2006-05-03-Allianz-links-yr7e0006.jpg"
xml_path   = "/home/ss/Kirti/lat/EAGLE_Dataset_public/Val/label_xmls/2006-05-03-Allianz-links-yr7e0006.xml"
out_txt    = "/home/ss/Kirti/lat/debug_yolo_obb.txt"

# ---- Conventions (match your verified visualization) ----
ANGLE_AXIS = "w"      # "w": angle is along width/heading
CLOCKWISE  = True     # dataset uses clockwise angles
ANGLE_DEG  = True
ANGLE_OFFSET_DEG = 0  # keep 0 since your plot looked correct

# ---------------------------
# CLASS MAPPING (EDIT THIS)
# ---------------------------
# EAGLE XML may store class as text like "small-vehicle", "large-vehicle"
# Map those names to YOLO class IDs.
NAME_TO_ID = {
    "small-vehicle": 0,
    "large-vehicle": 1,
}

DEFAULT_CLASS_ID = 0  # fallback if class name not found


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


def parse_xml_objects(xml_file):
    """
    Robust-ish XML parser:
    - extracts cx,cy,w,h,a
    - tries to extract class name from tags: name/class/label/type
    Returns list of tuples: (name_or_none, cx,cy,w,h,a)
    """
    tree = ET.parse(xml_file)
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


def normalize_pts(pts_xy, W, H):
    pts = pts_xy.copy().astype(np.float32)
    pts[:, 0] /= float(W)
    pts[:, 1] /= float(H)
    return pts


def main():
    # read image just to get width/height for normalization
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    H, W = img.shape[:2]

    objs = parse_xml_objects(xml_path)
    print("Found objects:", len(objs))

    lines_out = []
    for name, cx, cy, w, h, a in objs:
        pts = corners_from_cxcywha(cx, cy, w, h, a)        # pixel corners (4,2)
        pts_n = normalize_pts(pts, W, H)                   # normalized corners (0..1)

        # class id
        cls_name = (name or "").strip().lower()
        cls_id = NAME_TO_ID.get(cls_name, DEFAULT_CLASS_ID)

        # YOLO-OBB line: cls x1 y1 x2 y2 x3 y3 x4 y4
        flat = pts_n.reshape(-1).tolist()
        line = f"{cls_id} " + " ".join(f"{v:.6f}" for v in flat)
        lines_out.append(line)

    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(out_txt).write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")

    print("Wrote:", out_txt)


if __name__ == "__main__":
    main()
