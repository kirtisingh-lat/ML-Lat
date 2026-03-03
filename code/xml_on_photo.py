import cv2
import numpy as np
import xml.etree.ElementTree as ET

image_path = "/home/ss/Kirti/lat/EAGLE_Dataset_public/Val/images/2006-05-03-Allianz-links-yr7e0006.jpg"
label_path = "/home/ss/Kirti/lat/EAGLE_Dataset_public/Val/label_xmls/2006-05-03-Allianz-links-yr7e0006.xml"
out_path   = "/home/ss/Kirti/lat/debug.png"

# ---- Conventions (these two usually fix EAGLE-like issues) ----
ANGLE_AXIS = "w"      # "w" means angle is along width/heading, "h" means along height
CLOCKWISE  = True     # try True first for aerial datasets; if wrong, set False
ANGLE_DEG  = True     # EAGLE is typically degrees
ANGLE_OFFSET_DEG = 0  # if still off by 90, try +90 or -90 here

THICKNESS = 2
COLOR = (0, 255, 0)   # BGR green


def corners_from_cxcywha(cx, cy, w, h, a):
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)

    theta = float(a)
    if ANGLE_DEG:
        theta = np.deg2rad(theta + ANGLE_OFFSET_DEG)

    # If dataset uses clockwise angles, flip sign to get standard math rotation
    if CLOCKWISE:
        theta = -theta

    # Define unit vectors for width-axis (u) and height-axis (v)
    # If ANGLE_AXIS == "w": u is along angle, v is perpendicular
    # If ANGLE_AXIS == "h": v is along angle, u is perpendicular
    if ANGLE_AXIS.lower() == "w":
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)      # width direction
        v = np.array([-u[1], u[0]], dtype=np.float32)                      # height direction
    else:
        v = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)      # height direction
        u = np.array([-v[1], v[0]], dtype=np.float32)                      # width direction

    c  = np.array([cx, cy], dtype=np.float32)
    du = (w / 2.0) * u
    dv = (h / 2.0) * v

    p1 = c - du - dv
    p2 = c + du - dv
    p3 = c + du + dv
    p4 = c - du + dv
    return np.stack([p1, p2, p3, p4], axis=0)


def draw_poly(img, pts, color, thickness=2):
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i], True, color, thickness)


def parse_xml_cxcywha(xml_path):
    """
    EAGLE XMLs vary a bit by converter. This looks for cx,cy,w,h,a inside any object-like node.
    If your XML uses different tags/paths, tell me and I’ll hardcode the exact xpath.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def find_text(node, tags):
        tags = {t.lower() for t in tags}
        for ch in node.iter():
            if ch.tag.lower() in tags and ch.text:
                t = ch.text.strip()
                if t:
                    return t
        return None

    out = []
    for node in root.iter():
        # try to find a complete set anywhere under this node
        cx = find_text(node, ["cx", "centerx", "xcenter"])
        cy = find_text(node, ["cy", "centery", "ycenter"])
        w  = find_text(node, ["w", "width"])
        h  = find_text(node, ["h", "height"])
        a  = find_text(node, ["a", "angle", "theta", "rotation"])
        if cx and cy and w and h and a:
            out.append((float(cx), float(cy), float(w), float(h), float(a)))
    return out


def main():
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    dets = parse_xml_cxcywha(label_path)
    print("Found boxes:", len(dets))

    for cx, cy, w, h, a in dets:
        pts = corners_from_cxcywha(cx, cy, w, h, a)
        draw_poly(img, pts, COLOR, THICKNESS)

    cv2.imwrite(out_path, img)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
