import json
import argparse
from pathlib import Path


def clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def convert_auair_dict(data: dict, out: Path):
    """AUAIR dict format: {categories:[str,...], annotations:[{image_name, image_width:, image_height, bbox:[{top,left,height,width,class}]}]}"""
    class_names = data["categories"]  # list of strings, index = class id

    written = 0
    for entry in data["annotations"]:
        # Note: dataset has a typo — key is "image_width:" (with colon)
        W = entry.get("image_width") or entry.get("image_width:")
        H = entry["image_height"]
        stem = Path(entry["image_name"]).stem
        lines = []
        for box in entry.get("bbox", []):
            x = box["left"]
            y = box["top"]
            w = box["width"]
            h = box["height"]
            cls = box["class"]
            cx = clamp((x + w / 2) / W)
            cy = clamp((y + h / 2) / H)
            nw = clamp(w / W)
            nh = clamp(h / H)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (out / f"{stem}.txt").write_text("\n".join(lines))
        written += 1

    (out / "classes.txt").write_text("\n".join(class_names))
    print(f"Converted {written} images -> {out}")
    print(f"Classes ({len(class_names)}): {class_names}")


def convert_coco(data: dict, out: Path, yaml_path: Path | None = None):
    """Standard COCO format: {images, annotations, categories}"""
    images = {img["id"]: img for img in data["images"]}

    categories = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    names = [c["name"] for c in categories]

    annotations = data.get("annotations", [])
    ann_by_image: dict = {}
    for ann in annotations:
        if ann.get("iscrowd", 0):
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    if not annotations:
        print("[Info] No annotations in this split (test set). Writing empty label files.")

    written = 0
    for img in images.values():
        img_id = img["id"]
        W, H = img["width"], img["height"]
        stem = Path(img["file_name"]).stem
        lines = []
        for ann in ann_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cx = clamp((x + w / 2) / W)
            cy = clamp((y + h / 2) / H)
            nw = clamp(w / W)
            nh = clamp(h / H)
            cls = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Converted {written} images -> {out}")
    print(f"  Annotated: {len(ann_by_image)}  |  Empty: {written - len(ann_by_image)}")
    print(f"Classes ({len(names)}): {names}")

    if yaml_path:
        names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
        yaml_path.write_text(f"nc: {len(names)}\nnames:\n{names_block}\n", encoding="utf-8")
        print(f"dataset.yaml written -> {yaml_path}")


def json2yolo(json_path: str, output_dir: str, yaml_path: str | None = None):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    yp = Path(yaml_path) if yaml_path else None

    if isinstance(data, dict) and "annotations" in data and isinstance(data.get("categories"), list) and isinstance(data["categories"][0], str):
        convert_auair_dict(data, out)
    elif isinstance(data, dict) and "images" in data:
        convert_coco(data, out, yaml_path=yp)
    else:
        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        raise ValueError(f"Unrecognised JSON format. Top-level type/keys: {keys}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO/AUAIR JSON to YOLO labels")
    parser.add_argument("--json",   required=True, help="Path to annotations JSON file")
    parser.add_argument("--out",    required=True, help="Output folder for .txt label files")
    parser.add_argument("--yaml",   default=None,  help="(Optional) Write dataset.yaml to this path")
    args = parser.parse_args()

    json2yolo(args.json, args.out, args.yaml)
