"""
Dataset Unifier v2
==================
- Auto-discovers classes from all datasets
- Remaps all classes to: 0=person, 1=vehicle (ignores rest)
- Converts COCO/VOC/YOLO -> YOLO  (no image resizing — originals are copied as-is)
- Splits into Train/Val/Test with a fixed random seed (reproducible)
- Deduplicates images across datasets by MD5 content hash
- Reports image-size statistics per dataset in the info file
- Writes a detailed run_info.txt report on completion

Workflow:
  # Step 1 — generate mapping JSON for review (script exits after this)
  python unified_dataset.py --root ROOT --out OUT --dump-mapping mapping.json

  # Step 2 — edit mapping.json if needed, then run for real
  python unified_dataset.py --root ROOT --out OUT --mapping mapping.json

  # One-shot (auto-mapping, no overrides)
  python unified_dataset.py --root ROOT --out OUT

Requirements:
    pip install opencv-python pyyaml tqdm
"""

import cv2
import hashlib
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import argparse

IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASTER_CLASSES = {0: "person", 1: "vehicle"}
SKIP_TXT_NAMES = {"classes.txt", "obj.names", "_darknet.labels", "data.txt", "predefined_classes.txt"}
PRIORITY_YAML  = {"data.yaml", "data.yml", "dataset.yaml"}

PERSON_KEYWORDS  = {"person", "people", "pedestrian", "human", "man", "woman",
                    "rider", "crowd", "civilian", "walker", "cyclist"}
VEHICLE_KEYWORDS = {"vehicle", "car", "truck", "bus", "van", "motorcycle",
                    "motorbike", "bicycle", "bike", "scooter", "tricycle",
                    "awning-tricycle", "motor", "jeep", "suv", "pickup",
                    "trailer", "lorry", "ambulance", "taxi", "auto"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ─── Step 1: Class discovery ──────────────────────────────────────────────────

def find_class_files(dataset_dir: Path) -> list:
    """
    Shallow search only — class files are never inside images/ or labels/ dirs.
    Checks dataset root + up to 3 levels of non-data subdirectories only.
    """
    KNOWN_NAMES = PRIORITY_YAML | {"classes.txt", "obj.names", "_darknet.labels"}
    # Directories that contain bulk data (images/labels) — skip recursing into them
    SKIP_DIRS = {"images", "image", "data", "labels", "labels_yolo", "label", "annotations",
                 "overlays", "raw", "bin", "__pycache__", ".git"}
    found = []
    def _walk(p: Path, depth: int):
        if depth > 3:
            return
        try:
            for child in p.iterdir():
                if child.is_file() and child.name in KNOWN_NAMES:
                    found.append(child)
                elif child.is_dir() and child.name.lower() not in SKIP_DIRS:
                    _walk(child, depth + 1)
        except PermissionError:
            pass
    _walk(dataset_dir, 0)
    return found


def parse_class_file(path: Path) -> list[str]:
    try:
        if path.suffix in {".yaml", ".yml"}:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                names = data.get("names", [])
                if isinstance(names, list):
                    return [str(n).lower().strip() for n in names]
                if isinstance(names, dict):
                    # FIX: sort by integer key so order is correct
                    return [str(v).lower().strip() for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
        else:
            with open(path, encoding="utf-8") as f:
                lines = [ln.strip().lower() for ln in f if ln.strip()]
            if not any("/" in ln or "\\" in ln for ln in lines):
                return lines
    except Exception as e:
        print(f"  [warn] Could not parse {path.name}: {e}")
    return []


def discover_all_classes(root: Path, warnings: list) -> dict:
    """Returns { dataset_name: {class_id: class_name} }"""
    dataset_classes = {}
    for ds in sorted(root.iterdir()):
        if not ds.is_dir():
            continue
        class_files = find_class_files(ds)
        if not class_files:
            msg = f"No class file found in: {ds.name}"
            print(f"  [warn] {msg}")
            warnings.append(msg)
            continue

        # FIX: prefer data.yaml/dataset.yaml, then sort alphabetically for stability
        class_files.sort(key=lambda p: (p.name not in PRIORITY_YAML, p.name))
        primary_file = class_files[0]
        primary = parse_class_file(primary_file)

        if not primary:
            msg = f"Empty class list in {ds.name} ({primary_file.name})"
            print(f"  [warn] {msg}")
            warnings.append(msg)
            continue

        # FIX: warn on conflicts across multiple class files (all splits, etc.)
        for cf in class_files[1:]:
            other = parse_class_file(cf)
            if other and other != primary:
                msg = f"{ds.name}: {cf.name} differs from {primary_file.name} — using primary"
                print(f"  [warn] {msg}")
                warnings.append(msg)

        dataset_classes[ds.name] = {i: c for i, c in enumerate(primary)}
        print(f"  [{ds.name}] {len(primary)} classes  (from {primary_file.name})")

    return dataset_classes


# ─── Step 2: Class mapping ────────────────────────────────────────────────────

def auto_map_class(name: str) -> int:
    n = name.lower().strip()
    if n in PERSON_KEYWORDS or any(k in n for k in PERSON_KEYWORDS):
        return 0
    if n in VEHICLE_KEYWORDS or any(k in n for k in VEHICLE_KEYWORDS):
        return 1
    return -1


def build_mappings(dataset_classes: dict,
                   override_path: str | None,
                   dump_path: str | None,
                   warnings: list) -> dict:
    """
    Returns { dataset_name: {src_class_id: master_class_id} }

    If dump_path is set  -> writes auto-mapping JSON and returns (caller exits).
    If override_path set -> loads user-edited JSON and applies all overrides at once.
    """
    mappings = {
        ds: {cid: auto_map_class(cname) for cid, cname in id_map.items()}
        for ds, id_map in dataset_classes.items()
    }

    if dump_path:
        dump = {}
        for ds_name, mapping in mappings.items():
            id_to_name = dataset_classes[ds_name]
            dump[ds_name] = {
                str(cid): {
                    "name": id_to_name[cid],
                    "mapped_to": mid,
                    "label": MASTER_CLASSES.get(mid, "IGNORE"),
                }
                for cid, mid in mapping.items()
            }
        Path(dump_path).write_text(json.dumps(dump, indent=2), encoding="utf-8")
        print(f"\nAuto-mapping written -> {dump_path}")
        print("Edit 'mapped_to' values (0=person, 1=vehicle, -1=ignore) for any class.")
        print("Then re-run with --mapping <file> to apply all overrides at once.")

    if override_path:
        # FIX: apply ALL overrides from file in one pass (not limited to one edit)
        with open(override_path, encoding="utf-8") as f:
            overrides = json.load(f)
        n_changes = 0
        for ds_name, cls_overrides in overrides.items():
            if ds_name not in mappings:
                msg = f"Override: dataset '{ds_name}' not in discovered datasets"
                print(f"  [warn] {msg}")
                warnings.append(msg)
                continue
            for cid_str, val in cls_overrides.items():
                cid = int(cid_str)
                new_val = val if isinstance(val, int) else int(val.get("mapped_to", -1))
                old_val = mappings[ds_name].get(cid, "?")
                if old_val != new_val:
                    mappings[ds_name][cid] = new_val
                    n_changes += 1
                    name = dataset_classes.get(ds_name, {}).get(cid, "?")
                    print(f"  Override [{ds_name}] {cid}('{name}'): {old_val} -> {new_val}")
        print(f"Applied {n_changes} mapping change(s) from {override_path}")

    print("\n--- Final class mappings (0=person | 1=vehicle | -1=ignore) ---")
    for ds_name, mapping in mappings.items():
        id_to_name = dataset_classes.get(ds_name, {})
        mapped  = {id_to_name.get(c, c): MASTER_CLASSES[m] for c, m in mapping.items() if m >= 0}
        ignored = [id_to_name.get(c, c) for c, m in mapping.items() if m == -1]
        print(f"  [{ds_name}]: keep={mapped}  ignore={ignored}")

    return mappings


# ─── Step 3: Find image↔label pairs (structure-aware, no full tree walk) ───────

def _find_images_dirs(ds: Path) -> list[Path]:
    """Find all directories named 'images' up to 4 levels deep (no full recursion)."""
    found = []
    def _walk(p: Path, depth: int):
        if depth > 4:
            return
        try:
            for child in p.iterdir():
                if child.is_dir():
                    if child.name.lower() in {"images", "image", "data"}:
                        found.append(child)
                    else:
                        _walk(child, depth + 1)
        except PermissionError:
            pass
    _walk(ds, 0)
    return found


def _label_dir_for(images_dir: Path) -> Path | None:
    """Find the best label directory sibling to an images/ dir."""
    parent = images_dir.parent
    for name in ("labels_yolo", "labels", "label", "annotations"):
        p = parent / name
        if p.is_dir():
            return p
    return None


def find_image_label_pairs(ds: Path, warnings: list) -> list:
    """
    Structure-aware: find images/ dirs, then scan only their sibling labels/ dirs.
    Avoids walking 300k+ unrelated files on large datasets over network drives.
    """
    COCO_JSON_SIZE_LIMIT = 5 * 1024 * 1024  # 5 MB

    img_dirs = _find_images_dirs(ds)
    if not img_dirs:
        warnings.append(f"{ds.name}: no 'images/' directory found")
        return []

    pairs = []
    unpaired = 0

    for img_dir in img_dirs:
        lbl_dir = _label_dir_for(img_dir)

        # Build label index from the sibling labels/ dir only
        txt_index:  dict[str, Path] = {}
        xml_index:  dict[str, Path] = {}
        coco_index: dict[str, tuple] = {}

        if lbl_dir:
            for lbl_file in lbl_dir.iterdir():
                if not lbl_file.is_file():
                    continue
                ext = lbl_file.suffix.lower()
                if ext == ".txt":
                    if lbl_file.name not in SKIP_TXT_NAMES:
                        txt_index[lbl_file.stem] = lbl_file
                elif ext == ".xml":
                    xml_index[lbl_file.stem] = lbl_file
                elif ext == ".json" and lbl_file.stat().st_size <= COCO_JSON_SIZE_LIMIT:
                    try:
                        with open(lbl_file, encoding="utf-8") as f:
                            data = json.load(f)
                        if "images" in data and "annotations" in data:
                            id_to_img = {img["id"]: img for img in data["images"]}
                            ann_by_id: dict = defaultdict(list)
                            for ann in data["annotations"]:
                                ann_by_id[ann["image_id"]].append(ann)
                            for img_id, anns in ann_by_id.items():
                                info = id_to_img.get(img_id)
                                if info:
                                    stem = Path(info["file_name"]).stem
                                    if stem not in coco_index:
                                        coco_index[stem] = (anns, info["width"], info["height"])
                    except Exception:
                        pass

        # Match each image to its label
        for img_file in img_dir.iterdir():
            if not img_file.is_file() or img_file.suffix.lower() not in IMG_EXTS:
                continue
            stem = img_file.stem
            if stem in txt_index and txt_index[stem].stat().st_size > 0:
                pairs.append((img_file, txt_index[stem], "yolo", None))
            elif stem in xml_index:
                pairs.append((img_file, xml_index[stem], "voc", None))
            elif stem in coco_index:
                pairs.append((img_file, None, "coco", coco_index[stem]))
            else:
                unpaired += 1

    if unpaired:
        warnings.append(f"{ds.name}: {unpaired} image(s) had no matching label -- skipped")

    return pairs


# ─── Step 4: Parse labels ─────────────────────────────────────────────────────

def parse_yolo(label_path: Path, mapping: dict) -> list:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        master = mapping.get(cid, -1)
        if master == -1:
            continue
        if len(parts) == 9:
            # YOLO-OBB format: cls x1 y1 x2 y2 x3 y3 x4 y4 (normalized corners)
            # Convert to axis-aligned bbox via min/max of corners
            coords = list(map(float, parts[1:9]))
            xs = coords[0::2]
            ys = coords[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            cx = max(0.0, min(1.0, (xmin + xmax) / 2))
            cy = max(0.0, min(1.0, (ymin + ymax) / 2))
            w  = max(0.0, min(1.0, xmax - xmin))
            h  = max(0.0, min(1.0, ymax - ymin))
        else:
            cx, cy, w, h = map(float, parts[1:5])
        boxes.append((master, cx, cy, w, h))
    return boxes


def parse_voc(xml_path: Path) -> list:
    # FIX: VOC uses class names not IDs — always use auto_map_class (name-based)
    # Numeric mapping dict is irrelevant here; keyword-based mapping is correct.
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        iw = int(size.find("width").text)  if size is not None else 1
        ih = int(size.find("height").text) if size is not None else 1
        for obj in root.findall("object"):
            cname = obj.find("name").text.lower().strip()
            master = auto_map_class(cname)
            if master == -1:
                continue
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            cx = clamp(((xmin + xmax) / 2) / iw)
            cy = clamp(((ymin + ymax) / 2) / ih)
            bw = clamp((xmax - xmin) / iw)
            bh = clamp((ymax - ymin) / ih)
            boxes.append((master, cx, cy, bw, bh))
    except Exception as e:
        print(f"  [warn] VOC parse error {xml_path.name}: {e}")
    return boxes


def parse_coco_anns(annotations: list, mapping: dict, img_w: int, img_h: int) -> list:
    boxes = []
    for ann in annotations:
        cid = ann.get("category_id", -1)
        master = mapping.get(cid, -1)
        if master == -1:
            continue
        x, y, w, h = ann["bbox"]
        boxes.append((master, clamp((x + w/2)/img_w), clamp((y + h/2)/img_h),
                      clamp(w/img_w), clamp(h/img_h)))
    return boxes


# ─── Step 5: Process and write output ─────────────────────────────────────────

def write_yolo_label(path: Path, boxes: list):
    path.write_text(
        "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes) + "\n",
        encoding="utf-8",
    )


_WIN_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

def sanitize_stem(s: str) -> str:
    """Replace Windows-invalid filename characters with underscores."""
    return _WIN_INVALID.sub("_", s)


def process_datasets(root: Path, output: Path, split_ratio: tuple,
                     seed: int, mappings: dict, warnings: list) -> dict:
    random.seed(seed)  # FIX: fixed seed for reproducible splits

    for split in ["train", "val", "test"]:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    all_samples = []  # (img_path, label_path_or_None, fmt, extra, mapping, ds_name)
    per_ds: dict[str, dict] = {}

    print("\nScanning datasets for image-label pairs...")
    for ds in sorted(root.iterdir()):
        if not ds.is_dir():
            continue
        mapping = mappings.get(ds.name, {})
        pairs = find_image_label_pairs(ds, warnings)
        per_ds[ds.name] = {
            "pairs":       len(pairs),
            "written":     0,
            "dup":         0,
            "no_boxes":    0,  # FIX: track per-dataset skips
            "read_err":    0,
            "img_sizes":   [],
        }
        print(f"  {ds.name}: {len(pairs)} pairs")
        for item in pairs:
            all_samples.append((*item, mapping, ds.name))

    random.shuffle(all_samples)
    n       = len(all_samples)
    n_train = int(n * split_ratio[0])
    n_val   = int(n * split_ratio[1])
    splits  = [
        ("train", all_samples[:n_train]),
        ("val",   all_samples[n_train:n_train + n_val]),
        ("test",  all_samples[n_train + n_val:]),
    ]
    print(f"\nTotal: {n}  ->  train={n_train}, val={n_val}, test={n - n_train - n_val}")

    seen_hashes: set[str] = set()   # FIX: MD5 deduplication across datasets
    used_stems:  set[str] = set()   # FIX: global stem registry to prevent collisions
    split_counts = {"train": 0, "val": 0, "test": 0}

    for split_name, samples in splits:
        img_out = output / split_name / "images"
        lbl_out = output / split_name / "labels"

        for item in tqdm(samples, desc=f"Processing {split_name}"):
            img_path, label_path, fmt, extra, mapping, ds_name = item
            stats = per_ds[ds_name]

            # ── Deduplicate by file content hash ─────────────────────────────
            h = md5_file(img_path)
            if h in seen_hashes:
                stats["dup"] += 1
                continue
            seen_hashes.add(h)

            # ── Load image for size info (no resize) ─────────────────────────
            img = cv2.imread(str(img_path))
            if img is None:
                stats["read_err"] += 1
                warnings.append(f"Could not read: {img_path}")
                continue
            H, W = img.shape[:2]
            stats["img_sizes"].append((W, H))

            # ── Parse boxes ───────────────────────────────────────────────────
            try:
                if fmt == "yolo":
                    boxes = parse_yolo(label_path, mapping)
                elif fmt == "voc":
                    boxes = parse_voc(label_path)   # FIX: no longer passes broken mapping
                elif fmt == "coco":
                    anns, coco_w, coco_h = extra
                    boxes = parse_coco_anns(anns, mapping, coco_w, coco_h)
                else:
                    continue
            except Exception as e:
                stats["read_err"] += 1
                warnings.append(f"Label parse error {img_path.name}: {e}")
                continue

            # FIX: log per-dataset no-box skips (not silently lumped into a total)
            if not boxes:
                stats["no_boxes"] += 1
                continue

            # ── Build unique output stem ──────────────────────────────────────
            # FIX: use top-level dataset folder name, not parent.parent which is fragile
            base_stem   = sanitize_stem(f"{ds_name}_{img_path.stem}")
            unique_stem = base_stem
            idx = 1
            # FIX: collision check against a global set (not just file existence)
            # so img and lbl stems always stay in sync
            while unique_stem in used_stems:
                unique_stem = f"{base_stem}_{idx}"
                idx += 1
            used_stems.add(unique_stem)

            out_img = img_out / (unique_stem + img_path.suffix)
            out_lbl = lbl_out / (unique_stem + ".txt")

            # ── Copy image as-is (no resize) ─────────────────────────────────
            shutil.copy2(img_path, out_img)
            write_yolo_label(out_lbl, boxes)

            stats["written"] += 1
            split_counts[split_name] += 1

    return {"per_ds": per_ds, "split_counts": split_counts,
            "total_written": sum(split_counts.values()),
            "total_dup":      sum(s["dup"]      for s in per_ds.values()),
            "total_no_boxes": sum(s["no_boxes"] for s in per_ds.values()),
            "total_read_err": sum(s["read_err"] for s in per_ds.values())}


# ─── Step 6: data.yaml ────────────────────────────────────────────────────────

def write_data_yaml(output_dir: Path):
    data = {
        "path" : str(output_dir),
        "train": "train/images",
        "val"  : "val/images",
        "test" : "test/images",
        "nc"   : 2,
        "names": ["person", "vehicle"],
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"data.yaml saved -> {yaml_path}")


# ─── Step 7: Info report ──────────────────────────────────────────────────────

def write_info_report(output_dir: Path, cfg: dict, mappings: dict,
                      dataset_classes: dict, results: dict, warnings: list):
    lines = []
    sep = "=" * 70

    lines += [sep, "DATASET UNIFIER v2 — RUN REPORT",
              f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", sep]

    lines += ["\n[CONFIGURATION]",
              f"  root         : {cfg['root']}",
              f"  output       : {cfg['output']}",
              f"  split ratio  : train={cfg['split'][0]:.0%} / val={cfg['split'][1]:.0%} / test={cfg['split'][2]:.0%}",
              f"  random seed  : {cfg['seed']}  (reproducible splits)",
              f"  mapping file : {cfg.get('mapping_file') or 'auto-generated (no overrides)'}",
              f"  image resize : DISABLED — images copied at original resolution"]

    lines += ["\n[CHANGES FROM v1]",
              "  #1  Removed input() prompts — fully CLI-driven via argparse",
              "  #3  Mapping overrides via JSON file (unlimited changes, all at once)",
              "  #4  Fixed COCO cj loop-variable capture bug in find_image_label_pairs",
              "  #5  Built stem->path index once per dataset (no rglob per image)",
              "  #6  All class files checked; conflicts warned; stable sort for yaml dict names",
              "  #7  Collision uses global used_stems set; img+lbl stems always in sync",
              "  #8  random.seed() applied before shuffle for reproducible splits",
              "  #9  Per-dataset no-box skip counts logged in this report",
              "  #11 Image resize removed; image size stats reported per dataset",
              "  #12 MD5 content-hash deduplication across all datasets"]

    lines += ["\n[CLASS MAPPINGS]", "  0=person | 1=vehicle | -1=ignore"]
    for ds_name, mapping in mappings.items():
        id_to_name = dataset_classes.get(ds_name, {})
        lines.append(f"\n  [{ds_name}]")
        for cid, mid in sorted(mapping.items()):
            label = MASTER_CLASSES.get(mid, "IGNORE")
            lines.append(f"    class {cid:3d}  '{id_to_name.get(cid, '?'):<22}' -> {mid:2d}  ({label})")

    lines += ["\n[PER-DATASET STATISTICS]"]
    for ds_name, stats in results["per_ds"].items():
        lines.append(f"\n  {ds_name}:")
        lines.append(f"    pairs found     : {stats['pairs']}")
        lines.append(f"    written         : {stats['written']}")
        lines.append(f"    skipped no-boxes: {stats['no_boxes']}")
        lines.append(f"    skipped dup     : {stats['dup']}")
        lines.append(f"    skipped read err: {stats['read_err']}")
        sizes = stats["img_sizes"]
        if sizes:
            ws = [s[0] for s in sizes]
            hs = [s[1] for s in sizes]
            lines.append(f"    image width     : min={min(ws):5d}  max={max(ws):5d}  mean={sum(ws)//len(ws):5d}")
            lines.append(f"    image height    : min={min(hs):5d}  max={max(hs):5d}  mean={sum(hs)//len(hs):5d}")
        else:
            lines.append(f"    image sizes     : no data (all skipped)")

    sc = results["split_counts"]
    lines += ["\n[OVERALL RESULTS]",
              f"  Total written        : {results['total_written']}",
              f"  Duplicates removed   : {results['total_dup']}",
              f"  Skipped (no boxes)   : {results['total_no_boxes']}",
              f"  Skipped (read error) : {results['total_read_err']}",
              f"  Train images         : {sc['train']}",
              f"  Val   images         : {sc['val']}",
              f"  Test  images         : {sc['test']}"]

    if warnings:
        lines += [f"\n[WARNINGS]  ({len(warnings)} total)"]
        for w in warnings:
            lines.append(f"  ! {w}")
    else:
        lines.append("\n[WARNINGS]  None")

    report_path = output_dir / "run_info.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Run report saved -> {report_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Dataset Unifier v2 — merge multi-format datasets into unified YOLO format")
    ap.add_argument("--root",         required=True,
                    help="Folder containing all dataset subfolders")
    ap.add_argument("--out",          required=True,
                    help="Output directory for unified dataset")
    ap.add_argument("--split",        nargs=3, type=float, default=[0.70, 0.20, 0.10],
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="Split ratios, must sum to 1.0 (default: 0.70 0.20 0.10)")
    ap.add_argument("--seed",         type=int, default=42,
                    help="Random seed for reproducible shuffling (default: 42)")
    ap.add_argument("--mapping",      default=None,
                    help="Path to mapping override JSON (edit from --dump-mapping output)")
    ap.add_argument("--dump-mapping", default=None, metavar="PATH",
                    help="Write auto-generated mapping JSON to PATH for review, then exit")
    args = ap.parse_args()

    split_ratio = tuple(args.split)
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        ap.error("--split values must sum to 1.0")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATASET UNIFIER v2")
    print("=" * 60)
    print(f"Root  : {args.root}")
    print(f"Output: {args.out}")
    print(f"Split : train={split_ratio[0]:.0%} / val={split_ratio[1]:.0%} / test={split_ratio[2]:.0%}")
    print(f"Seed  : {args.seed}")

    warnings: list[str] = []

    print("\n--- Discovering classes ---")
    dataset_classes = discover_all_classes(Path(args.root), warnings)
    if not dataset_classes:
        print("[ERROR] No class definitions found. Check --root path.")
        return

    mappings = build_mappings(
        dataset_classes,
        override_path=args.mapping,
        dump_path=args.dump_mapping,
        warnings=warnings,
    )

    if args.dump_mapping:
        print("\nReview the mapping file then re-run with --mapping to apply overrides.")
        return

    print("\n--- Processing images & labels ---")
    results = process_datasets(
        Path(args.root), output, split_ratio, args.seed, mappings, warnings)

    write_data_yaml(output)

    write_info_report(
        output,
        cfg={"root": args.root, "output": args.out, "split": split_ratio,
             "seed": args.seed, "mapping_file": args.mapping},
        mappings=mappings,
        dataset_classes=dataset_classes,
        results=results,
        warnings=warnings,
    )

    print("\n" + "=" * 60)
    print(f"DONE!  Written={results['total_written']}  "
          f"Dup={results['total_dup']}  "
          f"NoBoxes={results['total_no_boxes']}  "
          f"Errors={results['total_read_err']}")
    print(f"Output: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
