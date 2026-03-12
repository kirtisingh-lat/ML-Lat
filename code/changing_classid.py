"""
Change multiple class IDs in YOLO .txt label files.

Supports any structure — labels are found recursively:

  Flat:
    labels/foo.txt

  Labels as root with splits inside:
    labels/train/foo.txt
    labels/val/foo.txt

  Dataset root with labels inside splits:        <-- pass dataset root here
    dataset/train/labels/foo.txt
    dataset/val/labels/foo.txt
    dataset/test/labels/foo.txt

Output mirrors the input structure under a new directory so originals are preserved.

Usage:
    # Remap class 0->2 and 5->0; pass dataset root (contains train/val/test)
    python changing_classid.py --dir path/to/dataset --map 0:2 5:0

    # Or pass a labels dir directly
    python changing_classid.py --dir path/to/labels --map 0:2 1:3 5:0

    # Custom output location
    python changing_classid.py --dir path/to/dataset --map 0:1 --out path/to/output
"""

import argparse
import shutil
from pathlib import Path


def remap_file(src: Path, dst: Path, mapping: dict[int, int]):
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        parts[0] = str(mapping.get(cls, cls))
        out_lines.append(" ".join(parts))
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Remap YOLO class IDs in label files.")
    parser.add_argument("--dir", required=True, help="Root labels directory")
    parser.add_argument(
        "--map", nargs="+", required=True, metavar="OLD:NEW",
        help="One or more ID remappings, e.g.  0:2  1:3  5:0",
    )
    parser.add_argument("--out", default=None, help="Output directory (default: <dir>_updated)")
    args = parser.parse_args()

    # Parse mapping
    mapping: dict[int, int] = {}
    for entry in args.map:
        old, new = entry.split(":")
        mapping[int(old)] = int(new)

    src_root = Path(args.dir)
    dst_root = Path(args.out) if args.out else src_root.parent / (src_root.name + "_updated")

    print(f"Source : {src_root}")
    print(f"Output : {dst_root}")
    print(f"Mapping: {mapping}")

    # Collect all .txt files (skip classes.txt / obj.names style files)
    SKIP_NAMES = {"classes.txt", "obj.names", "_darknet.labels", "predefined_classes.txt"}
    txt_files = [f for f in src_root.rglob("*.txt") if f.name not in SKIP_NAMES]

    if not txt_files:
        print("No label .txt files found.")
        return

    changed = skipped = 0
    for src in txt_files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        remap_file(src, dst, mapping)

        # Count changed vs unchanged boxes
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and int(parts[0]) in mapping:
                changed += 1
            elif parts:
                skipped += 1

    # Copy non-label files (classes.txt etc.) as-is
    for f in src_root.rglob("*"):
        if f.is_file() and f.suffix != ".txt":
            rel = f.relative_to(src_root)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)

    print(f"\nDone.")
    print(f"  Files processed : {len(txt_files)}")
    print(f"  Boxes remapped  : {changed}")
    print(f"  Boxes unchanged : {skipped}")
    print(f"  Updated labels written to: {dst_root}")


if __name__ == "__main__":
    main()
