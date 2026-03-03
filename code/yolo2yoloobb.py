#!/usr/bin/env python3
"""
Convert YOLO (axis-aligned) labels:  class cx cy w h
to YOLO-OBB (4 corners) labels:      class x1 y1 x2 y2 x3 y3 x4 y4
Assumes angle = 0 degrees (no rotation).

Input/output coordinates are normalized (0..1), same as standard YOLO.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def yolo_bbox_to_corners(cx: float, cy: float, w: float, h: float, clip: bool) -> Tuple[float, ...]:
    # Axis-aligned corners for 0° rotation
    left   = cx - w / 2.0
    right  = cx + w / 2.0
    top    = cy - h / 2.0
    bottom = cy + h / 2.0

    pts = (
        left,  top,      # x1 y1 (top-left)
        right, top,      # x2 y2 (top-right)
        right, bottom,   # x3 y3 (bottom-right)
        left,  bottom,   # x4 y4 (bottom-left)
    )

    if clip:
        pts = tuple(clamp01(p) for p in pts)

    return pts


def convert_label_file(in_file: Path, out_file: Path, clip: bool, precision: int, with_angle: bool) -> None:
    out_lines: List[str] = []

    raw = in_file.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(raw, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            # Skip malformed lines quietly (or raise if you prefer)
            continue

        cls = parts[0]
        try:
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            # Skip lines that can't be parsed
            continue

        corners = yolo_bbox_to_corners(cx, cy, w, h, clip=clip)

        fmt = f"{{:.{precision}f}}"
        corner_str = " ".join(fmt.format(v) for v in corners)

        if with_angle:
            # Some formats want an explicit angle; here it is always 0
            out_lines.append(f"{cls} {corner_str} 0")
        else:
            out_lines.append(f"{cls} {corner_str}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def iter_txt_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(path.glob(pattern))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="Input .txt file or directory containing YOLO labels")
    ap.add_argument("output", type=str, help="Output .txt file or directory for OBB labels")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders when input is a directory")
    ap.add_argument("--clip", action="store_true", help="Clamp output coords into [0,1]")
    ap.add_argument("--precision", type=int, default=6, help="Decimal precision for output floats (default: 6)")
    ap.add_argument("--with-angle", action="store_true", help="Append an explicit angle=0 at end of each line")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    in_files = iter_txt_files(in_path, recursive=args.recursive)

    # If converting a single file -> output should be a file path
    if in_path.is_file():
        convert_label_file(in_path, out_path, clip=args.clip, precision=args.precision, with_angle=args.with_angle)
        return

    # Directory mode: mirror relative structure into output directory
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    for f in in_files:
        rel = f.relative_to(in_path)
        out_f = out_path / rel
        convert_label_file(f, out_f, clip=args.clip, precision=args.precision, with_angle=args.with_angle)


if __name__ == "__main__":
    main()
