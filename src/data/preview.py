"""Draw YOLO labels on a few random images, to check conversions with your own eyes.

Usage: python -m src.data.preview --dir data/interim/rome --n 8 --out reports/preview_rome
Images are read like the trainer reads them (EXIF orientation applied), so a box in the wrong
place here means the label would be wrong in training too.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from src.data.common import CLASS_NAMES, IMAGE_EXTS

COLORS = {0: (0, 0, 255), 1: (255, 128, 0)}  # BGR: pothole red, crack blue


def draw_labels(image_path: Path, label_path: Path, max_side: int = 1280):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"cannot read {image_path}")
    h, w = img.shape[:2]
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = (float(v) for v in parts[1:])
            p1 = (int((xc - bw / 2) * w), int((yc - bh / 2) * h))
            p2 = (int((xc + bw / 2) * w), int((yc + bh / 2) * h))
            color = COLORS.get(cls, (0, 255, 0))
            cv2.rectangle(img, p1, p2, color, max(2, w // 400))
            cv2.putText(
                img,
                CLASS_NAMES[cls],
                (p1[0], max(15, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, w / 1500),
                color,
                2,
            )
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def preview(data_dir: Path, out_dir: Path, n: int = 8, seed: int = 0, only_labeled: bool = True):
    images = sorted(p for p in (data_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if only_labeled:
        images = [p for p in images if (data_dir / "labels" / f"{p.stem}.txt").read_text().strip()]
    random.Random(seed).shuffle(images)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for p in images[:n]:
        out = out_dir / f"{p.stem}.jpg"
        cv2.imwrite(str(out), draw_labels(p, data_dir / "labels" / f"{p.stem}.txt"))
        written.append(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="folder with images/ and labels/")
    ap.add_argument("--out", type=Path, default=Path("reports/preview"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    files = preview(args.dir, args.out, args.n, args.seed)
    print(f"wrote {len(files)} preview images to {args.out}")


if __name__ == "__main__":
    main()
