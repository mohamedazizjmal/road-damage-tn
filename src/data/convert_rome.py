"""Rome/Sacrofano road damage dataset (already YOLO) -> our 2-class taxonomy.

Class ids are remapped by NAME using the dataset's own data.yaml, never by assumed id order.
pothole -> 0, crack -> 1, manhole -> dropped (manholes become background, spec 2.1).
An unknown class name raises an error instead of being silently dropped.

Layout assumption (verify on the real download): any `.../labels/*.txt` with a matching image in
the sibling `.../images/` folder; a data.yaml with `names` somewhere under the root.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml

from src.data.common import CRACK, IMAGE_EXTS, POTHOLE, format_yolo_line, place_image

ROME_CLASS_MAP: dict[str, int] = {"pothole": POTHOLE, "crack": CRACK}
ROME_DROPPED: set[str] = {"manhole"}
SPLIT_NAMES = {"train", "valid", "val", "test"}
META_FIELDS = [
    "image_id",
    "source",
    "orig_split",
    "n_pothole",
    "n_crack",
    "n_dropped",
    "n_malformed",
]


def load_names(root: Path) -> dict[int, str]:
    candidates = [root / "data.yaml", *sorted(root.rglob("data.yaml"))]
    yaml_path = next((c for c in candidates if c.exists()), None)
    if yaml_path is None:
        raise FileNotFoundError(f"no data.yaml found under {root}")
    names = yaml.safe_load(yaml_path.read_text())["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): str(v).strip().lower() for k, v in names.items()}


def find_pairs(root: Path):
    """Yield (orig_split, label_path, image_path) for every label with an image."""
    for txt in sorted(root.rglob("*.txt")):
        if txt.parent.name != "labels":
            continue
        img_dir = txt.parent.parent / "images"
        img = next(
            (
                img_dir / f"{txt.stem}{e}"
                for e in IMAGE_EXTS
                if (img_dir / f"{txt.stem}{e}").exists()
            ),
            None,
        )
        if img is None:
            continue
        split_dir = txt.parent.parent
        orig_split = split_dir.name if split_dir.name.lower() in SPLIT_NAMES else "all"
        yield orig_split, txt, img


def convert_rome(root: Path, out_dir: Path, mode: str = "copy"):
    names = load_names(root)
    unknown = set(names.values()) - set(ROME_CLASS_MAP) - ROME_DROPPED
    if unknown:
        raise ValueError(f"unknown class names in data.yaml: {sorted(unknown)} (decide a mapping)")

    out_images, out_labels = out_dir / "images", out_dir / "labels"
    out_labels.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for orig_split, label_path, img_path in find_pairs(root):
        lines, counts = [], Counter()
        for raw in label_path.read_text().splitlines():
            parts = raw.split()
            if not parts:
                continue
            if len(parts) != 5:  # e.g. polygon lines: not handled, counted, not guessed
                counts["malformed"] += 1
                continue
            name = names.get(int(parts[0]))
            if name in ROME_DROPPED:
                counts["dropped"] += 1
                continue
            if name is None:
                raise ValueError(f"{label_path}: class id {parts[0]} not in data.yaml")
            cls = ROME_CLASS_MAP[name]
            lines.append(format_yolo_line(cls, tuple(float(v) for v in parts[1:])))
            counts[cls] += 1

        image_id = f"rome_{orig_split}_{label_path.stem}"
        place_image(img_path, out_images / f"{image_id}{img_path.suffix.lower()}", mode)
        (out_labels / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append(
            {
                "image_id": image_id,
                "source": "rome",
                "orig_split": orig_split,
                "n_pothole": counts[POTHOLE],
                "n_crack": counts[CRACK],
                "n_dropped": counts["dropped"],
                "n_malformed": counts["malformed"],
            }
        )

    with (out_dir / "meta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/interim/rome"))
    p.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    args = p.parse_args()
    rows = convert_rome(args.root, args.out, args.mode)
    print(f"converted {len(rows)} images -> {args.out}")
    print(f"manhole boxes dropped: {sum(r['n_dropped'] for r in rows)}")
    print(f"malformed (non-5-field) lines skipped: {sum(r['n_malformed'] for r in rows)}")


if __name__ == "__main__":
    main()
