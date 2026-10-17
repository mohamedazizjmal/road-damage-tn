"""COCO json (Rome/Sacrofano dataset) -> YOLO txt with our 2-class taxonomy.

The Rome download ships `annotations_coco.json` + `images/` (no YOLO labels), so we convert from
COCO ourselves. Class ids are remapped by NAME read from the json, never by assumed id order.
Default: pothole -> 0, crack -> 1, manhole -> dropped (becomes background, spec 2.1).
An unknown class name raises an error (nothing is dropped silently); override with --class-map.

COCO bbox = [x, y, width, height] in pixels, in the coordinate system declared by the json image
entry. We normalize with that declared size and count images whose declared size differs from the
size OpenCV reads (a sign of EXIF rotation on phone photos): check them with src.data.preview.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.data.common import (
    CRACK,
    POTHOLE,
    format_yolo_line,
    place_image,
    read_image_size,
    xyxy_to_yolo,
)

DEFAULT_CLASS_MAP: dict[str, int | None] = {"pothole": POTHOLE, "crack": CRACK, "manhole": None}
TARGETS: dict[str, int | None] = {"pothole": POTHOLE, "crack": CRACK, "drop": None}
META_FIELDS = [
    "image_id",
    "source",
    "width",
    "height",
    "n_pothole",
    "n_crack",
    "n_dropped",
    "n_degenerate",
    "size_mismatch",
]


def parse_class_map(text: str) -> dict[str, int | None]:
    """'Pothole=pothole,Crack=crack,Manhole=drop' -> {'pothole': 0, 'crack': 1, 'manhole': None}."""
    result: dict[str, int | None] = {}
    for item in text.split(","):
        name, _, target = item.partition("=")
        target = target.strip().lower()
        if target not in TARGETS:
            raise ValueError(f"bad target '{target}' in --class-map (use pothole, crack or drop)")
        result[name.strip().lower()] = TARGETS[target]
    return result


def list_classes(json_path: Path) -> dict[str, int]:
    """Category name -> number of annotations (for a quick look before converting)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    counts = Counter(
        names.get(a["category_id"], f"<id {a['category_id']}>") for a in data["annotations"]
    )
    return {n: counts.get(n, 0) for n in names.values()} | {
        k: v for k, v in counts.items() if k not in names.values()
    }


def convert_coco(
    json_path: Path,
    images_dir: Path,
    out_dir: Path,
    source: str = "rome",
    class_map: dict[str, int | None] | None = None,
    mode: str = "copy",
):
    """Returns (rows, missing_image_files)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    cmap = {k.strip().lower(): v for k, v in (class_map or DEFAULT_CLASS_MAP).items()}
    cats = {c["id"]: c["name"].strip().lower() for c in data["categories"]}
    unknown = sorted(set(cats.values()) - set(cmap))
    if unknown:
        raise ValueError(
            f"class names without a mapping: {unknown}. Use --class-map "
            f"'name=pothole|crack|drop,...' (run with --list-classes to see all names)."
        )

    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)

    out_images, out_labels = out_dir / "images", out_dir / "labels"
    out_labels.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    missing: list[str] = []

    for im in data["images"]:
        file_name = Path(im["file_name"]).name
        img_path = images_dir / file_name
        if not img_path.exists():
            missing.append(file_name)
            continue
        real = read_image_size(img_path)
        if real is None:
            missing.append(file_name)
            continue
        declared = (
            (int(im["width"]), int(im["height"])) if im.get("width") and im.get("height") else None
        )
        width, height = declared or real  # boxes live in the json's coordinate system

        lines, counts = [], Counter()
        for ann in by_image.get(im["id"], []):
            cls = cmap[cats[ann["category_id"]]]
            if cls is None:
                counts["dropped"] += 1
                continue
            x, y, w, h = ann["bbox"]
            box = xyxy_to_yolo(x, y, x + w, y + h, width, height)
            if box is None:
                counts["degenerate"] += 1
                continue
            lines.append(format_yolo_line(cls, box))
            counts[cls] += 1

        image_id = f"{source}_{Path(file_name).stem}"
        place_image(img_path, out_images / f"{image_id}{img_path.suffix.lower()}", mode)
        (out_labels / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append(
            {
                "image_id": image_id,
                "source": source,
                "width": width,
                "height": height,
                "n_pothole": counts[POTHOLE],
                "n_crack": counts[CRACK],
                "n_dropped": counts["dropped"],
                "n_degenerate": counts["degenerate"],
                "size_mismatch": int(declared is not None and declared != real),
            }
        )

    with (out_dir / "meta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, missing


def find_json(root: Path) -> Path:
    direct = root / "annotations_coco.json"
    if direct.exists():
        return direct
    found = sorted(root.rglob("annotations_coco.json"))
    if not found:
        raise FileNotFoundError(f"annotations_coco.json not found under {root}")
    return found[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--root", type=Path, required=True, help="folder containing annotations_coco.json"
    )
    p.add_argument("--out", type=Path, default=Path("data/interim/rome"))
    p.add_argument(
        "--images", type=Path, default=None, help="images folder (default: <json folder>/images)"
    )
    p.add_argument(
        "--class-map",
        type=str,
        default=None,
        help="e.g. 'Pothole=pothole,Crack=crack,Manhole=drop'",
    )
    p.add_argument(
        "--list-classes", action="store_true", help="print category names and counts, then exit"
    )
    p.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    args = p.parse_args()

    json_path = find_json(args.root)
    if args.list_classes:
        for name, n in list_classes(json_path).items():
            print(f"{name}: {n} annotations")
        return
    cmap = parse_class_map(args.class_map) if args.class_map else None
    images_dir = args.images or json_path.parent / "images"
    rows, missing = convert_coco(json_path, images_dir, args.out, class_map=cmap, mode=args.mode)
    print(f"converted {len(rows)} images -> {args.out}")
    print(
        f"potholes: {sum(r['n_pothole'] for r in rows)}, cracks: {sum(r['n_crack'] for r in rows)}"
    )
    print(f"dropped boxes (e.g. manholes): {sum(r['n_dropped'] for r in rows)}")
    n_mismatch = sum(r["size_mismatch"] for r in rows)
    print(f"images whose json size differs from the real image size: {n_mismatch}")
    print(f"images listed in json but missing/unreadable on disk: {len(missing)}")


if __name__ == "__main__":
    main()
