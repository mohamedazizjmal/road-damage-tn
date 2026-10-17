"""RDD2022 (Pascal VOC XML, per country) -> YOLO txt with the 2-class taxonomy.

Mapping (spec 8.1): D00, D10, D20 -> crack (1); D40 -> pothole (0); every other code is ignored
(and counted, so nothing disappears silently).

Expected layout (verify against your download):
    <root>/[RDD2022/]<Country>/train/images/*.jpg
    <root>/[RDD2022/]<Country>/train/annotations/xmls/*.xml
Test folders have no public annotations and are skipped.
"""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.data.common import (
    CRACK,
    IMAGE_EXTS,
    POTHOLE,
    format_yolo_line,
    place_image,
    read_image_size,
    xyxy_to_yolo,
)

RDD_CLASS_MAP: dict[str, int] = {"D00": CRACK, "D10": CRACK, "D20": CRACK, "D40": POTHOLE}
META_FIELDS = [
    "image_id",
    "source",
    "country",
    "width",
    "height",
    "n_pothole",
    "n_crack",
    "n_ignored",
    "n_degenerate",
]


@dataclass
class VocObject:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def parse_voc_xml(path: Path) -> tuple[tuple[int, int] | None, list[VocObject]]:
    """Return (declared (w, h) or None, objects)."""
    root = ET.parse(path).getroot()
    size = None
    size_el = root.find("size")
    if size_el is not None:
        try:
            size = (int(float(size_el.findtext("width"))), int(float(size_el.findtext("height"))))
        except (TypeError, ValueError):
            size = None
    objects = []
    for obj in root.iter("object"):
        box = obj.find("bndbox")
        name = (obj.findtext("name") or "").strip()
        if box is None or not name:
            continue
        objects.append(
            VocObject(
                name,
                float(box.findtext("xmin")),
                float(box.findtext("ymin")),
                float(box.findtext("xmax")),
                float(box.findtext("ymax")),
            )
        )
    return size, objects


def find_country_dirs(root: Path) -> list[Path]:
    """Folders that contain `train/annotations/xmls`, at any depth (handles Czech/Czech/train)."""
    found = {x.parent.parent.parent for x in root.rglob("xmls") if x.parent.parent.name == "train"}
    return sorted(found)


def convert_rdd2022(root: Path, out_dir: Path, mode: str = "copy", limit: int | None = None):
    """Convert all countries. Returns (rows, ignored_code_counter, size_mismatches)."""
    out_images, out_labels = out_dir / "images", out_dir / "labels"
    out_labels.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    ignored_codes: Counter[str] = Counter()
    size_mismatches = 0

    countries = find_country_dirs(root)
    if not countries:
        raise FileNotFoundError(f"no <Country>/train/annotations/xmls folder found under {root}")

    for country_dir in countries:
        country = country_dir.name
        xml_dir = country_dir / "train" / "annotations" / "xmls"
        img_dir = country_dir / "train" / "images"
        for n, xml_path in enumerate(sorted(xml_dir.glob("*.xml"))):
            if limit is not None and n >= limit:
                break
            img_path = next(
                (
                    img_dir / f"{xml_path.stem}{e}"
                    for e in IMAGE_EXTS
                    if (img_dir / f"{xml_path.stem}{e}").exists()
                ),
                None,
            )
            if img_path is None:
                continue  # annotation without image: skip
            declared, objects = parse_voc_xml(xml_path)
            real = read_image_size(img_path)
            if real is None:
                continue  # unreadable image: skip
            if declared is not None and declared != real:
                size_mismatches += 1
            width, height = real  # trust the real image, not the XML

            lines, counts = [], Counter()
            for o in objects:
                cls = RDD_CLASS_MAP.get(o.name)
                if cls is None:
                    ignored_codes[o.name] += 1
                    counts["ignored"] += 1
                    continue
                box = xyxy_to_yolo(o.xmin, o.ymin, o.xmax, o.ymax, width, height)
                if box is None:
                    counts["degenerate"] += 1
                    continue
                lines.append(format_yolo_line(cls, box))
                counts[cls] += 1

            image_id = f"{country}_{xml_path.stem}"
            place_image(img_path, out_images / f"{image_id}{img_path.suffix.lower()}", mode)
            (out_labels / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            rows.append(
                {
                    "image_id": image_id,
                    "source": "rdd2022",
                    "country": country,
                    "width": width,
                    "height": height,
                    "n_pothole": counts[POTHOLE],
                    "n_crack": counts[CRACK],
                    "n_ignored": counts["ignored"],
                    "n_degenerate": counts["degenerate"],
                }
            )

    with (out_dir / "meta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, ignored_codes, size_mismatches


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="folder containing the country dirs")
    p.add_argument("--out", type=Path, default=Path("data/interim/rdd2022"))
    p.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    p.add_argument("--limit", type=int, default=None, help="max images per country (smoke tests)")
    args = p.parse_args()
    rows, ignored, mismatches = convert_rdd2022(args.root, args.out, args.mode, args.limit)
    print(f"converted {len(rows)} images -> {args.out}")
    print(f"ignored class codes (not damage classes we use): {dict(ignored)}")
    print(f"images whose XML size differed from the real image size: {mismatches}")


if __name__ == "__main__":
    main()
