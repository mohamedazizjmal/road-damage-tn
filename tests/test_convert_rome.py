import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.common import validate_yolo_dir
from src.data.convert_rome import convert_coco, list_classes, parse_class_map
from src.data.preview import preview


def _make(tmp: Path, categories: list[dict], declared=(100, 50)):
    images = tmp / "images"
    images.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(images / name), np.zeros((50, 100, 3), np.uint8))  # 100 wide, 50 high
    coco = {
        "categories": categories,
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": declared[0], "height": declared[1]},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 50},
            {"id": 3, "file_name": "ghost.jpg", "width": 100, "height": 50},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20]},  # Pothole
            {"id": 2, "image_id": 1, "category_id": 5, "bbox": [0, 0, 100, 50]},  # Crack
            {"id": 3, "image_id": 1, "category_id": 9, "bbox": [50, 10, 10, 10]},  # Manhole
            {"id": 4, "image_id": 1, "category_id": 7, "bbox": [30, 30, 0, 5]},  # degenerate
            {"id": 5, "image_id": 2, "category_id": 9, "bbox": [1, 1, 5, 5]},  # manhole only
        ],
    }
    path = tmp / "annotations_coco.json"
    path.write_text(json.dumps(coco))
    return path, images


CATS = [
    {"id": 5, "name": "Crack"},
    {"id": 7, "name": "Pothole"},
    {"id": 9, "name": "Manhole"},
]


def test_coco_conversion(tmp_path: Path) -> None:
    jpath, images = _make(tmp_path / "rome", CATS)
    out = tmp_path / "out"
    rows, missing = convert_coco(jpath, images, out)

    assert (out / "labels" / "rome_a.txt").read_text().splitlines() == [
        "0 0.200000 0.400000 0.200000 0.400000",
        "1 0.500000 0.500000 1.000000 1.000000",
    ]
    assert (out / "labels" / "rome_b.txt").read_text() == ""
    by_id = {r["image_id"]: r for r in rows}
    assert by_id["rome_a"]["n_dropped"] == 1 and by_id["rome_a"]["n_degenerate"] == 1
    assert by_id["rome_a"]["size_mismatch"] == 0
    assert missing == ["ghost.jpg"]
    assert validate_yolo_dir(out / "images", out / "labels") == []


def test_size_mismatch_is_counted(tmp_path: Path) -> None:
    jpath, images = _make(tmp_path / "rome", CATS, declared=(50, 100))  # json says rotated
    rows, _ = convert_coco(jpath, images, tmp_path / "out")
    assert {r["image_id"]: r["size_mismatch"] for r in rows}["rome_a"] == 1


def test_unknown_class_raises_and_override_works(tmp_path: Path) -> None:
    cats = [*CATS[:2], {"id": 9, "name": "Puddle"}]
    jpath, images = _make(tmp_path / "rome", cats)
    with pytest.raises(ValueError, match="puddle"):
        convert_coco(jpath, images, tmp_path / "out")
    rows, _ = convert_coco(
        jpath,
        images,
        tmp_path / "out2",
        class_map=parse_class_map("Pothole=pothole,Crack=crack,Puddle=drop"),
    )
    assert len(rows) == 2


def test_parse_class_map_rejects_bad_target() -> None:
    with pytest.raises(ValueError):
        parse_class_map("Pothole=hole")


def test_list_classes(tmp_path: Path) -> None:
    jpath, _ = _make(tmp_path / "rome", CATS)
    assert list_classes(jpath) == {"Crack": 1, "Pothole": 2, "Manhole": 2}


def test_preview_writes_images(tmp_path: Path) -> None:
    jpath, images = _make(tmp_path / "rome", CATS)
    out = tmp_path / "out"
    convert_coco(jpath, images, out)
    files = preview(out, tmp_path / "prev", n=5)
    assert [f.name for f in files] == ["rome_a.jpg"]  # only labeled images are previewed
    assert files[0].exists()
