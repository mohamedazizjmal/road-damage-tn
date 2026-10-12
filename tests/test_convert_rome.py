from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.common import validate_yolo_dir
from src.data.convert_rome import convert_rome, load_names


def _make(root: Path, split: str, stem: str, lines: str) -> None:
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / split / "images" / f"{stem}.jpg"), np.zeros((10, 10, 3), np.uint8))
    (root / split / "labels" / f"{stem}.txt").write_text(lines)


def test_remap_by_name_not_by_id(tmp_path: Path) -> None:
    root, out = tmp_path / "rome", tmp_path / "out"
    root.mkdir()
    # Deliberately unusual id order: 0=manhole, 1=crack, 2=pothole
    (root / "data.yaml").write_text("names: [Manhole, crack, pothole]\n")
    _make(root, "train", "x", "0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.2 0.2\n2 0.6 0.6 0.2 0.2\n")
    _make(root, "valid", "y", "0 0.5 0.5 0.1 0.1\n")  # manhole only -> empty label
    _make(root, "test", "z", "1 0.5 0.5 0.1 0.1 0.2 0.2\n")  # polygon-like line: counted, skipped

    rows = convert_rome(root, out)
    by_id = {r["image_id"]: r for r in rows}

    assert (out / "labels" / "rome_train_x.txt").read_text().splitlines() == [
        "1 0.300000 0.300000 0.200000 0.200000",
        "0 0.600000 0.600000 0.200000 0.200000",
    ]
    assert (out / "labels" / "rome_valid_y.txt").read_text() == ""
    assert by_id["rome_train_x"]["n_dropped"] == 1
    assert by_id["rome_test_z"]["n_malformed"] == 1
    assert by_id["rome_valid_y"]["orig_split"] == "valid"
    assert validate_yolo_dir(out / "images", out / "labels") == []


def test_unknown_class_name_raises(tmp_path: Path) -> None:
    root = tmp_path / "rome"
    root.mkdir()
    (root / "data.yaml").write_text("names:\n  0: pothole\n  1: puddle\n")
    _make(root, "train", "x", "0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="puddle"):
        convert_rome(root, tmp_path / "out")


def test_load_names_dict_form(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").write_text("names:\n  0: Pothole\n  1: crack\n")
    assert load_names(tmp_path) == {0: "pothole", 1: "crack"}
