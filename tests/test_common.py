from pathlib import Path

import cv2
import numpy as np

from src.data.common import format_yolo_line, validate_yolo_dir, xyxy_to_yolo


def test_xyxy_to_yolo_basic() -> None:
    assert xyxy_to_yolo(10, 10, 30, 30, 100, 50) == (0.2, 0.4, 0.2, 0.4)


def test_xyxy_to_yolo_clamps_and_rejects_degenerate() -> None:
    assert xyxy_to_yolo(-10, -10, 200, 80, 100, 50) == (0.5, 0.5, 1.0, 1.0)
    assert xyxy_to_yolo(20, 20, 20, 40, 100, 50) is None
    assert xyxy_to_yolo(150, 10, 180, 30, 100, 50) is None  # entirely outside


def test_xyxy_to_yolo_swapped_corners() -> None:
    assert xyxy_to_yolo(30, 30, 10, 10, 100, 50) == (0.2, 0.4, 0.2, 0.4)


def test_format_yolo_line() -> None:
    assert format_yolo_line(1, (0.5, 0.5, 1.0, 1.0)) == "1 0.500000 0.500000 1.000000 1.000000"


def _img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((10, 10, 3), dtype=np.uint8))


def test_validate_yolo_dir(tmp_path: Path) -> None:
    images, labels = tmp_path / "images", tmp_path / "labels"
    labels.mkdir()
    for stem in ("ok", "nolabel", "badclass", "badcoord"):
        _img(images / f"{stem}.jpg")
    (labels / "ok.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (labels / "badclass.txt").write_text("5 0.5 0.5 0.2 0.2\n")
    (labels / "badcoord.txt").write_text("0 0.5 0.5 1.2 0.2\n")
    (labels / "orphan.txt").write_text("")
    problems = " | ".join(validate_yolo_dir(images, labels))
    assert "image without label file: nolabel" in problems
    assert "label without image: orphan" in problems
    assert "invalid class id 5" in problems
    assert "coordinate outside" in problems
    assert "ok" not in problems.replace("coordinate outside", "")
