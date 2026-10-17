"""CPU smoke test: train 1 epoch on ~20 synthetic images to check the whole pipeline wiring.

    python -m src.train.smoke        (needs: pip install -e ".[train]")

Uses a model built from scratch (yolov8n.yaml): no weight download, results are meaningless.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data.common import CLASS_NAMES


def make_tiny_dataset(root: Path, n_train: int = 16, n_val: int = 4, size: int = 64) -> Path:
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
            x0, y0 = rng.integers(8, 28, 2)
            cv2.rectangle(
                img, (int(x0), int(y0)), (int(x0) + 20, int(y0) + 20), (255, 255, 255), -1
            )
            cv2.imwrite(str(root / "images" / split / f"s{i}.jpg"), img)
            xc, yc, w = (x0 + 10) / size, (y0 + 10) / size, 20 / size
            (root / "labels" / split / f"s{i}.txt").write_text(
                f"{i % 2} {xc:.4f} {yc:.4f} {w:.4f} {w:.4f}\n"
            )
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return yaml_path


def main() -> None:
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        data_yaml = make_tiny_dataset(Path(tmp) / "tiny")
        model = YOLO("yolov8n.yaml")
        model.train(
            data=str(data_yaml), epochs=1, imgsz=64, batch=4, workers=0, device="cpu",
            plots=False, project=str(Path(tmp) / "runs"), name="smoke", pretrained=False,
            amp=False, verbose=False,
        )  # fmt: skip
        last = Path(tmp) / "runs" / "smoke" / "weights" / "last.pt"
        assert last.exists(), "training did not produce weights"
    print("smoke test OK")


if __name__ == "__main__":
    main()
