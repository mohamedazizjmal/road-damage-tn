"""Train a YOLO detector from a YAML config (experiments A, B, C all use this script).

    python -m src.train.train --config configs/train_baseline_a.yaml \
        --data-root /path/to/yolo_foreign

Command-line options override the config (--epochs, --batch, --device, --name, --fraction...).
The data YAML is rewritten with an ABSOLUTE dataset path before training, because Ultralytics
resolves relative paths against its own datasets folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

TRAIN_KEYS = (
    "imgsz", "epochs", "batch", "patience", "seed", "workers", "device",
    "project", "name", "fraction", "exist_ok", "plots", "cache",
)  # fmt: skip


def load_config(path: Path, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for required in ("data", "model"):
        if required not in cfg:
            raise KeyError(f"config {path} needs a '{required}' entry")
    return cfg


def resolve_data_yaml(data_yaml: Path, data_root: Path | None, out_path: Path) -> Path:
    """Write a copy of the data YAML with an absolute `path` (and no missing splits)."""
    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data_root or data["path"]).expanduser().resolve()
    if not (root / "images" / "train").is_dir():
        raise FileNotFoundError(f"{root / 'images' / 'train'} not found (run make_splits first?)")
    data["path"] = str(root)
    for split in ("val", "test"):
        if split in data and not (root / data[split]).is_dir():
            if split == "val":
                raise FileNotFoundError(f"{root / data['val']} not found")
            del data[split]  # no test split built yet: leave it out instead of crashing
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return out_path


def train(cfg: dict, data_yaml: Path):
    from ultralytics import YOLO  # imported here so tests and linting do not need torch

    model = YOLO(cfg["model"])
    kwargs = {k: cfg[k] for k in TRAIN_KEYS if k in cfg}
    if "project" in kwargs:  # absolute path, otherwise Ultralytics nests it under runs/detect/
        kwargs["project"] = str(Path(kwargs["project"]).resolve())
    return model.train(data=str(data_yaml), **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None, help="folder with images/ and labels/")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None, help="e.g. 0 or cpu")
    p.add_argument("--name", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--fraction", type=float, default=None, help="use only this share of train")
    p.add_argument("--model", default=None, help="weights to start from (.pt or .yaml)")
    a = p.parse_args()

    keys = ("epochs", "batch", "imgsz", "device", "name", "project", "fraction", "model")
    overrides = {k: getattr(a, k) for k in keys}
    cfg = load_config(a.config, overrides)
    out_dir = Path(cfg.get("project", "runs")) / f"{cfg.get('name', 'run')}_config"
    data_yaml = resolve_data_yaml(Path(cfg["data"]), a.data_root, out_dir / "data.yaml")
    (out_dir / "train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"data yaml: {data_yaml}")
    train(cfg, data_yaml)


if __name__ == "__main__":
    main()
