from pathlib import Path

import pytest
import yaml

from src.eval.evaluate import summarize
from src.train.smoke import make_tiny_dataset
from src.train.train import load_config, resolve_data_yaml


def test_load_config_overrides(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("data: d.yaml\nmodel: m.pt\nepochs: 50\n")
    cfg = load_config(p, {"epochs": 2, "batch": None})
    assert cfg["epochs"] == 2 and "batch" not in cfg


def test_load_config_requires_data_and_model(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("epochs: 1\n")
    with pytest.raises(KeyError):
        load_config(p)


def test_resolve_data_yaml_makes_path_absolute_and_drops_missing_test(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    make_tiny_dataset(root)
    template = tmp_path / "data.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "path": "somewhere/else",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "pothole", 1: "crack"},
            }
        )  # fmt: skip
    )
    out = resolve_data_yaml(template, root, tmp_path / "run" / "data.yaml")
    data = yaml.safe_load(out.read_text())
    assert Path(data["path"]).is_absolute() and Path(data["path"]) == root.resolve()
    assert "test" not in data  # images/test does not exist yet
    assert data["names"] == {0: "pothole", 1: "crack"}


def test_resolve_data_yaml_missing_dataset(tmp_path: Path) -> None:
    template = tmp_path / "data.yaml"
    template.write_text("path: nope\ntrain: images/train\nval: images/val\n")
    with pytest.raises(FileNotFoundError):
        resolve_data_yaml(template, None, tmp_path / "o.yaml")


def test_repo_configs_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "train_baseline_a.yaml")
    data = yaml.safe_load((root / cfg["data"]).read_text())
    assert data["names"] == {0: "pothole", 1: "crack"}


def test_summarize_with_fake_metrics() -> None:
    class Box:
        ap_class_index = [0, 1]
        p, r, ap50, ap = [0.5, 0.6], [0.4, 0.7], [0.45, 0.65], [0.2, 0.3]
        map50, map, mp, mr = 0.55, 0.25, 0.55, 0.55

    out = summarize(Box())
    assert out["per_class"]["crack"]["ap50"] == 0.65 and out["map50"] == 0.55
