"""Evaluate trained weights on a split and save metrics as JSON (mAP, per-class P/R/AP).

    python -m src.eval.evaluate --weights runs/baseline_a/weights/best.pt \
        --data runs/baseline_a_config/data.yaml --split val --out reports/baseline_a_val.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.common import CLASS_NAMES


def summarize(box, names=CLASS_NAMES) -> dict:
    """Turn an Ultralytics `metrics.box` object into plain numbers."""
    per_class = {}
    for i, cls_idx in enumerate(box.ap_class_index):
        per_class[names[int(cls_idx)]] = {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]),
            "ap50_95": float(box.ap[i]),
        }
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="data YAML with an absolute path")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=Path("reports/eval.json"))
    a = p.parse_args()

    from ultralytics import YOLO

    metrics = YOLO(str(a.weights)).val(
        data=str(a.data), split=a.split, imgsz=a.imgsz, device=a.device, plots=True
    )
    result = {"weights": str(a.weights), "split": a.split, **summarize(metrics.box)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
