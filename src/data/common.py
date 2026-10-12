"""Shared helpers for dataset converters (YOLO format, 2-class taxonomy)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2

# Unified taxonomy (spec section 2.1)
CLASS_NAMES: tuple[str, ...] = ("pothole", "crack")
POTHOLE, CRACK = 0, 1
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def xyxy_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Pixel corners -> normalized (xc, yc, w, h). Clamps to the image; None if degenerate."""
    xmin, xmax = sorted((xmin, xmax))
    ymin, ymax = sorted((ymin, ymax))
    xmin, xmax = max(0.0, xmin), min(float(width), xmax)
    ymin, ymax = max(0.0, ymin), min(float(height), ymax)
    if xmax - xmin <= 0 or ymax - ymin <= 0:
        return None
    return (
        (xmin + xmax) / 2 / width,
        (ymin + ymax) / 2 / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def format_yolo_line(cls: int, box: tuple[float, float, float, float]) -> str:
    return f"{cls} " + " ".join(f"{v:.6f}" for v in box)


def read_image_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from the real image file, or None if unreadable."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def place_image(src: Path, dst: Path, mode: str = "copy") -> None:
    """Copy / symlink / hardlink an image into the output folder."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode}")


def validate_yolo_dir(images_dir: Path, labels_dir: Path, n_classes: int = len(CLASS_NAMES)):
    """Data tests from the spec: label per image, normalized coords, valid class ids.

    Returns a list of human-readable problems (empty list = OK).
    """
    problems: list[str] = []
    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in labels_dir.glob("*.txt")}
    for stem in sorted(images.keys() - labels.keys()):
        problems.append(f"image without label file: {stem}")
    for stem in sorted(labels.keys() - images.keys()):
        problems.append(f"label without image: {stem}")
    for path in (labels[s] for s in sorted(labels)):
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{path.name}:{i} expected 5 fields, got {len(parts)}")
                continue
            try:
                cls = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                problems.append(f"{path.name}:{i} non-numeric value")
                continue
            if not 0 <= cls < n_classes:
                problems.append(f"{path.name}:{i} invalid class id {cls}")
            if any(not 0.0 <= v <= 1.0 for v in vals):
                problems.append(f"{path.name}:{i} coordinate outside [0, 1]")
    return problems
