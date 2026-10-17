"""Leak-free train/val(/test) splits for our YOLO-format folders (spec 7.3).

Why not a random split: consecutive frames of a drive are near-duplicates, so a random split puts
almost the same picture in train and val and inflates the scores. We split by GROUPS of frames:

- RDD2022: no trip information is public, only numbered files per country. ASSUMPTION (unverified):
  neighbouring numbers are neighbouring frames. We cut each country into blocks of `block_size`
  consecutive numbers, give whole blocks to one split, and drop the first `buffer` frames of a
  block whose predecessor went to another split.
- Rome: phone photos and VLC snapshots are timestamped. Frames closer than `cluster_gap` seconds
  form one session; whole sessions go to one split. `vlcsnap-<number>` files use blocks.
- Anything we cannot parse goes to train (never to val/test) and is reported.

Then automated leakage tests run: no two frames of the same sequence closer than the minimum
separation in different splits, and no byte-identical image in two splits (cross-split copies are
removed from the later split). Everything is deterministic given --seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2

from src.data.common import CLASS_NAMES, IMAGE_EXTS, place_image

SPLIT_NAMES = ("train", "val", "test")

PHONE_RE = re.compile(r"^(\d{8})_(\d{6})")
VLC_TS_RE = re.compile(r"^vlcsnap[-_](\d{4})-(\d{2})-(\d{2})-(\d{2})h(\d{2})m(\d{2})s(\d{3})")
VLC_NUM_RE = re.compile(r"^vlcsnap[-_](\d+)$")
TRAILING_NUM_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class Record:
    image_id: str
    source: str
    namespace: str  # frames are only "neighbours" inside the same namespace
    pos: float  # frame number or epoch seconds
    strategy: str  # "block", "cluster" or "fixed_train"
    image_path: Path
    label_path: Path


# ----------------------------------------------------------------------------- parsing


def _epoch(*parts: int) -> float:
    return datetime(*parts, tzinfo=timezone.utc).timestamp()


def parse_rome_stem(stem: str) -> tuple[str, float, str]:
    """(namespace, position, strategy) from a Rome file stem."""
    m = PHONE_RE.match(stem)
    if m:
        d, t = m.groups()
        ts = _epoch(int(d[:4]), int(d[4:6]), int(d[6:]), int(t[:2]), int(t[2:4]), int(t[4:]))
        return "rome:phone", ts, "cluster"
    m = VLC_TS_RE.match(stem)
    if m:
        y, mo, d, h, mi, s, ms = (int(v) for v in m.groups())
        return "rome:vlc-time", _epoch(y, mo, d, h, mi, s) + ms / 1000, "cluster"
    m = VLC_NUM_RE.match(stem)
    if m:
        return "rome:vlc-number", float(int(m.group(1))), "block"
    return "rome:unknown", 0.0, "fixed_train"


def load_records(interim_dir: Path) -> list[Record]:
    """Read meta.csv + the images/labels folders written by our converters."""
    images = {
        p.stem: p for p in (interim_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS
    }
    records: list[Record] = []
    with (interim_dir / "meta.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            image_id, source = row["image_id"], row["source"]
            if image_id not in images:
                continue
            label = interim_dir / "labels" / f"{image_id}.txt"
            if source == "rome":
                ns, pos, strategy = parse_rome_stem(image_id.removeprefix("rome_"))
            else:
                num = TRAILING_NUM_RE.search(image_id)
                ns = f"{source}:{row.get('country') or 'all'}"
                pos, strategy = (float(int(num.group(1))), "block") if num else (0.0, "fixed_train")
            records.append(Record(image_id, source, ns, pos, strategy, images[image_id], label))
    return records


# ----------------------------------------------------------------------------- grouping


def build_groups(
    records: list[Record], block_size: int, cluster_gap: float
) -> dict[tuple[str, int | str], list[Record]]:
    """Group frames that must never be separated (keys: (namespace, group id))."""
    groups: dict[tuple[str, int | str], list[Record]] = defaultdict(list)
    by_ns: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_ns[r.namespace].append(r)
    for ns, items in by_ns.items():
        strategy = items[0].strategy
        if strategy == "fixed_train":
            groups[(ns, "all")].extend(items)
        elif strategy == "block":
            for r in items:
                groups[(ns, int(r.pos // block_size))].append(r)
        else:  # cluster: a new session starts after a gap longer than cluster_gap
            gid, prev = 0, None
            for r in sorted(items, key=lambda x: x.pos):
                if prev is not None and r.pos - prev > cluster_gap:
                    gid += 1
                groups[(ns, gid)].append(r)
                prev = r.pos
    return groups


def assign_splits(
    groups: dict[tuple[str, int | str], list[Record]],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, str]:
    """image_id -> split. Whole groups go to one split; fractions are per namespace."""
    assignment: dict[str, str] = {}
    by_ns: dict[str, list[tuple[int | str, list[Record]]]] = defaultdict(list)
    for (ns, gid), items in groups.items():
        by_ns[ns].append((gid, items))
    rng = random.Random(seed)
    for ns in sorted(by_ns):
        entries = sorted(by_ns[ns], key=lambda e: str(e[0]))
        if entries[0][1][0].strategy == "fixed_train":
            for _, items in entries:
                assignment.update({r.image_id: "train" for r in items})
            continue
        rng.shuffle(entries)
        total = sum(len(items) for _, items in entries)
        want = {"val": round(val_frac * total), "test": round(test_frac * total)}
        have = {"val": 0, "test": 0}
        for _, items in entries:
            split = "train"
            for name in ("val", "test"):
                if have[name] < want[name]:
                    split = name
                    break
            if split != "train":
                have[split] += len(items)
            assignment.update({r.image_id: split for r in items})
    return assignment


def drop_buffer_frames(
    records: list[Record], assignment: dict[str, str], block_size: int, buffer: int
) -> list[str]:
    """Remove the first `buffer` frames of any block whose previous block is in another split."""
    block_split: dict[tuple[str, int], str] = {}
    for r in records:
        if r.strategy == "block" and r.image_id in assignment:
            block_split[(r.namespace, int(r.pos // block_size))] = assignment[r.image_id]
    dropped = []
    for r in records:
        if r.strategy != "block" or r.image_id not in assignment:
            continue
        b = int(r.pos // block_size)
        prev = block_split.get((r.namespace, b - 1))
        if prev is not None and prev != assignment[r.image_id] and r.pos < b * block_size + buffer:
            dropped.append(r.image_id)
    for image_id in dropped:
        del assignment[image_id]
    return sorted(dropped)


# ----------------------------------------------------------------------------- leakage tests


def find_leaks(
    records: list[Record], assignment: dict[str, str], buffer: int, cluster_gap: float
) -> list[str]:
    """Pairs of neighbouring frames (same namespace) that ended up in different splits."""
    by_ns: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        if r.image_id in assignment and r.strategy != "fixed_train":
            by_ns[r.namespace].append(r)
    problems = []
    for ns, items in by_ns.items():
        items.sort(key=lambda r: r.pos)
        min_sep = buffer if items[0].strategy == "block" else cluster_gap
        for a, b in zip(items, items[1:], strict=False):
            if assignment[a.image_id] != assignment[b.image_id] and b.pos - a.pos <= min_sep:
                problems.append(
                    f"{ns}: {a.image_id} ({assignment[a.image_id]}) and {b.image_id} "
                    f"({assignment[b.image_id]}) are only {b.pos - a.pos:g} apart"
                )
    return problems


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drop_cross_split_duplicates(records: list[Record], assignment: dict[str, str]) -> list[str]:
    """Byte-identical images in several splits: keep the earliest split, drop the others."""
    by_hash: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        if r.image_id in assignment:
            by_hash[file_md5(r.image_path)].append(r)
    order = {name: i for i, name in enumerate(SPLIT_NAMES)}
    dropped = []
    for items in by_hash.values():
        splits = {assignment[r.image_id] for r in items}
        if len(splits) < 2:
            continue
        keep = min(splits, key=order.__getitem__)
        dropped += [r.image_id for r in items if assignment[r.image_id] != keep]
    for image_id in dropped:
        del assignment[image_id]
    return sorted(dropped)


# ----------------------------------------------------------------------------- output


def _write_image(src: Path, dst: Path, mode: str, max_side: int | None) -> None:
    if max_side:
        img = cv2.imread(str(src))  # EXIF orientation applied, like the trainer does
        if img is not None and max(img.shape[:2]) > max_side:
            h, w = img.shape[:2]
            scale = max_side / max(h, w)
            img = cv2.resize(
                img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return
    place_image(src, dst, mode)


def _count_boxes(label_path: Path) -> Counter:
    counts: Counter = Counter()
    for line in label_path.read_text().splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
    return counts


def run(
    interim_dirs: list[Path],
    out_dir: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.0,
    seed: int = 0,
    block_size: int = 100,
    buffer: int = 5,
    cluster_gap: float = 60.0,
    resize_max_side: int | None = None,
    mode: str = "auto",
    skip_hash: bool = False,
    overwrite: bool = False,
) -> dict:
    records = [r for d in interim_dirs for r in load_records(d)]
    if not records:
        raise ValueError("no images found in the given --interim folders")

    groups = build_groups(records, block_size, cluster_gap)
    assignment = assign_splits(groups, val_frac, test_frac, seed)
    dropped_buffer = drop_buffer_frames(records, assignment, block_size, buffer)
    dropped_dups = [] if skip_hash else drop_cross_split_duplicates(records, assignment)
    leaks = find_leaks(records, assignment, buffer, cluster_gap)
    if leaks:
        raise RuntimeError("leakage detected:\n" + "\n".join(leaks[:20]))

    if (out_dir / "images").exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already has data; use --overwrite to rebuild it")
        shutil.rmtree(out_dir / "images")
        shutil.rmtree(out_dir / "labels", ignore_errors=True)

    stats: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        split = assignment.get(r.image_id)
        if split is None:
            continue
        ext = r.image_path.suffix.lower()
        _write_image(
            r.image_path, out_dir / "images" / split / f"{r.image_id}{ext}", mode, resize_max_side
        )
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.label_path, out_dir / "labels" / split / f"{r.image_id}.txt")
        stats[f"{split}/{r.source}"]["images"] += 1
        for cls, n in _count_boxes(r.label_path).items():
            stats[f"{split}/{r.source}"][CLASS_NAMES[cls]] += n

    unparsed = sorted(r.image_id for r in records if r.strategy == "fixed_train")
    result = {
        "params": {
            "val_frac": val_frac,
            "test_frac": test_frac,
            "seed": seed,
            "block_size": block_size,
            "buffer": buffer,
            "cluster_gap_s": cluster_gap,
            "resize_max_side": resize_max_side,
        },
        "counts": {k: dict(v) for k, v in sorted(stats.items())},
        "dropped_buffer": dropped_buffer,
        "dropped_duplicates": dropped_dups,
        "unparsed_forced_to_train": unparsed,
        "assignment": dict(sorted(assignment.items())),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--interim", type=Path, nargs="+", required=True, help="converter output folders"
    )
    p.add_argument("--out", type=Path, default=Path("data/processed/yolo_foreign"))
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=100)
    p.add_argument("--buffer", type=int, default=5)
    p.add_argument("--cluster-gap", type=float, default=60.0, help="seconds between sessions")
    p.add_argument("--resize-max-side", type=int, default=None, help="shrink bigger images (px)")
    p.add_argument("--mode", choices=["auto", "copy", "symlink", "hardlink"], default="auto")
    p.add_argument("--skip-hash", action="store_true", help="skip the duplicate-image check")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    res = run(
        a.interim, a.out, a.val_frac, a.test_frac, a.seed, a.block_size, a.buffer,
        a.cluster_gap, a.resize_max_side, a.mode, a.skip_hash, a.overwrite,
    )  # fmt: skip
    print(f"written to {a.out} (splits.json holds the full assignment)")
    for key, c in res["counts"].items():
        print(f"{key:>14}: {dict(c)}")
    print(f"dropped as boundary buffer: {len(res['dropped_buffer'])}")
    print(f"dropped as cross-split duplicates: {len(res['dropped_duplicates'])}")
    print(f"unparsed names forced to train: {len(res['unparsed_forced_to_train'])}")
    print("leakage tests: passed")


if __name__ == "__main__":
    main()
