import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.make_splits import (
    Record,
    assign_splits,
    build_groups,
    drop_buffer_frames,
    drop_cross_split_duplicates,
    find_leaks,
    load_records,
    parse_rome_stem,
    run,
)


def _rec(image_id: str, ns: str, pos: float, strategy: str, tmp: Path | None = None) -> Record:
    p = tmp or Path(".")
    return Record(
        image_id, ns.split(":")[0], ns, pos, strategy, p / f"{image_id}.jpg", p / f"{image_id}.txt"
    )


def test_parse_rome_stems() -> None:
    ns, t1, strat = parse_rome_stem("20250219_164325")
    assert (ns, strat) == ("rome:phone", "cluster")
    _, t2, _ = parse_rome_stem("20250219_164425")
    assert t2 - t1 == 60
    ns, _, strat = parse_rome_stem("vlcsnap-2025-02-18-18h36m32s090")
    assert (ns, strat) == ("rome:vlc-time", "cluster")
    assert parse_rome_stem("vlcsnap_2025-03-16-15h17m12s949")[0] == "rome:vlc-time"
    assert parse_rome_stem("vlcsnap-00058") == ("rome:vlc-number", 58.0, "block")
    assert parse_rome_stem("something_else")[2] == "fixed_train"


def test_block_split_has_no_leaks_and_buffer_is_dropped() -> None:
    records = [_rec(f"Japan_{i:06d}", "rdd2022:Japan", i, "block") for i in range(1000)]
    groups = build_groups(records, block_size=100, cluster_gap=60)
    assignment = assign_splits(groups, val_frac=0.2, test_frac=0.0, seed=0)
    assert sum(v == "val" for v in assignment.values()) == 200  # two whole blocks
    dropped = drop_buffer_frames(records, assignment, block_size=100, buffer=5)
    assert dropped  # at least one train/val boundary exists
    assert all(int(d.split("_")[1]) % 100 < 5 for d in dropped)
    assert find_leaks(records, assignment, buffer=5, cluster_gap=60) == []


def test_random_frame_split_would_be_caught() -> None:
    records = [_rec(f"a_{i}", "rdd2022:A", i, "block") for i in range(10)]
    bad = {r.image_id: ("val" if r.pos % 2 else "train") for r in records}
    problems = find_leaks(records, bad, buffer=5, cluster_gap=60)
    assert problems and "only 1 apart" in problems[0]


def test_sessions_stay_together() -> None:
    # two bursts of photos 10 minutes apart
    times = [0, 10, 20, 30, 600, 610, 620, 630]
    records = [_rec(f"p{t}", "rome:phone", t, "cluster") for t in times]
    groups = build_groups(records, block_size=100, cluster_gap=60)
    assert len(groups) == 2
    assignment = assign_splits(groups, val_frac=0.5, test_frac=0.0, seed=1)
    burst_a = {assignment[f"p{t}"] for t in (0, 10, 20, 30)}
    burst_b = {assignment[f"p{t}"] for t in (600, 610, 620, 630)}
    assert len(burst_a) == 1 and len(burst_b) == 1 and burst_a != burst_b
    assert find_leaks(records, assignment, buffer=5, cluster_gap=60) == []


def test_unparsed_names_go_to_train() -> None:
    records = [
        _rec("weird_a", "rome:unknown", 0, "fixed_train"),
        _rec("weird_b", "rome:unknown", 0, "fixed_train"),
    ]
    groups = build_groups(records, 100, 60)
    assert set(assign_splits(groups, 0.5, 0.0, 0).values()) == {"train"}


def test_same_seed_same_split() -> None:
    records = [_rec(f"x_{i}", "rdd2022:X", i, "block") for i in range(500)]
    groups = build_groups(records, 50, 60)
    assert assign_splits(groups, 0.2, 0.1, 3) == assign_splits(groups, 0.2, 0.1, 3)


def test_cross_split_duplicates_removed(tmp_path: Path) -> None:
    img = np.zeros((10, 10, 3), np.uint8)
    for name in ("a", "b"):
        cv2.imwrite(str(tmp_path / f"{name}.jpg"), img)  # identical bytes
    cv2.imwrite(str(tmp_path / "c.jpg"), img + 5)
    records = [_rec(n, "x:y", i, "block", tmp_path) for i, n in enumerate("abc")]
    assignment = {"a": "train", "b": "val", "c": "val"}
    assert drop_cross_split_duplicates(records, assignment) == ["b"]
    assert assignment == {"a": "train", "c": "val"}


def _make_interim(root: Path, source: str, ids: list[str], size=(40, 60)) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    with (root / "meta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "source", "country"])
        writer.writeheader()
        for i, image_id in enumerate(ids):
            writer.writerow({"image_id": image_id, "source": source, "country": "Japan"})
            img = np.full((size[0], size[1], 3), i % 250, np.uint8)
            cv2.imwrite(str(root / "images" / f"{image_id}.jpg"), img)
            (root / "labels" / f"{image_id}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"
            )
    return root


def test_end_to_end(tmp_path: Path) -> None:
    rdd = _make_interim(tmp_path / "rdd", "rdd2022", [f"Japan_Japan_{i:06d}" for i in range(400)])
    rome = _make_interim(
        tmp_path / "rome",
        "rome",
        [f"rome_20250219_16{m:02d}{s:02d}" for m in (10, 40) for s in range(0, 50, 10)],
    )
    out = tmp_path / "out"
    res = run(
        [rdd, rome],
        out,
        val_frac=0.25,
        block_size=50,
        buffer=3,
        resize_max_side=32,
        mode="copy",
        skip_hash=True,
    )

    assert (out / "splits.json").exists()
    saved = json.loads((out / "splits.json").read_text())
    assert saved["assignment"] == res["assignment"]
    n_images = sum(len(list((out / "images" / s).glob("*.jpg"))) for s in ("train", "val"))
    n_labels = sum(len(list((out / "labels" / s).glob("*.txt"))) for s in ("train", "val"))
    assert n_images == n_labels == len(res["assignment"])
    assert "val/rdd2022" in res["counts"] and "train/rome" in res["counts"]
    # images bigger than 32 px were shrunk (labels are normalized, so they stay valid)
    some = next((out / "images" / "train").glob("*.jpg"))
    assert max(cv2.imread(str(some)).shape[:2]) == 32
    with pytest.raises(FileExistsError):
        run([rdd, rome], out, resize_max_side=32)


def test_load_records_reads_meta(tmp_path: Path) -> None:
    d = _make_interim(tmp_path / "d", "rdd2022", ["Japan_Japan_000007"])
    rec = load_records(d)[0]
    assert (rec.namespace, rec.pos, rec.strategy) == ("rdd2022:Japan", 7.0, "block")
