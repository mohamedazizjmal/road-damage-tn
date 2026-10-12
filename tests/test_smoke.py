"""Sanity checks for the repo skeleton."""

from pathlib import Path

import src

ROOT = Path(__file__).resolve().parents[1]


def test_package_importable() -> None:
    assert src.__doc__


def test_trips_csv_header() -> None:
    header = (ROOT / "data/metadata/trips.csv").read_text().splitlines()[0]
    assert header.startswith("trip_id,date,start_time_local")
