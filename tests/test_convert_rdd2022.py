from pathlib import Path

import cv2
import numpy as np

from src.data.common import validate_yolo_dir
from src.data.convert_rdd2022 import convert_rdd2022, parse_voc_xml

XML = """<annotation>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  {objects}
</annotation>"""
OBJ = (
    "<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
    "<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
)


def _make(root: Path, country: str, stem: str, w: int, h: int, objs: list, xml_wh=None) -> None:
    train = root / "RDD2022" / country / "train"
    (train / "images").mkdir(parents=True, exist_ok=True)
    (train / "annotations" / "xmls").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(train / "images" / f"{stem}.jpg"), np.zeros((h, w, 3), dtype=np.uint8))
    xw, xh = xml_wh or (w, h)
    body = "".join(OBJ.format(n=n, a=a, b=b, c=c, d=d) for n, a, b, c, d in objs)
    (train / "annotations" / "xmls" / f"{stem}.xml").write_text(
        XML.format(w=xw, h=xh, objects=body)
    )


def test_parse_voc_xml(tmp_path: Path) -> None:
    p = tmp_path / "a.xml"
    p.write_text(XML.format(w=100, h=50, objects=OBJ.format(n="D40", a=1, b=2, c=3, d=4)))
    size, objs = parse_voc_xml(p)
    assert size == (100, 50)
    assert (objs[0].name, objs[0].xmax) == ("D40", 3.0)


def test_conversion_mapping_and_edge_cases(tmp_path: Path) -> None:
    root, out = tmp_path / "raw", tmp_path / "out"
    _make(
        root,
        "Japan",
        "a",
        100,
        50,
        [
            ("D40", 10, 10, 30, 30),  # pothole
            ("D00", 0, 0, 100, 50),  # crack, full image
            ("D10", 50, 10, 90, 20),  # crack
            ("D43", 0, 0, 10, 10),  # ignored code
            ("D20", 40, 40, 40, 45),  # degenerate
        ],
    )
    _make(root, "India", "b", 80, 80, [("D44", 0, 0, 5, 5)])  # only an ignored code -> empty label
    _make(
        root, "India", "c", 60, 60, [("D20", 10, 10, 20, 20)], xml_wh=(999, 999)
    )  # wrong XML size

    rows, ignored, mismatches = convert_rdd2022(root, out)

    lines = (out / "labels" / "Japan_a.txt").read_text().splitlines()
    assert lines[0] == "0 0.200000 0.400000 0.200000 0.400000"
    assert lines[1] == "1 0.500000 0.500000 1.000000 1.000000"
    assert len(lines) == 3  # D43 and degenerate D20 not written
    assert (out / "labels" / "India_b.txt").read_text() == ""
    assert ignored == {"D43": 1, "D44": 1}
    assert mismatches == 1  # real image size is used, not the XML one
    assert (out / "labels" / "India_c.txt").read_text().startswith("1 0.250000")

    by_id = {r["image_id"]: r for r in rows}
    assert by_id["Japan_a"]["n_pothole"] == 1 and by_id["Japan_a"]["n_crack"] == 2
    assert by_id["Japan_a"]["n_degenerate"] == 1
    assert by_id["India_b"]["country"] == "India"
    assert (out / "meta.csv").exists()
    assert validate_yolo_dir(out / "images", out / "labels") == []


def test_limit_per_country(tmp_path: Path) -> None:
    root, out = tmp_path / "raw", tmp_path / "out"
    for stem in ("a", "b", "c"):
        _make(root, "Czech", stem, 20, 20, [("D40", 1, 1, 5, 5)])
    rows, _, _ = convert_rdd2022(root, out, limit=2)
    assert len(rows) == 2
