# road-damage-tn

Road damage (potholes and cracks) detection for Tunisian roads: a new dataset, trained models,
an API, a live demo and a prioritized repair map.

> Status: work in progress. Results will be added here only once they come from real runs
> (`{{MAP50}}` placeholders until then).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
make setup
make lint test
```

## Data preparation (foreign data for Experiment A)
```bash
python -m src.data.convert_rdd2022 --root <rdd_raw> --out data/interim/rdd2022
python -m src.data.convert_rome --root <rome_raw> --out data/interim/rome
python -m src.data.preview --dir data/interim/rome --n 8 --out reports/preview_rome
python -m src.data.make_splits --interim data/interim/rdd2022 data/interim/rome \
    --out data/processed/yolo_foreign --resize-max-side 1024
```
Splits are made by group (blocks of frames / timestamp sessions), never by random frame, and the
leakage tests run automatically. See `src/data/make_splits.py` for the assumptions.

## Training
```bash
pip install -e ".[train]"
make smoke                                   # 1 epoch on 20 synthetic images (CPU)
python -m src.train.train --config configs/train_baseline_a.yaml --data-root <yolo_foreign>
python -m src.eval.evaluate --weights runs/baseline_a/weights/best.pt \
    --data runs/baseline_a_config/data.yaml --split val --out reports/baseline_a_val.json
```

## Datasets and licenses
- RDD2022: link and cite the official release (Figshare DOI 10.6084/m9.figshare.21431547); check its license.
- Rome road damage dataset: CC BY 4.0, Zenodo DOI 10.5281/zenodo.17834373.
Neither dataset is republished here; they are used for baseline training and comparison only.

## Project spec
See [CLAUDE.md](CLAUDE.md) for the full specification.

## License
Code: to be set (note: Ultralytics is AGPL-3.0, so this repo stays public and states its license).
Dataset license: to be decided (CC BY 4.0 proposed).
