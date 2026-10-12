# Road Damage Detection for Tunisia — End-to-End Computer Vision Project

> Hand this file to Claude Code as the project specification (save it as `CLAUDE.md` or `docs/PROJECT_SPEC.md` at the repo root and point Claude Code at it).
> Language of the project: English (code, docs, commit messages).

---

## 0. Working agreements for Claude Code (read first)

1. **Never fabricate results.** Do not invent metrics, dataset sizes, image counts, file URLs, or benchmark numbers. Anything measured must come from a real run. Use `TODO`/placeholders like `{{MAP50}}` in docs until real numbers exist.
2. **Do not invent dataset links.** Use only the sources listed in Section 5. If a link fails, tell the human and ask instead of guessing.
3. **Some steps are human-only** (Section 3). Build the tooling for them, then stop and tell the human exactly what to do. Do not simulate them.
4. **Everything must be free.** Only free/open-source tools and free tiers (Section 4). Never add a paid service or require a credit card.
5. **Reproducibility first:** pinned dependencies, fixed random seeds, config-driven scripts (YAML), no hard-coded paths.
6. **Small verified steps.** After each phase, run tests/lint, summarize what changed, and list open items. Commit often with clear messages (Conventional Commits).
7. **Privacy:** no unblurred faces or license plates may ever be committed, uploaded, or published.
8. **Ask before large decisions** that change scope (class taxonomy, dataset license, hosting provider).
9. Prefer simple, readable code with type hints and docstrings over clever code.

---

## 1. Goal and success criteria

### 1.1 Goal
Build, evaluate, deploy and document an end-to-end computer vision system that **detects road damage (potholes and cracks) on Tunisian roads**, geolocates detections, and produces a **prioritized repair map**. Publish a **new Tunisian road damage dataset** as the project's original contribution.

### 1.2 Why this matters (for the portfolio)
- Shows the full ML lifecycle: data collection, annotation, training, evaluation, deployment, and impact.
- Original contribution: to our knowledge no public Tunisian road damage dataset was found during scoping (this must be re-verified by a quick search before publishing any claim; phrase it as "we did not find" rather than "none exists").
- Key scientific story: **domain shift** between foreign datasets and Tunisian roads, and how much local data closes the gap.

### 1.3 Deliverables (definition of done)
| # | Deliverable | Acceptance criterion |
|---|---|---|
| D1 | Tunisian dataset | Published on Kaggle / Zenodo / Hugging Face with data card, splits, license, privacy-blurred images |
| D2 | Trained models | 3 model variants (A, B, C — Section 8) with reproducible configs and saved weights |
| D3 | Evaluation report | Results table, per-class metrics, domain-shift analysis, error analysis with examples |
| D4 | API | FastAPI + Docker, `/predict` works locally with the ONNX model |
| D5 | Live demo | Gradio app running on Hugging Face Spaces (free CPU) |
| D6 | Damage map | Interactive HTML map (Folium) from a real trip: GeoJSON + severity per 100 m segment |
| D7 | Repo & docs | Clean README, data card, annotation guide, model card, CI green |
| D8 | Demo video + write-up | 60–90 s video; 3–6 page write-up (Markdown/PDF) |

### 1.4 Non-goals
- Not a certified pavement inspection tool. No depth/area-in-cm² measurement.
- No real-time embedded deployment (optional stretch only).
- No paid GPU, paid hosting, or paid annotation tools.

---

## 2. Scope and design decisions (defaults — confirm with human if changing)

### 2.1 Class taxonomy
Primary task (2 classes) so that heterogeneous datasets can be merged:

| id | name | Notes |
|---|---|---|
| 0 | `pothole` | RDD2022 `D40`; Rome dataset `pothole` |
| 1 | `crack` | RDD2022 `D00`, `D10`, `D20` merged; Rome dataset `crack` |

- **Manholes:** the Rome dataset has a `manhole` class. RDD2022 does not. Do **not** train on manholes as a class (missing labels elsewhere would add noise). Drop manhole boxes when converting for training so manholes act as background. For the **Tunisian** data, annotate `manhole` as an extra label for **error analysis only** (are manholes being confused with potholes?), and drop it in the training export.
- **Optional fine-grained experiment:** 4-class RDD-style (`D00`, `D10`, `D20`, `D40`) evaluated only where labels exist. Only do this after the main pipeline works.

### 2.2 Model family
- Ultralytics YOLO (v8n / v8s as the default; a newer YOLO release is acceptable if it installs cleanly). Note: Ultralytics is **AGPL-3.0**. Keep the repo public and state the license clearly. Mention this in the README.
- Input size 640 by default. Try 960 only if small cracks are missed and GPU time allows.

### 2.3 Evaluation principle
- **Split by trip / location, never randomly by frame.** Consecutive video frames are near-duplicates; random splitting leaks and inflates scores.
- Report mAP@50, mAP@50-95, precision, recall, F1 per class, and confusion matrix.
- Report confidence intervals via bootstrap over test images (or multiple seeds if time allows).

---

## 3. Human-only tasks (Claude Code must prompt the human, not do these)

| Task | Why human | What Claude Code should provide |
|---|---|---|
| Create accounts: GitHub, Kaggle, Hugging Face, DagsHub (optional), Roboflow (optional) | Requires the human | A checklist and token instructions |
| Record road videos + GPS logs | Physical world | A recording protocol (Section 6.1) and a metadata template |
| Annotate images | Needs eyes + judgement | Label-Studio/CVAT setup instructions, label config, annotation guide |
| Second-annotator review of ~10% | Independence | A script that picks the review sample and computes agreement |
| Run GPU training on Kaggle/Colab | Claude Code usually has no GPU | Ready-to-run notebooks/scripts + smoke-test on a tiny CPU subset |
| Publish dataset/demo | Account ownership, license choice | Ready-to-upload folders and checklists |
| Safety and legality of filming | Human responsibility | Rules in Section 6.1 |

---

## 4. Free technology stack

| Layer | Tool | Notes |
|---|---|---|
| Language / env | Python 3.10–3.12, `uv` or `venv` + `pip`, `pyproject.toml` | Pin versions in `requirements.lock` or `uv.lock` |
| Compute | Kaggle Notebooks (free weekly GPU quota), Google Colab free | Save checkpoints often; sessions can be killed |
| Video/frames | `ffmpeg`, OpenCV | |
| GPS | GPX from any free GPS-logger phone app; `gpxpy` to parse | |
| Privacy | `deface` for faces; plates: an open-source plate detector (verify license) or manual blur during review | Never publish unblurred images |
| Annotation | CVAT (free, open-source/hosted free tier) or Label Studio (open-source) | Export in YOLO format |
| Detection | Ultralytics YOLO | AGPL-3.0 |
| Tracking experiments | MLflow (local file store) or Weights & Biases (free personal) | Ultralytics has built-in integrations |
| Data versioning | DVC with a free remote (DagsHub or Google Drive) or Hugging Face Datasets | Do not commit large data to git |
| Geo | `geopandas`, `shapely`, `pyproj`, `folium` | OpenStreetMap tiles (respect their usage policy; attribution required) |
| API | FastAPI, Uvicorn, ONNX Runtime | |
| Demo | Gradio on Hugging Face Spaces (free CPU) | |
| Containers | Docker | Local build; no paid registry required |
| CI | GitHub Actions | `ruff`, `pytest`, Docker build |
| Quality | `ruff`, `black` (or ruff-format), `pytest`, `pre-commit`, `mypy` (optional) | |

> Free-tier limits change. Before relying on a limit (GPU hours, storage, hosting), check the provider's current page and tell the human what you found.

---

## 5. Data sources

### 5.1 Public datasets (for pretraining / baseline)
1. **RDD2022** (multi-country: Japan, India, Czech Republic, Norway, United States, China; four damage types D00, D10, D20, D40; about 47k images). Source: the RDD2022 paper (arXiv 2209.08538, Geoscience Data Journal 2024) and the project's GitHub organization `sekilab` (repo `RoadDamageDetector`). **Verify the current download link and license from those pages.** Annotations are Pascal VOC XML per country and must be converted to YOLO.
2. **Rome/Sacrofano road damage dataset** (about 2,009 YOLO-annotated images; classes pothole, crack, manhole). Kaggle: `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`; also on Zenodo. Check the license on the dataset page before use/redistribution.
3. Optional additional crack datasets (classification-only) are **not** needed for the detection pipeline.

### 5.2 Original data (Tunisian)
Collected by the human (Section 6). Target: **1,500–3,000 useful images** (more is better if annotation time allows), from **at least 8–10 distinct trips** covering different road types and conditions.

Rule: RDD2022 and the Rome data are used **only** for baseline training and comparison. Do not republish them in our dataset; link to them and respect their licenses.

---

## 6. Data collection protocol (human) and tooling (Claude Code)

### 6.1 Recording protocol (put this in `docs/collection_protocol.md`)
- Safety first: record only as a passenger or with a fixed dashboard mount while the vehicle is stopped/legally operated. Never handle the phone while driving.
- Phone landscape, 1080p, 30 fps, fixed mount at windshield/dashboard, stable horizon; clean windshield.
- Run a free GPS-logger app at 1 Hz (GPX) at the same time. Note the video start time (clock-synced phone) or clap/flash sync event.
- Cover variety: highways, urban streets, suburban roads, rural roads/tracks; day, low sun, overcast, after rain (if safe); different vehicles/heights if possible.
- Record a metadata row per trip in `data/metadata/trips.csv`:
  `trip_id, date, start_time_local, city_or_region, road_type, weather, time_of_day, device, mount_height_cm, notes, gpx_file, video_file`
- Public-road filming only. Avoid identifiable people, private property interiors, and sensitive sites (military, government buildings).

### 6.2 Tooling to build (Claude Code)
`src/data/extract_frames.py`
- Input: video path, trip_id, config (fps, blur threshold, min speed, output dir).
- Steps: `ffmpeg` extract at 1–2 fps → drop blurry frames (variance of Laplacian below threshold; threshold configurable and reported) → drop near-duplicates (perceptual hash and/or minimum distance travelled if GPS available) → drop frames when stationary (speed below ~2 m/s).
- Output: `data/interim/frames/{trip_id}/{trip_id}_{frame_idx:06d}.jpg` plus `frames_index.csv` with `trip_id, frame_file, video_time_s, kept, reason_dropped`.

`src/geo/sync_gps.py`
- Parse GPX (`gpxpy`), interpolate lat/lon/speed at each frame timestamp (`video_start_time + video_time_s + optional_offset_s`).
- Config: `time_offset_s` per trip (to fix clock drift), documented.
- Output: `frames_index.csv` extended with `lat, lon, speed_mps, gps_ok`.

`src/data/anonymize.py`
- Blur faces with `deface` (or equivalent). For plates, integrate an open-source detector if license allows; else provide a manual-review workflow (flag frames with vehicles close to the camera).
- Output to `data/processed/blurred/{trip_id}/...`. Keep the un-blurred originals **out of git and out of any published artifact**.
- Provide a verification report: number of frames processed, number flagged for manual check.

`src/data/select_for_annotation.py`
- Sample frames for annotation with diversity: stratify by trip, road type, and time of day; oversample frames that a baseline model flags as likely damage (active learning), but keep a random portion (e.g., 30%) to avoid bias.

---

## 7. Annotation

### 7.1 Setup
- Use CVAT or Label Studio with a label config: `pothole`, `crack`, `manhole` (manhole for analysis only).
- Provide `docs/annotation_guide.md` with clear rules and example images:
  - `pothole`: a bowl-shaped hole/depression or missing pavement; box tightly around the visible hole.
  - `crack`: linear or network cracks in pavement. For alligator (network) cracks, box the whole affected patch. Long cracks: one box per continuous visible segment (define a max length rule, e.g., split at large gaps).
  - Ignore: repaired patches (unless cracking through), road markings, shadows, oil stains, speed bumps (document as hard negatives).
  - Minimum size rule: skip objects smaller than e.g. 12×12 px at 640 px width (state the final rule you choose).
  - Occlusion/truncation rules.
- Export format: YOLO txt (`class x_center y_center width height`, normalized).

### 7.2 Quality control
`src/eval/annotation_agreement.py`
- Randomly select ~10% of images for double annotation by a second person.
- Compute IoU-matched agreement (precision/recall/F1 between annotators at IoU 0.5) and class agreement (Cohen's kappa on matched boxes). Report in the data card.

### 7.3 Splits (critical)
`src/data/make_splits.py`
- Assign whole **trips** (or geographic clusters of a trip, e.g., 500 m tiles) to `train / val / test` (≈70/15/15 by images, balanced by road type when possible).
- Ensure the **test set contains roads not seen in training** (different route or area).
- Add an automated leakage test: no two frames closer than N meters/seconds may fall in different splits from the same trip segment; verify no duplicate hashes across splits.
- Save `splits.json` and DVC-track it.

---

## 8. Modeling and experiments

### 8.1 Data converters
- `src/data/convert_rdd2022.py`: VOC XML → YOLO; map `D00, D10, D20 → crack(1)`, `D40 → pothole(0)`; ignore other codes; keep per-country tag in a CSV for analysis.
- `src/data/convert_rome.py`: map `pothole → 0`, `crack → 1`, drop `manhole`.
- `src/data/convert_tn.py`: from CVAT/Label Studio export to the same format; drop `manhole` for training but keep a separate copy with manholes for analysis.
- Unit tests for each converter with tiny fixtures.

### 8.2 Dataset YAML files (Ultralytics style)
`configs/data_foreign.yaml`, `configs/data_tn.yaml`, `configs/data_merged.yaml`
```yaml
path: data/processed/yolo_tn
train: images/train
val: images/val
test: images/test
names:
  0: pothole
  1: crack
```

### 8.3 Experiment matrix
| ID | Training data | Init | Test on |
|---|---|---|---|
| **A** | RDD2022 (+ Rome) | COCO-pretrained YOLO | Foreign val **and** Tunisian test |
| **B** | Tunisian train only | COCO-pretrained YOLO | Tunisian test |
| **C** | Tunisian train | Start from A's weights, fine-tune | Tunisian test |
| C-small (optional) | Tunisian train subsets (10%, 25%, 50%, 100%) | From A | Tunisian test |

The C-small data-efficiency curve shows how much local data is needed, which is a strong result for the write-up.

Optional extras: model size comparison (n vs s), image size (640 vs 960), augmentation ablation (mosaic on/off, HSV, blur), class-imbalance handling.

### 8.4 Training scripts (config-driven)
`src/train/train.py`
```python
# Sketch — implement with argparse + YAML config, seed everything
from ultralytics import YOLO
model = YOLO(cfg["init_weights"])           # e.g. "yolov8s.pt" or path to A's best.pt
model.train(
    data=cfg["data"], imgsz=cfg.get("imgsz", 640), epochs=cfg["epochs"],
    batch=cfg["batch"], seed=cfg.get("seed", 42), project="runs", name=cfg["run_name"],
    patience=cfg.get("patience", 30), cos_lr=True,
)
```
- Log to MLflow/W&B (config, metrics, artifacts). Save `best.pt`, `last.pt`, results CSV.
- Provide `notebooks/train_kaggle.ipynb` (clone repo, install pinned deps, mount data as a Kaggle dataset, run configs, save outputs) and a Colab equivalent. Handle session limits: save checkpoints frequently and support `resume=True`.
- Provide a **CPU smoke test** (`make smoke`) that trains 1–2 epochs on ~20 images to validate the pipeline in CI/locally.

### 8.5 Evaluation
`src/eval/evaluate.py`
- Run `model.val(split="test")` and store: mAP@50, mAP@50-95, per-class P/R/AP, F1 curve, confusion matrix, inference speed (CPU ms/image).
- `src/eval/bootstrap_ci.py`: bootstrap over test images for mAP@50 confidence intervals.
- `src/eval/error_analysis.py`: collect false positives/negatives; tag by category (shadow, wet road, patch, manhole, distance/small, night/low light, motion blur); save galleries into `reports/error_gallery/`. Support manual tagging via a CSV.
- Explainability: Grad-CAM-style or attention visualizations for a small sample (optional; document method and limits for detection models).
- `src/eval/domain_shift.py`: produce the main table and plots: A on foreign vs A on Tunisia vs B vs C; plus optional data-efficiency curve.

Reports live in `reports/` as Markdown + PNG (no raw private data).

---

## 9. Geo layer and severity map

### 9.1 Inference on trips
`src/geo/run_trip.py`
- Input: trip frames + `frames_index.csv` (with GPS) + model weights.
- Output: `reports/trips/{trip_id}/detections.geojson` (FeatureCollection of Points).

GeoJSON feature properties:
```json
{
  "trip_id": "T001", "frame_file": "T001_000123.jpg", "time_s": 123.4,
  "class": "pothole", "confidence": 0.81, "bbox_area_ratio": 0.012,
  "speed_mps": 8.2
}
```

### 9.2 De-duplication
Same damage appears in consecutive frames. Cluster detections of the same class within `R` meters (default 8 m; configurable) and keep the highest-confidence (or average location). Document the assumption and known GPS error (a few meters).

### 9.3 Severity score (heuristic — document clearly as such)
Per road segment of length `L` (default 100 m, along the GPS track):
```
segment_score = sum_over_detections( w_class * conf * min(1, bbox_area_ratio / a_ref) )
w_pothole = 3.0, w_crack = 1.0, a_ref = 0.02   (defaults, in config)
severity_bin = low / medium / high by tertiles or fixed thresholds (state which)
```
- Normalize by segment length. Produce `segments.geojson` with `segment_id, length_m, n_potholes, n_cracks, score, severity_bin`.
- The formula is a **prioritization heuristic**, not an engineering standard. State it in the README and model card. If an engineer contact can review it, note their feedback.

### 9.4 Map
`src/geo/make_map.py` → `reports/maps/{trip_id}_map.html` with Folium:
- Colored polyline segments by severity, markers for detections (popup: class, confidence, thumbnail path/URL if published), layer control, legend, OSM attribution.
- Also export a ranked CSV of the worst segments: `worst_segments.csv`.

---

## 10. Serving

### 10.1 ONNX export
`src/serve/export_onnx.py` using `model.export(format="onnx", imgsz=640, simplify=True)`. Verify ONNX vs PyTorch output parity on a small set; record CPU latency.

### 10.2 FastAPI
`api/main.py`
- `GET /health` → `{"status": "ok", "model": "<name>", "version": "<git sha>"}`
- `POST /predict` (multipart image) → JSON: list of `{class, confidence, bbox_xyxy}` plus image size and inference time. Validate file type/size, thresholds via query params (`conf`, `iou`).
- `POST /predict/annotated` → returns the image with boxes drawn.
- Load the model once at startup. No image persistence by default (privacy).
- Tests with `pytest` + `httpx` using a tiny sample image.

### 10.3 Docker
`api/Dockerfile` (slim Python image, non-root user, pinned deps, CPU only). `docker-compose.yml` for local run. CI must build the image.

### 10.4 Hugging Face Space
`app/app.py` Gradio: upload an image, adjust confidence slider, show annotated output + table. Include example images (blurred, licensed for release) and a clear disclaimer about limits. Provide `app/README.md` with the Space metadata header and deployment steps for the human.

---

## 11. Repository layout

```
road-damage-tn/
├── CLAUDE.md                     # this spec or a short pointer to it
├── README.md
├── LICENSE                       # code license (mind AGPL-3.0 for Ultralytics)
├── pyproject.toml
├── Makefile                      # setup, lint, test, smoke, data, train, eval, serve
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/                      # yaml: data, train, extraction, geo
├── data/                         # DVC-tracked, git-ignored
│   ├── raw/                      # videos, GPX (never published)
│   ├── interim/                  # extracted frames
│   ├── processed/                # blurred + YOLO-formatted
│   └── metadata/trips.csv
├── docs/
│   ├── collection_protocol.md
│   ├── annotation_guide.md
│   ├── data_card.md
│   ├── model_card.md
│   └── privacy_and_ethics.md
├── notebooks/                    # EDA, Kaggle/Colab training, error analysis
├── src/
│   ├── data/                     # extract_frames, anonymize, converters, splits
│   ├── train/
│   ├── eval/
│   ├── geo/
│   └── serve/
├── api/                          # FastAPI + Dockerfile
├── app/                          # Gradio app for HF Spaces
├── reports/                      # figures, tables, maps (no private data)
└── tests/
```

---

## 12. Testing and CI

- `pytest` unit tests: converters, split leakage checks, GPS interpolation (synthetic track), de-duplication, severity scoring, API endpoints.
- `ruff` lint and format check.
- CI (GitHub Actions): install deps, lint, run tests, CPU smoke test on fixtures, build Docker image.
- Data tests: assert every image has a label file (or explicit empty), labels are normalized in [0,1], class ids valid, no duplicate hashes across splits.

---

## 13. Documentation requirements

1. **README:** problem, result highlights (real numbers only), demo link, quickstart, repo tour, limitations, license, citation.
2. **Data card** (Datasheets-for-Datasets style): motivation, collection method, locations (coarse — region/city level, not precise home/work), devices, time range, annotation process and guide, class definitions, agreement stats, split protocol, privacy steps, known biases, intended and out-of-scope uses, license, contact.
3. **Model card:** intended use, training data, metrics with CIs, domain-shift findings, failure modes, ethical considerations, license notes.
4. **Write-up (3–6 pages):** Introduction, related work (RDD2022, etc.), data collection, annotation, methods, experiments, results (domain shift, data efficiency), error analysis, deployment, limitations, future work.
5. **Privacy & ethics note:** what was blurred, how verified, what raw data is kept private, how to request removal.

---

## 14. Timeline (about 6 weeks, part-time; adjust as needed)

| Week | Goal | Outputs |
|---|---|---|
| 1 | Repo, env, CI, converters, baseline A on foreign data | Working pipeline, baseline metrics |
| 1–2 | Recording + extraction + GPS sync + blur tooling | First trips processed |
| 2–3 | Annotation (with guide and QC) | 1.5–3k labeled images, agreement stats |
| 3–4 | Splits, experiments B and C, error analysis | Domain-shift table and plots |
| 4–5 | ONNX, API, Docker, HF Space, geo layer and map | Live demo and map |
| 6 | Docs, data card, dataset publication, video, write-up | All deliverables (D1–D8) |

---

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Not enough annotated data | Active-learning selection; start with a smaller high-quality set; use data-efficiency curve as a result in itself |
| Split leakage inflates scores | Trip/location-level splits + automated leakage tests |
| Model confuses shadows/patches/manholes | Include hard-negative frames; dedicated error categories; mosaic/HSV augmentation |
| GPU quota exhausted | Smaller model/size first, checkpointing, resume support, run experiments in priority order (A → C → B) |
| GPS drift/offset | Per-trip time offset config, visual check on a sample, document uncertainty |
| Privacy breach | Blur pipeline + manual review sample + never publish raw video |
| License conflicts (AGPL, dataset licenses) | Record licenses in README/data card; ask the human before choosing the dataset license |
| Free hosting changes | Keep the app portable (Docker + Gradio); HF Spaces as default |

---

## 16. Optional stretch goals (only after D1–D8)

- Data-efficiency curve (10/25/50/100% of Tunisian data).
- Semi-supervised/pseudo-labeling on unlabeled Tunisian frames (report carefully).
- TFLite/ONNX-mobile demo running on a phone.
- Simple crack-length estimation using camera geometry (clearly labeled as approximate).
- Arabic/French README translations.

---

## 17. Suggested first prompts for Claude Code

1. "Read `CLAUDE.md`. Set up the repo skeleton from Section 11, `pyproject.toml`, Makefile, pre-commit, and a CI workflow. Do not add data. Run lint and tests and report."
2. "Implement the converters in Section 8.1 with unit tests using tiny fixtures. Show me the class mapping table you used."
3. "Implement `extract_frames.py`, `sync_gps.py`, and `anonymize.py` per Section 6.2 with configs and tests on synthetic data. Then write `docs/collection_protocol.md` and `docs/annotation_guide.md`."
4. "Implement `make_splits.py` with the leakage tests from Section 7.3."
5. "Write training and evaluation scripts (Section 8) plus the Kaggle notebook and CPU smoke test."
6. "Implement the geo layer (Section 9), API (Section 10), Gradio app, Dockerfile, and docs (Section 13)."

At the end of each step, list what the human must do next (recording, annotation, GPU runs, account setup, publishing).

---

## 18. Open questions for the human (Claude Code should ask early)

1. Preferred license for the Tunisian dataset (e.g., CC BY 4.0 or CC BY-NC 4.0)?
2. Which GPS-logger app and phone will be used (for the GPX/time-sync notes)?
3. Which cities/regions will be covered first?
4. Where will data be versioned (DagsHub, Google Drive, or Hugging Face)?
5. Who is the second annotator for the 10% agreement check?
6. Is there an engineer or municipal contact who could review the severity heuristic?
