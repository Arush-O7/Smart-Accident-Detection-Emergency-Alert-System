# ML pipeline: crash vs speed breaker vs pothole

Trains, compares and exports the on-device classifier used by `transmitter_LORA/ml_crash_detector`.
Results and caveats: [results/REPORT.md](results/REPORT.md).

## Reproduce

Python 3.11+ and a C compiler. Two datasets, two models:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r ml/requirements.txt
```

**Model A — crash detector (VZCrash, ~7.7 GB, gated).** Create a free HuggingFace account, accept the
terms at https://huggingface.co/datasets/vzc-research-chapter/VZCrash, then:

```bash
hf auth login
python ml/scripts/download_vzcrash.py --split train --split validation --split test
python ml/scripts/build_vzcrash.py          # ~4 min: trigger windows + 40 features
python ml/scripts/train_vzcrash.py          # ~30 s: 12 models, official splits
```

**Model B — road-event classifier (open datasets, ~400 MB).**

```bash
python ml/scripts/download_data.py          # PVS, NHTSA, DriverBehavior, potholes
python ml/scripts/build_dataset.py          # 2.5 s trigger windows
python ml/scripts/train_road_model.py       # leave-one-vehicle-out CV
```

**Ship them.**

```bash
python ml/scripts/export_model.py           # -> crash_model.h, road_model.h, demo/data/model.js
python ml/scripts/verify_firmware.py        # compiles the C and checks it matches Python
python ml/scripts/export_demo.py            # -> demo/data (replay, event library, tables)
```

`train_compare.py` is the earlier crash model built from NHTSA crash tests, kept for comparison;
`build_dataset.py` still produces its data. Model A now comes from VZCrash instead.

On macOS, `xgboost` and `lightgbm` need an OpenMP runtime: `brew install libomp`.

LightGBM training is not bit-reproducible across runs (multithreaded histogram ties), so retraining
moves the last decimal. What matters is that the exported header matches the saved model:
`verify_firmware.py` checks exactly that, and must be re-run after any retrain.

## Layout

| Path | What it does |
|---|---|
| `src/signal_config.py` | Window, trigger and sensor constants shared with the firmware |
| `src/signals.py` | Resampling, MPU6050 emulation, the streaming trigger (same logic as the firmware) |
| `src/features.py` | The 30 accelerometer + 7 gyroscope + 3 GPS features, and the document's 8-feature 200 ms vector (mirrored in `crash_features.h` and `demo/crash_ai.js`) |
| `src/tree_export.py` | Flattens LightGBM / XGBoost ensembles into C arrays and JSON |
| `scripts/download_data.py` | Fetches the open datasets from their original public sources |
| `scripts/download_vzcrash.py` | Fetches VZCrash from HuggingFace (needs a free account) |
| `scripts/build_vzcrash.py` | Replays VZCrash events through the firmware trigger into feature rows |
| `scripts/train_vzcrash.py` | Model A: compares 12 crash detectors on the official splits |
| `scripts/train_road_model.py` | Model B: the 4-class road-event classifier |
| `scripts/build_dataset.py` | Labels events, assembles crash windows, sets up leakage-safe folds |
| `scripts/train_compare.py` | First-generation crash model from NHTSA crash tests (superseded) |
| `scripts/export_model.py` | Writes both firmware headers and the dashboard model file |
| `scripts/verify_firmware.py` | Compiles the firmware headers on the host and compares outputs against Python |
| `scripts/export_demo.py` | Builds the dashboard data, using a model trained without the replayed vehicle |

## Changing the firmware constants

`TRIGGER_G`, `EMA_ALPHA` and the window sizes live in `src/signal_config.py`, `crash_features.h` and the sketch. Change them in all three, then rebuild the dataset and retrain. `verify_firmware.py` fails if the C features drift from Python.
