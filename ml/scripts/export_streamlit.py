"""Export the models and demo data for the Streamlit app (app/data/).

The app runs the same flattened trees the ESP32 runs, so it needs numpy only - no xgboost,
lightgbm or scikit-learn at runtime. That keeps the hosted app small and quick to start.

  app/data/models.npz   both models as flat arrays (feat, thr, no, roots, bias)
  app/data/demo.npz     the replay drive and the event library, as int16 sensor counts
  app/data/meta.json    everything text: class names, event descriptions, result tables
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import _paths  # noqa: F401
import export_demo as ed
from features import FEATURE_NAMES, FEATURE_NAMES_EXT, N_FEATURES, N_FEATURES_EXT
from signal_config import EMA_ALPHA, FS, N_POST, N_PRE, TRIGGER_G, WIN
from tree_export import from_lightgbm, from_xgboost

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "ml"
OUT = ROOT / "app" / "data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crash = joblib.load(ML / "models" / "crash_model_vzcrash.joblib")
    road = joblib.load(ML / "models" / "road_model.joblib")
    road_classes = json.loads((ML / "results" / "road_model.json").read_text(encoding="utf-8"))["classes"]

    vz = np.load(ML / "data" / "processed" / "vzcrash_test.npz")["X"][::50, :N_FEATURES_EXT]
    road_ref = np.load(ML / "data" / "processed" / "features_train.npy")[::40, :N_FEATURES]
    specs = {"crash": from_lightgbm(crash.booster_, vz),
             "road": from_xgboost(road.get_booster(), len(road_classes), road_ref)}

    arrays = {}
    for name, spec in specs.items():
        arrays[f"{name}_feat"] = spec["feat"].astype(np.int16)
        arrays[f"{name}_thr"] = spec["thr"].astype(np.float32)
        arrays[f"{name}_no"] = spec["no"].astype(np.int32)
        arrays[f"{name}_roots"] = spec["roots"].astype(np.int32)
        arrays[f"{name}_bias"] = spec["bias"].astype(np.float32)
    np.savez_compressed(OUT / "models.npz", **arrays)

    replay = ed.replay_stream()
    events = ed.vzcrash_events() + ed.road_events()
    np.savez_compressed(
        OUT / "demo.npz",
        replay_acc=np.array(replay["samples"], np.int16).reshape(-1, 3),
        replay_gyro=np.array(replay["gyro"], np.int16).reshape(-1, 3),
        replay_speed=np.array(replay["speed_kmh"], np.float32),
        event_acc=np.stack([np.array(e["samples"], np.int16).reshape(WIN, 3) for e in events]),
        event_gyro=np.stack([np.array(e["gyro"], np.int16).reshape(WIN, 3) if e["gyro"]
                             else np.zeros((WIN, 3), np.int16) for e in events]),
    )

    comparison = pd.read_csv(ML / "results" / "vzcrash_comparison.csv")
    meta = dict(
        models={n: dict(kind=s["kind"], cmp=s["cmp"], n_classes=int(s["n_classes"]),
                        n_trees=len(s["roots"]), n_nodes=len(s["feat"])) for n, s in specs.items()},
        road_classes=road_classes,
        features=dict(accel=FEATURE_NAMES, all=FEATURE_NAMES_EXT),
        config=dict(fs=FS, win=WIN, n_pre=N_PRE, n_post=N_POST, trigger_g=TRIGGER_G, ema_alpha=EMA_ALPHA),
        replay={k: replay[k] for k in ("source", "truth", "road", "crash_time", "fs")},
        events=[{k: v for k, v in e.items() if k not in ("samples", "gyro")} for e in events],
        comparison=json.loads(comparison.to_json(orient="records")),
        vzcrash=json.loads((ML / "results" / "metrics_vzcrash.json").read_text(encoding="utf-8")),
        road=json.loads((ML / "results" / "road_model.json").read_text(encoding="utf-8")),
        parity=(ML / "results" / "firmware_parity.txt").read_text(encoding="utf-8"),
    )
    (OUT / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    for f in ("models.npz", "demo.npz", "meta.json"):
        print(f, (OUT / f).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
