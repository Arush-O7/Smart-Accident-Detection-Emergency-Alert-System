"""Export both on-device models for the ESP32 (C headers) and the web dashboard (demo/data/model.js).

  model A  crash detector      LightGBM on 37 accel+gyro features, trained on VZCrash (real crashes)
  model B  road-event namer    XGBoost on the 30 accelerometer features, trained on PVS road data

Writes transmitter_LORA/ml_crash_detector/crash_model.h, .../road_model.h and demo/data/model.js.
"""
import json
from pathlib import Path

import joblib
import numpy as np

import _paths  # noqa: F401
from features import FEATURE_NAMES, FEATURE_NAMES_EXT, N_FEATURES, N_FEATURES_EXT
from signal_config import EMA_ALPHA, N_POST, N_PRE, TRIGGER_G
from tree_export import emit_c, flash_kb, from_lightgbm, from_xgboost, to_dict

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "ml"
FW = ROOT / "transmitter_LORA" / "ml_crash_detector"
WEB = ROOT / "demo" / "data" / "model.js"
ROAD_CLASSES = json.loads((ML / "results" / "road_model.json").read_text(encoding="utf-8"))["classes"]


def main():
    crash = joblib.load(ML / "models" / "crash_model_vzcrash.joblib")
    road = joblib.load(ML / "models" / "road_model.joblib")

    vz = np.load(ML / "data" / "processed" / "vzcrash_test.npz")["X"][::50, :N_FEATURES_EXT]
    road_ref = np.load(ML / "data" / "processed" / "features_train.npy")[::40, :N_FEATURES]

    crash_spec = from_lightgbm(crash.booster_, vz)
    road_spec = from_xgboost(road.get_booster(), len(ROAD_CLASSES), road_ref)
    for name, spec in (("crash", crash_spec), ("road", road_spec)):
        print(f"{name:6s} {len(spec['roots'])} trees, {len(spec['feat'])} nodes, "
              f"max error {spec['max_error']:.2e}, ~{flash_kb(spec):.0f} KB flash")
        assert spec["max_error"] < 1e-3, f"{name} model does not reproduce the trained model"

    (FW / "crash_model.h").write_text(emit_c(
        crash_spec, "crash_model",
        "Crash detector: LightGBM trained on VZCrash (real on-road crashes), 37 accel+gyro features.",
        extra_defines=['CRASH_MODEL_NAME "LightGBM-VZCrash"', f"CRASH_MODEL_N_FEATURES {N_FEATURES_EXT}"]))
    (FW / "road_model.h").write_text(emit_c(
        road_spec, "road_model",
        "Road-event classifier: XGBoost trained on PVS / DriverBehavior, 30 accelerometer features.",
        classes=ROAD_CLASSES,
        extra_defines=['ROAD_MODEL_NAME "XGBoost-road"', f"ROAD_MODEL_N_FEATURES {N_FEATURES}"]))

    WEB.parent.mkdir(parents=True, exist_ok=True)
    WEB.write_text(
        "window.CRASH_MODEL = " + json.dumps(to_dict(
            crash_spec, name="LightGBM-VZCrash", features=FEATURE_NAMES_EXT, trigger_g=TRIGGER_G,
            ema_alpha=EMA_ALPHA, n_pre=N_PRE, n_post=N_POST), separators=(",", ":")) + ";\n"
        "window.ROAD_MODEL = " + json.dumps(to_dict(
            road_spec, name="XGBoost-road", classes=ROAD_CLASSES, features=FEATURE_NAMES),
            separators=(",", ":")) + ";\n")
    print("wrote", (FW / "crash_model.h").relative_to(ROOT), (FW / "road_model.h").relative_to(ROOT), WEB.relative_to(ROOT))


if __name__ == "__main__":
    main()
