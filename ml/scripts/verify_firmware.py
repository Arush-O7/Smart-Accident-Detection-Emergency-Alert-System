"""Prove the firmware computes what Python computes, for both on-device models.

Compiles crash_features.h + crash_model.h + road_model.h into a host program, feeds it real
windows (VZCrash events for the crash model, PVS road events for the road model) and compares
features and probabilities against the Python pipeline. ESP32 timing must be measured on the
board; the sketch prints it with every event.
"""
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import _paths  # noqa: F401
from features import FEATURE_NAMES_EXT, N_FEATURES, N_FEATURES_EXT, extract, extract_gyro
from signal_config import WIN

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT / "transmitter_LORA" / "ml_crash_detector"
PROC = ROOT / "ml" / "data" / "processed"
ALERT_P = 0.80

HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "crash_features.h"
#include "crash_model.h"
#include "road_model.h"
int main(int argc, char **argv) {
  FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
  int n = atoi(argv[3]);
  static float acc[CF_WIN][3], gyro[CF_WIN][3];
  float f[CRASH_MODEL_N_FEATURES], road[ROAD_MODEL_N_CLASSES];
  double total = 0;
  for (int s = 0; s < n; s++) {
    if (fread(acc, sizeof(float), CF_WIN * 3, in) != CF_WIN * 3) return 1;
    if (fread(gyro, sizeof(float), CF_WIN * 3, in) != CF_WIN * 3) return 1;
    clock_t t0 = clock();
    crash_features(acc, f);
    crash_features_gyro(acc, gyro, f + CF_N_FEATURES);
    float p = crash_model_probability(f);
    road_model_predict(f, road);
    total += (double)(clock() - t0) / CLOCKS_PER_SEC;
    fwrite(f, sizeof(float), CRASH_MODEL_N_FEATURES, out);
    fwrite(&p, sizeof(float), 1, out);
    fwrite(road, sizeof(float), ROAD_MODEL_N_CLASSES, out);
  }
  printf("%.1f\n", total / n * 1e6);
  return 0;
}
"""


def vzcrash_windows():
    acc, gyro = [], []
    for split in ("validation", "test", "train"):
        d = np.load(PROC / f"vzcrash_{split}.npz", allow_pickle=True)
        if len(d["raw_acc"]):
            acc.append(d["raw_acc"])
            gyro.append(d["raw_gyro"])
    return np.concatenate(acc), np.concatenate(gyro)


def main():
    acc_vz, gyro_vz = vzcrash_windows()
    meta = pd.read_csv(PROC / "train_meta.csv")
    road_w = np.load(PROC / "train_windows.npy")[(meta.is_aug == 0).to_numpy()][::3]
    acc = np.concatenate([acc_vz, road_w])
    gyro = np.concatenate([gyro_vz, np.zeros_like(road_w)])   # PVS windows carry no gyro here
    n_vz = len(acc_vz)

    F_py = np.stack([np.concatenate([extract(a), extract_gyro(a, g)]) for a, g in zip(acc, gyro)])
    p_crash_py = joblib.load(ROOT / "ml" / "models" / "crash_model_vzcrash.joblib").predict_proba(F_py[:, :N_FEATURES_EXT])[:, 1]
    p_road_py = joblib.load(ROOT / "ml" / "models" / "road_model.joblib").predict_proba(F_py[:, :N_FEATURES])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "harness.c").write_text(HARNESS)
        subprocess.run(["cc", "-O2", "-std=c99", f"-I{FW}", "-o", tmp / "harness", tmp / "harness.c", "-lm"], check=True)
        with open(tmp / "in.bin", "wb") as fh:
            for a, g in zip(acc, gyro):
                fh.write(np.ascontiguousarray(a, np.float32).tobytes())
                fh.write(np.ascontiguousarray(g, np.float32).tobytes())
        us = subprocess.run([tmp / "harness", tmp / "in.bin", tmp / "out.bin", str(len(acc))],
                            check=True, capture_output=True, text=True).stdout.strip()
        out = np.fromfile(tmp / "out.bin", np.float32).reshape(len(acc), -1)

    F_c = out[:, :N_FEATURES_EXT]
    p_crash_c = out[:, N_FEATURES_EXT]
    p_road_c = out[:, N_FEATURES_EXT + 1:]

    rel = np.abs(F_c - F_py) / (np.abs(F_py) + 1e-3)
    for name, e in sorted(zip(FEATURE_NAMES_EXT, rel.max(axis=0)), key=lambda x: -x[1])[:3]:
        print(f"  worst feature error  {name:18s} {e:.2e}")
    dp = np.abs(p_crash_c - p_crash_py)
    same_alert = ((p_crash_c >= ALERT_P) == (p_crash_py >= ALERT_P)).mean()
    same_road = (p_road_c.argmax(1) == p_road_py.argmax(1)).mean()
    print(f"windows compared:             {len(acc)} ({WIN} samples each; {n_vz} VZCrash, {len(acc) - n_vz} road)")
    print(f"feature max relative error:   {rel.max():.2e}")
    print(f"crash model max |dP|:         {dp.max():.2e} (99th pct {np.percentile(dp, 99):.1e})")
    print(f"same alert decision (p>=.8):  {same_alert:.4%}")
    print(f"road model same class:        {same_road:.4%}")
    print(f"C inference on this host:     {us} us per event (features + both models)")
    assert same_alert == 1.0 and same_road > 0.999, "firmware and Python disagree"


if __name__ == "__main__":
    main()
