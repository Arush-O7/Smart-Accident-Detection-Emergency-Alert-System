"""Turn VZCrash events into triggered windows, exactly as the ESP32 would see them.

Each VZCrash event is 16 s of 100 Hz accelerometer (g) + gyroscope (deg/s) and 1 Hz GPS speed,
reviewed by three humans as crash / near_miss / normal_driving. We replay every event through the
firmware trigger (0.45 g over an EMA gravity estimate) and keep one row per triggered window.

Window labels
  crash events        the window with the strongest jolt is the crash; the other windows in that
                      event are dropped, because they may sit before or after the impact
  near_miss / normal  every triggered window is a negative (the whole 16 s contains no crash)

Outputs ml/data/processed/vzcrash_{split}.npz with the 40 features (30 accel + 7 gyro + 3 GPS
speed), the document's 8-feature 200 ms vector for the baseline, labels and event metadata, plus a
small sample of raw windows for the demo.

Usage: ml/.venv/bin/python ml/scripts/build_vzcrash.py [--jobs 8]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import _paths  # noqa: F401
from features import N_FEATURES_FULL, extract, extract_gyro, extract_pdf200, extract_speed
from signal_config import ACC_FULL_SCALE_G, ACC_LSB_PER_G, FS, N_PRE, WIN
from signals import mpu6050, stream_triggers, window_at

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "vzcrash" / "data"
OUT = ROOT / "data" / "processed"
GYRO_LSB_PER_DPS = 16.4      # MPU6050 at +-2000 deg/s
GYRO_FULL_SCALE = 2000.0
KEEP_RAW_EVERY = 400         # keep this many-th window's raw signal for the demo


def as_mpu6050_gyro(gyro_dps):
    g = np.clip(np.asarray(gyro_dps, np.float64), -GYRO_FULL_SCALE, GYRO_FULL_SCALE)
    return (np.round(g * GYRO_LSB_PER_DPS) / GYRO_LSB_PER_DPS).astype(np.float32)


def event_windows(acc, gyro, speed):
    """Yield (anchor, features, peak, raw_acc, raw_gyro) for every triggered window in one event."""
    for anchor in stream_triggers(acc):
        w_acc = window_at(acc, anchor)
        w_gyro = window_at(gyro, anchor)
        if w_acc is None or w_gyro is None:
            continue
        f_acc = extract(w_acc)
        f_gyro = extract_gyro(w_acc, w_gyro)
        pre_s = min(max((anchor - N_PRE) // FS, 0), len(speed) - 1)
        post_s = min((anchor + WIN - N_PRE) // FS, len(speed) - 1)
        f_speed = extract_speed(float(speed[pre_s]), float(speed[post_s]))
        f_pdf = extract_pdf200(w_acc, w_gyro, float(speed[pre_s]), float(speed[post_s]))
        yield anchor, np.concatenate([f_acc, f_gyro, f_speed]), f_pdf, float(f_acc[0]), w_acc, w_gyro


def process_row_group(task):
    path, rg = task
    table = pq.ParquetFile(path).read_row_group(rg, columns=["event_id", "label", "vehiclesize", "gsensor", "gyro", "gps_speed"])
    event_ids = table["event_id"].to_pylist()
    labels = table["label"].to_pylist()
    sizes = table["vehiclesize"].to_pylist()
    gs, gy, sp = table["gsensor"].to_pylist(), table["gyro"].to_pylist(), table["gps_speed"].to_pylist()

    feats, pdf_feats, meta, raws = [], [], [], []
    for i, eid in enumerate(event_ids):
        acc_raw = np.asarray(gs[i], np.float32)
        gyro_raw = np.asarray(gy[i], np.float32)
        speed = np.asarray(sp[i], np.float32)
        if acc_raw.ndim != 2 or len(acc_raw) < WIN + 10 or len(gyro_raw) != len(acc_raw) or len(speed) == 0:
            continue
        acc = mpu6050(np.clip(acc_raw, -ACC_FULL_SCALE_G, ACC_FULL_SCALE_G - 1 / ACC_LSB_PER_G))
        gyro = as_mpu6050_gyro(gyro_raw)
        found = list(event_windows(acc, gyro, speed))
        if not found:
            continue
        strongest = int(np.argmax([f[3] for f in found]))
        for k, (anchor, f, f_pdf, _peak, w_acc, w_gyro) in enumerate(found):
            primary = k == strongest
            if labels[i] == 0 and not primary:
                continue  # ambiguous: could be before or after the impact
            feats.append(f)
            pdf_feats.append(f_pdf)
            meta.append((eid, labels[i], anchor, int(primary), len(found), sizes[i]))
            raws.append((w_acc, w_gyro) if len(feats) % KEEP_RAW_EVERY == 0 else None)
    return feats, pdf_feats, meta, raws


def build_split(split, jobs):
    files = sorted(RAW.glob(f"{split}-*.parquet"))
    tasks = [(str(f), rg) for f in files for rg in range(pq.ParquetFile(f).num_row_groups)]
    feats, pdf_feats, meta, raw_acc, raw_gyro = [], [], [], [], []
    done = 0
    with ProcessPoolExecutor(jobs) as ex:
        for f, pf, m, r in ex.map(process_row_group, tasks, chunksize=1):
            feats += f
            pdf_feats += pf
            meta += m
            for item in r:
                if item is not None:
                    raw_acc.append(item[0])
                    raw_gyro.append(item[1])
            done += 1
            print(f"  {split}: {done}/{len(tasks)} row groups, {len(feats)} windows", end="\r", flush=True)

    X = np.asarray(feats, np.float32).reshape(-1, N_FEATURES_FULL)
    X_pdf = np.asarray(pdf_feats, np.float32).reshape(-1, 8)
    event_id = np.array([m[0] for m in meta], np.int64)
    label = np.array([m[1] for m in meta], np.int8)
    anchor = np.array([m[2] for m in meta], np.int32)
    primary = np.array([m[3] for m in meta], np.int8)
    n_windows = np.array([m[4] for m in meta], np.int16)
    size = np.array([m[5] for m in meta])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"vzcrash_{split}.npz", X=X, X_pdf=X_pdf, event_id=event_id, label=label, anchor=anchor,
                        primary=primary, n_windows=n_windows, vehiclesize=size,
                        raw_acc=np.asarray(raw_acc, np.float32), raw_gyro=np.asarray(raw_gyro, np.float32))
    events = len(np.unique(event_id))
    print(f"\n  {split}: {len(X)} windows from {events} events "
          f"(crash {int((label == 0).sum())}, near_miss {int((label == 1).sum())}, normal {int((label == 2).sum())}), "
          f"{len(raw_acc)} raw windows kept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=["validation", "test", "train"])
    args = ap.parse_args()
    for split in args.splits:
        build_split(split, args.jobs)


if __name__ == "__main__":
    main()
