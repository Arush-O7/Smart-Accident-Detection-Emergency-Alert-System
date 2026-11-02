"""Write demo/data/demo_data.js for the HTML dashboard."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import _paths  # noqa: F401
import build_dataset as bd
from build_vzcrash import as_mpu6050_gyro, event_windows
from features import FEATURE_NAMES_EXT, N_FEATURES, N_FEATURES_EXT
from signal_config import ACC_FULL_SCALE_G, ACC_LSB_PER_G, FS, G
from signals import mpu6050, stream_triggers, window_at

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "ml"
PROC = ML / "data" / "processed"
VZ = ML / "data" / "raw" / "vzcrash" / "data"
DEMO = ROOT / "demo" / "data"
DRIVE, T0, T1 = 7, 500.0, 700.0          # PVS drive 7 = vehicle 3, held out of the road model
ROAD_CLASSES = json.loads((ML / "results" / "road_model.json").read_text(encoding="utf-8"))["classes"]


def size_phrase(size):
    return "vehicle of unrecorded size" if size == "Unknown" else f"{size.lower()}-duty vehicle"
rng = np.random.default_rng(11)


def int16(a):
    return np.round(np.asarray(a) * 2048).astype(int).ravel().tolist()


def int16_gyro(a):
    return np.round(np.asarray(a) * 16.4).astype(int).ravel().tolist()


def vzcrash_events(n_per_class=(14, 6, 6)):
    """Sample real test-split events: crashes across severities, near-misses, normal driving."""
    crash_model = joblib.load(ML / "models" / "crash_model_vzcrash.joblib")
    road_model = joblib.load(ML / "models" / "road_model.joblib")
    picked = {0: [], 1: [], 2: []}
    want = dict(zip((0, 1, 2), n_per_class))
    for f in sorted(VZ.glob("test-*.parquet")):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            if all(len(picked[k]) >= want[k] for k in picked):
                break
            t = pf.read_row_group(rg, columns=["event_id", "label", "vehiclesize", "gsensor", "gyro", "gps_speed"])
            labels = t["label"].to_pylist()
            for i, lab in enumerate(labels):
                if len(picked[lab]) >= want[lab] or rng.random() < 0.985:
                    continue
                acc_raw = np.asarray(t["gsensor"][i].as_py(), np.float32)
                gyro_raw = np.asarray(t["gyro"][i].as_py(), np.float32)
                speed = np.asarray(t["gps_speed"][i].as_py(), np.float32)
                if acc_raw.ndim != 2 or len(acc_raw) < 300:
                    continue
                acc = mpu6050(np.clip(acc_raw, -ACC_FULL_SCALE_G, ACC_FULL_SCALE_G - 1 / ACC_LSB_PER_G))
                gyro = as_mpu6050_gyro(gyro_raw)
                found = list(event_windows(acc, gyro, speed))
                if not found:
                    continue
                anchor, feats, _pdf, peak, w_acc, w_gyro = max(found, key=lambda x: x[3])
                p = float(crash_model.predict_proba(feats[None, :N_FEATURES_EXT])[0, 1])
                road = road_model.predict_proba(feats[None, :N_FEATURES])[0]
                picked[lab].append(dict(
                    source="VZCrash", truth=["CRASH", "NEAR_MISS", "NORMAL_DRIVING"][lab],
                    why={0: "real crash, reviewed from dashcam footage", 1: "near miss: evasive manoeuvre, no impact",
                         2: "normal driving"}[lab],
                    desc=f"VZCrash event {t['event_id'][i].as_py()} ({size_phrase(t['vehiclesize'][i].as_py())}), "
                         f"{speed[0]:.0f} to {speed[-1]:.0f} km/h, peak {peak:.1f} g",
                    samples=int16(w_acc), gyro=int16_gyro(w_gyro), p_crash=round(p, 4),
                    road_probs=[round(float(x), 4) for x in road],
                    v1_alarm=bool(feats[4] > 2.2), peak_g=round(float(peak), 2), dv_kmh=round(float(feats[11]), 1),
                    w_peak=round(float(feats[30]), 0)))
    out = picked[0] + picked[1] + picked[2]
    out.sort(key=lambda e: -e["peak_g"])
    return out


def road_events(n=12):
    """PVS / DriverBehavior events with the road model's out-of-fold verdict."""
    meta = pd.read_csv(PROC / "train_meta.csv")
    X = np.load(PROC / "features_train.npy")
    W = np.load(PROC / "train_windows.npy")
    oof = np.load(ML / "results" / "road_oof.npy")
    keep = (meta.label_name != "CRASH").to_numpy()
    meta_r = meta[keep].reset_index(drop=True)
    X_r, W_r = X[keep], W[keep]
    crash_model = joblib.load(ML / "models" / "crash_model_vzcrash.joblib")

    real = meta_r[(meta_r.is_aug == 0)]
    picks = []
    for label, count, why in [("SPEED_BUMP", 4, "labelled speed bump"),
                              ("POTHOLE_ROUGH_ROAD", 4, "pothole / broken road"),
                              ("HARSH_MANEUVER", 2, "aggressive braking or cornering"),
                              ("NORMAL", 2, "ordinary jolt")]:
        pool = real[real.label_name == label]
        rough_first = pool[X_r[pool.index, 4] > 2.2]           # prefer ones that trip the old rule
        idx = list(rough_first.index[:count // 2]) + list(rng.choice(pool.index, count, replace=False))
        for i in list(dict.fromkeys(idx))[:count]:
            picks.append((int(i), why))

    out = []
    for i, why in picks:
        row = meta_r.loc[i]
        feats = np.concatenate([X_r[i, :N_FEATURES], np.zeros(N_FEATURES_EXT - N_FEATURES, np.float32)])
        p = float(crash_model.predict_proba(feats[None])[0, 1])
        out.append(dict(source="PVS", truth=row.label_name, why=why, desc=row.desc,
                        samples=int16(W_r[i]), gyro=None, p_crash=round(p, 4),
                        road_probs=[round(float(x), 4) for x in oof[i]],
                        v1_alarm=bool(X_r[i, 4] > 2.2), peak_g=round(float(X_r[i, 0]), 2),
                        dv_kmh=round(float(X_r[i, 11]), 1), w_peak=None))
    return out


def replay_stream():
    """Real drive (PVS vehicle 3) + a real VZCrash crash spliced on, in the drive's sensor frame."""
    df = pd.read_parquet(ML / "data" / "raw" / "pvs" / f"pvs{DRIVE}.parquet")
    acc = mpu6050(df[["acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard"]].to_numpy() / G)
    gyro = as_mpu6050_gyro(df[["gyro_x_dashboard", "gyro_y_dashboard", "gyro_z_dashboard"]].to_numpy())
    speed = df.speed.to_numpy() * 3.6
    a, b = int(T0 * FS), int(T1 * FS)
    seg, seg_gyro, seg_speed = acc[a:b].astype(np.float64), gyro[a:b].astype(np.float64), speed[a:b]

    # pick a clearly visible crash from the test split
    crash = None
    pf = pq.ParquetFile(sorted(VZ.glob("test-*.parquet"))[0])
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["event_id", "label", "vehiclesize", "gsensor", "gyro", "gps_speed"])
        for i, lab in enumerate(t["label"].to_pylist()):
            if lab != 0:
                continue
            ca = np.asarray(t["gsensor"][i].as_py(), np.float32)
            cg = np.asarray(t["gyro"][i].as_py(), np.float32)
            cs = np.asarray(t["gps_speed"][i].as_py(), np.float32)
            if ca.ndim != 2 or len(ca) < 1000:
                continue
            base = ca[:200].mean(axis=0)
            peak = np.linalg.norm(ca - base, axis=1).max()
            if 2.5 < peak < 12 and cs[0] > 25:
                crash = dict(acc=ca.astype(np.float64), gyro=cg.astype(np.float64), speed=cs,
                             event_id=int(t["event_id"][i].as_py()), size=t["vehiclesize"][i].as_py(),
                             peak=float(peak))
                break
        if crash:
            break

    # rotate crash into the drive's frame
    g_mean = seg[-300:].mean(axis=0)
    crash_acc = bd.rotate_onto(crash["acc"], g_mean)
    R_ = np.linalg.lstsq(crash["acc"] - crash["acc"].mean(axis=0), crash_acc - crash_acc.mean(axis=0), rcond=None)[0]
    crash_gyro = crash["gyro"] @ R_
    fade = np.clip(np.arange(50) / 50.0, 0, 1)[:, None]
    crash_acc[:50] = (1 - fade) * seg[-1] + fade * crash_acc[:50]

    stream = mpu6050(np.concatenate([seg, crash_acc]))
    stream_gyro = as_mpu6050_gyro(np.concatenate([seg_gyro, crash_gyro]))
    speeds = np.concatenate([seg_speed[::FS], crash["speed"]])

    bumps = (df.speed_bump_asphalt.to_numpy() + df.speed_bump_cobblestone.to_numpy())[a:b] > 0
    edges = np.flatnonzero(np.diff(np.r_[0, bumps.astype(int), 0]))
    truth = [dict(t=round(s / FS, 2), end=round(e / FS, 2), label="SPEED_BUMP (labelled)")
             for s, e in zip(edges[::2], edges[1::2])]
    crash_peak_t = (len(seg) + int(np.argmax(np.linalg.norm(crash_acc - crash_acc[:200].mean(axis=0), axis=1)))) / FS
    truth.append(dict(t=round(crash_peak_t - 0.3, 2), end=round(crash_peak_t + 0.3, 2),
                      label=f"CRASH (VZCrash event {crash['event_id']})"))
    rough = ((df.bad_road_left > 0) | (df.unpaved_road > 0) | (df.cobblestone_road > 0)).to_numpy()[a:b]
    road = ["rough" if rough[i:i + FS].mean() > 0.5 else "smooth" for i in range(0, len(rough), FS)]
    road += ["crash recording"] * (len(speeds) - len(road))
    return dict(fs=FS, samples=int16(stream), gyro=int16_gyro(stream_gyro), road=road,
                speed_kmh=[round(float(v), 1) for v in speeds],
                source=f"PVS drive {DRIVE} ({T0:.0f}-{T1:.0f} s, Fiat Palio, dashboard MPU-9250) followed by "
                       f"VZCrash event {crash['event_id']}: a real crash, {size_phrase(crash['size'])}, "
                       f"{crash['speed'][0]:.0f} km/h, peak {crash['peak']:.1f} g",
                truth=truth, crash_time=round(crash_peak_t, 2))


def main():
    DEMO.mkdir(parents=True, exist_ok=True)
    events = vzcrash_events() + road_events()
    comparison = pd.read_csv(ML / "results" / "vzcrash_comparison.csv")
    vz_metrics = json.loads((ML / "results" / "metrics_vzcrash.json").read_text(encoding="utf-8"))
    road_metrics = json.loads((ML / "results" / "road_model.json").read_text(encoding="utf-8"))
    data = dict(replay=replay_stream(), events=events,
                comparison=json.loads(comparison.to_json(orient="records")),
                vzcrash=vz_metrics, road=road_metrics,
                road_classes=ROAD_CLASSES, features=FEATURE_NAMES_EXT)
    (DEMO / "demo_data.js").write_text("window.DEMO_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    for f in ("model.js", "demo_data.js"):
        print(f, (DEMO / f).stat().st_size // 1024, "KB")
    print(f"{len(events)} events; replay {len(data['replay']['samples']) // 3} samples, "
          f"crash at {data['replay']['crash_time']} s")


if __name__ == "__main__":
    main()
