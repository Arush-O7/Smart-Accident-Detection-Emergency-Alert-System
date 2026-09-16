"""Turn the raw downloads into triggered 2.5 s event windows, as the ESP32 would see them.

Every stream is converted to g at 100 Hz, passed through an MPU6050 model (+-16 g, 2048 LSB/g)
and cut into windows by the same trigger logic the firmware runs.

Classes
  NORMAL              jolts on good/regular asphalt, non-aggressive manoeuvres        (PVS, DriverBehavior)
                      (PVS windows come from the dashboard MPU; the body MPU adds training-only views)
  SPEED_BUMP          labelled speed bumps                                             (PVS)
  POTHOLE_ROUGH_ROAD  jolts on labelled bad / unpaved / cobblestone roads              (PVS)
  HARSH_MANEUVER      aggressive braking, acceleration, turns, lane changes            (DriverBehavior)
  CRASH               real NHTSA crash-test pulses measured on the vehicle body, placed into
                      real PVS driving so the pre-crash and post-crash context is real sensor data;
                      speed-scaled copies (training + stress test only) cover minor collisions

Folds (leakage control): fold k holds PVS vehicle k (drives 1-3, 4-6, 7-9), one DriverBehavior trip
group and a third of the NHTSA tests. Crash windows in fold k only use fold-k driving as background,
so no sensor stream is shared across folds.

External test (never trained on): iPhone drives with human-labelled potholes (Austin, TX).
"""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from signal_config import CLASSES, FS, G, N_PRE, WIN
from signals import mpu6050, resample_rate, resample_uniform, stream_triggers, window_at

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
C = {name: i for i, name in enumerate(CLASSES)}
N_FOLDS = 3
rng = np.random.default_rng(2026)


class Store:
    def __init__(self):
        self.windows, self.rows = [], []

    def add(self, w, label, fold, source, group, desc, is_aug=0, **extra):
        self.windows.append(np.asarray(w, np.float32))
        self.rows.append(dict(label=C[label], label_name=label, fold=fold, source=source, group=group,
                              desc=desc, is_aug=is_aug, **extra))


# ------------------------------------------------------------------------------------ PVS
def load_pvs():
    drives = {}
    for n in range(1, 10):
        df = pd.read_parquet(RAW / "pvs" / f"pvs{n}.parquet")
        acc = mpu6050(df[["acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard"]].to_numpy() / G)
        body = mpu6050(df[["acc_x_above_suspension", "acc_y_above_suspension", "acc_z_above_suspension"]].to_numpy() / G)
        bump = (df.speed_bump_asphalt + df.speed_bump_cobblestone).to_numpy() > 0
        rough = ((df.bad_road_left > 0) | (df.unpaved_road > 0) | (df.cobblestone_road > 0)).to_numpy()
        drives[n] = dict(acc=acc, body=body, bump=bump, rough=rough, speed=df.speed.to_numpy(), fold=(n - 1) // 3)
    return drives


def pvs_events(store, drives):
    # The dashboard sensor is the evaluated view. The second MPU bolted to the body above the
    # suspension saw the same drive; its windows are extra training data (sensor placement variety).
    for n, d in drives.items():
        for view, is_aug in (("acc", 0), ("body", 1)):
            pvs_view_events(store, n, d, d[view], is_aug, "dashboard" if is_aug == 0 else "body")


def pvs_view_events(store, n, d, acc, is_aug, where):
    bump, rough = d["bump"], d["rough"]
    taken = np.zeros(len(acc), bool)
    for a in stream_triggers(acc):
        w = window_at(acc, a)
        if w is None:
            continue
        near = slice(max(a - 50, 0), a + 100)
        taken[near] = True
        if bump[near].any():
            label = "SPEED_BUMP"
        elif rough[a]:
            label = "POTHOLE_ROUGH_ROAD"
        else:
            label = "NORMAL"
        store.add(w, label, d["fold"], "PVS", f"pvs{n}", f"PVS drive {n} ({where}) @ {a / FS:.1f}s, "
                  f"{d['speed'][a] * 3.6:.0f} km/h", is_aug=is_aug, t=a / FS)
    # Some gentle bumps stay under the trigger; still teach the model what they look like.
    edges = np.flatnonzero(np.diff(bump.astype(np.int8)))
    for s, e in zip(edges[::2] + 1, edges[1::2] + 1):
        if taken[s:e].any():
            continue
        a = s + int(np.argmax(np.linalg.norm(np.diff(acc[s:e + 1], axis=0), axis=1)))
        w = window_at(acc, a)
        if w is not None:
            store.add(w, "SPEED_BUMP", d["fold"], "PVS", f"pvs{n}",
                      f"PVS drive {n} ({where}) @ {a / FS:.1f}s (below trigger)", is_aug=is_aug, t=a / FS)


# -------------------------------------------------------------------------- DriverBehavior
HARSH = {"freada_agressiva": "aggressive braking", "aceleracao_agressiva": "aggressive acceleration",
         "curva_direita_agressiva": "aggressive right turn", "curva_esquerda_agressiva": "aggressive left turn",
         "troca_faixa_direita_agressiva": "aggressive lane change", "troca_faixa_esquerda_agressiva": "aggressive lane change"}
TRIP_FOLD = {16: 0, 17: 1, 20: 2, 21: 0}


def load_driver_behavior():
    trips = {}
    for trip, fold in TRIP_FOLD.items():
        folder = RAW / "driver_behavior" / str(trip)
        df = pd.read_csv(folder / "acelerometro_terra.csv")
        t = (df.uptimeNanos.to_numpy() - df.uptimeNanos.iloc[0]) / 1e9
        _, acc = resample_uniform(t, df[["x", "y", "z"]].to_numpy() / G)
        gt = pd.read_csv(folder / "groundTruth.csv", skipinitialspace=True)
        trips[trip] = dict(acc=mpu6050(acc), gt=gt, fold=fold)
    return trips


def driver_behavior_events(store, trips):
    for trip, d in trips.items():
        acc = d["acc"]
        for _, ev in d["gt"].iterrows():
            s, e = int(ev.inicio * FS), int(ev.fim * FS)
            seg = acc[s:e]
            if len(seg) < 10:
                continue
            base = acc[max(s - 50, 0):s].mean(axis=0)
            a = s + int(np.argmax(np.linalg.norm(seg - base, axis=1)))
            label = "HARSH_MANEUVER" if ev.evento in HARSH else "NORMAL"
            kind = HARSH.get(ev.evento, "normal manoeuvre")
            for k, jitter in enumerate([0, -30, -15, 15, 30]):
                w = window_at(acc, a + jitter)
                if w is not None:
                    store.add(w, label, d["fold"], "DriverBehavior", f"trip{trip}",
                              f"Trip {trip}: {kind} @ {ev.inicio:.0f}s", is_aug=int(k > 0))


# ---------------------------------------------------------------------------------- NHTSA
def load_pulse(folder, pulse, meta):
    """Return the body acceleration pulse (SAE frame: X fwd, Y right, Z down, g) at 100 Hz from t=0."""
    axes = {}
    for axis in "XYZ":
        f = folder / f"v{pulse['vehicleNo']}_{axis}.tsv"
        if axis in pulse["curves"] and f.exists():
            raw = (pd.read_csv(f, sep=r"\s+", header=None, usecols=[0, 1], on_bad_lines="skip")
                   .apply(pd.to_numeric, errors="coerce").dropna().to_numpy())
            if len(raw) > 100:
                axes[axis] = raw
    if "X" not in axes and "Y" not in axes:
        return None
    ref = axes.get("X", axes.get("Y"))
    t = ref[:, 0]
    dt = float(np.median(np.diff(t)))
    if not (0 < dt <= 0.001):
        return None  # corrupt time base, or sampled too slowly to resolve the pulse
    sig = np.zeros((len(t), 3))
    for k, axis in enumerate("XYZ"):
        if axis in axes:
            sig[:, k] = np.interp(t, axes[axis][:, 0], axes[axis][:, 1])
    pre = t < 0
    sig -= sig[pre].mean(axis=0) if pre.sum() > 20 else sig[:5].mean(axis=0)
    sig100 = resample_rate(sig, 1.0 / dt)
    t100 = t[0] + np.arange(len(sig100)) * 0.01
    pulse100 = sig100[t100 >= 0]
    pulse100 = pulse100[: int(0.6 * FS)]
    if len(pulse100) < 15:
        return None
    ramp = np.linspace(1, 0, 5)[:, None]
    pulse100[-5:] *= ramp
    horiz_peak = np.linalg.norm(pulse100[:, :2], axis=1).max()
    dv = np.linalg.norm(pulse100[:, :2].sum(axis=0)) * 0.01 * G * 3.6
    if not (1.0 < horiz_peak < 250) or dv > 1.6 * meta["closingSpeed"] + 15 or np.abs(pulse100[-3:]).max() > 5:
        return None  # broken channel (baseline shift, clipped, wrong units)
    return pulse100


def random_frame(up):
    """Orthonormal forward/right/down vectors for a sensor whose gravity reading points along `up`."""
    down = -up / np.linalg.norm(up)
    x = rng.normal(size=3)
    fwd = x - np.dot(x, down) * down
    fwd /= np.linalg.norm(fwd)
    return fwd, np.cross(down, fwd), down


def rotate_onto(seg, target_mean):
    """Rotate a background segment so its gravity vector matches `target_mean` (keeps it seamless)."""
    m = seg.mean(axis=0)
    a, b = m / np.linalg.norm(m), target_mean / np.linalg.norm(target_mean)
    v, c = np.cross(a, b), float(np.dot(a, b))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx / (1 + c)
    return (seg - m) @ R.T + target_mean


def background_pools(drives, trips):
    pools = {k: dict(driving=[], rest=[]) for k in range(N_FOLDS)}
    for d in drives.values():
        moving, still = d["speed"] > 4.0, d["speed"] < 0.3
        pools[d["fold"]]["driving"].append((d["acc"], moving))
        pools[d["fold"]]["rest"].append((d["acc"], still))
    for d in trips.values():
        pools[d["fold"]]["driving"].append((d["acc"], np.ones(len(d["acc"]), bool)))
    return pools


def draw_segment(pool, length):
    starts_by_entry = []
    for acc, ok in pool:
        bad = np.concatenate([[0], np.cumsum(~ok)])
        starts_by_entry.append(np.flatnonzero(bad[length:] - bad[:-length] == 0))
    weights = np.array([len(x) for x in starts_by_entry], float)
    if weights.sum() == 0:
        raise RuntimeError("no background segment")
    k = rng.choice(len(pool), p=weights / weights.sum())
    s = int(rng.choice(starts_by_entry[k]))
    return pool[k][0][s:s + length].astype(np.float64)


def compose_crash(pulse, pools, fold, config, scale):
    PRE, TOTAL = 400, 700
    pool = pools[fold]
    struck_while_stopped = config == "IMPACTOR INTO VEHICLE" and rng.random() < 0.4
    pre = draw_segment(pool["rest" if struck_while_stopped else "driving"], TOTAL)
    after = draw_segment(pool["rest"] if rng.random() < 0.65 else pool["driving"], TOTAL)
    g_mean = pre[:PRE].mean(axis=0)
    after = rotate_onto(after, g_mean)
    blend = np.clip((np.arange(TOTAL) - PRE - 10) / 60.0, 0, 1)[:, None]
    acc = (1 - blend) * pre + blend * after
    fwd, right, down = random_frame(g_mean)
    p = pulse * scale
    body = np.outer(p[:, 0], fwd) + np.outer(p[:, 1], right) + np.outer(p[:, 2], down)
    acc[PRE:PRE + len(p)] += body
    acc = mpu6050(acc, rng)
    anchors = [a for a in stream_triggers(acc, start=300) if a >= 300]
    if not anchors or not (PRE - 3 <= anchors[0] <= PRE + 12):
        return None
    return window_at(acc, anchors[0])


def fold_of_test(test_no):
    return zlib.crc32(str(test_no).encode()) % N_FOLDS


def nhtsa_events(store, pools):
    n_tests = 0
    for meta_path in sorted((RAW / "nhtsa").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        fold = fold_of_test(meta["testNo"])
        kind = meta["configuration"].lower()
        for pulse_meta in meta["pulses"]:
            pulse = load_pulse(meta_path.parent, pulse_meta, meta)
            if pulse is None:
                continue
            n_tests += 1
            desc = (f"NHTSA test {meta['testNo']}: {meta['vehicle'].title()}, {kind}, "
                    f"{meta['closingSpeed']:.0f} km/h")
            # copy 0 = the real pulse (evaluated as-is). Other copies are scaled: in a linear-spring
            # crash model peak deceleration grows with impact speed while pulse duration stays put,
            # so scale s approximates the same vehicle hitting at s x the test speed. Two copies are
            # minor collisions down to 8 km/h equivalent.
            lo = min(0.5, max(0.08, 8.0 / meta["closingSpeed"]))
            plan = [1.0, float(rng.uniform(0.6, 1.0)), float(rng.uniform(lo, 0.5)), float(rng.uniform(lo, 0.5))]
            for k, scale in enumerate(plan):
                for _ in range(6):
                    w = compose_crash(pulse, pools, fold, meta["configuration"], scale)
                    if w is not None:
                        store.add(w, "CRASH", fold, "NHTSA", f"nhtsa{meta['testNo']}", desc, is_aug=int(k > 0),
                                  closing_speed=meta["closingSpeed"], configuration=meta["configuration"],
                                  pulse_scale=scale, equivalent_speed=meta["closingSpeed"] * scale)
                        break
    return n_tests


# ----------------------------------------------------------------------- external potholes
def pothole_external():
    """iPhone data has gravity removed; restore 1 g along the dominant vibration (vertical) axis.

    The features only use |vertical|, so the sign of that axis does not matter. Windows are taken at
    every human-reviewed event and at every other trigger during the drives.
    """
    labels = pd.read_csv(RAW / "potholes" / "labels.csv")
    store = Store()
    for d in range(1, 7):
        drive = f"drive_{d:02d}"
        df = pd.read_csv(RAW / "potholes" / drive / "sensor.csv")
        # logged at 50 Hz despite info.json saying 100 Hz
        t, user = resample_uniform(df.seconds_elapsed.to_numpy(), df[["accel_x", "accel_y", "accel_z"]].to_numpy() / G)
        _, vecs = np.linalg.eigh(np.cov((user - user.mean(axis=0)).T))
        acc = mpu6050(user + vecs[:, -1])
        dyn = np.linalg.norm(acc - acc.mean(axis=0), axis=1)
        lab = labels[labels.drive_id == drive]
        reviewed = []
        for _, ev in lab.iterrows():
            c = int(np.searchsorted(t, ev.t_center_s))
            if not 50 <= c < len(acc) - 50:
                continue
            a = c - 50 + int(np.argmax(dyn[c - 50:c + 50]))
            w = window_at(acc, a)
            if w is not None:
                reviewed.append(a)
                store.add(w, "NORMAL", -1, "PotholeDrives", drive, f"{drive} @ {a / FS:.1f}s ({ev.label})",
                          human_label=ev.label, t=a / FS)
        for a in stream_triggers(acc):
            w = window_at(acc, a)
            if w is not None and all(abs(a - r) > FS for r in reviewed):
                store.add(w, "NORMAL", -1, "PotholeDrives", drive, f"{drive} @ {a / FS:.1f}s (trigger)",
                          human_label="unreviewed trigger", t=a / FS)
    return store


def save(store, name):
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}_windows.npy", np.stack(store.windows))
    pd.DataFrame(store.rows).to_csv(OUT / f"{name}_meta.csv", index=False)


def main():
    store = Store()
    print("PVS drives ...")
    drives = load_pvs()
    pvs_events(store, drives)
    print("DriverBehavior trips ...")
    trips = load_driver_behavior()
    driver_behavior_events(store, trips)
    print("NHTSA crash pulses ...")
    n_pulses = nhtsa_events(store, background_pools(drives, trips))
    save(store, "train")
    meta = pd.DataFrame(store.rows)
    print(f"\n{len(meta)} windows ({n_pulses} valid crash pulses)")
    print(pd.crosstab(meta.label_name, [meta.fold, meta.is_aug]))

    print("\nExternal pothole drives ...")
    ext = pothole_external()
    save(ext, "external")
    print(pd.DataFrame(ext.rows).human_label.value_counts())


if __name__ == "__main__":
    main()
