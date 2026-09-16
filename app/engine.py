"""Crash AI engine for the Streamlit app.

Runs the exact models the ESP32 runs: the same trigger, the same features, and the same flattened
trees exported by ml/scripts/export_model.py. Needs numpy only, so the hosted app stays small.

  model A  crash detector      LightGBM, 37 accel+gyro features, sigmoid(sum of leaves)
  model B  road-event namer    XGBoost, 30 accelerometer features, softmax over 4 classes
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from features import extract, extract_gyro  # noqa: E402
from signal_config import EMA_ALPHA, N_POST, N_PRE, TRIGGER_G, WIN  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
ACC_LSB, GYRO_LSB = 2048.0, 16.4


def load():
    arrays = np.load(DATA / "models.npz")
    meta = json.loads((DATA / "meta.json").read_text())
    models = {}
    for name in ("crash", "road"):
        models[name] = dict(
            feat=arrays[f"{name}_feat"], thr=arrays[f"{name}_thr"], no=arrays[f"{name}_no"],
            roots=arrays[f"{name}_roots"], bias=arrays[f"{name}_bias"], **meta["models"][name])
    return models, meta


def margins(model, x):
    feat, thr, no, roots = model["feat"], model["thr"], model["no"], model["roots"]
    k = model["n_classes"]
    out = model["bias"].astype(np.float64).copy()
    le = model["cmp"] == "<="
    xf = np.asarray(x, np.float32)
    for t, root in enumerate(roots):
        i = root
        while feat[i] >= 0:
            v = xf[feat[i]]
            i = i + 1 if (v <= thr[i] if le else v < thr[i]) else no[i]
        out[t % k] += thr[i]
    return out


def crash_probability(models, x):
    return float(1.0 / (1.0 + np.exp(-margins(models["crash"], x)[0])))


def road_probabilities(models, x):
    m = margins(models["road"], x)
    e = np.exp(m - m.max())
    return e / e.sum()


def window_features(acc_window, gyro_window):
    """37 features: 30 accelerometer + 7 gyroscope, as the firmware computes them."""
    return np.concatenate([extract(acc_window), extract_gyro(acc_window, gyro_window)])


def classify(models, acc_window, gyro_window):
    x = window_features(acc_window, gyro_window)
    p = crash_probability(models, x)
    road = road_probabilities(models, x[:30])
    return dict(p_crash=p, road=road, features=x, peak_g=float(x[0]), dv_kmh=float(x[11]),
                w_peak=float(x[30]), v1_alarm=bool(x[4] > 2.2))


class Detector:
    """Streaming detector, identical in behaviour to ml_crash_detector.ino."""

    def __init__(self, models, trigger_g=TRIGGER_G, alpha=EMA_ALPHA):
        self.models, self.trigger_g, self.alpha = models, trigger_g, alpha
        self.acc = np.zeros((WIN, 3), np.float32)
        self.gyro = np.zeros((WIN, 3), np.float32)
        self.head = self.i = 0
        self.ema = None
        self.collecting = False
        self.anchor = self.peak = 0

    def push(self, a, w):
        self.acc[self.head] = a
        self.gyro[self.head] = w
        self.head = (self.head + 1) % WIN
        if self.ema is None:
            self.ema = np.array(a, np.float64)
        self.ema += self.alpha * (np.asarray(a) - self.ema)
        dyn = float(np.linalg.norm(np.asarray(a) - self.ema))
        i = self.i
        self.i += 1
        event = None
        if not self.collecting:
            if i >= N_PRE and dyn > self.trigger_g:
                self.collecting, self.anchor, self.peak = True, i, dyn
        else:
            if i - self.anchor > 10 and dyn > max(1.0, 1.5 * self.peak):
                self.anchor, self.peak = i, dyn
            self.peak = max(self.peak, dyn)
            if i == self.anchor + N_POST - 1:
                order = (np.arange(WIN) + self.head) % WIN
                event = classify(self.models, self.acc[order], self.gyro[order])
                event["trigger"] = self.anchor
                self.collecting = False
        return dyn, event


def run_stream(models, acc, gyro, alert_p=0.80):
    """Feed a whole recording through the detector; returns the events it would have classified."""
    det = Detector(models)
    events = []
    for a, w in zip(acc, gyro):
        _, ev = det.push(a, w)
        if ev:
            ev["t"] = ev["trigger"] / 100.0
            ev["verdict"] = "CRASH" if ev["p_crash"] >= 0.5 else None
            events.append(ev)
    return events


def v1_alarms(acc, cooldown=300):
    """The original firmware rule: |a| > 2.2 g, with its 3 s cooldown."""
    mag = np.linalg.norm(acc, axis=1)
    out, last = [], -10 ** 9
    for i in np.flatnonzero(mag > 2.2):
        if i - last > cooldown:
            out.append(int(i))
            last = i
    return out


def to_g(counts):
    return np.asarray(counts, np.float32) / ACC_LSB


def to_dps(counts):
    return np.asarray(counts, np.float32) / GYRO_LSB


def split_components(acc_window, n_base=50):
    """Gravity-referenced vertical / horizontal split, for plotting."""
    a = np.asarray(acc_window, np.float64)
    g0 = a[:n_base].mean(axis=0)
    gmag = np.linalg.norm(g0)
    u = g0 / gmag if gmag > 0.3 else np.array([0.0, 0.0, 1.0])
    d = a - g0
    v = d @ u
    h = np.linalg.norm(d - np.outer(v, u), axis=1)
    return v, h
