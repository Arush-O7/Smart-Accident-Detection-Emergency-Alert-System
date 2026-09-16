"""Train and compare crash detectors on VZCrash (real on-road crashes).

Uses the dataset's own train / validation / test splits: models are fitted on train, the alert
threshold is chosen on validation, and every number reported comes from the untouched test split.

Feature sets (all from signals this hardware can produce):
  accel            30 accelerometer features                       (today's firmware)
  accel+gyro       37 = accel + gyroscope                          (MPU6050, no extra parts)
  accel+gyro+gps   40 = accel + gyro + GPS speed change            (needs the NEO-6M wired up)
  doc-200ms         8 features over a 200 ms window                (Assessment-6 document, Section 1.1)

Two views of the same result:
  per window  every triggered 2.5 s window the device would classify
  per EVENT   the deployment question: during this 16 s event, did the device raise an alert?
              Crash events that never cross the 0.45 g trigger count as misses.

Outputs ml/results/vzcrash_comparison.csv and ml/results/metrics_vzcrash.json.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import _paths  # noqa: F401
from features import FEATURE_NAMES, FEATURE_NAMES_FULL, N_FEATURES, N_FEATURES_EXT, PDF200_FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, MODELS = ROOT / "data" / "processed", ROOT / "results", ROOT / "models"
VZ = ROOT / "data" / "raw" / "vzcrash" / "data"
SEED = 42
CRASH_LABEL = 0          # VZCrash: 0 crash, 1 near_miss, 2 normal_driving
FIRMWARE_ALERT_P = 0.80  # P(crash) that sends the LoRa alert immediately


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def scaled(model):
    return make_pipeline(FunctionTransformer(signed_log), StandardScaler(), model)


def load(split):
    d = np.load(PROC / f"vzcrash_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in ("X", "X_pdf", "event_id", "label", "anchor", "primary", "vehiclesize")}


def split_event_totals(split):
    """Every event in the split, including those that never cross the trigger."""
    labels = []
    for f in sorted(VZ.glob(f"{split}-*.parquet")):
        labels.append(pq.read_table(f, columns=["label"])["label"].to_numpy())
    labels = np.concatenate(labels)
    return {"total": len(labels), "crash": int((labels == 0).sum()),
            "near_miss": int((labels == 1).sum()), "normal": int((labels == 2).sum())}


def event_scores(data, p, totals):
    """Best crash score per event, with untriggered events scored 0 (the device stays silent)."""
    ids, inv = np.unique(data["event_id"], return_inverse=True)
    best = np.zeros(len(ids))
    np.maximum.at(best, inv, p)
    lab = np.zeros(len(ids), np.int8)
    lab[inv] = data["label"]
    seen_crash = int((lab == CRASH_LABEL).sum())
    missed_crash = totals["crash"] - seen_crash            # never triggered -> never alerted
    missed_negatives = totals["total"] - totals["crash"] - int((lab != CRASH_LABEL).sum())
    scores = np.concatenate([best, np.zeros(missed_crash + max(missed_negatives, 0))])
    truth = np.concatenate([lab == CRASH_LABEL, np.ones(missed_crash, bool), np.zeros(max(missed_negatives, 0), bool)])
    return scores, truth


def rates(truth, scores, thr):
    pred = scores >= thr
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(recall=rec, precision=prec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                fpr=fp / max(fp + tn, 1), tp=tp, fp=fp, fn=fn, tn=tn)


def threshold_for_event_fpr(truth, scores, target=0.005):
    """Lowest threshold whose event-level false-alarm rate stays under the target."""
    for thr in np.unique(np.round(scores, 3))[::-1]:
        if rates(truth, scores, thr)["fpr"] > target:
            return min(1.0, thr + 0.001)
    return 0.001


class ThresholdRule:
    """The firmware's original rule: |a| > 2.2 g anywhere in the window."""
    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        p = np.zeros((len(X), 2))
        hit = X[:, FEATURE_NAMES.index("amag_peak")] > 2.2
        p[hit, 1] = 1.0
        p[~hit, 0] = 1.0
        return p


FEATURE_SETS = {
    "accel": (slice(0, N_FEATURES), FEATURE_NAMES),
    "accel+gyro": (slice(0, N_FEATURES_EXT), FEATURE_NAMES_FULL[:N_FEATURES_EXT]),
    "accel+gyro+gps": (slice(0, len(FEATURE_NAMES_FULL)), FEATURE_NAMES_FULL),
}

RUNS = [
    # name, feature set, factory, deployable on the ESP32 today
    ("V1 rule |a| > 2.2 g (current firmware)", "accel", ThresholdRule, True),
    ("Document V2: 200 ms + 8 features + logistic", "doc-200ms", lambda: scaled(LogisticRegression(max_iter=3000, class_weight="balanced")), True),
    ("Document V2 features + XGBoost", "doc-200ms", lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, n_jobs=8, random_state=SEED), True),
    ("Logistic Regression", "accel+gyro", lambda: scaled(LogisticRegression(max_iter=3000, class_weight="balanced")), True),
    ("Decision Tree", "accel+gyro", lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=SEED), True),
    ("Random Forest", "accel+gyro", lambda: RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), True),
    ("Extra Trees", "accel+gyro", lambda: ExtraTreesClassifier(n_estimators=200, max_depth=16, min_samples_leaf=4, class_weight="balanced", n_jobs=-1, random_state=SEED), True),
    ("MLP neural network", "accel+gyro", lambda: scaled(MLPClassifier(hidden_layer_sizes=(96, 48), alpha=1e-3, max_iter=400, early_stopping=True, random_state=SEED)), True),
    ("LightGBM", "accel+gyro", lambda: LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.06, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED), True),
    ("XGBoost (accelerometer only)", "accel", lambda: XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.9, colsample_bytree=0.8, n_jobs=8, random_state=SEED), True),
    ("XGBoost (accel + gyro)", "accel+gyro", lambda: XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.9, colsample_bytree=0.8, n_jobs=8, random_state=SEED), True),
    ("XGBoost (accel + gyro + GPS)", "accel+gyro+gps", lambda: XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.9, colsample_bytree=0.8, n_jobs=8, random_state=SEED), True),
]


def features_for(data, name):
    if name == "doc-200ms":
        return data["X_pdf"]
    return data["X"][:, FEATURE_SETS[name][0]]


def main():
    tr, va, te = load("train"), load("validation"), load("test")
    totals = {s: split_event_totals(s) for s in ("train", "validation", "test")}
    print("events per split (all, including never-triggered):", json.dumps(totals))
    y_tr = (tr["label"] == CRASH_LABEL).astype(int)
    y_te = (te["label"] == CRASH_LABEL).astype(int)
    w_tr = compute_sample_weight("balanced", y_tr)

    rows, keep = [], {}
    for name, fset, factory, deployable in RUNS:
        Xtr, Xva, Xte = features_for(tr, fset), features_for(va, fset), features_for(te, fset)
        t0 = time.time()
        model = factory()
        try:
            model.fit(Xtr, y_tr, sample_weight=w_tr)
        except TypeError:
            model.fit(Xtr, y_tr)
        except ValueError:
            model.fit(Xtr, y_tr, **{f"{model.steps[-1][0]}__sample_weight": w_tr})
        fit_s = time.time() - t0

        p_va = model.predict_proba(Xva)[:, 1]
        t0 = time.perf_counter()
        p_te = model.predict_proba(Xte)[:, 1]
        infer_us = (time.perf_counter() - t0) / len(Xte) * 1e6

        s_va, t_va = event_scores(va, p_va, totals["validation"])
        thr = threshold_for_event_fpr(t_va, s_va, 0.005)
        s_te, t_te = event_scores(te, p_te, totals["test"])

        win = rates(y_te.astype(bool), p_te, 0.5)
        win80 = rates(y_te.astype(bool), p_te, FIRMWARE_ALERT_P)
        ev80 = rates(t_te, s_te, FIRMWARE_ALERT_P)
        evt = rates(t_te, s_te, thr)
        row = dict(
            model=name, features=fset, n_features=Xtr.shape[1], deployable_on_esp32=deployable,
            window_recall=win["recall"], window_precision=win["precision"], window_f1=win["f1"],
            window_fpr=win["fpr"], window_pr_auc=average_precision_score(y_te, p_te),
            window_roc_auc=roc_auc_score(y_te, p_te),
            event_pr_auc=average_precision_score(t_te, s_te),
            event_recall_at_p80=ev80["recall"], event_fpr_at_p80=ev80["fpr"],
            event_precision_at_p80=ev80["precision"], event_f1_at_p80=ev80["f1"],
            event_accuracy_at_p80=(ev80["tp"] + ev80["tn"]) / max(sum(ev80[k] for k in ("tp", "tn", "fp", "fn")), 1),
            event_recall_tuned=evt["recall"], event_fpr_tuned=evt["fpr"], event_precision_tuned=evt["precision"],
            event_f1_tuned=evt["f1"], tuned_threshold=thr,
            false_alarms_per_1000_events=evt["fpr"] * 1000, missed_crashes=evt["fn"], detected_crashes=evt["tp"],
            fit_seconds=fit_s, host_inference_us=infer_us,
        )
        rows.append(row)
        keep[name] = (model, p_te, s_te, t_te)
        print(f"{name:44s} eventAP={row['event_pr_auc']:.4f} recall@0.8={row['event_recall_at_p80']:.4f} "
              f"FPR@0.8={row['event_fpr_at_p80']:.4%} | tuned thr={thr:.3f} recall={row['event_recall_tuned']:.4f} "
              f"FPR={row['event_fpr_tuned']:.4%} | windowAP={row['window_pr_auc']:.4f} ({fit_s:.0f}s)")

    df = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / "vzcrash_comparison.csv", index=False)

    # Selected model: best event-level AP among ESP32-deployable models that need no extra hardware.
    cand = df[df.deployable_on_esp32 & (df.features != "accel+gyro+gps") & ~df.model.str.startswith("V1")]
    best_name = cand.sort_values("event_pr_auc", ascending=False).iloc[0].model
    best_model, p_te, s_te, t_te = keep[best_name]
    print("\nSELECTED:", best_name)

    # Where does it stand on severity, vehicle size and gyro resolution?
    crash_win = te["label"] == CRASH_LABEL
    peak = te["X"][:, 0]
    by_severity = {}
    for lo, hi, tag in [(0, 1.5, "<1.5 g"), (1.5, 3, "1.5-3 g"), (3, 6, "3-6 g"), (6, 99, ">6 g")]:
        m = crash_win & (peak >= lo) & (peak < hi)
        if m.sum():
            by_severity[tag] = dict(n=int(m.sum()), recall=float((p_te[m] >= FIRMWARE_ALERT_P).mean()))
    by_size = {}
    for size in np.unique(te["vehiclesize"]):
        m = crash_win & (te["vehiclesize"] == size)
        if m.sum():
            by_size[str(size)] = dict(n=int(m.sum()), recall=float((p_te[m] >= FIRMWARE_ALERT_P).mean()))

    best_est = best_model[-1] if hasattr(best_model, "steps") else best_model
    imp = getattr(best_est, "feature_importances_", None)
    names = FEATURE_SETS[df[df.model == best_name].iloc[0].features][1]
    json.dump(dict(
        selected=best_name, totals=totals, firmware_alert_p=FIRMWARE_ALERT_P,
        window_counts={s: dict(windows=int(len(d["X"])), crash=int((d["label"] == CRASH_LABEL).sum()))
                       for s, d in (("train", tr), ("validation", va), ("test", te))},
        crash_recall_by_peak_g=by_severity, crash_recall_by_vehicle_size=by_size,
        feature_importance=None if imp is None else dict(sorted(zip(names, map(float, imp / imp.sum())), key=lambda kv: -kv[1])),
        paper_baselines=dict(note="VZCrash paper, whole-dataset AP; our detector only sees events above the 0.45 g trigger",
                             physical_baseline=0.8962, cnn_rnn=0.9756, cnn_transformer=0.9754, chronos_bolt=0.9664),
    ), open(RES / "metrics_vzcrash.json", "w"), indent=2)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS / "crash_model_vzcrash.joblib")
    np.save(RES / "vzcrash_test_scores.npy", np.stack([s_te, t_te.astype(float)]))
    print(json.dumps(dict(by_severity=by_severity, by_size=by_size), indent=1))


if __name__ == "__main__":
    main()
