"""Event-level precision-recall curves, written to ml/results/pr_curves.json."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import _paths  # noqa: F401
from features import N_FEATURES, N_FEATURES_EXT
from train_vzcrash import CRASH_LABEL, FIRMWARE_ALERT_P, event_scores, load, rates, signed_log, split_event_totals

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def thin(recall, precision, n=120):
    """Keep ~n points along the curve so it can be embedded in a document."""
    order = np.argsort(recall)
    r, p = recall[order], precision[order]
    grid = np.linspace(0, 1, n)
    idx = np.searchsorted(r, grid).clip(0, len(r) - 1)
    keep = np.unique(idx)
    return [[round(float(r[i]), 4), round(float(p[i]), 4)] for i in keep]


def main():
    tr, va, te = load("train"), load("validation"), load("test")
    totals = {s: split_event_totals(s) for s in ("validation", "test")}
    y_tr = (tr["label"] == CRASH_LABEL).astype(int)
    w = compute_sample_weight("balanced", y_tr)

    curves = {}

    # shipped model
    shipped = joblib.load(ROOT / "models" / "crash_model_vzcrash.joblib")
    p_te = shipped.predict_proba(te["X"][:, :N_FEATURES_EXT])[:, 1]
    runs = {"Proposed: LightGBM, 2.5 s window, 37 accel+gyro features": p_te}

    # V2 design (200 ms, 8 features)
    doc = make_pipeline(FunctionTransformer(signed_log), StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced"))
    doc.fit(tr["X_pdf"], y_tr, logisticregression__sample_weight=w)
    runs["Document V2: 200 ms window, 8 features, logistic"] = doc.predict_proba(te["X_pdf"])[:, 1]

    # accel only
    acc_only = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.9,
                             colsample_bytree=0.8, n_jobs=8, random_state=42)
    acc_only.fit(tr["X"][:, :N_FEATURES], y_tr, sample_weight=w)
    runs["Accelerometer only: XGBoost, 30 features"] = acc_only.predict_proba(te["X"][:, :N_FEATURES])[:, 1]

    # with GPS speed
    gps = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.9,
                        colsample_bytree=0.8, n_jobs=8, random_state=42)
    gps.fit(tr["X"], y_tr, sample_weight=w)
    runs["With GPS speed: XGBoost, 40 features"] = gps.predict_proba(te["X"])[:, 1]

    for name, p in runs.items():
        scores, truth = event_scores(te, p, totals["test"])
        prec, rec, _ = precision_recall_curve(truth, scores)
        at80 = rates(truth, scores, FIRMWARE_ALERT_P)
        curves[name] = dict(ap=float(average_precision_score(truth, scores)), curve=thin(rec, prec),
                            operating_point=dict(recall=at80["recall"], precision=at80["precision"],
                                                 fpr=at80["fpr"], threshold=FIRMWARE_ALERT_P))
        print(f"{name:62s} AP={curves[name]['ap']:.4f} recall@0.8={at80['recall']:.3f} precision@0.8={at80['precision']:.3f}")

    # old 2.2 g rule is a single point
    v1 = (te["X"][:, 4] > 2.2).astype(float)
    scores, truth = event_scores(te, v1, totals["test"])
    r = rates(truth, scores, 0.5)
    curves["V1 rule: |a| > 2.2 g (current firmware)"] = dict(
        ap=float(average_precision_score(truth, scores)), curve=[],
        operating_point=dict(recall=r["recall"], precision=r["precision"], fpr=r["fpr"], threshold=2.2))
    print(f"{'V1 rule |a| > 2.2 g':62s} AP={curves['V1 rule: |a| > 2.2 g (current firmware)']['ap']:.4f} "
          f"recall={r['recall']:.3f} precision={r['precision']:.3f}")

    json.dump(dict(curves=curves, test_events=totals["test"],
                   crash_prevalence=totals["test"]["crash"] / totals["test"]["total"]),
              open(RES / "pr_curves.json", "w"), indent=1)
    print("wrote", (RES / "pr_curves.json").name)


if __name__ == "__main__":
    main()
