"""Train the road-event classifier (model B): what kind of non-crash jolt was that?

Crash detection is model A, trained on VZCrash (real crashes). This model only names the jolts that
model A cleared: NORMAL, SPEED_BUMP, POTHOLE_ROUGH_ROAD or HARSH_MANEUVER. It uses the 30
accelerometer features and the PVS / Driver Behavior data built by build_dataset.py, scored with
leave-one-vehicle-out cross-validation.

Outputs ml/models/road_model.joblib and ml/results/road_model.json.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import _paths  # noqa: F401
from features import FEATURE_NAMES, N_FEATURES

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, MODELS = ROOT / "data" / "processed", ROOT / "results", ROOT / "models"
ROAD_CLASSES = ["NORMAL", "SPEED_BUMP", "POTHOLE_ROUGH_ROAD", "HARSH_MANEUVER"]
SEED = 42


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


MODELS_TO_TRY = {
    "XGBoost": lambda: XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9,
                                     colsample_bytree=0.8, n_jobs=8, random_state=SEED),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=2,
                                                    class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    "Logistic Regression": lambda: make_pipeline(FunctionTransformer(signed_log), StandardScaler(),
                                                 LogisticRegression(max_iter=3000, class_weight="balanced")),
}


def main():
    meta = pd.read_csv(PROC / "train_meta.csv")
    X = np.load(PROC / "features_train.npy")[:, :N_FEATURES]
    keep = (meta.label_name != "CRASH").to_numpy()
    meta, X = meta[keep].reset_index(drop=True), X[keep]
    y = np.array([ROAD_CLASSES.index(n) for n in meta.label_name])
    folds = meta.fold.to_numpy()
    real = (meta.is_aug == 0).to_numpy()
    print(meta[real].label_name.value_counts().to_string())

    results = {}
    for name, factory in MODELS_TO_TRY.items():
        oof = np.zeros((len(y), len(ROAD_CLASSES)))
        for k in sorted(set(folds)):
            tr, te = folds != k, folds == k
            model = factory()
            w = compute_sample_weight("balanced", y[tr])
            try:
                model.fit(X[tr], y[tr], sample_weight=w)
            except (TypeError, ValueError):
                model.fit(X[tr], y[tr])
            oof[te] = model.predict_proba(X[te])
        macro = f1_score(y[real], oof[real].argmax(1), average="macro")
        acc = float((oof[real].argmax(1) == y[real]).mean())
        results[name] = dict(macro_f1=macro, accuracy=acc,
                             per_class_f1=dict(zip(ROAD_CLASSES, f1_score(y[real], oof[real].argmax(1), average=None).tolist())))
        print(f"{name:22s} macro F1={macro:.4f} accuracy={acc:.4f}")
        results[name]["_oof"] = oof

    best = max(results, key=lambda n: results[n]["macro_f1"])
    oof = results[best].pop("_oof")
    for r in results.values():
        r.pop("_oof", None)
    print("best road model:", best)

    final = MODELS_TO_TRY[best]()
    try:
        final.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    except (TypeError, ValueError):
        final.fit(X, y)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "road_model.joblib")

    est = final[-1] if hasattr(final, "steps") else final
    imp = getattr(est, "feature_importances_", None)
    json.dump(dict(selected=best, classes=ROAD_CLASSES, comparison=results,
                   counts=meta[real].label_name.value_counts().to_dict(),
                   confusion=confusion_matrix(y[real], oof[real].argmax(1), labels=range(len(ROAD_CLASSES))).tolist(),
                   feature_importance=None if imp is None else dict(sorted(zip(FEATURE_NAMES, map(float, imp / imp.sum())), key=lambda kv: -kv[1]))),
              open(RES / "road_model.json", "w"), indent=2)
    np.save(RES / "road_oof.npy", oof)
    print(pd.DataFrame(confusion_matrix(y[real], oof[real].argmax(1), labels=range(len(ROAD_CLASSES))),
                       index=ROAD_CLASSES, columns=ROAD_CLASSES).to_string())


if __name__ == "__main__":
    main()
