"""Old NHTSA-based crash model, compared with leave-one-vehicle-out CV."""
import inspect
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import _paths  # noqa: F401
from features import FEATURE_NAMES, extract_many
from signal_config import CLASSES, CRASH, N_BASE

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, MODELS = ROOT / "data" / "processed", ROOT / "results", ROOT / "models"
K = len(CLASSES)
SEED = 42


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def scaled(model):
    return make_pipeline(FunctionTransformer(signed_log), StandardScaler(), model)


class ThresholdRule:
    """The existing firmware rule (V1): |a| > 2.2 g anywhere in the window means crash."""
    deployable = True

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        p = np.zeros((len(X), K))
        crash = X[:, FEATURE_NAMES.index("amag_peak")] > 2.2
        p[crash, CRASH] = 1.0
        p[~crash, 0] = 1.0
        return p


class CNN1D:
    """Small 1D-CNN on the raw orientation-invariant signals (vertical, horizontal, magnitude)."""

    def __init__(self, epochs=40):
        self.epochs = epochs

    @staticmethod
    def channels(W):
        g0 = W[:, :N_BASE].mean(axis=1, keepdims=True)
        u = g0 / np.linalg.norm(g0, axis=2, keepdims=True)
        d = W - g0
        v = (d * u).sum(axis=2)
        h = np.linalg.norm(d - v[..., None] * u, axis=2)
        m = np.linalg.norm(d, axis=2)
        x = np.stack([v, h, m], axis=1)
        return torch.tensor(np.sign(x) * np.log1p(4 * np.abs(x)), dtype=torch.float32)

    def build(self):
        return torch.nn.Sequential(
            torch.nn.Conv1d(3, 16, 7, padding=3), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(16, 32, 5, padding=2), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(32, 32, 5, padding=2), torch.nn.ReLU(), torch.nn.AdaptiveMaxPool1d(1),
            torch.nn.Flatten(), torch.nn.Linear(32, K))

    def fit(self, W, y, sample_weight):
        torch.manual_seed(SEED)
        self.net = self.build()
        X, Y = self.channels(W), torch.tensor(y)
        wts = torch.tensor(sample_weight, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 128):
                b = perm[i:i + 128]
                loss = (torch.nn.functional.cross_entropy(self.net(X[b]), Y[b], reduction="none") * wts[b]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, W):
        with torch.no_grad():
            return torch.softmax(self.net(self.channels(W)), dim=1).numpy()


def model_zoo():
    return {
        "V1 threshold rule (current)": (ThresholdRule, "features", True),
        "Logistic Regression": (lambda: scaled(LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")), "features", True),
        "Decision Tree": (lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=SEED), "features", True),
        "Random Forest": (lambda: RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), "features", True),
        "Extra Trees": (lambda: ExtraTreesClassifier(n_estimators=200, max_depth=14, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=SEED), "features", True),
        "XGBoost": (lambda: XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, n_jobs=8, random_state=SEED), "features", True),
        "LightGBM": (lambda: LGBMClassifier(n_estimators=250, num_leaves=15, learning_rate=0.06, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, min_child_samples=10, verbose=-1, random_state=SEED), "features", True),
        "SVM (RBF)": (lambda: scaled(SVC(C=10, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)), "features", False),
        "k-Nearest Neighbours": (lambda: scaled(KNeighborsClassifier(n_neighbors=7, weights="distance")), "features", False),
        "MLP neural network": (lambda: scaled(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=800, early_stopping=True, random_state=SEED)), "features", True),
        "1D-CNN (raw signal)": (CNN1D, "raw", False),
    }


def fit(model, X, y, w):
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if "sample_weight" not in inspect.signature(est.fit).parameters:
        return model.fit(X, y)
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def binary_metrics(y, proba, thr=None):
    pred_crash = proba.argmax(1) == CRASH if thr is None else proba[:, CRASH] >= thr
    true_crash = y == CRASH
    tp = int((pred_crash & true_crash).sum())
    fp = int((pred_crash & ~true_crash).sum())
    fn = int((~pred_crash & true_crash).sum())
    tn = int((~pred_crash & ~true_crash).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(recall=rec, precision=prec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                fpr=fp / max(fp + tn, 1), tp=tp, fp=fp, fn=fn, tn=tn)


def model_size_kb(model, name):
    """Approximate flash needed on the ESP32 when exported as C arrays."""
    if name.startswith("V1"):
        return 0.0
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "estimators_") and hasattr(est.estimators_[0], "tree_"):
        return sum(t.tree_.node_count for t in est.estimators_) * 16 / 1024
    if hasattr(est, "tree_"):
        return est.tree_.node_count * 16 / 1024
    if isinstance(est, XGBClassifier):
        return est.get_booster().trees_to_dataframe().shape[0] * 16 / 1024
    if isinstance(est, LGBMClassifier):
        return est.booster_.trees_to_dataframe().shape[0] * 16 / 1024
    if isinstance(est, LogisticRegression):
        return est.coef_.size * 4 / 1024
    if isinstance(est, MLPClassifier):
        return sum(c.size + i.size for c, i in zip(est.coefs_, est.intercepts_)) * 4 / 1024
    if isinstance(est, SVC):
        return est.support_vectors_.size * 4 / 1024
    if isinstance(est, KNeighborsClassifier):
        return est._fit_X.size * 4 / 1024
    if isinstance(est, CNN1D):
        return sum(p.numel() for p in est.net.parameters()) * 4 / 1024
    return float("nan")


def main():
    W = np.load(PROC / "train_windows.npy")
    meta = pd.read_csv(PROC / "train_meta.csv")
    W_ext = np.load(PROC / "external_windows.npy")
    meta_ext = pd.read_csv(PROC / "external_meta.csv")

    feat_path = PROC / "features_train.npy"
    if feat_path.exists() and len(np.load(feat_path)) == len(W):
        X, X_ext = np.load(feat_path), np.load(PROC / "features_external.npy")
    else:
        print("extracting features ...")
        X, X_ext = extract_many(W), extract_many(W_ext)
        np.save(feat_path, X)
        np.save(PROC / "features_external.npy", X_ext)
    y = meta.label.to_numpy()
    folds = meta.fold.to_numpy()
    real = meta.is_aug.to_numpy() == 0
    print(pd.crosstab(meta.label_name, meta.is_aug))

    rows, oof_all, ext_all = [], {}, {}
    for name, (factory, kind, deployable) in model_zoo().items():
        oof = np.zeros((len(y), K))
        t0 = time.time()
        for k in sorted(set(folds)):
            tr, te = folds != k, folds == k
            data = W if kind == "raw" else X
            w = compute_sample_weight("balanced", y[tr])
            model = fit(factory(), data[tr], y[tr], w)
            oof[te] = model.predict_proba(data[te])
        train_s = time.time() - t0

        final = fit(factory(), W if kind == "raw" else X, y, compute_sample_weight("balanced", y))
        ext_data = W_ext if kind == "raw" else X_ext
        t0 = time.perf_counter()
        p_ext = final.predict_proba(ext_data)
        infer_us = (time.perf_counter() - t0) / len(ext_data) * 1e6

        yr, pr = y[real], oof[real]
        bm = binary_metrics(yr, pr)
        bm80 = binary_metrics(yr, pr, thr=0.8)
        crash_real = real & (y == CRASH)
        speeds = meta.closing_speed.to_numpy()
        low = crash_real & (speeds <= 35)
        eq = meta.equivalent_speed.to_numpy()
        minor = (meta.is_aug.to_numpy() == 1) & (y == CRASH) & (eq <= 30)
        very_minor = minor & (eq <= 15)
        row = {
            "model": name,
            "deployable_on_esp32": deployable,
            "crash_recall": bm["recall"],
            "crash_precision": bm["precision"],
            "crash_f1": bm["f1"],
            "false_positive_rate": bm["fpr"],
            "false_alarms": bm["fp"],
            "missed_crashes": bm["fn"],
            "fpr_at_p80": bm80["fpr"],
            "recall_at_p80": bm80["recall"],
            "pr_auc": average_precision_score(yr == CRASH, pr[:, CRASH]),
            "macro_f1_5class": f1_score(yr, pr.argmax(1), average="macro") if not name.startswith("V1") else np.nan,
            "accuracy_5class": float((pr.argmax(1) == yr).mean()) if not name.startswith("V1") else np.nan,
            "recall_low_speed_crash_le35kmh": float((oof[low].argmax(1) == CRASH).mean()),
            "recall_minor_collision_stress": float((oof[minor].argmax(1) == CRASH).mean()),
            "recall_minor_collision_le15kmh": float((oof[very_minor].argmax(1) == CRASH).mean()),
            "v1_style_miss_rate_minor": float((X[minor, FEATURE_NAMES.index("amag_peak")] <= 2.2).mean()),
            "external_pothole_drive_false_alarm_rate": float((p_ext.argmax(1) == CRASH).mean()),
            "model_size_kb": model_size_kb(final, name),
            "host_inference_us": infer_us,
            "cv_train_seconds": train_s,
        }
        for c, cname in enumerate(CLASSES):
            if c != CRASH:
                mask = real & (y == c)
                row[f"fpr_{cname}"] = float((oof[mask].argmax(1) == CRASH).mean())
        rows.append(row)
        oof_all[name], ext_all[name] = oof, p_ext
        joblib.dump(final, MODELS / f"{name.split(' (')[0].replace(' ', '_').lower()}.joblib") if deployable and not name.startswith("V1") else None
        print(f"{name:30s} F1={row['crash_f1']:.4f} recall={row['crash_recall']:.4f} FPR={row['false_positive_rate']:.4%} "
              f"minor<=30={row['recall_minor_collision_stress']:.3f} minor<=15={row['recall_minor_collision_le15kmh']:.3f} "
              f"ext_FA={row['external_pothole_drive_false_alarm_rate']:.4f} macroF1={row['macro_f1_5class']:.3f} "
              f"size={row['model_size_kb']:.0f}KB")

    df = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / "model_comparison.csv", index=False)

    # crash F1 is nearly tied, so pick by macro F1 then FPR among ESP32-sized models
    cand = df[df.deployable_on_esp32 & ~df.model.str.startswith("V1")].copy()
    cand = cand[cand.crash_f1 >= cand.crash_f1.max() - 0.005].sort_values(
        ["macro_f1_5class", "false_positive_rate", "model_size_kb"], ascending=[False, True, True])
    best = cand.iloc[0].model
    print("\nBEST:", best)

    yr = y[real]
    cm = confusion_matrix(yr, oof_all[best][real].argmax(1), labels=range(K))
    cm_v1 = confusion_matrix(yr, oof_all["V1 threshold rule (current)"][real].argmax(1), labels=range(K))
    per_config = {}
    crash_real = real & (y == CRASH)
    for cfg, grp in meta[crash_real].groupby("configuration"):
        per_config[cfg] = dict(n=len(grp), recall=float((oof_all[best][grp.index].argmax(1) == CRASH).mean()))
    ext_labels = meta_ext.human_label.fillna("unreviewed")
    ext_classes = {h: pd.Series(np.array(CLASSES)[ext_all[best][ext_labels == h].argmax(1)]).value_counts().to_dict()
                   for h in ext_labels.unique()}
    est = joblib.load(MODELS / f"{best.split(' (')[0].replace(' ', '_').lower()}.joblib")
    importance = getattr(est[-1] if hasattr(est, "steps") else est, "feature_importances_", None)
    json.dump(dict(
        best_model=best, classes=CLASSES, features=FEATURE_NAMES,
        counts=meta[real].label_name.value_counts().to_dict(),
        counts_augmented=meta[~real].label_name.value_counts().to_dict(),
        external_windows=len(meta_ext), external_human_labels=ext_labels.value_counts().to_dict(),
        external_predicted_classes=ext_classes,
        confusion_best=cm.tolist(), confusion_v1=cm_v1.tolist(), crash_recall_by_configuration=per_config,
        n_crash_tests=int(meta[crash_real].group.nunique()),
        feature_importance=None if importance is None else dict(zip(FEATURE_NAMES, map(float, importance / importance.sum()))),
    ), open(RES / "metrics.json", "w"), indent=2)
    np.save(RES / "oof_best.npy", oof_all[best])
    joblib.dump(est, MODELS / "best_model.joblib")


if __name__ == "__main__":
    main()
