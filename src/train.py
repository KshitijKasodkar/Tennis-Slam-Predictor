"""Step 2 - tune, select, calibrate and test the match-level models.

Protocol (fixed before any model was compared):
  train 2010-2021  -> hyper-parameters tuned with TimeSeriesSplit (past folds predict future folds)
  val   2022       -> model selection (raw log-loss) + fitting the Platt calibrator
  test  2023-2024  -> touched once, after everything above is frozen

    python -m src.train
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config as C
from .calib import EPS, Platt, logit

warnings.filterwarnings("ignore", category=FutureWarning)
MODEL_DIR = C.PROC_DIR / "models"
NAMES = {"rank_only": "Rank-only logistic (baseline)", "lasso": "Logistic regression + Lasso",
         "svm": "SVM (RBF kernel)", "rf": "Random forest", "xgb": "XGBoost"}


# ----------------------------------------------------------------------------- helpers
def metrics(y, p):
    return {"log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS))),
            "accuracy": float(accuracy_score(y, p >= 0.5)),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "n": int(len(y))}


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 10:
            out.append({"bin": b, "pred": float(p[m].mean()), "obs": float(y[m].mean()), "n": int(m.sum())})
    return out


def paired_bootstrap(y, p1, p2, reps=2000, seed=C.SEED):
    """95% CI for mean per-match log-loss(p1) - log-loss(p2). Negative = model 1 better."""
    rng = np.random.default_rng(seed)
    l1 = -(y * np.log(np.clip(p1, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p1, EPS, 1)))
    l2 = -(y * np.log(np.clip(p2, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p2, EPS, 1)))
    d = l1 - l2
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
    return {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5))}


def candidates():
    imp = lambda: SimpleImputer(strategy="median")        # medians learned from training folds only
    return {
        "lasso": (Pipeline([("imp", imp()), ("sc", StandardScaler()),
                            ("clf", LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=2000))]),
                  {"clf__C": [0.001, 0.01, 0.1, 1.0]}),
        "svm": (Pipeline([("imp", imp()), ("sc", StandardScaler()),
                          ("clf", SVC(kernel="rbf", probability=True, random_state=C.SEED))]),
                {"clf__C": [0.3, 1.0, 3.0]}),
        "rf": (Pipeline([("imp", imp()),
                         ("clf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                        n_jobs=-1, random_state=C.SEED))]),
               {"clf__min_samples_leaf": [20, 50, 100]}),
        "xgb": (XGBClassifier(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=5, eval_metric="logloss", n_jobs=4,
                              random_state=C.SEED),                         # handles NaN natively
                {"n_estimators": [200, 500], "max_depth": [2, 3, 4]}),
    }


# ----------------------------------------------------------------------------- main
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    f = pd.read_pickle(C.PROC_DIR / "features.pkl")
    f = f[f.year >= C.TRAIN_YEARS[0]]
    n_ret = int(f.retired.sum())
    f = f[~f.retired].reset_index(drop=True)          # retirements: outcome reflects injury, not form
    tr = f[f.year.between(*C.TRAIN_YEARS)]
    va = f[f.year == C.VAL_YEAR]
    te = f[f.year.between(*C.TEST_YEARS)]
    X = lambda d: d[C.FEATURES].astype(float)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  (retirements excluded: {n_ret})")

    models, cv_info, raw = {}, {}, {"val": {}, "test": {}}
    # --- baseline: one feature, no tuning
    rank_only = Pipeline([("clf", LogisticRegression())]).fit(tr[["log_rank_diff"]], tr.y)
    models["rank_only"] = rank_only
    for split, d in (("val", va), ("test", te)):
        raw[split]["rank_only"] = rank_only.predict_proba(d[["log_rank_diff"]])[:, 1]
    # --- candidates: tuned on the training years only
    tss = TimeSeriesSplit(n_splits=4)
    for key, (est, grid) in candidates().items():
        Xtr, ytr = X(tr), tr.y
        if key == "svm":                                  # O(n^2) kernel: most recent 12k training rows
            Xtr, ytr = Xtr.iloc[-12000:], ytr.iloc[-12000:]
        gs = GridSearchCV(est, grid, cv=tss, scoring="neg_log_loss", n_jobs=1 if key in ("rf", "xgb") else -1)
        gs.fit(Xtr, ytr)
        models[key] = gs.best_estimator_
        cv_info[key] = {"best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                        "cv_log_loss": float(-gs.best_score_), "train_rows": int(len(Xtr)),
                        "grid": [{"params": {k.replace("clf__", ""): v for k, v in prm.items()},
                                  "cv_log_loss": float(-s)}
                                 for prm, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"])]}
        for split, d in (("val", va), ("test", te)):
            raw[split][key] = models[key].predict_proba(X(d))[:, 1]
        print(f"{key:6s} cv={cv_info[key]['cv_log_loss']:.4f} val={log_loss(va.y, raw['val'][key]):.4f} {cv_info[key]['best_params']}")

    # --- selection on validation (raw probabilities), calibrators fitted on validation
    val_metrics = {k: metrics(va.y.values, p) for k, p in raw["val"].items()}
    selected = min([k for k in raw["val"] if k != "rank_only"], key=lambda k: val_metrics[k]["log_loss"])
    web_model = min(["xgb", "lasso"], key=lambda k: val_metrics[k]["log_loss"])   # exportable to JS
    cal = {k: Platt().fit(p, va.y.values) for k, p in raw["val"].items()}
    print("selected:", selected, "| website model:", web_model)

    # --- test: evaluated once, all models reported
    yt = te.y.values
    test_raw = {k: metrics(yt, p) for k, p in raw["test"].items()}
    test_cal = {k: metrics(yt, cal[k](p)) for k, p in raw["test"].items()}
    p_sel = cal[selected](raw["test"][selected])
    p_base = cal["rank_only"](raw["test"]["rank_only"])
    p_lasso = cal["lasso"](raw["test"]["lasso"])
    fav_acc = float(((te.A_rank.fillna(9999) < te.B_rank.fillna(9999)) == (yt == 1)).mean())

    # --- error analysis on the test years (selected model, calibrated)
    t = te.assign(p=p_sel, p_base=p_base)
    t["p_winner"] = np.where(t.y == 1, t.p, 1 - t.p)
    t["correct"] = t.p_winner > 0.5
    t["gap"] = pd.cut(t.log_rank_diff.abs(), [-0.01, 0.5, 1.0, 2.0, 99],
                      labels=["similar rank (<1.6x)", "1.6-2.7x", "2.7-7.4x", ">7.4x"])
    def grp(col):
        out = []
        for g, d in t.groupby(col, observed=True):
            out.append({"group": str(g), "n": int(len(d)), **{k: round(v, 4) for k, v in metrics(d.y.values, d.p.values).items() if k != "n"},
                        "base_log_loss": float(log_loss(d.y, np.clip(d.p_base, EPS, 1 - EPS), labels=[0, 1]))})
        return out
    t["level_g"] = np.where(t.level == "G", "Grand Slam", "Other tour-level")
    misses = t[~t.correct].nsmallest(12, "p_winner")
    misses = [{"date": r.date.strftime("%Y-%m-%d"), "event": r.tourney_name, "round": r["round"],
               "surface": r.surface, "winner": r.A_name if r.y == 1 else r.B_name,
               "loser": r.B_name if r.y == 1 else r.A_name,
               "winner_rank": None if pd.isna(r.A_rank if r.y == 1 else r.B_rank) else int(r.A_rank if r.y == 1 else r.B_rank),
               "loser_rank": None if pd.isna(r.B_rank if r.y == 1 else r.A_rank) else int(r.B_rank if r.y == 1 else r.A_rank),
               "p_winner": round(float(r.p_winner), 3)} for _, r in misses.iterrows()]

    # --- interpretation: permutation importance on validation (never on test)
    pi = permutation_importance(models[selected], X(va), va.y, scoring="neg_log_loss",
                                n_repeats=5, random_state=C.SEED)
    importance = sorted([{"feature": c, "mean": float(m), "std": float(s)} for c, m, s in
                         zip(C.FEATURES, pi.importances_mean, pi.importances_std)], key=lambda d: -d["mean"])
    coefs = models["lasso"].named_steps["clf"].coef_[0]
    lasso_coef = [{"feature": c, "coef": float(v)} for c, v in zip(C.FEATURES, coefs)]

    for k, m in models.items():
        joblib.dump(m, MODEL_DIR / f"{k}.joblib")
    joblib.dump(cal, MODEL_DIR / "calibrators.joblib")

    results = {
        "split": {"train": list(C.TRAIN_YEARS), "val": C.VAL_YEAR, "test": list(C.TEST_YEARS),
                  "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                  "retirements_excluded": n_ret},
        "names": NAMES, "features": C.FEATURES, "cv": cv_info,
        "val": val_metrics, "selected": selected, "web_model": web_model,
        "calibration": {k: {"a": c.a, "b": c.b} for k, c in cal.items()},
        "test_raw": test_raw, "test_cal": test_cal,
        "favourite_accuracy_test": fav_acc,
        "bootstrap": {"selected_vs_rank_only": paired_bootstrap(yt, p_sel, p_base),
                      "selected_vs_lasso": paired_bootstrap(yt, p_sel, p_lasso)},
        "reliability": {"raw": reliability(yt, raw["test"][selected]), "cal": reliability(yt, p_sel),
                        "rank_only": reliability(yt, p_base)},
        "by_surface": grp("surface"), "by_level": grp("level_g"), "by_rank_gap": grp("gap"),
        "worst_misses": misses, "permutation_importance_val": importance, "lasso_coef": lasso_coef,
    }
    (C.PROC_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"test_cal": {k: round(v["log_loss"], 4) for k, v in test_cal.items()},
                      "bootstrap": results["bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
