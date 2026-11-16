"""Step 4 - export everything the static website needs into docs/data/.

docs/data/results.json  metrics, tables, error analysis, Slam backtest scores
docs/data/slams.json    Monte Carlo output per Slam (title odds, route to final)
docs/data/h2h.json      player snapshot at the data cut-off + the calibrated model itself
                        (XGBoost trees or Lasso coefficients) so the browser can score any pair

    python -m src.export_site
"""
import json
import math
import pickle

import joblib
import numpy as np

from . import config as C

MODEL_DIR = C.PROC_DIR / "models"


def nan_to_none(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {str(k): nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nan_to_none(v) for v in o]
    if isinstance(o, (np.floating,)):
        return nan_to_none(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def export_xgb(model):
    booster = model.get_booster()
    j = json.loads(booster.save_raw("json"))
    base = float(str(j["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    trees = []
    for t in j["learner"]["gradient_booster"]["model"]["trees"]:
        L, R = t["left_children"], t["right_children"]
        cond, hess = t["split_conditions"], t["sum_hessian"]
        mean = [0.0] * len(L)

        def fill(n):                                   # cover-weighted node means (Saabas contributions)
            if L[n] == -1:
                mean[n] = cond[n]
            else:
                fill(L[n]); fill(R[n])
                mean[n] = (mean[L[n]] * hess[L[n]] + mean[R[n]] * hess[R[n]]) / (hess[L[n]] + hess[R[n]])
        fill(0)
        trees.append({"l": L, "r": R, "f": t["split_indices"], "t": cond,
                      "d": t["default_left"], "m": [round(x, 7) for x in mean]})
    return {"type": "xgb", "base_margin": math.log(base / (1 - base)), "trees": trees}


def export_lasso(pipe):
    return {"type": "lasso",
            "median": pipe.named_steps["imp"].statistics_.tolist(),
            "mean": pipe.named_steps["sc"].mean_.tolist(), "scale": pipe.named_steps["sc"].scale_.tolist(),
            "coef": pipe.named_steps["clf"].coef_[0].tolist(),
            "intercept": float(pipe.named_steps["clf"].intercept_[0])}


def data_summary():
    """Focused summaries of the modelling data for the website's Data section.
    Descriptive statistics that shape a modelling choice use the training years only."""
    import pandas as pd
    f = pd.read_pickle(C.PROC_DIR / "features.pkl")
    f = f.assign(year=f.year.clip(lower=C.FIRST_YEAR))          # Dec-2007 events in the 2008 file -> warm-up
    split_of = lambda y: ("warm-up" if y < C.TRAIN_YEARS[0] else "train" if y <= C.TRAIN_YEARS[1]
                          else "val" if y == C.VAL_YEAR else "test" if y <= C.TEST_YEARS[1] else "later")
    per_year = [{"year": int(y), "n": int(n), "split": split_of(y)} for y, n in f.groupby("year").size().items()]

    m = f[~f.retired & (f.year >= C.TRAIN_YEARS[0])].copy()
    m["split"] = m.year.map(split_of)
    a_fav = m.log_rank_diff < 0                                # player A is the better-ranked one
    m["fav_won"] = np.where(a_fav, m.y, 1 - m.y)
    splits = []
    for s in ("train", "val", "test", "later"):
        d = m[m.split == s]
        splits.append({"split": s, "years": f"{d.year.min()}–{d.year.max()}", "n": int(len(d)),
                       "hard": float((d.surface == "Hard").mean()), "clay": float((d.surface == "Clay").mean()),
                       "grass": float((d.surface == "Grass").mean()), "slam": float((d.level == "G").mean()),
                       "fav_win": float(d.fav_won.mean())})
    tr = m[m.split == "train"]
    gap = pd.cut(tr.log_rank_diff.abs(), [-0.01, 0.5, 1.0, 2.0, 99],
                 labels=["similar rank (<1.6x)", "1.6-2.7x", "2.7-7.4x", ">7.4x"])
    fav_by_gap = [{"group": str(g), "n": int(len(d)), "fav_win": float(d.fav_won.mean())}
                  for g, d in tr.groupby(gap, observed=True)]
    # single-feature signal: how often the player who is "ahead" on one feature wins (training years)
    signal = []
    for c in C.FEATURES[:8]:
        x = -tr[c] if c == "log_rank_diff" else tr[c]
        ok = x.notna() & (x != 0)
        won = np.where(x[ok] > 0, tr.y[ok], 1 - tr.y[ok])
        signal.append({"feature": c, "coverage": float(ok.mean()), "missing": float(tr[c].isna().mean()),
                       "ahead_wins": float(won.mean())})
    missing = {"rank": float((m.A_rank.isna() | m.B_rank.isna()).mean()),
               "height": float(m.height_diff.isna().mean()), "age": float(m.age_diff.isna().mean())}
    return {"per_year": per_year, "splits": splits, "fav_by_gap_train": fav_by_gap,
            "feature_signal_train": signal, "missing": missing, "label_balance": float(m.y.mean())}


def main():
    C.SITE_DATA.mkdir(parents=True, exist_ok=True)
    res = json.loads((C.PROC_DIR / "results.json").read_text())
    res["data_summary"] = data_summary()
    res["data"] = json.loads((C.PROC_DIR / "data_info.json").read_text())
    res["slam_eval"] = json.loads((C.PROC_DIR / "slam_eval.json").read_text())
    res["config"] = {"form_window_days": C.FORM_WINDOW, "surface_window_days": C.SURFACE_WINDOW,
                     "shrink_k": C.SHRINK_K, "unranked_rank": C.UNRANKED_RANK, "levels": sorted(C.LEVELS)}
    (C.SITE_DATA / "results.json").write_text(json.dumps(nan_to_none(res), indent=1))
    slams = json.loads((C.PROC_DIR / "slams.json").read_text())
    (C.SITE_DATA / "slams").mkdir(exist_ok=True)
    for old in (C.SITE_DATA / "slams").glob("*.json"):
        old.unlink()
    index = []
    for s in slams["slams"]:                     # one file per Slam, loaded on demand by the page
        (C.SITE_DATA / "slams" / f"{s['key']}.json").write_text(json.dumps(s, separators=(",", ":")))
        index.append({k: s.get(k) for k in ("key", "label", "slam", "year", "surface", "split", "kind", "champion",
                                             "horizon_months")})
    (C.SITE_DATA / "slams.json").write_text(json.dumps({"n_sims": slams["n_sims"], "slams": index}, indent=1))

    snaps = pickle.load(open(C.PROC_DIR / "snapshots.pkl", "rb"))
    cur = snaps["current"]
    wm = res["web_model"]
    model = joblib.load(MODEL_DIR / f"{wm}.joblib")
    cal = joblib.load(MODEL_DIR / "calibrators.joblib")[wm]
    exported = export_xgb(model) if wm == "xgb" else export_lasso(model)
    exported.update({"name": res["names"][wm], "features": C.FEATURES, "platt": {"a": cal.a, "b": cal.b}})
    h2h = {"cutoff": cur["cutoff"], "players": list(cur["players"].values()), "h2h": cur["h2h"],
           "constants": {"shrink_k": C.SHRINK_K, "unranked_rank": C.UNRANKED_RANK}, "model": exported}
    (C.SITE_DATA / "h2h.json").write_text(json.dumps(nan_to_none(h2h), separators=(",", ":")))
    for f in sorted(C.SITE_DATA.glob("**/*.json"))[:6]:
        print(f"{f.name:14s} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
