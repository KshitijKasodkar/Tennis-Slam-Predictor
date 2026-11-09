"""Step 1 - load raw ATP matches and build leakage-safe differential features.

Every feature for a match is computed from a player-state object that has only seen
matches processed *before* it (strict chronological order: tournament date, then round).
The state is updated with the result only after the features have been recorded.

    python -m src.features
"""
import json
import math
import pickle
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------------------- loading
def load_raw() -> pd.DataFrame:
    files = sorted(list(C.RAW_DIR.glob("atp_matches_[0-9][0-9][0-9][0-9].csv")) +
                   list(C.RAW_DIR.glob("[0-9][0-9][0-9][0-9].csv")), key=lambda f: f.stem[-4:])
    files = [f for f in files if int(f.stem[-4:]) >= C.FIRST_YEAR]
    if not files:
        raise SystemExit(f"No YYYY.csv / atp_matches_YYYY.csv files in {C.RAW_DIR}. See README 'Data access'.")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    n0 = len(df)
    df["tourney_level"] = df["tourney_level"].astype(str)
    df = df[df["tourney_level"].isin(C.LEVELS)].copy()
    for c in ("winner_id", "loser_id"):          # Sackmann uses numbers, TML uses ATP codes -> strings
        df[c] = df[c].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["score"] = df["score"].fillna("").astype(str)
    walkover = df["score"].str.contains("W/O|Walkover|walkover", regex=True) | (df["score"].str.strip() == "")
    df["walkover"] = walkover   # kept for draw reconstruction, skipped for modelling
    df["retired"] = df["score"].str.contains("RET|DEF|Def", regex=True)
    df["date"] = pd.to_datetime(df["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    df["year"] = df["date"].dt.year
    df["round_order"] = df["round"].map(C.ROUND_ORDER).fillna(0)
    df["surface"] = df["surface"].fillna("Hard").replace({"Carpet": "Hard"})
    df["slam"] = df["tourney_name"].str.lower().str.strip().map(C.SLAM_NAMES)
    df.loc[df["tourney_level"] != "G", "slam"] = np.nan
    df.loc[df["slam"].notna(), "tourney_name"] = df["slam"]       # "Roland Garros" -> "French Open" etc.
    df = df.sort_values(["date", "tourney_id", "round_order", "match_num"], kind="mergesort")
    df = df.reset_index(drop=True)
    info = {"files": [f.name for f in files], "rows_raw": int(n0), "rows_tour_level": int(len(df)),
            "walkovers_skipped": int(df["walkover"].sum()), "retirements": int(df["retired"].sum()),
            "synthetic": (C.RAW_DIR / "SYNTHETIC_DATA").exists()}
    return df, info


# ----------------------------------------------------------------------------- state
def shrunk(w, n):
    return (w + C.SHRINK_K * 0.5) / (n + C.SHRINK_K)


def num(x, default=np.nan):
    try:
        x = float(x)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default


class PlayerState:
    """Rolling per-player history plus head-to-head counts. Knows only the past."""

    def __init__(self):
        self.hist = defaultdict(deque)        # pid -> deque[(day, won, surface)]
        self.h2h = defaultdict(int)           # (a, b) -> wins of a over b
        self.profile = {}                     # pid -> latest rank/points/age/height/name/day

    def stats(self, pid, day, surface):
        w6 = n6 = ws = ns = 0
        for d, won, s in self.hist[pid]:
            age = day - d
            if age <= C.FORM_WINDOW:
                n6 += 1; w6 += won
            if age <= C.SURFACE_WINDOW and s == surface:
                ns += 1; ws += won
        return {"w6": w6, "n6": n6, "ws": ws, "ns": ns}

    def update(self, w, l, day, surface):
        for pid, won in ((w, 1), (l, 0)):
            h = self.hist[pid]
            h.append((day, won, surface))
            while h and day - h[0][0] > C.SURFACE_WINDOW:
                h.popleft()
        self.h2h[(w, l)] += 1


def make_features(a, b, surface, best_of, h2h_ab, h2h_ba):
    """Single source of truth for the feature vector (mirrored in docs/js/model.js).
    a, b: dicts with rank, points, age, height, w6, n6, ws, ns."""
    ra = a["rank"] if a["rank"] == a["rank"] else C.UNRANKED_RANK
    rb = b["rank"] if b["rank"] == b["rank"] else C.UNRANKED_RANK
    pa = a["points"] if a["points"] == a["points"] else 0.0
    pb = b["points"] if b["points"] == b["points"] else 0.0
    return {
        "log_rank_diff": math.log(ra) - math.log(rb),
        "rank_points_diff": pa - pb,
        "win_pct_6m_diff": shrunk(a["w6"], a["n6"]) - shrunk(b["w6"], b["n6"]),
        "matches_6m_diff": a["n6"] - b["n6"],
        "surface_win_pct_diff": shrunk(a["ws"], a["ns"]) - shrunk(b["ws"], b["ns"]),
        "h2h_win_diff": h2h_ab - h2h_ba,
        "age_diff": a["age"] - b["age"],            # NaN propagates; handled by the model
        "height_diff": a["height"] - b["height"],
        "best_of_5": int(best_of == 5),
        "surface_clay": int(surface == "Clay"),
        "surface_grass": int(surface == "Grass"),
    }


def side(row, p):
    return {"id": str(row[f"{p}_id"]), "name": row[f"{p}_name"],
            "rank": num(row[f"{p}_rank"]), "points": num(row[f"{p}_rank_points"]),
            "age": num(row[f"{p}_age"]), "height": num(row[f"{p}_ht"]),
            "seed": num(row[f"{p}_seed"]), "entry": row.get(f"{p}_entry")}


# ----------------------------------------------------------------------------- main pass
def build(df):
    rng = np.random.default_rng(C.SEED)
    st = PlayerState()
    rows, slam_snap = [], {}
    first_snap_year = C.VAL_YEAR
    day0 = df["date"].min()
    df = df.assign(day=(df["date"] - day0).dt.days)

    for tid, t in df.groupby("tourney_id", sort=False):
        r0 = t.iloc[0]
        day, surface = int(r0["day"]), r0["surface"]
        # Snapshot of every Slam main-draw entrant, frozen BEFORE the first ball is hit.
        if isinstance(r0["slam"], str) and r0["year"] >= first_snap_year:
            r128 = t[t["round"] == "R128"]
            if len(r128) == 64:
                ents = {}
                for _, r in r128.iterrows():
                    for p in ("winner", "loser"):
                        s = side(r, p)
                        s.update(st.stats(s["id"], day, surface))
                        ents[s["id"]] = s
                ids = list(ents)
                h2h = {f"{a}-{b}": st.h2h[(a, b)] for a in ids for b in ids if st.h2h.get((a, b))}
                slam_snap[tid] = {"tourney_id": tid, "slam": r0["slam"], "year": int(r0["year"]),
                                  "date": r0["date"].strftime("%Y-%m-%d"), "surface": surface,
                                  "entrants": ents, "h2h": h2h}

        for _, r in t.iterrows():
            if r["walkover"]:
                continue                                   # no match was played
            w, l = side(r, "winner"), side(r, "loser")
            w.update(st.stats(w["id"], day, surface)); l.update(st.stats(l["id"], day, surface))
            a_is_w = rng.random() < 0.5                       # symmetric framing
            A, B = (w, l) if a_is_w else (l, w)
            f = make_features(A, B, surface, int(r["best_of"]),
                              st.h2h[(A["id"], B["id"])], st.h2h[(B["id"], A["id"])])
            f.update({"date": r["date"], "year": int(r["year"]), "tourney_id": tid,
                      "tourney_name": r["tourney_name"], "level": r["tourney_level"],
                      "surface": surface, "round": r["round"], "retired": bool(r["retired"]),
                      "A_id": A["id"], "B_id": B["id"], "A_name": A["name"], "B_name": B["name"],
                      "A_rank": A["rank"], "B_rank": B["rank"], "y": int(a_is_w)})
            rows.append(f)
            for s in (w, l):
                st.profile[s["id"]] = {**{k: s[k] for k in ("name", "rank", "points", "age", "height")},
                                       "day": day}
            st.update(w["id"], l["id"], day, surface)

    feats = pd.DataFrame(rows)
    # Snapshot of active players at the data cut-off, for the head-to-head simulator.
    ref = int(df["day"].max()) + 1
    cutoff = (day0 + pd.Timedelta(days=ref)).strftime("%Y-%m-%d")
    active = [(pid, p) for pid, p in st.profile.items()
              if ref - p["day"] <= 365 and p["rank"] == p["rank"]]
    active.sort(key=lambda x: x[1]["rank"])
    players = {}
    for pid, p in active[:150]:
        s = {"id": pid, "name": p["name"], "rank": p["rank"], "points": p["points"],
             "last_played": (day0 + pd.Timedelta(days=p["day"])).strftime("%Y-%m-%d"),
             "weeks_inactive": round((ref - p["day"]) / 7, 1),
             "age": p["age"] + (ref - p["day"]) / 365.25 if p["age"] == p["age"] else np.nan,
             "height": p["height"]}
        for surf in ("Hard", "Clay", "Grass"):
            x = st.stats(pid, ref, surf)
            s["w6"], s["n6"] = x["w6"], x["n6"]
            s[f"ws_{surf}"], s[f"ns_{surf}"] = x["ws"], x["ns"]
        players[pid] = s
    ids = list(players)
    h2h = {f"{a}-{b}": st.h2h[(a, b)] for a in ids for b in ids if st.h2h.get((a, b))}
    current = {"cutoff": cutoff, "players": players, "h2h": h2h}
    return feats, slam_snap, current


def main():
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    df, info = load_raw()
    print(f"loaded {info['rows_tour_level']} tour-level matches from {len(info['files'])} files")
    raw_slams = df[df["slam"].notna()][["tourney_id", "slam", "year", "round", "winner_id",
                                        "loser_id", "winner_name", "loser_name", "score",
                                        "walkover"]]
    feats, slam_snap, current = build(df)
    feats.to_pickle(C.PROC_DIR / "features.pkl")
    raw_slams.to_pickle(C.PROC_DIR / "slam_matches.pkl")
    with open(C.PROC_DIR / "snapshots.pkl", "wb") as fh:
        pickle.dump({"slams": slam_snap, "current": current}, fh)
    info.update({"rows_modelled": int(len(feats)), "years": [int(feats.year.min()), int(feats.year.max())],
                 "slam_snapshots": len(slam_snap), "cutoff": current["cutoff"]})
    (C.PROC_DIR / "data_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
