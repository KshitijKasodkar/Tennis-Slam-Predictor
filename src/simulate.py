"""Step 3 - Monte Carlo Grand Slam simulation.

For every Slam with a stored pre-tournament snapshot, the real 128-player draw is rebuilt from
the results tree (who-could-meet-whom is exact), a calibrated, symmetric win-probability matrix
is computed for every pair of entrants with features frozen at the tournament start, and the
bracket is played N_SIMS times. The same is done with the rank-only baseline for comparison.
A projection for the next Slam after the data cut-off uses random seeded draws.

    python -m src.simulate
"""
import json
import math
import pickle

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import make_features

N_SIMS = 10000
ROUNDS = C.SLAM_ROUNDS                    # R128 ... F
STAGES = ["R64", "R32", "R16", "QF", "SF", "F", "W"]   # stage k = won k matches
SLAM_SURFACE = {"Australian Open": "Hard", "French Open": "Clay", "Wimbledon": "Grass", "US Open": "Hard"}
SLAM_MONTH = [("Australian Open", 1), ("French Open", 5), ("Wimbledon", 7), ("US Open", 8)]
MODEL_DIR = C.PROC_DIR / "models"


def clean(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


# ----------------------------------------------------------------------------- probabilities
def prob_matrix(model, cal, ents, surface, h2h, cols):
    n = len(ents)
    rows, idx = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                a, b = ents[i], ents[j]
                rows.append(make_features(a, b, surface, 5, h2h.get(f"{a['id']}-{b['id']}", 0),
                                          h2h.get(f"{b['id']}-{a['id']}", 0)))
                idx.append((i, j))
    X = pd.DataFrame(rows)[cols].astype(float)
    p = cal(model.predict_proba(X)[:, 1])
    P = np.full((n, n), 0.5)
    ii, jj = np.array(idx).T
    P[ii, jj] = p
    return (P + 1 - P.T) / 2                      # enforce P[i,j] + P[j,i] = 1


# ----------------------------------------------------------------------------- draw
def rebuild_draw(m):
    """Return the 128 player ids in bracket order from a Slam's match list (incl. walkovers)."""
    won = {(r.round, r.winner_id): r for r in m.itertuples()}
    final = m[m["round"] == "F"].iloc[0]

    def slots(match, depth):
        if depth == 0:                            # R128 match: its two players
            return [match.winner_id, match.loser_id]
        prev = ROUNDS[depth - 1]
        halves = [slots(won[(prev, pid)], depth - 1) for pid in (match.winner_id, match.loser_id)]
        return halves[0] + halves[1]

    order = slots(final, len(ROUNDS) - 1)
    r128 = {frozenset((r.winner_id, r.loser_id)) for r in m[m["round"] == "R128"].itertuples()}
    assert len(order) == 128 and all(frozenset(order[k:k + 2]) in r128 for k in range(0, 128, 2))
    return order


def actual_stage(m):
    """Number of matches each entrant won (0 = lost in R128 ... 7 = champion)."""
    wins = m.groupby("winner_id").size().to_dict()
    ids = set(m.winner_id) | set(m.loser_id)
    return {pid: int(wins.get(pid, 0)) for pid in ids}


SEED_SLOTS = [[0], [127], [63, 64], [31, 32, 95, 96], [15, 16, 47, 48, 79, 80, 111, 112],
              [7, 8, 23, 24, 39, 40, 55, 56, 71, 72, 87, 88, 103, 104, 119, 120]]
SEED_GROUPS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]


def random_seeded_draws(n_sims, rng):
    """Each row: a draw with players 0..31 seeded on Grand-Slam seed lines, 32..127 random."""
    draws = np.empty((n_sims, 128), dtype=int)
    seeded = set(s for g in SEED_SLOTS for s in g)
    free = np.array([s for s in range(128) if s not in seeded])
    for k in range(n_sims):
        d = np.empty(128, dtype=int)
        for (lo, hi), pos in zip(SEED_GROUPS, SEED_SLOTS):
            d[rng.permutation(pos)] = np.arange(lo, hi)
        d[free] = rng.permutation(np.arange(32, 128))
        draws[k] = d
    return draws


# ----------------------------------------------------------------------------- simulation
def simulate(P, draws, rng):
    """draws: (S, 128) entrant indices in bracket order. Returns stage counts and opponent counts."""
    S, n = draws.shape[0], P.shape[0]
    reach = np.zeros((n, 8))
    reach[:, 0] = S
    opp = np.zeros((n, 7, n))
    slots = draws
    for r in range(7):
        a, b = slots[:, 0::2], slots[:, 1::2]
        np.add.at(opp, (a.ravel(), r, b.ravel()), 1)
        np.add.at(opp, (b.ravel(), r, a.ravel()), 1)
        slots = np.where(rng.random(a.shape) < P[a, b], a, b)
        reach[:, r + 1] = np.bincount(slots.ravel(), minlength=n)
    return reach / S, opp / S


def summarise(ents, P, reach, opp, actual=None, top_opp=3):
    players = []
    for i, e in enumerate(ents):
        route = []
        for r in range(7):
            o = np.argsort(-opp[i, r])[:top_opp]
            pr = reach[i, r]
            route.append({"round": ROUNDS[r], "p_reach": round(float(pr), 4),
                          "p_win_round": round(float(reach[i, r + 1] / pr), 4) if pr > 0 else 0,
                          "opponents": [{"name": ents[j]["name"], "p_face": round(float(opp[i, r, j] / pr), 3) if pr > 0 else 0,
                                         "p_beat": round(float(P[i, j]), 3)} for j in o if opp[i, r, j] > 0]})
        players.append({"id": str(e["id"]), "name": e["name"], "rank": clean(e["rank"]),
                        "seed": clean(e.get("seed")), "last_played": e.get("last_played"),
                        "weeks_inactive": e.get("weeks_inactive"),
                        "p": {s: round(float(reach[i, k + 1]), 4) for k, s in enumerate(STAGES)},
                        "route": route,
                        "actual": None if actual is None else actual.get(str(e["id"]), 0)})
    players.sort(key=lambda d: -d["p"]["W"])
    return players


def bracket_scores(reach, actual_wins):
    """Proper scores of the whole bracket: mean log-likelihood of each entrant's exact stage,
    and Brier score of 'reached QF' / 'won title'."""
    exact = np.clip(reach[:, :7] - reach[:, 1:8], 0, 1)
    exact = np.column_stack([exact, reach[:, 7]])
    ll = np.mean(np.log(np.clip(exact[np.arange(len(actual_wins)), actual_wins], 1e-6, 1)))
    qf = np.mean((reach[:, 4] - (actual_wins >= 4)) ** 2)
    w = np.mean((reach[:, 7] - (actual_wins == 7)) ** 2)
    return {"stage_loglik": float(ll), "brier_qf": float(qf), "brier_title": float(w)}


def main():
    res = json.loads((C.PROC_DIR / "results.json").read_text())
    sel = res["selected"]
    cal = joblib.load(MODEL_DIR / "calibrators.joblib")
    model, base = joblib.load(MODEL_DIR / f"{sel}.joblib"), joblib.load(MODEL_DIR / "rank_only.joblib")
    snaps = pickle.load(open(C.PROC_DIR / "snapshots.pkl", "rb"))
    sm = pd.read_pickle(C.PROC_DIR / "slam_matches.pkl")
    rng = np.random.default_rng(C.SEED)
    out, evals = [], []

    for tid, s in sorted(snaps["slams"].items(), key=lambda kv: kv[1]["date"]):
        m = sm[sm.tourney_id == tid]
        try:
            order = rebuild_draw(m)
        except (KeyError, IndexError, AssertionError):
            print("skip (draw not reconstructable):", tid); continue
        ents = [s["entrants"][pid] for pid in order]
        P = prob_matrix(model, cal[sel], ents, s["surface"], s["h2h"], C.FEATURES)
        Pb = prob_matrix(base, cal["rank_only"], ents, s["surface"], s["h2h"], ["log_rank_diff"])
        draws = np.tile(np.arange(128), (N_SIMS, 1))
        reach, opp = simulate(P, draws, rng)
        reach_b, _ = simulate(Pb, draws, rng)
        act = actual_stage(m)
        actual_wins = np.array([act[e["id"]] for e in ents])
        champ = int(np.argmax(actual_wins == 7))
        split = "val" if s["year"] == C.VAL_YEAR else ("test" if s["year"] <= C.TEST_YEARS[1] else "post-test")
        players = summarise(ents, P, reach, opp, act)
        title_order = np.argsort(-reach[:, 7])
        ev = {"key": tid, "label": f"{s['slam']} {s['year']}", "split": split,
              "champion": ents[champ]["name"], "champion_rank": clean(ents[champ]["rank"]),
              "p_champion_model": float(reach[champ, 7]), "p_champion_base": float(reach_b[champ, 7]),
              "champion_pred_position": int(np.where(title_order == champ)[0][0]) + 1,
              "model": bracket_scores(reach, actual_wins), "base": bracket_scores(reach_b, actual_wins)}
        evals.append(ev)
        base_top = sorted([{"name": e["name"], "p": round(float(reach_b[i, 7]), 4)} for i, e in enumerate(ents)],
                          key=lambda d: -d["p"])[:16]
        out.append({"key": tid, "label": ev["label"], "slam": s["slam"], "year": s["year"],
                    "date": s["date"], "surface": s["surface"], "split": split, "kind": "real draw",
                    "champion": ev["champion"], "players": players, "baseline_top": base_top,
                    "draw": [e["name"] for e in ents]})
        print(f"{ev['label']:24s} champ={ev['champion']:<18s} p_model={ev['p_champion_model']:.3f} "
              f"p_base={ev['p_champion_base']:.3f} pos={ev['champion_pred_position']}")

    # --- projections for the next four Slams after the cut-off (hypothetical seeded draws)
    cur = snaps["current"]
    cutoff = pd.Timestamp(cur["cutoff"])
    upcoming = [(cutoff.year, n) for n, mth in SLAM_MONTH if mth > cutoff.month] + \
               [(cutoff.year + 1, n) for n, _ in SLAM_MONTH]
    pl = sorted(cur["players"].values(), key=lambda p: p["rank"])[:128]
    for year, slam in upcoming[:4] if len(pl) == 128 else []:
        surf = SLAM_SURFACE[slam]
        ents = [{**p, "ws": p[f"ws_{surf}"], "ns": p[f"ns_{surf}"], "seed": i + 1 if i < 32 else None}
                for i, p in enumerate(pl)]
        P = prob_matrix(model, cal[sel], ents, surf, cur["h2h"], C.FEATURES)
        reach, opp = simulate(P, random_seeded_draws(N_SIMS, rng), rng)
        start = pd.Timestamp(year, dict(SLAM_MONTH)[slam], 15)
        horizon = int(round((start - cutoff).days / 30.4))
        out.append({"key": f"projection-{year}-{slam.replace(' ', '')}", "horizon_months": horizon, "label": f"{slam} {year} (projection)",
                    "slam": slam, "year": year, "date": None, "surface": surf, "split": "projection",
                    "kind": "random seeded draws", "champion": None, "players": summarise(ents, P, reach, opp),
                    "baseline_top": [], "draw": None})
        print("projection:", out[-1]["label"], [(p["name"], p["p"]["W"]) for p in out[-1]["players"][:4]])

    test_ev = [e for e in evals if e["split"] != "val"]
    agg = {k: {m: float(np.mean([e[k][m] for e in test_ev])) for m in ("stage_loglik", "brier_qf", "brier_title")}
           for k in ("model", "base")} if test_ev else {}
    if test_ev:
        agg["mean_p_champion_model"] = float(np.mean([e["p_champion_model"] for e in test_ev]))
        agg["mean_p_champion_base"] = float(np.mean([e["p_champion_base"] for e in test_ev]))
        agg["n_slams"] = len(test_ev)
    (C.PROC_DIR / "slams.json").write_text(json.dumps({"n_sims": N_SIMS, "slams": out}, separators=(",", ":")))
    (C.PROC_DIR / "slam_eval.json").write_text(json.dumps({"per_slam": evals, "aggregate": agg}, indent=2))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
