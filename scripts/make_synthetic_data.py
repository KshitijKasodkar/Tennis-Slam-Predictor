"""Generate a STAND-IN dataset with the same columns as Jeff Sackmann's atp_matches_YYYY.csv.

Only for testing the pipeline end-to-end when the real files are not available.
Results produced from this data are meaningless and the website shows a banner when it is used.

    python scripts/make_synthetic_data.py --out data/raw
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date",
        "match_num", "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand",
        "winner_ht", "winner_ioc", "winner_age", "loser_id", "loser_seed", "loser_entry",
        "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age", "score", "best_of",
        "round", "minutes", "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"]

SLAMS = [("Australian Open", "Hard", 2), ("Roland Garros", "Clay", 21),
         ("Wimbledon", "Grass", 26), ("US Open", "Hard", 35)]


def main(out: Path, start=2008, end=2024, n_players=420, seed=7):
    rng = np.random.default_rng(seed)
    pid = np.arange(100000, 100000 + n_players)
    names = [f"Player {chr(65 + i % 26)}{i:03d}" for i in range(n_players)]
    base = rng.normal(0, 1, n_players)
    surf_off = {s: rng.normal(0, 0.35, n_players) for s in ["Hard", "Clay", "Grass"]}
    birth = rng.uniform(start - 36, end - 17, n_players)   # birth year
    height = rng.normal(188, 7, n_players).round()
    height[rng.random(n_players) < 0.15] = np.nan            # realistic missingness
    out.mkdir(parents=True, exist_ok=True)
    h2h_noise = rng.normal(0, 0.15, (n_players, n_players))

    def skill(i, year, surface):
        age = year - birth[i]
        arc = -0.012 * (age - 27) ** 2                         # career curve peaking ~27
        active = 1.0 if 17 <= age <= 37 else -5.0
        return base[i] + arc + surf_off[surface][i] + active

    for year in range(start, end + 1):
        rows = []
        cal = [(w, n, s, "G", 128, 5) for n, s, w in SLAMS]
        for k in range(9):
            cal.append((5 + k * 5, f"Masters {k + 1}", ["Hard", "Clay", "Hard"][k % 3], "M", 64, 3))
        for k in range(38):
            surf = "Clay" if 14 <= k <= 22 else ("Grass" if 23 <= k <= 26 else "Hard")
            cal.append((1 + k * 51 // 38, f"Open {k + 1}", surf, "A", 32, 3))
        cal.sort()
        for week, tname, surf, lvl, draw, bo in cal:
            date = pd.Timestamp(year, 1, 1) + pd.Timedelta(weeks=week)
            yr = year + week / 52
            sk = np.array([skill(i, yr, "Hard") for i in range(n_players)])
            noisy = sk + rng.normal(0, 0.3, n_players)
            order = np.argsort(-noisy)
            rank = np.empty(n_players, int)
            rank[order] = np.arange(1, n_players + 1)
            pts = (12000 / rank ** 0.85).round()
            pool = order[: int(draw * 1.6)]
            entrants = rng.choice(pool, draw, replace=False)
            entrants = entrants[np.argsort(rank[entrants])]
            seeds = entrants[: draw // 4]
            rest = rng.permutation(entrants[draw // 4:])
            slots = np.empty(draw, int)                     # simple seeded draw
            seed_pos = np.linspace(0, draw - 1, draw // 4).round().astype(int)
            slots[:] = -1
            slots[seed_pos] = seeds
            slots[slots == -1] = rest
            tid = f"{year}-{tname.replace(' ', '')[:6]}{week}"
            rounds = ["R128", "R64", "R32", "R16", "QF", "SF", "F"][-int(np.log2(draw)):]
            cur = list(slots)
            mnum = 1
            for rnd in rounds:
                nxt = []
                for a, b in zip(cur[0::2], cur[1::2]):
                    d = skill(a, yr, surf) - skill(b, yr, surf) + h2h_noise[a, b]
                    pset = 1 / (1 + np.exp(-0.9 * d))
                    need = bo // 2 + 1
                    sa = sb = 0
                    while sa < need and sb < need:
                        if rng.random() < pset: sa += 1
                        else: sb += 1
                    w, l = (a, b) if sa > sb else (b, a)
                    nxt.append(w)
                    ret = rng.random() < 0.02
                    score = " ".join(["6-4"] * max(sa, sb)) + (" RET" if ret else "")
                    if rng.random() < 0.004: score = "W/O"
                    sd = {s: n + 1 for n, s in enumerate(seeds)}
                    rows.append([tid, tname, surf, draw, lvl, int(date.strftime("%Y%m%d")), mnum,
                                 pid[w], sd.get(w), None, names[w], "R", height[w], "XXX",
                                 round(yr - birth[w], 1), pid[l], sd.get(l), None, names[l], "R",
                                 height[l], "XXX", round(yr - birth[l], 1), score, bo, rnd,
                                 None, rank[w], pts[w], rank[l], pts[l]])
                    mnum += 1
                cur = nxt
        df = pd.DataFrame(rows, columns=COLS).sample(frac=1, random_state=year)  # real files are not in bracket order
        df.to_csv(out / f"atp_matches_{year}.csv", index=False)
    (out / "SYNTHETIC_DATA").write_text("These CSVs are synthetic stand-ins, not real ATP data.\n")
    print(f"wrote synthetic {start}-{end} to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    a = ap.parse_args()
    main(a.out)
