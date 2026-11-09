"""Check that docs/js/model.js reproduces the Python model to ~1e-6 on random player pairs.

    python scripts/verify_js_model.py          (needs Node.js)
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C                      # noqa: E402
from src.features import make_features           # noqa: E402

D = json.loads((C.SITE_DATA / "h2h.json").read_text())
wm = json.loads((C.SITE_DATA / "results.json").read_text())["web_model"]
model = joblib.load(C.PROC_DIR / "models" / f"{wm}.joblib")
cal = joblib.load(C.PROC_DIR / "models" / "calibrators.joblib")[wm]
rng = np.random.default_rng(0)
P = D["players"]
cases = []
for _ in range(300):
    i, j = rng.choice(len(P), 2, replace=False)
    cases.append((int(i), int(j), ["Hard", "Clay", "Grass"][rng.integers(3)], int(rng.choice([3, 5]))))


def py_one(a, b, surf, bo):
    na = lambda p: {**{k: (np.nan if p[k] is None else p[k]) for k in ("rank", "points", "age", "height")},
                    "w6": p["w6"], "n6": p["n6"], "ws": p[f"ws_{surf}"], "ns": p[f"ns_{surf}"]}
    f = make_features(na(a), na(b), surf, bo, D["h2h"].get(f"{a['id']}-{b['id']}", 0),
                      D["h2h"].get(f"{b['id']}-{a['id']}", 0))
    return cal(model.predict_proba(pd.DataFrame([f])[C.FEATURES].astype(float))[:, 1])[0]


py = [(py_one(P[i], P[j], s, bo) + 1 - py_one(P[j], P[i], s, bo)) / 2 for i, j, s, bo in cases]
js_src = f"""
const fs=require('fs'); const M=require('{(ROOT / 'docs/js/model.js').as_posix()}');
const D=JSON.parse(fs.readFileSync('{(C.SITE_DATA / 'h2h.json').as_posix()}'));
const cases={json.dumps(cases)};
console.log(JSON.stringify(cases.map(([i,j,s,bo])=>M.predict(D,D.players[i],D.players[j],s,bo).p)));
"""
js = json.loads(subprocess.run(["node", "-e", js_src], capture_output=True, text=True, check=True).stdout)
err = np.abs(np.array(py) - np.array(js))
print(f"model={wm}  pairs={len(cases)}  max |python - js| = {err.max():.2e}")
sys.exit(0 if err.max() < 1e-4 else 1)
