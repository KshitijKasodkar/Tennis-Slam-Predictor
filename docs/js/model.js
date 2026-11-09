// In-browser copy of the trained, calibrated match model.
// Mirrors src/features.py::make_features exactly; verified against Python by scripts/verify_js_model.py.
(function (root) {
  "use strict";

  function shrunk(w, n, k) { return (w + k * 0.5) / (n + k); }
  function nz(x) { return x === null || x === undefined ? NaN : x; }

  // a, b: player snapshot objects; surface: "Hard" | "Clay" | "Grass"
  function features(a, b, surface, bestOf, h2hAB, h2hBA, K) {
    const ra = Number.isNaN(nz(a.rank)) ? K.unranked_rank : a.rank;
    const rb = Number.isNaN(nz(b.rank)) ? K.unranked_rank : b.rank;
    const pa = Number.isNaN(nz(a.points)) ? 0 : a.points;
    const pb = Number.isNaN(nz(b.points)) ? 0 : b.points;
    const ws = (p) => nz(p["ws_" + surface] !== undefined ? p["ws_" + surface] : p.ws);
    const ns = (p) => nz(p["ns_" + surface] !== undefined ? p["ns_" + surface] : p.ns);
    return {
      log_rank_diff: Math.log(ra) - Math.log(rb),
      rank_points_diff: pa - pb,
      win_pct_6m_diff: shrunk(a.w6, a.n6, K.shrink_k) - shrunk(b.w6, b.n6, K.shrink_k),
      matches_6m_diff: a.n6 - b.n6,
      surface_win_pct_diff: shrunk(ws(a), ns(a), K.shrink_k) - shrunk(ws(b), ns(b), K.shrink_k),
      h2h_win_diff: h2hAB - h2hBA,
      age_diff: nz(a.age) - nz(b.age),
      height_diff: nz(a.height) - nz(b.height),
      best_of_5: bestOf === 5 ? 1 : 0,
      surface_clay: surface === "Clay" ? 1 : 0,
      surface_grass: surface === "Grass" ? 1 : 0,
    };
  }

  // Returns {margin, contrib[], bias} in log-odds units.
  function scoreXGB(M, x) {
    const contrib = new Array(x.length).fill(0);
    let margin = M.base_margin, bias = M.base_margin;
    for (const t of M.trees) {
      let n = 0;
      bias += t.m[0];
      while (t.l[n] !== -1) {
        const v = x[t.f[n]];
        const next = Number.isNaN(v) ? (t.d[n] ? t.l[n] : t.r[n]) : (Math.fround(v) < Math.fround(t.t[n]) ? t.l[n] : t.r[n]);
        contrib[t.f[n]] += t.m[next] - t.m[n];
        n = next;
      }
      margin += t.t[n];                       // leaf value is stored in split_conditions
    }
    return { margin, contrib, bias };
  }

  function scoreLasso(M, x) {
    let margin = M.intercept;
    const contrib = x.map((v, i) => {
      const z = ((Number.isNaN(v) ? M.median[i] : v) - M.mean[i]) / M.scale[i];
      const c = M.coef[i] * z;
      margin += c;
      return c;
    });
    return { margin, contrib, bias: M.intercept };
  }

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const logit = (p) => { p = Math.min(Math.max(p, 1e-6), 1 - 1e-6); return Math.log(p / (1 - p)); };

  // One direction: calibrated P(A beats B)
  function oneWay(D, a, b, surface, bestOf) {
    const M = D.model, K = D.constants;
    const f = features(a, b, surface, bestOf, D.h2h[a.id + "-" + b.id] || 0, D.h2h[b.id + "-" + a.id] || 0, K);
    const x = M.features.map((k) => f[k]);
    const s = M.type === "xgb" ? scoreXGB(M, x) : scoreLasso(M, x);
    const raw = sigmoid(s.margin);
    const cal = sigmoid(M.platt.a * logit(raw) + M.platt.b);
    return { f, x, raw, cal, contrib: s.contrib, bias: s.bias };
  }

  // Symmetric prediction exactly as in src/simulate.py: average P(A>B) and 1 - P(B>A)
  function predict(D, a, b, surface, bestOf) {
    const ab = oneWay(D, a, b, surface, bestOf);
    const ba = oneWay(D, b, a, surface, bestOf);
    const contrib = ab.contrib.map((c, i) => (c - ba.contrib[i]) / 2);
    return { p: (ab.cal + 1 - ba.cal) / 2, features: ab.f, contrib, names: D.model.features };
  }

  const api = { features, predict, oneWay };
  if (typeof module !== "undefined") module.exports = api; else root.TennisModel = api;
})(this);
