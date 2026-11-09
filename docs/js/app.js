// Slam Predictor front end. Static: reads docs/data/*.json, no server, no libraries.
(async function () {
  "use strict";
  // ------------------------------------------------------------------ helpers
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => [...document.querySelectorAll(s)];
  const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
  const pct = (x, d = 1) => (x === null || x === undefined ? "—" : (100 * x).toFixed(d) + "%");
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const NS = "http://www.w3.org/2000/svg";
  const tip = $("#tip");
  const showTip = (e, h) => { tip.innerHTML = h; tip.style.opacity = 1; tip.style.left = Math.min(e.clientX + 14, innerWidth - 290) + "px"; tip.style.top = e.clientY + 14 + "px"; };
  const hideTip = () => (tip.style.opacity = 0);
  const hover = (node, html) => { node.addEventListener("mousemove", (e) => showTip(e, typeof html === "function" ? html() : html)); node.addEventListener("mouseleave", hideTip); };
  function el(tag, attrs = {}, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }
  function table(id, head, rows, hl = () => false) {
    document.getElementById(id).innerHTML = "<tr>" + head.map((h) => `<th class="${h.n ? "n" : ""}">${h.t}</th>`).join("") + "</tr>" +
      rows.map((r, i) => `<tr class="${hl(r, i) ? "hl" : ""}">` + r.map((c, j) => `<td class="${head[j].n ? "n" : ""}">${c}</td>`).join("") + "</tr>").join("");
  }
  // Horizontal bars sized to the container (1 SVG unit = 1 CSS px, so text stays 12px).
  function hbar(box, items, { min = 0, max, fmtV = (v) => fmt(v), labelW = 180, barH = 20 } = {}) {
    const W = Math.max(300, box.clientWidth), gap = 8, rowH = barH + gap, H = items.length * rowH + 22;
    labelW = Math.min(labelW, W * 0.42);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
    const x0 = labelW, x1 = W - 52;
    max = max ?? (Math.max(...items.map((d) => Math.max(d.value, d.value2 ?? 0))) * 1.05 || 1);
    const sx = (v) => x0 + ((v - min) / (max - min)) * (x1 - x0);
    const g = el("g", { class: "grid" }, svg), nt = W < 520 ? 3 : 4;
    for (let k = 0; k <= nt; k++) {
      const v = min + ((max - min) * k) / nt;
      el("line", { x1: sx(v), x2: sx(v), y1: 0, y2: H - 18 }, g);
      el("text", { x: sx(v), y: H - 3, "text-anchor": "middle" }, svg).textContent = fmtV(v);
    }
    items.forEach((d, i) => {
      const y = i * rowH, h2 = d.value2 !== undefined ? Math.round(barH * 0.3) : 0;
      el("text", { x: x0 - 8, y: y + barH / 2 + 4, "text-anchor": "end" }, svg).textContent = d.label;
      el("rect", { x: x0, y, width: Math.max(1, sx(d.value) - x0), height: barH - (h2 ? h2 + 2 : 0), rx: 4, fill: d.color || "var(--a)" }, svg);
      if (h2) el("rect", { x: x0, y: y + barH - h2, width: Math.max(1, sx(d.value2) - x0), height: h2, rx: 2, fill: "var(--base)" }, svg);
      el("text", { x: sx(Math.max(d.value, d.value2 ?? -1)) + 6, y: y + barH / 2 + 4, style: "fill:var(--ink);font-weight:600" }, svg).textContent = fmtV(d.value);
      hover(el("rect", { x: 0, y: y - gap / 2, width: W, height: rowH, fill: "transparent" }, svg), d.tip || `${esc(d.label)}: ${fmtV(d.value)}`);
    });
    box.replaceChildren(svg);
  }

  // ------------------------------------------------------------------ data
  const get = (f) => fetch(`data/${f}`).then((r) => r.json());
  const [R, S, H] = await Promise.all([get("results.json"), get("slams.json"), get("h2h.json")]);
  const cache = {};
  const loadSlam = (key) => (cache[key] ??= get(`slams/${key}.json`));
  if (R.data.synthetic) $("#synthetic-banner").hidden = false;
  const N = R.names, sel = R.selected, tc = R.test_cal, agg = R.slam_eval.aggregate;
  const past = R.slam_eval.per_slam;
  const SLAM = {
    "Australian Open": { c: "var(--ao)", short: "AO", when: "January", surf: "Hard" },
    "French Open": { c: "var(--fo)", short: "FO", when: "late May", surf: "Clay" },
    "Wimbledon": { c: "var(--wim)", short: "WIM", when: "late June", surf: "Grass" },
    "US Open": { c: "var(--uso)", short: "USO", when: "late August", surf: "Hard" },
  };
  const ROUND_OUT = ["lost in round 1", "lost in round 2", "lost in round 3", "lost in round 4", "lost in the quarter-final", "lost in the semi-final", "lost the final", "won the title"];
  const STALE = 6;   // weeks without a match after which rank/form are flagged as out of date
  const since = (p) => new Date(p.last_played).toLocaleDateString("en-GB", { month: "short", year: "numeric" });
  const stale = (p) => (p.weeks_inactive ?? 0) > STALE;
  const warn = (p) => (stale(p) ? `<span class="warn" title="No matches in the data since ${since(p)}; ranking and form are as of then.">⚠ no matches since ${since(p)}</span>` : "");
  const conf = (h) => (h <= 3 ? ["hi", "Most reliable"] : h <= 6 ? ["md", "Medium confidence"] : ["lo", "Early outlook"]);

  const bind = {
    cutoff: new Date(R.data.cutoff).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" }),
    nSims: S.n_sims.toLocaleString(), years: `${R.data.years[0] + 1}–${R.data.years[1]}`,
    rowsTour: R.data.rows_tour_level.toLocaleString(), wo: R.data.walkovers_skipped.toLocaleString(), ret: R.data.retirements.toLocaleString(),
    nSnap: R.data.slam_snapshots, unranked: R.config.unranked_rank, k: R.config.shrink_k, k2: R.config.shrink_k / 2, nSlams: agg.n_slams,
    trainY: `${R.split.train.join("–")} (${R.split.n_train.toLocaleString()})`, valY: `${R.split.val} (${R.split.n_val.toLocaleString()})`,
    testY: `${R.split.test.join("–")} (${R.split.n_test.toLocaleString()})`,
  };
  $$("[data-k]").forEach((n) => { if (bind[n.dataset.k] !== undefined) n.textContent = bind[n.dataset.k]; });

  // charts render lazily, only when their container is visible (hidden tabs have zero width)
  const charts = {};
  function renderCharts(force = false) {
    $$("[data-chart]").forEach((c) => {
      if (!c.offsetParent || !c.clientWidth || !charts[c.dataset.chart]) return;
      if (!force && c._w === c.clientWidth && c._v === charts[c.dataset.chart].v) return;
      c._w = c.clientWidth; c._v = charts[c.dataset.chart].v;
      charts[c.dataset.chart].fn(c);
    });
  }
  const setChart = (name, fn) => { charts[name] = { fn, v: (charts[name]?.v || 0) + 1 }; renderCharts(); };
  addEventListener("resize", () => renderCharts());

  // ------------------------------------------------------------------ router
  function route() {
    let h = location.hash.slice(1) || "forecast";
    let section = null;
    if (h.startsWith("sec-")) { section = h; h = "method"; }
    if (!["forecast", "h2h", "history", "method"].includes(h)) h = "forecast";
    $$(".view").forEach((v) => (v.hidden = v.id !== "view-" + h));
    $$("#tabs a").forEach((a) => a.classList.toggle("on", a.dataset.view === h));
    if (section) { const d = document.getElementById(section); if (d) { d.open = true; d.scrollIntoView({ behavior: "smooth" }); } }
    else scrollTo(0, 0);
    renderCharts();
  }
  addEventListener("hashchange", route);

  // ================================================================== 1. FORECAST
  const projIdx = S.slams.filter((s) => s.split === "projection");
  const proj = await Promise.all(projIdx.map((s) => loadSlam(s.key)));
  const fc = { i: 0, player: 0, n: 12 };

  $("#slam-cards").innerHTML = proj.map((s, i) => {
    const m = SLAM[s.slam] || {}, [cc, ct] = conf(s.horizon_months ?? 99), f = s.players[0];
    return `<button class="slam-card" data-i="${i}" style="--c:${m.c}">
      <span class="conf ${cc}">${ct}</span>
      <div class="nm">${esc(s.slam)}</div><div class="dt">${m.when} ${s.year} · ${s.surface}</div>
      <div class="fav">Favourite<b>${esc(f.name)}</b>${pct(f.p.W, 0)} to win</div></button>`;
  }).join("");
  $$(".slam-card").forEach((b) => b.addEventListener("click", () => { fc.i = +b.dataset.i; fc.player = 0; fc.n = 12; renderForecast(); }));
  $("#fc-more").addEventListener("click", () => { fc.n = fc.n === 12 ? 32 : 12; renderForecast(); });

  function routeHTML(p) {
    return p.route.map((r) => `<div class="rd">${r.round}</div><div class="row">
        <div class="bar2"><div><span style="width:${100 * r.p_reach}%"></span></div>${pct(r.p_reach, 0)}</div>
        <div class="opp">${r.opponents.slice(0, 3).map((o) => `<b>${esc(o.name)}</b> ${pct(o.p_face, 0)} · wins ${pct(o.p_beat, 0)}`).join(" &nbsp;·&nbsp; ")}</div></div>`).join("") +
      `<div class="rd">🏆</div><div class="row"><div class="bar2"><div><span style="width:${100 * p.p.W}%;background:var(--ball)"></span></div><b>${pct(p.p.W)}</b></div></div>`;
  }
  function renderForecast() {
    const s = proj[fc.i], [cc, ct] = conf(s.horizon_months ?? 99);
    $$(".slam-card").forEach((b) => b.classList.toggle("on", +b.dataset.i === fc.i));
    $("#fc-title").textContent = `${s.slam} ${s.year} — title chances`;
    $("#fc-conf").className = "conf " + cc; $("#fc-conf").textContent = ct;
    const h = s.horizon_months;
    $("#fc-note").textContent = (h <= 3
      ? `Starts in about ${h} months — closest to our data, so this is the forecast to trust most.`
      : `Starts in about ${h} months. Form, injuries and rankings will change before then, so treat this as an early outlook.`) +
      ` Draw not yet known: averaged over ${S.n_sims.toLocaleString()} random seeded draws.` +
      (s.players.slice(0, 5).some(stale) ? ` ⚠ ${s.players.slice(0, 5).filter(stale).map((p) => p.name).join(", ")} has not played since ${since(s.players.slice(0, 5).find(stale))}: his ranking is out of date and an injury or absence would make this forecast too high. The model only partly accounts for it (low 6-month match count).` : "");
    const top = s.players.slice(0, fc.n), max = top[0].p.W;
    $("#fc-list").innerHTML = top.map((p, i) => `<div class="cont ${i === fc.player ? "on" : ""}" data-i="${i}">
        <div class="rk">${i + 1}</div>
        <div><div class="nm"><span>${esc(p.name)} ${warn(p)}</span><small>#${p.rank}${stale(p) ? ` (${since(p)})` : ""} · reach final ${pct(p.p.F, 0)}</small></div><div class="track"><span style="width:${(100 * p.p.W) / max}%"></span></div></div>
        <div class="pc">${pct(p.p.W, p.p.W < 0.1 ? 1 : 0)}</div></div>`).join("");
    $$("#fc-list .cont").forEach((d) => d.addEventListener("click", () => { fc.player = +d.dataset.i; renderForecast(); }));
    $("#fc-more").textContent = fc.n === 12 ? "Show top 32" : "Show top 12";
    const p = s.players[fc.player];
    $("#rt-who").innerHTML = `<b>${esc(p.name)}</b><span>ATP #${p.rank}${stale(p) ? ` (as of ${since(p)})` : ""}${p.seed ? ` · seed ${p.seed}` : ""} · ${pct(p.p.W)} title chance at ${esc(s.slam)} ${s.year}</span>${warn(p)}`;
    $("#rt-body").innerHTML = routeHTML(p);
  }

  // heat table across the four 2027 Slams
  (function () {
    const names = new Map();
    proj.forEach((s) => s.players.slice(0, 6).forEach((p) => names.set(p.name, true)));
    const rows = [...names.keys()].map((n) => ({ n, v: proj.map((s) => s.players.find((p) => p.name === n)?.p.W ?? 0) }))
      .sort((a, b) => b.v.reduce((x, y) => x + y) - a.v.reduce((x, y) => x + y));
    const mx = Math.max(...rows.flatMap((r) => r.v));
    $("#heat").innerHTML = `<tr><th>Player</th>${proj.map((s) => `<th class="c"><span class="dot" style="background:${SLAM[s.slam]?.c}"></span>${SLAM[s.slam]?.short || s.slam} ${String(s.year).slice(2)}</th>`).join("")}</tr>` +
      rows.map((r) => `<tr><td><b>${esc(r.n)}</b></td>${r.v.map((v, j) => {
        const a = v / mx;
        return `<td class="c" title="${esc(r.n)} · ${esc(proj[j].slam)}: ${pct(v)}" style="background:rgba(42,120,214,${(0.08 + 0.85 * a).toFixed(2)});color:${a > 0.5 ? "#fff" : "var(--ink)"}">${pct(v, 0)}</td>`;
      }).join("")}</tr>`).join("");
  })();

  const top2 = past.filter((e) => e.split !== "val" && e.champion_pred_position <= 2).length;
  $("#trust").innerHTML = `
    <div><b>${top2} of ${agg.n_slams}</b><span>past Slams where the eventual champion was in the model's top 2 before round 1.</span> <a href="#history">See them →</a></div>
    <div><b>${pct(agg.mean_p_champion_model, 0)} vs ${pct(agg.mean_p_champion_base, 0)}</b><span>average title chance given to the real champion — model vs ranking-only baseline.</span></div>
    <div><b>${fmt(tc[sel].log_loss, 3)} vs ${fmt(tc.rank_only.log_loss, 3)}</b><span>match-level test log-loss (lower is better), ${R.split.n_test.toLocaleString()} unseen matches.</span> <a href="#sec-results">Method →</a></div>`;

  // ================================================================== 2. HEAD TO HEAD
  const P = H.players.slice().sort((a, b) => a.rank - b.rank || (a.last_played < b.last_played ? 1 : -1));
  const latest = P.reduce((m, p) => (p.last_played > m ? p.last_played : m), "");
  const dayFmt = (d) => new Date(d).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
  // ranks come from each player's latest match; show the week when it is not the latest week
  const rankTxt = (p) => `#${p.rank}${stale(p) ? ` in ${since(p)}` : p.last_played < latest ? ` · ${dayFmt(p.last_played)}` : ""}`;
  const opts = P.map((p, i) => `<option value="${i}">${esc(p.name)} (${rankTxt(p)})</option>`).join("");
  $("#h2h-a").innerHTML = opts; $("#h2h-b").innerHTML = opts;
  const hs = { a: P[0], b: P[1], surf: "Hard", bo: 5 };
  const sync = () => { $("#h2h-a").value = P.indexOf(hs.a); $("#h2h-b").value = P.indexOf(hs.b); };
  sync();
  $("#h2h-a").addEventListener("change", (e) => { hs.a = P[+e.target.value]; renderH2H(); });
  $("#h2h-b").addEventListener("change", (e) => { hs.b = P[+e.target.value]; renderH2H(); });
  $("#h2h-swap").addEventListener("click", () => { [hs.a, hs.b] = [hs.b, hs.a]; sync(); renderH2H(); });
  const seg = (id, key, num) => $$(`#${id} button`).forEach((b) => b.addEventListener("click", () => {
    $$(`#${id} button`).forEach((x) => x.classList.toggle("on", x === b)); hs[key] = num ? +b.dataset.v : b.dataset.v; renderH2H(); }));
  seg("h2h-surf", "surf"); seg("h2h-bo", "bo", true);
  const shr = (w, n) => (w + H.constants.shrink_k / 2) / (n + H.constants.shrink_k);

  function renderH2H() {
    const { a, b, surf, bo } = hs;
    if (a.id === b.id) { $("#duel").innerHTML = ""; $("#h2h-meta").textContent = "Pick two different players."; return; }
    const r = TennisModel.predict(H, a, b, surf, bo);
    $("#duel").innerHTML = `<div style="flex:${Math.max(r.p, 0.12)};background:var(--a)"><b>${pct(r.p)}</b><small>${esc(a.name)}</small></div>
      <div class="r" style="flex:${Math.max(1 - r.p, 0.12)};background:var(--b)"><b>${pct(1 - r.p)}</b><small>${esc(b.name)}</small></div>`;
    const hab = H.h2h[a.id + "-" + b.id] || 0, hba = H.h2h[b.id + "-" + a.id] || 0;
    $("#h2h-meta").innerHTML = `${H.model.name} (calibrated) · ${surf} court · best of ${bo} · form as of ${bind.cutoff}. ` +
      [a, b].filter(stale).map((p) => `⚠ ${esc(p.name)} has no matches since ${since(p)} — his rank is as of then.`).join(" ");
    const ws = (p) => [p["ws_" + surf], p["ns_" + surf]];
    // d > 0 means the row favours A (shown bold); 0 = neutral
    const rows = [
      ["ATP rank", "#" + a.rank + (stale(a) ? ` (${since(a)})` : ""), "#" + b.rank + (stale(b) ? ` (${since(b)})` : ""), b.rank - a.rank],
      ["Ranking points", (a.points ?? 0).toLocaleString(), (b.points ?? 0).toLocaleString(), (a.points ?? 0) - (b.points ?? 0)],
      ["Last 6 months", `${a.w6}–${a.n6 - a.w6}`, `${b.w6}–${b.n6 - b.w6}`, shr(a.w6, a.n6) - shr(b.w6, b.n6)],
      [`${surf}, 24 months`, `${ws(a)[0]}–${ws(a)[1] - ws(a)[0]}`, `${ws(b)[0]}–${ws(b)[1] - ws(b)[0]}`, shr(...ws(a)) - shr(...ws(b))],
      ["Head-to-head wins", hab, hba, hab - hba],
      ["Age", fmt(a.age, 1), fmt(b.age, 1), 0],
      ["Height", a.height ? a.height + " cm" : "–", b.height ? b.height + " cm" : "–", 0],
    ];
    $("#tape").innerHTML = `<div class="l hd">${esc(a.name)}</div><div class="m hd"></div><div class="hd">${esc(b.name)}</div>` +
      rows.map(([k, x, y, d]) => `<div class="l ${d > 0 ? "w" : ""}">${x}</div><div class="m">${k}</div><div class="${d < 0 ? "w" : ""}">${y}</div>`).join("");
    setChart("contrib", (box) => {
      const items = r.names.map((n, i) => ({ n, c: r.contrib[i] })).filter((d) => Math.abs(d.c) > 1e-4).sort((x, y) => Math.abs(y.c) - Math.abs(x.c)).slice(0, 8);
      const W = Math.max(300, box.clientWidth), rowH = 26, Hh = items.length * rowH + 20, x0 = Math.min(170, W * 0.38), mid = x0 + (W - x0) / 2;
      const mx = Math.max(0.05, ...items.map((d) => Math.abs(d.c)));
      const svg = el("svg", { viewBox: `0 0 ${W} ${Hh}` });
      el("line", { x1: mid, x2: mid, y1: 0, y2: Hh - 16, stroke: "var(--ink3)" }, svg);
      items.forEach((d, i) => {
        const y = i * rowH, w = (Math.abs(d.c) / mx) * ((W - x0) / 2 - 40);
        el("text", { x: x0 - 8, y: y + 15, "text-anchor": "end" }, svg).textContent = d.n;
        el("rect", { x: d.c > 0 ? mid : mid - w, y: y + 3, width: Math.max(w, 1), height: rowH - 8, rx: 4, fill: d.c > 0 ? "var(--a)" : "var(--b)" }, svg);
        el("text", { x: d.c > 0 ? mid + w + 5 : mid - w - 5, y: y + 15, "text-anchor": d.c > 0 ? "start" : "end" }, svg).textContent = (d.c > 0 ? "+" : "") + d.c.toFixed(2);
        hover(el("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }, svg), `${d.n} = ${fmt(r.features[d.n], 3)}<br>pushes toward ${esc(d.c > 0 ? a.name : b.name)}`);
      });
      el("text", { x: mid, y: Hh - 2, "text-anchor": "middle" }, svg).textContent = "← favours B · log-odds · favours A →";
      box.replaceChildren(svg);
    });
  }

  // ================================================================== 3. HISTORY
  const tested = past.filter((e) => e.split !== "val");
  const top1 = tested.filter((e) => e.champion_pred_position === 1).length;
  $("#hist-tiles").innerHTML = `
    <div><b>${top1} / ${tested.length}</b><span>champion was the model's #1 favourite</span></div>
    <div><b>${top2} / ${tested.length}</b><span>champion was in the model's top 2</span></div>
    <div><b>${pct(agg.mean_p_champion_model, 0)}</b><span>avg title chance given to the champion (baseline ${pct(agg.mean_p_champion_base, 0)})</span></div>
    <div><b>${fmt(agg.model.stage_loglik, 2)} vs ${fmt(agg.base.stage_loglik, 2)}</b><span>bracket log-score, model vs baseline (higher = better)</span></div>`;
  const shortOf = (label) => { const [y] = label.match(/\d{4}/); const n = label.replace(/\s*\d{4}.*/, ""); return `${SLAM[n]?.short || n} '${y.slice(2)}`; };
  // compact scoreboard: rows = Slam, columns = year
  const slamOf = (label) => label.replace(/\s*\d{4}.*/, ""), yearOf = (label) => +label.match(/\d{4}/)[0];
  const years = [...new Set(past.map((e) => yearOf(e.label)))].sort();
  $("#chips").innerHTML = `<tr><th></th>${years.map((y) => `<th class="c">${y}<small>${y === R.split.val ? "validation" : y <= R.split.test[1] ? "test" : "after test"}</small></th>`).join("")}</tr>` +
    Object.keys(SLAM).map((sn) => `<tr><td><span class="s" style="background:${SLAM[sn].c}"></span><b>${sn}</b></td>${years.map((y) => {
      const i = past.findIndex((e) => slamOf(e.label) === sn && yearOf(e.label) === y);
      if (i < 0) return "<td></td>";
      const e = past[i];
      return `<td class="c"><button class="cell" data-i="${i}" title="${esc(e.label)}"><span class="pos ${e.champion_pred_position === 1 ? "p1" : ""}">#${e.champion_pred_position}</span>${esc(e.champion.split(" ").slice(-1)[0])}</button></td>`;
    }).join("")}</tr>`).join("");
  $$("#chips .cell").forEach((c) => c.addEventListener("click", () => renderHistory(+c.dataset.i)));
  async function renderHistory(i) {
    const e = past[i], s = await loadSlam(e.key);
    $$("#chips .cell").forEach((c) => c.classList.toggle("on", +c.dataset.i === i));
    $("#hist-title").textContent = `${e.label}: pre-tournament favourites`;
    $("#hist-split").className = "conf " + (e.split === "val" ? "lo" : "hi");
    $("#hist-split").textContent = e.split === "val" ? "validation year" : e.split === "test" ? "test year" : "after test period";
    $("#hist-note").innerHTML = `Champion <strong>${esc(e.champion)}</strong> (ATP #${e.champion_rank ?? "–"}) was ranked <strong>#${e.champion_pred_position}</strong> by the model with a ${pct(e.p_champion_model)} title chance; the ranking-only baseline gave him ${pct(e.p_champion_base)}.` +
      (e.split === "val" ? " (2022 was also used to calibrate the model, so it is not a clean test.)" : "");
    const base = Object.fromEntries((s.baseline_top || []).map((d) => [d.name, d.p]));
    let top = s.players.slice(0, 12);
    const ci = s.players.findIndex((p) => p.actual === 7);
    if (ci >= 12) top = top.concat([s.players[ci]]);
    setChart("hist", (box) => hbar(box, top.map((p) => ({
      label: `${p.name}${p.seed ? ` [${p.seed}]` : ""}`, value: p.p.W, value2: base[p.name] ?? 0, color: p.actual === 7 ? "var(--champ)" : "var(--a)",
      tip: `<b>${esc(p.name)}</b> · ATP #${p.rank ?? "–"}<br>model ${pct(p.p.W)} · baseline ${pct(base[p.name] ?? 0)}<br>actually ${ROUND_OUT[p.actual ?? 0]}`,
    })), { fmtV: (v) => pct(v, 0), labelW: 200 }));
  }

  // ================================================================== 4. METHOD (rubric sections)
  $("#expand-all").addEventListener("click", () => { const open = !$$("details.sec").every((d) => d.open); $$("details.sec").forEach((d) => (d.open = open)); renderCharts(); });
  $$("details.sec").forEach((d) => d.addEventListener("toggle", () => renderCharts()));
  $("#kpis").innerHTML = `
    <div><b>${fmt(tc[sel].log_loss, 4)}</b><span>test log-loss · ${N[sel]} (baseline ${fmt(tc.rank_only.log_loss, 4)})</span></div>
    <div><b>${pct(tc[sel].accuracy)}</b><span>test accuracy (higher-rank-wins ${pct(R.favourite_accuracy_test)})</span></div>
    <div><b>${R.split.n_test.toLocaleString()}</b><span>test matches, ${R.split.test.join("–")}</span></div>
    <div><b>${agg.n_slams}</b><span>held-out Slams simulated on real draws</span></div>`;
  table("cv-table", [{ t: "Model" }, { t: "Best setting" }, { t: "CV log-loss", n: 1 }, { t: "Rows", n: 1 }],
    Object.entries(R.cv).map(([k, v]) => [N[k], Object.entries(v.best_params).map(([a, b]) => `<code>${a}=${b}</code>`).join(" "), fmt(v.cv_log_loss, 4), v.train_rows.toLocaleString()]));
  $("#timeline").innerHTML = `<div style="flex:12;background:var(--a)">Train ${R.split.train.join("–")}</div><div style="flex:2;background:#1baf7a">Val ${R.split.val}</div>` +
    `<div style="flex:3;background:var(--b)">Test ${R.split.test.join("–")}</div><div style="flex:3;background:var(--base);color:var(--ink)">Later Slams · 2027 forecast</div>`;
  const order = ["rank_only", "lasso", "svm", "rf", "xgb"];
  table("model-table", [{ t: "Model" }, { t: "Val (raw)", n: 1 }, { t: "Test raw", n: 1 }, { t: "Test calibrated", n: 1 }, { t: "Accuracy", n: 1 }, { t: "AUC", n: 1 }, { t: "Brier", n: 1 }],
    order.map((k) => [N[k] + (k === sel ? " ✓" : ""), fmt(R.val[k].log_loss, 4), fmt(R.test_raw[k].log_loss, 4), fmt(tc[k].log_loss, 4), pct(tc[k].accuracy), fmt(tc[k].roc_auc, 3), fmt(tc[k].brier, 4)]),
    (r, i) => order[i] === sel);
  const b1 = R.bootstrap.selected_vs_rank_only, b2 = R.bootstrap.selected_vs_lasso;
  const sig = (b) => (b.hi < 0 ? "clearly better (95% CI below zero)" : b.lo > 0 ? "worse" : "statistically tied (CI includes zero)");
  $("#results-summary").innerHTML = `Validation picked <strong>${N[sel]}</strong>. On ${R.split.n_test.toLocaleString()} unseen test matches its calibrated log-loss is
    <strong>${fmt(tc[sel].log_loss, 4)}</strong> vs <strong>${fmt(tc.rank_only.log_loss, 4)}</strong> for the baseline (Δ ${fmt(b1.diff, 4)}, CI ${fmt(b1.lo, 4)} to ${fmt(b1.hi, 4)}): ${sig(b1)}.
    Versus Lasso: Δ ${fmt(b2.diff, 4)} (${fmt(b2.lo, 4)} to ${fmt(b2.hi, 4)}), ${sig(b2)}. Accuracy ${pct(tc[sel].accuracy)} vs ${pct(R.favourite_accuracy_test)} for “higher rank wins” — upsets are common, so gains are modest.
    ${R.web_model !== sel ? `The head-to-head tool runs ${N[R.web_model]} (val log-loss ${fmt(R.val[R.web_model].log_loss, 4)}, nearly identical), because a tree-boosted model can be shipped to the browser compactly.` : ""}`;
  setChart("ll", (box) => {
    const it = order.map((k) => ({ label: N[k].replace(" (baseline)", ""), value: tc[k].log_loss, color: k === "rank_only" ? "var(--base)" : k === sel ? "var(--a)" : "color-mix(in srgb, var(--a) 45%, transparent)" }));
    const lo = Math.min(...it.map((d) => d.value)), hi = Math.max(...it.map((d) => d.value));
    hbar(box, it, { min: Math.floor((lo - 0.01) * 100) / 100, max: hi + 0.004, labelW: 190 });
  });
  setChart("rel", (box) => {
    const W = Math.max(280, box.clientWidth), Hh = Math.min(340, W * 0.8), m = { l: 40, r: 10, t: 8, b: 34 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${Hh}` });
    const sx = (v) => m.l + v * (W - m.l - m.r), sy = (v) => Hh - m.b - v * (Hh - m.t - m.b);
    const g = el("g", { class: "grid" }, svg);
    for (let k = 0; k <= 5; k++) {
      const v = k / 5;
      el("line", { x1: sx(v), x2: sx(v), y1: sy(0), y2: sy(1) }, g); el("line", { x1: sx(0), x2: sx(1), y1: sy(v), y2: sy(v) }, g);
      el("text", { x: sx(v), y: Hh - m.b + 15, "text-anchor": "middle" }, svg).textContent = v.toFixed(1);
      el("text", { x: m.l - 6, y: sy(v) + 4, "text-anchor": "end" }, svg).textContent = v.toFixed(1);
    }
    el("text", { x: (W + m.l) / 2, y: Hh - 3, "text-anchor": "middle" }, svg).textContent = "predicted";
    el("line", { x1: sx(0), y1: sy(0), x2: sx(1), y2: sy(1), stroke: "var(--base)", "stroke-width": 2, "stroke-dasharray": "5 4" }, svg);
    [["raw", "var(--b)"], ["cal", "var(--a)"]].forEach(([k, c]) => {
      const pts = R.reliability[k];
      el("polyline", { points: pts.map((p) => `${sx(p.pred)},${sy(p.obs)}`).join(" "), fill: "none", stroke: c, "stroke-width": 2 }, svg);
      pts.forEach((p) => hover(el("circle", { cx: sx(p.pred), cy: sy(p.obs), r: 5, fill: c, stroke: "var(--panel)", "stroke-width": 2 }, svg),
        `${k === "cal" ? "calibrated" : "raw"}: predicted ${pct(p.pred)} → won ${pct(p.obs)} (${p.n} matches)`));
    });
    box.replaceChildren(svg);
  });
  setChart("imp", (box) => hbar(box, R.permutation_importance_val.map((d) => ({ label: d.feature, value: Math.max(0, d.mean), tip: `${d.feature}: +${fmt(d.mean, 4)} log-loss when shuffled` })), { labelW: 170 }));
  const eh = [{ t: "Group" }, { t: "Matches", n: 1 }, { t: "Accuracy", n: 1 }, { t: "Log-loss", n: 1 }, { t: "Baseline", n: 1 }];
  const er = (a) => a.map((g) => [g.group, g.n.toLocaleString(), pct(g.accuracy), fmt(g.log_loss, 4), fmt(g.base_log_loss, 4)]);
  table("err-surface", eh, er(R.by_surface)); table("err-level", eh, er(R.by_level));
  table("err-gap", [{ t: "Rank ratio" }, ...eh.slice(1)], er(R.by_rank_gap));
  const gp = R.by_rank_gap;
  $("#err-gap-note").textContent = `Near a coin flip between similarly ranked players (${pct(gp[0].accuracy)} accuracy), much more reliable with a big ranking gap (${pct(gp.at(-1).accuracy)}).`;
  table("err-miss", [{ t: "Date" }, { t: "Event" }, { t: "Rd" }, { t: "Winner (rank)" }, { t: "Loser (rank)" }, { t: "P(winner)", n: 1 }],
    R.worst_misses.map((m) => [m.date, esc(m.event), m.round, `${esc(m.winner)} (${m.winner_rank ?? "–"})`, `${esc(m.loser)} (${m.loser_rank ?? "–"})`, pct(m.p_winner)]));
  const f0 = proj[0];
  $("#conclusion-text").innerHTML = `Adding form, surface record, activity and head-to-head to the ranking gives a small but statistically clear gain in match probabilities
    (test log-loss ${fmt(tc[sel].log_loss, 4)} vs ${fmt(tc.rank_only.log_loss, 4)}); tree ensembles and Lasso perform about the same, so the signal is mostly additive.
    Simulated over real draws, these probabilities put the eventual champion in the model's top 2 in ${top2} of ${agg.n_slams} held-out Slams and gave him ${pct(agg.mean_p_champion_model, 0)} title chance on average (baseline ${pct(agg.mean_p_champion_base, 0)}).
    For 2027 the model's clearest call is the ${esc(f0.slam)}, where ${esc(f0.players[0].name)} leads at ${pct(f0.players[0].p.W, 0)} — a favourite, not a certainty. The model ranks contenders well; it cannot name a single winner.`;

  // ------------------------------------------------------------------ start
  renderForecast();
  renderH2H();
  renderHistory(past.length - 1);
  route();
})();
