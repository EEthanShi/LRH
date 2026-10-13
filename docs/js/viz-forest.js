/* SOG card, illustration of Proposition 2: 72 samples in the plane from three seeded blobs form model A, and model B
   adds independent Gaussian noise with sd d times the median nearest-neighbor distance of A. Both minimum spanning
   trees (Prim), both nearest-neighbor forests, MST overlap, one-nearest-neighbor overlap and the bound are exact. */
(function () {
  "use strict";
  var V = window.LRHViz, root = document.getElementById("viz-forest");
  if (!V || !root) return;
  var M = 72, BLOBS = [[30, -1.05, 0.55, 0.44], [24, 0.95, 0.85, 0.34], [18, 0.1, -1.3, 0.27]];
  function key(i, j) { return i < j ? i * M + j : j * M + i; }

  function sample(seed) {
    var r = V.rng(9973 * seed + 17), P = [], U = [];
    BLOBS.forEach(function (b) { for (var n = 0; n < b[0]; n++) P.push([b[1] + b[3] * V.gauss(r), b[2] + b[3] * V.gauss(r)]); });
    for (var i = 0; i < M; i++) U.push([V.gauss(r), V.gauss(r)]);
    var nn = model(P).nnd.slice().sort(function (a, b) { return a - b; });
    return { P: P, U: U, med: Math.sqrt((nn[35] + nn[36]) / 2) };
  }
  function shift(S, d) { return S.P.map(function (p, i) { return [p[0] + d * S.med * S.U[i][0], p[1] + d * S.med * S.U[i][1]]; }); }

  /* nearest neighbors, forest, mutual pairs and the MST (Prim) of one model */
  function model(Q) {
    var D = [], nn = new Int32Array(M), nnd = new Float64Array(M), i, j;
    for (i = 0; i < M; i++) { D.push(new Float64Array(M)); nnd[i] = Infinity; }
    for (i = 0; i < M; i++) for (j = 0; j < M; j++) {
      var dx = Q[i][0] - Q[j][0], dy = Q[i][1] - Q[j][1]; D[i][j] = dx * dx + dy * dy;
      if (i !== j && D[i][j] < nnd[i]) { nnd[i] = D[i][j]; nn[i] = j; }
    }
    var F = {}, nmut = 0;
    for (i = 0; i < M; i++) { F[key(i, nn[i])] = 1; if (nn[nn[i]] === i && i < nn[i]) nmut++; }
    var inT = new Uint8Array(M), best = new Float64Array(M).fill(Infinity), par = new Int32Array(M).fill(-1), T = [];
    best[0] = 0;
    for (var it = 0; it < M; it++) {
      var u = -1; for (var v = 0; v < M; v++) if (!inT[v] && (u < 0 || best[v] < best[u])) u = v;
      inT[u] = 1;
      if (par[u] >= 0) T.push({ i: u, j: par[u], w: D[u][par[u]], k: key(u, par[u]), f: !!F[key(u, par[u])] });
      for (v = 0; v < M; v++) if (!inT[v] && D[u][v] < best[v]) { best[v] = D[u][v]; par[v] = u; }
    }
    T.sort(function (a, b) { return a.w - b.w; }).forEach(function (e, n) { e.rank = n; });
    return { Q: Q, nn: nn, nnd: nnd, F: F, nmut: nmut, T: T, nF: M - nmut };
  }
  function compare(a, b) {
    var inB = {}, s = { sf: 0, sb: 0, fa: 0, n1: 0 };
    b.T.forEach(function (e) { inB[e.k] = 1; });
    a.T.forEach(function (e) { e.sh = !!inB[e.k]; if (e.f) s.fa++; if (e.sh) { if (e.f) s.sf++; else s.sb++; } });
    var inA = {}; a.T.forEach(function (e) { inA[e.k] = 1; }); b.T.forEach(function (e) { e.sh = !!inA[e.k]; });
    for (var i = 0; i < M; i++) if (a.nn[i] === b.nn[i]) s.n1++;
    s.n1 /= M; s.st = (s.sf + s.sb) / (M - 1);
    s.bound = Math.max(M * s.n1 - a.nmut, M * s.n1 / 2) / (M - 1);
    return s;
  }

  /* ---------- drawing ---------- */
  var svA = root.querySelector(".sv-a"), svB = root.querySelector(".sv-b"), slider = root.querySelector("input[type=range]"), out = root.querySelector("output");
  var seed = 1, S = sample(seed), d = +slider.value / 100, frame = null, cur = null, stopBuild = null;
  function fit() {
    var all = S.P.concat(shift(S, 1.2)), xs = all.map(function (p) { return p[0]; }), ys = all.map(function (p) { return p[1]; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var sc = Math.min(216 / (x1 - x0), 216 / (y1 - y0));
    frame = function (p) { return [120 + sc * (p[0] - (x0 + x1) / 2), 120 - sc * (p[1] - (y0 + y1) / 2)]; };
  }
  function drawTree(sv, m) {
    sv.textContent = "";
    var gE = V.el("g", {}, sv), gP = V.el("g", {}, sv), Q = m.Q.map(frame);
    m.T.forEach(function (e) {
      e.el = V.el("line", { "class": "vf-e " + (e.f ? "f" : "b") + (e.sh ? " sh" : ""), x1: Q[e.i][0].toFixed(1), y1: Q[e.i][1].toFixed(1), x2: Q[e.j][0].toFixed(1), y2: Q[e.j][1].toFixed(1) }, gE);
    });
    Q.forEach(function (q) { V.el("circle", { "class": "vf-pt", cx: q[0].toFixed(1), cy: q[1].toFixed(1), r: 2.6 }, gP); });
  }
  function put(sel, v) { root.querySelector('[data-read="' + sel + '"]').textContent = v; }
  function update(build) {
    var a = model(S.P), b = model(shift(S, d)), s = compare(a, b);
    cur = { a: a, b: b, s: s };
    drawTree(svA, a); drawTree(svB, b);
    out.textContent = "d = " + d.toFixed(2); slider.setAttribute("aria-valuetext", "noise of " + d.toFixed(2) + " times the median nearest-neighbor distance");
    put("st", s.st.toFixed(2)); put("sf", (s.sf / (M - 1)).toFixed(2)); put("sb", (s.sb / (M - 1)).toFixed(2));
    put("n1", s.n1.toFixed(2)); put("bd", s.bound.toFixed(2));
    put("fa", s.fa + " of " + (M - 1));
    var sfa = root.querySelector('[data-read="sfa"]'); sfa.textContent = "";
    ["Of the ", s.sf + s.sb, " shared tree edges, ", s.sf, " lie in the forest of A"].forEach(function (t, n) { sfa.appendChild(n % 2 ? document.createElement("b") : document.createTextNode("")).textContent = t; });
    var bar = root.querySelector(".v-bar");
    bar.style.setProperty("--f", s.sf / (M - 1)); bar.style.setProperty("--b", s.sb / (M - 1));
    bar.style.setProperty("--n1", s.n1); bar.style.setProperty("--bd", s.bound);
    var txt = "MST overlap " + s.st.toFixed(2) + ", of which " + s.sf + " shared edges lie in the nearest-neighbor forest of model A and " + s.sb + " are bridges; one-nearest-neighbor overlap " + s.n1.toFixed(2) + ".";
    V.say(svA, "Minimum spanning tree of model A, with nearest-neighbor forest edges and bridges. " + txt);
    V.say(svB, "Minimum spanning tree of model B, with nearest-neighbor forest edges and bridges. " + txt);
    if (build) grow();
  }
  /* both trees appear in Kruskal order, shortest edge first */
  function grow() {
    if (stopBuild) stopBuild();
    if (V.REDUCED) return;
    var lines = cur.a.T.concat(cur.b.T);
    lines.forEach(function (e) { e.el.style.opacity = 0; });
    stopBuild = V.tween(2400, function (p) {
      var n = p * (M - 1);
      lines.forEach(function (e) { if (e.rank < n && e.el.style.opacity === "0") e.el.style.opacity = ""; });
    });
  }
  var touched = false;
  root.addEventListener("pointerdown", function () { touched = true; });
  root.addEventListener("keydown", function (ev) { if (ev.key !== "Tab" && ev.key !== "Shift") touched = true; });
  slider.addEventListener("input", function () { touched = true; d = +slider.value / 100; if (stopBuild) stopBuild(); update(false); });
  root.querySelector("[data-new]").addEventListener("click", function () { S = sample(++seed); fit(); update(true); });
  root.querySelector("[data-replay]").addEventListener("click", function () { update(true); });
  fit(); update(false);
  if (!V.REDUCED) { cur.a.T.concat(cur.b.T).forEach(function (e) { e.el.style.opacity = 0; }); V.inView(root, function () { setTimeout(function () { if (!touched) grow(); else cur.a.T.concat(cur.b.T).forEach(function (e) { e.el.style.opacity = ""; }); }, 250); }); }

  /* ---------- self-test over 50 samples and four noise levels ---------- */
  V.test("SOG forest", "forest inside each tree, at least half of it, and MST overlap above the bound (200 cases)", function () {
    var bad = [];
    for (var sd = 101; sd < 151; sd++) {
      var T = sample(sd);
      [0, 0.25, 0.5, 1].forEach(function (dd) {
        var a = model(T.P), b = model(shift(T, dd)), s = compare(a, b);
        [a, b].forEach(function (m) {
          var inT = {}; m.T.forEach(function (e) { inT[e.k] = 1; });
          if (Object.keys(m.F).some(function (k) { return !inT[k]; })) bad.push(sd + "/" + dd + " forest not in tree");
          if (Object.keys(m.F).length !== m.nF || m.nF < M / 2) bad.push(sd + "/" + dd + " forest size");
        });
        if (s.st < s.bound - 1e-12) bad.push(sd + "/" + dd + " bound");
        var shared = {}; a.T.forEach(function (e) { if (e.sh) shared[e.k] = 1; });
        for (var i = 0; i < M; i++) if (a.nn[i] === b.nn[i] && !shared[key(i, a.nn[i])]) bad.push(sd + "/" + dd + " shared nearest-neighbor edge");
        if (dd === 0 && (s.st !== 1 || s.n1 !== 1)) bad.push(sd + " identical models");
      });
    }
    return { pass: !bad.length, detail: bad.slice(0, 4).join("; ") || "200 cases" };
  });
})();
