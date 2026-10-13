/* Overview example: the squared distances of model B are those of model A passed through the link
   g_t(D) = 2(1 - exp(-D/(2t))) of Theorem 3. Four scores are computed live from the two matrices:
   mKNN (k = 3), MST overlap (Prim), Spearman over the entries i < j, and CKA of HD_AH and HD_BH. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var M = 48, K = 3, T_MAX = 4, T_MIN = 0.1, SEED = 11;
  var REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(r) { var u = 0; while (u === 0) u = r(); var v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  // 48 points in 2-D, two overlapping elongated clusters, fixed seed
  function makePoints() {
    var r = mulberry32(SEED), P = [];
    for (var i = 0; i < M; i++) {
      var c = i < 26 ? 0 : 1, a = c ? 0.6 : -0.5, cx = c ? 0.75 : -0.75, cy = c ? 0.3 : -0.3;
      var u = gauss(r) * 0.65, v = gauss(r) * 0.45;
      P.push([cx + u * Math.cos(a) - v * Math.sin(a), cy + u * Math.sin(a) + v * Math.cos(a)]);
    }
    return P;
  }
  function sqDist(P) {
    var D = [], mx = 0, i, j;
    for (i = 0; i < M; i++) D.push(new Float64Array(M));
    for (i = 0; i < M; i++) for (j = 0; j < M; j++) {
      var dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1];
      D[i][j] = dx * dx + dy * dy; if (D[i][j] > mx) mx = D[i][j];
    }
    for (i = 0; i < M; i++) for (j = 0; j < M; j++) D[i][j] *= 4 / mx; // largest squared distance = 4
    return D;
  }
  function g(d, t) { return -2 * Math.expm1(-d / (2 * t)); } // keeps distinct values at small t
  function bend(D, t) {
    var B = [];
    for (var i = 0; i < M; i++) { var row = new Float64Array(M); for (var j = 0; j < M; j++) row[j] = i === j ? 0 : g(D[i][j], t); B.push(row); }
    return B;
  }
  function knn(D, k) {
    var out = [];
    for (var i = 0; i < M; i++) {
      var idx = []; for (var j = 0; j < M; j++) if (j !== i) idx.push(j);
      idx.sort(function (a, b) { return D[i][a] - D[i][b]; });
      out.push(idx.slice(0, k));
    }
    return out;
  }
  function mknn(nA, nB, k) {
    var s = 0;
    for (var i = 0; i < M; i++) for (var a = 0; a < k; a++) if (nB[i].indexOf(nA[i][a]) >= 0) s++;
    return s / (M * k);
  }
  function mst(D) {
    var inT = new Array(M).fill(false), best = new Float64Array(M).fill(Infinity), par = new Int32Array(M).fill(-1), E = [];
    best[0] = 0;
    for (var it = 0; it < M; it++) {
      var u = -1;
      for (var v = 0; v < M; v++) if (!inT[v] && (u < 0 || best[v] < best[u])) u = v;
      inT[u] = true;
      if (par[u] >= 0) E.push(Math.min(u, par[u]) + "-" + Math.max(u, par[u]));
      for (v = 0; v < M; v++) if (!inT[v] && D[u][v] < best[v]) { best[v] = D[u][v]; par[v] = u; }
    }
    return E;
  }
  function mstOverlap(eA, eB) { var s = 0, set = {}; eB.forEach(function (e) { set[e] = 1; }); eA.forEach(function (e) { if (set[e]) s++; }); return s / (M - 1); }
  function upper(D) { var x = []; for (var i = 0; i < M; i++) for (var j = i + 1; j < M; j++) x.push(D[i][j]); return x; }
  function ranks(x) {
    var idx = x.map(function (_, i) { return i; }).sort(function (a, b) { return x[a] - x[b]; });
    var r = new Float64Array(x.length); idx.forEach(function (v, i) { r[v] = i; }); return r;
  }
  function pearson(x, y) {
    var n = x.length, mx = 0, my = 0, i; for (i = 0; i < n; i++) { mx += x[i]; my += y[i]; } mx /= n; my /= n;
    var sxy = 0, sxx = 0, syy = 0; for (i = 0; i < n; i++) { var a = x[i] - mx, b = y[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
    return sxy / Math.sqrt(sxx * syy);
  }
  function centered(D) {
    var rm = new Float64Array(M), gm = 0, i, j;
    for (i = 0; i < M; i++) { for (j = 0; j < M; j++) rm[i] += D[i][j]; rm[i] /= M; gm += rm[i]; }
    gm /= M;
    var C = []; for (i = 0; i < M; i++) { var row = new Float64Array(M); for (j = 0; j < M; j++) row[j] = D[i][j] - rm[i] - rm[j] + gm; C.push(row); }
    return C;
  }
  function cka(CA, B) {
    var CB = centered(B), ab = 0, aa = 0, bb = 0;
    for (var i = 0; i < M; i++) for (var j = 0; j < M; j++) { ab += CA[i][j] * CB[i][j]; aa += CA[i][j] * CA[i][j]; bb += CB[i][j] * CB[i][j]; }
    return ab / Math.sqrt(aa * bb);
  }

  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function init() {
    var root = document.getElementById("bend");
    if (!root) return;
    var P = makePoints(), DA = sqDist(P);
    var nA = knn(DA, K), eA = mst(DA), rA = ranks(upper(DA)), CA = centered(DA);
    var pairsA = upper(DA);

    // --- points panel (model A, with its spanning tree and the neighbors of one sample) ---
    var sp = root.querySelector("#bend-points");
    var xs = P.map(function (p) { return p[0]; }), ys = P.map(function (p) { return p[1]; });
    var minx = Math.min.apply(null, xs), maxx = Math.max.apply(null, xs), miny = Math.min.apply(null, ys), maxy = Math.max.apply(null, ys);
    var sc = Math.min(212 / (maxx - minx), 212 / (maxy - miny));
    var ox = 120 - sc * (minx + maxx) / 2, oy = 120 + sc * (miny + maxy) / 2;
    var Q = P.map(function (p) { return [ox + sc * p[0], oy - sc * p[1]]; });
    var gMst = el("g", {}, sp), gNn = el("g", {}, sp), gPts = el("g", {}, sp), gHit = el("g", {}, sp);
    eA.forEach(function (e) {
      var ij = e.split("-").map(Number);
      el("line", { "class": "bp-mst", x1: Q[ij[0]][0], y1: Q[ij[0]][1], x2: Q[ij[1]][0], y2: Q[ij[1]][1] }, gMst);
    });
    var ptEls = Q.map(function (q) { return el("circle", { "class": "bp-pt", cx: q[0], cy: q[1], r: 3.3 }, gPts); });
    var ring = el("circle", { "class": "bp-ring", r: 7.5 }, gPts);
    var sel = 7;
    function select(i) {
      sel = i;
      gNn.innerHTML = "";
      nA[i].forEach(function (j) { el("line", { "class": "bp-nn", x1: Q[i][0], y1: Q[i][1], x2: Q[j][0], y2: Q[j][1] }, gNn); });
      ptEls.forEach(function (c, j) { c.setAttribute("class", "bp-pt" + (j === i ? " sel" : nA[i].indexOf(j) >= 0 ? " nb" : "")); c.setAttribute("r", j === i ? 4.2 : 3.3); });
      ring.setAttribute("cx", Q[i][0]); ring.setAttribute("cy", Q[i][1]);
    }
    var COARSE = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
    Q.forEach(function (q, i) {
      var h = el("circle", { "class": "bp-hit", cx: q[0], cy: q[1], r: COARSE ? 16 : 9 }, gHit);
      h.addEventListener("click", function () { select(i); });
    });
    sp.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowRight" || ev.key === "ArrowUp") { ev.preventDefault(); select((sel + 1) % M); }
      if (ev.key === "ArrowLeft" || ev.key === "ArrowDown") { ev.preventDefault(); select((sel + M - 1) % M); }
    });
    select(sel);

    // --- curve panel: g_t against the identity, rescaled to the same largest value ---
    var cp = root.querySelector("#bend-curve");
    var X0 = 40, X1 = 226, Y0 = 200, Y1 = 14;
    function px(d) { return X0 + (X1 - X0) * d / 4; }
    function py(d) { return Y0 - (Y0 - Y1) * d / 4; }
    el("line", { "class": "bc-axis", x1: X0, y1: Y0, x2: X1, y2: Y0 }, cp);
    el("line", { "class": "bc-axis", x1: X0, y1: Y0, x2: X0, y2: Y1 }, cp);
    [0, 2, 4].forEach(function (v) {
      var tx = el("text", { "class": "bc-tick", x: px(v), y: Y0 + 15, "text-anchor": "middle" }, cp); tx.textContent = v;
      var ty = el("text", { "class": "bc-tick", x: X0 - 6, y: py(v) + 3.5, "text-anchor": "end" }, cp); ty.textContent = v;
    });
    var lx = el("text", { "class": "bc-lab", x: (X0 + X1) / 2, y: 234, "text-anchor": "middle" }, cp); lx.textContent = "squared distance, model A";
    var ly = el("text", { "class": "bc-lab", x: 13, y: (Y0 + Y1) / 2, "text-anchor": "middle", transform: "rotate(-90 13 " + (Y0 + Y1) / 2 + ")" }, cp); ly.textContent = "model B, rescaled";
    el("line", { "class": "bc-id", x1: px(0), y1: py(0), x2: px(4), y2: py(4) }, cp);
    var gPairs = el("g", {}, cp);
    var pairEls = pairsA.map(function (d) { return el("circle", { "class": "bc-pair", cx: px(d), cy: py(d), r: 1.7 }, gPairs); });
    var curve = el("path", { "class": "bc-curve" }, cp);

    // --- bars ---
    var rows = {};
    root.querySelectorAll(".bar-row").forEach(function (r) { rows[r.getAttribute("data-score")] = { fill: r.querySelector(".b-fill"), val: r.querySelector(".b-val") }; });
    function setBar(key, v) { rows[key].fill.style.transform = "scaleX(" + Math.max(0, Math.min(1, v)) + ")"; rows[key].val.textContent = v.toFixed(2); }

    var slider = root.querySelector("#bend-t"), out = root.querySelector("#bend-t-out");
    function tFromSlider(v) { return T_MAX * Math.pow(T_MIN / T_MAX, v / 1000); }
    function sliderFromT(t) { return Math.round(1000 * Math.log(t / T_MAX) / Math.log(T_MIN / T_MAX)); }

    function update(t) {
      var B = bend(DA, t), norm = 4 / g(4, t);
      var d = "";
      for (var s = 0; s <= 80; s++) { var x = 4 * s / 80; d += (s ? "L" : "M") + px(x).toFixed(2) + " " + py(g(x, t) * norm).toFixed(2); }
      curve.setAttribute("d", d);
      for (var i = 0; i < pairEls.length; i++) pairEls[i].setAttribute("cy", py(g(pairsA[i], t) * norm).toFixed(2));
      setBar("mknn", mknn(nA, knn(B, K), K));
      setBar("mst", mstOverlap(eA, mst(B)));
      setBar("order", pearson(rA, ranks(upper(B))));
      setBar("cka", cka(CA, B));
      var txt = "t = " + t.toFixed(2);
      out.textContent = txt; slider.setAttribute("aria-valuetext", txt);
    }

    var raf = 0, playing = false;
    function stop() { cancelAnimationFrame(raf); playing = false; }
    function play() {
      stop(); playing = true;
      var start = performance.now(), dur = 6000;
      (function frame(now) {
        var p = Math.min(1, (now - start) / dur);
        var e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
        var v = Math.round(1000 * e);
        slider.value = v; update(tFromSlider(v));
        if (p < 1 && playing) raf = requestAnimationFrame(frame); else playing = false;
      })(start);
    }
    slider.addEventListener("input", function () { stop(); update(tFromSlider(+slider.value)); });
    root.querySelector("#bend-replay").addEventListener("click", function () { if (REDUCED) { slider.value = 0; update(T_MAX); } else play(); });

    if (REDUCED) { slider.value = sliderFromT(0.3); update(tFromSlider(+slider.value)); }
    else {
      slider.value = 0; update(T_MAX);
      if ("IntersectionObserver" in window) {
        var io = new IntersectionObserver(function (entries) {
          entries.forEach(function (en) { if (en.isIntersecting) { io.disconnect(); setTimeout(play, 350); } });
        }, { threshold: 0.45 });
        io.observe(root);
      } else play();
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
