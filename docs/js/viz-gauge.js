/* Theorem 2 card, illustration of parts 1, 3, 4 and Corollary 2: 48 evaluation samples z in the latent plane of a
   linear autoencoder. Run A has Omega = I and h_A = z. Under gradient flow, run B has Omega_B = R diag(f(s), f(-s)) R^T
   with f(l) = (l + sqrt(l^2 + 4))/2, i.e. an initial imbalance C_0 with eigenvalues s and -s, and h_B = Q Omega_B^(1/2) z.
   Under weight decay or tied weights, Omega_B = Omega_A and h_B = Q z. */
(function () {
  "use strict";
  var V = window.LRHViz, root = document.getElementById("viz-gauge");
  if (!V || !root) return;
  var M = 48, K = 3, r = V.rng(23), Z = [], q = 2 * Math.PI * r(), thr = V.rng(5);
  for (var n = 0; n < M; n++) Z.push([V.gauss(r), 0.7 * V.gauss(r)]);
  function f(l) { return (l + Math.sqrt(l * l + 4)) / 2; }
  function cond(s) { return f(s) / f(-s); }
  function runB(s, th) {
    var a = Math.sqrt(f(s)), b = Math.sqrt(f(-s)), c = Math.cos(th), si = Math.sin(th), cq = Math.cos(q), sq = Math.sin(q);
    var o11 = a * c * c + b * si * si, o12 = (a - b) * c * si, o22 = a * si * si + b * c * c; // Omega^(1/2)
    return Z.map(function (z) { var u = o11 * z[0] + o12 * z[1], v = o12 * z[0] + o22 * z[1]; return [cq * u - sq * v, sq * u + cq * v]; });
  }
  function sqd(H) { var D = []; for (var i = 0; i < M; i++) { D.push(new Float64Array(M)); for (var j = 0; j < M; j++) { var x = H[i][0] - H[j][0], y = H[i][1] - H[j][1]; D[i][j] = x * x + y * y; } } return D; }
  function order(D, i) { var idx = []; for (var j = 0; j < M; j++) if (j !== i) idx.push(j); return idx.sort(function (a, b) { return D[i][a] - D[i][b]; }); }
  function cka(HA, HB) {
    function cen(H) { var mx = 0, my = 0; H.forEach(function (h) { mx += h[0]; my += h[1]; }); mx /= M; my /= M; return H.map(function (h) { return [h[0] - mx, h[1] - my]; }); }
    function cross(X, Y) { var c = [0, 0, 0, 0]; for (var i = 0; i < M; i++) { c[0] += X[i][0] * Y[i][0]; c[1] += X[i][0] * Y[i][1]; c[2] += X[i][1] * Y[i][0]; c[3] += X[i][1] * Y[i][1]; } return c[0] * c[0] + c[1] * c[1] + c[2] * c[2] + c[3] * c[3]; }
    var A = cen(HA), B = cen(HB);
    return cross(A, B) / Math.sqrt(cross(A, A) * cross(B, B)); // linear CKA of the centered Gram matrices
  }
  function projector(H) { // onto the column space of the probe Gram matrix H H^T, i.e. H (H^T H)^-1 H^T
    var a = 0, b = 0, c = 0; H.forEach(function (h) { a += h[0] * h[0]; b += h[0] * h[1]; c += h[1] * h[1]; });
    var det = a * c - b * b, i11 = c / det, i12 = -b / det, i22 = a / det, P = [];
    for (var i = 0; i < M; i++) { P.push(new Float64Array(M)); for (var j = 0; j < M; j++) P[i][j] = H[i][0] * (i11 * H[j][0] + i12 * H[j][1]) + H[i][1] * (i12 * H[j][0] + i22 * H[j][1]); }
    return P;
  }
  function measure(s, th) {
    var HB = runB(s, th), DA = sqd(Z), DB = sqd(HB), cd = cond(s), same = [], cert = [], nA = [], nB = [];
    for (var i = 0; i < M; i++) {
      var oa = order(DA, i), ob = order(DB, i), a = oa.slice(0, K), b = ob.slice(0, K);
      nA.push(a); nB.push(b);
      same.push(a.every(function (j) { return b.indexOf(j) >= 0; }));
      cert.push(DA[i][oa[K]] / DA[i][oa[K - 1]] > cd);
    }
    var PA = projector(Z), PB = projector(HB), pd = 0;
    for (i = 0; i < M; i++) for (var j = 0; j < M; j++) pd += (PA[i][j] - PB[i][j]) * (PA[i][j] - PB[i][j]);
    var cnt = function (x) { return x.filter(Boolean).length / M; };
    return { HB: HB, cond: cd, same: same, cert: cert, nA: nA, nB: nB, pSame: cnt(same), pCert: cnt(cert), cka: cka(Z, HB), proj: Math.sqrt(pd) };
  }

  /* ---------- drawing ---------- */
  var svA = root.querySelector(".sv-a"), svB = root.querySelector(".sv-b"), slider = root.querySelector("input[type=range]"), out = root.querySelector("output");
  /* sample 14 changes its 3 nearest neighbors in run B during the opening sweep */
  var st = { proc: "gf", s: +slider.value / 100, se: +slider.value / 100, th: 0.6, sel: 14 }, stop = null, res = null, live = 0, touched = false;
  var selOut = root.querySelector('[data-read="sel"]');
  /* run A fills 90% of its panel; run B keeps this scale unless it would exceed 90% of the panel */
  function maxAbs(H) { var m = 0; H.forEach(function (h) { m = Math.max(m, Math.abs(h[0]), Math.abs(h[1])); }); return m; }
  var SCA = 108 / maxAbs(Z);
  function frame(H, sc) { return H.map(function (h) { return [120 + sc * h[0], 120 - sc * h[1]]; }); }
  function mark(g, p, i, cls) {
    var x = p[0].toFixed(1), y = p[1].toFixed(1);
    if (!res.same[i]) return V.el("path", { "class": "vg-pt chg" + cls, d: "M" + x + " " + (p[1] - 4.6).toFixed(1) + "l4.6 4.6-4.6 4.6-4.6-4.6z" }, g);
    return V.el("circle", { "class": "vg-pt " + (res.cert[i] ? "cert" : "same") + cls, cx: x, cy: y, r: res.cert[i] ? 3.4 : 3 }, g);
  }
  function panel(sv, H, nb, sc) {
    sv.textContent = "";
    var P = frame(H, sc), gP = V.el("g", {}, sv), gL = V.el("g", {}, sv), gH = V.el("g", {}, sv), p0 = P[st.sel], R0 = 6.5, R1 = 5.5;
    P.forEach(function (p, i) {
      mark(gP, p, i, i === st.sel ? " sel" : "");
      var hit = V.el("circle", { "class": "vg-hit", cx: p[0], cy: p[1], r: V.COARSE ? 15 : 9 }, gH);
      hit.addEventListener("click", function () { select(i); });
    });
    /* the lines to the 3 nearest neighbors run between the rings, above the points, with a halo */
    nb[st.sel].forEach(function (j) {
      var p = P[j], dx = p[0] - p0[0], dy = p[1] - p0[1], L = Math.sqrt(dx * dx + dy * dy);
      if (L > R0 + R1 + 1) {
        var a = { x1: (p0[0] + dx * R0 / L).toFixed(1), y1: (p0[1] + dy * R0 / L).toFixed(1), x2: (p[0] - dx * R1 / L).toFixed(1), y2: (p[1] - dy * R1 / L).toFixed(1) };
        V.el("line", Object.assign({ "class": "vg-halo" }, a), gL); V.el("line", Object.assign({ "class": "vg-nn" }, a), gL);
      }
      V.el("circle", { "class": "vg-ring nb", cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: R1 }, gL);
    });
    V.el("circle", { "class": "vg-ring", cx: p0[0].toFixed(1), cy: p0[1].toFixed(1), r: R0 }, gL);
  }
  function select(i) {
    touched = true; st.sel = i; draw();
    selOut.textContent = "Sample " + (i + 1) + " selected: run B " + (res.same[i] ? "keeps its 3 nearest neighbors, and the sample is " + (res.cert[i] ? "certified." : "not certified.") : "changes its 3 nearest neighbors.");
  }
  function bar(name, v, w) { var b = root.querySelector('[data-bar="' + name + '"]'); b.querySelector(".val").textContent = v.toFixed(2); b.style.setProperty("--v", w == null ? v : w); }
  function draw() {
    res = measure(st.se, st.th);
    panel(svA, Z, res.nA, SCA); panel(svB, res.HB, res.nB, Math.min(SCA, 108 / maxAbs(res.HB)));
    bar("same", res.pSame); root.querySelector('[data-read="cert"]').textContent = res.pCert.toFixed(2);
    root.querySelector('[data-bar="same"]').style.setProperty("--c2", res.pCert);
    bar("cka", res.cka);
    var ok = res.proj < 1e-9, chk = root.querySelector(".vg-check");
    chk.classList.toggle("ok", ok); chk.querySelector("b").textContent = ok ? "same in both runs" : "different";
    out.textContent = "cond = " + res.cond.toFixed(2);
    slider.setAttribute("aria-valuetext", "imbalance " + st.s.toFixed(2) + ", condition number " + res.cond.toFixed(2));
    var txt = "Condition number " + res.cond.toFixed(2) + ": " + Math.round(res.pSame * M) + " of 48 samples keep their 3 nearest neighbors, " + Math.round(res.pCert * M) + " of them certified, and CKA is " + res.cka.toFixed(2) + ".";
    V.say(svA, "Run A, the 48 samples with the 3 nearest neighbors of the selected sample. Click a sample or press an arrow key to select another sample. " + txt);
    V.say(svB, "Run B, the same samples under the gauge " + (st.proc === "wd" ? "fixed by weight decay or tied weights. " : "selected by its initialization. ") + txt);
  }
  function morph(to, dur) {
    if (stop) stop();
    var from = st.se;
    if (V.REDUCED || !dur) { st.se = to; draw(); return; }
    stop = V.tween(dur, function (p) { st.se = from + (to - from) * V.ease(p); draw(); });
  }
  [svA, svB].forEach(function (sv) {
    sv.addEventListener("keydown", function (ev) {
      var dlt = ev.key === "ArrowRight" || ev.key === "ArrowUp" ? 1 : ev.key === "ArrowLeft" || ev.key === "ArrowDown" ? M - 1 : 0;
      if (dlt) { ev.preventDefault(); select((st.sel + dlt) % M); }
    });
  });
  var setProc = V.seg(root.querySelector(".seg"), function (b) {
    st.proc = b.getAttribute("data-proc"); slider.disabled = st.proc === "wd";
    root.classList.toggle("is-wd", slider.disabled);
    morph(st.proc === "wd" ? 0 : st.s, 600);
  });
  root.addEventListener("pointerdown", function () { touched = true; clearTimeout(live); });
  slider.addEventListener("input", function () { touched = true; clearTimeout(live); if (stop) stop(); st.s = st.se = +slider.value / 100; draw(); });
  root.querySelector("[data-new]").addEventListener("click", function () { st.th = Math.PI * thr(); morph(st.se, 0); });
  root.querySelector("[data-replay]").addEventListener("click", replay);
  function replay() {
    if (st.proc === "wd") { setProc("gf", "data-proc"); st.proc = "gf"; slider.disabled = false; root.classList.remove("is-wd"); }
    if (V.REDUCED) { st.s = 1; slider.value = 100; morph(1, 0); return; }
    if (stop) stop();
    stop = V.tween(4000, function (p) { st.s = st.se = 1.5 * V.ease(p); slider.value = Math.round(100 * st.s); draw(); });
  }
  if (V.REDUCED) { st.s = st.se = 1; slider.value = 100; draw(); }
  else { st.se = 0; slider.value = 0; draw(); V.inView(root, function () { if (!touched) live = setTimeout(function () { if (!touched) replay(); }, 300); }); }

  /* ---------- self-test ---------- */
  V.test("Theorem 2 gauge", "certified ⊆ same on a grid of 8 directions and 8 imbalances", function () {
    var bad = 0;
    for (var a = 0; a < 8; a++) for (var b = 1; b <= 8; b++) { var m = measure(b / 4, a * Math.PI / 8); m.cert.forEach(function (c, i) { if (c && !m.same[i]) bad++; }); }
    return { pass: !bad, detail: bad + " certified samples with a changed neighbor set" };
  });
  V.test("Theorem 2 gauge", "weight decay: same = 1 and |1 − CKA| < 1e-12; gradient flow with s > 0: CKA < 1", function () {
    var wd = measure(0, 1.1), gf = [0.1, 0.5, 1, 2].every(function (s) { return measure(s, 1.1).cka < 1; });
    return { pass: wd.pSame === 1 && Math.abs(1 - wd.cka) < 1e-12 && gf, detail: "1 − CKA under weight decay = " + Math.abs(1 - wd.cka).toExponential(1) };
  });
  V.test("Theorem 2 gauge", "displayed cond equals ((s + √(s² + 4))/2)² to three decimals", function () {
    return [0, 0.5, 1, 1.5, 2].every(function (s) { var w = Math.pow((s + Math.sqrt(s * s + 4)) / 2, 2); return Math.abs(cond(s) - w) < 5e-4; }) && Math.abs(cond(1) - 2.618) < 5e-4;
  });
  V.test("Theorem 2 gauge", "projectors onto the column space agree to 1e-9 (Frobenius)", function () {
    var worst = 0; [0, 0.7, 1.4, 2].forEach(function (s) { worst = Math.max(worst, measure(s, 0.3).proj); });
    return { pass: worst < 1e-9, detail: worst.toExponential(1) };
  });
})();
