/* BPC card, illustration of Theorem 1: 16,384 samples uniform in the unit square form model A, and model B adds
   independent Gaussian noise (sd 0.016 per coordinate). Nested evaluation sets are the first m samples. For a
   fixed k = 10 or k = m/100, the neighbor boundary at an anchor is its kth distance; the panels show the samples
   pushed out and pulled in by model B, and mKNN over the first 256 samples as anchors. */
(function () {
  "use strict";
  var V = window.LRHViz, root = document.getElementById("viz-cross");
  if (!V || !root) return;
  var N = 16384, NA = 256, SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384], SIG = 0.016;
  var xa = new Float64Array(N), ya = new Float64Array(N), xb = new Float64Array(N), yb = new Float64Array(N);
  (function () { var r = V.rng(7), i; for (i = 0; i < N; i++) { xa[i] = r(); ya[i] = r(); } for (i = 0; i < N; i++) { xb[i] = xa[i] + SIG * V.gauss(r); yb[i] = ya[i] + SIG * V.gauss(r); } })();
  function kOf(m, rule) { return rule === "prop" ? Math.round(m / 100) : 10; }
  function dists(i) {
    var a = new Float64Array(N), b = new Float64Array(N);
    for (var j = 0; j < N; j++) { var u = xa[j] - xa[i], v = ya[j] - ya[i], p = xb[j] - xb[i], q = yb[j] - yb[i]; a[j] = u * u + v * v; b[j] = p * p + q * q; }
    return { i: i, a: a, b: b };
  }
  var scrA = new Float64Array(N), scrB = new Float64Array(N);
  function sorted(d, m, scr) { scr.set(d.subarray(0, m)); return scr.subarray(0, m).sort(); } // index 0 is the anchor itself

  /* shared neighbors for every anchor, size and rule, computed in short chunks */
  var SH = new Uint16Array(NA * 7 * 2), done = 0, readyFns = [];
  /* the kth smallest of the first m values of a scratch array, by quickselect */
  function select(scr, m, k) {
    var lo = 0, hi = m - 1;
    while (hi > lo) {
      var piv = scr[(lo + hi) >> 1], i = lo, j = hi;
      while (i <= j) { while (scr[i] < piv) i++; while (scr[j] > piv) j--; if (i <= j) { var t = scr[i]; scr[i] = scr[j]; scr[j] = t; i++; j--; } }
      if (k <= j) hi = j; else if (k >= i) lo = i; else break;
    }
    return scr[k];
  }
  /* the k1th and k2th smallest of the first m distances (index 0 is the anchor itself) */
  function two(d, m, scr, k1, k2) {
    var lo = Math.min(k1, k2), hi = Math.max(k1, k2); scr.set(d.subarray(0, m));
    var vh = select(scr, m, hi), vl = lo === hi ? vh : select(scr, hi, lo);
    return k1 <= k2 ? [vl, vh] : [vh, vl];
  }
  function stats(d) {
    for (var mi = 0; mi < 7; mi++) {
      var m = SIZES[mi], k2 = kOf(m, "prop"), A = two(d.a, m, scrA, 10, k2), B = two(d.b, m, scrB, 10, k2);
      var a1 = A[0], b1 = B[0], a2 = A[1], b2 = B[1], s1 = 0, s2 = 0;
      for (var j = 0; j < m; j++) { if (j === d.i) continue; var u = d.a[j], v = d.b[j]; if (u <= a1 && v <= b1) s1++; if (u <= a2 && v <= b2) s2++; }
      SH[(d.i * 7 + mi) * 2] = s1; SH[(d.i * 7 + mi) * 2 + 1] = s2;
    }
  }
  /* the work starts at idle time after load and speeds up once the figure comes near the viewport */
  var started = false, urgent = false;
  function chunk(budget) {
    var t0 = performance.now();
    while (done < NA && performance.now() - t0 < budget) { stats(dists(done)); done++; }
    progress();
    if (done < NA) schedule(); else readyFns.forEach(function (f) { f(); });
  }
  function schedule() {
    if (urgent || !window.requestIdleCallback) setTimeout(function () { chunk(urgent ? 12 : 6); }, urgent ? 0 : 30);
    else requestIdleCallback(function (dl) { chunk(Math.max(3, Math.min(10, dl.timeRemaining()))); }, { timeout: 800 });
  }
  function precompute(now) {
    if (now) urgent = true;
    if (started) return; started = true; schedule();
  }
  var ready = new Promise(function (res) { readyFns.push(res); });
  function mknn(mi, rule) { var s = 0, k = kOf(SIZES[mi], rule), o = rule === "prop" ? 1 : 0; for (var i = 0; i < NA; i++) s += SH[(i * 7 + mi) * 2 + o]; return s / (NA * k); }

  /* ---------- state and controls ---------- */
  var cv1 = root.querySelector(".cv-near"), cv2 = root.querySelector(".cv-dist"), sv = root.querySelector(".sv-mknn");
  var slider = root.querySelector("input[type=range]"), out = root.querySelector("output");
  var rRep = root.querySelector("[data-read=rep]"), rMk = root.querySelector("[data-read=mknn]"), status = root.querySelector(".v-status");
  /* START is the anchor whose count of replaced neighbors at m = 4,096 and k = 10 equals the median over the
     256 anchors; the self-test recomputes this median */
  var START = 3, st = { i: START, mi: 4, rule: "fixed", from: 4, p: 1 }, cur = null, C = {}, stop = null, live = 0, touched = false;
  function colors() { ["--fg", "--fg-2", "--fg-3", "--line", "--line-strong", "--bg", "--surface", "--d-nb", "--d-alt"].forEach(function (n) { C[n] = V.css(n); }); }
  function progress() { if (status) status.textContent = done < NA ? "Computing mKNN over 256 anchors of 16,384 samples, " + Math.round(100 * done / NA) + "%" : ""; }

  /* classification of the current view */
  function view() {
    if (!cur || cur.i !== st.i) cur = dists(st.i);
    var m = SIZES[st.mi], k = kOf(m, st.rule), ta = sorted(cur.a, m, scrA)[k], tb = sorted(cur.b, m, scrB)[k];
    var pushed = 0, pulled = 0;
    for (var j = 0; j < m; j++) {
      if (j === st.i) continue;
      var ia = cur.a[j] <= ta, ib = cur.b[j] <= tb;
      if (ia && !ib) pushed++;
      if (!ia && ib) pulled++;
    }
    var hw = Math.max(2.3 * Math.sqrt(ta), 1.45 * Math.sqrt(tb));
    return { m: m, k: k, ta: ta, tb: tb, pushed: pushed, pulled: pulled, hw: hw };
  }
  var vPrev = null, vNow = null;

  function ring(ctx, x, y, r, col, lw) { ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.stroke(); }
  function dot(ctx, x, y, r, col) { ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fillStyle = col; ctx.fill(); }
  function diamond(ctx, x, y, r, col) { ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); ctx.fillStyle = col; ctx.fill(); }
  function label(ctx, t, x, y, align, col) { ctx.font = "500 11.5px Inter, system-ui, sans-serif"; ctx.textAlign = align; ctx.lineWidth = 3.5; ctx.strokeStyle = C["--bg"]; ctx.strokeText(t, x, y); ctx.fillStyle = col; ctx.fillText(t, x, y); }
  function cls(j, v) { if (j >= v.m || j === st.i) return 0; var ia = cur.a[j] <= v.ta, ib = cur.b[j] <= v.tb; return ia ? (ib ? 1 : 2) : (ib ? 3 : 0); } // 1 kept, 2 pushed out, 3 pulled in

  function draw() {
    var v = vNow, e = V.ease(st.p), hw = vPrev ? vPrev.hw + (v.hw - vPrev.hw) * e : v.hw;
    var mLo = Math.min(SIZES[st.from], v.m), mHi = Math.max(SIZES[st.from], v.m), grow = v.m >= SIZES[st.from];
    function alpha(j) { return j < mLo ? 1 : grow ? e : 1 - e; }
    var i = st.i;

    /* panel 1: model A near the anchor, with the shift of each sample in model B */
    var c = V.canvas(cv1), ctx = c.ctx, W = c.w, cx = W / 2, cy = c.h / 2, s = (W / 2 - 8) / hw;
    var lim = hw * 1.15, pts = [];
    for (var j = 0; j < mHi; j++) {
      if (j === i) continue;
      var dx = xa[j] - xa[i], dy = ya[j] - ya[i], k0 = cls(j, v);
      if ((dx > lim || dx < -lim || dy > lim || dy < -lim) && k0 !== 3) continue; // samples pulled in from outside keep their shift
      pts.push(j);
    }
    pts.forEach(function (j) {
      var k = cls(j, v), a = alpha(j);
      ctx.globalAlpha = a * (k ? 0.8 : 0.2);
      ctx.beginPath(); ctx.moveTo(cx + s * (xa[j] - xa[i]), cy - s * (ya[j] - ya[i])); ctx.lineTo(cx + s * (xb[j] - xb[i]), cy - s * (yb[j] - yb[i]));
      ctx.strokeStyle = k === 1 ? C["--d-nb"] : k ? C["--d-alt"] : C["--fg-3"]; ctx.lineWidth = k > 1 ? 1.3 : 1; ctx.stroke();
      if (k > 1) dot(ctx, cx + s * (xb[j] - xb[i]), cy - s * (yb[j] - yb[i]), 1.8, C["--d-alt"]);
    });
    ctx.globalAlpha = 1;
    ring(ctx, cx, cy, Math.sqrt(v.ta) * s, C["--fg"], 1.5);
    ctx.setLineDash([4, 3]); ring(ctx, cx, cy, Math.sqrt(v.tb) * s, C["--fg-2"], 1.5); ctx.setLineDash([]);
    pts.forEach(function (j) {
      var k = cls(j, v), x = cx + s * (xa[j] - xa[i]), y = cy - s * (ya[j] - ya[i]);
      ctx.globalAlpha = alpha(j);
      if (k === 1) dot(ctx, x, y, 3.6, C["--d-nb"]);
      else if (k === 2) { dot(ctx, x, y, 4, C["--bg"]); ring(ctx, x, y, 3.8, C["--d-alt"], 1.9); }
      else if (k === 3) diamond(ctx, x, y, 5, C["--d-alt"]);
      else dot(ctx, x, y, 1.9, C["--fg-3"]);
    });
    ctx.globalAlpha = 1;
    ring(ctx, cx, cy, 5.5, C["--fg"], 1.8); dot(ctx, cx, cy, 1.6, C["--fg"]);
    var L = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001].filter(function (l) { return l * s <= 80; })[0] || 0.001;
    var y0 = c.h - 12; ctx.beginPath(); ctx.moveTo(10, y0 - 4); ctx.lineTo(10, y0); ctx.lineTo(10 + L * s, y0); ctx.lineTo(10 + L * s, y0 - 4);
    ctx.strokeStyle = C["--fg-2"]; ctx.lineWidth = 1.3; ctx.stroke();
    label(ctx, String(L), 10 + L * s / 2, y0 - 7, "center", C["--fg-2"]);

    /* panel 2: squared distances from the anchor in both models, log axes fixed for every m */
    c = V.canvas(cv2); ctx = c.ctx; W = c.w;
    var H = c.h, x0 = 46, x1 = W - 8, y1 = 8, y2 = H - 34;
    function X(d) { return x0 + (x1 - x0) * (Math.max(-5, Math.min(0, Math.log10(d))) + 5) / 5; }
    function Y(d) { return y2 - (y2 - y1) * (Math.max(-5, Math.min(0, Math.log10(d))) + 5) / 5; }
    ctx.strokeStyle = C["--line"]; ctx.lineWidth = 1;
    for (var t = -5; t <= 0; t++) { var gx = X(Math.pow(10, t)), gy = Y(Math.pow(10, t)); ctx.beginPath(); ctx.moveTo(gx, y1); ctx.lineTo(gx, y2); ctx.moveTo(x0, gy); ctx.lineTo(x1, gy); ctx.stroke(); }
    ctx.setLineDash([3, 4]); ctx.strokeStyle = C["--fg-3"]; ctx.beginPath(); ctx.moveTo(X(1e-5), Y(1e-5)); ctx.lineTo(X(1), Y(1)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C["--fg-3"];
    [[0, mLo, 0.42], [mLo, mHi, 0.42 * (grow ? e : 1 - e)]].forEach(function (r) {
      ctx.globalAlpha = r[2]; ctx.beginPath();
      for (var j = r[0]; j < r[1]; j++) if (j !== i && !cls(j, v)) ctx.rect(X(cur.a[j]) - 0.8, Y(cur.b[j]) - 0.8, 1.6, 1.6);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
    for (j = 0; j < v.m; j++) {
      var k2 = cls(j, v); if (!k2) continue;
      var px = X(cur.a[j]), py = Y(cur.b[j]);
      if (k2 === 1) dot(ctx, px, py, 2.4, C["--d-nb"]); else if (k2 === 2) ring(ctx, px, py, 3.4, C["--d-alt"], 1.7); else diamond(ctx, px, py, 4.2, C["--d-alt"]);
    }
    ctx.strokeStyle = C["--fg"]; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(v.ta), y1); ctx.lineTo(X(v.ta), y2); ctx.stroke();
    ctx.setLineDash([5, 3]); ctx.beginPath(); ctx.moveTo(x0, Y(v.tb)); ctx.lineTo(x1, Y(v.tb)); ctx.stroke(); ctx.setLineDash([]);
    /* each count names its quadrant: pushed out lies left of the boundary in A and above the boundary in B,
       pulled in lies right of the boundary in A and below the boundary in B */
    ctx.font = "500 11.5px Inter, system-ui, sans-serif";
    var tP = "pushed out " + v.pushed, tQ = "pulled in " + v.pulled, wP = ctx.measureText(tP).width, wQ = ctx.measureText(tQ).width;
    var xa0 = X(v.ta), yb0 = Y(v.tb);
    if (xa0 - x0 >= wP + 12) label(ctx, tP, xa0 - 6, y1 + 15, "right", C["--fg"]);
    else if (x1 - xa0 >= wP + 12) label(ctx, tP, xa0 + 6, y1 + 15, "left", C["--fg"]);
    else label(ctx, tP, x1 - 4, y1 + 15, "right", C["--fg"]); // narrow panel: the halo keeps the label legible over the boundary
    var yQ = y2 - yb0 >= 24 ? y2 - 7 : yb0 - 7;
    if (x1 - xa0 >= wQ + 12) label(ctx, tQ, x1 - 5, yQ, "right", C["--fg"]);
    else label(ctx, tQ, xa0 - 6, yQ, "right", C["--fg"]);
    ["10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "1"].forEach(function (lab, n) {
      if (n % 2 && n !== 5) return;
      label(ctx, lab, X(Math.pow(10, n - 5)), y2 + 14, "center", C["--fg-3"]);
      label(ctx, lab, x0 - 5, Y(Math.pow(10, n - 5)) + 4, "right", C["--fg-3"]);
    });
    label(ctx, W < 240 ? "model A" : "squared distance in model A", (x0 + x1) / 2, H - 4, "center", C["--fg-2"]);
    ctx.save(); ctx.translate(11, (y1 + y2) / 2); ctx.rotate(-Math.PI / 2); label(ctx, W < 240 ? "model B" : "squared distance in model B", 0, 0, "center", C["--fg-2"]); ctx.restore();

    /* readouts */
    var mk = done >= NA ? mknn(st.mi, st.rule) : NaN;
    rRep.textContent = v.pushed + " of " + v.k;
    rMk.textContent = isNaN(mk) ? "…" : mk.toFixed(2);
    var txt = "At m = " + V.int(v.m) + " and k = " + v.k + ", model B replaces " + v.pushed + " of the " + v.k + " neighbors of this anchor" + (isNaN(mk) ? "." : "; mKNN over 256 anchors equals " + mk.toFixed(2) + ".");
    V.say(cv1, "Model A near one anchor. " + txt); V.say(cv2, "Squared distances from the anchor in model A against model B. " + txt);
    curve();
  }

  /* panel 3: mKNN against m for both rules, with chance levels */
  var PX0 = 46, PX1 = 270, PY0 = 252, PY1 = 14, LO = Math.log10(0.0005);
  function SX(mi) { return PX0 + (PX1 - PX0) * mi / 6; }
  function SY(v) { return PY0 - (PY0 - PY1) * (Math.log10(Math.max(0.0005, v)) - LO) / -LO; }
  var g = {};
  (function axes() {
    [0.001, 0.01, 0.1, 1].forEach(function (v) {
      V.el("line", { "class": "vx-grid", x1: PX0, x2: PX1, y1: SY(v), y2: SY(v) }, sv);
      V.el("text", { "class": "vx-tick", x: PX0 - 6, y: SY(v) + 4, "text-anchor": "end" }, sv, String(v));
    });
    [0, 2, 4, 6].forEach(function (mi) { V.el("text", { "class": "vx-tick", x: SX(mi), y: PY0 + 16, "text-anchor": "middle" }, sv, V.int(SIZES[mi])); });
    V.el("line", { "class": "vx-axis", x1: PX0, x2: PX1, y1: PY0, y2: PY0 }, sv);
    V.el("text", { "class": "vx-lab", x: (PX0 + PX1) / 2, y: 290, "text-anchor": "middle" }, sv, "evaluation size m");
    ["fixed", "prop"].forEach(function (r) {
      g["c" + r] = V.el("path", { "class": "vx-chance" }, sv);
      g["l" + r] = V.el("path", { "class": "vx-line" + (r === "prop" ? " dash" : "") }, sv);
      g["t" + r] = V.el("text", { "class": "vx-name", x: PX1 + 8 }, sv, r === "prop" ? "k = m/100" : "k = 10");
      g["h" + r] = V.el("text", { "class": "vx-tick", x: PX1 + 8 }, sv);
      V.el("tspan", { x: PX1 + 8, dy: "-0.55em" }, g["h" + r], "chance,");
      V.el("tspan", { x: PX1 + 8, dy: "1.15em" }, g["h" + r], r === "prop" ? "k = m/100" : "k = 10");
    });
    g.ring = V.el("circle", { "class": "vx-now", r: 5 }, sv);
    g.ring.style.visibility = g.tfixed.style.visibility = g.tprop.style.visibility = "hidden";
  })();
  function curve() {
    ["fixed", "prop"].forEach(function (r) {
      var on = r === st.rule, ch = "", ln = "";
      SIZES.forEach(function (m, mi) { ch += (mi ? "L" : "M") + SX(mi) + " " + SY(kOf(m, r) / (m - 1)).toFixed(1); });
      g["c" + r].setAttribute("d", ch); g["c" + r].style.opacity = on ? 1 : 0.4;
      g["h" + r].setAttribute("y", SY(kOf(16384, r) / 16383) + 4);
      if (done >= NA) {
        SIZES.forEach(function (m, mi) { ln += (mi ? "L" : "M") + SX(mi) + " " + SY(mknn(mi, r)).toFixed(1); });
        g["l" + r].setAttribute("d", ln); g["t" + r].setAttribute("y", SY(mknn(6, r)) + 4);
      }
      g["l" + r].style.opacity = on ? 1 : 0.4;
      g["t" + r].style.fontWeight = on ? 600 : 500;
    });
    if (done >= NA) { g.ring.setAttribute("cx", SX(st.mi)); g.ring.setAttribute("cy", SY(mknn(st.mi, st.rule))); }
    g.ring.style.visibility = g.tfixed.style.visibility = g.tprop.style.visibility = done >= NA ? "" : "hidden";
    sv.setAttribute("aria-label", "mKNN over 256 anchors against the evaluation size m for k = 10 and k = m/100, with the chance level k/(m − 1) of each rule.");
  }

  function setM(mi, animate) {
    if (stop) { stop(); stop = null; }
    vPrev = vNow; st.from = vNow ? st.mi : mi; st.mi = mi; vNow = view();
    slider.value = mi; out.textContent = "m = " + V.int(SIZES[mi]); slider.setAttribute("aria-valuetext", out.textContent);
    if (animate && !V.REDUCED) stop = V.tween(450, function (p) { st.p = p; draw(); });
    else { st.p = 1; vPrev = null; st.from = mi; draw(); }
  }
  function replay() {
    var mi = 0; setM(0, false);
    if (V.REDUCED) { setM(4, false); return; }
    (function next() { live = setTimeout(function () { if (++mi > 6) return; setM(mi, true); next(); }, 700); })();
  }
  function halt() { clearTimeout(live); }
  /* once the reader acts on the figure, the first autoplay is skipped */
  function touch() { touched = true; halt(); }
  root.addEventListener("pointerdown", touch);
  root.addEventListener("keydown", function (ev) { if (ev.key !== "Tab" && ev.key !== "Shift") touch(); });
  slider.addEventListener("input", function () { touch(); setM(+slider.value, false); });
  var setRule = V.seg(root.querySelector(".seg"), function (b) { st.rule = b.getAttribute("data-rule"); setM(st.mi, false); });
  root.querySelector("[data-anchor]").addEventListener("click", function () { st.i = (st.i + 1) % NA; setM(st.mi, false); });
  root.querySelector("[data-replay]").addEventListener("click", function () { halt(); replay(); });
  V.onResize(cv1, function () { if (vNow) draw(); });
  V.onTheme(function () { colors(); if (vNow) draw(); });

  /* panels 1 and 2 need one anchor only and appear at once; panel 3 fills in when all 256 anchors are done */
  function first() {
    if (vNow) return;
    colors(); if (done < NA) root.setAttribute("aria-busy", "true");
    setM(V.REDUCED ? 4 : 0, false);
  }
  V.near(root, function () { precompute(true); first(); });
  window.addEventListener("load", function () { setTimeout(function () { precompute(false); }, 600); });
  ready.then(function () {
    root.removeAttribute("aria-busy");
    if (vNow) draw();
    if (!V.REDUCED) V.inView(root, function () { if (!touched) live = setTimeout(function () { if (!touched) replay(); }, 300); });
  });
  colors();

  /* ---------- self-test ---------- */
  function theorem() {
    var bad = 0, n = 0, sand = 0;
    for (var i = 0; i < NA; i++) {
      var d = dists(i);
      for (var mi = 0; mi < 7; mi++) ["fixed", "prop"].forEach(function (r) {
        var m = SIZES[mi], k = kOf(m, r), ta = sorted(d.a, m, scrA)[k], tb = sorted(d.b, m, scrB)[k];
        var X = 0, Y = 0, Xa = 0, Ya = 0, rep = k - SH[(i * 7 + mi) * 2 + (r === "prop" ? 1 : 0)];
        for (var j = 0; j < m; j++) {
          if (j === i) continue;
          var ia = d.a[j] <= ta;
          if (ia && d.b[j] > tb) X++; if (!ia && d.b[j] <= tb) Y++;
          if (ia && d.b[j] > ta) Xa++; if (!ia && d.b[j] <= ta) Ya++;
        }
        n++; if (X !== rep || Y !== rep) bad++;
        if (rep < Math.min(Xa, Ya) || rep > Math.max(Xa, Ya)) sand++;
      });
    }
    return { pass: !bad && !sand, detail: n + " anchor-size-rule cases; " + bad + " with pushed out ≠ pulled in ≠ k − shared; " + sand + " outside [X, Y] at the boundary of model A" };
  }
  V.test("BPC crossings", "Theorem 1: pushed out = pulled in = k − shared, and the bounds X, Y hold", function () { return ready.then(theorem); });
  V.test("BPC crossings", "the starting anchor has the median count of replaced neighbors at m = 4,096 and k = 10", function () {
    return ready.then(function () {
      var rep = []; for (var i = 0; i < NA; i++) rep.push(10 - SH[(i * 7 + 4) * 2]);
      var med = rep.slice().sort(function (a, b) { return a - b; })[NA >> 1];
      return { pass: rep.indexOf(med) === START, detail: "median " + med + " at anchor " + rep.indexOf(med) };
    });
  });
  V.test("BPC crossings", "k = 10 falls with m, ending below one third of its start", function () {
    return ready.then(function () {
      var v = SIZES.map(function (m, mi) { return mknn(mi, "fixed"); }), mono = v.every(function (x, n) { return !n || x < v[n - 1]; });
      return { pass: mono && v[6] <= v[0] / 3, detail: v.map(function (x) { return x.toFixed(3); }).join(" ") };
    });
  });
  V.test("BPC crossings", "k = m/100 stays within 0.05 of its value at 1,024 for m ≥ 512, and both rules agree at 1,024", function () {
    return ready.then(function () {
      var v = SIZES.map(function (m, mi) { return mknn(mi, "prop"); }), ok = v.slice(1).every(function (x) { return Math.abs(x - v[2]) <= 0.05; });
      return { pass: ok && Math.abs(v[2] - mknn(2, "fixed")) < 1e-12, detail: v.map(function (x) { return x.toFixed(3); }).join(" ") };
    });
  });
  V.test("BPC crossings", "every chance level lies below both curves", function () {
    return ready.then(function () {
      return SIZES.every(function (m, mi) { return ["fixed", "prop"].every(function (r) { var c = kOf(m, r) / (m - 1); return c < mknn(mi, "fixed") && c < mknn(mi, "prop"); }); });
    });
  });
  V.test("BPC crossings", "one redraw at m = 16,384 takes under 16 ms (median of 5)", function () {
    return ready.then(function () {
      var keep = st.mi, ts = []; for (var n = 0; n < 5; n++) { var t0 = performance.now(); setM(6, false); ts.push(performance.now() - t0); }
      setM(keep, false); ts.sort(function (a, b) { return a - b; });
      return { pass: ts[2] < 16, detail: ts[2].toFixed(1) + " ms" };
    });
  });
})();
