/* ARH card: size trends of CKA and mKNN before and after calibration along four language-model families,
   for the maximum over layers and for a fixed layer at mid depth (preprocessing of PRH). Values are the
   raw s_obs and calibrated s_cal columns of the appendix tables tab:app-layermax(-llama, -openllama, -pythia). */
(function () {
  "use strict";
  var V = window.LRHViz, root = document.getElementById("viz-cal");
  if (!V || !root) return;
  var FAM = [
    { name: "BLOOMZ", sizes: "5 sizes from 0.56B to 7.1B", kept: false, max: { cka: [0.89, -0.02], mknn: [1.00, 1.00] }, mid: { cka: [0.88, 0.70], mknn: [0.85, 0.84] } },
    { name: "LLaMA-1", sizes: "3 sizes from 7B to 30B", kept: false, max: { cka: [0.68, -0.85], mknn: [0.41, 0.38] }, mid: { cka: [1.00, 0.97], mknn: [0.82, 0.82] } },
    { name: "OpenLLaMA", sizes: "3 sizes from 3B to 13B", kept: false, max: { cka: [0.88, 0.18], mknn: [1.00, 1.00] }, mid: { cka: [0.68, 0.50], mknn: [0.68, 0.68] } },
    { name: "Pythia", sizes: "7 sizes from 70M to 6.9B", kept: true, max: { cka: [0.93, 0.89], mknn: [1.00, 1.00] }, mid: { cka: [0.98, 0.96], mknn: [0.99, 0.99] } }
  ];
  var PROTO = { max: "maximum over layers", mid: "fixed layer at mid depth" };
  var MET = { cka: "CKA", mknn: "mKNN" };
  var WHY = ["The largest random-pairing baseline grows at least as fast as raw CKA", "Raw CKA grows faster than the largest random-pairing baseline"];
  var chart = root.querySelector(".cal"), tip = document.createElement("div");
  tip.className = "cal-tip"; tip.setAttribute("aria-hidden", "true");
  var proto = "max", phase = "raw", timer = 0, lanes = [];

  function h(tag, cls, parent, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; if (parent) parent.appendChild(e); return e; }
  var rows = h("div", "cal-rows", chart);
  FAM.forEach(function (f) {
    var row = h("div", "cal-fam", rows), nm = h("div", "cal-name", row);
    h("b", "", nm, f.name); h("span", "", nm, f.sizes); h("em", "cal-why", nm, WHY[f.kept ? 1 : 0]);
    var box = h("div", "cal-lanes", row);
    ["cka", "mknn"].forEach(function (m) {
      var lane = h("div", "cal-lane m-" + m, box);
      lane.tabIndex = lanes.length ? -1 : 0; lane.setAttribute("role", "img");
      h("span", "cal-lab", lane, MET[m]);
      var plot = h("div", "cal-plot", lane);
      var L = { f: f, m: m, lane: lane, arrow: h("i", "cal-arrow", plot), ring: h("i", "cal-ring", plot), dot: h("i", "cal-dot", plot), val: h("span", "cal-val", plot), rawv: h("span", "cal-rawv", plot) };
      lane.addEventListener("mouseenter", function () { showTip(L); });
      lane.addEventListener("focus", function () { lanes.forEach(function (o) { o.lane.tabIndex = o === L ? 0 : -1; }); showTip(L); });
      lane.addEventListener("keydown", function (ev) {
        var n = lanes.indexOf(L), to = ev.key === "ArrowDown" || ev.key === "ArrowRight" ? n + 1 : ev.key === "ArrowUp" || ev.key === "ArrowLeft" ? n - 1 : ev.key === "Home" ? 0 : ev.key === "End" ? lanes.length - 1 : -2;
        if (to === -2) return;
        ev.preventDefault(); lanes[Math.max(0, Math.min(lanes.length - 1, to))].lane.focus();
      });
      lane.addEventListener("mouseleave", hideTip);
      lane.addEventListener("blur", hideTip);
      lanes.push(L);
    });
  });
  var axis = h("div", "cal-axis", chart); h("span", "", axis);
  var ticks = h("div", "cal-ticks", axis);
  [-1, -0.5, 0, 0.5, 1].forEach(function (v) { var t = h("span", "", ticks, V.fmt(v, v % 1 ? 1 : 0)); t.style.setProperty("--x", (v + 1) / 2); });
  var at = h("a", "term-link cal-title", axis, "Size trend"); at.href = "#term-trend";
  chart.appendChild(tip);

  function text(L) {
    var v = L.f[proto][L.m];
    return L.f.name + " (" + L.f.sizes + "), " + MET[L.m] + ", " + PROTO[proto] + ": " + V.fmt(v[0]) + " before and " + V.fmt(v[1]) + " after calibration";
  }
  function showTip(L) {
    var v = L.f[proto][L.m]; // the lane already names the sizes, so the tooltip leaves them out
    tip.textContent = L.f.name + ", " + MET[L.m] + ", " + PROTO[proto] + ": " + V.fmt(v[0]) + " before and " + V.fmt(v[1]) + " after calibration";
    var c = chart.getBoundingClientRect(), d = L.dot.getBoundingClientRect();
    tip.style.maxWidth = Math.min(400, c.width - 8) + "px";
    var x = d.left + d.width / 2 - c.left, w = tip.offsetWidth, below = d.top - c.top < tip.offsetHeight + 14;
    tip.classList.toggle("below", below);
    tip.style.top = (below ? d.bottom - c.top + 10 : d.top - c.top - 10) + "px";
    tip.style.left = Math.max(w / 2 + 4, Math.min(c.width - w / 2 - 4, x)) + "px";
    tip.classList.add("on");
  }
  function hideTip() { tip.classList.remove("on"); }

  function side(node, cls) { node.classList.remove("l", "r"); node.classList.add(cls); }
  function render(noAnim) {
    if (noAnim) chart.classList.add("snap");
    chart.classList.toggle("mid", proto === "mid");
    lanes.forEach(function (L) {
      var v = L.f[proto][L.m], r = v[0], c = phase === "cal" ? v[1] : r, xr = (r + 1) / 2, xc = (c + 1) / 2;
      var dv = Math.abs(v[1] - v[0]), moved = dv >= 0.05; // close pairs get one label and no arrow
      L.lane.style.setProperty("--xr", xr); L.lane.style.setProperty("--xc", xc);
      L.lane.style.setProperty("--a", Math.min(xr, xc)); L.lane.style.setProperty("--w", Math.abs(xc - xr));
      L.lane.classList.toggle("arrow", moved && phase === "cal");
      L.lane.classList.toggle("left", v[1] < v[0]);
      L.val.textContent = moved ? V.fmt(v[1]) : dv < 0.005 ? V.fmt(v[1]) + ", unchanged" : V.fmt(r) + " → " + V.fmt(v[1]);
      L.rawv.textContent = moved ? V.fmt(r) : "";
      var vs = moved ? (v[1] < v[0] ? "l" : "r") : (Math.max(xr, xc) > 0.8 ? "l" : "r");
      side(L.val, vs);
      L.lane.style.setProperty("--xl", moved || phase !== "cal" ? xc : (vs === "r" ? Math.max(xr, xc) : Math.min(xr, xc)));
      side(L.rawv, v[1] < v[0] ? "r" : "l");
      L.lane.setAttribute("aria-label", text(L));
    });
    if (noAnim) { void chart.offsetWidth; chart.classList.remove("snap"); }
  }
  function replay() {
    clearTimeout(timer); phase = "raw"; chart.classList.add("pre"); render(true);
    if (V.REDUCED) { phase = "cal"; chart.classList.remove("pre"); render(true); return; }
    timer = setTimeout(function () { phase = "cal"; chart.classList.remove("pre"); render(false); }, 500);
  }
  V.seg(root.querySelector(".seg"), function (b) {
    proto = b.getAttribute("data-proto");
    if (phase === "raw") { clearTimeout(timer); phase = "cal"; chart.classList.remove("pre"); }
    render(false); hideTip();
  });
  root.querySelector("[data-replay]").addEventListener("click", replay);

  if (V.REDUCED) { phase = "cal"; render(true); }
  else { chart.classList.add("pre"); render(true); V.inView(root, function () { setTimeout(replay, 250); }); }

  /* self-test: the values shown equal those of the table typed separately from the paper */
  V.test("ARH trends", "chart values equal the table of values (16 pairs)", function () {
    var bad = [], n = 0;
    root.querySelectorAll(".viz-table tbody tr").forEach(function (tr) {
      var c = tr.querySelectorAll("td"), f = FAM.filter(function (x) { return x.name === c[0].textContent.trim(); })[0];
      var p = /mid/.test(c[1].textContent) ? "mid" : "max";
      var got = [2, 3, 4, 5].map(function (i) { return +c[i].textContent.replace("−", "-"); });
      var want = f[p].cka.concat(f[p].mknn);
      got.forEach(function (g, i) { n++; if (Math.abs(g - want[i]) > 1e-9) bad.push(f.name + " " + p + " col " + i); });
    });
    var keep = proto; proto = "max"; var t1 = lanes.map(text).join("|"); proto = "mid"; var t2 = lanes.map(text).join("|"); proto = keep;
    ["0.89 before and −0.02", "0.68 before and −0.85", "0.88 before and 0.18", "0.93 before and 0.89"].forEach(function (s) { if (t1.indexOf(s) < 0) bad.push("tooltip " + s); });
    if (t2.indexOf("0.88 before and 0.70") < 0) bad.push("tooltip mid BLOOMZ");
    return { pass: n === 32 && !bad.length, detail: n + " values compared; " + (bad.join(", ") || "no mismatch") };
  });
  V.test("ARH trends", "LLaMA-1 CKA crosses zero; Pythia CKA moves by 0.04 or less", function () {
    var l = FAM[1].max.cka, p = FAM[3].max.cka;
    return l[0] > 0 && l[1] < 0 && Math.abs(p[1] - p[0]) <= 0.04 + 1e-9;
  });
})();
