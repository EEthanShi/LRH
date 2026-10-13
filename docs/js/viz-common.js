/* Shared helpers for the four instruments of Sections 03 and 04: seeded random numbers, theme tokens,
   canvas setup, lazy start, one-time play on view, tweening, segmented controls, and the ?selftest runner. */
(function () {
  "use strict";
  var V = window.LRHViz = {};
  var mq = function (q) { return window.matchMedia ? window.matchMedia(q) : { matches: false }; };
  V.REDUCED = mq("(prefers-reduced-motion: reduce)").matches;
  V.COARSE = mq("(pointer: coarse)").matches;
  V.SELFTEST = /[?&]selftest\b/.test(location.search);
  V.NS = "http://www.w3.org/2000/svg";

  V.rng = function (a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; };
  V.gauss = function (r) { var u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  V.css = function (name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); };
  V.el = function (name, attrs, parent, text) {
    var e = document.createElementNS(V.NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  };
  V.fmt = function (x, d) { var s = Math.abs(x).toFixed(d == null ? 2 : d); return (x < 0 && +s !== 0 ? "−" : "") + s; };
  V.int = function (n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ","); };
  V.ease = function (p) { return p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2; };

  /* canvas at device resolution (capped at 2) */
  V.canvas = function (cv) {
    var w = cv.clientWidth || 260, h = cv.clientHeight || w, dpr = Math.min(2, window.devicePixelRatio || 1);
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    var ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  };

  /* theme changes: the toggle sets data-theme, the system preference can also change */
  var themeFns = [];
  V.onTheme = function (fn) { themeFns.push(fn); };
  function fire() { requestAnimationFrame(function () { themeFns.forEach(function (f) { f(); }); }); }
  if (window.MutationObserver) new MutationObserver(fire).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  var dmq = mq("(prefers-color-scheme: dark)");
  if (dmq.addEventListener) dmq.addEventListener("change", fire);

  /* start work when an element comes near the viewport; play once when it is in view */
  function io(el, fn, opts) {
    if (!("IntersectionObserver" in window)) { fn(); return; }
    var o = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting) { o.disconnect(); fn(); } }); }, opts);
    o.observe(el);
  }
  V.near = function (el, fn) { if (V.SELFTEST) fn(); else io(el, fn, { rootMargin: "600px 0px" }); };
  V.inView = function (el, fn) { io(el, fn, { threshold: 0.4 }); };
  V.onResize = function (el, fn) {
    if (!window.ResizeObserver) { window.addEventListener("resize", fn); return; }
    var w0 = 0; new ResizeObserver(function () { var w = el.clientWidth; if (w && w !== w0) { w0 = w; fn(); } }).observe(el);
  };

  /* a tween drives fn(p) with p from 0 to 1; returns a stop function */
  V.tween = function (dur, fn, done) {
    var t0 = performance.now(), raf = 0, live = true;
    (function step(now) {
      if (!live) return;
      var p = Math.min(1, (now - t0) / dur); fn(p);
      if (p < 1) raf = requestAnimationFrame(step); else { live = false; if (done) done(); }
    })(t0);
    return function () { live = false; cancelAnimationFrame(raf); };
  };

  /* segmented control: buttons with aria-pressed inside a group */
  V.seg = function (group, onPick) {
    var bs = Array.prototype.slice.call(group.querySelectorAll("button"));
    function set(b) { bs.forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); }); }
    bs.forEach(function (b) { b.addEventListener("click", function () { if (b.getAttribute("aria-pressed") === "true") return; set(b); onPick(b); }); });
    return function (val, attr) { bs.forEach(function (b) { if (b.getAttribute(attr) === val) set(b); }); };
  };

  /* polite live text, throttled */
  V.say = function (node, text) {
    clearTimeout(node._t);
    node._t = setTimeout(function () { node.setAttribute("aria-label", text); }, 300);
  };

  /* self-test: each check returns true/false or a promise of a {pass, detail} record */
  var tests = [];
  V.test = function (group, name, fn) { tests.push({ group: group, name: name, fn: fn }); };
  function runTests() {
    var rows = [];
    return tests.reduce(function (chain, t) {
      return chain.then(function () { return Promise.resolve().then(t.fn); }).then(function (r) {
        var pass = r && typeof r === "object" ? !!r.pass : !!r;
        rows.push({ instrument: t.group, check: t.name, result: pass ? "pass" : "FAIL", detail: r && r.detail ? r.detail : "" });
      }, function (e) { rows.push({ instrument: t.group, check: t.name, result: "FAIL", detail: String(e) }); });
    }, Promise.resolve()).then(function () {
      V.selftest = { rows: rows, failed: rows.filter(function (r) { return r.result !== "pass"; }).length };
      if (console.table) console.table(rows); else console.log(rows);
      console.log("LRH self-test: " + (rows.length - V.selftest.failed) + " of " + rows.length + " checks passed");
      return V.selftest;
    });
  }
  if (V.SELFTEST) window.addEventListener("load", function () { setTimeout(function () { V.selftestDone = runTests(); }, 50); });
})();
