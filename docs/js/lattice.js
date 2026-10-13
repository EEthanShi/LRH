/* Interactive lattice of observables (hero) and its static copy in the PRH card.
   Geometry and study marks follow Figure 1 and Proposition 1 of the paper: the only open circle is the
   criterion behind the claim of PRH, and filled circles are the criteria applied by each study.
   Arrows run from a finer observable to a coarser observable. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var W = 600, H = 570, BW = 150, BH = 46;

  // sym: [base, subscript, optional sub-subscript]; tip may hold <i> for variables
  var NODES = [
    { id: "G",   label: "All distances",    sym: ["T", "G"],        x: 300, y: 46,  tip: "All squared distances between the samples." },
    { id: "GR",  label: "CKA",              sym: ["T", "G/ℝ", "+"], x: 300, y: 112, tip: "The distances up to a common rescaling, which determine linear CKA." },
    { id: "GA",  label: "Pearson / RSA",    sym: ["T", "G/Aff"],    x: 300, y: 178, tip: "The distances up to a positive affine map, which determine the Pearson correlation of two distance matrices." },
    { id: "E",   label: "Global order",     sym: ["T", "E"],        x: 300, y: 244, tip: "The order of all pairwise distances, which determines their Spearman correlation." },
    { id: "O",   label: "Rankings",         sym: ["T", "O"],        x: 170, y: 334, tip: "For each sample, the ranking of the other samples by their distance to this sample." },
    { id: "SL",  label: "Single linkage",   sym: ["T", "SL"],       x: 430, y: 334, tip: "The single-linkage dendrogram, i.e., the MST edges together with their order of addition." },
    { id: "NK",  label: "Neighbors (mKNN)", sym: ["T", "N,k"],      x: 130, y: 434, tip: "For each sample, the set of its <i>k</i> nearest neighbors." },
    { id: "N1",  label: "1-NN",             sym: ["T", "N,1"],      x: 300, y: 434, tip: "The map sending each sample to its nearest neighbor." },
    { id: "MST", label: "Spanning tree",    sym: ["T", "MST"],      x: 470, y: 434, tip: "The edge set of the minimum spanning tree (MST)." },
    { id: "CK",  label: "Components",       sym: ["T", "C,k"],      x: 130, y: 534, tip: "The connected components of the <i>k</i>-nearest-neighbor graph." },
    { id: "TAU", label: "MST + lengths",    sym: ["τ"],             x: 470, y: 534, tip: "The length-weighted spanning-tree overlap of SOG, in which a mismatch between the normalized lengths of a shared MST edge in the two models lowers the weight of this edge, with length tolerance τ.", dashed: true }
  ];
  // [from, to, startDx, endDx, dashed]
  var EDGES = [
    ["G", "GR", 0, 0], ["GR", "GA", 0, 0], ["GA", "E", 0, 0],
    ["E", "O", -32, 12], ["E", "SL", 32, -12],
    ["O", "NK", -22, 8], ["O", "N1", 26, -22],
    ["SL", "N1", -26, 22], ["SL", "MST", 22, -8],
    ["NK", "CK", 0, 0],
    ["TAU", "MST", 0, 0, true]
  ];
  // arrow-key moves between boxes, level by level
  var NAV = {
    G:   { ArrowDown: "GR" },
    GR:  { ArrowUp: "G", ArrowDown: "GA" },
    GA:  { ArrowUp: "GR", ArrowDown: "E" },
    E:   { ArrowUp: "GA", ArrowDown: "O", ArrowLeft: "O", ArrowRight: "SL" },
    O:   { ArrowUp: "E", ArrowDown: "NK", ArrowRight: "SL" },
    SL:  { ArrowUp: "E", ArrowDown: "MST", ArrowLeft: "O" },
    NK:  { ArrowUp: "O", ArrowDown: "CK", ArrowRight: "N1" },
    N1:  { ArrowUp: "O", ArrowLeft: "NK", ArrowRight: "MST" },
    MST: { ArrowUp: "SL", ArrowDown: "TAU", ArrowLeft: "N1" },
    CK:  { ArrowUp: "NK", ArrowRight: "TAU" },
    TAU: { ArrowUp: "MST", ArrowLeft: "CK" }
  };
  var DOTS = { GR: ["ARH"], NK: ["PRH", "ARH", "BPC", "SOG"], MST: ["SOG"], TAU: ["SOG"] };
  var STUDIES = ["PRH", "ARH", "BPC", "SOG"];

  var STATES = {
    All: {
      dots: { PRH: ["NK"], ARH: ["GR", "NK"], BPC: ["NK"], SOG: ["NK", "MST", "TAU"] },
      rings: { PRH: ["G"] }, gap: true,
      caption: "Each solid box is an observable, i.e., one relation among the evaluation samples, and each solid arrow runs from a finer observable to a coarser observable. The dashed box marks the length-weighted spanning-tree overlap of SOG.",
      spec: null
    },
    PRH: {
      dots: { PRH: ["NK"] }, rings: { PRH: ["G"] }, gap: true,
      caption: "Under the LRH, the convergence claimed by the PRH is agreement at the top of the lattice, whereas its evidence comes from neighbor overlap, a criterion near the bottom.",
      spec: ["10", "1,024", "Max over all pairs", "None"]
    },
    ARH: {
      dots: { ARH: ["GR", "NK"] }, rings: {}, tags: ["arh"],
      caption: "ARH restricts the claimed convergence to local neighborhoods, and the loss of the CKA trend behind this restriction can come from calibrating a maximum over layers.",
      spec: ["10–100", "1,024", "Max over all pairs", "Permutation"]
    },
    BPC: {
      dots: { BPC: ["NK"] }, rings: {}, tags: ["bpc"],
      caption: "BPC infers coarse shared structure from the collapse of neighbor overlap on larger evaluation sets, whereas we trace this collapse at least partly to a fixed number of neighbors covering shorter distances.",
      spec: ["1 or 10; <i>m</i>/100", "1,024 to 1.5×10⁷", "Fixed pair", "None"]
    },
    SOG: {
      dots: { SOG: ["NK", "MST", "TAU"] }, rings: {}, glow: true,
      caption: "On the studied vision–language pairs, about 88% of the shared spanning-tree edges join a sample to its nearest neighbor, so the spanning-tree overlap of SOG mostly repeats one-nearest-neighbor overlap. The number of other shared edges exceeds its chance level without an established size trend.",
      spec: ["10", "1,024", "Max over all pairs", "Permutation; τ"]
    }
  };
  var SPEC_KEYS = [
    { html: "<i>k</i>", aria: "k", v: true },
    { html: "<i>m</i>", aria: "m", v: true },
    { html: "Layer pairs" },
    { html: "Calibration / tolerance" }
  ];
  var REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var NARROW = window.matchMedia ? window.matchMedia("(max-width: 480px)") : null;
  var allSvgs = [];
  function setViewBox(svg) {
    if (allSvgs.indexOf(svg) < 0) allSvgs.push(svg);
    svg.setAttribute("viewBox", NARROW && NARROW.matches ? "24 0 528 " + H : "0 0 " + W + " " + H);
  }

  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function plain(html) { return html.replace(/<[^>]+>/g, ""); }
  function byId(id) { for (var i = 0; i < NODES.length; i++) if (NODES[i].id === id) return NODES[i]; }
  function arrowHead(x, y, ang, parent, size) {
    var s = size || 7, a = 0.42;
    var p1 = [x - s * Math.cos(ang - a), y - s * Math.sin(ang - a)];
    var p2 = [x - s * Math.cos(ang + a), y - s * Math.sin(ang + a)];
    return el("polygon", { points: x + "," + y + " " + p1.join(",") + " " + p2.join(",") }, parent);
  }
  function symText(parent, n) {
    var t = el("text", { x: n.x, y: n.y + 15, "text-anchor": "middle", "class": "n-sym" }, parent);
    var base = el("tspan", {}, t); base.textContent = n.sym[0];
    if (n.sym[1]) {
      var sub = el("tspan", { "class": "sub", dy: "3.5" }, t); sub.textContent = n.sym[1];
      if (n.sym[2]) { var ss = el("tspan", { "class": "sub sub2", dy: "3" }, t); ss.textContent = n.sym[2]; }
    }
    return t;
  }
  // tag line given as [[text, italic], ...]
  function tagLine(t, parts) {
    parts.forEach(function (p) {
      var ts = el("tspan", p[1] ? { "class": "var" } : {}, t);
      ts.textContent = p[0];
    });
  }

  function build(container, opts) {
    opts = opts || {};
    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H, "class": "lattice-svg lat", role: "group",
      "aria-label": opts.label || "Lattice of observables, from all distances at the top to components and spanning-tree lengths at the bottom"
    });
    var under = el("g", { "class": "under" }, svg);

    // axis cue (right edge): stronger at the top, weaker at the bottom
    var ax = el("g", { "class": "axis", "aria-hidden": "true" }, svg);
    el("line", { x1: 586, y1: 100, x2: 586, y2: 480 }, ax);
    arrowHead(586, 488, Math.PI / 2, ax, 7);
    var t1 = el("text", { x: 586, y: 30, transform: "rotate(90 586 30)", dy: "-3.5" }, ax); t1.textContent = "STRONGER";
    var t2 = el("text", { x: 586, y: 498, transform: "rotate(90 586 498)", dy: "-3.5" }, ax); t2.textContent = "WEAKER";

    // SOG glow on the 1-NN box only (under nodes and edges)
    var n1 = byId("N1");
    el("rect", { "class": "glow-rect", x: n1.x - BW / 2, y: n1.y - BH / 2, width: BW, height: BH, rx: 10, "data-study": "SOG" }, under);

    // edges
    var edgeEls = [];
    var eg = el("g", { "class": "edges", "aria-hidden": "true" }, svg);
    EDGES.forEach(function (e) {
      var a = byId(e[0]), b = byId(e[1]);
      var g = el("g", { "class": "edge" + (e[4] ? " dashed" : "") }, eg);
      var x1, y1, x2, y2;
      if (e[0] === "TAU") { x1 = a.x; y1 = a.y - BH / 2; x2 = b.x; y2 = b.y + BH / 2 + 1; }
      else { x1 = a.x + e[2]; y1 = a.y + BH / 2; x2 = b.x + e[3]; y2 = b.y - BH / 2 - 1; }
      var ang = Math.atan2(y2 - y1, x2 - x1);
      el("line", { x1: x1, y1: y1, x2: x2 - 5 * Math.cos(ang), y2: y2 - 5 * Math.sin(ang) }, g);
      arrowHead(x2, y2, ang, g, 7.5);
      edgeEls.push({ from: e[0], to: e[1], g: g });
    });

    // incomparable arc between neighbor sets and spanning trees
    var ic = el("g", { "class": "incomp", "aria-hidden": "true" }, svg);
    el("path", { d: "M 186 459 Q 300 517 414 459" }, ic);
    var ict = el("text", { x: 300, y: 505, "text-anchor": "middle" }, ic); ict.textContent = "incomparable";

    // PRH claim / evidence gap
    var gap = el("g", { "class": "gap", "data-study": "PRH", "aria-hidden": "true" }, svg);
    var gN = byId("G"), nk = byId("NK");
    el("path", { d: "M " + (gN.x - BW / 2) + " " + gN.y + " H 30 V " + nk.y + " H " + (nk.x - BW / 2 - 7) }, gap);
    arrowHead(nk.x - BW / 2 - 1, nk.y, 0, gap, 7.5);
    var gt = el("text", { x: 22, y: 240, "text-anchor": "middle", transform: "rotate(-90 22 240)" }, gap); gt.textContent = "claim / evidence gap";

    // nodes
    var nodeEls = {};
    var ng = el("g", { "class": "nodes" }, svg);
    NODES.forEach(function (n, i) {
      var g = el("g", {
        "class": "node", "data-id": n.id, role: "img",
        "aria-label": n.label + ". " + plain(n.tip)
      }, ng);
      if (opts.interactive) g.setAttribute("tabindex", i === 0 ? "0" : "-1");
      el("rect", { "class": "n-rect" + (n.dashed ? " dashed" : ""), x: n.x - BW / 2, y: n.y - BH / 2, width: BW, height: BH, rx: 10 }, g);
      var lt = el("text", { "class": "n-label", x: n.x, y: n.y - 2, "text-anchor": "middle" }, g); lt.textContent = n.label;
      symText(g, n);
      nodeEls[n.id] = g;
    });

    // tags (ARH, BPC); widths are measured once the svg is in the document
    var tags = {}, tagSpecs = [];
    function tag(key, study, x, y, lines, alignRight) {
      var g = el("g", { "class": "tag", "data-study": study, "aria-hidden": "true" }, svg);
      var rect = el("rect", { x: x, y: y, width: 10, height: 10, rx: 7 }, g);
      var texts = lines.map(function (l) { var t = el("text", { x: x + 8, y: y }, g); tagLine(t, l); return t; });
      g.style.opacity = 0; tags[key] = g;
      tagSpecs.push({ rect: rect, texts: texts, x: x, y: y, lines: lines, alignRight: alignRight });
    }
    var gr = byId("GR");
    // on phones the tags are hidden: the scale-and-protocol row under the lattice carries the same values
    tag("arh", "ARH", gr.x - BW / 2 - 10, gr.y - 11.5, [[["max over layers → calibration", false]]], true);
    tag("bpc", "BPC", 8, 365, [[["k", true], [" fixed, ", false], ["m", true], [" grows", false]], [["k", true], [" = ", false], ["m", true], ["/100", false]]]);
    function sizeTags() {
      tagSpecs.forEach(function (t) {
        var w = 0;
        t.texts.forEach(function (tx, i) {
          var len = 0;
          try { len = tx.getComputedTextLength(); } catch (e) {}
          w = Math.max(w, (len || 16 * 6.1) + 16);
          tx.setAttribute("y", t.y + 16 + 15 * i);
        });
        var x0 = t.alignRight ? t.x - w : t.x;
        t.rect.setAttribute("x", x0); t.rect.setAttribute("width", w);
        t.rect.setAttribute("height", 8 + 15 * t.texts.length);
        t.texts.forEach(function (tx) { tx.setAttribute("x", x0 + 8); });
      });
    }

    // study marks: dots on the bottom border, the PRH ring on the top-right corner
    var dotEls = [], ringEls = [];
    var mg = el("g", { "class": "marks", "aria-hidden": "true" }, svg);
    Object.keys(DOTS).forEach(function (id) {
      var n = byId(id);
      DOTS[id].forEach(function (s, j) {
        var d = el("circle", { "class": "dot", "data-study": s, "data-node": id, cx: n.x - BW / 2 + 18 + 15 * j, cy: n.y + BH / 2, r: 6 }, mg);
        dotEls.push(d);
      });
    });
    var gRing = byId("G");
    ringEls.push(el("circle", { "class": "ring", "data-study": "PRH", "data-node": "G", cx: gRing.x + BW / 2 - 16, cy: gRing.y - BH / 2, r: 6.5 }, mg));

    container.appendChild(svg);
    setViewBox(svg);
    sizeTags();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(sizeTags);
    return { svg: svg, nodes: nodeEls, edges: edgeEls, dots: dotEls, rings: ringEls, tags: tags, gap: gap, sizeTags: sizeTags };
  }

  if (NARROW) {
    var onNarrow = function () { allSvgs.forEach(setViewBox); built.forEach(function (L) { L.sizeTags(); }); };
    if (NARROW.addEventListener) NARROW.addEventListener("change", onNarrow); else if (NARROW.addListener) NARROW.addListener(onNarrow);
  }
  var built = [];

  function apply(L, name) {
    var st = STATES[name];
    L.svg.setAttribute("class", "lattice-svg lat state-" + name);
    L.dots.forEach(function (d) {
      var s = d.getAttribute("data-study"), id = d.getAttribute("data-node");
      var on = st.dots[s] && st.dots[s].indexOf(id) >= 0;
      d.style.opacity = on ? 1 : 0;
      d.style.transform = "";
    });
    L.rings.forEach(function (r) {
      var s = r.getAttribute("data-study"), id = r.getAttribute("data-node");
      r.style.opacity = st.rings[s] && st.rings[s].indexOf(id) >= 0 ? 1 : 0;
    });
    L.gap.style.opacity = st.gap ? 1 : 0;
    Object.keys(L.tags).forEach(function (k) { L.tags[k].style.opacity = st.tags && st.tags.indexOf(k) >= 0 ? 1 : 0; });
    Object.keys(L.nodes).forEach(function (id) {
      var g = L.nodes[id];
      g.classList.remove("active"); g.removeAttribute("data-study");
      if (name !== "All") {
        var s = name, on = (st.dots[s] && st.dots[s].indexOf(id) >= 0) || (st.rings[s] && st.rings[s].indexOf(id) >= 0);
        if (on) { g.classList.add("active"); g.setAttribute("data-study", s); }
      }
    });
  }

  // observables determined by an observable (follow the solid arrows from finer to coarser)
  function descendants(id) {
    var out = {}, stack = [id];
    if (byId(id).dashed) return out;
    while (stack.length) {
      var c = stack.pop();
      EDGES.forEach(function (e) { if (e[0] === c && !e[4] && !out[e[1]]) { out[e[1]] = true; stack.push(e[1]); } });
    }
    return out;
  }

  function initHero() {
    var plate = document.getElementById("lattice-plate");
    if (!plate) return;
    var box = plate.querySelector(".lattice-box");
    var L = build(box, { interactive: true });
    built.push(L);
    var chips = plate.querySelectorAll(".chip");
    var slot = plate.querySelector(".cap-slot");
    var cap = plate.querySelector(".plate-cap");
    var def = plate.querySelector(".plate-def");
    var spec = plate.querySelector(".spec");
    var current = "All";

    function renderSpec(st) {
      spec.innerHTML = "";
      spec.className = "spec" + (st.spec ? "" : " is-empty");
      var dl = document.createElement("dl"); dl.className = "spec-grid";
      SPEC_KEYS.forEach(function (key, i) {
        var d = document.createElement("div");
        var dt = document.createElement("dt"); dt.innerHTML = key.html; if (key.v) dt.className = "var-key";
        var dd = document.createElement("dd");
        if (st.spec) dd.innerHTML = st.spec[i];
        else { dd.textContent = "–"; dd.setAttribute("aria-label", "none chosen"); }
        d.appendChild(dt); d.appendChild(dd); dl.appendChild(d);
      });
      spec.appendChild(dl);
    }
    function setState(name, fromUser) {
      if (!STATES[name]) return;
      current = name;
      apply(L, name);
      chips.forEach(function (c) { c.setAttribute("aria-pressed", c.getAttribute("data-study") === name ? "true" : "false"); });
      cap.textContent = STATES[name].caption;
      renderSpec(STATES[name]);
      if (fromUser) try { history.replaceState(null, "", name === "All" ? location.pathname + location.search : "#lattice-" + name.toLowerCase()); } catch (e) {}
    }
    chips.forEach(function (c) {
      c.addEventListener("click", function () { stopIntro(); setState(c.getAttribute("data-study"), true); });
    });

    // hover / focus: the definition replaces the caption, and the determined observables are highlighted
    function showNode(id) {
      var n = byId(id);
      var desc = descendants(id);
      L.svg.classList.add("hovering");
      Object.keys(L.nodes).forEach(function (k) {
        L.nodes[k].classList.toggle("hi-src", k === id);
        L.nodes[k].classList.toggle("hi", !!desc[k]);
      });
      L.edges.forEach(function (e) { e.g.classList.toggle("hi", !!((e.from === id || desc[e.from]) && desc[e.to])); });
      def.innerHTML = "";
      var b = document.createElement("b"); b.textContent = n.label; def.appendChild(b);
      var t = document.createElement("span"); t.innerHTML = n.tip; def.appendChild(t);
      if (Object.keys(desc).length) {
        var h = document.createElement("span"); h.className = "def-hint";
        h.textContent = "Highlighted boxes: the observables determined by this observable.";
        def.appendChild(h);
      }
      slot.classList.add("def");
    }
    function hideNode() {
      L.svg.classList.remove("hovering");
      Object.keys(L.nodes).forEach(function (k) { L.nodes[k].classList.remove("hi", "hi-src"); });
      L.edges.forEach(function (e) { e.g.classList.remove("hi"); });
      slot.classList.remove("def");
    }
    Object.keys(L.nodes).forEach(function (id) {
      var g = L.nodes[id];
      g.addEventListener("mouseenter", function () { showNode(id); });
      g.addEventListener("mouseleave", function () { if (document.activeElement !== g) hideNode(); });
      g.addEventListener("focus", function () { showNode(id); });
      g.addEventListener("blur", hideNode);
      g.addEventListener("click", function () { g.focus(); showNode(id); });
      g.addEventListener("keydown", function (ev) {
        if (ev.key === "Escape") { g.blur(); return; }
        var to = NAV[id] && NAV[id][ev.key];
        if (!/^Arrow/.test(ev.key)) return;
        ev.preventDefault();
        if (to) {
          g.setAttribute("tabindex", "-1");
          var t = L.nodes[to]; t.setAttribute("tabindex", "0"); t.focus();
        }
      });
    });

    // intro: the study marks appear one study at a time, then the All state holds
    var timers = [];
    function stopIntro() { timers.forEach(clearTimeout); timers = []; }
    setState("All");
    var hash = (location.hash || "").replace("#lattice-", "").toUpperCase();
    if (STATES[hash] && hash !== "ALL") {
      setState(hash);
      var r = plate.getBoundingClientRect();
      if (r.top > window.innerHeight * 0.45 || r.bottom < 0) {
        requestAnimationFrame(function () { plate.scrollIntoView({ block: "start" }); });
      }
    } else if (!REDUCED) {
      L.dots.forEach(function (d) { d.style.opacity = 0; d.style.transform = "scale(.3)"; });
      L.rings.forEach(function (r) { r.style.opacity = 0; });
      L.gap.style.opacity = 0;
      STUDIES.forEach(function (s, i) {
        timers.push(setTimeout(function () {
          L.dots.forEach(function (d) { if (d.getAttribute("data-study") === s) { d.style.opacity = 1; d.style.transform = "scale(1)"; } });
          if (s === "PRH") L.rings.forEach(function (r) { r.style.opacity = 1; });
        }, 650 + 420 * i));
      });
      timers.push(setTimeout(function () { L.gap.style.opacity = 1; L.dots.forEach(function (d) { d.style.transform = ""; }); }, 650 + 420 * 4 + 150));
    }

    window.addEventListener("hashchange", function () {
      var h = (location.hash || "").replace("#lattice-", "").toUpperCase();
      if (STATES[h] && h !== "ALL") {
        stopIntro(); hideNode(); setState(h);
        var r = plate.getBoundingClientRect();
        if (r.top > window.innerHeight * 0.45 || r.bottom < 0) plate.scrollIntoView({ block: "start" });
      }
    });

    window.LatticeHero = {
      show: function (name) { stopIntro(); setState(name, true); }
    };
  }

  function initStatic() {
    document.querySelectorAll("[data-lattice-static]").forEach(function (holder) {
      var name = holder.getAttribute("data-lattice-static");
      var L = build(holder, { interactive: false, label: holder.getAttribute("data-label") });
      built.push(L);
      L.svg.setAttribute("role", "img");
      apply(L, name);
    });
  }

  function init() { initHero(); initStatic(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
