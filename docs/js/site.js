/* Page behavior: theme toggle, top bar with reading progress, current section, reveals, dialogs, figure stepping, copy, counters, math. */
(function () {
  "use strict";
  var doc = document.documentElement;
  var REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var darkMQ = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  /* ---------- theme ---------- */
  function currentTheme() {
    var t = doc.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return darkMQ && darkMQ.matches ? "dark" : "light";
  }
  function syncThemeMeta() {
    var meta = document.querySelector('meta[name="theme-color"]:not([media])');
    if (!meta) { meta = document.createElement("meta"); meta.name = "theme-color"; document.head.appendChild(meta); }
    meta.content = currentTheme() === "dark" ? "#0b0f15" : "#fbfaf7";
    document.querySelectorAll(".theme-btn").forEach(function (b) {
      b.setAttribute("aria-label", currentTheme() === "dark" ? "Switch to the light theme" : "Switch to the dark theme");
    });
  }
  document.querySelectorAll(".theme-btn").forEach(function (b) {
    b.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      doc.setAttribute("data-theme", next);
      try { localStorage.setItem("lrh-theme", next); } catch (e) {}
      syncThemeMeta();
    });
  });
  if (darkMQ && darkMQ.addEventListener) darkMQ.addEventListener("change", syncThemeMeta);
  syncThemeMeta();

  /* ---------- top bar over the hero ---------- */
  var topbar = document.getElementById("topbar");
  var hero = document.getElementById("top");
  var ticking = false;
  var progress = document.querySelector(".read-progress i");
  function onScroll() {
    ticking = false;
    if (!topbar || !hero) return;
    topbar.classList.toggle("at-top", window.scrollY < 24);
    topbar.classList.toggle("in-hero", window.scrollY < hero.offsetHeight - 140);
    var max = document.documentElement.scrollHeight - window.innerHeight;
    if (progress) progress.style.transform = "scaleX(" + (max > 0 ? Math.min(1, window.scrollY / max) : 0).toFixed(4) + ")";
    chapterFill();
  }
  /* the underline of the current chapter fills with the reading progress through that chapter */
  function chapterFill() {
    var a = document.querySelector('.chapters a[aria-current="true"]');
    var s = a && document.querySelector(a.getAttribute("href"));
    if (!s) return;
    var p = (window.scrollY + window.innerHeight * 0.42 - s.offsetTop) / s.offsetHeight;
    a.style.setProperty("--p", Math.max(0.06, Math.min(1, p)).toFixed(3));
  }
  window.addEventListener("scroll", function () { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
  window.addEventListener("resize", onScroll);
  onScroll();

  /* ---------- current section in the chapter nav ---------- */
  var nav = document.querySelector(".chapters");
  var links = nav ? Array.prototype.slice.call(nav.querySelectorAll("a")) : [];
  var sections = links.map(function (a) { return document.querySelector(a.getAttribute("href")); }).filter(Boolean);
  function markCurrent(id) {
    links.forEach(function (a) {
      var on = a.getAttribute("href") === "#" + id;
      if (on) {
        a.setAttribute("aria-current", "true");
        if (nav.scrollWidth > nav.clientWidth) {
          var left = a.offsetLeft - nav.clientWidth / 2 + a.offsetWidth / 2;
          nav.scrollTo({ left: left, behavior: REDUCED ? "auto" : "smooth" });
        }
      } else a.removeAttribute("aria-current");
    });
    chapterFill();
  }
  if ("IntersectionObserver" in window && sections.length) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) markCurrent(en.target.id); });
    }, { rootMargin: "-40% 0px -55% 0px" });
    sections.forEach(function (s) { io.observe(s); });
  }

  /* ---------- reveal on scroll ---------- */
  var revealEls = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && !REDUCED) {
    var rio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add("in"); rio.unobserve(en.target); } });
    }, { threshold: 0.08, rootMargin: "0px 0px -6% 0px" });
    revealEls.forEach(function (e) { rio.observe(e); });
  } else revealEls.forEach(function (e) { e.classList.add("in"); });

  /* ---------- dialogs ---------- */
  function openDialog(d) {
    if (!d) return;
    if (typeof d.showModal === "function") { d.showModal(); }
    else d.setAttribute("open", "");
  }
  document.querySelectorAll("dialog.dlg").forEach(function (d) {
    d.addEventListener("click", function (ev) { if (ev.target === d) d.close(); });
    d.querySelectorAll("[data-close]").forEach(function (b) { b.addEventListener("click", function () { d.close(); }); });
  });

  // cite
  var dlgCite = document.getElementById("dlg-cite");
  document.querySelectorAll("[data-open-cite]").forEach(function (a) {
    a.addEventListener("click", function (ev) { if (dlgCite && typeof dlgCite.showModal === "function") { ev.preventDefault(); openDialog(dlgCite); } });
  });

  // terms: inline links open the glossary entry in a dialog
  var dlgTerm = document.getElementById("dlg-term");
  document.addEventListener("click", function (ev) {
    var a = ev.target.closest && ev.target.closest("a.term-link");
    if (!a || !dlgTerm || typeof dlgTerm.showModal !== "function") return;
    if (ev.metaKey || ev.ctrlKey || ev.shiftKey) return;
    var entry = document.querySelector(a.getAttribute("href"));
    if (!entry) return;
    ev.preventDefault();
    dlgTerm.querySelector("#dlg-term-title").textContent = entry.querySelector("dt").textContent;
    dlgTerm.querySelector(".dlg-body").innerHTML = entry.querySelector("dd").innerHTML;
    var more = dlgTerm.querySelector(".dlg-foot a");
    if (more) more.setAttribute("href", a.getAttribute("href"));
    openDialog(dlgTerm);
  });
  if (dlgTerm) {
    var moreLink = dlgTerm.querySelector(".dlg-foot a");
    if (moreLink) moreLink.addEventListener("click", function () { dlgTerm.close(); });
  }

  // figures: open a larger copy, then step through all paper figures in page order
  var dlgFig = document.getElementById("dlg-fig");
  var figLinks = Array.prototype.slice.call(document.querySelectorAll("a.fig-img"));
  var figAt = 0, figOpener = null;
  function showFig(n) {
    figAt = (n + figLinks.length) % figLinks.length;
    var a = figLinks[figAt], img = a.querySelector("img"), big = dlgFig.querySelector("img");
    big.src = a.getAttribute("href"); big.alt = img ? img.alt : "";
    var fig = a.closest("figure"), cap = fig && fig.querySelector("figcaption");
    dlgFig.querySelector("figcaption").innerHTML = cap ? cap.innerHTML : "";
    var cnt = dlgFig.querySelector(".dlg-count");
    if (cnt) cnt.textContent = (figAt + 1) + " of " + figLinks.length;
    var pan = dlgFig.querySelector(".fig-img");
    if (pan) { pan.scrollLeft = 0; pan.scrollTop = 0; }
  }
  function openFig(a, opener) {
    if (!dlgFig || typeof dlgFig.showModal !== "function") return false;
    figOpener = opener || a; showFig(figLinks.indexOf(a));
    if (!dlgFig.open) openDialog(dlgFig);
    return true;
  }
  figLinks.forEach(function (a) {
    a.addEventListener("click", function (ev) {
      if (ev.metaKey || ev.ctrlKey) return;
      if (openFig(a)) ev.preventDefault();
    });
  });
  document.querySelectorAll("[data-enlarge]").forEach(function (b) {
    var a = b.closest("figure").querySelector("a.fig-img");
    if (a) a.tabIndex = -1; // one tab stop per figure: the Enlarge button; the image still opens on click
    b.addEventListener("click", function () { if (a) openFig(a, b); });
  });
  if (dlgFig) {
    dlgFig.querySelectorAll("[data-step]").forEach(function (b) { b.addEventListener("click", function () { showFig(figAt + (+b.getAttribute("data-step"))); }); });
    dlgFig.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") { ev.preventDefault(); showFig(figAt + (ev.key === "ArrowLeft" ? -1 : 1)); }
    });
    dlgFig.addEventListener("close", function () { if (figOpener) { figOpener.focus({ preventScroll: true }); figOpener = null; } });
  }

  /* ---------- copy BibTeX ---------- */
  document.querySelectorAll("[data-copy]").forEach(function (b) {
    b.addEventListener("click", function () {
      var src = document.querySelector(b.getAttribute("data-copy"));
      if (!src) return;
      var text = src.textContent;
      var label = b.querySelector(".lbl") || b;
      var orig = label.textContent;
      function done(ok) { label.textContent = ok ? "Copied" : "Selected for copying"; setTimeout(function () { label.textContent = orig; }, 1400); }
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(function () { done(true); }, function () { selectText(src); done(false); });
      else { selectText(src); done(false); }
    });
  });
  function selectText(node) { var r = document.createRange(); r.selectNodeContents(node); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); }

  /* ---------- show a study on the hero lattice ---------- */
  document.querySelectorAll("[data-show-study]").forEach(function (b) {
    b.addEventListener("click", function () {
      var plate = document.getElementById("lattice-plate");
      if (window.LatticeHero) window.LatticeHero.show(b.getAttribute("data-show-study"));
      if (plate) plate.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "center" });
      var chip = plate && plate.querySelector('.chip[data-study="' + b.getAttribute("data-show-study") + '"]');
      if (chip) setTimeout(function () { chip.focus({ preventScroll: true }); }, REDUCED ? 0 : 600);
    });
  });

  /* ---------- explore button: where the hero stacks, go to the lattice just below; otherwise to Section 02 ---------- */
  var enter = document.querySelector(".hero .enter");
  if (enter) enter.addEventListener("click", function (ev) {
    if (window.matchMedia("(max-width: 900px)").matches) {
      var plate = document.getElementById("lattice-plate");
      if (plate) { ev.preventDefault(); plate.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" }); }
    }
  });

  /* ---------- the numbers of Section 05 count up once when half visible ---------- */
  if (!REDUCED && "IntersectionObserver" in window) {
    var bigs = Array.prototype.filter.call(document.querySelectorAll(".stat .big"), function (b) { return !b.querySelector("sup") && /\d/.test(b.textContent); });
    var cio = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { cio.unobserve(en.target); countUp(en.target); } });
    }, { threshold: 0.5 });
    bigs.forEach(function (b) { b.setAttribute("aria-label", b.textContent); cio.observe(b); });
  }
  function countUp(b) {
    var html = b.innerHTML, txt = b.textContent, t0 = performance.now();
    (function frame(now) {
      var p = Math.min(1, (now - t0) / 900), e = 1 - Math.pow(1 - p, 3);
      if (p < 1) {
        b.textContent = txt.replace(/\d[\d,]*/g, function (s) { return String(Math.round(+s.replace(/,/g, "") * e)).replace(/\B(?=(\d{3})+(?!\d))/g, ","); });
        requestAnimationFrame(frame);
      } else b.innerHTML = html;
    })(t0);
  }

  /* ---------- math ---------- */
  function renderMath() {
    if (typeof window.renderMathInElement !== "function") return;
    window.renderMathInElement(document.body, {
      delimiters: [{ left: "\\(", right: "\\)", display: false }, { left: "\\[", right: "\\]", display: true }],
      ignoredClasses: ["bib", "no-math"],
      throwOnError: false
    });
  }
  renderMath();
})();
