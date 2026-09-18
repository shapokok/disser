/* Shared runtime: API client, theme + language switches, toasts, DOM helpers, chart defaults. */
(function () {
  const API_BASE = window.CROPAI_API_BASE || (location.protocol.startsWith("http") ? "" : "http://localhost:5001");

  // ----------------------------------------------------------------- DOM helpers
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      Object.entries(attrs).forEach(([k, v]) => {
        if (v === null || v === undefined || v === false) return;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k === "html") node.innerHTML = v; // only for trusted, static markup (icons)
        else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
        else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
        else if (k === "dataset") Object.assign(node.dataset, v);
        else node.setAttribute(k, v === true ? "" : v);
      });
    }
    (Array.isArray(children) ? children : [children]).forEach((c) => {
      if (c === null || c === undefined || c === false) return;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    });
    return node;
  }
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

  const ICONS = {
    leaf: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>',
    sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>',
    moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>',
    menu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 6h16M4 12h16M4 18h16"/></svg>',
    upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4m0 0 4 4m-4-4-4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>',
    download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v12m0 0 4-4m-4 4-4-4"/><path d="M4 18v1a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-1"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 5 5L20 7"/></svg>',
    warn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>',
    image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/></svg>',
    cpu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>',
    eye: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></svg>',
    layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/></svg>',
    pill: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/></svg>',
    file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z"/><path d="M14 2v6h6M8 13h8M8 17h8"/></svg>',
    field: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 20c4-6 8-6 12-12"/><path d="M3 14c4-4 6-4 9-8"/><path d="M12 20c3-3 6-4 9-4"/></svg>',
    search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  };
  function icon(name, cls) { const s = el("span", { class: cls || "icon" }); s.innerHTML = ICONS[name] || ""; return s.firstChild; }

  // ----------------------------------------------------------------- toasts
  let toastRoot;
  function toast(message, kind, title) {
    if (!toastRoot) { toastRoot = el("div", { class: "toasts", role: "status", "aria-live": "polite" }); document.body.append(toastRoot); }
    const node = el("div", { class: "toast " + (kind || "") }, [title ? el("strong", { text: title }) : null, el("span", { text: message })]);
    toastRoot.append(node);
    setTimeout(() => { node.style.opacity = "0"; node.style.transition = "opacity .3s"; setTimeout(() => node.remove(), 320); }, kind === "error" ? 8000 : 4000);
    return node;
  }

  // ----------------------------------------------------------------- API
  async function api(path, opts) {
    const o = Object.assign({ method: "GET" }, opts || {});
    if (o.json !== undefined) {
      o.body = JSON.stringify(o.json);
      o.headers = Object.assign({ "Content-Type": "application/json" }, o.headers || {});
      delete o.json;
      if (o.method === "GET") o.method = "POST";
    }
    let res;
    try {
      res = await fetch(API_BASE + path, o);
    } catch (e) {
      const err = new Error(t("common.offline"));
      err.offline = true;
      throw err;
    }
    const type = res.headers.get("content-type") || "";
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      if (type.includes("application/json")) { try { const d = await res.json(); msg = d.error || msg; } catch (e) { /* ignore */ } }
      const err = new Error(msg);
      err.status = res.status;
      throw err;
    }
    if (type.includes("application/json")) return res.json();
    return res;
  }

  async function downloadBlob(path, payload, fallbackName) {
    const res = await api(path, { method: "POST", json: payload });
    const blob = await res.blob();
    const cd = res.headers.get("content-disposition") || "";
    const m = /filename\*?=(?:UTF-8'')?"?([^";]+)/i.exec(cd);
    const name = m ? decodeURIComponent(m[1]) : fallbackName;
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  // ----------------------------------------------------------------- theme
  const THEME_KEY = "cropai.theme";
  function currentTheme() {
    const explicit = document.documentElement.getAttribute("data-theme");
    if (explicit) return explicit;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function applyTheme(theme) {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
    const btn = $("#themeToggle");
    if (btn) { clear(btn).append(icon(currentTheme() === "dark" ? "sun" : "moon")); }
    document.dispatchEvent(new CustomEvent("themechange", { detail: { theme: currentTheme() } }));
  }
  function toggleTheme() {
    const next = currentTheme() === "dark" ? "light" : "dark";
    try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
    applyTheme(next);
  }
  (function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (e) { /* ignore */ }
    if (saved === "dark" || saved === "light") document.documentElement.setAttribute("data-theme", saved);
  })();

  // ----------------------------------------------------------------- formatting
  const fmt = {
    pct: (x, d = 1) => (x === null || x === undefined || isNaN(x) ? t("common.na") : (x * 100).toFixed(d) + " %"),
    num: (x) => (x === null || x === undefined ? t("common.na") : new Intl.NumberFormat(i18n.lang === "ru" ? "ru-RU" : "en-US").format(x)),
    ms: (x) => (x === null || x === undefined ? t("common.na") : `${Number(x).toFixed(x < 10 ? 1 : 0)} ${t("common.ms")}`),
    mb: (x) => (x === null || x === undefined ? t("common.na") : `${Number(x).toFixed(1)} MB`),
    cls: (entry) => (i18n.lang === "ru" ? entry.class_ru || entry.class : entry.class || entry.class_ru),
    plant: (entry) => (i18n.lang === "ru" ? entry.plant_ru || entry.plant : entry.plant),
    disease: (entry) => (i18n.lang === "ru" ? entry.disease_ru || entry.disease : entry.disease),
    modelLabel: (m) => (i18n.lang === "ru" ? m.label_ru || m.label : m.label),
    modelDesc: (m) => (i18n.lang === "ru" ? m.description_ru || m.description : m.description),
    modelTag: (m) => (i18n.lang === "ru" ? m.tagline_ru || m.tagline : m.tagline),
  };

  // Fixed categorical slots per model: identity never follows rank or order.
  const MODEL_SLOT = { baseline: "--s1", efficientnet: "--s2", mobilenet: "--s3", hybrid: "--s4", ensemble: "--s7", field: "--s6" };
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function modelColor(name) { return cssVar(MODEL_SLOT[name] || "--s8"); }

  // ----------------------------------------------------------------- Chart.js defaults
  function chartDefaults() {
    if (!window.Chart) return;
    const C = window.Chart;
    C.defaults.font.family = cssVar("--font") || "system-ui, sans-serif";
    C.defaults.font.size = 12;
    C.defaults.color = cssVar("--ink-2");
    C.defaults.borderColor = cssVar("--grid");
    C.defaults.plugins.legend.labels.boxWidth = 12;
    C.defaults.plugins.legend.labels.boxHeight = 12;
    C.defaults.plugins.legend.labels.usePointStyle = false;
    C.defaults.plugins.tooltip.backgroundColor = cssVar("--ink");
    C.defaults.plugins.tooltip.titleColor = cssVar("--page");
    C.defaults.plugins.tooltip.bodyColor = cssVar("--page");
    C.defaults.plugins.tooltip.cornerRadius = 8;
    C.defaults.plugins.tooltip.padding = 8;
    C.defaults.elements.line.borderWidth = 2;
    C.defaults.elements.line.borderJoinStyle = "round";
    C.defaults.elements.line.borderCapStyle = "round";
    C.defaults.elements.point.radius = 4;
    C.defaults.elements.point.hoverRadius = 6;
    C.defaults.elements.point.borderWidth = 2;
    C.defaults.elements.point.borderColor = cssVar("--surface");
    C.defaults.elements.bar.borderRadius = 4;
    C.defaults.elements.bar.borderSkipped = "start";
    C.defaults.maxBarThickness = 24;
    C.defaults.scale.grid.color = cssVar("--grid");
    C.defaults.scale.grid.lineWidth = 1;
    C.defaults.scale.border.color = cssVar("--axis");
    C.defaults.scale.ticks.color = cssVar("--ink-3");
    C.defaults.animation.duration = 300;
  }
  // Direct end-labels for line charts (name at the last point), so identity is not colour-alone.
  const endLabelPlugin = {
    id: "endLabels",
    afterDatasetsDraw(chart, args, opts) {
      if (!opts || opts.enabled === false) return;
      const { ctx } = chart;
      ctx.save();
      ctx.font = `600 11px ${cssVar("--font")}`;
      ctx.fillStyle = cssVar("--ink-2");
      ctx.textBaseline = "middle";
      const used = [];
      chart.data.datasets.forEach((ds, i) => {
        const meta = chart.getDatasetMeta(i);
        if (meta.hidden || !meta.data.length) return;
        const last = meta.data[meta.data.length - 1];
        let y = last.y;
        used.forEach((u) => { if (Math.abs(u - y) < 12) y = u + 12; });
        used.push(y);
        ctx.fillText(ds.label, last.x + 8, y);
      });
      ctx.restore();
    },
  };
  // Horizontal error bars (95% CI) for the DA comparison chart.
  const errorBarPlugin = {
    id: "errorBars",
    afterDatasetsDraw(chart, args, opts) {
      if (!opts || !opts.ci) return;
      const { ctx, scales } = chart;
      const x = scales.x;
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      ctx.strokeStyle = cssVar("--ink-2");
      ctx.lineWidth = 1.5;
      meta.data.forEach((bar, i) => {
        const ci = opts.ci[i];
        if (!ci) return;
        const x1 = x.getPixelForValue(ci[0]);
        const x2 = x.getPixelForValue(ci[1]);
        const y = bar.y;
        ctx.beginPath();
        ctx.moveTo(x1, y); ctx.lineTo(x2, y);
        ctx.moveTo(x1, y - 5); ctx.lineTo(x1, y + 5);
        ctx.moveTo(x2, y - 5); ctx.lineTo(x2, y + 5);
        ctx.stroke();
      });
      ctx.restore();
    },
  };

  // ----------------------------------------------------------------- page chrome
  function initChrome() {
    const themeBtn = $("#themeToggle");
    if (themeBtn) themeBtn.addEventListener("click", toggleTheme);
    applyTheme(document.documentElement.getAttribute("data-theme"));
    $$(".lang-switch button").forEach((b) => b.addEventListener("click", () => i18n.setLang(b.dataset.lang)));
    const menu = $("#menuToggle");
    const links = $("#navLinks");
    if (menu && links) menu.addEventListener("click", () => links.classList.toggle("open"));
    const page = document.body.dataset.page;
    $$(".nav-links a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
    $$("[data-icon]").forEach((n) => { clear(n).append(icon(n.dataset.icon)); });
    i18n.apply();
    chartDefaults();
    document.addEventListener("themechange", chartDefaults);
  }

  window.App = { el, $, $$, clear, icon, ICONS, toast, api, downloadBlob, fmt, modelColor, cssVar, initChrome, endLabelPlugin, errorBarPlugin, API_BASE };
  document.addEventListener("DOMContentLoaded", initChrome);
})();
