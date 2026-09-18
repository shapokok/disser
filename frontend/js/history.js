(function () {
  const { el, $, $$, clear, icon, api, toast, fmt, cssVar, modelColor } = App;
  const state = { stats: null, items: [], total: 0, offset: 0, limit: 25, models: [], charts: {}, filters: { model: "", plant: "", healthy: "", q: "" } };
  let debounce;

  function chart(id, config) {
    if (state.charts[id]) state.charts[id].destroy();
    const ctx = $(`#${id}`);
    if (!ctx || !window.Chart) return null;
    state.charts[id] = new Chart(ctx, config);
    return state.charts[id];
  }

  function renderTiles() {
    const s = state.stats;
    const tiles = clear($("#historyTiles"));
    if (!s) return;
    const top = s.by_class.find((c) => !c.healthy) || null;
    const last7 = s.by_day.slice(-7).reduce((a, d) => a + d.n, 0);
    tiles.append(
      el("div", { class: "card tile hero-figure" }, [el("div", { class: "label", text: t("history.total") }), el("div", { class: "value", text: fmt.num(s.total) }), el("div", { class: "delta", text: s.last_at ? `${t("history.last")}: ${new Date(s.last_at).toLocaleString(i18n.lang === "ru" ? "ru-RU" : "en-US")}` : "" })]),
      el("div", { class: "card tile" }, [el("div", { class: "label", text: t("history.healthy_share") }), el("div", { class: "value", text: s.healthy_share === null ? "—" : fmt.pct(s.healthy_share, 0) }), el("div", { class: "delta", text: `${fmt.num(s.healthy)} / ${fmt.num(s.total)}` })]),
      el("div", { class: "card tile" }, [el("div", { class: "label", text: t("history.top_disease") }), el("div", { class: "value", style: { fontSize: "1.25rem" }, text: top ? fmt.cls(top) : "—" }), el("div", { class: "delta", text: top ? `${fmt.num(top.n)} · ${fmt.pct(top.conf, 0)}` : "" })]),
      el("div", { class: "card tile" }, [el("div", { class: "label", text: t("history.last7") }), el("div", { class: "value", text: fmt.num(last7) }), el("div", { class: "delta", text: s.by_day.length ? `${s.by_day.filter((d) => d.unhealthy).length} ${t("history.days_with_disease")}` : "" })]),
    );
  }

  function renderCharts() {
    const s = state.stats;
    if (!s) return;
    const one = cssVar("--s1");
    chart("daysChart", {
      type: "bar",
      data: { labels: s.by_day.map((d) => d.day.slice(5)), datasets: [{ label: t("history.analyses"), data: s.by_day.map((d) => d.n), backgroundColor: one, maxBarThickness: 22 }] },
      options: { maintainAspectRatio: false, scales: { y: { beginAtZero: true, ticks: { precision: 0 } }, x: { grid: { display: false } } }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.parsed.y} (${s.by_day[c.dataIndex].unhealthy} ${t("history.unhealthy")})` } } } },
    });
    const classes = s.by_class.slice(0, 8);
    chart("classesChart", {
      type: "bar",
      data: { labels: classes.map((c) => fmt.cls(c)), datasets: [{ label: t("history.analyses"), data: classes.map((c) => c.n), backgroundColor: classes.map((c) => (c.healthy ? cssVar("--ink-3") : one)), maxBarThickness: 20 }] },
      options: { indexAxis: "y", maintainAspectRatio: false, scales: { x: { beginAtZero: true, ticks: { precision: 0 } }, y: { grid: { display: false }, ticks: { callback: (v, i) => { const l = classes[i] ? fmt.cls(classes[i]) : ""; return l.length > 28 ? l.slice(0, 26) + "…" : l; } } } }, plugins: { legend: { display: false } } },
    });
  }

  function renderFilters() {
    const s = state.stats || { by_plant: [], by_model: [] };
    const selModel = $("#filterModel"), selPlant = $("#filterPlant"), selHealth = $("#filterHealth");
    const keep = (sel, val) => { sel.value = [...sel.options].some((o) => o.value === val) ? val : ""; };
    clear(selModel).append(el("option", { value: "", text: t("history.all_models") }), ...s.by_model.filter((m) => m.model).map((m) => { const info = state.models.find((x) => x.name === m.model); return el("option", { value: m.model, text: `${info ? fmt.modelLabel(info) : m.model} (${m.n})` }); }));
    clear(selPlant).append(el("option", { value: "", text: t("history.all_plants") }), ...s.by_plant.filter((p) => p.plant).map((p) => el("option", { value: p.plant, text: `${p.plant} (${p.n})` })));
    clear(selHealth).append(el("option", { value: "", text: t("history.all_states") }), el("option", { value: "0", text: t("history.only_disease") }), el("option", { value: "1", text: t("history.only_healthy") }));
    keep(selModel, state.filters.model); keep(selPlant, state.filters.plant); keep(selHealth, state.filters.healthy);
    $("#filterQuery").value = state.filters.q;
  }

  function renderTable() {
    const wrap = clear($("#historyTable"));
    const pager = clear($("#historyPager"));
    if (!state.items.length) {
      wrap.append(el("div", { class: "empty" }, [el("div", { "data-icon": "file" }), el("h3", { text: t("history.empty_title") }), el("p", { text: t("history.empty_sub") })]));
      $$("[data-icon]", wrap).forEach((n) => { clear(n).append(icon(n.dataset.icon)); });
      return;
    }
    const rows = state.items.map((it) => {
      const p = it.prediction;
      const conf = it.confidence || 0;
      const m = state.models.find((x) => x.name === it.model);
      return el("tr", {}, [
        el("td", {}, it.thumbnail ? el("img", { src: `data:image/jpeg;base64,${it.thumbnail}`, alt: "", style: { width: "44px", height: "44px", objectFit: "cover", borderRadius: "10px" } }) : ""),
        el("td", {}, [el("div", { style: { fontWeight: 600 }, text: p ? fmt.disease(p) : "—" }), el("div", { class: "small muted", text: p ? fmt.plant(p) : "" })]),
        el("td", {}, p && p.healthy ? el("span", { class: "chip good" }, [icon("check"), t("history.healthy")]) : el("span", { class: "chip " + (conf >= 0.8 ? "serious" : "warning"), text: t("history.disease") })),
        el("td", { class: "bar-cell" }, [el("div", { class: "small", text: it.prediction ? it.prediction.confidence_percent : "" }), el("div", { class: "bar" }, el("span", { style: { width: `${Math.round(conf * 100)}%` } }))]),
        el("td", {}, [el("span", { style: { display: "inline-block", width: "10px", height: "10px", borderRadius: "3px", background: modelColor(it.model), marginRight: ".4rem", verticalAlign: "middle" } }), it.model_label || (m ? fmt.modelLabel(m) : it.model), el("div", { class: "small muted", text: `${(it.explanation || "").toUpperCase()} · ${it.dataset_type === "field" ? t("analyze.mode_field") : t("analyze.mode_controlled")}` })]),
        el("td", { class: "small muted", text: new Date(it.ts).toLocaleString(i18n.lang === "ru" ? "ru-RU" : "en-US") }),
        el("td", { class: "small muted", style: { maxWidth: "180px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, title: it.image_name, text: it.image_name }),
        el("td", {}, el("button", { class: "btn sm ghost", title: t("history.delete"), text: "×", onclick: () => removeItem(it.id) })),
      ]);
    });
    wrap.append(el("table", { class: "data" }, [
      el("thead", {}, el("tr", {}, [el("th"), el("th", { text: t("analyze.diagnosis") }), el("th", { text: t("history.state") }), el("th", { text: t("common.confidence") }), el("th", { text: t("common.model") }), el("th", { text: t("history.when") }), el("th", { text: t("history.file") }), el("th")])),
      el("tbody", {}, rows),
    ]));
    const pages = Math.ceil(state.total / state.limit);
    const page = Math.floor(state.offset / state.limit) + 1;
    pager.append(
      el("span", { class: "small muted", text: `${state.total} · ${page}/${pages}` }),
      el("span", { class: "grow" }),
      el("button", { class: "btn sm", disabled: state.offset === 0, text: "←", onclick: () => { state.offset = Math.max(0, state.offset - state.limit); loadItems(); } }),
      el("button", { class: "btn sm", disabled: state.offset + state.limit >= state.total, text: "→", onclick: () => { state.offset += state.limit; loadItems(); } }),
    );
  }

  async function removeItem(id) {
    try { await api(`/api/history/${id}`, { method: "DELETE" }); toast(t("history.deleted"), "success"); await load(); } catch (e) { toast(e.message, "error", t("common.error")); }
  }

  async function loadItems() {
    const f = state.filters;
    const qs = new URLSearchParams({ limit: state.limit, offset: state.offset, model: f.model, plant: f.plant, healthy: f.healthy, q: f.q });
    const data = await api(`/api/history?${qs}`);
    state.items = data.items;
    state.total = data.total;
    renderTable();
  }

  async function load() {
    try {
      const [stats, models] = await Promise.all([api("/api/history/stats"), api("/api/models")]);
      state.stats = stats;
      state.models = models.models;
      renderTiles();
      renderCharts();
      renderFilters();
      await loadItems();
    } catch (e) {
      toast(e.message, "error", t("common.error"));
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    load();
    const onFilter = () => { state.filters = { model: $("#filterModel").value, plant: $("#filterPlant").value, healthy: $("#filterHealth").value, q: $("#filterQuery").value.trim() }; state.offset = 0; loadItems().catch((e) => toast(e.message, "error")); };
    ["#filterModel", "#filterPlant", "#filterHealth"].forEach((s) => $(s).addEventListener("change", onFilter));
    $("#filterQuery").addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(onFilter, 250); });
    $("#clearHistory").addEventListener("click", async () => {
      if (!window.confirm(t("history.clear_confirm"))) return;
      try { await api("/api/history", { method: "DELETE" }); toast(t("history.cleared"), "success"); await load(); } catch (e) { toast(e.message, "error", t("common.error")); }
    });
    document.addEventListener("langchange", () => { renderTiles(); renderCharts(); renderFilters(); renderTable(); });
    document.addEventListener("themechange", () => { if (state.stats) renderCharts(); });
  });
})();
