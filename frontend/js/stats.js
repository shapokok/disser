(function () {
  const { el, $, $$, clear, api, toast, fmt, modelColor, cssVar, endLabelPlugin, errorBarPlugin } = App;
  const state = { stats: null, models: null, reports: null, research: null, charts: {}, classModel: null, sort: { key: "f1_score", dir: 1 }, hideDiag: false };
  let tooltipNode;

  // ------------------------------------------------------------------ tabs
  function showTab(name) {
    $$("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    $$(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
    if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    Object.values(state.charts).forEach((c) => c.resize());
  }

  // ------------------------------------------------------------------ helpers
  function chart(id, config) {
    if (state.charts[id]) state.charts[id].destroy();
    const ctx = $(`#${id}`);
    if (!ctx || !window.Chart) return null;
    state.charts[id] = new Chart(ctx, config);
    return state.charts[id];
  }
  const barValuePlugin = {
    id: "barValues",
    afterDatasetsDraw(chart, args, opts) {
      if (!opts || !opts.format) return;
      const { ctx } = chart;
      ctx.save();
      ctx.font = `600 11px ${cssVar("--font")}`;
      ctx.fillStyle = cssVar("--ink-2");
      ctx.textBaseline = "middle";
      const meta = chart.getDatasetMeta(0);
      meta.data.forEach((bar, i) => {
        const v = chart.data.datasets[0].data[i];
        if (v === null || v === undefined) return;
        ctx.fillText(opts.format(v), bar.x + 6, bar.y);
      });
      ctx.restore();
    },
  };
  function modelLabel(name) {
    if (name === "ensemble") return t("stats.tile_ensemble");
    const m = (state.models || []).find((x) => x.name === name);
    return m ? fmt.modelLabel(m) : name;
  }
  function tooltip(html, x, y) {
    if (!tooltipNode) { tooltipNode = el("div", { class: "tooltip", role: "tooltip" }); document.body.append(tooltipNode); }
    clear(tooltipNode).append(...html);
    tooltipNode.style.display = "block";
    const w = tooltipNode.offsetWidth, h = tooltipNode.offsetHeight;
    tooltipNode.style.left = `${Math.min(x + 14, window.innerWidth - w - 8)}px`;
    tooltipNode.style.top = `${Math.min(y + 14, window.innerHeight - h - 8)}px`;
  }
  function hideTooltip() { if (tooltipNode) tooltipNode.style.display = "none"; }
  function trainedModels() { return (state.models || []).filter((m) => m.trained); }

  // ------------------------------------------------------------------ overview
  function renderOverview() {
    const s = state.stats;
    const tiles = clear($("#overviewTiles"));
    const best = s.best_model && s.statistics[s.best_model];
    tiles.append(el("div", { class: "card tile hero-figure" }, [el("div", { class: "label", text: `${t("stats.tile_best")}${s.best_model ? " · " + modelLabel(s.best_model) : ""}` }), el("div", { class: "value", text: best ? fmt.pct(best.accuracy, 2) : t("common.na") }), el("div", { class: "delta", text: best ? `${t("common.f1")} ${fmt.pct(best.f1_score, 2)}` : "" })]));
    const ens = s.statistics.ensemble;
    tiles.append(el("div", { class: "card tile" }, [el("div", { class: "label", text: t("stats.tile_ensemble") }), el("div", { class: "value", text: ens ? fmt.pct(ens.accuracy, 2) : t("common.na") }), el("div", { class: "delta", text: ens ? (Array.isArray(ens.models_combined) ? ens.models_combined.map(modelLabel).join(" + ") : `${ens.models_combined} ×`) : "" })]));
    tiles.append(el("div", { class: "card tile" }, [el("div", { class: "label", text: t("stats.evaluated_on") }), el("div", { class: "value", text: best ? fmt.num(best.validation_samples) : t("common.na") }), el("div", { class: "delta", text: t("stats.valid_images") })]));
    const f = s.field_model || {};
    tiles.append(el("div", { class: "card tile" }, [el("div", { class: "label", text: t("stats.tile_field") }), el("div", { class: "value", text: f.available && f.accuracy_test !== undefined ? fmt.pct(f.accuracy_test) : t("common.na") }), el("div", { class: "delta", text: f.method || t("stats.research_pending").split(".")[0] })]));
    $("#evalInfo").textContent = best && best.evaluated_at ? `${t("stats.evaluated_on")}: ${new Date(best.evaluated_at).toLocaleString(i18n.lang === "ru" ? "ru-RU" : "en-US")}` : "";

    // table
    const head = ["common.model", "", "common.accuracy", "common.top5", "common.precision", "common.recall", "common.f1", "stats.ece", "common.inference", "common.params", "common.size"];
    const rows = state.models.map((m) => {
      const st = s.statistics[m.name] || {};
      return el("tr", {}, [
        el("td", {}, [el("span", { style: { display: "inline-block", width: "10px", height: "10px", borderRadius: "3px", background: modelColor(m.name), marginRight: ".5rem" } }), fmt.modelLabel(m)]),
        el("td", {}, m.trained ? el("span", { class: "chip good", text: m.name === s.best_model ? t("common.best") : t("common.trained") }) : el("span", { class: "chip warning", text: t("common.needs_training") })),
        el("td", { class: "num", text: fmt.pct(st.accuracy, 2) }),
        el("td", { class: "num", text: fmt.pct(st.top5_accuracy, 2) }),
        el("td", { class: "num", text: fmt.pct(st.precision, 2) }),
        el("td", { class: "num", text: fmt.pct(st.recall, 2) }),
        el("td", { class: "num", text: fmt.pct(st.f1_score, 2) }),
        el("td", { class: "num", text: st.ece === undefined || st.ece === null ? t("common.na") : st.ece.toFixed(3) }),
        el("td", { class: "num", text: fmt.ms(st.inference_time_ms) }),
        el("td", { class: "num", text: st.parameters || m.parameters || t("common.na") }),
        el("td", { class: "num", text: fmt.mb(st.size_mb) }),
      ]);
    });
    if (ens) rows.push(el("tr", {}, [el("td", {}, [el("span", { style: { display: "inline-block", width: "10px", height: "10px", borderRadius: "3px", background: modelColor("ensemble"), marginRight: ".5rem" } }), t("stats.tile_ensemble")]), el("td", {}, el("span", { class: "chip accent", text: (Array.isArray(ens.models_combined) ? ens.models_combined.length : ens.models_combined) + " ×" })), el("td", { class: "num", text: fmt.pct(ens.accuracy, 2) }), el("td", { class: "num", text: t("common.na") }), el("td", { class: "num", text: fmt.pct(ens.precision, 2) }), el("td", { class: "num", text: fmt.pct(ens.recall, 2) }), el("td", { class: "num", text: fmt.pct(ens.f1_score, 2) }), el("td", { class: "num", text: ens.ece === undefined || ens.ece === null ? t("common.na") : ens.ece.toFixed(3) }), el("td", { class: "num", text: fmt.ms(ens.inference_time_ms) }), el("td", { class: "num", text: ens.parameters }), el("td", { class: "num", text: fmt.mb(ens.size_mb) })]));
    clear($("#modelsTable")).append(el("table", { class: "data" }, [el("thead", {}, el("tr", {}, head.map((k, i) => el("th", { class: i >= 2 ? "num" : "", text: k ? t(k) : "" })))), el("tbody", {}, rows)]));

    // charts: nominal categories -> one hue
    const names = state.models.map((m) => m.name).concat(ens ? ["ensemble"] : []);
    const labels = names.map(modelLabel);
    const acc = names.map((n) => (s.statistics[n] || {}).accuracy ?? null);
    const times = names.map((n) => (s.statistics[n] || {}).inference_time_ms ?? null);
    const one = cssVar("--s1");
    chart("accChart", {
      type: "bar",
      data: { labels, datasets: [{ label: t("common.accuracy"), data: acc.map((x) => (x === null ? null : x * 100)), backgroundColor: one, maxBarThickness: 22 }] },
      options: { indexAxis: "y", maintainAspectRatio: false, layout: { padding: { right: 56 } }, scales: { x: { min: 0, max: 100, ticks: { callback: (v) => v + " %" } }, y: { grid: { display: false } } }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.parsed.x.toFixed(2)} %` } }, barValues: { format: (v) => v.toFixed(1) + " %" } } },
      plugins: [barValuePlugin],
    });
    chart("timeChart", {
      type: "bar",
      data: { labels, datasets: [{ label: t("common.inference"), data: times, backgroundColor: one, maxBarThickness: 22 }] },
      options: { indexAxis: "y", maintainAspectRatio: false, layout: { padding: { right: 56 } }, scales: { x: { min: 0, ticks: { callback: (v) => v + " ms" } }, y: { grid: { display: false } } }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.parsed.x} ms` } }, barValues: { format: (v) => v.toFixed(1) + " ms" } } },
      plugins: [barValuePlugin],
    });
  }

  // ------------------------------------------------------------------ training
  function renderTraining() {
    const hist = state.stats.training_history || {};
    const names = Object.keys(hist).filter((n) => hist[n] && hist[n].epochs);
    const maxEpochs = Math.max(0, ...names.map((n) => hist[n].epochs));
    const labels = Array.from({ length: maxEpochs }, (_, i) => i + 1);
    const ds = (key, scale) => names.map((n) => ({ label: modelLabel(n), data: hist[n][key].map((v) => v * scale), borderColor: modelColor(n), backgroundColor: modelColor(n), tension: 0.25, spanGaps: true }));
    const common = { maintainAspectRatio: false, layout: { padding: { right: 110 } }, interaction: { mode: "index", intersect: false }, plugins: { legend: { display: names.length > 1, position: "bottom" }, endLabels: { enabled: names.length <= 4 } }, scales: { x: { title: { display: true, text: i18n.lang === "ru" ? "Эпоха" : "Epoch" }, grid: { display: false } } } };
    chart("curvesChart", { type: "line", data: { labels, datasets: ds("val_accuracies", 1) }, options: { ...common, scales: { ...common.scales, y: { ticks: { callback: (v) => v + " %" } } }, plugins: { ...common.plugins, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(2)} %` } } } }, plugins: [endLabelPlugin] });
    chart("lossChart", { type: "line", data: { labels, datasets: ds("train_losses", 1) }, options: { ...common, scales: { ...common.scales, y: { beginAtZero: true } }, plugins: { ...common.plugins, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(4)}` } } } }, plugins: [endLabelPlugin] });

    const rows = names.map((n) => {
      const h = hist[n];
      return el("tr", {}, [el("td", { text: modelLabel(n) }), el("td", { class: "num", text: h.epochs }), el("td", { class: "num", text: h.batch_size ?? "—" }), el("td", { class: "num", text: h.learning_rate ?? "—" }), el("td", { text: h.optimizer || "Adam" }), el("td", { text: h.device || "cuda" }), el("td", { class: "num", text: h.total_time_minutes ? `${h.total_time_minutes.toFixed(0)} ${t("stats.min")}` : "—" }), el("td", { class: "num", text: h.best_val_accuracy ? `${h.best_val_accuracy.toFixed(2)} %` : "—" })]);
    });
    clear($("#trainingTable")).append(el("table", { class: "data" }, [el("thead", {}, el("tr", {}, [el("th", { text: t("common.model") }), el("th", { class: "num", text: t("stats.epochs") }), el("th", { class: "num", text: t("stats.batch") }), el("th", { class: "num", text: t("stats.lr") }), el("th", { text: t("stats.optimizer") }), el("th", { text: t("stats.device") }), el("th", { class: "num", text: t("stats.time_total") }), el("th", { class: "num", text: t("stats.best_val") })])), el("tbody", {}, rows.length ? rows : el("tr", {}, el("td", { colspan: 8, class: "muted", text: t("stats.no_data") })))]));
  }

  // ------------------------------------------------------------------ per class
  function fillModelSelects() {
    const names = Object.keys(state.reports || {});
    [$("#classModelSelect"), $("#cmModelSelect")].forEach((sel) => {
      const prev = sel.value;
      clear(sel);
      names.forEach((n) => sel.append(el("option", { value: n, text: modelLabel(n) })));
      sel.value = names.includes(prev) ? prev : (names.includes(state.stats.best_model) ? state.stats.best_model : names[0] || "");
    });
    state.classModel = $("#classModelSelect").value;
  }

  function renderClasses() {
    const rep = state.reports && state.reports[$("#classModelSelect").value];
    const hl = clear($("#classHighlights"));
    const table = clear($("#classTable"));
    if (!rep) { table.append(el("p", { class: "muted", style: { padding: "1rem" }, text: t("stats.no_data") })); return; }
    hl.append(
      el("div", { class: "grid grid-2" }, [
        el("div", {}, [el("strong", { class: "small", text: t("stats.best_classes") }), el("div", { class: "row", style: { marginTop: ".35rem" } }, rep.best_classes.map((c) => el("span", { class: "chip good", text: `${c.label} · ${fmt.pct(c.f1_score)}` })))]),
        el("div", {}, [el("strong", { class: "small", text: t("stats.worst_classes") }), el("div", { class: "row", style: { marginTop: ".35rem" } }, rep.worst_classes.map((c) => el("span", { class: "chip serious", text: `${c.label} · ${fmt.pct(c.f1_score)}` })))]),
      ])
    );
    const q = ($("#classSearch").value || "").toLowerCase();
    const { key, dir } = state.sort;
    const rows = rep.per_class.filter((c) => !q || c.label.toLowerCase().includes(q) || c.class_name.toLowerCase().includes(q)).sort((a, b) => (a[key] > b[key] ? 1 : a[key] < b[key] ? -1 : 0) * dir);
    const th = (label, k, num) => el("th", { class: num ? "num" : "" }, el("button", { type: "button", text: `${label}${state.sort.key === k ? (dir > 0 ? " ↑" : " ↓") : ""}`, onclick: () => { state.sort = { key: k, dir: state.sort.key === k ? -dir : 1 }; renderClasses(); } }));
    table.append(el("table", { class: "data" }, [
      el("thead", {}, el("tr", {}, [th(t("common.class"), "label"), th(t("common.precision"), "precision", true), th(t("common.recall"), "recall", true), th(t("common.f1"), "f1_score", true), el("th", { class: "bar-cell", text: "" }), th(t("stats.support"), "support", true)])),
      el("tbody", {}, rows.map((c) => el("tr", {}, [el("td", { text: c.label }), el("td", { class: "num", text: fmt.pct(c.precision) }), el("td", { class: "num", text: fmt.pct(c.recall) }), el("td", { class: "num", text: fmt.pct(c.f1_score) }), el("td", { class: "bar-cell" }, el("div", { class: "bar" }, el("span", { style: { width: `${Math.round(c.f1_score * 100)}%` } }))), el("td", { class: "num", text: c.support })]))),
    ]));
  }

  // ------------------------------------------------------------------ confusion matrix
  function seqColor(rate) {
    // sequential blue ramp, light -> dark; zero recedes to the surface
    if (!rate) return cssVar("--surface-2");
    const stops = [cssVar("--seq-100"), cssVar("--seq-250"), cssVar("--seq-400"), cssVar("--seq-550"), cssVar("--seq-700")];
    const x = Math.log10(1 + 9 * Math.min(1, rate)); // log scale so 1-5 % errors are visible
    return stops[Math.min(stops.length - 1, Math.floor(x * stops.length))];
  }
  function renderConfusion() {
    const rep = state.reports && state.reports[$("#cmModelSelect").value];
    const grid = clear($("#cmGrid"));
    const pairs = clear($("#cmPairs"));
    $("#cmHideDiagLabel").textContent = i18n.lang === "ru" ? "Скрыть диагональ" : "Hide diagonal";
    if (!rep) { grid.append(el("p", { class: "muted", text: t("stats.no_data") })); return; }
    const n = rep.labels.length;
    const cm = rep.confusion_matrix;
    const rowSums = cm.map((r) => r.reduce((a, b) => a + b, 0));
    const wrap = el("div", { style: { display: "grid", gridTemplateColumns: "140px 1fr", gap: "4px", minWidth: `${140 + n * 16}px` } });
    // column labels (rotated) + grid
    const colLabels = el("div", { class: "cm-labels", style: { gridColumn: "2", display: "grid", gridTemplateColumns: `repeat(${n}, 1fr)`, gap: "2px", height: "120px" } }, rep.labels.map((l) => el("div", { style: { writingMode: "vertical-rl", transform: "rotate(180deg)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontSize: ".6rem" }, text: l.split(" — ").pop() })));
    wrap.append(el("div"), colLabels);
    const rowLabels = el("div", { class: "cm-labels", style: { display: "grid", gridTemplateRows: `repeat(${n}, 1fr)`, gap: "2px" } }, rep.labels.map((l) => el("div", { style: { whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", lineHeight: "1", display: "flex", alignItems: "center" }, text: l })));
    const cells = el("div", { class: "cm", style: { gridTemplateColumns: `repeat(${n}, 1fr)` } });
    cm.forEach((row, i) => row.forEach((v, j) => {
      const rate = rowSums[i] ? v / rowSums[i] : 0;
      const hidden = state.hideDiag && i === j;
      const cell = el("div", { class: "cm-cell", tabindex: 0, style: { background: hidden ? cssVar("--surface-2") : seqColor(rate) } });
      const show = (e) => tooltip([el("strong", { text: `${v} (${(rate * 100).toFixed(1)} %)` }), el("div", { text: `${t("stats.true_class")}: ${rep.labels[i]}` }), el("div", { text: `${t("stats.predicted_class")}: ${rep.labels[j]}` })], e.clientX || 0, e.clientY || 0);
      cell.addEventListener("pointermove", show);
      cell.addEventListener("focus", (e) => { const r = cell.getBoundingClientRect(); show({ clientX: r.left, clientY: r.top }); });
      cell.addEventListener("pointerleave", hideTooltip);
      cell.addEventListener("blur", hideTooltip);
      cells.append(cell);
    }));
    wrap.append(rowLabels, cells);
    grid.append(wrap, el("div", { class: "legend", style: { marginTop: ".75rem" } }, [0, 0.02, 0.1, 0.4, 1].map((r) => el("span", { class: "key" }, [el("span", { class: "sw", style: { background: seqColor(r) } }), r ? `${r * 100} %` : "0"]))));

    pairs.append(el("table", { class: "data" }, [
      el("thead", {}, el("tr", {}, [el("th", { text: t("stats.true_class") }), el("th", { text: t("stats.predicted_class") }), el("th", { class: "num", text: t("stats.count") }), el("th", { class: "num", text: t("stats.rate") })])),
      el("tbody", {}, rep.top_confused_pairs.length ? rep.top_confused_pairs.map((p) => el("tr", {}, [el("td", { text: p.true_label }), el("td", { text: p.predicted_label }), el("td", { class: "num", text: p.count }), el("td", { class: "num", text: fmt.pct(p.rate) })])) : el("tr", {}, el("td", { colspan: 4, class: "muted", text: "—" }))),
    ]));
  }

  // ------------------------------------------------------------------ calibration
  function renderCalibration() {
    const table = clear($("#calibrationTable"));
    const mc = clear($("#mcnemarTable"));
    const reports = state.reports || {};
    const names = Object.keys(reports).filter((n) => reports[n].calibration);
    if (!names.length) {
      table.append(el("p", { class: "muted", style: { padding: "1rem" }, text: t("stats.no_data") }));
      chart("reliabilityChart", { type: "bar", data: { labels: [], datasets: [] }, options: { maintainAspectRatio: false } });
      return;
    }
    const s = state.stats.statistics || {};
    table.append(el("table", { class: "data" }, [
      el("thead", {}, el("tr", {}, [el("th", { text: t("common.model") }), el("th", { class: "num", text: "ECE" }), el("th", { class: "num", text: "ECE (T)" }), el("th", { class: "num", text: "T" }), el("th", { class: "num", text: "NLL" })])),
      el("tbody", {}, names.map((n) => { const c = reports[n].calibration; return el("tr", {}, [el("td", { text: modelLabel(n) }), el("td", { class: "num", text: c.ece.toFixed(4) }), el("td", { class: "num", text: c.ece_after_temperature.toFixed(4) }), el("td", { class: "num", text: c.temperature.toFixed(2) }), el("td", { class: "num", text: c.nll.toFixed(3) })]); })),
    ]));
    const comp = state.comparison;
    if (comp && comp.mcnemar && comp.mcnemar.length) {
      mc.append(el("table", { class: "data" }, [
        el("thead", {}, el("tr", {}, [el("th", { text: "A" }), el("th", { text: "B" }), el("th", { class: "num", text: "Acc A" }), el("th", { class: "num", text: "Acc B" }), el("th", { class: "num", text: t("stats.a_only") }), el("th", { class: "num", text: t("stats.b_only") }), el("th", { class: "num", text: "p" }), el("th")])),
        el("tbody", {}, comp.mcnemar.map((r) => { const sig = r.p_value < 0.05; return el("tr", {}, [el("td", { text: modelLabel(r.a) }), el("td", { text: modelLabel(r.b) }), el("td", { class: "num", text: fmt.pct(r.accuracy_a, 2) }), el("td", { class: "num", text: fmt.pct(r.accuracy_b, 2) }), el("td", { class: "num", text: r.a_only }), el("td", { class: "num", text: r.b_only }), el("td", { class: "num", text: r.p_value < 0.001 ? "< 0.001" : r.p_value.toFixed(3) }), el("td", {}, el("span", { class: "chip " + (sig ? "accent" : ""), text: sig ? t("stats.significant") : t("stats.not_significant") }))]); })),
      ]));
    }
    // reliability diagram: accuracy per confidence bin for the trained models (+ the diagonal)
    const trained = names.filter((n) => (s[n] || {}).trained !== false);
    const bins = reports[trained[0]].calibration.bins;
    const labels = bins.map((b) => `${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}`);
    const datasets = trained.slice(0, 4).map((n) => ({ label: modelLabel(n), data: reports[n].calibration.bins.map((b) => (b.accuracy === null ? null : b.accuracy * 100)), backgroundColor: modelColor(n), maxBarThickness: 14 }));
    datasets.push({ type: "line", label: t("stats.perfect"), data: bins.map((b) => ((b.lo + b.hi) / 2) * 100), borderColor: cssVar("--ink-3"), borderDash: [4, 4], pointRadius: 0, borderWidth: 1.5 });
    chart("reliabilityChart", {
      type: "bar",
      data: { labels, datasets },
      options: { maintainAspectRatio: false, scales: { x: { title: { display: true, text: t("stats.conf_bin") }, grid: { display: false } }, y: { min: 0, max: 100, ticks: { callback: (v) => v + " %" }, title: { display: true, text: t("common.accuracy") } } }, plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y === null ? "—" : c.parsed.y.toFixed(1) + " %"}` } } } },
    });
  }

  // ------------------------------------------------------------------ research
  function renderResearch() {
    const body = clear($("#researchBody"));
    const meta = clear($("#researchMeta"));
    const r = state.research;
    if (!r || !r.available || !r.summary) {
      body.append(el("div", { class: "card" }, el("div", { class: "card-body" }, [el("p", { class: "note", text: t("stats.research_pending") }), el("p", { class: "small muted", text: t("stats.legacy_note") })])));
      return;
    }
    const s = r.summary;
    const splits = (s.protocol && s.protocol.splits) || {};
    meta.append(el("span", { text: `${t("stats.splits")}: ${Object.entries(splits).map(([k, v]) => `${k} = ${v.n}`).join(", ")}` }), el("span", { text: `${t("stats.seeds")}: ${(s.seeds || []).join(", ")}` }), el("span", { text: s.generated_at ? new Date(s.generated_at).toLocaleString() : "" }));

    const methods = Object.entries(s.methods || {});
    const rows = methods.map(([k, m]) => el("tr", {}, [
      el("td", { text: k === "baseline" ? t("stats.zero_shot") : m.label }),
      el("td", { class: "num", text: `${fmt.pct(m.eval_open.accuracy_mean)}${m.eval_open.runs > 1 ? ` ± ${(m.eval_open.accuracy_std * 100).toFixed(1)}` : ""}` }),
      el("td", { class: "num", text: `${(m.eval_open.ci95_pooled[0] * 100).toFixed(0)}–${(m.eval_open.ci95_pooled[1] * 100).toFixed(0)}` }),
      el("td", { class: "num", text: fmt.pct(m.eval_open.macro_f1_mean) }),
      el("td", { class: "num", text: fmt.pct(m.test_open.accuracy_mean) }),
      el("td", { class: "num", text: fmt.pct(m.eval_restricted.accuracy_mean) }),
    ]));
    body.append(el("div", { class: "card" }, [el("div", { class: "card-head" }, el("h3", { text: t("stats.chart_da_title") })), el("div", { class: "table-wrap" }, el("table", { class: "data" }, [
      el("thead", {}, el("tr", {}, [el("th", { text: t("stats.method") }), el("th", { class: "num", text: t("stats.eval_acc") }), el("th", { class: "num", text: t("stats.ci") }), el("th", { class: "num", text: t("stats.macro_f1") || t("common.f1") }), el("th", { class: "num", text: t("stats.test_acc") }), el("th", { class: "num", text: t("stats.restricted") })])),
      el("tbody", {}, rows),
    ]))]));

    // chart: methods (emphasis: baseline gray, adapted methods one hue), CI error bars
    const c1 = el("canvas", { id: "daChart" });
    const c2 = el("canvas", { id: "pcChart" });
    body.append(el("div", { class: "grid grid-2" }, [
      el("div", { class: "card chart-card" }, el("div", { class: "card-body" }, [el("p", { class: "chart-title", text: t("stats.chart_da_title") }), el("p", { class: "chart-sub", text: t("stats.chart_da_sub") }), el("div", { class: "chart-box" }, c1)])),
      el("div", { class: "card chart-card" }, el("div", { class: "card-body" }, [el("p", { class: "chart-title", text: t("stats.chart_pc_title") }), el("p", { class: "chart-sub", text: t("stats.chart_pc_sub") }), el("div", { class: "chart-box" }, c2)])),
    ]));
    const labels = methods.map(([k, m]) => (k === "baseline" ? t("stats.zero_shot") : m.label));
    const vals = methods.map(([, m]) => m.eval_open.accuracy_mean * 100);
    const ci = methods.map(([, m]) => m.eval_open.ci95_pooled.map((x) => x * 100));
    const colors = methods.map(([k]) => (k === "baseline" ? cssVar("--ink-3") : cssVar("--s1")));
    chart("daChart", {
      type: "bar",
      data: { labels, datasets: [{ data: vals, backgroundColor: colors, maxBarThickness: 22 }] },
      options: { indexAxis: "y", maintainAspectRatio: false, layout: { padding: { right: 60 } }, scales: { x: { min: 0, max: 100, ticks: { callback: (v) => v + " %" } }, y: { grid: { display: false } } }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.parsed.x.toFixed(1)} % (95 % CI ${ci[c.dataIndex][0].toFixed(0)}–${ci[c.dataIndex][1].toFixed(0)})` } }, errorBars: { ci }, barValues: { format: (v) => v.toFixed(1) + " %" } } },
      plugins: [errorBarPlugin, barValuePlugin],
    });
    const pc = Object.entries(s.per_class_eval_open || {});
    chart("pcChart", {
      type: "bar",
      data: { labels: pc.map(([k]) => k), datasets: [
        { label: t("stats.zero_shot"), data: pc.map(([, v]) => v.baseline * 100), backgroundColor: cssVar("--ink-3"), maxBarThickness: 18 },
        { label: t("stats.adapted"), data: pc.map(([, v]) => v.joint * 100), backgroundColor: cssVar("--s1"), maxBarThickness: 18 },
      ] },
      options: { maintainAspectRatio: false, scales: { y: { min: 0, max: 100, ticks: { callback: (v) => v + " %" } }, x: { grid: { display: false } } }, plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(1)} %` } } } },
    });
    const a27 = s.baseline_all27_test;
    body.append(el("div", { class: "card" }, el("div", { class: "card-body small" }, [
      a27 ? el("p", { text: `${t("stats.all27")}: ${fmt.pct(a27.open.accuracy)} (n = ${a27.open.n}, top-5 ${fmt.pct(a27.top5_open)})` }) : null,
      el("p", { class: "muted", text: t("stats.legacy_note") }),
    ])));
  }

  // ------------------------------------------------------------------ load
  async function load() {
    try {
      const [stats, models, reports, research, comparison] = await Promise.all([
        api("/api/stats"), api("/api/models"), api(`/api/validation/all?lang=${i18n.lang}`).catch(() => ({ reports: {} })), api("/api/research").catch(() => null), api("/api/comparison").catch(() => null),
      ]);
      state.stats = stats; state.models = models.models; state.reports = reports.reports || {}; state.research = research; state.comparison = comparison;
      renderAll();
    } catch (e) { toast(e.message, "error", t("common.error")); }
  }
  function renderAll() {
    if (!state.stats) return;
    renderOverview();
    renderTraining();
    fillModelSelects();
    renderClasses();
    renderConfusion();
    renderCalibration();
    renderResearch();
  }

  document.addEventListener("DOMContentLoaded", () => {
    $$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
    const initial = (location.hash || "#overview").slice(1);
    showTab(["overview", "training", "classes", "confusion", "calibration", "research"].includes(initial) ? initial : "overview");
    $("#classModelSelect").addEventListener("change", renderClasses);
    $("#classSearch").addEventListener("input", renderClasses);
    $("#cmModelSelect").addEventListener("change", renderConfusion);
    $("#cmHideDiag").addEventListener("change", (e) => { state.hideDiag = e.target.checked; renderConfusion(); });
    load();
    document.addEventListener("langchange", async () => {
      try { state.reports = (await api(`/api/validation/all?lang=${i18n.lang}`)).reports || {}; } catch (e) { /* keep */ }
      renderAll();
    });
    document.addEventListener("themechange", () => { if (state.stats) renderAll(); });
  });
})();
