(function () {
  const { el, $, $$, clear, icon, api, downloadBlob, toast, fmt, modelColor, ring } = App;
  const MAX_MB = 16;
  const state = { files: [], results: [], compare: null, models: null, field: null, explanation: "gradcam", mode: "controlled", busy: false };

  // ------------------------------------------------------------------ settings
  async function loadModels() {
    try {
      const data = await api("/api/models");
      state.models = data.models;
      state.field = data.field_model;
      state.bestModel = data.best_model;
      renderModelSelect();
      renderModeHint();
    } catch (e) {
      toast(e.message, "error", t("common.error"));
    }
  }

  function renderModelSelect() {
    const sel = $("#modelSelect");
    const prev = sel.value;
    clear(sel);
    const trained = (state.models || []).filter((m) => m.trained);
    if (trained.length >= 2) sel.append(el("option", { value: "ensemble", text: t("analyze.ensemble_opt") }));
    (state.models || []).forEach((m) => {
      const suffix = m.trained ? (m.name === state.bestModel ? ` — ${t("common.recommended")}` : "") : ` — ${t("common.needs_training")}`;
      sel.append(el("option", { value: m.name, text: `${fmt.modelLabel(m)}${suffix}` }));
    });
    sel.value = [...sel.options].some((o) => o.value === prev) ? prev : (state.bestModel && trained.length ? (trained.length >= 2 ? "ensemble" : state.bestModel) : sel.options[0] && sel.options[0].value);
    renderModelHint();
  }

  function renderModelHint() {
    const sel = $("#modelSelect");
    const hint = $("#modelHint");
    const m = (state.models || []).find((x) => x.name === sel.value);
    clear(hint);
    if (sel.value === "ensemble") {
      hint.textContent = (state.models || []).filter((x) => x.trained).map(fmt.modelLabel).join(" + ");
    } else if (m) {
      hint.append(fmt.modelDesc(m), " ");
      if (m.accuracy !== null && m.accuracy !== undefined) hint.append(el("span", { class: "chip " + (m.trained ? "good" : "warning"), text: `${t("common.accuracy")} ${fmt.pct(m.accuracy)}` }));
      if (!m.trained) hint.append(el("div", { class: "note", style: { marginTop: ".4rem" }, text: `${t("analyze.untrained_warn")} ${m.name}` }));
    }
  }

  function renderModeHint() {
    const hint = clear($("#modeHint"));
    if (state.mode !== "field") return;
    if (state.field && state.field.available) {
      hint.append(t("analyze.mode_hint_on"), " ", (state.field.covered_classes || []).map((c) => c.replace("___", " — ").replace(/_/g, " ")).join(", "));
    } else {
      hint.append(el("span", { class: "chip warning" }, [icon("warn"), t("analyze.mode_hint_off")]));
    }
  }

  function segmented(id, onChange) {
    $$(`#${id} button`).forEach((b) =>
      b.addEventListener("click", () => {
        $$(`#${id} button`).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        onChange(b.dataset.value);
      })
    );
  }

  // ------------------------------------------------------------------ files
  function addFiles(list) {
    const errors = [];
    Array.from(list).forEach((f) => {
      const ext = (f.name.split(".").pop() || "").toLowerCase();
      if (!["jpg", "jpeg", "png", "jfif", "webp", "bmp"].includes(ext)) return errors.push(`${f.name}: ${ext || "?"}`);
      if (f.size > MAX_MB * 1024 * 1024) return errors.push(`${f.name}: > ${MAX_MB} MB`);
      if (state.files.some((x) => x.file.name === f.name && x.file.size === f.size)) return;
      state.files.push({ file: f, url: URL.createObjectURL(f) });
    });
    if (errors.length) toast(errors.join("; "), "warning", t("analyze.errors_upload"));
    renderPreviews();
  }

  function renderPreviews() {
    const box = clear($("#previews"));
    state.files.forEach((f, i) => {
      box.append(
        el("div", { class: "preview" }, [
          el("img", { src: f.url, alt: f.file.name }),
          el("div", { class: "name", text: f.file.name }),
          el("button", { class: "remove", type: "button", title: t("analyze.remove"), text: "×", onclick: () => { URL.revokeObjectURL(f.url); state.files.splice(i, 1); renderPreviews(); } }),
        ])
      );
    });
    $("#analyzeBtn").disabled = !state.files.length || state.busy;
    $("#compareBtn").disabled = !state.files.length || state.busy;
  }

  function setBusy(v) {
    state.busy = v;
    $("#analyzeBtn").disabled = v || !state.files.length;
    $("#compareBtn").disabled = v || !state.files.length;
  }

  async function upload() {
    const fd = new FormData();
    state.files.forEach((f) => fd.append("files", f.file, f.file.name));
    const data = await api("/api/upload", { method: "POST", body: fd });
    if (data.errors && data.errors.length) toast(data.errors.join("; "), "warning", t("analyze.errors_upload"));
    if (!data.uploaded.length) throw new Error(t("analyze.errors_upload"));
    return data.uploaded;
  }

  // ------------------------------------------------------------------ progress
  function progressCard() {
    const bar = el("span", { style: { width: "0%" } });
    const text = el("div", { class: "row" }, [el("div", { class: "spinner" }), el("span", { class: "muted", text: t("analyze.progress_upload") })]);
    const card = el("div", { class: "card progress" }, [text, el("div", { class: "bar" }, bar)]);
    return {
      card,
      set(msg, frac) { text.lastChild.textContent = msg; bar.style.width = `${Math.round(frac * 100)}%`; },
      done() { card.remove(); },
    };
  }

  // ------------------------------------------------------------------ analyze
  async function analyze() {
    if (!state.files.length || state.busy) return;
    setBusy(true);
    $("#emptyState").classList.add("hidden");
    const results = $("#results");
    const prog = progressCard();
    results.prepend(prog.card);
    try {
      const uploaded = await upload();
      const model = $("#modelSelect").value;
      for (let i = 0; i < uploaded.length; i++) {
        prog.set(`${t("analyze.progress_predict")} (${i + 1} ${t("analyze.progress_of")} ${uploaded.length})`, (i + 0.3) / uploaded.length);
        const payload = { image_path: uploaded[i].saved_name, explanation: state.explanation, dataset_type: state.mode, lang: i18n.lang };
        const path = model === "ensemble" ? "/api/ensemble" : "/api/predict";
        if (model !== "ensemble") payload.model = model;
        try {
          const r = await api(path, { json: payload });
          r._localUrl = (state.files[i] || {}).url;
          state.results.unshift(r);
          results.insertBefore(resultCard(r), prog.card.nextSibling);
        } catch (e) {
          toast(`${uploaded[i].original_name}: ${e.message}`, "error", t("common.error"));
        }
      }
      toast(t("analyze.done"), "success");
      renderExportBar();
    } catch (e) {
      toast(e.message, "error", t("common.error"));
      if (!state.results.length) $("#emptyState").classList.remove("hidden");
    } finally {
      prog.done();
      setBusy(false);
    }
  }

  function confidenceClass(c) { return c >= 0.8 ? "good" : c >= 0.5 ? "warning" : "critical"; }

  function vizBlock(r) {
    const imgs = r.images || {};
    const tabs = [];
    if (imgs.overlay) tabs.push(["overlay", t("analyze.overlay")]);
    if (imgs.heatmap) tabs.push(["heatmap", t("analyze.heatmap")]);
    if (imgs.lime) tabs.push(["lime", t("analyze.lime_tab")]);
    if (imgs.lime_regions) tabs.push(["lime_regions", t("analyze.lime_regions")]);
    if (imgs.original) tabs.push(["original", t("analyze.original")]);
    const img = el("img", { alt: fmt.cls(r.prediction) });
    const show = (key) => { img.src = `data:image/jpeg;base64,${imgs[key]}`; $$("button", tabBar).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key))); };
    const tabBar = el("div", { class: "viz-tabs" }, tabs.map(([key, label]) => el("button", { type: "button", dataset: { key }, text: label, onclick: () => show(key) })));
    if (tabs.length) show(tabs[0][0]);
    else if (r._localUrl) img.src = r._localUrl;
    const dl = el("a", { class: "btn sm ghost", download: `${r.image_name}_${r.explanation}.jpg`, href: "#" }, [icon("download"), t("common.download")]);
    dl.addEventListener("click", (e) => { e.preventDefault(); const a = el("a", { href: img.src, download: dl.download }); document.body.append(a); a.click(); a.remove(); });
    tabBar.append(el("span", { class: "grow" }), dl);
    return el("div", { class: "viz" }, [img, tabBar]);
  }

  function treatmentBlock(tr) {
    if (!tr) return null;
    const lists = [["treatments", "analyze.treatments"], ["prevention", "analyze.prevention"], ["organic_options", "analyze.organic"]];
    const body = [];
    if (tr.pathogen) body.push(el("p", { class: "small" }, [el("strong", { text: t("analyze.pathogen") + ": " }), tr.pathogen]));
    if (tr.severity) body.push(el("p", { class: "small" }, [el("strong", { text: t("analyze.severity") + ": " }), tr.severity]));
    if (tr.symptoms) body.push(el("h4", { text: t("analyze.symptoms") }), el("p", { class: "small", text: tr.symptoms }));
    lists.forEach(([key, label]) => {
      if (tr[key] && tr[key].length) body.push(el("h4", { text: t(label) }), el("ul", { class: "small" }, tr[key].map((x) => el("li", { text: x }))));
    });
    if (tr.generic) body.push(el("p", { class: "small muted", text: t("analyze.generic_note") }));
    return el("div", { class: "treatment" }, el("details", { open: !tr.healthy }, [el("summary", { text: `${t("analyze.treatment")}: ${tr.disease_name}` }), ...body]));
  }

  function ensembleBlock(ens) {
    if (!ens) return null;
    const unc = ens.uncertainty_metrics || {};
    const level = unc.uncertainty_level || "high";
    const chipCls = level === "low" ? "good" : level === "medium" ? "warning" : "critical";
    return el("div", { class: "stack", style: { marginTop: "1rem" } }, [
      el("div", { class: "row" }, [
        el("strong", { text: t("analyze.ensemble") }),
        el("span", { class: "chip", text: `${t("analyze.agreement")}: ${ens.agreement_percent}` }),
        el("span", { class: "chip " + chipCls }, [icon(level === "low" ? "check" : "warn"), `${t("analyze.uncertainty")}: ${t("analyze.unc_" + level)}`]),
      ]),
      el("div", { class: "ensemble-grid" }, Object.entries(ens.individual_predictions || {}).map(([name, p]) => {
        const m = (state.models || []).find((x) => x.name === name);
        return el("div", { class: "mini" }, [
          el("span", {}, [el("span", { class: "swatch", style: { display: "inline-block", width: "10px", height: "10px", borderRadius: "3px", background: modelColor(name), marginRight: ".4rem" } }), m ? fmt.modelLabel(m) : name]),
          el("strong", { text: fmt.cls(p) }),
          el("span", { class: "muted", text: p.confidence_percent }),
        ]);
      })),
      el("p", { class: "small muted", text: `${t("analyze.explained_with")}: ${ens.explained_with}. H = ${unc.entropy}, σ² = ${unc.prediction_variance}` }),
    ]);
  }

  function resultCard(r) {
    const p = r.prediction;
    const conf = p.confidence;
    const modeChip = el("span", { class: "chip", text: r.dataset_type === "field" ? t("analyze.mode_field") : t("analyze.mode_controlled") });
    const head = el("div", { class: "result-head" }, [
      el("div", {}, [el("strong", { text: r.image_name }), el("div", { class: "small muted", text: new Date(r.timestamp).toLocaleString(i18n.lang === "ru" ? "ru-RU" : "en-US") })]),
      el("div", { class: "row" }, [
        el("span", { class: "chip accent" }, [el("span", { style: { width: "10px", height: "10px", borderRadius: "3px", background: modelColor(r.model), display: "inline-block" } }), r.model_label]),
        el("span", { class: "chip", text: r.explanation.toUpperCase() }),
        modeChip,
        el("span", { class: "chip", text: `${fmt.ms(r.inference_time_ms)}` }),
      ]),
    ]);

    const confColor = conf >= 0.8 ? "var(--good)" : conf >= 0.5 ? "var(--warning)" : "var(--critical)";
    const diagnosis = el("div", { class: "diagnosis stack" }, [
      el("div", { class: "diagnosis-head" }, [
        ring(conf, confColor, { size: "lg", label: `${(conf * 100).toFixed(1)}%` }),
        el("div", {}, [
          el("div", { class: "plant", text: `${t("analyze.diagnosis")} · ${fmt.plant(p)}` }),
          el("div", { class: "disease", text: fmt.disease(p) }),
          p.healthy ? el("span", { class: "chip good" }, [icon("check"), t("analyze.healthy")]) : el("span", { class: "chip " + confidenceClass(conf), text: `${t("common.confidence")}: ${p.confidence_percent}` }),
        ]),
      ]),
      r.field_mode ? el("div", { class: "note " + (r.field_mode.adapted_model ? "info" : "") , text: r.field_mode.adapted_model ? t("analyze.field_used") : t("analyze.field_missing") }) : null,
      el("div", {}, [
        el("strong", { class: "small", text: t("analyze.top5") }),
        el("ul", { class: "topk" }, (r.top_predictions || []).map((x) => el("li", {}, [
          el("div", {}, [el("div", { class: "name", text: fmt.cls(x) }), el("div", { class: "bar" }, el("span", { style: { width: `${Math.max(2, x.confidence * 100)}%` } }))]),
          el("span", { class: "pct", text: x.confidence_percent }),
        ]))),
      ]),
      ensembleBlock(r.ensemble),
      treatmentBlock(r.treatment),
    ]);

    const footer = el("div", { class: "result-footer" }, [
      el("button", { class: "btn sm", onclick: () => exportOne(r, "pdf") }, [icon("file"), t("analyze.report")]),
      el("button", { class: "btn sm", onclick: () => exportOne(r, "excel") }, "Excel"),
      el("button", { class: "btn sm", onclick: () => exportOne(r, "csv") }, "CSV"),
      el("button", { class: "btn sm", onclick: () => exportOne(r, "json") }, "JSON"),
    ]);
    const card = el("article", { class: "card result-card", dataset: { ts: r.timestamp } }, [head, el("div", { class: "result-body" }, [vizBlock(r), diagnosis]), footer]);
    return card;
  }

  async function exportOne(r, fmtName) {
    try {
      await downloadBlob(`/api/export/${fmtName}?lang=${i18n.lang}`, { results: [r], lang: i18n.lang }, `result.${fmtName === "excel" ? "xlsx" : fmtName}`);
      toast(t("analyze.export_done"), "success");
    } catch (e) { toast(e.message, "error", t("common.error")); }
  }

  async function exportAll(fmtName) {
    if (!state.results.length) return;
    try {
      await downloadBlob(`/api/export/${fmtName}?lang=${i18n.lang}`, { results: state.results, lang: i18n.lang }, `results.${fmtName === "excel" ? "xlsx" : fmtName}`);
      toast(t("analyze.export_done"), "success");
    } catch (e) { toast(e.message, "error", t("common.error")); }
  }

  function renderExportBar() {
    let bar = $("#exportBar");
    if (!state.results.length) { if (bar) bar.remove(); return; }
    if (!bar) { bar = el("div", { class: "card", id: "exportBar" }); $("#results").prepend(bar); }
    clear(bar).append(el("div", { class: "result-footer", style: { borderTop: 0 } }, [
      el("strong", { class: "small", text: `${t("analyze.export_all")} (${state.results.length})` }),
      el("span", { class: "grow" }),
      el("button", { class: "btn sm", onclick: () => exportAll("pdf") }, [icon("file"), t("common.pdf")]),
      el("button", { class: "btn sm", onclick: () => exportAll("excel") }, "Excel"),
      el("button", { class: "btn sm", onclick: () => exportAll("csv") }, "CSV"),
      el("button", { class: "btn sm", onclick: () => exportAll("json") }, "JSON"),
    ]));
  }

  // ------------------------------------------------------------------ compare
  async function compare() {
    if (!state.files.length || state.busy) return;
    setBusy(true);
    $("#emptyState").classList.add("hidden");
    const prog = progressCard();
    $("#results").prepend(prog.card);
    try {
      const uploaded = await upload();
      prog.set(t("analyze.progress_predict"), 0.5);
      const data = await api("/api/compare", { json: { image_path: uploaded[0].saved_name, lang: i18n.lang } });
      data._localUrl = state.files[0].url;
      state.compare = data;
      $("#results").insertBefore(compareCard(data), prog.card.nextSibling);
      toast(t("analyze.compare_done"), "success");
    } catch (e) {
      toast(e.message, "error", t("common.error"));
    } finally { prog.done(); setBusy(false); }
  }

  function compareCard(data) {
    const rows = Object.entries(data.comparisons).map(([name, r]) => {
      const m = (state.models || []).find((x) => x.name === name);
      return el("tr", {}, [
        el("td", {}, [el("span", { style: { display: "inline-block", width: "10px", height: "10px", borderRadius: "3px", background: modelColor(name), marginRight: ".5rem" } }), r.label || (m ? fmt.modelLabel(m) : name), r.trained ? null : el("span", { class: "chip warning", style: { marginLeft: ".5rem" }, text: t("common.needs_training") })]),
        el("td", { text: fmt.cls(r) }),
        el("td", { class: "bar-cell" }, [el("div", { class: "row between small" }, [el("span", { text: r.confidence_percent })]), el("div", { class: "bar" }, el("span", { style: { width: `${Math.round(r.confidence * 100)}%` } }))]),
        el("td", { class: "num", text: fmt.ms(r.inference_time_ms) }),
      ]);
    });
    const agr = data.agreement;
    return el("article", { class: "card result-card" }, [
      el("div", { class: "result-head" }, [
        el("div", {}, [el("strong", { text: t("analyze.compare_title") }), el("div", { class: "small muted", text: data.image_name })]),
        el("span", { class: "chip accent" }, [icon("check"), `${agr.models_agreeing}/${agr.models_total} ${t("analyze.compare_agree")}: ${fmt.cls(agr.class)}`]),
      ]),
      el("div", { class: "result-body compare-body" }, [
        el("div", { class: "viz" }, el("img", { src: data._localUrl, alt: data.image_name })),
        el("div", { class: "table-wrap" }, el("table", { class: "data" }, [
          el("thead", {}, el("tr", {}, [el("th", { text: t("common.model") }), el("th", { text: t("common.class") }), el("th", { text: t("common.confidence") }), el("th", { class: "num", text: t("analyze.time") })])),
          el("tbody", {}, rows),
        ])),
      ]),
      el("div", { class: "result-footer" }, [
        el("button", { class: "btn sm", onclick: () => exportCompare(data, "pdf") }, [icon("file"), t("analyze.report")]),
        el("button", { class: "btn sm", onclick: () => exportCompare(data, "excel") }, "Excel"),
        el("button", { class: "btn sm", onclick: () => exportCompare(data, "csv") }, "CSV"),
      ]),
    ]);
  }

  async function exportCompare(data, fmtName) {
    try {
      await downloadBlob(`/api/export/comparison/${fmtName}?lang=${i18n.lang}`, Object.assign({}, data, { lang: i18n.lang, _localUrl: undefined }), `comparison.${fmtName === "excel" ? "xlsx" : fmtName}`);
      toast(t("analyze.export_done"), "success");
    } catch (e) { toast(e.message, "error", t("common.error")); }
  }

  // ------------------------------------------------------------------ language switch: re-render results
  async function rerender() {
    renderModelSelect();
    renderModeHint();
    const results = $("#results");
    $$("article.result-card", results).forEach((n) => n.remove());
    for (const r of state.results) {
      try { r.treatment = (await api(`/api/treatment/${encodeURIComponent(r.prediction.class_raw)}?lang=${i18n.lang}`)).treatment; } catch (e) { /* keep old */ }
    }
    const bar = $("#exportBar");
    [...state.results].reverse().forEach((r) => results.insertBefore(resultCard(r), bar ? bar.nextSibling : results.firstChild));
    if (state.compare) results.insertBefore(compareCard(state.compare), bar ? bar.nextSibling : results.firstChild);
    renderExportBar();
  }

  // ------------------------------------------------------------------ init
  document.addEventListener("DOMContentLoaded", () => {
    loadModels();
    $("#modelSelect").addEventListener("change", renderModelHint);
    segmented("explanationSeg", (v) => { state.explanation = v; });
    segmented("modeSeg", (v) => { state.mode = v; renderModeHint(); });
    const dz = $("#dropzone");
    const input = $("#fileInput");
    dz.addEventListener("click", () => input.click());
    dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
    const cam = $("#cameraInput");
    $("#cameraBtn").addEventListener("click", () => cam.click());
    cam.addEventListener("change", () => { addFiles(cam.files); cam.value = ""; });
    ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
    document.addEventListener("paste", (e) => { const items = [...(e.clipboardData || {}).files || []]; if (items.length) addFiles(items); });
    $("#analyzeBtn").addEventListener("click", analyze);
    $("#compareBtn").addEventListener("click", compare);
    $("#clearBtn").addEventListener("click", () => {
      state.files.forEach((f) => URL.revokeObjectURL(f.url));
      state.files = []; state.results = []; state.compare = null;
      renderPreviews();
      $$("#results > *:not(#emptyState)").forEach((n) => n.remove());
      $("#emptyState").classList.remove("hidden");
    });
    document.addEventListener("langchange", rerender);
  });
})();
