(function () {
  const { el, $, clear, api, fmt, toast, modelColor, ring, countUp, initReveal } = App;
  let data = null;

  function render() {
    if (!data) return;
    const { stats, models, research } = data;
    countUp($("#kpiImages"), stats.dataset.total_images);
    countUp($("#kpiClasses"), stats.total_classes);
    countUp($("#kpiPlants"), stats.dataset.plant_types);
    const best = stats.best_model && stats.statistics[stats.best_model];
    $("#kpiBest").textContent = best ? fmt.pct(best.accuracy, 2) : t("common.na");
    const bestInfo = models.models.find((m) => m.name === stats.best_model);
    $("#kpiBestModel").textContent = bestInfo ? fmt.modelLabel(bestInfo) : "";
    $("#version").textContent = "v" + (data.health.version || "");
    $("#fieldStatus").textContent = models.field_model.available ? t("home.field_available") : t("home.field_missing");

    const cards = clear($("#modelCards"));
    models.models.forEach((m) => {
      const chip = m.trained
        ? el("span", { class: "chip good", text: m.name === stats.best_model ? t("common.best") : fmt.modelTag(m) })
        : el("span", { class: "chip warning", text: t("common.needs_training") });
      const color = modelColor(m.name);
      cards.append(
        el("div", { class: "card model-card lift reveal", style: { "--model-color": color } }, [
          el("div", { class: "row between" }, [el("h3", {}, [el("span", { class: "swatch" }), fmt.modelLabel(m)]), chip]),
          el("div", { class: "ring-row" }, [
            ring(m.accuracy || 0, color, { label: m.accuracy === null || m.accuracy === undefined ? "—" : `${(m.accuracy * 100).toFixed(1)}%` }),
            el("div", { class: "kv" }, [
              el("div", {}, [t("common.f1"), el("strong", { text: fmt.pct(m.f1_score, 2) })]),
              el("div", {}, [t("common.inference"), el("strong", { text: fmt.ms(m.inference_time_ms) })]),
            ]),
          ]),
          el("p", { class: "desc", text: fmt.modelDesc(m) }),
          el("div", { class: "metrics" }, [
            el("div", {}, [t("common.params"), el("strong", { text: m.parameters || t("common.na") })]),
            el("div", {}, [t("common.size"), el("strong", { text: fmt.mb(m.size_mb) })]),
          ]),
        ])
      );
    });

    const tiles = clear($("#researchTiles"));
    const methods = research && research.summary && research.summary.methods;
    if (methods) {
      const base = methods.baseline && methods.baseline.eval_open;
      const bestMethod = Object.entries(methods).filter(([k]) => k !== "baseline").sort((a, b) => b[1].eval_open.accuracy_mean - a[1].eval_open.accuracy_mean)[0];
      tiles.append(el("div", { class: "card tile" }, [el("div", { class: "label", text: t("stats.zero_shot") }), el("div", { class: "value", text: base ? fmt.pct(base.accuracy_mean) : "—" })]));
      if (bestMethod) tiles.append(el("div", { class: "card tile" }, [el("div", { class: "label", text: t("stats.adapted") }), el("div", { class: "value", text: fmt.pct(bestMethod[1].eval_open.accuracy_mean) }), el("div", { class: "delta", text: bestMethod[1].label })]));
    } else {
      tiles.append(el("div", { class: "card tile" }, [el("div", { class: "label", text: t("stats.tab_research") }), el("div", { class: "muted small", text: t("stats.research_pending") })]));
    }
    initReveal();
  }

  async function load() {
    try {
      const [health, stats, models, research] = await Promise.all([api("/api/health"), api("/api/stats"), api("/api/models"), api("/api/research").catch(() => null)]);
      data = { health, stats, models, research };
      render();
    } catch (e) {
      toast(e.message, "error", t("common.error"));
    }
  }
  document.addEventListener("DOMContentLoaded", load);
  document.addEventListener("langchange", render);
})();
