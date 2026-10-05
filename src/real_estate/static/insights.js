/* Render prepared server aggregates and saved model diagnostics; no ML preprocessing here. */
(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const colours = { ink: "#183734", muted: "#657370", green: "#176455", teal: "#64a293", gold: "#bf9559", rose: "#a86865", grid: "#edf1ec" };
  const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const decimal = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
  const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
  const usdDetail = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
  const numericLabel = (value, formatter) => isNumber(value) ? formatter.format(value) : "—";
  const listingNoun = (count) => count === 1 ? "listing" : "listings";
  // Plotly supports a subset of HTML in labels. Escape all labels originating in records.
  const plotText = (value) => String(value ?? "Not reported").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  const featureNames = { surface_total: "Total area (m²)", bedrooms: "Bedrooms", bathrooms: "Bathrooms", lat: "Latitude", lon: "Longitude", price_usd: "Asking price (USD)", property_type: "Property type", l2: "Region", l4: "Locality" };
  const modelNames = { dummy: "Median baseline", ridge: "Ridge", hist_gradient_boosting: "Histogram Gradient Boosting", random_forest: "Random Forest" };
  const modelLabel = (name) => modelNames[name] || name || "Selected model";
  const allChartIds = ["chart-price", "chart-area", "chart-types", "chart-regions", "chart-region-prices", "chart-scatter", "chart-correlation", "chart-predictions", "chart-residuals", "chart-residual-distribution", "chart-importance"];
  const exploratoryIds = allChartIds.slice(0, 7);
  const state = { data: null, controller: null, revision: 0 };
  const plotConfig = { responsive: true, displaylogo: false, scrollZoom: false, modeBarButtonsToRemove: ["lasso2d", "select2d"], toImageButtonOptions: { format: "png", scale: 2 } };

  function baseLayout(extra = {}) {
    return {
      font: { family: '"Segoe UI", Arial, sans-serif', size: 11, color: colours.ink },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 65, r: 22, t: 15, b: 60 },
      hoverlabel: { bgcolor: "#ffffff", bordercolor: "#dce5df", font: { size: 12, color: colours.ink }, namelength: -1 },
      xaxis: { automargin: true, gridcolor: colours.grid, zeroline: false, tickfont: { color: colours.muted }, title: { standoff: 12 }, autorange: true },
      yaxis: { automargin: true, gridcolor: colours.grid, zeroline: false, tickfont: { color: colours.muted }, title: { standoff: 12 }, autorange: true },
      legend: { orientation: "h", x: 0, y: 1.1, font: { size: 11 }, bgcolor: "rgba(0,0,0,0)" },
      modebar: { orientation: "h", bgcolor: "rgba(255,255,255,0.94)", color: "#789185", activecolor: colours.green },
      uirevision: String(state.revision), ...extra,
    };
  }

  function renderPlot(id, traces, extra = {}) {
    const target = byId(id);
    target.querySelectorAll(".insights-plot-empty").forEach((placeholder) => placeholder.remove());
    const layout = baseLayout(extra);
    // Preserve common styling when a chart supplies a more specific axis.
    layout.xaxis = { ...baseLayout().xaxis, ...(extra.xaxis || {}) };
    layout.yaxis = { ...baseLayout().yaxis, ...(extra.yaxis || {}) };
    return window.Plotly.react(target, traces, layout, plotConfig);
  }

  function clearChart(id, message = "No chart data available") {
    const target = byId(id);
    if (window.Plotly) window.Plotly.purge(target);
    target.classList.remove("js-plotly-plot");
    const note = document.createElement("p");
    note.className = "insights-plot-empty";
    note.textContent = message;
    target.replaceChildren(note);
  }

  function setStatus(message, error = false) {
    const element = byId("insights-status");
    element.textContent = message;
    element.className = error ? "status-message error-message" : "status-message";
  }

  function summary(data) {
    const values = data.summary || {};
    byId("summary-listings").textContent = numericLabel(values.listings, integer);
    byId("summary-regions").textContent = numericLabel(values.regions, integer);
    byId("summary-price").textContent = numericLabel(values.median_price_usd, usd);
    byId("summary-area").textContent = isNumber(values.median_area_m2) ? `${decimal.format(values.median_area_m2)} m²` : "—";
    byId("summary-casa").textContent = numericLabel(values.casa, integer);
    byId("summary-departamento").textContent = numericLabel(values.departamento, integer);
    byId("overview-scope").textContent = `${integer.format(values.listings || 0)} matching retained ${listingNoun(values.listings)}. Summaries use all matching observations; the relationship scatter may use a bounded sample.`;
  }

  function histogram(id, edges, counts, title, dollars = false, log = false) {
    if (!Array.isArray(edges) || !Array.isArray(counts) || !counts.length) {
      clearChart(id, "No matching observations");
      return Promise.resolve();
    }
    const centres = [], widths = [], ranges = [];
    for (let i = 0; i < counts.length; i += 1) {
      const low = edges[i], high = edges[i + 1];
      centres.push((low + high) / 2);
      widths.push(high - low);
      const formatter = dollars ? usd : decimal;
      ranges.push(`${formatter.format(low)} – ${formatter.format(high)}${dollars ? " USD" : " m²"}`);
    }
    return renderPlot(id, [{ type: "bar", x: centres, y: counts, width: widths, marker: { color: colours.green, line: { color: "#ffffff", width: 0.7 } }, customdata: ranges, hovertemplate: "%{customdata}<br>Listings: %{y:,}<extra></extra>" }], {
      bargap: 0.03,
      xaxis: { title: { text: title }, type: log ? "log" : "linear", tickformat: dollars ? "$~s" : "~s" },
      yaxis: { title: { text: "Listing count" }, rangemode: "tozero", tickformat: ",d" },
      uirevision: `${state.revision}-${log ? "log" : "linear"}`,
    });
  }

  function priceHistogram() {
    const values = state.data.price_distribution || {};
    const log = byId("price-scale").value === "log";
    return histogram("chart-price", log ? values.log_edges : values.edges, log ? values.log_counts : values.counts, "Asking price (USD)", true, log);
  }

  function propertyTypes(data) {
    const items = data.property_types || [];
    return renderPlot("chart-types", [{ type: "bar", x: items.map((item) => plotText(item.property_type)), y: items.map((item) => item.count), marker: { color: items.map((item) => item.property_type === "Casa" ? colours.green : colours.gold) }, text: items.map((item) => integer.format(item.count)), textposition: "outside", cliponaxis: false, hovertemplate: "%{x}<br>Listings: %{y:,}<extra></extra>" }], { xaxis: { title: { text: "Property type" } }, yaxis: { title: { text: "Listing count" }, rangemode: "tozero", tickformat: ",d" }, bargap: 0.55, margin: { l: 65, r: 20, t: 30, b: 55 } });
  }

  function regions(data) {
    const items = data.regions || [];
    const counts = [...items].sort((a, b) => b.count - a.count);
    const prices = [...items].filter((item) => isNumber(item.median_price_usd)).sort((a, b) => b.median_price_usd - a.median_price_usd);
    const setHeight = (id, rows) => { byId(id).style.height = `${Math.max(390, rows * 23 + 105)}px`; };
    setHeight("chart-regions", counts.length);
    setHeight("chart-region-prices", prices.length);
    return Promise.all([
      renderPlot("chart-regions", [{ type: "bar", orientation: "h", x: counts.map((item) => item.count), y: counts.map((item) => plotText(item.region)), marker: { color: colours.green }, hovertemplate: "%{y}<br>Listings: %{x:,}<extra></extra>" }], { xaxis: { title: { text: "Listing count" }, tickformat: ",d", rangemode: "tozero" }, yaxis: { type: "category", autorange: "reversed", gridcolor: "rgba(0,0,0,0)" }, margin: { l: 110, r: 20, t: 15, b: 60 } }),
      renderPlot("chart-region-prices", [{ type: "bar", orientation: "h", x: prices.map((item) => item.median_price_usd), y: prices.map((item) => plotText(item.region)), customdata: prices.map((item) => item.count), marker: { color: colours.teal }, hovertemplate: "%{y}<br>Median asking price: $%{x:,.0f} USD<br>Listings: %{customdata:,}<extra></extra>" }], { xaxis: { title: { text: "Median asking price (USD)" }, tickformat: "$~s", rangemode: "tozero" }, yaxis: { type: "category", autorange: "reversed", gridcolor: "rgba(0,0,0,0)" }, margin: { l: 110, r: 20, t: 15, b: 60 } }),
    ]);
  }

  function relationshipScatter() {
    const scatter = state.data.scatter || {};
    const points = scatter.points || [];
    const log = byId("scatter-scale").value === "log";
    byId("scatter-source").textContent = scatter.sampled ? `${integer.format(scatter.shown || points.length)} of ${integer.format(scatter.total)} listings · deterministic sample, including extrema` : `All ${integer.format(scatter.total || points.length)} matching ${listingNoun(scatter.total || points.length)}`;
    const traces = ["Casa", "Departamento"].map((propertyType) => {
      const group = points.filter((point) => point.property_type === propertyType);
      return { type: "scattergl", mode: "markers", name: propertyType, x: group.map((point) => point.surface_total), y: group.map((point) => point.price_usd), marker: { color: propertyType === "Casa" ? colours.green : colours.gold, size: 6, opacity: 0.65 }, customdata: group.map((point) => [plotText(point.property_type), plotText(point.l2), plotText(point.l4)]), hovertemplate: "%{customdata[0]}<br>Asking price: $%{y:,.0f} USD<br>Total area: %{x:,.1f} m²<br>Region: %{customdata[1]}<br>Locality: %{customdata[2]}<extra></extra>" };
    });
    return renderPlot("chart-scatter", traces, { xaxis: { title: { text: "Total area (m²)" }, type: log ? "log" : "linear", tickformat: "~s" }, yaxis: { title: { text: "Asking price (USD)" }, type: log ? "log" : "linear", tickformat: "$~s" }, margin: { l: 70, r: 22, t: 45, b: 60 }, uirevision: `${state.revision}-${log ? "log" : "linear"}` });
  }

  function correlation(data) {
    const values = data.correlation || {};
    const variables = values.variables || [];
    const labels = variables.map((key) => plotText(featureNames[key] || key));
    if (!variables.length) { clearChart("chart-correlation"); return Promise.resolve(); }
    const matrix = (values.matrix || []).map((row) => row.map((value) => isNumber(value) ? value : null));
    const nullX = [], nullY = [], nullCounts = [];
    matrix.forEach((row, rowIndex) => row.forEach((value, colIndex) => {
      if (value === null) { nullX.push(labels[colIndex]); nullY.push(labels[rowIndex]); nullCounts.push((values.pair_counts || [])[rowIndex]?.[colIndex] ?? 0); }
    }));
    const traces = [{ type: "heatmap", x: labels, y: labels, z: matrix, customdata: values.pair_counts || [], zmin: -1, zmax: 1, zmid: 0, colorscale: [[0, colours.rose], [0.5, "#f5f7f2"], [1, colours.green]], colorbar: { title: { text: "r" }, thickness: 10, len: 0.8, outlinewidth: 0, tickvals: [-1, -0.5, 0, 0.5, 1] }, hoverongaps: false, xgap: 2, ygap: 2, hovertemplate: "%{y} × %{x}<br>Pearson r: %{z:.3f}<br>Paired observations: %{customdata:,}<extra></extra>" }];
    if (nullX.length) traces.push({ type: "scatter", mode: "text", x: nullX, y: nullY, text: nullX.map(() => "—"), textfont: { color: "#9ba9a1", size: 15 }, customdata: nullCounts, showlegend: false, hovertemplate: "%{y} × %{x}<br>Correlation: undefined<br>Paired observations: %{customdata:,}<extra></extra>" });
    return renderPlot("chart-correlation", traces, { xaxis: { tickangle: -35, type: "category", showgrid: false }, yaxis: { type: "category", autorange: "reversed", showgrid: false }, margin: { l: 115, r: 40, t: 15, b: 100 } });
  }

  function testPairs(evaluation) {
    const predictions = evaluation.predictions || {};
    return (predictions.actual_usd || []).map((actual, index) => ({ actual, predicted: (predictions.predicted_usd || [])[index], residual: (predictions.residual_usd || [])[index] })).filter((point) => isNumber(point.actual) && isNumber(point.predicted) && isNumber(point.residual));
  }

  function predictedActual() {
    const evaluation = state.data.evaluation || {};
    const points = testPairs(evaluation);
    if (!points.length) { clearChart("chart-predictions", "Saved test predictions are unavailable"); return Promise.resolve(); }
    const allPositive = points.every((point) => point.actual > 0 && point.predicted > 0);
    const option = byId("prediction-scale").querySelector('option[value="log"]');
    option.disabled = !allPositive;
    if (!allPositive) byId("prediction-scale").value = "linear";
    const log = byId("prediction-scale").value === "log";
    const extent = points.reduce((range, point) => ({ low: Math.min(range.low, point.actual, point.predicted), high: Math.max(range.high, point.actual, point.predicted) }), { low: Infinity, high: -Infinity });
    const low = log ? extent.low : Math.min(0, extent.low);
    const high = extent.high;
    byId("prediction-source").textContent = `${modelLabel(evaluation.selected_model)} · ${integer.format(points.length)} final test predictions${!allPositive ? " · log axes unavailable for nonpositive values" : ""}`;
    return renderPlot("chart-predictions", [
      { type: "scattergl", mode: "markers", name: "Final test predictions", x: points.map((point) => point.actual), y: points.map((point) => point.predicted), marker: { color: colours.green, size: 5, opacity: 0.6 }, hovertemplate: "Actual: $%{x:,.0f} USD<br>Predicted: $%{y:,.0f} USD<extra></extra>" },
      { type: "scatter", mode: "lines", name: "Perfect prediction", x: [low, high], y: [low, high], line: { color: colours.gold, dash: "dash", width: 1.5 }, hoverinfo: "skip" },
    ], { xaxis: { title: { text: "Actual asking price (USD)" }, type: log ? "log" : "linear", tickformat: "$~s" }, yaxis: { title: { text: "Predicted asking price (USD)" }, type: log ? "log" : "linear", tickformat: "$~s" }, margin: { l: 72, r: 20, t: 45, b: 60 }, uirevision: `${state.revision}-${log ? "log" : "linear"}` });
  }

  function residuals(evaluation) {
    const points = testPairs(evaluation);
    if (!points.length) { clearChart("chart-residuals", "Saved test residuals are unavailable"); clearChart("chart-residual-distribution", "Saved test residuals are unavailable"); return Promise.resolve(); }
    const predicted = points.map((point) => point.predicted);
    const values = points.map((point) => point.residual);
    return Promise.all([
      renderPlot("chart-residuals", [{ type: "scattergl", mode: "markers", x: predicted, y: values, marker: { color: colours.green, size: 5, opacity: 0.6 }, customdata: points.map((point) => point.actual), hovertemplate: "Predicted: $%{x:,.0f} USD<br>Actual: $%{customdata:,.0f} USD<br>Residual: $%{y:,.0f} USD<extra></extra>" }], { xaxis: { title: { text: "Predicted asking price (USD)" }, tickformat: "$~s" }, yaxis: { title: { text: "Actual − predicted (USD)" }, tickformat: "$~s" }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: colours.gold, width: 1.5, dash: "dash" } }], margin: { l: 76, r: 20, t: 15, b: 60 } }),
      renderPlot("chart-residual-distribution", [{ type: "histogram", x: values, nbinsx: 45, marker: { color: colours.teal, line: { color: "#ffffff", width: 0.7 } }, hovertemplate: "Residual bin: $%{x:,.0f} USD<br>Test observations: %{y:,}<extra></extra>" }], { xaxis: { title: { text: "Residual = actual − predicted (USD)" }, tickformat: "$~s" }, yaxis: { title: { text: "Test observation count" }, rangemode: "tozero", tickformat: ",d" }, shapes: [{ type: "line", yref: "paper", y0: 0, y1: 1, x0: 0, x1: 0, line: { color: colours.gold, width: 1.5, dash: "dash" } }] }),
    ]);
  }

  function importance(evaluation) {
    const items = [...(evaluation.importance || [])].filter((item) => isNumber(item.mean_mae_increase_usd)).sort((a, b) => b.mean_mae_increase_usd - a.mean_mae_increase_usd);
    byId("importance-content").hidden = !items.length;
    byId("importance-status").textContent = items.length ? "" : "Validation permutation scores are unavailable in the current artifacts.";
    if (!items.length) { clearChart("chart-importance"); return Promise.resolve(); }
    return renderPlot("chart-importance", [{ type: "bar", orientation: "h", x: items.map((item) => item.mean_mae_increase_usd), y: items.map((item) => plotText(featureNames[item.feature] || item.feature)), error_x: { type: "data", array: items.map((item) => isNumber(item.std_mae_increase_usd) ? item.std_mae_increase_usd : 0), color: "#47685c", thickness: 1, width: 4 }, customdata: items.map((item) => item.std_mae_increase_usd), marker: { color: items.map((item) => item.mean_mae_increase_usd >= 0 ? colours.green : colours.rose) }, hovertemplate: "%{y}<br>Mean MAE increase: $%{x:,.2f} USD<br>Standard deviation: $%{customdata:,.2f} USD<extra></extra>" }], { xaxis: { title: { text: "Increase in validation MAE (USD)" }, tickformat: "$~s", zeroline: true, zerolinecolor: "#a5b4aa", zerolinewidth: 1 }, yaxis: { type: "category", autorange: "reversed", showgrid: false }, margin: { l: 125, r: 35, t: 15, b: 60 } });
  }

  function modelEvaluation(data) {
    const evaluation = data.evaluation || {};
    const available = evaluation.available === true;
    byId("evaluation-content").hidden = !available;
    byId("evaluation-status").textContent = available ? "" : (evaluation.reason || "Verified model diagnostics are unavailable. Follow the repository instructions to train and prepare local artifacts.");
    if (!available) {
      allChartIds.slice(7).forEach((id) => clearChart(id));
      byId("importance-content").hidden = true;
      byId("importance-status").textContent = "Validation importance is unavailable without verified model diagnostics.";
      return Promise.resolve();
    }
    const metrics = evaluation.held_out_test || {};
    byId("evaluation-mae").textContent = numericLabel(metrics.mae_usd, usdDetail);
    byId("evaluation-rmse").textContent = numericLabel(metrics.rmse_usd, usdDetail);
    byId("evaluation-r2").textContent = isNumber(metrics.r2) ? metrics.r2.toFixed(4) : "—";
    const rows = evaluation.split?.rows || {};
    const rowDescription = ["train", "validation", "test"].filter((name) => isNumber(rows[name])).map((name) => `${integer.format(rows[name])} ${name}`).join(" / ");
    const limitedFit = isNumber(metrics.r2) && metrics.r2 < 0.5 ? "The final test fit is limited. " : "";
    byId("evaluation-context").textContent = `${modelLabel(evaluation.selected_model)}${rowDescription ? ` · ${rowDescription} observations.` : "."} ${limitedFit}Extreme asking prices can strongly affect errors and RMSE. These historical diagnostics do not establish reliable current-market valuation.`;
    return Promise.all([predictedActual(), residuals(evaluation), importance(evaluation)]);
  }

  async function apply(params) {
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;
    byId("exploratory-charts").setAttribute("aria-busy", "true");
    setStatus("Updating the selected listings…");
    try {
      const response = await fetch(`/api/insights?${params.toString()}`, { signal: controller.signal, cache: "no-store", headers: { Accept: "application/json" } });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : "Prepared analytics are unavailable. Check the repository's local data preparation instructions.");
      if (controller.signal.aborted) return;
      state.data = data;
      state.revision += 1;
      summary(data);
      const empty = !data.summary?.listings;
      byId("insights-empty").hidden = !empty;
      byId("exploratory-charts").hidden = empty;
      if (empty) exploratoryIds.forEach((id) => clearChart(id, "No matching listings"));
      else await Promise.all([priceHistogram(), histogram("chart-area", data.area_distribution?.edges, data.area_distribution?.counts, "Total area (m²)"), propertyTypes(data), regions(data), relationshipScatter(), correlation(data)]);
      if (controller.signal.aborted) return;
      await modelEvaluation(data);
      if (controller.signal.aborted) return;
      setStatus(empty ? "No listings match these filters. Model diagnostics remain the fixed independent evaluation." : `${integer.format(data.summary.listings)} retained ${listingNoun(data.summary.listings)} ${data.summary.listings === 1 ? "matches" : "match"} the current filters. Model diagnostics are independent of this selection.`);
    } catch (error) {
      if (error.name === "AbortError" || controller.signal.aborted) return;
      state.data = null;
      ["summary-listings", "summary-regions", "summary-price", "summary-area", "summary-casa", "summary-departamento"].forEach((id) => { byId(id).textContent = "—"; });
      byId("overview-scope").textContent = "Prepared data could not be loaded for this selection.";
      byId("insights-empty").hidden = true;
      byId("exploratory-charts").hidden = true;
      byId("evaluation-content").hidden = true;
      byId("importance-content").hidden = true;
      byId("evaluation-status").textContent = "Model diagnostics could not be loaded. No saved values are displayed.";
      byId("importance-status").textContent = "Validation permutation scores could not be loaded.";
      allChartIds.forEach((id) => clearChart(id));
      setStatus(error.message || "Unable to load analytics. Please try again.", true);
    } finally {
      if (state.controller === controller) byId("exploratory-charts").setAttribute("aria-busy", "false");
    }
  }

  async function start() {
    if (!window.Plotly || !window.ExplorerFilters) {
      setStatus("A local dashboard script could not be loaded. Check that the repository's static assets are installed.", true);
      byId("evaluation-status").textContent = "Model diagnostics could not be loaded.";
      byId("importance-status").textContent = "Validation permutation scores could not be loaded.";
      return;
    }
    const form = byId("listing-filters");
    window.ExplorerFilters.bind(form, apply);
    byId("price-scale").addEventListener("change", () => { if (state.data) priceHistogram(); });
    byId("scatter-scale").addEventListener("change", () => { if (state.data) relationshipScatter(); });
    byId("prediction-scale").addEventListener("change", () => { if (state.data) predictedActual(); });
    try { await window.ExplorerFilters.loadOptions(form); } catch (_) { /* The insights endpoint reports the authoritative missing-data error. */ }
    await apply(window.ExplorerFilters.params(form));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
