/* File: static/js/scripts.js */

let currentFilePath = null;
let chartInstance = null;

// ===== Elements =====
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadAlert = document.getElementById("upload-alert");

const tickerInput = document.getElementById("ticker-input");
const downloadBtn = document.getElementById("download-btn");
const downloadAlert = document.getElementById("download-alert");

const modelSection = document.getElementById("model-section");
const modelSelect = document.getElementById("model-select");
const timeStepInput = document.getElementById("time-step-input");
const retrainCheckbox = document.getElementById("retrain-checkbox");

const trainBtn = document.getElementById("train-btn");
const predictBtn = document.getElementById("predict-btn");
const trainAlert = document.getElementById("train-alert");
const trainingProgress = document.getElementById("training-progress");

const predictionText = document.getElementById("prediction-text");
const chartCanvas = document.getElementById("train-test-plot");
const errorBlock = document.getElementById("error-block");

// Evaluate & Verify
const evaluateBtn = document.getElementById("evaluate-btn");
const evaluateAlert = document.getElementById("evaluate-alert");
const evalResults = document.getElementById("eval-results");
const evalTable = document.getElementById("eval-table");

const verifyBtn = document.getElementById("verify-btn");
const verifyAlert = document.getElementById("verify-alert");
const verifyOutput = document.getElementById("verify-output");

// === Model comparison elements ===
const compareSection = document.getElementById("compare-section");
const compareOverlayBtn = document.getElementById("compare-overlay-btn");
const compareMetricsBtn = document.getElementById("compare-metrics-btn");
const compareAlert = document.getElementById("compare-alert");
const compareError = document.getElementById("compare-error");

const overlayCanvas = document.getElementById("compare-overlay-chart");
const metricsCanvas = document.getElementById("metrics-bar-chart");
const metricsNote = document.getElementById("metrics-note");

let overlayChart = null;
let metricsChart = null;

// Same-origin by default; set window.API_BASE in HTML if front-end is served elsewhere
const API_BASE = window.API_BASE || "";

// ===== Socket.IO (training progress) =====
let socket;
try {
  socket = io(API_BASE || undefined);
  socket.on("training_progress", (data) => {
    if (!trainingProgress) return;
    trainingProgress.style.display = "block";
    trainingProgress.value = data?.progress ?? 0;
  });
} catch (e) {
  console.warn("Socket.IO not available:", e);
}

// ===== Helpers =====
function show(el, text) {
  if (!el) return;
  if (typeof text === "string") el.textContent = text;
  el.style.display = "";
}
function hide(el) {
  if (!el) return;
  el.style.display = "none";
}
function setPredictionHTML(html) {
  if (!predictionText) return;
  predictionText.innerHTML = html || "";
}
function urlEncode(obj) {
  return new URLSearchParams(obj);
}
async function tryPOST(url, bodyParams) {
  const res = await fetch(API_BASE + url, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: typeof bodyParams === "string" ? bodyParams : urlEncode(bodyParams),
  });
  let json = {};
  try { json = await res.json(); } catch (_) {}
  if (!res.ok || json.error) {
    const err = new Error(json.error || `HTTP ${res.status}`);
    err.status = res.status;
    err.payload = json;
    throw err;
  }
  return json;
}

function getSelectedModel() {
  return (modelSelect?.value || "lstm").trim().toLowerCase();
}
function getTimeStep() {
  const v = parseInt(timeStepInput?.value ?? "60", 10);
  return Number.isFinite(v) && v >= 2 ? v : 60;
}
function getRetrainFlag(defaultVal = true) {
  if (!retrainCheckbox) return defaultVal;
  return !!retrainCheckbox.checked;
}

function showError(msg) {
  if (!errorBlock) { alert(msg); return; }
  errorBlock.style.display = "block";
  errorBlock.textContent = msg || "Something went wrong";
}
function clearError() {
  if (!errorBlock) return;
  errorBlock.style.display = "none";
  errorBlock.textContent = "";
}

// ===== Charting =====
function ensureChart() {
  if (!chartCanvas) return;
  if (chartInstance) chartInstance.destroy();
  const ctx = chartCanvas.getContext("2d");
  chartInstance = new Chart(ctx, {
    type: "line",
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: true, position: "top" } },
      scales: {
        x: { title: { display: true, text: "Index (oldest → newest)" } },
        y: { title: { display: true, text: "Value" }, beginAtZero: false },
      },
    },
  });
}

function displayChart(graphData, modelKey = "lstm") {
  if (!chartCanvas || !graphData) return;

  const isLogit = modelKey === "logit";
  const trueArr = (graphData.true_prices || []).map(Number);
  const trainPredict = (graphData.train_predict || []).map(Number);
  const testPredict = (graphData.test_predict || []).map(Number);
  const ts = Number(graphData.time_step || 60);

  const labels = Array.from({ length: trueArr.length }, (_, i) => i + 1);

  // Null-padded alignment
  const trainPlot = new Array(trueArr.length).fill(null);
  const testPlot = new Array(trueArr.length).fill(null);

  const trainStart = Math.min(ts, trueArr.length);
  for (let i = 0; i < trainPredict.length; i++) {
    const idx = trainStart + i;
    if (idx < trainPlot.length) trainPlot[idx] = trainPredict[i];
  }

  const testStart = ts + trainPredict.length;
  for (let i = 0; i < testPredict.length; i++) {
    const idx = testStart + i;
    if (idx < testPlot.length) testPlot[idx] = testPredict[i];
  }

  ensureChart();

  chartInstance.data.labels = labels;
  chartInstance.data.datasets = [
    {
      label: isLogit ? "True Close (ref)" : "True Prices",
      data: trueArr,
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      tension: 0.1,
    },
    {
      label: isLogit ? "Train Prob (Up)" : "Training Predictions",
      data: trainPlot,
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      tension: 0.1,
      borderDash: [6, 4],
    },
    {
      label: isLogit ? "Test Prob (Up)" : "Testing Predictions",
      data: testPlot,
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      tension: 0.1,
    },
  ];

  if (isLogit) {
    chartInstance.options.scales.y.beginAtZero = true;
    chartInstance.options.scales.y.suggestedMax = 1;
    chartInstance.options.scales.y.title.text = "Probability (Up)";
  } else {
    chartInstance.options.scales.y.beginAtZero = false;
    chartInstance.options.scales.y.suggestedMax = undefined;
    chartInstance.options.scales.y.title.text = "Price";
  }

  chartInstance.update();
}

function renderPrediction(modelKey, predictions) {
  if (!predictions) { setPredictionHTML(""); return; }

  if (modelKey === "logit") {
    const dir = predictions.next_day_direction || "?";
    const p = predictions.next_day_proba != null ? (predictions.next_day_proba * 100).toFixed(1) : "—";
    const acc = predictions.accuracy != null ? (predictions.accuracy * 100).toFixed(1) : null;
    setPredictionHTML(
      `<strong>Direction:</strong> ${dir} &nbsp; <strong>P(Up):</strong> ${p}%` +
      (acc ? ` &nbsp; <strong>Acc:</strong> ${acc}%` : "")
    );
  } else {
    const price = predictions.next_day != null ? Number(predictions.next_day).toFixed(2) : "—";
    const rmse = predictions.rmse != null && !Number.isNaN(predictions.rmse)
      ? Number(predictions.rmse).toFixed(2) : null;
    setPredictionHTML(
      `<strong>Next Day Price:</strong> ${price}` +
      (rmse ? ` &nbsp; <strong>RMSE:</strong> ${rmse}` : "")
    );
  }
}

// ===== Upload CSV =====
if (uploadForm) {
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();
    if (!fileInput?.files?.length) {
      showError("Please choose a CSV file.");
      return;
    }
    show(uploadAlert, "Uploading...");

    try {
      const fd = new FormData();
      fd.append("file", fileInput.files[0]);

      const res = await fetch(API_BASE + "/uploads", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "Upload failed");

      hide(uploadAlert);
      currentFilePath = data.file_path;

      if (modelSection) modelSection.style.display = "";

      // Auto fast-predict with current selection
      await predictFast();
    } catch (err) {
      hide(uploadAlert);
      showError(`Upload error: ${err.message || err}`);
    }
  });
}

// ===== Download by ticker =====
if (downloadBtn) {
  downloadBtn.addEventListener("click", async () => {
    clearError();
    const ticker = (tickerInput?.value || "").trim();
    if (!ticker) return showError("Please enter a stock ticker.");

    show(downloadAlert, "Downloading...");
    try {
      const data = await tryPOST("/download", { ticker });
      hide(downloadAlert);
      currentFilePath = data.file_path;

      if (modelSection) modelSection.style.display = "";

      await predictFast();
    } catch (err) {
      hide(downloadAlert);
      showError(`Download error: ${err.message || err}`);
    }
  });
}

// ===== Train (selected model) =====
if (trainBtn) {
  trainBtn.addEventListener("click", async () => {
    clearError();
    if (!currentFilePath) return showError("No file selected for training.");

    const model = getSelectedModel();
    const time_step = getTimeStep();
    const retrain = getRetrainFlag(true) ? "1" : "0";

    show(trainAlert, "Training...");
    if (trainingProgress) {
      trainingProgress.value = 0;
      trainingProgress.style.display = "block";
    }

    try {
      const data = await tryPOST("/train", {
        file_path: currentFilePath,
        model,
        time_step,
        retrain,
      });

      hide(trainAlert);
      if (trainingProgress) trainingProgress.style.display = "none";

      renderPrediction(model, data?.predictions);
      displayChart(data?.graph_data, model);
    } catch (err) {
      hide(trainAlert);
      if (trainingProgress) trainingProgress.style.display = "none";
      showError(`Training error: ${err.message || err}`);
    }
  });
}

// ===== Predict (fast path, no retrain) =====
if (predictBtn) {
  predictBtn.addEventListener("click", async () => {
    await predictFast();
  });
}

async function predictFast() {
  clearError();
  if (!currentFilePath) return showError("Please upload or download a CSV first.");

  const model = getSelectedModel();
  const time_step = getTimeStep();

  try {
    const data = await tryPOST("/predict", {
      file_path: currentFilePath,
      model,
      time_step,
      retrain: "0",
    });

    renderPrediction(model, data?.predictions);
    displayChart(
      data?.graph_data || { true_prices: [], train_predict: [], test_predict: [], time_step },
      model
    );
  } catch (err) {
    if (err.status === 404) {
      console.warn("Fast /predict not available. You can still Train.");
      return;
    }
    showError(`Predict error: ${err.message || err}`);
  }
}

// ===== Evaluate All Models =====
if (evaluateBtn) {
  evaluateBtn.addEventListener("click", async () => {
    clearError();
    if (!currentFilePath) return showError("Please upload or download a CSV first.");
    const time_step = getTimeStep();

    show(evaluateAlert, "Evaluating...");
    hide(evalResults);

    try {
      const payload = await tryPOST("/evaluate", {
        file_path: currentFilePath,
        time_step,
        retrain: "0"
        // omit "models" to evaluate all registered models
      });

      hide(evaluateAlert);
      buildEvaluationTable(payload?.results || {});
      show(evalResults);
    } catch (err) {
      hide(evaluateAlert);
      showError(`Evaluate error: ${err.message || err}`);
    }
  });
}

function buildEvaluationTable(results) {
  if (!evalTable) return;
  const thead = evalTable.querySelector("thead");
  const tbody = evalTable.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  // Collect metric keys across models
  const rows = [];
  const metricsSet = new Set();

  // Normalize results into rows
  Object.entries(results).forEach(([modelKey, obj]) => {
    const type = obj?.type || "";
    let metrics = obj?.metrics || {};
    const row = { model: modelKey.toUpperCase(), type, metrics };
    rows.push(row);

    Object.keys(metrics).forEach(k => metricsSet.add(k));
  });

  const metricKeys = Array.from(metricsSet);
  // Header
  const h = document.createElement("tr");
  h.innerHTML =
    `<th>Model</th><th>Type</th>` +
    metricKeys.map(k => `<th>${k.toUpperCase()}</th>`).join("");
  thead.appendChild(h);

  // Determine best values per metric (min for error metrics, max for scores)
  const metricArrays = {};
  metricKeys.forEach(k => {
    metricArrays[k] = rows
      .map(r => ({ model: r.model, val: safeNumber(r.metrics[k]) }))
      .filter(x => isFinite(x.val));
  });

  const bestMap = {}; // k -> bestVal
  metricKeys.forEach(k => {
    const arr = metricArrays[k];
    if (!arr.length) return;
    const lowerIsBetter = /rmse|mae|mape|mse/i.test(k);
    bestMap[k] = lowerIsBetter
      ? Math.min(...arr.map(x => x.val))
      : Math.max(...arr.map(x => x.val));
  });

  // Body
  rows.forEach(r => {
    const tr = document.createElement("tr");
    const cells = [
      `<td><strong>${r.model}</strong></td>`,
      `<td>${r.type || "—"}</td>`,
      ...metricKeys.map(k => {
        const v = r.metrics[k];
        const num = safeNumber(v);
        const text = isFinite(num) ? formatNumber(k, num) : (v ?? "—");

        let cls = "";
        const best = bestMap[k];
        if (isFinite(num) && isFinite(best) && nearlyEqual(num, best)) {
          cls = "table-success font-weight-bold"; // highlight best
        }
        return `<td class="${cls}">${text}</td>`;
      })
    ];
    tr.innerHTML = cells.join("");
    tbody.appendChild(tr);
  });
}

function safeNumber(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}
function formatNumber(metric, val) {
  if (/mape/i.test(metric)) return `${val.toFixed(2)}%`;
  if (/rmse|mae|mse/i.test(metric)) return val.toFixed(4);
  if (/r2|roc|acc/i.test(metric)) return val.toFixed(4);
  return String(val);
}
function nearlyEqual(a, b, eps = 1e-9) {
  return Math.abs(a - b) <= eps;
}

// ===== Verify Latest Prediction =====
if (verifyBtn) {
  verifyBtn.addEventListener("click", async () => {
    clearError();
    if (!currentFilePath) return showError("Please upload or download a CSV first.");
    const model = getSelectedModel();
    show(verifyAlert, "Verifying..."); hide(verifyOutput);
    try {
      const payload = await tryPOST("/verify-latest", {
        file_path: currentFilePath,
        model,
        time_step: getTimeStep()
      });
      hide(verifyAlert);
      verifyOutput.textContent = JSON.stringify(payload, null, 2);
      show(verifyOutput);
    } catch (err) {
      hide(verifyAlert);
      showError(`Verify error: ${err.message || err}`);
    }
  });
}

// ===== Model Comparison: overlay of test predictions =====
function allRegressionModels() {
  // Excludes logistic (classification)
  return ["lstm", "linear", "rf", "svm", "xgb"];
}
function allModelsForMetrics() {
  return ["lstm", "linear", "rf", "svm", "xgb", "logit", "naive"];
}
function showCompareAlert(msg) {
  if (compareError) { compareError.style.display = "none"; compareError.textContent = ""; }
  if (compareAlert) { compareAlert.style.display = ""; compareAlert.textContent = msg || "Working…"; }
}
function showCompareError(msg) {
  if (compareAlert) compareAlert.style.display = "none";
  if (compareError) { compareError.style.display = ""; compareError.textContent = msg || "Something went wrong"; }
}
function hideCompareAlerts() {
  if (compareAlert) compareAlert.style.display = "none";
  if (compareError) compareError.style.display = "none";
}

async function buildOverlayComparison() {
  if (!currentFilePath) return showCompareError("Please upload or download a CSV first.");
  showCompareAlert("Collecting model predictions…");

  const time_step = getTimeStep();
  const models = allRegressionModels();

  // 1) Fetch graph_data for each model (fast path, no retrain)
  const series = []; // { key, true_prices, ts, train_len, test_pred[] }
  for (const m of models) {
    try {
      const res = await tryPOST("/predict", {
        file_path: currentFilePath,
        model: m,
        time_step,
        retrain: "0"
      });
      const g = res?.graph_data || {};
      const true_prices = (g.true_prices || []).map(Number);
      const train_predict = (g.train_predict || []).map(Number);
      const test_predict = (g.test_predict || []).map(Number);
      const ts = Number(g.time_step || time_step);

      series.push({
        key: m.toUpperCase(),
        true_prices,
        ts,
        train_len: train_predict.length,
        test_pred: test_predict
      });
    } catch (e) {
      console.warn("Predict failed for", m, e);
    }
  }

  if (!series.length) return showCompareError("No model predictions available.");

  // 2) Build aligned arrays sized to the true series length of the *longest* truth we found
  hideCompareAlerts();

  const maxTruthLen = Math.max(...series.map(s => s.true_prices.length));
  const labels = Array.from({ length: maxTruthLen }, (_, i) => i + 1);

  // True series: choose the longest; pad others if needed (nulls)
  const mainTruth = series.reduce((a, b) => (a.true_prices.length >= b.true_prices.length ? a : b)).true_prices;
  const truthPlot = new Array(maxTruthLen).fill(null);
  for (let i = 0; i < mainTruth.length; i++) truthPlot[i] = mainTruth[i];

  // For each model, create a null-padded array with only the TEST window filled
  const modelDatasets = [];
  for (const s of series) {
    const arr = new Array(maxTruthLen).fill(null);
    const testStart = s.ts + s.train_len;
    for (let i = 0; i < s.test_pred.length; i++) {
      const idx = testStart + i;
      if (idx < arr.length) arr[idx] = s.test_pred[i];
    }
    modelDatasets.push({
      label: `${s.key} (Test)`,
      data: arr,
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      tension: 0.1,
    });
  }

  // 3) Render the overlay chart
  if (overlayChart) overlayChart.destroy();
  const ctx = overlayCanvas.getContext("2d");
  overlayChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "True Prices",
          data: truthPlot,
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          tension: 0.1,
        },
        ...modelDatasets
      ]
    },
    options: {
      responsive: true,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: true, position: "top" } },
      scales: {
        x: { title: { display: true, text: "Index (oldest → newest)" } },
        y: { title: { display: true, text: "Price" }, beginAtZero: false }
      }
    }
  });
}

// ===== Metrics bar chart (RMSE across models) =====
async function buildMetricsBars() {
  if (!currentFilePath) return showCompareError("Please upload or download a CSV first.");
  showCompareAlert("Evaluating models…");
  metricsNote.style.display = "none";
  metricsNote.textContent = "";

  try {
    const payload = await tryPOST("/evaluate", {
      file_path: currentFilePath,
      time_step: getTimeStep(),
      retrain: "0"
    });
    hideCompareAlerts();

    const results = payload?.results || {};
    // Collect RMSE for regression models (including 'naive' if present)
    const labels = [];
    const values = [];
    let logitNote = "";

    for (const [k, obj] of Object.entries(results)) {
      const type = obj?.type;
      const mets = obj?.metrics || {};
      if (type === "regression" && Number.isFinite(Number(mets.rmse))) {
        labels.push(k.toUpperCase());
        values.push(Number(mets.rmse));
      }
    }

    // Logistic (classification) note
    if (results.logit && results.logit.metrics) {
      const acc = results.logit.metrics.accuracy;
      const auc = results.logit.metrics.roc_auc;
      const accTxt = (acc != null && isFinite(Number(acc))) ? `${(Number(acc)*100).toFixed(2)}%` : "—";
      const aucTxt = (auc != null && isFinite(Number(auc))) ? Number(auc).toFixed(4) : "—";
      logitNote = `Logistic (direction): Accuracy ${accTxt}, ROC-AUC ${aucTxt}.`;
    }

    if (metricsChart) metricsChart.destroy();
    const ctx = metricsCanvas.getContext("2d");
    metricsChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "RMSE (lower is better)",
          data: values
        }]
      },
      options: {
        responsive: true,
        animation: false,
        plugins: {
          legend: { display: true, position: "top" },
          tooltip: { callbacks: { label: (it) => `RMSE: ${Number(it.raw).toFixed(4)}` } }
        },
        scales: {
          y: { beginAtZero: false, title: { display: true, text: "RMSE" } }
        }
      }
    });

    if (logitNote) {
      metricsNote.textContent = logitNote;
      metricsNote.style.display = "";
    }
  } catch (err) {
    showCompareError(`Metrics error: ${err.message || err}`);
  }
}

// Wire up comparison buttons
if (compareOverlayBtn) {
  compareOverlayBtn.addEventListener("click", async () => {
    hideCompareAlerts();
    await buildOverlayComparison();
  });
}
if (compareMetricsBtn) {
  compareMetricsBtn.addEventListener("click", async () => {
    hideCompareAlerts();
    await buildMetricsBars();
  });
}

// ===== Init =====
function ensureChartOnLoad() { ensureChart(); }
ensureChartOnLoad();
