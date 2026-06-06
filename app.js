const API_BASE = "http://127.0.0.1:8000";

let datasetProfile = null;
let savedModels = {};
let selectedAlgorithm = "xgboost";

const fmt = new Intl.NumberFormat("es-MX");
const pct = (value) => `${(value * 100).toFixed(1)}%`;
const modelOrder = ["linear_regression", "random_forest", "xgboost", "neural_network"];
const featureDescriptions = {
  Age: "Edad del solicitante. En este dataset suele relacionarse con estabilidad financiera y riesgo histórico.",
  Income: "Ingreso anual del solicitante. Representa capacidad de pago.",
  LoanAmount: "Monto solicitado del préstamo. Montos altos pueden elevar el riesgo si superan la capacidad de pago.",
  CreditScore: "Puntaje crediticio de 300 a 850. Resume historial de pagos y comportamiento crediticio.",
  MonthsEmployed: "Meses de antigüedad laboral. Más estabilidad laboral suele reducir riesgo.",
  NumCreditLines: "Número de líneas de crédito activas. Refleja exposición crediticia actual.",
  InterestRate: "Tasa de interés del préstamo. Tasas más altas suelen asociarse a mayor riesgo.",
  LoanTerm: "Plazo del préstamo en meses. Mide por cuánto tiempo se mantiene la exposición.",
  DTIRatio: "Relación deuda-ingreso. Valores altos indican mayor carga financiera.",
  HasMortgage: "Indica si el cliente tiene hipoteca.",
  HasDependents: "Indica si el cliente tiene dependientes económicos.",
  HasCoSigner: "Indica si el cliente tiene cofirmante o aval.",
};

function setApiState(text, ok = true) {
  const el = document.getElementById("api-state");
  el.textContent = text;
  el.className = `api-state ${ok ? "ok" : "bad"}`;
}

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function loadDataset() {
  try {
    setApiState("API conectada", true);
    datasetProfile = await api("/dataset");
    renderDataset(datasetProfile);
    await loadModels();
  } catch (err) {
    setApiState("API no disponible", false);
    document.getElementById("dataset-grid").innerHTML = `
      <div class="empty wide">No se pudo cargar el dataset: ${err.message}</div>
    `;
  }
}

async function uploadDataset(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  const body = new FormData();
  body.append("file", file);

  document.getElementById("dataset-grid").innerHTML = `<div class="empty wide">Subiendo CSV...</div>`;
  try {
    datasetProfile = await api("/dataset/upload", { method: "POST", body });
    renderDataset(datasetProfile);
    await loadModels();
  } catch (err) {
    document.getElementById("dataset-grid").innerHTML = `
      <div class="empty wide">No se pudo subir el dataset: ${err.message}</div>
    `;
  }
}

function renderDataset(d) {
  document.getElementById("dataset-grid").innerHTML = `
    <div class="data-card"><span>Filas</span><strong>${fmt.format(d.rows)}</strong></div>
    <div class="data-card"><span>Columnas</span><strong>${fmt.format(d.columns)}</strong></div>
    <div class="data-card"><span>Variables predictoras</span><strong>${fmt.format(d.feature_count)}</strong></div>
    <div class="data-card"><span>Tasa de default</span><strong>${d.default_rate.toFixed(2)}%</strong></div>
  `;
}

async function loadModels() {
  const data = await api("/models");
  savedModels = data.models || {};
  renderPredictModelOptions();
  renderModelsDashboard();
  renderRunsList();
}

function renderModelAvailability() {
  if (!document.querySelector(".algo-card")) return;
  document.querySelectorAll(".algo-card").forEach((card) => {
    const algorithm = card.dataset.algorithm;
    const model = savedModels[algorithm];
    card.classList.toggle("missing", !model);
    const small = card.querySelector("small");
    if (model) {
      const auc = model.metrics?.auc != null ? model.metrics.auc.toFixed(3) : "--";
      const f1 = model.metrics?.f1 != null ? model.metrics.f1.toFixed(3) : "--";
      small.textContent = `Guardado - AUC ${auc} - F1 ${f1}`;
    }
  });

  const status = document.getElementById("train-status");
  if (!status) return;
  const count = Object.keys(savedModels).length;
  status.textContent = count
    ? `${count} modelo(s) guardado(s). Selecciona uno para ver sus resultados.`
    : "No hay modelos guardados. Ejecuta train_saved_models.py una sola vez.";
}

function selectAlgorithm(button) {
  document.querySelectorAll(".algo-card").forEach((el) => el.classList.remove("active"));
  button.classList.add("active");
  selectedAlgorithm = button.dataset.algorithm;
  showSelectedRun();
}

function renderRunSelect() {
  const select = document.getElementById("run-select");
  if (!select) {
    renderPredictModelOptions();
    return;
  }
  const entries = modelOrder.map((key) => savedModels[key]).filter(Boolean);
  if (!entries.length) {
    select.innerHTML = `<option value="">Sin modelos guardados</option>`;
    renderPredictModelOptions();
    return;
  }

  select.innerHTML = entries
    .map((run) => `<option value="${run.algorithm}">${run.model_name} - ${run.id}</option>`)
    .join("");
  select.value = selectedAlgorithm in savedModels ? selectedAlgorithm : entries[0].algorithm;
  selectedAlgorithm = select.value;
  syncAlgorithmCards();
  renderPredictModelOptions();
}

function showSelectedRun() {
  const select = document.getElementById("run-select");
  if (!select) {
    renderModelsDashboard();
    return;
  }
  if (select.value) selectedAlgorithm = select.value;
  syncAlgorithmCards();
  const run = savedModels[selectedAlgorithm];

  if (!run) {
    renderEmptyDashboard();
    return;
  }

  renderMetrics(run);
  renderMatrix(run);
  renderImportance(run);
  renderRunsList();
}

function syncAlgorithmCards() {
  document.querySelectorAll(".algo-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.algorithm === selectedAlgorithm);
  });
}

function renderPredictModelOptions() {
  const select = document.getElementById("p-model");
  if (!select) return;

  const entries = modelOrder.map((key) => savedModels[key]).filter(Boolean);
  const current = select.value || selectedAlgorithm;

  select.innerHTML = [
    `<option value="all">Comparar todos</option>`,
    ...entries.map((run) => `<option value="${run.algorithm}">${run.model_name}</option>`),
  ].join("");

  select.value = current in savedModels ? current : "all";
}

function metricCard(label, value, hint) {
  return `
    <div class="mcard">
      <div class="ml">${label}</div>
      <div class="mn">${value}</div>
      <div class="mh">${hint}</div>
    </div>
  `;
}

function tooltip(label, text) {
  return `
    <span class="tooltip-wrap">
      ${label}
      <span class="info-icon">?</span>
      <span class="tooltip-box">${text}</span>
    </span>
  `;
}

function inlineTip(label, text) {
  return `<span class="hover-tip" data-tip="${text}">${label}</span>`;
}

function metricMini(label, value, hint) {
  return `
    <div class="model-metric">
      <span>${tooltip(label, hint)}</span>
      <strong>${value}</strong>
    </div>
  `;
}

function matrixCell(label, value, cls, description) {
  return `
    <div class="matrix-cell ${cls}">
      <span>${inlineTip(label, description)}</span>
      <strong>${fmt.format(value || 0)}</strong>
    </div>
  `;
}

function renderModelMatrix(run) {
  const c = run.confusion_matrix;
  if (!c) {
    return `<div class="empty">Este modelo no tiene matriz disponible.</div>`;
  }

  return `
    <div class="matrix compact-matrix">
      ${matrixCell("TN", c.tn, "good", "Verdaderos negativos: clientes que no cayeron en default y el modelo clasificó correctamente como no default.")}
      ${matrixCell("FP", c.fp, "warn", "Falsos positivos: clientes sanos marcados como riesgo. Puede reducir aprobaciones aunque protege contra pérdidas.")}
      ${matrixCell("FN", c.fn, "bad", "Falsos negativos: clientes que sí cayeron en default pero el modelo no detectó. En crédito suele ser el error más costoso.")}
      ${matrixCell("TP", c.tp, "good", "Verdaderos positivos: defaults detectados correctamente. Ayuda a controlar riesgo crediticio.")}
    </div>
  `;
}

function renderModelImportance(run) {
  const rows = run.feature_importance || [];
  if (!rows.length) {
    return `<div class="empty">Este modelo no expone importancia directa de variables.</div>`;
  }

  const max = Math.max(...rows.map((row) => row.importance));
  return `
    <div class="bars model-bars">
      ${rows
        .slice(0, 8)
        .map((row) => {
          const width = Math.max(4, (row.importance / max) * 100);
          return `
            <div class="bar-row">
              <div class="bar-label">${inlineTip(row.feature, featureDescriptions[row.feature] || `Peso relativo de ${row.feature} dentro de este modelo. Una barra mayor significa que influyó más en la predicción aprendida.`)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
              <div class="bar-value">${(row.importance * 100).toFixed(1)}%</div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderModelsDashboard() {
  const wrap = document.getElementById("models-dashboard");
  if (!wrap) return;

  const entries = modelOrder.map((key) => savedModels[key]).filter(Boolean);
  if (!entries.length) {
    wrap.innerHTML = `<div class="empty">Aún no hay modelos guardados para mostrar.</div>`;
    return;
  }

  wrap.innerHTML = entries
    .map((run) => {
      const m = run.metrics || {};
      return `
        <article class="model-dashboard-card ${run.algorithm === selectedAlgorithm ? "active" : ""}">
          <div class="model-card-head">
            <div>
              <h3>${run.model_name}</h3>
              <span>${run.features?.length || run.dataset?.features_used || "--"} variables · ${run.dataset?.test_rows ? fmt.format(run.dataset.test_rows) + " filas test" : "artefacto del notebook"}</span>
            </div>
            <button onclick="selectRunFromList('${run.algorithm}')">Usar</button>
          </div>

          <div class="model-metrics-inline">
            ${metricMini("Accuracy", m.accuracy != null ? pct(m.accuracy) : "--", "Porcentaje total de clasificaciones correctas. En datasets desbalanceados debe leerse junto con recall y F1.")}
            ${metricMini("AUC", m.auc != null ? m.auc.toFixed(3) : "--", "Capacidad del modelo para separar clientes con y sin default. Más cercano a 1 es mejor.")}
            ${metricMini("Recall", m.recall != null ? pct(m.recall) : "--", "Proporción de defaults reales que el modelo logró detectar. Es clave para riesgo crediticio.")}
            ${metricMini("F1", m.f1 != null ? m.f1.toFixed(3) : "--", "Balance entre precisión y recall. Útil cuando hay desbalance de clases.")}
          </div>

          <div class="model-detail-grid">
            <section>
              <div class="mini-title">${tooltip("Matriz de confusión", "Resume aciertos y errores del modelo. En crédito conviene vigilar especialmente los falsos negativos.")}</div>
              ${renderModelMatrix(run)}
            </section>
            <section>
              <div class="mini-title">${tooltip("Variables más influyentes", "Muestra qué variables pesaron más para este modelo según el artefacto entrenado o la interpretación calculada.")}</div>
              ${renderModelImportance(run)}
            </section>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderMetrics(run) {
  const m = run.metrics;
  document.getElementById("metrics-grid").innerHTML = [
    metricCard("Accuracy", m?.accuracy != null ? pct(m.accuracy) : "--", "clasificacion total"),
    metricCard("AUC", m?.auc != null ? m.auc.toFixed(3) : "--", "separacion entre clases"),
    metricCard("Recall", m?.recall != null ? pct(m.recall) : "--", "defaults detectados"),
    metricCard("F1-score", m?.f1 != null ? m.f1.toFixed(3) : "--", "balance precision/recall"),
  ].join("");
}

function renderMatrix(run) {
  const c = run.confusion_matrix;
  if (!c) {
    document.getElementById("matrix-caption").textContent = run.model_name;
    document.getElementById("matrix").innerHTML = `<div class="empty wide">Este artefacto no incluye matriz de confusión guardada.</div>`;
    return;
  }
  document.getElementById("matrix-caption").textContent =
    `${run.model_name} - ${fmt.format(run.dataset?.test_rows || 0)} filas de prueba`;
  document.getElementById("matrix").innerHTML = `
    <div class="matrix-cell good"><span>TN</span><strong>${fmt.format(c.tn)}</strong><small>No default correcto</small></div>
    <div class="matrix-cell warn"><span>FP</span><strong>${fmt.format(c.fp)}</strong><small>Alerta falsa</small></div>
    <div class="matrix-cell bad"><span>FN</span><strong>${fmt.format(c.fn)}</strong><small>Default no detectado</small></div>
    <div class="matrix-cell good"><span>TP</span><strong>${fmt.format(c.tp)}</strong><small>Default detectado</small></div>
  `;
}

function renderImportance(run) {
  const rows = run.feature_importance || [];
  const wrap = document.getElementById("importance-bars");
  document.getElementById("importance-caption").textContent =
    `${run.dataset?.features_used || run.features?.length || "--"} variables usadas`;

  if (!rows.length) {
    wrap.innerHTML = `<div class="empty">Este modelo no expone importancia directa de variables.</div>`;
    return;
  }

  const max = Math.max(...rows.map((row) => row.importance));
  wrap.innerHTML = rows
    .map((row) => {
      const width = Math.max(4, (row.importance / max) * 100);
      return `
        <div class="bar-row">
          <div class="bar-label">${row.feature}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <div class="bar-value">${(row.importance * 100).toFixed(1)}%</div>
        </div>
      `;
    })
    .join("");
}

function renderRunsList() {
  const entries = Object.values(savedModels).sort((a, b) => (b.metrics?.auc || 0) - (a.metrics?.auc || 0));
  const wrap = document.getElementById("runs-list");

  if (!entries.length) {
    wrap.innerHTML = `<div class="empty">Aun no hay modelos persistidos.</div>`;
    return;
  }

  wrap.innerHTML = entries
    .map((run) => `
      <button class="run-item ${run.algorithm === selectedAlgorithm ? "active" : ""}"
              onclick="selectRunFromList('${run.algorithm}')">
        <div>
          <strong>${run.model_name}</strong>
          <span>${run.created_at ? new Date(run.created_at).toLocaleString("es-MX") : "Artefacto guardado"} - ${fmt.format(run.dataset?.rows_used || 0)} filas</span>
        </div>
        <div class="run-score">AUC ${run.metrics?.auc != null ? run.metrics.auc.toFixed(3) : "--"}</div>
      </button>
    `)
    .join("");
}

function selectRunFromList(algorithm) {
  selectedAlgorithm = algorithm;
  const select = document.getElementById("run-select");
  if (select) select.value = algorithm;
  document.querySelectorAll(".algo-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.algorithm === algorithm);
  });
  renderModelsDashboard();
  renderRunsList();
  const predictSelect = document.getElementById("p-model");
  if (predictSelect && algorithm in savedModels) predictSelect.value = algorithm;
}

function renderEmptyDashboard() {
  document.getElementById("metrics-grid").innerHTML = [
    metricCard("Accuracy", "--", "sin modelo"),
    metricCard("AUC", "--", "sin modelo"),
    metricCard("Recall", "--", "sin modelo"),
    metricCard("F1-score", "--", "sin modelo"),
  ].join("");
  document.getElementById("matrix").innerHTML = `<div class="empty wide">Selecciona un modelo guardado.</div>`;
  document.getElementById("importance-bars").innerHTML = `<div class="empty">Sin importancias disponibles.</div>`;
  renderRunsList();
}

async function trainModel() {
  const button = document.getElementById("train-btn");
  const sampleRaw = document.getElementById("sample-size").value;
  const payload = {
    algorithm: selectedAlgorithm,
    sample_size: sampleRaw ? Number(sampleRaw) : null,
    test_size: Number(document.getElementById("test-size").value),
  };

  button.disabled = true;
  document.getElementById("train-status").textContent =
    "Preparando version guardada. Esto se hace solo cuando quieres actualizar artefactos.";

  try {
    const run = await api("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    savedModels[run.algorithm] = run;
    selectedAlgorithm = run.algorithm;
    renderModelAvailability();
    renderRunSelect();
    showSelectedRun();
    document.getElementById("train-status").textContent =
      `Version guardada: ${run.model_name} - AUC ${run.metrics.auc.toFixed(3)}.`;
  } catch (err) {
    document.getElementById("train-status").textContent = `No se pudo guardar version: ${err.message}`;
  } finally {
    button.disabled = false;
  }
}

function collectClientPayload() {
  return {
    Age: Number(document.getElementById("p-age").value),
    Income: Number(document.getElementById("p-income").value),
    LoanAmount: Number(document.getElementById("p-loan").value),
    CreditScore: Number(document.getElementById("p-score").value),
    MonthsEmployed: Number(document.getElementById("p-employed").value),
    NumCreditLines: Number(document.getElementById("p-lines").value),
    InterestRate: Number(document.getElementById("p-rate").value),
    LoanTerm: Number(document.getElementById("p-term").value),
    DTIRatio: Number(document.getElementById("p-dti").value),
    HasMortgage: Number(document.getElementById("p-mortgage").checked),
    HasDependents: Number(document.getElementById("p-dependents").checked),
    HasCoSigner: Number(document.getElementById("p-cosigner").checked),
    Education: document.getElementById("p-education").value,
    EmploymentType: document.getElementById("p-employment").value,
    MaritalStatus: document.getElementById("p-marital").value,
    LoanPurpose: document.getElementById("p-purpose").value,
  };
}

function decisionClass(decision) {
  if (decision === "APROBADO") return "approved";
  if (decision === "RECHAZADO") return "rejected";
  return "review";
}

function renderSinglePrediction(data) {
  const cls = decisionClass(data.decision);
  document.getElementById("prediction-result").innerHTML = `
    <div class="res-card ${cls}">
      <div class="res-status">${data.decision}</div>
      <div class="res-decision">${data.risk_level}</div>
      <div class="res-detail">
        Probabilidad de default: <strong>${pct(data.default_probability)}</strong><br>
        Modelo: <strong>${data.model_name}</strong><br>
        ${data.message}
      </div>
    </div>
  `;
}

function renderAllPredictions(data) {
  const rows = data.predictions
    .map((item) => {
      if (item.error) {
        return `
          <tr>
            <td>${item.model_name}</td>
            <td colspan="3">${item.error}</td>
          </tr>
        `;
      }
      return `
        <tr>
          <td>${item.model_name}</td>
          <td>${pct(item.default_probability)}</td>
          <td><span class="decision-pill ${decisionClass(item.decision)}">${item.decision}</span></td>
          <td>${item.risk_level}</td>
        </tr>
      `;
    })
    .join("");

  const consensus = data.consensus;
  const summary = consensus
    ? `
      <div class="res-card ${decisionClass(consensus.decision)}">
        <div class="res-status">Consenso</div>
        <div class="res-decision">${consensus.decision} - ${consensus.risk_level}</div>
        <div class="res-detail">
          Promedio de default: <strong>${pct(consensus.average_default_probability)}</strong><br>
          ${consensus.message}
        </div>
      </div>
    `
    : "";

  document.getElementById("prediction-result").innerHTML = `
    ${summary}
    <div class="table-wrap">
      <table class="prediction-table">
        <thead>
          <tr>
            <th>Modelo</th>
            <th>Probabilidad</th>
            <th>Decisión</th>
            <th>Riesgo</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

async function evaluateClient() {
  const result = document.getElementById("prediction-result");
  const selected = document.getElementById("p-model").value;
  const payload = collectClientPayload();

  result.innerHTML = `<div class="empty">Evaluando perfil...</div>`;

  try {
    if (selected === "all") {
      const data = await api("/predict-all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      renderAllPredictions(data);
      return;
    }

    const data = await api(`/predict?model=${encodeURIComponent(selected)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderSinglePrediction(data);
  } catch (err) {
    result.innerHTML = `<div class="empty">No se pudo evaluar: ${err.message}</div>`;
  }
}

function loadDemoClient() {
  document.getElementById("p-age").value = 38;
  document.getElementById("p-income").value = 72000;
  document.getElementById("p-loan").value = 45000;
  document.getElementById("p-score").value = 680;
  document.getElementById("p-employed").value = 84;
  document.getElementById("p-lines").value = 3;
  document.getElementById("p-rate").value = 12.5;
  document.getElementById("p-term").value = 36;
  document.getElementById("p-dti").value = 0.42;
  document.getElementById("p-mortgage").checked = true;
  document.getElementById("p-dependents").checked = true;
  document.getElementById("p-cosigner").checked = false;
  document.getElementById("p-education").value = "Master's";
  document.getElementById("p-employment").value = "Full-time";
  document.getElementById("p-marital").value = "Married";
  document.getElementById("p-purpose").value = "Home";
}

window.addEventListener("DOMContentLoaded", loadDataset);
