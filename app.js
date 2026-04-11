// ═══════════════════════════════════════════════════════════════════════════
// app.js — Sistema Crediticio · Frontend ↔ FastAPI
// Compatible con el HTML y CSS actuales
// ═══════════════════════════════════════════════════════════════════════════

const API_BASE = "http://127.0.0.1:8000";
const USE_BACKEND = true;

// ── Historial en memoria ──────────────────────────────────────────────────
const sessionHistory = [];

// ── Navegación entre pestañas ─────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
  document.querySelectorAll(".nav-tab").forEach((b) => b.classList.remove("active"));

  document.getElementById("tab-" + name).classList.add("active");

  document.querySelectorAll(".nav-tab").forEach((b) => {
    const txt = b.textContent.toLowerCase();
    if (
      (name === "consulta" && txt.includes("rápida")) ||
      (name === "registro" && txt.includes("registro")) ||
      (name === "historial" && txt.includes("historial"))
    ) {
      b.classList.add("active");
    }
  });

  if (name === "historial") renderHistory();
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

function creditScoreLabel(cs) {
  if (cs < 580) return "Deficiente";
  if (cs < 670) return "Regular";
  if (cs < 740) return "Bueno";
  return "Excelente";
}

function localPredict(age, income, loanAmount, creditScore, monthsEmp, numLines) {
  const b = [-0.00260, -0.00000812, 0.00000935, -0.00310, -0.00185, 0.0412];
  const intercept = 1.05;

  const z =
    intercept +
    b[0] * age +
    b[1] * income +
    b[2] * loanAmount +
    b[3] * creditScore +
    b[4] * monthsEmp +
    b[5] * numLines;

  return 1 / (1 + Math.exp(-z));
}

function interpretRisk(prob) {
  if (prob < 0.25) {
    return {
      level: "Riesgo bajo",
      cls: "approved",
      status: "APROBABLE",
      message: "Perfil sólido. Préstamo recomendado sin condiciones especiales."
    };
  }
  if (prob < 0.45) {
    return {
      level: "Riesgo moderado",
      cls: "review",
      status: "REVISAR",
      message: "Perfil aceptable. Considerar monto menor o garantía adicional."
    };
  }
  if (prob < 0.65) {
    return {
      level: "Riesgo medio-alto",
      cls: "review",
      status: "REVISAR",
      message: "Revisar historial detallado. Se recomienda tasa diferenciada."
    };
  }
  return {
    level: "Riesgo alto",
    cls: "rejected",
    status: "RECHAZAR",
    message: "Alta probabilidad de default. Préstamo no recomendado."
  };
}

async function predecirRiesgo(payload) {
  const res = await fetch(`${API_BASE}/predict?model=xgb`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    throw new Error(`Error HTTP ${res.status}`);
  }

  return await res.json();
}

function setStepState(activeStep) {
  const ids = ["rs1", "rs2", "rs3"];
  ids.forEach((id, idx) => {
    const el = document.getElementById(id);
    if (!el) return;

    el.classList.remove("active", "done");

    if (idx + 1 < activeStep) el.classList.add("done");
    if (idx + 1 === activeStep) el.classList.add("active");
  });
}

function rClear(fieldId) {
  document.getElementById(fieldId)?.classList.remove("invalid");
}

function showTyping() {
  return `<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
}

// ═══════════════════════════════════════════════════════════════════════════
// PESTAÑA 1 — CONSULTA RÁPIDA
// ═══════════════════════════════════════════════════════════════════════════

function renderQuickResult(prob, modelName, message = null) {
  const pct = (prob * 100).toFixed(1);
  const risk = interpretRisk(prob);

  const result = document.getElementById("q-result");
  if (!result) return;

  result.innerHTML = `
    <div class="res-card ${risk.cls}">
      <div class="res-status">${risk.status}</div>
      <div class="res-decision">${risk.level}</div>
      <div class="res-detail">
        Probabilidad de default: <strong>${pct}%</strong><br>
        Modelo: <strong>${modelName}</strong><br>
        ${message || risk.message}
      </div>
    </div>
  `;
}

let qDebounce = null;

async function qCompute() {
  const age = +document.getElementById("q-age").value;
  const emp = +document.getElementById("q-emp").value;
  const inc = +document.getElementById("q-inc").value;
  const loan = +document.getElementById("q-loan").value;
  const cred = +document.getElementById("q-cred").value;
  const cs = +document.getElementById("q-cs").value;

  document.getElementById("q-age-v").textContent = `${age} años`;
  document.getElementById("q-emp-v").textContent = `${emp} meses`;
  document.getElementById("q-inc-v").textContent = `$${inc.toLocaleString()}`;
  document.getElementById("q-loan-v").textContent = `$${loan.toLocaleString()}`;
  document.getElementById("q-cred-v").textContent = `${cred}`;
  document.getElementById("q-cs-v").textContent = `${cs} — ${creditScoreLabel(cs)}`;

  clearTimeout(qDebounce);

  qDebounce = setTimeout(async () => {
    const payload = {
      Age: age,
      Income: inc,
      LoanAmount: loan,
      CreditScore: cs,
      MonthsEmployed: emp,
      NumCreditLines: cred
    };

    if (!USE_BACKEND) {
      const prob = localPredict(age, inc, loan, cs, emp, cred);
      renderQuickResult(prob, "Estimación local");
      return;
    }

    try {
      const data = await predecirRiesgo(payload);
      renderQuickResult(
        data.default_probability,
        data.model,
        data.message
      );
    } catch (err) {
      console.error("Error API /predict:", err);
      const prob = localPredict(age, inc, loan, cs, emp, cred);
      renderQuickResult(
        prob,
        "Estimación local",
        "No se pudo consultar la API. Se muestra una estimación local de respaldo."
      );
    }
  }, 250);
}

// ═══════════════════════════════════════════════════════════════════════════
// PESTAÑA 2 — REGISTRO DE CLIENTE
// ═══════════════════════════════════════════════════════════════════════════

let rCurrentData = null;

function rDemo() {
  document.getElementById("r-nombre").value = "María García López";
  document.getElementById("r-age").value = 38;
  document.getElementById("r-income").value = 72000;
  document.getElementById("r-loan").value = 45000;
  document.getElementById("r-cs").value = 680;
  document.getElementById("r-emp").value = 84;
  document.getElementById("r-cred").value = 3;

  ["rf-nombre", "rf-age", "rf-income", "rf-loan", "rf-cs", "rf-emp", "rf-cred"].forEach(rClear);
}

function rValidate() {
  let ok = true;

  const rules = [
    { id: "r-nombre", field: "rf-nombre", test: (v) => v.trim().length >= 2 },
    { id: "r-age", field: "rf-age", test: (v) => v >= 18 && v <= 85 },
    { id: "r-income", field: "rf-income", test: (v) => v >= 1000 },
    { id: "r-loan", field: "rf-loan", test: (v) => v >= 1000 },
    { id: "r-cs", field: "rf-cs", test: (v) => v >= 300 && v <= 850 },
    { id: "r-emp", field: "rf-emp", test: (v) => v >= 0 },
    { id: "r-cred", field: "rf-cred", test: (v) => v >= 0 && v <= 20 }
  ];

  rules.forEach((r) => {
    const raw = document.getElementById(r.id).value;
    const val = r.id === "r-nombre" ? raw : parseFloat(raw);

    if (!r.test(val)) {
      document.getElementById(r.field).classList.add("invalid");
      ok = false;
    } else {
      document.getElementById(r.field).classList.remove("invalid");
    }
  });

  return ok;
}

async function rEval() {
  if (!rValidate()) return;

  const payload = {
    Age: +document.getElementById("r-age").value,
    Income: +document.getElementById("r-income").value,
    LoanAmount: +document.getElementById("r-loan").value,
    CreditScore: +document.getElementById("r-cs").value,
    MonthsEmployed: +document.getElementById("r-emp").value,
    NumCreditLines: +document.getElementById("r-cred").value
  };

  document.getElementById("rp1").classList.add("hidden");
  document.getElementById("rp2").classList.remove("hidden");
  setStepState(2);

  document.getElementById("r-ai-text").innerHTML = showTyping();

  let prob, modelName, riskLevel, message;

  if (USE_BACKEND) {
    try {
      const data = await predecirRiesgo(payload);
      prob = data.default_probability;
      modelName = data.model;
      riskLevel = data.risk_level;
      message = data.message;
    } catch (err) {
      console.error("Error API /predict:", err);
      prob = localPredict(
        payload.Age,
        payload.Income,
        payload.LoanAmount,
        payload.CreditScore,
        payload.MonthsEmployed,
        payload.NumCreditLines
      );
      modelName = "Estimación local";
      riskLevel = interpretRisk(prob).level;
      message = "No se pudo consultar la API. Se muestra una estimación local de respaldo.";
    }
  } else {
    prob = localPredict(
      payload.Age,
      payload.Income,
      payload.LoanAmount,
      payload.CreditScore,
      payload.MonthsEmployed,
      payload.NumCreditLines
    );
    modelName = "Estimación local";
    riskLevel = interpretRisk(prob).level;
    message = "Modo sin backend activo.";
  }

  const risk = interpretRisk(prob);

  rCurrentData = {
    nombre: document.getElementById("r-nombre").value.trim(),
    ...payload,
    prob,
    modelName,
    riskLevel,
    message,
    timestamp: new Date().toLocaleString("es-MX")
  };

  document.getElementById("r-result-wrap").innerHTML = `
    <div class="res-card ${risk.cls}">
      <div class="res-status">${risk.status}</div>
      <div class="res-decision">${riskLevel}</div>
      <div class="res-detail">
        Probabilidad de default: <strong>${(prob * 100).toFixed(1)}%</strong><br>
        Modelo: <strong>${modelName}</strong><br>
        ${message}
      </div>
    </div>
  `;

  fetchAIAnalysis(rCurrentData);
}

async function fetchAIAnalysis(d) {
  const debtIncomeRatio = d.Income > 0 ? (d.LoanAmount / d.Income) : 0;
  const csLabel = creditScoreLabel(d.CreditScore);

  let analysis = "";

  if (d.prob < 0.25) {
    analysis = `El perfil de ${d.nombre} es favorable para otorgamiento. Tiene un riesgo estimado bajo, apoyado por un puntaje crediticio ${csLabel.toLowerCase()} y una relación préstamo-ingreso de ${(debtIncomeRatio * 100).toFixed(1)}%. Se recomienda aprobación bajo condiciones estándar.`;
  } else if (d.prob < 0.45) {
    analysis = `El perfil de ${d.nombre} muestra riesgo moderado. Aunque existen elementos positivos, conviene revisar con mayor detalle el equilibrio entre ingreso, monto solicitado y estabilidad laboral. Se recomienda considerar un monto menor o solicitar garantías adicionales.`;
  } else if (d.prob < 0.65) {
    analysis = `El perfil de ${d.nombre} presenta señales de riesgo medio-alto. El análisis sugiere cautela por la combinación de score crediticio, carga financiera y/o estabilidad laboral. Se recomienda una evaluación complementaria antes de aprobar el crédito.`;
  } else {
    analysis = `El perfil de ${d.nombre} presenta alto riesgo estimado de incumplimiento. La probabilidad de default es elevada y no se recomienda aprobar el crédito en las condiciones actuales.`;
  }

  document.getElementById("r-ai-text").textContent = analysis;
}

function rBack() {
  document.getElementById("rp2").classList.add("hidden");
  document.getElementById("rp1").classList.remove("hidden");
  setStepState(1);
}

function rConfirm() {
  if (!rCurrentData) return;

  sessionHistory.push({ ...rCurrentData });

  document.getElementById("rp2").classList.add("hidden");
  document.getElementById("rp3").classList.remove("hidden");
  setStepState(3);

  document.getElementById("r-confirm-card").innerHTML = `
    <div class="success-banner">
      Cliente registrado exitosamente.
    </div>
    <div class="form-card" style="margin-top:10px;">
      <div class="sec-lbl" style="margin-top:0;">Resumen del expediente</div>
      <p><strong>Nombre:</strong> ${rCurrentData.nombre}</p>
      <p><strong>Riesgo:</strong> ${rCurrentData.riskLevel}</p>
      <p><strong>Probabilidad de default:</strong> ${(rCurrentData.prob * 100).toFixed(1)}%</p>
      <p><strong>Modelo:</strong> ${rCurrentData.modelName}</p>
      <p><strong>Fecha:</strong> ${rCurrentData.timestamp}</p>
    </div>
  `;
}

function rNew() {
  rCurrentData = null;

  ["r-nombre", "r-age", "r-income", "r-loan", "r-cs", "r-emp", "r-cred"].forEach((id) => {
    document.getElementById(id).value = "";
  });

  ["rf-nombre", "rf-age", "rf-income", "rf-loan", "rf-cs", "rf-emp", "rf-cred"].forEach(rClear);

  document.getElementById("rp2").classList.add("hidden");
  document.getElementById("rp3").classList.add("hidden");
  document.getElementById("rp1").classList.remove("hidden");
  setStepState(1);
}

// ═══════════════════════════════════════════════════════════════════════════
// PESTAÑA 3 — HISTORIAL
// ═══════════════════════════════════════════════════════════════════════════

function renderHistory() {
  const wrap = document.getElementById("hist-wrap");

  if (sessionHistory.length === 0) {
    wrap.innerHTML = `<div class="empty">Aún no hay clientes registrados.</div>`;
    return;
  }

  wrap.innerHTML = sessionHistory
    .slice()
    .reverse()
    .map((c) => {
      const risk = interpretRisk(c.prob);
      const dotClass = risk.cls;
      const badgeClass = risk.cls;

      return `
        <div class="hist-item">
          <div class="hdot ${dotClass}"></div>
          <div>
            <div class="hname">${c.nombre}</div>
            <div class="hmeta">
              Score ${c.CreditScore} · $${c.LoanAmount.toLocaleString()} · ${c.timestamp}
            </div>
          </div>
          <div class="hpct">${(c.prob * 100).toFixed(1)}%</div>
          <div class="hbadge ${badgeClass}">${c.riskLevel}</div>
        </div>
      `;
    })
    .join("");
}

// ── Inicialización ─────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  qCompute();
  setStepState(1);
});