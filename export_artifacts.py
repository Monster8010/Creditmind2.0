# ═══════════════════════════════════════════════════════════════════════════
# BLOQUE DE EXPORTACIÓN DE ARTEFACTOS — Pegar al final del notebook
# Ejecutar UNA SOLA VEZ después de entrenar los modelos
# ═══════════════════════════════════════════════════════════════════════════

import joblib
import json
import os

# Ruta destino (ajusta si tu estructura de carpetas es distinta)
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── 1. Modelo principal: XGBoost ────────────────────────────────────────────
joblib.dump(best_xgb, os.path.join(ARTIFACTS_DIR, "model_xgb.pkl"))
print("✓ model_xgb.pkl  guardado")

# ── 2. Modelo secundario: Random Forest ─────────────────────────────────────
joblib.dump(best_rf, os.path.join(ARTIFACTS_DIR, "model_rf.pkl"))
print("✓ model_rf.pkl   guardado")

# ── 3. Features (orden exacto que usa el modelo) ─────────────────────────────
#    XGBoost y RF son tree-based → NO necesitan StandardScaler.
#    El scaler solo se usó para LR y SVM.
features_list = FEATURES  # ['Age','Income','LoanAmount','CreditScore','MonthsEmployed','NumCreditLines']
with open(os.path.join(ARTIFACTS_DIR, "features.json"), "w") as f:
    json.dump(features_list, f)
print("✓ features.json  guardado →", features_list)

# ── 4. Verificación rápida ───────────────────────────────────────────────────
import numpy as np

modelo_test = joblib.load(os.path.join(ARTIFACTS_DIR, "model_xgb.pkl"))
ejemplo     = np.array([[38, 72000, 45000, 680, 84, 3]])   # Age, Income, Loan, CS, Emp, CL
prob_test   = modelo_test.predict_proba(ejemplo)[0][1]
print(f"\n✓ Verificación de carga: prob_default = {prob_test:.4f} (debe ser ~0.30–0.45)")
print("\n══════════════════════════════════════════════════════")
print(" Artefactos listos. Cópialos a proyecto_crediticio/artifacts/")
print(" Luego ejecuta:  uvicorn app.main:app --reload")
print("══════════════════════════════════════════════════════")
