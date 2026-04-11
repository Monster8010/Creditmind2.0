"""
Sistema Crediticio — Backend FastAPI

Modelo principal: XGBoost
Features: Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator


# ── Rutas de artefactos ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model_xgb.pkl"
RF_PATH = ARTIFACTS_DIR / "model_rf.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"


# ── Inicialización de la app ─────────────────────────────────────────────────
app = FastAPI(
    title="API de Análisis Crediticio",
    description="Predice la probabilidad de default usando XGBoost",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para demo local. En producción usa tu dominio.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Variables globales ───────────────────────────────────────────────────────
model_xgb = None
model_rf = None
features = None


# ── Carga de artefactos al arrancar ──────────────────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global model_xgb, model_rf, features

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontró el modelo principal en: {MODEL_PATH}\n"
            "Genera primero artifacts/model_xgb.pkl desde tu notebook."
        )

    if not FEATURES_PATH.exists():
        raise RuntimeError(
            f"No se encontró el archivo de features en: {FEATURES_PATH}\n"
            "Genera primero artifacts/features.json desde tu notebook."
        )

    model_xgb = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)

    # Random Forest opcional
    if RF_PATH.exists():
        model_rf = joblib.load(RF_PATH)

    print(f"✓ Modelo XGBoost cargado")
    print(f"✓ Features cargadas: {features}")

    if model_rf is not None:
        print("✓ Random Forest cargado")
    else:
        print("• Random Forest no disponible, se usará solo XGBoost")


# ── Esquema de entrada ───────────────────────────────────────────────────────
class LoanRequest(BaseModel):
    Age: int = Field(..., ge=18, le=85, example=38)
    Income: float = Field(..., ge=1000, example=72000)
    LoanAmount: float = Field(..., ge=1000, example=45000)
    CreditScore: int = Field(..., ge=300, le=850, example=680)
    MonthsEmployed: int = Field(..., ge=0, example=84)
    NumCreditLines: int = Field(..., ge=0, le=20, example=3)

    @validator("Income", "LoanAmount")
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("El valor debe ser mayor que cero")
        return v


# ── Esquema de salida ────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    model: str
    default_probability: float
    predicted_class: int
    risk_level: str
    message: str
    timestamp: str


# ── Helpers ──────────────────────────────────────────────────────────────────
def interpret_risk(prob: float):
    if prob < 0.25:
        return (
            "Riesgo bajo",
            "Perfil sólido. Préstamo recomendado sin condiciones especiales.",
        )
    elif prob < 0.45:
        return (
            "Riesgo moderado",
            "Perfil aceptable. Considerar monto menor o garantía adicional.",
        )
    elif prob < 0.65:
        return (
            "Riesgo medio-alto",
            "Revisar historial detallado. Se recomienda tasa diferenciada.",
        )
    else:
        return (
            "Riesgo alto",
            "Alta probabilidad de default. Préstamo no recomendado.",
        )


def build_input_array(data: LoanRequest) -> np.ndarray:
    payload = {
        "Age": data.Age,
        "Income": data.Income,
        "LoanAmount": data.LoanAmount,
        "CreditScore": data.CreditScore,
        "MonthsEmployed": data.MonthsEmployed,
        "NumCreditLines": data.NumCreditLines,
    }

    # Respeta el orden definido en features.json
    try:
        values = [payload[feat] for feat in features]
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature faltante en features.json o payload: {str(e)}"
        )

    return np.array([values], dtype=float)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Sistema"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_xgb is not None,
        "rf_loaded": model_rf is not None,
        "features": features,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(data: LoanRequest, model: str = "xgb"):
    if model_xgb is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    X = build_input_array(data)

    if model == "rf":
        if model_rf is None:
            raise HTTPException(
                status_code=404,
                detail="Random Forest no está disponible. Usa model=xgb."
            )
        clf = model_rf
        model_name = "Random Forest"
    else:
        clf = model_xgb
        model_name = "XGBoost"

    prob = float(clf.predict_proba(X)[0][1])
    clase = int(clf.predict(X)[0])
    risk_level, message = interpret_risk(prob)

    return PredictionResponse(
        model=model_name,
        default_probability=round(prob, 4),
        predicted_class=clase,
        risk_level=risk_level,
        message=message,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model-info", tags=["Sistema"])
def model_info():
    return {
        "primary_model": "XGBoost",
        "secondary_model": "Random Forest" if model_rf is not None else None,
        "features": features,
        "metrics_testset": {
            "accuracy": 0.671,
            "auc": 0.720,
            "recall": 0.647,
            "dataset_default_rate": 0.116,
        },
        "thresholds": {
            "low_risk": "prob < 0.25",
            "moderate_risk": "0.25 ≤ prob < 0.45",
            "medium_high": "0.45 ≤ prob < 0.65",
            "high_risk": "prob ≥ 0.65",
        },
    }