"""
CreditMind - prototipo de entrenamiento y dashboard.

Flujo principal:
1. Cargar/perfilar el dataset de default crediticio.
2. Entrenar el algoritmo elegido.
3. Guardar modelo y resultados.
4. Exponer resultados para el dashboard.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "trained_models"
RESULTS_PATH = ARTIFACTS_DIR / "training_runs.json"
REGISTRY_PATH = ARTIFACTS_DIR / "model_registry.json"
NOTEBOOK_REGISTRY_PATH = ARTIFACTS_DIR / "model_registry_notebook.json"
LEGACY_XGB_PATH = ARTIFACTS_DIR / "model_xgb.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"

ENV_DATASET_PATH = os.getenv("CREDITMIND_DATASET")
DATASET_CANDIDATES = [
    Path(ENV_DATASET_PATH).expanduser() if ENV_DATASET_PATH else None,
    BASE_DIR / "Loan_default_limpio.csv",
    ARTIFACTS_DIR / "Loan_default_limpio.csv",
    BASE_DIR / "data" / "Loan_default_limpio.csv",
]
LOCAL_DATASET_PATH = BASE_DIR / "data" / "Loan_default_limpio.csv"
TARGET_COLUMN = "Default"

ARTIFACTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="CreditMind API",
    description="Carga datasets, entrena modelos de default y entrega resultados para dashboard.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    algorithm: str = Field(..., examples=["random_forest"])
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    sample_size: int | None = Field(
        80000,
        ge=1000,
        description="Submuestra estratificada para mantener la demo ágil. Usa null para todo el dataset.",
    )
    random_state: int = 42


class LoanRequest(BaseModel):
    Age: int = Field(..., ge=18, le=85)
    Income: float = Field(..., ge=0)
    LoanAmount: float = Field(..., ge=0)
    CreditScore: int = Field(..., ge=300, le=850)
    MonthsEmployed: int = Field(..., ge=0)
    NumCreditLines: int = Field(..., ge=0)
    InterestRate: float = Field(13.49, ge=0)
    LoanTerm: int = Field(36, ge=1)
    DTIRatio: float = Field(0.5, ge=0, le=1)
    HasMortgage: int = Field(0, ge=0, le=1)
    HasDependents: int = Field(0, ge=0, le=1)
    HasCoSigner: int = Field(0, ge=0, le=1)
    Education: str = "Bachelor"
    EmploymentType: str = "Full-time"
    MaritalStatus: str = "Divorced"
    LoanPurpose: str = "Auto"


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    algorithm: str
    model_name: str
    default_probability: float
    predicted_class: int
    risk_level: str
    decision: str
    message: str
    timestamp: str


def read_runs() -> list[dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_runs(runs: list[dict[str, Any]]) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)


def read_registry() -> dict[str, Any]:
    registry: dict[str, Any] = {}

    for path in [REGISTRY_PATH, NOTEBOOK_REGISTRY_PATH]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                registry.update(json.load(f))

    if "xgboost" not in registry and LEGACY_XGB_PATH.exists():
        registry["xgboost"] = {
            "id": "legacy_xgb",
            "algorithm": "xgboost",
            "model_name": "XGBoost",
            "model_path": str(LEGACY_XGB_PATH),
            "features": read_features_file(),
            "metrics": {
                "accuracy": 0.6707,
                "precision": 0.2068,
                "recall": 0.6474,
                "f1": 0.3135,
                "auc": 0.7196,
            },
        }

    return normalize_registry(registry)


def write_registry(registry: dict[str, Any]) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def read_features_file() -> list[str]:
    if not FEATURES_PATH.exists():
        return [
            "Age",
            "Income",
            "LoanAmount",
            "CreditScore",
            "MonthsEmployed",
            "NumCreditLines",
        ]
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_algorithm(algorithm: str) -> str:
    value = algorithm.lower().strip()
    aliases = {
        "lr": "linear_regression",
        "rl": "linear_regression",
        "logistic_regression": "linear_regression",
        "regresion_logistica": "linear_regression",
        "regresion_lineal": "linear_regression",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "xgb": "xgboost",
        "xgboost_optimizado": "xgboost",
        "nn": "neural_network",
        "mlp": "neural_network",
        "red_neuronal": "neural_network",
    }
    return aliases.get(value, value)


def normalize_registry(registry: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    display_names = {
        "linear_regression": "Regresión Logística",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "neural_network": "Red Neuronal",
    }

    for key, value in registry.items():
        canonical = normalize_algorithm(value.get("algorithm", key))
        item = dict(value)
        item["algorithm"] = canonical
        item["model_name"] = item.get("model_name") or display_names.get(canonical, canonical)
        item["id"] = item.get("id") or canonical
        normalized[canonical] = item

    return normalized


def dataset_path() -> Path:
    for candidate in DATASET_CANDIDATES:
        if candidate and candidate.exists():
            return candidate
    raise HTTPException(
        status_code=404,
        detail=(
            "No se encontró Loan_default_limpio.csv. Colócalo en la raíz del proyecto, "
            "en data/, en artifacts/ o define la variable CREDITMIND_DATASET."
        ),
    )


def available_dataset_path() -> Path | None:
    for candidate in DATASET_CANDIDATES:
        if candidate and candidate.exists():
            return candidate
    return None


def load_dataset() -> pd.DataFrame:
    path = dataset_path()
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"El dataset debe contener la columna objetivo '{TARGET_COLUMN}'.",
        )
    return df


def dataframe_profile(df: pd.DataFrame) -> dict[str, Any]:
    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    target_pct = (df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100)
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    correlations = (
        df[numeric_cols]
        .corr(numeric_only=False)[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN], errors="ignore")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(10)
        .round(4)
    )

    return {
        "source": str(dataset_path()),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": len(feature_cols),
        "target": TARGET_COLUMN,
        "missing_values": int(df.isna().sum().sum()),
        "default_count": int(target_counts.get(1, 0)),
        "non_default_count": int(target_counts.get(0, 0)),
        "default_rate": round(float(target_pct.get(1, 0)), 3),
        "columns_list": df.columns.tolist(),
        "top_correlations": [
            {"feature": idx, "correlation": float(value)}
            for idx, value in correlations.items()
        ],
    }


def stratified_sample(
    df: pd.DataFrame,
    sample_size: int | None,
    random_state: int,
) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df

    pieces = []
    for _, group in df.groupby(TARGET_COLUMN):
        n = max(1, round(sample_size * len(group) / len(df)))
        pieces.append(group.sample(n=min(n, len(group)), random_state=random_state))
    return pd.concat(pieces).sample(frac=1, random_state=random_state).reset_index(drop=True)


def build_model(algorithm: str, random_state: int):
    normalized = algorithm.lower().strip()

    if normalized in {"linear_regression", "regresion_lineal", "logistic_regression"}:
        return (
            "linear_regression",
            "Regresión lineal (logística binaria)",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=600,
                            random_state=random_state,
                            solver="liblinear",
                        ),
                    ),
                ]
            ),
        )

    if normalized in {"random_forest", "randomforest", "rf"}:
        return (
            "random_forest",
            "Random Forest",
            RandomForestClassifier(
                n_estimators=180,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
        )

    if normalized in {"neural_network", "red_neuronal", "mlp"}:
        return (
            "neural_network",
            "Red neuronal",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(48, 24),
                            activation="relu",
                            alpha=0.001,
                            batch_size=512,
                            learning_rate_init=0.001,
                            early_stopping=True,
                            max_iter=80,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        )

    raise HTTPException(
        status_code=400,
        detail="Algoritmo no soportado. Usa linear_regression, random_forest o neural_network.",
    )


def feature_strength(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    elif hasattr(estimator, "coefs_"):
        values = np.abs(estimator.coefs_[0]).sum(axis=1)
    else:
        return []

    total = float(np.sum(np.abs(values))) or 1.0
    rows = [
        {
            "feature": feature,
            "importance": round(float(abs(value) / total), 5),
        }
        for feature, value in zip(feature_names, values)
    ]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)[:12]


def class_probability(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    scores = model.decision_function(X_test)
    return 1 / (1 + np.exp(-scores))


MODEL_CACHE: dict[str, Any] = {}


def load_saved_model(algorithm: str) -> tuple[str, dict[str, Any], Any]:
    registry = read_registry()
    canonical = normalize_algorithm(algorithm)

    if canonical not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{algorithm}' no disponible. Modelos: {list(registry.keys())}",
        )

    meta = registry[canonical]
    model_path = Path(meta["model_path"])
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"El artefacto del modelo no existe: {model_path}",
        )

    cache_key = f"{canonical}:{model_path}"
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = joblib.load(model_path)

    return canonical, meta, MODEL_CACHE[cache_key]


def model_features(meta: dict[str, Any], model: Any) -> list[str]:
    if meta.get("features"):
        return list(meta["features"])

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if isinstance(model, Pipeline):
        for _, step in reversed(model.steps):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    df_columns = [c for c in load_dataset().columns.tolist() if c != TARGET_COLUMN]
    expected = meta.get("dataset", {}).get("features_used")
    if expected == len(df_columns):
        return df_columns

    return read_features_file()


def base_payload(data: LoanRequest) -> dict[str, Any]:
    payload = data.model_dump()

    one_hot = {
        "Education_High School": data.Education == "High School",
        "Education_Master's": data.Education == "Master's",
        "Education_PhD": data.Education == "PhD",
        "EmploymentType_Part-time": data.EmploymentType == "Part-time",
        "EmploymentType_Self-employed": data.EmploymentType == "Self-employed",
        "EmploymentType_Unemployed": data.EmploymentType == "Unemployed",
        "MaritalStatus_Married": data.MaritalStatus == "Married",
        "MaritalStatus_Single": data.MaritalStatus == "Single",
        "LoanPurpose_Business": data.LoanPurpose == "Business",
        "LoanPurpose_Education": data.LoanPurpose == "Education",
        "LoanPurpose_Home": data.LoanPurpose == "Home",
        "LoanPurpose_Other": data.LoanPurpose == "Other",
    }

    payload.update({key: int(value) for key, value in one_hot.items()})
    return payload


def build_prediction_frame(data: LoanRequest, features: list[str]) -> pd.DataFrame:
    payload = base_payload(data)
    df = load_dataset()

    values: dict[str, Any] = {}
    for feature in features:
        if feature in payload:
            values[feature] = payload[feature]
        elif feature in df.columns:
            if df[feature].dtype == bool:
                values[feature] = False
            else:
                values[feature] = float(df[feature].median())
        else:
            values[feature] = 0

    return pd.DataFrame([values], columns=features)


def interpret_risk(prob: float) -> tuple[str, str, str]:
    if prob < 0.30:
        return (
            "Riesgo bajo",
            "APROBADO",
            "Perfil favorable. El crédito puede avanzar bajo condiciones estándar.",
        )
    if prob < 0.60:
        return (
            "Riesgo medio",
            "REVISAR",
            "Conviene revisar capacidad de pago, monto solicitado o garantías.",
        )
    return (
        "Riesgo alto",
        "RECHAZADO",
        "Alta probabilidad estimada de default. No se recomienda aprobar en estas condiciones.",
    )


def predict_with_algorithm(data: LoanRequest, algorithm: str) -> PredictionResponse:
    try:
        canonical, meta, model = load_saved_model(algorithm)
        features = model_features(meta, model)
        X = build_prediction_frame(data, features)
        prob = float(class_probability(model, X)[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"No se pudo usar el modelo '{algorithm}'. "
                "Verifica que el backend tenga la misma versión de librerías con la que se entrenó. "
                f"Detalle: {type(exc).__name__}: {exc}"
            ),
        )

    predicted_class = int(prob >= 0.5)
    risk_level, decision, message = interpret_risk(prob)

    return PredictionResponse(
        algorithm=canonical,
        model_name=meta["model_name"],
        default_probability=round(prob, 4),
        predicted_class=predicted_class,
        risk_level=risk_level,
        decision=decision,
        message=message,
        timestamp=datetime.now().isoformat(),
    )


def enrich_model_for_dashboard(algorithm: str, meta: dict[str, Any]) -> dict[str, Any]:
    item = dict(meta)

    if item.get("confusion_matrix") and item.get("feature_importance"):
        return item

    try:
        canonical, _, model = load_saved_model(algorithm)
        features = model_features(item, model)
        df = load_dataset()
        X = df[features]
        y = df[TARGET_COLUMN].astype(int)
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        y_pred = model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

        item.setdefault("dataset", {})
        item["dataset"].setdefault("rows_used", int(df.shape[0]))
        item["dataset"].setdefault("features_used", len(features))
        item["dataset"].setdefault("test_rows", int(X_test.shape[0]))
        item["dataset"].setdefault("default_rate", round(float(y.mean() * 100), 3))
        item["confusion_matrix"] = {
            "tn": int(matrix[0][0]),
            "fp": int(matrix[0][1]),
            "fn": int(matrix[1][0]),
            "tp": int(matrix[1][1]),
        }
        item["feature_importance"] = feature_strength(model, features)
        item["algorithm"] = canonical
    except Exception as exc:
        item["dashboard_error"] = f"{type(exc).__name__}: {exc}"

    return item


@app.get("/health", tags=["Sistema"])
def health_check():
    dataset = available_dataset_path()
    registry = read_registry()
    return {
        "status": "ok",
        "dataset_available": dataset is not None,
        "dataset_path": str(dataset) if dataset else None,
        "models_available": list(registry.keys()),
        "runs_saved": len(read_runs()),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/", tags=["Sistema"])
def root():
    return {
        "service": "CreditMind API",
        "status": "online",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/dataset", tags=["Dataset"])
def get_dataset_profile():
    df = load_dataset()
    return dataframe_profile(df)


@app.post("/dataset/upload", tags=["Dataset"])
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Sólo se aceptan archivos CSV.")

    content = await file.read()
    LOCAL_DATASET_PATH.parent.mkdir(exist_ok=True)
    LOCAL_DATASET_PATH.write_bytes(content)

    df = load_dataset()
    return dataframe_profile(df)


@app.post("/train", tags=["Entrenamiento"])
def train_model(request: TrainRequest):
    df = load_dataset()
    df = stratified_sample(df, request.sample_size, request.random_state)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=request.test_size,
        random_state=request.random_state,
        stratify=y,
    )

    algorithm, model_name, model = build_model(request.algorithm, request.random_state)
    started = datetime.now()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = class_probability(model, X_test)
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    run_id = uuid4().hex[:10]
    model_path = MODELS_DIR / f"{algorithm}_{run_id}.pkl"
    joblib.dump(model, model_path)

    result = {
        "id": run_id,
        "algorithm": algorithm,
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "training_seconds": round((datetime.now() - started).total_seconds(), 2),
        "dataset": {
            "rows_used": int(df.shape[0]),
            "features_used": int(X.shape[1]),
            "test_rows": int(X_test.shape[0]),
            "default_rate": round(float(y.mean() * 100), 3),
        },
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        },
        "confusion_matrix": {
            "tn": int(matrix[0][0]),
            "fp": int(matrix[0][1]),
            "fn": int(matrix[1][0]),
            "tp": int(matrix[1][1]),
        },
        "feature_importance": feature_strength(model, X.columns.tolist()),
        "model_path": str(model_path),
    }

    runs = read_runs()
    runs.insert(0, result)
    write_runs(runs[:25])

    registry = read_registry()
    previous = registry.get(algorithm)
    if previous is None or result["metrics"]["auc"] >= previous["metrics"]["auc"]:
        registry[algorithm] = result
        write_registry(registry)

    return result


@app.get("/models", tags=["Modelos guardados"])
def list_saved_models():
    registry = read_registry()
    enriched = {
        algorithm: enrich_model_for_dashboard(algorithm, meta)
        for algorithm, meta in registry.items()
    }
    return {
        "models": enriched,
        "available_algorithms": list(enriched.keys()),
        "message": (
            "Estos son los modelos persistidos para uso normal del prototipo. "
            "El endpoint /train queda como preparacion offline para guardar nuevas versiones."
        ),
    }


@app.get("/models/{algorithm}", tags=["Modelos guardados"])
def get_saved_model(algorithm: str):
    registry = read_registry()
    canonical = normalize_algorithm(algorithm)
    if canonical not in registry:
        raise HTTPException(status_code=404, detail="Modelo guardado no encontrado.")
    return registry[canonical]


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(
    data: LoanRequest,
    model: str = Query("xgboost", description="linear_regression, random_forest, xgboost o neural_network"),
):
    return predict_with_algorithm(data, model)


@app.post("/predict-all", tags=["Predicción"])
def predict_all(data: LoanRequest):
    registry = read_registry()
    responses = []

    for algorithm in ["linear_regression", "random_forest", "xgboost", "neural_network"]:
        if algorithm not in registry:
            continue
        try:
            responses.append(predict_with_algorithm(data, algorithm).model_dump())
        except HTTPException as exc:
            responses.append(
                {
                    "algorithm": algorithm,
                    "model_name": registry[algorithm].get("model_name", algorithm),
                    "error": exc.detail,
                }
            )

    if not responses:
        raise HTTPException(status_code=404, detail="No hay modelos disponibles para comparar.")

    successful = [item for item in responses if "default_probability" in item]
    consensus = None
    if successful:
        avg_prob = float(np.mean([item["default_probability"] for item in successful]))
        risk_level, decision, message = interpret_risk(avg_prob)
        consensus = {
            "average_default_probability": round(avg_prob, 4),
            "risk_level": risk_level,
            "decision": decision,
            "message": message,
        }

    return {
        "predictions": responses,
        "consensus": consensus,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/runs", tags=["Resultados"])
def list_runs():
    return {"runs": read_runs()}


@app.get("/runs/{run_id}", tags=["Resultados"])
def get_run(run_id: str):
    for run in read_runs():
        if run["id"] == run_id:
            return run
    raise HTTPException(status_code=404, detail="Corrida no encontrada.")
