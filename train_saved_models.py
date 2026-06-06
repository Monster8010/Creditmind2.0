"""
Entrenamiento offline de modelos CreditMind.

Ejecuta este script una vez para generar las versiones persistidas que usara
el prototipo. El frontend normal no reentrena: solo lee artifacts/model_registry.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
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


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "trained_models"
REGISTRY_PATH = ARTIFACTS_DIR / "model_registry.json"
RUNS_PATH = ARTIFACTS_DIR / "training_runs.json"
DEFAULT_DATASET_PATH = Path(
    r"C:\Users\Antonio\Documents\Curso samsung\Course\Notebook\Loan_default_limpio.csv"
)
TARGET_COLUMN = "Default"

ARTIFACTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    local = ARTIFACTS_DIR / "Loan_default_limpio.csv"
    path = local if local.exists() else DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            "No encontre Loan_default_limpio.csv. Copialo a artifacts/ o conserva la ruta original."
        )
    return pd.read_csv(path)


def build_models(random_state: int):
    return {
        "linear_regression": (
            "Regresion lineal (logistica binaria)",
            Pipeline(
                [
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
        ),
        "random_forest": (
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
        ),
        "neural_network": (
            "Red neuronal",
            Pipeline(
                [
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
        ),
    }


def feature_strength(model, feature_names: list[str]):
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return []

    total = float(np.sum(np.abs(values))) or 1.0
    rows = [
        {"feature": feature, "importance": round(float(abs(value) / total), 5)}
        for feature, value in zip(feature_names, values)
    ]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)[:12]


def evaluate(model, X_test, y_test, feature_names, algorithm, model_name, rows_used):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    run_id = uuid4().hex[:10]
    model_path = MODELS_DIR / f"{algorithm}_{run_id}.pkl"
    joblib.dump(model, model_path)

    return {
        "id": run_id,
        "algorithm": algorithm,
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "training_seconds": None,
        "dataset": {
            "rows_used": int(rows_used),
            "features_used": len(feature_names),
            "test_rows": int(len(X_test)),
            "default_rate": round(float(y_test.mean() * 100), 3),
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
        "feature_importance": feature_strength(model, feature_names),
        "model_path": str(model_path),
    }


def main(sample_size: int | None = 80000, random_state: int = 42):
    df = load_dataset()
    if sample_size and sample_size < len(df):
        df = (
            df.groupby(TARGET_COLUMN, group_keys=False)
            .apply(lambda g: g.sample(max(1, round(sample_size * len(g) / len(df))), random_state=random_state))
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    registry = {}
    runs = []
    for algorithm, (model_name, model) in build_models(random_state).items():
        print(f"Entrenando {model_name}...")
        start = datetime.now()
        model.fit(X_train, y_train)
        result = evaluate(model, X_test, y_test, X.columns.tolist(), algorithm, model_name, len(df))
        result["training_seconds"] = round((datetime.now() - start).total_seconds(), 2)
        registry[algorithm] = result
        runs.append(result)
        print(f"  AUC={result['metrics']['auc']} F1={result['metrics']['f1']} guardado en {result['model_path']}")

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)

    print(f"\nListo. Registro guardado en {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
