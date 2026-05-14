from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Base-de-datos" / "cleaned" / "Hipertension_Arterial_Mexico_cleaned.csv"
MODELS_DIR = ROOT_DIR / "models"
RANDOM_STATE = 42


def evaluate(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "proba_min": float(np.min(y_proba)),
        "proba_median": float(np.median(y_proba)),
        "proba_max": float(np.max(y_proba)),
        "pct_proba_ge_90": float(np.mean(y_proba >= 0.90)),
        "pct_proba_ge_99": float(np.mean(y_proba >= 0.99)),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    target = "riesgo_hipertension"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "CalibratedLogisticRegression": CalibratedClassifierCV(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            method="sigmoid",
            cv=5,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=80,
            max_depth=2,
            min_samples_leaf=20,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }

    trained = {}
    metrics = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        trained[name] = model
        metrics[name] = evaluate(y_test, y_pred, y_proba)

    best_name = "CalibratedLogisticRegression"
    metadata = {
        "target_column": target,
        "class_labels": {"0": "Sin riesgo", "1": "Riesgo de hipertension"},
        "feature_columns": list(X.columns),
        "best_model_name": best_name,
        "selection_policy": "Se usa regresion logistica calibrada por prudencia medica; modelos de arboles quedan como auditoria por mostrar probabilidades demasiado extremas.",
        "models_compared": list(trained.keys()),
        "test_metrics": metrics,
        "imputation_medians": {},
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "scaler": "StandardScaler fitted on train only",
        "probability_note": "Probabilidad calibrada aproximada; no equivale a certeza ni diagnostico medico.",
        "probability_upper_bound": 0.95,
        "probability_upper_bound_reason": "Sin validacion clinica externa, la app limita HTA a 95% para evitar presentar certeza artificial en entradas fuera de distribucion.",
    }

    model = trained[best_name]
    joblib.dump(model, MODELS_DIR / "hipertension_best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "hipertension_scaler.joblib")
    joblib.dump({"model": model, "scaler": scaler, "metadata": metadata}, MODELS_DIR / "hipertension_model.joblib")
    (MODELS_DIR / "hipertension_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Modelo de hipertension guardado: CalibratedLogisticRegression")


if __name__ == "__main__":
    main()
