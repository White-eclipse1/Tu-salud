from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Base-de-datos" / "cleaned" / "Medicaldataset_cleaned.csv"
MODELS_DIR = ROOT_DIR / "models"
RANDOM_STATE = 25


def evaluate(model, X_train, X_test, y_train, y_test) -> tuple[object, dict[str, float]]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.full(len(y_test), np.nan)
    return model, {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def key(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Result"])
    y = df["Result"].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns, index=X_train_raw.index)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns, index=X_test_raw.index)

    candidates = {
        "Support Vector Machine": svm.SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }
    trained = {}
    metrics = {}
    for name, model in candidates.items():
        trained[name], metrics[key(name)] = evaluate(model, X_train, X_test, y_train, y_test)

    best_key = max(metrics, key=lambda name: metrics[name]["f1"])
    best_display_name = next(name for name in candidates if key(name) == best_key)
    best_model = trained[best_display_name]

    metadata = {
        "problem": "paro_cardiaco_prediction",
        "source_dataset": "Base-de-datos/raw/Medicaldataset.csv",
        "cleaned_dataset": "Base-de-datos/cleaned/Medicaldataset_cleaned.csv",
        "target_column": "Result",
        "class_labels": {"0": "Negative", "1": "Positive"},
        "best_model": best_key,
        "model_path": "models/paro_cardiaco_best_model.joblib",
        "scaler_path": "models/paro_cardiaco_scaler.joblib",
        "feature_columns": list(X.columns),
        "label_mapping": {"negative": 0, "positive": 1},
        "test_metrics_by_model": metrics,
        "preprocessing_note": "StandardScaler ajustado unicamente con X_train para evitar fuga de datos.",
    }
    joblib.dump(best_model, MODELS_DIR / "paro_cardiaco_best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "paro_cardiaco_scaler.joblib")
    joblib.dump({"model": best_model, "scaler": scaler, "metadata": metadata}, MODELS_DIR / "paro_cardiaco_model.joblib")
    (MODELS_DIR / "paro_cardiaco_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Modelo de paro cardiaco guardado: {best_key}")


if __name__ == "__main__":
    main()
