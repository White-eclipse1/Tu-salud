from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Base-de-datos" / "cleaned" / "diabetes_cleaned.csv"
MODELS_DIR = ROOT_DIR / "models"
RANDOM_STATE = 42


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 120],
        labels=["Young", "Adult", "Middle-aged", "Senior"],
    )
    df["Glucose_Category"] = pd.cut(
        df["Glucose"],
        bins=[0, 100, 125, 300],
        labels=["Normal", "Prediabetes", "Diabetes"],
    )
    df["Insulin_Category"] = pd.cut(
        df["Insulin"],
        bins=[0, 16, 166, 1000],
        labels=["Low", "Normal", "High"],
    )
    df["BP_Category"] = pd.cut(
        df["BloodPressure"],
        bins=[0, 80, 90, 200],
        labels=["Normal", "Elevated", "High"],
    )
    return df


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df_encoded = pd.get_dummies(
        add_features(df),
        columns=["BMI_Category", "AgeGroup", "Glucose_Category", "Insulin_Category", "BP_Category"],
        drop_first=True,
    )

    X = df_encoded.drop(columns=["Outcome"])
    y = df_encoded["Outcome"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    medians = {column: float(df[column].median()) for column in zero_invalid_cols}
    metadata = {
        "problem": "diabetes_prediction",
        "source_dataset": "Base-de-datos/raw/diabetes.csv",
        "cleaned_dataset": "Base-de-datos/cleaned/diabetes_cleaned.csv",
        "target_column": "Outcome",
        "class_labels": {"0": "No diabetes", "1": "Diabetes"},
        "model_type": "RandomForestClassifier",
        "model_path": "models/diabetes_random_forest.joblib",
        "feature_columns": list(X.columns),
        "zero_invalid_columns": zero_invalid_cols,
        "imputation_medians": medians,
        "categorical_columns": [
            "BMI_Category",
            "AgeGroup",
            "Glucose_Category",
            "Insulin_Category",
            "BP_Category",
        ],
        "category_definitions": {
            "BMI_Category": {"bins": [0, 18.5, 25, 30, 100], "labels": ["Underweight", "Normal", "Overweight", "Obese"]},
            "AgeGroup": {"bins": [0, 30, 45, 60, 100], "labels": ["Young", "Adult", "Middle-aged", "Senior"]},
            "Glucose_Category": {"bins": [0, 100, 125, 200], "labels": ["Normal", "Prediabetes", "Diabetes"]},
            "Insulin_Category": {"bins": [0, 16, 166, 1000], "labels": ["Low", "Normal", "High"]},
            "BP_Category": {"bins": [0, 80, 90, 200], "labels": ["Normal", "Elevated", "High"]},
        },
        "best_params": grid_search.best_params_,
        "best_cv_f1": float(grid_search.best_score_),
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
    }

    joblib.dump(model, MODELS_DIR / "diabetes_random_forest.joblib")
    joblib.dump({"model": model, "metadata": metadata}, MODELS_DIR / "diabetes_model.joblib")
    (MODELS_DIR / "diabetes_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Modelo de diabetes guardado")


if __name__ == "__main__":
    main()
