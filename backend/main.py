from __future__ import annotations

import math
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


app = FastAPI(title="Tu Salud API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_bundle(name: str) -> dict[str, Any]:
    path = MODELS_DIR / name
    if not path.exists():
        raise RuntimeError(f"No existe el artefacto: {path}")
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        return artifact
    return {"model": artifact, "metadata": {}}


@app.on_event("startup")
def startup() -> None:
    app.state.diabetes = load_bundle("diabetes_model.joblib")
    app.state.hipertension = load_bundle("hipertension_model.joblib")
    app.state.paro = load_bundle("paro_cardiaco_model.joblib")
    app.state.subtipos = load_bundle("subtipos_kmeans.joblib")
    with open(MODELS_DIR / "subtipos_metadata.json", encoding="utf-8") as file:
        app.state.subtipos_metadata = json.load(file)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict[str, Any]) -> dict[str, Any]:
    try:
        diabetes = predict_diabetes(app.state.diabetes, data)
        hipert = predict_hipertension(app.state.hipertension, data)
        cardio = predict_paro_cardiaco(app.state.paro, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {exc}") from exc

    general_pct = round(diabetes["pct"] * 0.3 + hipert["pct"] * 0.35 + cardio["pct"] * 0.35)
    return {
        "diabetes": diabetes,
        "hipert": hipert,
        "cardio": cardio,
        "general": {"pct": general_pct, "level": classify(general_pct)},
    }


@app.post("/cluster")
def cluster(data: dict[str, Any]) -> dict[str, Any]:
    try:
        diabetes = predict_cluster(
            "diabetes",
            app.state.subtipos["diabetes"],
            app.state.subtipos_metadata,
            build_diabetes_raw_features(data),
        )
        hipert = predict_cluster(
            "hipertension",
            app.state.subtipos["hipertension"],
            app.state.subtipos_metadata,
            build_hipertension_features(data),
        )
        cardio = predict_cluster(
            "paro_cardiaco",
            app.state.subtipos["paro_cardiaco"],
            app.state.subtipos_metadata,
            build_paro_cardiaco_features(data),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al calcular clusters: {exc}") from exc

    cluster_mean = round(
        diabetes["proba_media_modelo"] * 0.3
        + hipert["proba_media_modelo"] * 0.35
        + cardio["proba_media_modelo"] * 0.35,
        3,
    )
    return {
        "diabetes": diabetes,
        "hipert": hipert,
        "cardio": cardio,
        "general": {
            "cluster_risk_pct": round(cluster_mean * 100),
            "level": classify(round(cluster_mean * 100)),
        },
    }


def predict_diabetes(bundle: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle["metadata"]
    features = {column: 0.0 for column in metadata["feature_columns"]}

    raw_values = {
        "Pregnancies": number(data, "pregnancies", default=0),
        "Glucose": number(data, "glucose"),
        "BloodPressure": number(data, "bp_diastolic"),
        "SkinThickness": number(data, "skin_thickness"),
        "Insulin": number(data, "insulin"),
        "BMI": number(data, "bmi"),
        "DiabetesPedigreeFunction": number(data, "diabetes_pedigree"),
        "Age": number(data, "age"),
    }

    for column in metadata.get("zero_invalid_columns", []):
        if raw_values.get(column) == 0:
            raw_values[column] = metadata.get("imputation_medians", {}).get(column, raw_values[column])

    features.update({key: value for key, value in raw_values.items() if key in features})
    add_diabetes_categories(features, metadata, raw_values)
    probability = probability_for(bundle["model"], pd.DataFrame([features], columns=metadata["feature_columns"]))
    return format_risk(probability)


def build_diabetes_raw_features(data: dict[str, Any]) -> pd.DataFrame:
    features = {
        "Pregnancies": number(data, "pregnancies", default=0),
        "Glucose": number(data, "glucose"),
        "BloodPressure": number(data, "bp_diastolic"),
        "SkinThickness": number(data, "skin_thickness"),
        "Insulin": number(data, "insulin"),
        "BMI": number(data, "bmi"),
        "DiabetesPedigreeFunction": number(data, "diabetes_pedigree"),
        "Age": number(data, "age"),
    }
    return pd.DataFrame([features], columns=list(features.keys()))


def add_diabetes_categories(features: dict[str, float], metadata: dict[str, Any], raw: dict[str, float]) -> None:
    source_map = {
        "BMI_Category": raw["BMI"],
        "AgeGroup": raw["Age"],
        "Glucose_Category": raw["Glucose"],
        "Insulin_Category": raw["Insulin"],
        "BP_Category": raw["BloodPressure"],
    }
    for category_name, value in source_map.items():
        definition = metadata["category_definitions"][category_name]
        label = bucket_label(value, definition["bins"], definition["labels"])
        if label is None:
            continue
        dummy_column = f"{category_name}_{label}"
        if dummy_column in features:
            features[dummy_column] = 1.0


def predict_hipertension(bundle: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle["metadata"]
    features = {}
    for column in metadata["feature_columns"]:
        features[column] = optional_number(data, column)

    features["sexo"] = hypertension_sex_code(data)
    features["edad"] = number(data, "age")
    features["resultado_glucosa"] = number(data, "glucose")
    features["valor_insulina"] = number(data, "insulin")
    features["valor_colesterol_total"] = number(data, "cholesterol_total")
    features["tension_arterial"] = number(data, "bp_systolic")
    features["masa_corporal"] = number(data, "bmi")

    missing = [key for key, value in features.items() if value is None or not math.isfinite(value)]
    if missing:
        raise ValueError("Faltan campos para hipertension: " + ", ".join(missing))

    frame = pd.DataFrame([features], columns=metadata["feature_columns"])
    scaler = bundle.get("scaler")
    if scaler is not None:
        frame = pd.DataFrame(scaler.transform(frame), columns=metadata["feature_columns"])
    probability = probability_for(bundle["model"], frame)
    return format_risk(probability)


def build_hipertension_features(data: dict[str, Any]) -> pd.DataFrame:
    metadata = load_metadata("hipertension_metadata.json")
    features = {}
    for column in metadata["feature_columns"]:
        features[column] = optional_number(data, column)

    features["sexo"] = hypertension_sex_code(data)
    features["edad"] = number(data, "age")
    features["resultado_glucosa"] = number(data, "glucose")
    features["valor_insulina"] = number(data, "insulin")
    features["valor_colesterol_total"] = number(data, "cholesterol_total")
    features["tension_arterial"] = number(data, "bp_systolic")
    features["masa_corporal"] = number(data, "bmi")

    missing = [key for key, value in features.items() if value is None or not math.isfinite(value)]
    if missing:
        raise ValueError("Faltan campos para K-Means hipertension: " + ", ".join(missing))

    return pd.DataFrame([features], columns=metadata["feature_columns"])


def predict_paro_cardiaco(bundle: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle["metadata"]
    features = {
        "Age": number(data, "age"),
        "Gender": cardiac_gender_code(data),
        "Heart rate": number(data, "heart_rate"),
        "Systolic blood pressure": number(data, "bp_systolic"),
        "Diastolic blood pressure": number(data, "bp_diastolic"),
        "Blood sugar": number(data, "blood_sugar", default=number(data, "glucose")),
        "CK-MB": number(data, "ck_mb"),
        "Troponin": number(data, "troponin"),
    }
    frame = pd.DataFrame([features], columns=metadata["feature_columns"])
    scaler = bundle.get("scaler")
    if scaler is not None:
        frame = pd.DataFrame(scaler.transform(frame), columns=metadata["feature_columns"])
    probability = probability_for(bundle["model"], frame)
    return format_risk(probability)


def build_paro_cardiaco_features(data: dict[str, Any]) -> pd.DataFrame:
    features = {
        "Age": number(data, "age"),
        "Gender": cardiac_gender_code(data),
        "Heart rate": number(data, "heart_rate"),
        "Systolic blood pressure": number(data, "bp_systolic"),
        "Diastolic blood pressure": number(data, "bp_diastolic"),
        "Blood sugar": number(data, "blood_sugar", default=number(data, "glucose")),
        "CK-MB": number(data, "ck_mb"),
        "Troponin": number(data, "troponin"),
    }
    return pd.DataFrame([features], columns=list(features.keys()))


def predict_cluster(
    disease: str,
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    features: pd.DataFrame,
) -> dict[str, Any]:
    scaler = artifact["scaler"]
    kmeans = artifact["kmeans"]
    scaled = scaler.transform(features)
    cluster_id = int(kmeans.predict(scaled)[0])
    distances = kmeans.transform(scaled)[0]
    assigned_distance = float(distances[cluster_id])
    max_distance = max(float(distance) for distance in distances) or 1.0

    summary = next(
        row
        for row in metadata["summary_per_cluster"][disease]
        if int(row["cluster"]) == cluster_id
    )
    groups = []
    for row in metadata["summary_per_cluster"][disease]:
        group_cluster = int(row["cluster"])
        distance = float(distances[group_cluster])
        risk_pct = round(float(row["proba_media_modelo"]) * 100)
        groups.append(
            {
                "cluster": group_cluster,
                "n": int(row["n"]),
                "pct_positivos": float(row["pct_positivos"]),
                "proba_media_modelo": float(row["proba_media_modelo"]),
                "cluster_risk_pct": risk_pct,
                "level": classify(risk_pct),
                "distance_to_center": round(distance, 3),
                "affinity_pct": max(0, min(100, round((1 - distance / max_distance) * 100))),
                "is_patient_group": group_cluster == cluster_id,
            }
        )

    return {
        "cluster": cluster_id,
        "k": int(artifact["k"]),
        "silhouette": round(float(artifact["silhouette"]), 3),
        "n": int(summary["n"]),
        "pct_positivos": float(summary["pct_positivos"]),
        "proba_media_modelo": float(summary["proba_media_modelo"]),
        "cluster_risk_pct": round(float(summary["proba_media_modelo"]) * 100),
        "level": classify(round(float(summary["proba_media_modelo"]) * 100)),
        "distance_to_center": round(assigned_distance, 3),
        "group_label": cluster_group_label(disease, cluster_id),
        "groups": groups,
    }


def load_metadata(name: str) -> dict[str, Any]:
    with open(MODELS_DIR / name, encoding="utf-8") as file:
        return json.load(file)


def probability_for(model: Any, frame: Any) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X .* feature names.*")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(frame)[0]
            if len(probabilities) > 1:
                return float(probabilities[1])
            return float(probabilities[0])
        prediction = model.predict(frame)[0]
        return float(prediction)


def number(data: dict[str, Any], key: str, default: float | None = None) -> float:
    value = data.get(key, default)
    if value in ("", None):
        if default is None:
            raise ValueError(f"Falta el campo requerido: {key}")
        value = default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"El campo {key} debe ser numerico") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"El campo {key} debe ser finito")
    return parsed


def optional_number(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"El campo {key} debe ser numerico") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"El campo {key} debe ser finito")
    return parsed


def hypertension_sex_code(data: dict[str, Any]) -> float:
    value = str(data.get("sex", "")).strip().upper()
    if value in {"1", "M", "MASCULINO", "HOMBRE"}:
        return 1.0
    if value in {"2", "F", "FEMENINO", "MUJER"}:
        return 2.0
    raise ValueError("El campo sex debe ser M/F o 1/2")


def cardiac_gender_code(data: dict[str, Any]) -> float:
    value = str(data.get("sex", "")).strip().upper()
    if value in {"1", "M", "MASCULINO", "HOMBRE"}:
        return 1.0
    if value in {"0", "F", "FEMENINO", "MUJER"}:
        return 0.0
    raise ValueError("El campo sex debe ser M/F o 0/1 para paro cardiaco")


def bucket_label(value: float, bins: list[float], labels: list[str]) -> str | None:
    for index, label in enumerate(labels):
        lower = bins[index]
        upper = bins[index + 1] if index + 1 < len(bins) else float("inf")
        if lower < value <= upper or (index == 0 and lower <= value <= upper):
            return label
    return None


def format_risk(probability: float) -> dict[str, Any]:
    pct = max(0, min(100, round(probability * 100)))
    return {"pct": pct, "level": classify(pct)}


def classify(pct: int) -> str:
    if pct >= 70:
        return "critical"
    if pct >= 45:
        return "high"
    if pct >= 20:
        return "moderate"
    return "low"


def cluster_group_label(disease: str, cluster_id: int) -> str:
    labels = {
        "diabetes": {
            0: "Grupo metabolico alto",
            1: "Grupo metabolico bajo",
        },
        "hipertension": {
            0: "Grupo hipertensivo frecuente",
            1: "Grupo cardiovascular bajo",
            2: "Grupo hipertensivo metabolico",
            3: "Grupo hipertensivo avanzado",
        },
        "paro_cardiaco": {
            0: "Grupo cardiaco intermedio",
            1: "Grupo cardiaco atipico",
            2: "Grupo cardiometabolico comun",
            3: "Grupo cardiaco critico",
            4: "Grupo cardiaco muy alto",
        },
    }
    return labels.get(disease, {}).get(cluster_id, f"Grupo {cluster_id}")
