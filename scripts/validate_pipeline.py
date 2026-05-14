from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
sys.path.insert(0, str(ROOT_DIR))

from backend.main import cluster, predict, startup


SAMPLE_PATIENT = {
    "name": "Paciente DVC",
    "age": 45,
    "sex": "M",
    "pregnancies": 0,
    "glucose": 120,
    "bp_systolic": 120,
    "bp_diastolic": 80,
    "skin_thickness": 29,
    "insulin": 125,
    "bmi": 28,
    "diabetes_pedigree": 0.45,
    "heart_rate": 72,
    "blood_sugar": 120,
    "ck_mb": 2.4,
    "troponin": 0.01,
    "concentracion_hemoglobina": 14.2,
    "temperatura_ambiente": 22,
    "valor_acido_urico": 5.4,
    "valor_albumina": 4.3,
    "valor_colesterol_hdl": 48,
    "valor_colesterol_ldl": 115,
    "cholesterol_total": 190,
    "valor_creatina": 0.9,
    "valor_trigliceridos": 150,
    "resultado_glucosa_promedio": 110,
    "valor_hemoglobina_glucosilada": 5.7,
    "valor_ferritina": 80,
    "valor_folato": 12,
    "valor_homocisteina": 10,
    "valor_proteinac_reactiva": 2,
    "valor_transferrina": 280,
    "valor_vitamina_bdoce": 450,
    "valor_vitamina_d": 30,
    "peso": 78,
    "estatura": 1.68,
    "medida_cintura": 92,
    "segundamedicion_peso": 78,
    "segundamedicion_estatura": 1.68,
    "distancia_rodilla_talon": 50,
    "circunferencia_de_la_pantorrilla": 36,
    "segundamedicion_cintura": 92,
    "sueno_horas": 7,
    "actividad_total": 380,
}


def assert_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Falta artefacto requerido: {path}")


def assert_percent(name: str, value: float) -> None:
    if not 0 <= value <= 100:
        raise ValueError(f"{name} debe estar entre 0 y 100, recibio {value}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        "diabetes_model.joblib",
        "diabetes_random_forest.joblib",
        "diabetes_metadata.json",
        "paro_cardiaco_model.joblib",
        "paro_cardiaco_best_model.joblib",
        "paro_cardiaco_scaler.joblib",
        "paro_cardiaco_metadata.json",
        "hipertension_model.joblib",
        "hipertension_best_model.joblib",
        "hipertension_scaler.joblib",
        "hipertension_metadata.json",
        "subtipos_kmeans.joblib",
        "subtipos_metadata.json",
        "subtipos_summary.csv",
    ]
    for file_name in required:
        assert_file(MODELS_DIR / file_name)

    for file_name in required:
        if file_name.endswith(".joblib"):
            joblib.load(MODELS_DIR / file_name)

    startup()
    predictions = predict(SAMPLE_PATIENT)
    profiles = cluster(SAMPLE_PATIENT)

    for key in ("diabetes", "hipert", "cardio", "general"):
        assert_percent(f"predict.{key}.pct", float(predictions[key]["pct"]))
    for key in ("diabetes", "hipert", "cardio"):
        assert_percent(
            f"cluster.{key}.profile_reference_pct",
            float(profiles[key]["profile_reference_pct"]),
        )
    assert_percent("cluster.general.profile_reference_pct", float(profiles["general"]["profile_reference_pct"]))

    report = {
        "status": "ok",
        "checked_artifacts": required,
        "sample_prediction": predictions,
        "sample_profiles_general": profiles["general"],
    }
    (REPORTS_DIR / "pipeline_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Pipeline validado correctamente")


if __name__ == "__main__":
    main()
