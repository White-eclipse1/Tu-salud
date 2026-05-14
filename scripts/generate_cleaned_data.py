from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "Base-de-datos" / "raw"
CLEANED_DIR = ROOT_DIR / "Base-de-datos" / "cleaned"


def save_diabetes() -> None:
    df = pd.read_csv(RAW_DIR / "diabetes.csv")
    zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)
    df = df.fillna(df.median(numeric_only=True))
    df.to_csv(CLEANED_DIR / "diabetes_cleaned.csv", index=False)


def save_paro_cardiaco() -> None:
    df = pd.read_csv(RAW_DIR / "Medicaldataset.csv")
    df["Result"] = (
        df["Result"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"negative": 0, "positive": 1})
        .astype(int)
    )
    df.to_csv(CLEANED_DIR / "Medicaldataset_cleaned.csv", index=False)


def save_hipertension() -> None:
    df = pd.read_csv(RAW_DIR / "Hipertension_Arterial_Mexico.csv")

    id_like = [
        column
        for column in df.columns
        if column.lower() in {"id", "id_paciente", "paciente_id", "folio", "folio_i"}
    ]
    df = df.drop(columns=id_like)

    if "sexo" in df.columns and df["sexo"].dtype == object:
        mapping = {value: idx for idx, value in enumerate(sorted(df["sexo"].dropna().unique()))}
        df["sexo"] = df["sexo"].map(mapping)

    clinically_nonzero = [
        "glucosa",
        "presion_sistolica",
        "presion_diastolica",
        "hemoglobina",
        "colesterol_total",
        "colesterol_hdl",
        "colesterol_ldl",
        "trigliceridos",
        "peso",
        "talla",
        "imc",
        "masa_corporal",
        "frecuencia_cardiaca",
    ]
    for column in clinically_nonzero:
        if column in df.columns and (df[column] == 0).any():
            df[column] = df[column].replace(0, np.nan)

    for column in df.select_dtypes(include=np.number).columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna(float(df[column].median()))

    df.to_csv(CLEANED_DIR / "Hipertension_Arterial_Mexico_cleaned.csv", index=False)


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    save_diabetes()
    save_paro_cardiaco()
    save_hipertension()
    print(f"Datasets limpios guardados en {CLEANED_DIR}")


if __name__ == "__main__":
    main()
