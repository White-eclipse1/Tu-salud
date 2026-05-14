from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
CLEANED_DIR = ROOT_DIR / "Base-de-datos" / "cleaned"
MODELS_DIR = ROOT_DIR / "models"
RANDOM_STATE = 42


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 7)) -> dict[int, float]:
    scores = {}
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)
        scores[k] = float(silhouette_score(X_scaled, labels))
    return scores


def describe_cluster_profile(
    raw_features: pd.DataFrame,
    labels: np.ndarray,
    top_features: int = 6,
) -> tuple[pd.DataFrame, dict[int, list[dict[str, float | str]]]]:
    global_mean = raw_features.mean()
    global_std = raw_features.std(ddof=0).replace(0, 1)
    cluster_means = raw_features.assign(cluster=labels).groupby("cluster")[raw_features.columns].mean()
    z_means = (cluster_means - global_mean) / global_std

    descriptors = {}
    for cluster_id, row in z_means.iterrows():
        strongest = row.abs().sort_values(ascending=False).head(top_features).index
        descriptors[int(cluster_id)] = [
            {
                "feature": feature,
                "direction": "alto" if row[feature] > 0 else "bajo",
                "z_score": float(row[feature]),
                "mean": float(cluster_means.loc[cluster_id, feature]),
            }
            for feature in strongest
        ]
    return cluster_means, descriptors


def cluster_and_profile(
    raw_features: pd.DataFrame,
    y_truth: pd.Series,
    y_proba: np.ndarray,
) -> dict[str, object]:
    raw_features = raw_features.select_dtypes(include=np.number).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_features)
    scores = find_optimal_k(X_scaled)
    k = max(scores, key=scores.get)

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    profile = raw_features.copy()
    profile["cluster"] = labels

    summary = profile.groupby("cluster").agg(n=("cluster", "size"))
    summary["pct_muestra"] = summary["n"] / len(profile)
    summary["tasa_positivos_referencia"] = pd.Series(y_truth.values, index=profile.index).groupby(labels).mean().values
    summary["proba_media_supervisado_referencia"] = pd.Series(y_proba, index=profile.index).groupby(labels).mean().values
    summary = summary.round(4)

    cluster_means, descriptors = describe_cluster_profile(raw_features, labels)
    PCA(n_components=2).fit_transform(X_scaled)
    return {
        "kmeans": kmeans,
        "scaler": scaler,
        "k": int(k),
        "silhouette": float(scores[k]),
        "summary": summary,
        "descriptors": descriptors,
        "cluster_means": cluster_means,
    }


def diabetes_probabilities(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    bundle = joblib.load(MODELS_DIR / "diabetes_model.joblib")
    raw_features = df.drop(columns=["Outcome"])
    engineered = df.copy()
    engineered["BMI_Category"] = pd.cut(engineered["BMI"], bins=[0, 18.5, 25, 30, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])
    engineered["AgeGroup"] = pd.cut(engineered["Age"], bins=[0, 30, 45, 60, 120], labels=["Young", "Adult", "Middle-aged", "Senior"])
    engineered["Glucose_Category"] = pd.cut(engineered["Glucose"], bins=[0, 99, 125, 300], labels=["Normal", "Prediabetes", "Diabetes"])
    engineered["Insulin_Category"] = pd.cut(engineered["Insulin"], bins=[0, 16, 166, 1000], labels=["Low", "Normal", "High"])
    engineered["BP_Category"] = pd.cut(engineered["BloodPressure"], bins=[0, 80, 89, 200], labels=["Normal", "Elevated", "High"])
    engineered = pd.get_dummies(
        engineered,
        columns=["BMI_Category", "AgeGroup", "Glucose_Category", "Insulin_Category", "BP_Category"],
    )
    expected = bundle["metadata"]["feature_columns"]
    for column in expected:
        if column not in engineered.columns:
            engineered[column] = 0
    y_proba = bundle["model"].predict_proba(engineered[expected])[:, 1]
    return raw_features, df["Outcome"].astype(int), y_proba


def paro_probabilities(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    bundle = joblib.load(MODELS_DIR / "paro_cardiaco_model.joblib")
    raw_features = df.drop(columns=["Result"])
    scaled = bundle["scaler"].transform(raw_features)
    y_proba = bundle["model"].predict_proba(scaled)[:, 1]
    return raw_features, df["Result"].astype(int), y_proba


def hipertension_probabilities(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    bundle = joblib.load(MODELS_DIR / "hipertension_model.joblib")
    target = bundle["metadata"]["target_column"]
    raw_features = df.drop(columns=[target])
    expected = bundle["metadata"]["feature_columns"]
    scaled = bundle["scaler"].transform(raw_features[expected])
    y_proba = bundle["model"].predict_proba(scaled)[:, 1]
    upper = bundle["metadata"].get("probability_upper_bound")
    if upper is not None:
        y_proba = np.minimum(y_proba, float(upper))
    return raw_features, df[target].astype(int), y_proba


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "diabetes": diabetes_probabilities(pd.read_csv(CLEANED_DIR / "diabetes_cleaned.csv")),
        "paro_cardiaco": paro_probabilities(pd.read_csv(CLEANED_DIR / "Medicaldataset_cleaned.csv")),
        "hipertension": hipertension_probabilities(pd.read_csv(CLEANED_DIR / "Hipertension_Arterial_Mexico_cleaned.csv")),
    }
    results = {
        name: cluster_and_profile(raw_features, y_truth, y_proba)
        for name, (raw_features, y_truth, y_proba) in datasets.items()
    }

    artifact = {
        name: {
            "kmeans": result["kmeans"],
            "scaler": result["scaler"],
            "k": result["k"],
            "silhouette": result["silhouette"],
        }
        for name, result in results.items()
    }
    joblib.dump(artifact, MODELS_DIR / "subtipos_kmeans.joblib")

    combined = pd.concat(
        {name: result["summary"] for name, result in results.items()},
        names=["enfermedad", "cluster"],
    )
    combined.to_csv(MODELS_DIR / "subtipos_summary.csv")

    metadata = {
        "approach": "KMeans exploratorio por dataset; k elegido por silhouette en [2,6]. Las etiquetas y probabilidades se usan solo como referencia descriptiva posterior, no para definir ni diagnosticar clusters.",
        "interpretation_warning": "Un cluster indica similitud estadistica entre pacientes, no presencia ni ausencia de una enfermedad.",
        "datasets": {
            name: {"k": result["k"], "silhouette": result["silhouette"]}
            for name, result in results.items()
        },
        "summary_per_cluster": {
            name: result["summary"].reset_index().to_dict(orient="records")
            for name, result in results.items()
        },
        "profile_descriptors": {
            name: result["descriptors"]
            for name, result in results.items()
        },
    }
    (MODELS_DIR / "subtipos_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Perfiles K-Means guardados")


if __name__ == "__main__":
    main()
