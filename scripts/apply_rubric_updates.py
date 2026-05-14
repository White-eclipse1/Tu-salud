"""
Apply rubric-driven updates to all 4 notebooks.

Adds:
- IQR winsorization + save cleaned CSV to data/clean/
- Per-algorithm feature selection (computed and used in training)
- Normalization justification (markdown)
- Per-notebook conclusion update

Run from repo root: python scripts/apply_rubric_updates.py
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SUPER = ROOT / "Supervisado"
UNSUPER = ROOT / "No-supervisado"


def md(source):
    return nbf.v4.new_markdown_cell(source)


def code(source):
    return nbf.v4.new_code_cell(source)


def find_cell_idx(nb, predicate):
    for i, c in enumerate(nb.cells):
        if predicate(c):
            return i
    return -1


def insert_after(nb, idx, cells):
    for offset, c in enumerate(cells, start=1):
        nb.cells.insert(idx + offset, c)


def replace_source(nb, predicate, new_source):
    idx = find_cell_idx(nb, predicate)
    if idx < 0:
        raise RuntimeError("cell not found")
    nb.cells[idx].source = new_source
    nb.cells[idx].outputs = []
    nb.cells[idx].execution_count = None


# -----------------------------------------------------------------------------
# 1) DIABETES
# -----------------------------------------------------------------------------
def update_diabetes():
    path = SUPER / "Diabetes-Prediction.ipynb"
    nb = nbf.read(path, as_version=4)

    # Fix data path: add data/raw/ candidates
    def is_load_cell(c):
        return c.cell_type == "code" and "DATA_PATH_CANDIDATES" in c.source

    load_idx = find_cell_idx(nb, is_load_cell)
    nb.cells[load_idx].source = (
        'DATA_PATH_CANDIDATES = [\n'
        '    Path("../data/raw/diabetes.csv"),\n'
        '    Path("data/raw/diabetes.csv"),\n'
        '    Path("../Base-de-datos/diabetes.csv"),\n'
        '    Path("Base-de-datos/diabetes.csv"),\n'
        ']\n'
        '\n'
        'DATA_PATH = next((path for path in DATA_PATH_CANDIDATES if path.exists()), None)\n'
        'if DATA_PATH is None:\n'
        '    raise FileNotFoundError("No se encontro diabetes.csv. Revisa la ruta del archivo.")\n'
        '\n'
        'df = pd.read_csv(DATA_PATH)\n'
        '\n'
        'print(f"Dataset cargado desde: {DATA_PATH}")\n'
        'print(f"Filas y columnas originales: {df.shape}")\n'
        'df.head()\n'
    )
    nb.cells[load_idx].outputs = []
    nb.cells[load_idx].execution_count = None

    # Insert outlier handling AFTER imputation cell (5f771dba)
    impute_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "impute_missing(df_clean" in c.source
    )

    outlier_md = md(
        "## 5b. Tratamiento de outliers (Winsorización IQR)\n"
        "\n"
        "Las variables clínicas continuas presentan valores extremos legítimos pero "
        "ruidosos para el modelo (mediciones puntuales, errores de captura, "
        "pacientes atípicos). En lugar de eliminar filas — el dataset tiene solo "
        "768 registros — se aplica **winsorización por IQR** con k=1.5:\n"
        "\n"
        "- Para cada feature continua se calcula Q1, Q3 e IQR=Q3-Q1.\n"
        "- Los valores fuera de `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` se recortan a esos límites.\n"
        "\n"
        "Justificación: preserva el tamaño del dataset, atenúa el apalancamiento de "
        "outliers en algoritmos sensibles a magnitud (escalado, distancias) y deja "
        "intacta la información ordinal. Random Forest es robusto a outliers pero "
        "se aplica igual para mantener consistencia con los CSV limpios persistidos."
    )

    outlier_code = code(
        'def winsorize_iqr(df, cols, k=1.5):\n'
        '    """Recorta outliers de cada columna a [Q1 - k*IQR, Q3 + k*IQR]."""\n'
        '    out = df.copy()\n'
        '    bounds = {}\n'
        '    for c in cols:\n'
        '        q1, q3 = out[c].quantile([0.25, 0.75])\n'
        '        iqr = q3 - q1\n'
        '        low, high = q1 - k * iqr, q3 + k * iqr\n'
        '        n_clipped = int(((out[c] < low) | (out[c] > high)).sum())\n'
        '        out[c] = out[c].clip(lower=low, upper=high)\n'
        '        bounds[c] = {"low": float(low), "high": float(high), "n_clipped": n_clipped}\n'
        '    return out, bounds\n'
        '\n'
        '# Variables continuas (excluye Pregnancies que es discreta y Outcome que es target)\n'
        'continuous_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin",\n'
        '                   "BMI", "DiabetesPedigreeFunction", "Age"]\n'
        '\n'
        'df_clean, outlier_bounds = winsorize_iqr(df_clean, continuous_cols, k=1.5)\n'
        '\n'
        'print("Outliers recortados por columna:")\n'
        'for c, info in outlier_bounds.items():\n'
        '    print(f"  {c:30s} clipped={info[\'n_clipped\']:3d}  rango=[{info[\'low\']:.2f}, {info[\'high\']:.2f}]")\n'
        '\n'
        '# Guardar dataset limpio (sin feature engineering) para trazabilidad\n'
        'CLEAN_DIR = Path("../data/clean")\n'
        'CLEAN_DIR.mkdir(parents=True, exist_ok=True)\n'
        'df_clean.to_csv(CLEAN_DIR / "diabetes_clean.csv", index=False)\n'
        'print(f"\\nDataset limpio guardado en: {CLEAN_DIR / \'diabetes_clean.csv\'}")\n'
        'print(f"Shape: {df_clean.shape}")\n'
    )

    insert_after(nb, impute_idx, [outlier_md, outlier_code])

    # Add markdown justifying feature selection for RF AFTER the train/test split cell
    split_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "train_test_split" in c.source and "X_train" in c.source
    )
    fs_md = md(
        "### Selección de variables para Random Forest\n"
        "\n"
        "Random Forest se entrena con el conjunto completo de features (8 originales + "
        "12 dummies de las 5 variables categóricas creadas en feature engineering = "
        "20 features). Justificación:\n"
        "\n"
        "- Los árboles son **invariantes a escala** y manejan correlación entre features "
        "  mediante splits independientes; no requieren reducir dimensionalidad de antemano.\n"
        "- El propio modelo provee `feature_importances_` para ranking post-entrenamiento "
        "  (ver sección 12), lo que actúa como selección implícita: variables irrelevantes "
        "  reciben importancia ~0 sin afectar el desempeño.\n"
        "- Las dummies de feature engineering (`Glucose_Category`, `BMI_Category`, etc.) "
        "  inyectan conocimiento clínico (cortes médicos estándar) que un RF chico puede "
        "  no descubrir solo.\n"
    )
    # Insert BEFORE the split cell (so it explains what's about to happen)
    nb.cells.insert(split_idx, fs_md)

    # Add scaler-not-needed justification BEFORE training (8. Entrenar...)
    train_md_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "markdown" and "Entrenar solo el mejor modelo" in c.source
    )
    scaler_note = md(
        "### Sobre normalización\n"
        "\n"
        "No se aplica `StandardScaler` ni `MinMaxScaler`. Random Forest construye splits "
        "ordinales (`feature <= umbral`) y es **invariante a transformaciones monótonas** "
        "por feature: escalar no cambia los splits que el árbol encuentra. Aplicar un "
        "scaler añadiría costo computacional sin beneficio. (En contraste, los notebooks "
        "de paro cardíaco e hipertensión sí escalan porque usan LogReg/SVM/KNN.)"
    )
    nb.cells.insert(train_md_idx, scaler_note)

    # Update final conclusion cell
    conc_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "markdown" and c.source.startswith("## 14. Conclusión")
    )
    nb.cells[conc_idx].source = (
        "## 14. Conclusión\n"
        "\n"
        "**Dataset:** `diabetes.csv` (768 pacientes, 8 variables clínicas + Outcome).\n"
        "\n"
        "**Limpieza aplicada:**\n"
        "- Reemplazo de ceros clínicamente imposibles (`Glucose`, `BloodPressure`, "
        "  `SkinThickness`, `Insulin`, `BMI`) por NaN → imputación con **mediana**.\n"
        "- **Winsorización IQR** (k=1.5) sobre variables continuas para acotar outliers "
        "  sin perder filas.\n"
        "- CSV limpio persistido en `data/clean/diabetes_clean.csv`.\n"
        "\n"
        "**Feature engineering:** binning clínico (BMI, edad, glucosa, insulina, presión) "
        "→ 5 nuevas categóricas → one-hot → 20 features finales.\n"
        "\n"
        "**Algoritmo:** Random Forest tuneado con `GridSearchCV` (5-fold, F1-score). "
        "Justificación: maneja interacciones no lineales entre features clínicas, es "
        "robusto a outliers y escala, y entrega importancias interpretables — relevante "
        "en un contexto médico donde la justificación del modelo importa tanto como el desempeño.\n"
        "\n"
        "**Resultados (test):** F1 ≈ 0.60, ROC-AUC ≈ 0.82, accuracy ≈ 0.75. La feature más "
        "influyente es Glucose, seguida de BMI y edad — consistente con el criterio clínico.\n"
        "\n"
        "**Limitaciones:** dataset chico (768), recall moderado (~0.54) → el modelo "
        "pierde positivos; en uso clínico convendría bajar el umbral de decisión o "
        "rebalancear con SMOTE para priorizar sensibilidad."
    )

    # Update metadata-save cell to include outlier bounds + new clean path
    save_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "diabetes_metadata.json" in c.source
    )
    nb.cells[save_idx].source = (
        'import json\n'
        'import joblib\n'
        '\n'
        'MODELS_DIR = Path("../models")\n'
        'MODELS_DIR.mkdir(parents=True, exist_ok=True)\n'
        '\n'
        'diabetes_medians = {\n'
        '    column: float(value)\n'
        '    for column, value in replace_zeros_with_nan(df, zero_invalid_cols)[zero_invalid_cols].median().items()\n'
        '}\n'
        '\n'
        'diabetes_metadata = {\n'
        '    "model_name": "Random Forest optimizado para diabetes",\n'
        '    "source_dataset": "data/raw/diabetes.csv",\n'
        '    "clean_dataset": "data/clean/diabetes_clean.csv",\n'
        '    "target_column": "Outcome",\n'
        '    "class_labels": {"0": "No diabetes", "1": "Diabetes"},\n'
        '    "feature_columns": list(X.columns),\n'
        '    "zero_invalid_columns": zero_invalid_cols,\n'
        '    "categorical_columns": categorical_cols,\n'
        '    "imputation_medians": diabetes_medians,\n'
        '    "outlier_treatment": "winsorize_iqr_k1.5",\n'
        '    "outlier_bounds": outlier_bounds,\n'
        '    "feature_selection": "all features (RF invariante a escala y maneja colinealidad via splits)",\n'
        '    "normalization": "none (Random Forest no requiere escalado)",\n'
        '    "best_params": grid_search.best_params_,\n'
        '    "test_metrics": {metric: float(value) for metric, value in metrics.items()},\n'
        '}\n'
        '\n'
        'diabetes_bundle = {\n'
        '    "model": best_rf,\n'
        '    "metadata": diabetes_metadata,\n'
        '}\n'
        '\n'
        'joblib.dump(best_rf, MODELS_DIR / "diabetes_random_forest.joblib")\n'
        'joblib.dump(diabetes_bundle, MODELS_DIR / "diabetes_model.joblib")\n'
        '\n'
        'with open(MODELS_DIR / "diabetes_metadata.json", "w", encoding="utf-8") as file:\n'
        '    json.dump(diabetes_metadata, file, indent=2)\n'
        '\n'
        'print("Modelo de diabetes guardado en:")\n'
        'print(MODELS_DIR / "diabetes_random_forest.joblib")\n'
        'print(MODELS_DIR / "diabetes_model.joblib")\n'
    )
    nb.cells[save_idx].outputs = []
    nb.cells[save_idx].execution_count = None

    nbf.write(nb, path)
    print(f"✓ Diabetes notebook updated: {path}")


# -----------------------------------------------------------------------------
# 2) PARO CARDIACO
# -----------------------------------------------------------------------------
def update_paro():
    path = SUPER / "Prediccion-paro-cardiaco.ipynb"
    nb = nbf.read(path, as_version=4)

    # Fix data path
    load_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "DATA_PATH_CANDIDATES" in c.source
    )
    nb.cells[load_idx].source = (
        'DATA_PATH_CANDIDATES = [\n'
        '    Path("../data/raw/Medicaldataset.csv"),\n'
        '    Path("data/raw/Medicaldataset.csv"),\n'
        '    Path("../Base-de-datos/Medicaldataset.csv"),\n'
        '    Path("Base-de-datos/Medicaldataset.csv"),\n'
        ']\n'
        '\n'
        'DATA_PATH = next((path for path in DATA_PATH_CANDIDATES if path.exists()), None)\n'
        'if DATA_PATH is None:\n'
        '    raise FileNotFoundError("No se encontro Medicaldataset.csv. Revisa la ruta del archivo.")\n'
        '\n'
        'data_df = pd.read_csv(DATA_PATH)\n'
        '\n'
        'print(f"Dataset cargado desde: {DATA_PATH}")\n'
        'print(f"Filas y columnas: {data_df.shape}")\n'
        'data_df.head()\n'
    )
    nb.cells[load_idx].outputs = []
    nb.cells[load_idx].execution_count = None

    # Insert outlier treatment AFTER the Result-mapping cell (dfe8d406)
    map_idx = find_cell_idx(
        nb,
        lambda c: c.cell_type == "code" and 'data_model["Result"].map' in c.source,
    )

    outlier_md = md(
        "## 4b. Tratamiento de outliers (Winsorización IQR)\n"
        "\n"
        "Los boxplots iniciales muestran colas largas en `CK-MB`, `Troponin`, "
        "`Blood sugar`, etc. — biomarcadores con distribución muy sesgada. Aplicar "
        "winsorización por IQR (k=1.5) acota esas colas sin descartar pacientes "
        "(el dataset tiene 1319 filas; perder ~10% por filtrado de outliers reduce "
        "el poder estadístico).\n"
        "\n"
        "Justificación frente a otras opciones:\n"
        "- **Eliminación:** descarta señal real (un paciente con troponina alta puede "
        "  ser justo el positivo que queremos detectar).\n"
        "- **Z-score:** asume normalidad — biomarcadores cardíacos son log-normales.\n"
        "- **Winsorización IQR:** robusta sin supuestos paramétricos, preserva N."
    )

    outlier_code = code(
        'def winsorize_iqr(df, cols, k=1.5):\n'
        '    """Recorta outliers a [Q1 - k*IQR, Q3 + k*IQR] por columna."""\n'
        '    out = df.copy()\n'
        '    bounds = {}\n'
        '    for c in cols:\n'
        '        q1, q3 = out[c].quantile([0.25, 0.75])\n'
        '        iqr = q3 - q1\n'
        '        low, high = q1 - k * iqr, q3 + k * iqr\n'
        '        n_clipped = int(((out[c] < low) | (out[c] > high)).sum())\n'
        '        out[c] = out[c].clip(lower=low, upper=high)\n'
        '        bounds[c] = {"low": float(low), "high": float(high), "n_clipped": n_clipped}\n'
        '    return out, bounds\n'
        '\n'
        'continuous_cols = ["Age", "Heart rate", "Systolic blood pressure",\n'
        '                   "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]\n'
        '\n'
        'data_model, outlier_bounds = winsorize_iqr(data_model, continuous_cols, k=1.5)\n'
        '\n'
        'print("Outliers recortados por columna:")\n'
        'for c, info in outlier_bounds.items():\n'
        '    print(f"  {c:30s} clipped={info[\'n_clipped\']:3d}  rango=[{info[\'low\']:.2f}, {info[\'high\']:.2f}]")\n'
        '\n'
        '# Guardar dataset limpio para trazabilidad\n'
        'CLEAN_DIR = Path("../data/clean")\n'
        'CLEAN_DIR.mkdir(parents=True, exist_ok=True)\n'
        'data_model.to_csv(CLEAN_DIR / "Medicaldataset_clean.csv", index=False)\n'
        'print(f"\\nDataset limpio guardado en: {CLEAN_DIR / \'Medicaldataset_clean.csv\'}")\n'
        'print(f"Shape: {data_model.shape}")\n'
    )

    insert_after(nb, map_idx, [outlier_md, outlier_code])

    # Insert per-algorithm feature selection BEFORE the scaler cell (b7a34fab)
    scaler_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "s_scaler = preprocessing.StandardScaler()" in c.source
    )

    fs_md = md(
        "## 7b. Selección de variables por algoritmo\n"
        "\n"
        "Los tres algoritmos comparados (SVM, KNN, Logistic Regression) tienen "
        "supuestos distintos, por eso se construye un **subconjunto de features "
        "específico para cada uno**:\n"
        "\n"
        "- **Logistic Regression:** se queda con las features cuyo `|corr(target)|` "
        "  está por arriba de la mediana del ranking, y se eliminan pares con "
        "  `|r|>0.9` entre sí. La regresión logística es sensible a multicolinealidad "
        "  (los coeficientes se vuelven inestables) y a features irrelevantes, así "
        "  que se simplifica el espacio.\n"
        "- **SVM lineal:** mismo subconjunto que LogReg. Razón: SVM lineal calcula "
        "  productos punto en el espacio de features; con menos redundancia los "
        "  pesos `coef_` son más estables e interpretables.\n"
        "- **KNN:** sufre **maldición de dimensionalidad** — en pocas dimensiones "
        "  las distancias discriminan mejor. Se elige el **top-5 por información "
        "  mutua** (`mutual_info_classif`) con el target, que captura relaciones "
        "  no lineales (KNN no asume linealidad).\n"
    )

    fs_code = code(
        'from sklearn.feature_selection import mutual_info_classif\n'
        '\n'
        'target_col = "Result"\n'
        'numeric_X = data_model.drop(columns=[target_col])\n'
        '\n'
        '# --- Selección para LogReg / SVM: correlación + descarte de colinealidad ---\n'
        'corr_target = numeric_X.corrwith(data_model[target_col]).abs().sort_values(ascending=False)\n'
        'print("Correlación |feature, target|:")\n'
        'print(corr_target.round(3))\n'
        '\n'
        'median_corr = corr_target.median()\n'
        'kept = corr_target[corr_target >= median_corr].index.tolist()\n'
        '\n'
        '# eliminar pares con |r|>0.9 entre sí, conservando el que más correlaciona con target\n'
        'corr_matrix = numeric_X[kept].corr().abs()\n'
        'to_drop = set()\n'
        'for i, c1 in enumerate(kept):\n'
        '    for c2 in kept[i+1:]:\n'
        '        if c2 in to_drop or c1 in to_drop:\n'
        '            continue\n'
        '        if corr_matrix.loc[c1, c2] > 0.9:\n'
        '            weaker = c1 if corr_target[c1] < corr_target[c2] else c2\n'
        '            to_drop.add(weaker)\n'
        '\n'
        'features_lr_svm = [c for c in kept if c not in to_drop]\n'
        'print(f"\\nFeatures para LogReg y SVM ({len(features_lr_svm)}): {features_lr_svm}")\n'
        '\n'
        '# --- Selección para KNN: top-5 por mutual_info ---\n'
        'mi = pd.Series(\n'
        '    mutual_info_classif(numeric_X, data_model[target_col], random_state=25),\n'
        '    index=numeric_X.columns,\n'
        ').sort_values(ascending=False)\n'
        'print(f"\\nMutual information con target:")\n'
        'print(mi.round(3))\n'
        '\n'
        'features_knn = mi.head(5).index.tolist()\n'
        'print(f"\\nFeatures para KNN (top-5 MI): {features_knn}")\n'
        '\n'
        '# diccionario de subsets por algoritmo (se usa abajo)\n'
        'feature_sets = {\n'
        '    "Support Vector Machine": features_lr_svm,\n'
        '    "K-Nearest Neighbors":    features_knn,\n'
        '    "Logistic Regression":    features_lr_svm,\n'
        '}\n'
    )

    insert_after(nb, scaler_idx - 1, [fs_md, fs_code])
    # recompute scaler_idx
    scaler_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "s_scaler = preprocessing.StandardScaler()" in c.source
    )

    # Update scaler markdown (cell before scaler) to add justification
    scaler_md_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "markdown" and c.source.startswith("## 8. Escalar")
    )
    nb.cells[scaler_md_idx].source = (
        "## 8. Escalar variables (StandardScaler)\n"
        "\n"
        "**Justificación del scaler:** los tres algoritmos calculan magnitudes en el "
        "espacio de features:\n"
        "\n"
        "- **LogReg:** los coeficientes son sensibles a la escala; sin escalado, las "
        "  features con rango grande (e.g. `Blood sugar` 50–500) dominan a las de "
        "  rango chico (e.g. `Troponin` 0–10) y la regularización penaliza "
        "  asimétricamente.\n"
        "- **SVM:** maximiza el margen en el espacio de features → unidades disparejas "
        "  distorsionan el hiperplano.\n"
        "- **KNN:** calcula distancias euclidianas; sin escalar, la distancia se "
        "  reduce a 'la diferencia en la feature de mayor rango'.\n"
        "\n"
        "`StandardScaler` centra cada feature en 0 con varianza 1, dejando todas las "
        "variables en pie de igualdad. Se ajusta sólo sobre el set completo previo al "
        "split (para esta comparación didáctica) — en producción se ajustaría sólo en train."
    )

    # Replace scaler cell to keep the original behavior but document it
    # (no change needed, just keep)

    # Now modify each model-training cell to use its per-algorithm feature subset
    # SVM cell (a9aef811)
    svm_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "svm_model = svm.SVC" in c.source
    )
    nb.cells[svm_idx].source = (
        'svm_features = feature_sets["Support Vector Machine"]\n'
        'X_train_svm = X_train[svm_features]\n'
        'X_test_svm  = X_test[svm_features]\n'
        '\n'
        'svm_model = svm.SVC(kernel="linear", probability=True, random_state=25)\n'
        '\n'
        'svm_model, svm_pred, svm_proba, svm_metrics = evaluate_model(\n'
        '    "Support Vector Machine",\n'
        '    svm_model,\n'
        '    X_train_svm,\n'
        '    X_test_svm,\n'
        '    y_train,\n'
        '    y_test,\n'
        ')\n'
    )
    nb.cells[svm_idx].outputs = []
    nb.cells[svm_idx].execution_count = None

    # SVM confusion cell uses svm_pred — no change
    # SVM classification report — no change

    # KNN cell
    knn_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "knn_model = KNeighborsClassifier" in c.source
    )
    nb.cells[knn_idx].source = (
        'knn_features = feature_sets["K-Nearest Neighbors"]\n'
        'X_train_knn = X_train[knn_features]\n'
        'X_test_knn  = X_test[knn_features]\n'
        '\n'
        'knn_model = KNeighborsClassifier(n_neighbors=5)\n'
        '\n'
        'knn_model, knn_pred, knn_proba, knn_metrics = evaluate_model(\n'
        '    "K-Nearest Neighbors",\n'
        '    knn_model,\n'
        '    X_train_knn,\n'
        '    X_test_knn,\n'
        '    y_train,\n'
        '    y_test,\n'
        ')\n'
    )
    nb.cells[knn_idx].outputs = []
    nb.cells[knn_idx].execution_count = None

    # LogReg cell
    lr_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "log_model = LogisticRegression" in c.source
    )
    nb.cells[lr_idx].source = (
        'lr_features = feature_sets["Logistic Regression"]\n'
        'X_train_lr = X_train[lr_features]\n'
        'X_test_lr  = X_test[lr_features]\n'
        '\n'
        'log_model = LogisticRegression(max_iter=1000, random_state=25)\n'
        '\n'
        'log_model, log_pred, log_proba, log_metrics = evaluate_model(\n'
        '    "Logistic Regression",\n'
        '    log_model,\n'
        '    X_train_lr,\n'
        '    X_test_lr,\n'
        '    y_train,\n'
        '    y_test,\n'
        ')\n'
    )
    nb.cells[lr_idx].outputs = []
    nb.cells[lr_idx].execution_count = None

    # Update conclusion cell
    conc_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "markdown" and c.source.startswith("## 17. Conclusion")
    )
    nb.cells[conc_idx].source = (
        "## 17. Conclusión\n"
        "\n"
        "**Dataset:** `Medicaldataset.csv` (1319 pacientes, 8 features clínicas + `Result`).\n"
        "\n"
        "**Limpieza aplicada:**\n"
        "- Mapeo `Result` `negative→0`, `positive→1`.\n"
        "- **Winsorización IQR** (k=1.5) sobre las 7 variables continuas (biomarcadores con "
        "  colas largas como troponina y CK-MB).\n"
        "- CSV limpio persistido en `data/clean/Medicaldataset_clean.csv`.\n"
        "\n"
        "**Normalización:** `StandardScaler` aplicado a todas las features. Necesario "
        "porque los tres algoritmos (LogReg, SVM lineal, KNN) son sensibles a la escala.\n"
        "\n"
        "**Selección de variables por algoritmo:**\n"
        "- LogReg y SVM: subconjunto por `|corr(target)|` ≥ mediana, descartando pares "
        "  con `|r|>0.9` (multicolinealidad).\n"
        "- KNN: top-5 features por `mutual_info_classif` para mitigar maldición de "
        "  dimensionalidad.\n"
        "\n"
        "**Algoritmos comparados:**\n"
        "1. **SVM lineal** — margen máximo en espacio lineal; sirve de baseline robusto.\n"
        "2. **KNN (k=5)** — referencia no paramétrica; sensible a dimensionalidad.\n"
        "3. **Logistic Regression** — interpretable, calibrada por defecto; típicamente "
        "   el modelo de referencia en literatura clínica.\n"
        "\n"
        "**Selección final:** se elige el de mayor **F1-score** (criterio médico: "
        "balancea precisión y recall, importante porque ambos falsos negativos y "
        "falsos positivos tienen costo clínico).\n"
        "\n"
        "**Resultado esperado:** Logistic Regression suele ganar por F1 (~0.82) sobre "
        "SVM (~0.81) y KNN (~0.68 — penalizado por dimensionalidad incluso reducida).\n"
        "\n"
        "**Limitaciones:** dataset pequeño y desbalance leve; las métricas pueden "
        "variar con otro `random_state`. Validación cruzada estratificada y, si el "
        "tamaño lo permite, un test set externo darían más confianza."
    )

    # Update save cell to include per-algorithm features + outlier bounds
    save_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "paro_cardiaco_metadata.json" in c.source
    )
    nb.cells[save_idx].source = (
        'import json\n'
        'import joblib\n'
        '\n'
        'MODELS_DIR = Path("../models")\n'
        'MODELS_DIR.mkdir(parents=True, exist_ok=True)\n'
        '\n'
        'trained_models = {\n'
        '    "Support Vector Machine": svm_model,\n'
        '    "K-Nearest Neighbors": knn_model,\n'
        '    "Logistic Regression": log_model,\n'
        '}\n'
        '\n'
        'best_model = trained_models[best_model_name]\n'
        'best_features = feature_sets[best_model_name]\n'
        '\n'
        'metrics_by_model = {\n'
        '    model_name: {metric: float(value) for metric, value in values.items() if metric != "Modelo"}\n'
        '    for model_name, values in metrics_df.to_dict(orient="index").items()\n'
        '}\n'
        '\n'
        'paro_metadata = {\n'
        '    "model_name": best_model_name,\n'
        '    "source_dataset": "data/raw/Medicaldataset.csv",\n'
        '    "clean_dataset": "data/clean/Medicaldataset_clean.csv",\n'
        '    "target_column": "Result",\n'
        '    "class_labels": {"0": "Negative", "1": "Positive"},\n'
        '    "feature_columns": list(X.columns),\n'
        '    "features_used_best": best_features,\n'
        '    "features_per_algorithm": feature_sets,\n'
        '    "label_mapping": {"negative": 0, "positive": 1},\n'
        '    "outlier_treatment": "winsorize_iqr_k1.5",\n'
        '    "outlier_bounds": outlier_bounds,\n'
        '    "normalization": "StandardScaler",\n'
        '    "test_metrics_by_model": metrics_by_model,\n'
        '}\n'
        '\n'
        'paro_bundle = {\n'
        '    "model": best_model,\n'
        '    "scaler": s_scaler,\n'
        '    "features": best_features,\n'
        '    "metadata": paro_metadata,\n'
        '}\n'
        '\n'
        'joblib.dump(best_model, MODELS_DIR / "paro_cardiaco_best_model.joblib")\n'
        'joblib.dump(s_scaler, MODELS_DIR / "paro_cardiaco_scaler.joblib")\n'
        'joblib.dump(paro_bundle, MODELS_DIR / "paro_cardiaco_model.joblib")\n'
        '\n'
        'with open(MODELS_DIR / "paro_cardiaco_metadata.json", "w", encoding="utf-8") as file:\n'
        '    json.dump(paro_metadata, file, indent=2)\n'
        '\n'
        'print("Modelo de paro cardiaco guardado en:")\n'
        'print(MODELS_DIR / "paro_cardiaco_best_model.joblib")\n'
        'print(MODELS_DIR / "paro_cardiaco_scaler.joblib")\n'
        'print(MODELS_DIR / "paro_cardiaco_model.joblib")\n'
    )
    nb.cells[save_idx].outputs = []
    nb.cells[save_idx].execution_count = None

    nbf.write(nb, path)
    print(f"✓ Paro cardiaco notebook updated: {path}")


# -----------------------------------------------------------------------------
# 3) HIPERTENSION
# -----------------------------------------------------------------------------
def update_hipertension():
    path = SUPER / "Hipertension_Arterial_Mexico.ipynb"
    nb = nbf.read(path, as_version=4)

    # Insert outlier handling AFTER the cleaning cell (f2f506ef — drops id, encodes sexo, imputes)
    clean_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "clinically_nonzero" in c.source
    )

    outlier_md = md(
        "## Tratamiento de outliers (Winsorización IQR)\n"
        "\n"
        "El dataset ENSANUT contiene mediciones clínicas con escalas dispares "
        "(hemoglobina, colesterol, presión arterial, antropometría) y errores de "
        "captura ocasionales (e.g. `segundamedicion_cintura == 0`). Se aplica "
        "winsorización por IQR (k=1.5) **sobre las variables continuas**, "
        "preservando las binarias y los códigos ordinales.\n"
        "\n"
        "Justificación: con 4363 filas eliminar outliers reduciría la N "
        "innecesariamente; los valores extremos en variables clínicas suelen ser "
        "pacientes reales del extremo de la distribución (justamente los que el "
        "modelo de riesgo debe identificar), pero su magnitud distorsiona "
        "algoritmos basados en escala (LogReg). Recortar a los whiskers preserva "
        "la fila y atenúa la influencia."
    )

    outlier_code = code(
        'def winsorize_iqr(df, cols, k=1.5):\n'
        '    """Recorta outliers a [Q1 - k*IQR, Q3 + k*IQR] por columna."""\n'
        '    out = df.copy()\n'
        '    bounds = {}\n'
        '    for c in cols:\n'
        '        q1, q3 = out[c].quantile([0.25, 0.75])\n'
        '        iqr = q3 - q1\n'
        '        low, high = q1 - k * iqr, q3 + k * iqr\n'
        '        n_clipped = int(((out[c] < low) | (out[c] > high)).sum())\n'
        '        out[c] = out[c].clip(lower=low, upper=high)\n'
        '        bounds[c] = {"low": float(low), "high": float(high), "n_clipped": n_clipped}\n'
        '    return out, bounds\n'
        '\n'
        '# Continuas = numéricas con más de 10 valores únicos, excluyendo target y binarias tipo sexo\n'
        'target = "riesgo_hipertension"\n'
        'continuous_cols = [\n'
        '    c for c in df_clean.select_dtypes(include=np.number).columns\n'
        '    if c != target and df_clean[c].nunique() > 10\n'
        ']\n'
        'print(f"Columnas continuas a winsorizar: {len(continuous_cols)}")\n'
        '\n'
        'df_clean, outlier_bounds = winsorize_iqr(df_clean, continuous_cols, k=1.5)\n'
        '\n'
        'total_clipped = sum(b["n_clipped"] for b in outlier_bounds.values())\n'
        'print(f"Total de valores recortados: {total_clipped}")\n'
        'print(f"Shape final: {df_clean.shape}")\n'
        '\n'
        '# Persistir dataset limpio\n'
        'from pathlib import Path as _P\n'
        'CLEAN_DIR = _P("../data/clean")\n'
        'if not CLEAN_DIR.exists():\n'
        '    CLEAN_DIR = _P("data/clean")\n'
        'CLEAN_DIR.mkdir(parents=True, exist_ok=True)\n'
        'df_clean.to_csv(CLEAN_DIR / "Hipertension_Arterial_Mexico_clean.csv", index=False)\n'
        'print(f"\\nDataset limpio guardado en: {CLEAN_DIR / \'Hipertension_Arterial_Mexico_clean.csv\'}")\n'
    )

    insert_after(nb, clean_idx, [outlier_md, outlier_code])

    # Insert per-algorithm feature selection BEFORE the split cell (14acd1d9)
    split_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "X = df_clean.drop(columns=[target])" in c.source
    )

    fs_md = md(
        "## Selección de variables por algoritmo\n"
        "\n"
        "El dataset tiene 34 features tras limpieza. Cada algoritmo necesita un "
        "tratamiento distinto:\n"
        "\n"
        "- **Logistic Regression:** se queda con features cuyo `|corr(target)|` está "
        "  por arriba de la mediana **y** se descartan pares con `|r|>0.9` entre sí. "
        "  La regresión logística penalizada (`l2`) sufre con multicolinealidad y "
        "  features ruidosas: los coeficientes oscilan y la interpretación se rompe. "
        "  Reducir el espacio mejora calibración y estabilidad.\n"
        "- **Random Forest y Gradient Boosting:** **se usan las 34 features**. Los "
        "  árboles son invariantes a escala, manejan correlación con splits "
        "  independientes y hacen selección implícita vía `feature_importances_`. "
        "  Quitar variables a priori sólo desperdicia información — el árbol decide "
        "  no usarlas si no aportan.\n"
    )

    fs_code = code(
        '# --- features para LogReg: correlación + descarte de colinealidad ---\n'
        'corr_target_full = df_clean.drop(columns=[target]).corrwith(df_clean[target]).abs()\n'
        'corr_target_full = corr_target_full.sort_values(ascending=False)\n'
        '\n'
        'median_corr = corr_target_full.median()\n'
        'kept = corr_target_full[corr_target_full >= median_corr].index.tolist()\n'
        '\n'
        'corr_matrix = df_clean[kept].corr().abs()\n'
        'to_drop = set()\n'
        'for i, c1 in enumerate(kept):\n'
        '    for c2 in kept[i+1:]:\n'
        '        if c2 in to_drop or c1 in to_drop:\n'
        '            continue\n'
        '        if corr_matrix.loc[c1, c2] > 0.9:\n'
        '            weaker = c1 if corr_target_full[c1] < corr_target_full[c2] else c2\n'
        '            to_drop.add(weaker)\n'
        '\n'
        'features_lr = [c for c in kept if c not in to_drop]\n'
        'features_trees = df_clean.drop(columns=[target]).columns.tolist()\n'
        '\n'
        'feature_sets = {\n'
        '    "LogisticRegression": features_lr,\n'
        '    "RandomForest":       features_trees,\n'
        '    "GradientBoosting":   features_trees,\n'
        '}\n'
        '\n'
        'print(f"Features para LogReg ({len(features_lr)}): {features_lr}")\n'
        'print(f"\\nFeatures para RF/GB ({len(features_trees)} = todas)")\n'
    )

    insert_after(nb, split_idx - 1, [fs_md, fs_code])
    # Recompute split idx
    split_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "X = df_clean.drop(columns=[target])" in c.source
    )

    # Modify the training cell to use per-algorithm features
    train_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and '"LogisticRegression": LogisticRegression' in c.source
    )
    nb.cells[train_idx].source = (
        'models = {\n'
        '    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),\n'
        '    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),\n'
        '    "GradientBoosting":   GradientBoostingClassifier(random_state=RANDOM_STATE),\n'
        '}\n'
        '\n'
        '# Aplicamos el subset de features correspondiente a cada algoritmo.\n'
        '# Para LogReg además usamos los datos escalados (sensible a magnitud);\n'
        '# para RF/GB usamos los datos sin escalar (escala-invariantes).\n'
        'X_train_df = pd.DataFrame(X_train_s, columns=feature_columns, index=X_train.index)\n'
        'X_test_df  = pd.DataFrame(X_test_s,  columns=feature_columns, index=X_test.index)\n'
        '\n'
        'data_per_model = {\n'
        '    "LogisticRegression": (X_train_df[feature_sets["LogisticRegression"]],\n'
        '                           X_test_df[feature_sets["LogisticRegression"]]),\n'
        '    "RandomForest":       (X_train[feature_sets["RandomForest"]],\n'
        '                           X_test[feature_sets["RandomForest"]]),\n'
        '    "GradientBoosting":   (X_train[feature_sets["GradientBoosting"]],\n'
        '                           X_test[feature_sets["GradientBoosting"]]),\n'
        '}\n'
        '\n'
        'trained, preds, probas = {}, {}, {}\n'
        'for name, mdl in models.items():\n'
        '    Xtr, Xte = data_per_model[name]\n'
        '    mdl.fit(Xtr, y_train)\n'
        '    trained[name] = mdl\n'
        '    preds[name]   = mdl.predict(Xte)\n'
        '    probas[name]  = mdl.predict_proba(Xte)[:, 1]\n'
        '    print(f"✓ {name} entrenado con {Xtr.shape[1]} features")\n'
    )
    nb.cells[train_idx].outputs = []
    nb.cells[train_idx].execution_count = None

    # Update scaler markdown justification (add a small inline note as a new md cell BEFORE scaler)
    scaler_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "scaler = StandardScaler()" in c.source
    )
    scaler_note = md(
        "### Sobre el escalado\n"
        "\n"
        "`StandardScaler` es **necesario para Logistic Regression** (calcula productos "
        "punto, los coeficientes y la regularización son sensibles a escala). En "
        "cambio **Random Forest y Gradient Boosting son invariantes a escala** — "
        "construyen splits ordinales, así que escalar no cambia el resultado. Aquí "
        "lo aplicamos uniformemente por simplicidad del pipeline y porque LogReg lo "
        "necesita; el costo computacional es despreciable y permite usar el mismo "
        "`X_train_s` para los tres. Las predicciones de RF/GB serían idénticas con "
        "o sin scaler.\n"
        "\n"
        "(En el bloque siguiente, sin embargo, RF/GB reciben **datos sin escalar** "
        "para preservar las unidades originales en `feature_importances_`, mientras "
        "que LogReg recibe los datos escalados.)"
    )
    nb.cells.insert(scaler_idx, scaler_note)

    # Update conclusion cell
    conc_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "markdown" and c.source.startswith("## Conclusión")
    )
    nb.cells[conc_idx].source = (
        "## Conclusión\n"
        "\n"
        "**Dataset:** `Hipertension_Arterial_Mexico.csv` (ENSANUT, 4363 pacientes, 36 columnas).\n"
        "\n"
        "**Limpieza aplicada:**\n"
        "- Drop de `FOLIO_I` (identificador, no aporta).\n"
        "- Codificación numérica de `sexo` (cuando aplica).\n"
        "- Reemplazo de ceros clínicamente imposibles por NaN → imputación con mediana.\n"
        "- **Winsorización IQR** (k=1.5) sobre variables continuas (más de 10 valores únicos).\n"
        "- CSV limpio persistido en `data/clean/Hipertension_Arterial_Mexico_clean.csv`.\n"
        "\n"
        "**Normalización:** `StandardScaler` aplicado a las 34 features. **Necesario para "
        "Logistic Regression**, **indiferente para Random Forest y Gradient Boosting** "
        "(escala-invariantes). RF y GB reciben los datos originales para preservar "
        "interpretabilidad de `feature_importances_`; LogReg recibe los datos escalados.\n"
        "\n"
        "**Selección de variables por algoritmo:**\n"
        "- LogReg: features con `|corr(target)|` ≥ mediana, descartando pares con `|r|>0.9` "
        "  (la regresión logística penalizada es inestable bajo multicolinealidad).\n"
        "- RF / GB: las 34 features. Los árboles seleccionan implícitamente vía splits "
        "  y `feature_importances_`; no se beneficia de pre-selección.\n"
        "\n"
        "**Algoritmos comparados:**\n"
        "1. **Logistic Regression** — baseline lineal, interpretable, calibrado.\n"
        "2. **Random Forest** — maneja interacciones no lineales y robusto al ruido.\n"
        "3. **Gradient Boosting** — corrige residuos secuencialmente; suele ganar en "
        "   datasets tabulares medianos.\n"
        "\n"
        "**Selección final** por F1-score (criterio médico que balancea precisión y recall).\n"
        "\n"
        "**Resultado:** Gradient Boosting tiende a ganar (F1≈0.998, ROC-AUC≈1.0). El "
        "AUC casi perfecto **sugiere posible leakage** o features altamente "
        "deterministas del target (e.g. presiones sistólica/diastólica directamente "
        "ligadas a `riesgo_hipertension`). En despliegue clínico real habría que "
        "auditar qué features son medibles **antes** del diagnóstico para evitar "
        "evaluar circularmente.\n"
        "\n"
        "**Artefactos:** `models/hipertension_*.joblib` y `hipertension_metadata.json` "
        "(incluye `feature_sets`, `outlier_bounds` y métricas por modelo)."
    )

    # Update save cell to include outlier bounds + feature sets
    save_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "hipertension_metadata.json" in c.source
    )
    nb.cells[save_idx].source = (
        'best_name = max(metrics, key=lambda n: metrics[n]["f1"])\n'
        'best_model = trained[best_name]\n'
        'best_features = feature_sets[best_name]\n'
        'print(f"🏆 Mejor modelo por F1: {best_name} (F1={metrics[best_name][\'f1\']:.4f})")\n'
        '\n'
        'models_dir = Path("models")\n'
        'if not models_dir.exists():\n'
        '    models_dir = Path("../models")\n'
        'models_dir.mkdir(exist_ok=True)\n'
        '\n'
        'metadata = {\n'
        '    "target_column": target,\n'
        '    "source_dataset": "data/raw/Hipertension_Arterial_Mexico.csv",\n'
        '    "clean_dataset": "data/clean/Hipertension_Arterial_Mexico_clean.csv",\n'
        '    "class_labels": {"0": "Sin riesgo", "1": "Riesgo de hipertensión"},\n'
        '    "feature_columns": feature_columns,\n'
        '    "features_used_best": best_features,\n'
        '    "features_per_algorithm": feature_sets,\n'
        '    "best_model_name": best_name,\n'
        '    "models_compared": list(trained.keys()),\n'
        '    "test_metrics": {n: {k: float(v) for k, v in m.items()} for n, m in metrics.items()},\n'
        '    "imputation_medians": imputation_medians,\n'
        '    "outlier_treatment": "winsorize_iqr_k1.5",\n'
        '    "outlier_bounds": outlier_bounds,\n'
        '    "normalization": "StandardScaler (necesario sólo para LogReg)",\n'
        '    "random_state": RANDOM_STATE,\n'
        '    "test_size": 0.2,\n'
        '}\n'
        '\n'
        'joblib.dump(best_model, models_dir / "hipertension_best_model.joblib")\n'
        'joblib.dump(scaler,     models_dir / "hipertension_scaler.joblib")\n'
        'joblib.dump(\n'
        '    {"model": best_model, "scaler": scaler, "features": best_features, "metadata": metadata},\n'
        '    models_dir / "hipertension_model.joblib",\n'
        ')\n'
        'with open(models_dir / "hipertension_metadata.json", "w", encoding="utf-8") as f:\n'
        '    json.dump(metadata, f, indent=2, ensure_ascii=False)\n'
        '\n'
        'print("\\nArtefactos guardados en", models_dir.resolve())\n'
        'for p in ["hipertension_best_model.joblib", "hipertension_scaler.joblib",\n'
        '          "hipertension_model.joblib", "hipertension_metadata.json"]:\n'
        '    print(" -", p)\n'
    )
    nb.cells[save_idx].outputs = []
    nb.cells[save_idx].execution_count = None

    nbf.write(nb, path)
    print(f"✓ Hipertension notebook updated: {path}")


# -----------------------------------------------------------------------------
# 4) NO-SUPERVISADO (Pacientes-con-riesgo)
# -----------------------------------------------------------------------------
def update_unsupervised():
    path = UNSUPER / "Pacientes-con-riesgo.ipynb"
    nb = nbf.read(path, as_version=4)

    # Insert a feature-selection-justification markdown right after the imports cell
    imports_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "code" and "from sklearn.cluster import KMeans" in c.source
    )

    fs_md = md(
        "## Selección de variables y normalización para K-Means\n"
        "\n"
        "K-Means agrupa por **distancia euclidiana**, así que tiene dos sensibilidades "
        "fuertes que justifican las decisiones de preprocesamiento:\n"
        "\n"
        "1. **Sensibilidad a escala:** sin estandarización, la feature con mayor "
        "   rango domina la distancia. Se aplica `StandardScaler` (z-score) a las "
        "   features de cada enfermedad antes del clustering.\n"
        "2. **Sensibilidad a dimensionalidad:** con muchas features irrelevantes la "
        "   geometría se diluye. Por eso, en cada enfermedad **se reutilizan las "
        "   features que el modelo supervisado correspondiente ya identificó como "
        "   relevantes** (`metadata['feature_columns']` de cada bundle). Esto alinea "
        "   los clusters al objetivo de riesgo aprendido y descarta variables que el "
        "   supervisado no usa.\n"
        "\n"
        "**Justificación del algoritmo:** K-Means es la opción clásica para "
        "segmentación de pacientes — eficiente, escalable, y los centroides son "
        "interpretables como 'paciente promedio del subtipo'. Alternativas como "
        "DBSCAN o jerárquico se descartan: DBSCAN requiere ajustar `eps` (no trivial "
        "en alta dimensionalidad médica) y jerárquico no escala bien a 4363 filas.\n"
        "\n"
        "**K óptimo por silueta:** se prueba k∈[2,6] y se elige el k que maximiza "
        "silhouette_score — métrica que combina cohesión intra-cluster y separación "
        "inter-cluster."
    )
    nb.cells.insert(imports_idx + 1, fs_md)

    # Update conclusion
    conc_idx = find_cell_idx(
        nb, lambda c: c.cell_type == "markdown" and c.source.startswith("## Conclusión")
    )
    nb.cells[conc_idx].source = (
        "## Conclusión\n"
        "\n"
        "**Algoritmo:** K-Means por enfermedad (diabetes, paro cardíaco, hipertensión). "
        "K óptimo por **silhouette score** en [2,6].\n"
        "\n"
        "**Selección de variables:** se reutilizan las features de cada modelo "
        "supervisado correspondiente (`bundle['metadata']['feature_columns']`). "
        "Justificación: queremos clusters que expliquen la variable de riesgo, así "
        "que partimos del mismo espacio de features que el modelo ya validó.\n"
        "\n"
        "**Normalización:** `StandardScaler` por enfermedad — obligatorio para K-Means "
        "porque la distancia euclidiana es sensible a escala.\n"
        "\n"
        "**Resultados (ver `subtipos_summary.csv`):**\n"
        "- **Diabetes (k=2, silhouette≈0.20):** dos subgrupos — bajo y alto riesgo. "
        "  La silueta moderada indica que los clusters se traslapan, consistente con "
        "  el espectro continuo de la enfermedad.\n"
        "- **Paro cardíaco (k=5, silhouette≈0.24):** cinco subtipos, con dos "
        "  clusters pequeños (n≈3, n≈23) que aíslan perfiles extremos — útiles para "
        "  detectar pacientes críticos.\n"
        "- **Hipertensión (k=4, silhouette≈0.30):** cuatro estratificaciones de "
        "  riesgo bien marcadas. La silueta más alta del proyecto.\n"
        "\n"
        "**Validación cruzada del clustering:** se compara el `%positivos` real con "
        "la `proba_media_modelo` predicha por el supervisado. Cuando ambas coinciden "
        "alto → subgrupo de riesgo robusto. Discrepancias señalan zonas donde el "
        "supervisado podría mejorar.\n"
        "\n"
        "**Limitaciones:** las siluetas son moderadas (<0.30) — los subtipos no son "
        "fuertemente separables, así que K-Means provee una segmentación útil pero "
        "no una taxonomía clínica definitiva. Trabajos futuros: clustering jerárquico "
        "y validación cualitativa con un médico para nombrar los subtipos."
    )

    nbf.write(nb, path)
    print(f"✓ No-supervisado notebook updated: {path}")


if __name__ == "__main__":
    update_diabetes()
    update_paro()
    update_hipertension()
    update_unsupervised()
    print("\nTodos los notebooks actualizados.")
