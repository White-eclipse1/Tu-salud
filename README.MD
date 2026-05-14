# Tu Salud

Sistema educativo de analisis predictivo de riesgos medicos. El proyecto integra
limpieza de datos, entrenamiento de modelos supervisados, perfiles no
supervisados con K-Means, versionado de datos/modelos con DVC, una API FastAPI y
una interfaz web estatica.

> Importante: este proyecto no es un sistema de diagnostico medico. Las
> probabilidades son salidas de modelos entrenados con datasets especificos y
> deben interpretarse como apoyo academico o demostrativo.

## Resumen Del Proyecto

Tu Salud estima riesgos para tres rubros:

- Diabetes
- Hipertension arterial
- Paro cardiaco

La app hace dos lecturas complementarias:

- Modelos supervisados: predicen una probabilidad individual para cada rubro.
- K-Means no supervisado: asigna al paciente a perfiles exploratorios por
  similitud estadistica. Estos perfiles no significan que el paciente tenga una
  enfermedad; solo describen grupos parecidos dentro de los datos.

El flujo completo esta controlado por DVC:

```text
Base-de-datos/raw.dvc
        |
   clean_data
    /    |    \
train_diabetes
train_paro_cardiaco
train_hipertension
        \    |    /
 train_patient_profiles
        |
 validate_pipeline
```

## Estructura

```text
Tu-salud/
  App/
    index.html              Interfaz principal
    about.html              Explicacion visual del proyecto
    css/styles.css          Estilos
    js/app.js               Conexion con API y render de resultados

  backend/
    main.py                 API FastAPI
    README.md               Guia corta del backend

  Base-de-datos/
    raw.dvc                 Datos crudos versionados por DVC
    raw/                    Datasets originales, no versionados por Git
    cleaned/                Datasets limpios generados por DVC

  Supervisado/
    Diabetes-Prediction.ipynb
    Hipertension_Arterial_Mexico.ipynb
    Prediccion-paro-cardiaco.ipynb

  No-supervisado/
    Pacientes-con-riesgo.ipynb

  scripts/
    generate_cleaned_data.py
    train_diabetes.py
    train_hipertension.py
    train_paro_cardiaco.py
    train_patient_profiles.py
    validate_pipeline.py

  models/                  Artefactos generados por DVC
  reports/                 Reportes/metricas generados por DVC
  dvc.yaml                 Definicion del pipeline
  dvc.lock                 Estado reproducible del pipeline
  requirements.txt
```

## Datos

Los datos crudos estan en `Base-de-datos/raw/` y se versionan con DVC mediante
`Base-de-datos/raw.dvc`.

Datasets usados:

- `diabetes.csv`
- `Hipertension_Arterial_Mexico.csv`
- `Medicaldataset.csv`

Los datasets limpios se generan automaticamente en `Base-de-datos/cleaned/`:

- `diabetes_cleaned.csv`
- `Hipertension_Arterial_Mexico_cleaned.csv`
- `Medicaldataset_cleaned.csv`

`Base-de-datos/cleaned/` no debe editarse manualmente. Si cambian los datos
crudos o los scripts, se regenera con:

```powershell
dvc repro clean_data
```

## Pipeline DVC

El pipeline principal esta definido en `dvc.yaml`.

### 1. `clean_data`

Comando:

```powershell
python scripts/generate_cleaned_data.py
```

Hace:

- Diabetes: reemplaza ceros invalidos en glucosa, presion, skin thickness,
  insulina e IMC; despues imputa medianas.
- Paro cardiaco: normaliza `Result` a `0/1`.
- Hipertension: elimina columnas tipo ID, codifica sexo si aplica, reemplaza
  ceros clinicamente imposibles e imputa medianas.

### 2. `train_diabetes`

Comando:

```powershell
python scripts/train_diabetes.py
```

Modelo final:

- `RandomForestClassifier`

Feature engineering:

- Categorias de IMC
- Grupos de edad
- Categorias de glucosa
- Categorias de insulina
- Categorias de presion arterial

Artefactos:

- `models/diabetes_model.joblib`
- `models/diabetes_random_forest.joblib`
- `models/diabetes_metadata.json`

Metricas actuales:

```text
accuracy  = 0.7468
precision = 0.6744
recall    = 0.5370
f1_score  = 0.5979
roc_auc   = 0.8231
```

### 3. `train_paro_cardiaco`

Comando:

```powershell
python scripts/train_paro_cardiaco.py
```

Modelos comparados:

- Support Vector Machine
- K-Nearest Neighbors
- Logistic Regression

Modelo final:

- `LogisticRegression`

Correccion metodologica importante:

- El `StandardScaler` se ajusta solo con `X_train`.
- Esto evita fuga de datos desde test hacia train.

Artefactos:

- `models/paro_cardiaco_model.joblib`
- `models/paro_cardiaco_best_model.joblib`
- `models/paro_cardiaco_scaler.joblib`
- `models/paro_cardiaco_metadata.json`

Metricas del modelo final:

```text
accuracy  = 0.7803
precision = 0.8197
recall    = 0.8230
f1        = 0.8214
roc_auc   = 0.8677
```

### 4. `train_hipertension`

Comando:

```powershell
python scripts/train_hipertension.py
```

Modelos comparados:

- Calibrated Logistic Regression
- Random Forest
- Gradient Boosting

Modelo usado por la app:

- `CalibratedLogisticRegression`

Por que no se usa el arbol con mejor F1:

- Random Forest y Gradient Boosting alcanzan metricas muy altas en este dataset.
- En una aplicacion medica educativa, probabilidades casi perfectas pueden dar
  una falsa sensacion de certeza.
- Por prudencia, se usa un modelo calibrado y mas interpretable.

Medida adicional:

- `probability_upper_bound = 0.95`
- Esto evita que HTA se muestre como 99% o 100% de certeza sin validacion
  clinica externa.

Artefactos:

- `models/hipertension_model.joblib`
- `models/hipertension_best_model.joblib`
- `models/hipertension_scaler.joblib`
- `models/hipertension_metadata.json`

Metricas del modelo usado por la app:

```text
accuracy  = 0.8465
precision = 0.8499
recall    = 0.9254
f1        = 0.8861
roc_auc   = 0.9033
brier     = 0.1160
log_loss  = 0.3782
```

### 5. `train_patient_profiles`

Comando:

```powershell
python scripts/train_patient_profiles.py
```

Objetivo:

- Crear perfiles no supervisados con K-Means.
- Elegir `k` usando silhouette en el rango `[2, 6]`.
- Guardar tasas y probabilidades supervisadas solo como referencia posterior.

Interpretacion correcta:

- Un cluster indica similitud estadistica.
- Un cluster no significa diagnostico.
- Un cluster no debe llamarse "grupo con enfermedad".

Resultados actuales:

```text
diabetes       k = 2, silhouette = 0.1960
paro_cardiaco  k = 6, silhouette = 0.2253
hipertension   k = 6, silhouette = 0.3011
```

Artefactos:

- `models/subtipos_kmeans.joblib`
- `models/subtipos_metadata.json`
- `models/subtipos_summary.csv`

### 6. `validate_pipeline`

Comando:

```powershell
python scripts/validate_pipeline.py
```

Hace:

- Verifica que existan todos los artefactos esperados.
- Carga todos los `.joblib`.
- Inicializa el backend.
- Ejecuta una prediccion de prueba.
- Ejecuta una asignacion K-Means de prueba.
- Guarda el reporte en `reports/pipeline_validation.json`.

Metricas actuales de validacion:

```text
status                      = ok
diabetes                    = 35.0577%
hipertension                = 95.0000%
paro cardiaco               = 37.3086%
riesgo general              = 56.8253%
referencia general perfiles = 53.9305%
```

## Reproducir Todo El Proyecto

### 1. Crear o activar ambiente

Ejemplo con conda:

```powershell
conda activate mineriadedatos
```

O con venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Descargar datos/modelos desde DVC

Si el repo viene sin datos ni modelos locales:

```powershell
dvc pull
```

El remoto configurado actualmente es:

```text
onedrive -> ..\Tu-salud-dvc-remote
```

### 3. Ejecutar el pipeline completo

```powershell
dvc repro
```

### 4. Ver estado

```powershell
dvc status
```

Salida esperada:

```text
Data and pipelines are up to date.
```

### 5. Ver el DAG

```powershell
dvc dag
```

### 6. Ver metricas finales

```powershell
dvc metrics show
```

### 7. Subir cambios de datos/modelos al remoto

```powershell
dvc push
```

## Ejecutar La API

Desde la raiz del proyecto:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints:

```text
GET  /health
POST /predict
POST /cluster
```

`/predict` devuelve probabilidades supervisadas:

```json
{
  "diabetes": { "pct": 35.05, "level": "moderate" },
  "hipert": { "pct": 95.0, "level": "critical" },
  "cardio": { "pct": 37.30, "level": "moderate" },
  "general": { "pct": 56.82, "level": "high" }
}
```

`/cluster` devuelve perfiles K-Means:

```json
{
  "diabetes": {
    "cluster": 1,
    "group_label": "Perfil metabolico B",
    "profile_reference_pct": 18.92
  },
  "general": {
    "profile_reference_pct": 53.93,
    "level": "high"
  }
}
```

## Ejecutar La App Web

1. Levanta el backend:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

2. Abre en el navegador:

```text
App/index.html
```

La app primero llama a `/cluster` para obtener perfiles K-Means y despues a
`/predict` para obtener probabilidades supervisadas.

## Notebooks

Los notebooks documentan el analisis exploratorio y el entrenamiento:

- `Supervisado/Diabetes-Prediction.ipynb`
- `Supervisado/Hipertension_Arterial_Mexico.ipynb`
- `Supervisado/Prediccion-paro-cardiaco.ipynb`
- `No-supervisado/Pacientes-con-riesgo.ipynb`

Los scripts en `scripts/` son la version deterministica usada por DVC. Si se
quiere reproducibilidad, se debe usar `dvc repro`. Si se quiere explicacion,
graficas y exploracion, se pueden abrir los notebooks.

## Decisiones Metodologicas Importantes

### No redondeo interno de probabilidades

El backend conserva probabilidades como `float`. En frontend se muestran con dos
decimales para lectura, pero no se convierten a enteros.

### Hipertension no se reporta como 99% o 100%

El modelo de HTA puede saturarse por la forma del dataset y por variables que
parecen estar muy relacionadas con la etiqueta. Para evitar una salida
clinicamente enganosa:

- Se usa regresion logistica calibrada.
- Se conserva auditoria de modelos mas fuertes.
- Se aplica limite superior de 95% en la app.

### K-Means no diagnostica

K-Means no usa etiquetas para formar clusters. Por eso el proyecto habla de
perfiles exploratorios, no de "clusters enfermos". Las tasas por perfil son
descriptivas y se calculan despues de entrenar el cluster.

### Unidad de estatura

El dataset de hipertension usa estatura en centimetros. El formulario puede
recibir valores en metros, por ejemplo `1.68`; el backend los convierte a `168`
cm antes de alimentar el modelo.

## Archivos Generados Por DVC

Datos limpios:

```text
Base-de-datos/cleaned/
```

Modelos:

```text
models/
```

Reportes:

```text
reports/pipeline_validation.json
```

Estos archivos estan ignorados por Git y controlados por DVC.

## Comandos Utiles

```powershell
# Reproducir todo
dvc repro

# Reproducir una stage concreta
dvc repro train_hipertension

# Ver estado de datos y pipeline
dvc status

# Ver grafo del pipeline
dvc dag

# Ver metricas finales
dvc metrics show

# Descargar artefactos desde remoto
dvc pull

# Subir artefactos al remoto
dvc push

# Ejecutar API
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## Limitaciones

- No hay validacion clinica externa.
- Los datasets pueden no representar a toda la poblacion.
- Las probabilidades dependen de la calidad y distribucion de los datos usados.
- Los umbrales `low`, `moderate`, `high` y `critical` son criterios del proyecto,
  no guias clinicas oficiales.
- El riesgo general combina modelos con pesos definidos en la app:
  - Diabetes: 30%
  - Hipertension: 35%
  - Paro cardiaco: 35%
- El modelo no reemplaza evaluacion medica profesional.

## Estado Actual

Ultima validacion del pipeline:

```text
dvc repro        OK
dvc status       Data and pipelines are up to date
dvc metrics show status=ok
dvc push         14 files pushed
```

El proyecto queda listo para ser reproducido con DVC y usado localmente con la
API FastAPI y la interfaz web.
