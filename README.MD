# Tu Salud: Análisis Predictivo de Riesgos Médicos

## Descripción

Este proyecto tiene como objetivo desarrollar un sistema de predicción de riesgos médicos para pacientes. Utiliza modelos de aprendizaje automático para identificar pacientes con riesgo general mediante técnicas no supervisadas, y posteriormente predice riesgos específicos para tres condiciones: Diabetes, Hipertensión Arterial y Paro Cardíaco utilizando modelos supervisados.

El proyecto se basa en análisis de datos reales, implementando un flujo completo de ciencia de datos que incluye limpieza de datos, normalización, análisis exploratorio y aplicación de múltiples algoritmos de machine learning.

## Objetivo

El objetivo de esta actividad integradora es realizar un proyecto completo de análisis de datos, demostrando habilidades en el manejo de datos, modelado predictivo y buenas prácticas de desarrollo con control de versiones (Git) y gestión de datos (DVC).

## Requisitos Cumplidos

### Base de Datos
- Se utilizaron datasets médicos reales incluyendo:
  - `diabetes.csv`: Datos para predicción de diabetes
  - `Hipertension_Arterial_Mexico.csv`: Datos sobre hipertensión en México
  - `Medicaldataset.csv`: Dataset médico general

### Pasos del Proyecto

1. **Limpieza de Datos**
   - Tratamiento de valores faltantes mediante imputación apropiada
   - Identificación y manejo de outliers utilizando métodos estadísticos
   - Justificación de cada decisión tomada en el proceso de limpieza

2. **Normalización de Datos**
   - Aplicación de técnicas de normalización cuando fue necesario
   - Uso de escalado estándar o min-max según el algoritmo utilizado
   - Justificación de la elección de método de normalización

3. **Análisis de Variables**
   - Análisis exploratorio de datos (EDA)
   - Identificación de correlaciones entre variables
   - Selección de features relevantes para los modelos

4. **Algoritmos Utilizados**
   - **Aprendizaje No Supervisado**: Clustering (K-means o similar) para identificar pacientes de riesgo general
   - **Aprendizaje Supervisado**: Al menos 3 algoritmos adicionales como:
     - Regresión Logística
     - Árboles de Decisión
     - Random Forest
     - SVM
     - Redes Neuronales (opcional)

5. **Trazabilidad de Datos**
   - Uso de DVC para versionado de datasets
   - Control de versiones con Git para código y modelos
   - Documentación completa de cada transformación de datos

6. **Documentación**
   - Todos los pasos están justificados con explicaciones técnicas
   - Análisis documentados en notebooks Jupyter
   - Comentarios en el código explicando decisiones tomadas

7. **Jerarquía de Archivos**
   - `App/`: Interfaz de usuario (HTML)
   - `Base-de-datos/`: Datasets originales y procesados
   - `No-supervisado/`: Análisis no supervisado
   - `Supervisado/`: Modelos supervisados por condición médica

## Estructura del Proyecto

```
Tu-salud/
├── README.MD                    # Este archivo
├── App/
│   └── index.html              # Interfaz web (opcional)
├── Base-de-datos/
│   ├── diabetes.csv
│   ├── Hipertension_Arterial_Mexico.csv
│   └── Medicaldataset.csv
├── No-supervisado/
│   └── Pacientes-con-riesgo.ipynb
└── Supervisado/
    ├── Diabetes-Prediction.ipynb
    ├── Hipertension_Arterial_Mexico.ipynb
    └── Prediccion-paro-cardiaco.ipynb
```

## Instalación y Ejecución

### Requisitos Previos
- Python 3.8+
- Jupyter Notebook
- Git
- DVC

### Instalación
```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd Tu-salud

# Instalar dependencias
pip install -r requirements.txt

# Inicializar DVC
dvc init
dvc add Base-de-datos/
```

### Ejecución
```bash
# Ejecutar notebooks en orden
jupyter notebook No-supervisado/Pacientes-con-riesgo.ipynb
jupyter notebook Supervisado/Diabetes-Prediction.ipynb
# ... continuar con los demás notebooks
```

### Modelos guardados para la app

Los notebooks supervisados entrenan y guardan sus modelos al final usando `joblib`.

Para regenerarlos, ejecutar todas las celdas de:

- `Supervisado/Diabetes-Prediction.ipynb`
- `Supervisado/Prediccion-paro-cardiaco.ipynb`

Los artefactos se guardan en `models/`:

- `diabetes_random_forest.joblib`
- `diabetes_model.joblib`
- `diabetes_metadata.json`
- `paro_cardiaco_best_model.joblib`
- `paro_cardiaco_scaler.joblib`
- `paro_cardiaco_model.joblib`
- `paro_cardiaco_metadata.json`

Los archivos `*_model.joblib` son bundles pensados para la aplicacion, porque incluyen el modelo y los metadatos necesarios para preparar los datos antes de predecir.

## Conclusiones

[Espacio para conclusiones finales del proyecto]

- Resumen de los resultados obtenidos
- Comparación de rendimiento entre algoritmos
- Limitaciones identificadas
- Posibles mejoras futuras

## Autores

[Nombre del autor/equipo]

## Licencia

[Información de licencia si aplica] 
