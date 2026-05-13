# Tu Salud - Frontend

Interfaz web para capturar las variables que esperan los modelos guardados en `models/`.

## Como usar

1. Levanta el backend:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

2. Abre `App/index.html` en el navegador.

El formulario primero envia los datos a `http://localhost:8000/cluster` para ubicar al paciente en los perfiles K-Means no supervisados. Despues llama a `http://localhost:8000/predict` y compara esos clusters contra las probabilidades supervisadas.

## Archivos

```text
App/
  index.html      Formulario y resultados
  about.html      Informacion del proyecto
  css/styles.css  Estilos
  js/app.js       Conexion con backend y render de resultados
```

## Aviso

Las predicciones no constituyen diagnostico medico.
