# Backend Tu Salud

API local para usar los modelos guardados en `models/` desde el frontend.

## Requisitos

Se recomienda Python 3.11 o 3.12. Con Python 3.14 algunas librerias de ciencia de datos pueden intentar compilarse desde fuente y fallar en Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Ejecutar

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Luego abre `App/index.html`. El frontend llamara a:

```text
http://localhost:8000/predict
```

## Endpoints

- `GET /health`: estado del servicio.
- `POST /cluster`: ejecuta los modelos K-Means no supervisados y devuelve el cluster asignado por enfermedad.
- `POST /predict`: recibe los campos del formulario y devuelve:

```json
{
  "diabetes": { "pct": 42, "level": "moderate" },
  "hipert": { "pct": 18, "level": "low" },
  "cardio": { "pct": 73, "level": "critical" },
  "general": { "pct": 46, "level": "high" }
}
```
