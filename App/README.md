# Tu Salud — Frontend

Interfaz web del sistema de análisis predictivo de riesgos médicos.

## Estructura

```
App/
├── index.html        → Dashboard principal: formulario + resultados de riesgo
├── about.html        → Descripción del proyecto, modelos y datasets
├── css/
│   └── styles.css    → Estilos globales (tema oscuro médico, responsivo)
└── js/
    └── app.js        → Lógica: lectura de formulario, simulación y renderizado
```

## Cómo usar

Abrir `index.html` directamente en el navegador. No requiere servidor ni dependencias externas.

## Conectar con backend Python

En `js/app.js`, reemplazar la función `simulateRisks(data)` por una llamada al endpoint:

```js
async function getRisksFromModel(data) {
  const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return await response.json();
  // Espera: { diabetes:{pct,level}, hipert:{pct,level}, cardio:{pct,level}, general:{pct,level} }
}
```

## Tecnologías

- HTML5, CSS3, JavaScript (vanilla)
- Google Fonts: DM Sans + Space Mono
- Sin frameworks ni dependencias externas

## Aviso

Los resultados mostrados son simulaciones demostrativas y **no constituyen diagnóstico médico**.
