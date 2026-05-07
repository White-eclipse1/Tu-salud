/* ============================================================
   TU SALUD — app.js
   Lógica del formulario, simulación de riesgos y renderizado
   ============================================================ */

// ──────────────────────────────────────────────
// CONSTANTES Y CONFIG
// ──────────────────────────────────────────────
const RISK_LEVELS = {
  low:      { label: 'Bajo',     class: 'low',      badge: 'badge-low',      note: 'Los indicadores están dentro de rangos saludables. Se recomienda mantener hábitos preventivos.' },
  moderate: { label: 'Moderado', class: 'moderate',  badge: 'badge-moderate', note: 'Algunos indicadores sugieren vigilancia. Se recomienda consultar a un médico.' },
  high:     { label: 'Alto',     class: 'high',      badge: 'badge-high',     note: 'Indicadores fuera de rango. Evaluación médica profesional recomendada a la brevedad.' },
  critical: { label: 'Crítico',  class: 'critical',  badge: 'badge-critical', note: 'Valores críticos detectados. Se recomienda atención médica urgente.' },
};

const CONDITIONS = [
  { id: 'diabetes',   label: 'Riesgo — Diabetes',              icon: '🩸' },
  { id: 'hipert',     label: 'Riesgo — Hipertensión Arterial',  icon: '❤️' },
  { id: 'cardio',     label: 'Riesgo — Paro Cardíaco',          icon: '⚡' },
];

// ──────────────────────────────────────────────
// INICIALIZACIÓN
// ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initRangeInputs();
  initForm();
  initResetButton();
  setCurrentYear();
});

// ──────────────────────────────────────────────
// RANGE INPUTS — actualizar valor en pantalla
// ──────────────────────────────────────────────
function initRangeInputs() {
  const ranges = document.querySelectorAll('.form-range');
  ranges.forEach(range => {
    const display = document.getElementById(range.dataset.display);
    if (!display) return;
    const update = () => {
      display.textContent = `${range.value}${range.dataset.unit || ''}`;
      // Actualizar color de fondo del track
      const pct = ((range.value - range.min) / (range.max - range.min)) * 100;
      range.style.background = `linear-gradient(90deg, var(--blue-primary) ${pct}%, var(--bg-input) ${pct}%)`;
    };
    update();
    range.addEventListener('input', update);
  });
}

// ──────────────────────────────────────────────
// FORMULARIO — submit
// ──────────────────────────────────────────────
function initForm() {
  const form = document.getElementById('patient-form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn = form.querySelector('#analyze-btn');
    btn.classList.add('loading');
    btn.disabled = true;

    // Leer datos del formulario
    const data = readFormData(form);

    // Simular latencia de modelo (demo)
    await delay(1400);

    // Calcular riesgos simulados
    const risks = simulateRisks(data);

    // Renderizar resultados
    renderResults(data, risks);

    btn.classList.remove('loading');
    btn.disabled = false;

    // Scroll suave a resultados
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
}

function initResetButton() {
  const btn = document.getElementById('reset-btn');
  if (!btn) return;
  btn.addEventListener('click', () => {
    document.getElementById('patient-form').reset();
    initRangeInputs();
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) resultsSection.classList.add('hidden');
  });
}

// ──────────────────────────────────────────────
// LEER FORMULARIO
// ──────────────────────────────────────────────
function readFormData(form) {
  const fd = new FormData(form);
  return {
    name:           fd.get('patient-name')   || 'Paciente',
    age:            parseInt(fd.get('age'))  || 0,
    sex:            fd.get('sex')            || 'N/E',
    glucose:        parseFloat(fd.get('glucose'))  || 0,
    bp_systolic:    parseInt(fd.get('bp_systolic')) || 0,
    bmi:            parseFloat(fd.get('bmi'))       || 0,
    heart_rate:     parseInt(fd.get('heart_rate'))  || 0,
    cholesterol:    parseInt(fd.get('cholesterol')) || 0,
    smoking:        fd.get('smoking')        || 'no',
    physical_activity: fd.get('physical_activity') || 'moderate',
    family_history: fd.get('family_history') || 'no',
  };
}

// ──────────────────────────────────────────────
// SIMULACIÓN DE RIESGOS
// Nota: Algoritmo demostrativo, no médico.
// En producción conectar con modelos Python.
// ──────────────────────────────────────────────
function simulateRisks(d) {
  // Score base por condición — puntos de riesgo acumulados
  let scores = { diabetes: 0, hipert: 0, cardio: 0 };

  // ── Glucosa (normal < 100, pre-diabético 100-125, diabético >= 126)
  if (d.glucose >= 126)     scores.diabetes += 40;
  else if (d.glucose >= 100) scores.diabetes += 20;
  else if (d.glucose >= 90)  scores.diabetes += 5;

  // ── Presión sistólica (normal < 120, elevada 120-129, hipert1 130-139, hipert2 >= 140)
  if (d.bp_systolic >= 160)      scores.hipert += 50; scores.cardio += 20;
  if (d.bp_systolic >= 140)      { scores.hipert += 35; scores.cardio += 15; }
  else if (d.bp_systolic >= 130) { scores.hipert += 20; scores.cardio += 8; }
  else if (d.bp_systolic >= 120) { scores.hipert += 8; }

  // ── IMC (normal 18.5-24.9)
  if (d.bmi >= 35)       { scores.diabetes += 20; scores.cardio += 15; }
  else if (d.bmi >= 30)  { scores.diabetes += 12; scores.cardio += 8; }
  else if (d.bmi >= 25)  { scores.diabetes += 5; scores.cardio += 3; }

  // ── Frecuencia cardíaca (normal 60-100)
  if (d.heart_rate > 120 || d.heart_rate < 45) scores.cardio += 25;
  else if (d.heart_rate > 100 || d.heart_rate < 55) scores.cardio += 12;

  // ── Colesterol
  if (d.cholesterol >= 240)      { scores.cardio += 20; scores.hipert += 5; }
  else if (d.cholesterol >= 200) { scores.cardio += 10; }

  // ── Edad
  const ageFactor = d.age > 65 ? 1.4 : d.age > 50 ? 1.2 : d.age > 40 ? 1.05 : 1.0;
  Object.keys(scores).forEach(k => { scores[k] = Math.round(scores[k] * ageFactor); });

  // ── Tabaquismo
  if (d.smoking === 'yes') { scores.cardio += 20; scores.hipert += 10; }

  // ── Antecedentes familiares
  if (d.family_history === 'yes') {
    scores.diabetes += 15; scores.hipert += 10; scores.cardio += 15;
  }

  // ── Actividad física
  if (d.physical_activity === 'low') {
    scores.diabetes += 10; scores.cardio += 8;
  } else if (d.physical_activity === 'high') {
    scores.diabetes -= 8; scores.cardio -= 5; scores.hipert -= 5;
  }

  // Normalizar a porcentaje 0-100
  const normalize = (v, cap = 100) => Math.min(Math.max(Math.round(v), 0), cap);
  const pct = {
    diabetes: normalize(scores.diabetes),
    hipert:   normalize(scores.hipert),
    cardio:   normalize(scores.cardio),
  };

  // Riesgo general = promedio ponderado
  pct.general = normalize(Math.round(pct.diabetes * 0.3 + pct.hipert * 0.35 + pct.cardio * 0.35));

  // Clasificar nivel
  const classify = (v) =>
    v >= 70 ? 'critical' : v >= 45 ? 'high' : v >= 20 ? 'moderate' : 'low';

  return {
    diabetes: { pct: pct.diabetes, level: classify(pct.diabetes) },
    hipert:   { pct: pct.hipert,   level: classify(pct.hipert) },
    cardio:   { pct: pct.cardio,   level: classify(pct.cardio) },
    general:  { pct: pct.general,  level: classify(pct.general) },
  };
}

// ──────────────────────────────────────────────
// RENDERIZAR RESULTADOS
// ──────────────────────────────────────────────
function renderResults(data, risks) {
  const container = document.getElementById('results-section');
  if (!container) return;

  container.innerHTML = buildResultsHTML(data, risks);
  container.classList.remove('hidden');

  // Animar barras con delay
  requestAnimationFrame(() => {
    setTimeout(() => {
      container.querySelectorAll('.result-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    }, 100);
  });

  // Animar gauge circular
  animateGauge(risks.general.pct, risks.general.level);
}

function buildResultsHTML(data, risks) {
  const sexLabel = data.sex === 'M' ? 'Masculino' : data.sex === 'F' ? 'Femenino' : 'N/E';
  const generalInfo = RISK_LEVELS[risks.general.level];

  return `
    <div class="results-section">
      <!-- Resumen del paciente -->
      <div class="patient-summary">
        <div class="patient-avatar">${data.sex === 'F' ? '👩' : '👨'}</div>
        <div>
          <div class="patient-name">${escapeHTML(data.name)}</div>
          <div class="patient-meta">Análisis generado ${formatTime()}</div>
        </div>
        <div class="patient-chips">
          <span class="chip">${data.age} años</span>
          <span class="chip">${sexLabel}</span>
          <span class="chip">IMC ${data.bmi}</span>
          <span class="chip">Glucosa ${data.glucose} mg/dL</span>
          <span class="chip">PA ${data.bp_systolic} mmHg</span>
        </div>
      </div>

      <!-- Grid de resultados -->
      <div class="results-grid">

        <!-- Riesgo General -->
        <div class="result-card ${risks.general.level} result-general">
          <div class="result-general-gauge">
            <svg width="120" height="120" viewBox="0 0 120 120" id="gauge-svg">
              <circle cx="60" cy="60" r="50"
                fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="10"/>
              <circle cx="60" cy="60" r="50"
                fill="none" stroke-width="10"
                stroke-linecap="round"
                stroke-dasharray="314"
                stroke-dashoffset="314"
                id="gauge-circle"
                class="gauge-ring-${risks.general.level}"/>
            </svg>
            <div class="gauge-text">
              <span class="gauge-pct" id="gauge-pct-val" style="color:${gaugeColor(risks.general.level)}">0%</span>
              <span class="gauge-label">Riesgo</span>
            </div>
          </div>
          <div class="result-general-info">
            <div class="result-condition">Evaluación de Riesgo General</div>
            <h3>Nivel ${generalInfo.label}
              <span class="result-badge ${generalInfo.badge}" style="margin-left:8px">${generalInfo.label}</span>
            </h3>
            <p>${generalInfo.note}</p>
            <span class="text-muted" style="font-size:0.75rem">⚠️ Resultado demostrativo — No es un diagnóstico médico.</span>
          </div>
        </div>

        ${CONDITIONS.map(c => buildConditionCard(c, risks[c.id])).join('')}

      </div>

      <div style="margin-top:1.5rem; text-align:center">
        <button class="btn btn-secondary" onclick="window.print()">
          <span class="btn-icon">🖨️</span> Imprimir reporte
        </button>
        <button class="btn btn-ghost" id="new-analysis-btn" style="margin-left:0.5rem">
          <span class="btn-icon">↩</span> Nuevo análisis
        </button>
      </div>
    </div>
  `;
}

function buildConditionCard(condition, risk) {
  const info = RISK_LEVELS[risk.level];
  return `
    <div class="result-card ${risk.level}">
      <div class="result-card-header">
        <span class="result-condition">${condition.icon} ${condition.label}</span>
        <span class="result-badge ${info.badge}">${info.label}</span>
      </div>
      <div class="result-percentage">${risk.pct}%</div>
      <div class="result-label">Probabilidad de riesgo</div>
      <div class="result-bar-bg">
        <div class="result-bar-fill" data-width="${risk.pct}" style="width:0%"></div>
      </div>
      <div class="result-note">${info.note}</div>
    </div>
  `;
}

function animateGauge(pct, level) {
  const circle = document.getElementById('gauge-circle');
  const display = document.getElementById('gauge-pct-val');
  if (!circle || !display) return;

  const circumference = 314;
  const color = gaugeColor(level);
  circle.style.stroke = color;

  const targetDashoffset = circumference - (circumference * pct / 100);

  // Agregar transición
  circle.style.transition = 'stroke-dashoffset 1.4s cubic-bezier(0.4,0,0.2,1)';

  setTimeout(() => {
    circle.style.strokeDashoffset = targetDashoffset;
    // Contador numérico
    animateCounter(display, 0, pct, 1400, '%');
  }, 150);
}

function animateCounter(el, from, to, duration, suffix = '') {
  const start = performance.now();
  const update = (now) => {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(from + (to - from) * eased) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function gaugeColor(level) {
  return { low: '#00d4a0', moderate: '#ffb800', high: '#ff9f3f', critical: '#ff4757' }[level];
}

// ──────────────────────────────────────────────
// UTILIDADES
// ──────────────────────────────────────────────
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function escapeHTML(str) {
  return String(str).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
  );
}

function formatTime() {
  const now = new Date();
  return now.toLocaleString('es-MX', { dateStyle: 'medium', timeStyle: 'short' });
}

function setCurrentYear() {
  const el = document.getElementById('current-year');
  if (el) el.textContent = new Date().getFullYear();
}

// Delegación de eventos para "Nuevo análisis"
document.addEventListener('click', (e) => {
  if (e.target && e.target.id === 'new-analysis-btn') {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
});
