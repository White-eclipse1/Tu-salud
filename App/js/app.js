const API_URL = 'http://localhost:8000/predict';
const CLUSTER_URL = 'http://localhost:8000/cluster';

const RISK_LEVELS = {
  low: {
    label: 'Bajo',
    class: 'low',
    badge: 'badge-low',
    note: 'Los indicadores estan dentro de rangos saludables. Se recomienda mantener habitos preventivos.',
  },
  moderate: {
    label: 'Moderado',
    class: 'moderate',
    badge: 'badge-moderate',
    note: 'Algunos indicadores sugieren vigilancia. Se recomienda consultar a un medico.',
  },
  high: {
    label: 'Alto',
    class: 'high',
    badge: 'badge-high',
    note: 'Indicadores fuera de rango. Evaluacion medica profesional recomendada a la brevedad.',
  },
  critical: {
    label: 'Critico',
    class: 'critical',
    badge: 'badge-critical',
    note: 'Valores criticos detectados. Se recomienda atencion medica urgente.',
  },
};

const CONDITIONS = [
  { id: 'diabetes', label: 'Riesgo - Diabetes', icon: 'Dx' },
  { id: 'hipert', label: 'Riesgo - Hipertension Arterial', icon: 'HTA' },
  { id: 'cardio', label: 'Riesgo - Paro Cardiaco', icon: 'PC' },
];

document.addEventListener('DOMContentLoaded', () => {
  initRangeInputs();
  initForm();
  initResetButton();
  setCurrentYear();
});

function initRangeInputs() {
  const ranges = document.querySelectorAll('.form-range');
  ranges.forEach(range => {
    const display = document.getElementById(range.dataset.display);
    if (!display) return;
    const update = () => {
      display.textContent = `${range.value}${range.dataset.unit || ''}`;
      const pct = ((range.value - range.min) / (range.max - range.min)) * 100;
      range.style.background = `linear-gradient(90deg, var(--blue-primary) ${pct}%, var(--bg-input) ${pct}%)`;
    };
    update();
    range.addEventListener('input', update);
  });
}

function initForm() {
  const form = document.getElementById('patient-form');
  if (!form) return;

  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const btn = form.querySelector('#analyze-btn');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
      const data = readFormData(form);
      const clusters = await getClustersFromModel(data);
      renderUnsupervisedAnalysis(data, clusters);
      const risks = await getRisksFromModel(data);
      renderResults(data, risks, clusters);
    } catch (error) {
      renderError(error.message || 'No se pudo conectar con el backend de prediccion.');
    } finally {
      btn.classList.remove('loading');
      btn.disabled = false;
    }

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

function readFormData(form) {
  const fd = new FormData(form);
  const data = { name: fd.get('patient-name') || 'Paciente' };

  fd.forEach((value, key) => {
    if (key === 'patient-name' || value === '') return;
    const numericValue = Number(value);
    data[key] = Number.isNaN(numericValue) ? value : numericValue;
  });

  return data;
}

async function getRisksFromModel(data) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map(item => item.msg).join(', ')
      : payload.detail;
    throw new Error(detail || 'Error en la prediccion del backend.');
  }
  return payload;
}

async function getClustersFromModel(data) {
  const response = await fetch(CLUSTER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map(item => item.msg).join(', ')
      : payload.detail;
    throw new Error(detail || 'Error en el modelo no supervisado.');
  }
  return payload;
}

function renderUnsupervisedAnalysis(data, clusters) {
  const container = document.getElementById('results-section');
  if (!container) return;

  container.innerHTML = `
    <div class="results-section">
      ${buildClusterHTML(clusters, null)}
      <div class="card" style="margin-top:1.2rem">
        <div class="card-header" style="margin-bottom:0;padding-bottom:0;border-bottom:0">
          <div class="card-header-icon blue">ML</div>
          <div>
            <div class="card-title">Ejecutando modelos supervisados...</div>
            <div class="card-subtitle">Comparando perfiles exploratorios con las probabilidades individuales</div>
          </div>
        </div>
      </div>
    </div>
  `;
  container.classList.remove('hidden');
}

function renderResults(data, risks, clusters) {
  const container = document.getElementById('results-section');
  if (!container) return;

  container.innerHTML = buildResultsHTML(data, risks, clusters);
  container.classList.remove('hidden');

  requestAnimationFrame(() => {
    setTimeout(() => {
      container.querySelectorAll('.result-bar-fill').forEach(bar => {
        bar.style.width = `${clampPercent(Number(bar.dataset.width))}%`;
      });
    }, 100);
  });

  animateGauge(risks.general.pct, risks.general.level);
}

function renderError(message) {
  const container = document.getElementById('results-section');
  if (!container) return;
  container.innerHTML = `
    <div class="results-section">
      <div class="disclaimer-banner" style="margin-bottom:0">
        <span class="icon">!</span>
        <p><strong>No se pudo generar la prediccion:</strong> ${escapeHTML(message)}</p>
      </div>
    </div>
  `;
  container.classList.remove('hidden');
}

function buildResultsHTML(data, risks, clusters) {
  const sexLabel = data.sex === 'M' || data.sex === 1 ? 'Masculino' : data.sex === 'F' || data.sex === 2 ? 'Femenino' : 'N/E';
  const generalInfo = RISK_LEVELS[risks.general.level];

  return `
    <div class="results-section">
      ${clusters ? buildClusterHTML(clusters, risks) : ''}

      <div class="patient-summary">
        <div class="patient-avatar">${sexLabel === 'Femenino' ? 'F' : 'M'}</div>
        <div>
          <div class="patient-name">${escapeHTML(data.name)}</div>
          <div class="patient-meta">Analisis generado ${formatTime()}</div>
        </div>
        <div class="patient-chips">
          <span class="chip">${data.age} anos</span>
          <span class="chip">${sexLabel}</span>
          <span class="chip">IMC ${data.bmi}</span>
          <span class="chip">Glucosa ${data.glucose} mg/dL</span>
          <span class="chip">PA ${data.bp_systolic}/${data.bp_diastolic} mmHg</span>
        </div>
      </div>

      <div class="results-grid">
        <div class="result-card ${risks.general.level} result-general">
          <div class="result-general-gauge">
            <svg width="120" height="120" viewBox="0 0 120 120" id="gauge-svg">
              <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(20,96,112,0.10)" stroke-width="10"/>
              <circle cx="60" cy="60" r="50" fill="none" stroke-width="10" stroke-linecap="round" stroke-dasharray="314" stroke-dashoffset="314" id="gauge-circle"/>
            </svg>
            <div class="gauge-text">
              <span class="gauge-pct" id="gauge-pct-val" style="color:${gaugeColor(risks.general.level)}">0%</span>
              <span class="gauge-label">Riesgo</span>
            </div>
          </div>
          <div class="result-general-info">
            <div class="result-condition">Evaluacion de Riesgo General</div>
            <h3>Nivel ${generalInfo.label}
              <span class="result-badge ${generalInfo.badge}" style="margin-left:8px">${generalInfo.label}</span>
            </h3>
            <p>${generalInfo.note}</p>
            <span class="text-muted" style="font-size:0.75rem">Resultado generado por modelos entrenados. No es un diagnostico medico.</span>
          </div>
        </div>

        ${CONDITIONS.map(condition => buildConditionCard(condition, risks[condition.id])).join('')}
      </div>

      ${clusters ? buildClusterComparisonHTML(clusters, risks) : ''}

      <div style="margin-top:1.5rem; text-align:center">
        <button class="btn btn-secondary" onclick="window.print()">
          <span class="btn-icon">PDF</span> Imprimir reporte
        </button>
        <button class="btn btn-ghost" id="new-analysis-btn" style="margin-left:0.5rem">
          <span class="btn-icon">+</span> Nuevo analisis
        </button>
      </div>
    </div>
  `;
}

function buildClusterHTML(clusters, risks) {
  const rows = [
    { key: 'diabetes', label: 'Diabetes', color: '#0f7c93' },
    { key: 'hipert', label: 'Hipertension', color: '#c2413a' },
    { key: 'cardio', label: 'Paro cardiaco', color: '#18a878' },
  ];

  return `
    <div class="card cluster-card">
      <div class="card-header">
        <div class="card-header-icon green">KM</div>
        <div>
          <div class="card-title">Modelo no supervisado: perfiles K-Means</div>
          <div class="card-subtitle">Primero se ubica al paciente dentro del perfil mas parecido a sus datos</div>
        </div>
      </div>

      <div class="cluster-map-grid" aria-label="Mapa grafico de pertenencia por clusters">
        ${rows.map(row => buildClusterMap(row, clusters[row.key], risks ? risks[row.key] : null)).join('')}
      </div>

      <div class="cluster-summary">
        <div>
          <span class="stat-label">Referencia general por perfiles</span>
          <strong>${formatPercent(clusters.general.profile_reference_pct ?? clusters.general.cluster_risk_pct)}</strong>
        </div>
        <p>El perfil marcado indica similitud estadistica. El porcentaje es una referencia descriptiva posterior, no un diagnostico ni una etiqueta de enfermedad.</p>
      </div>
    </div>
  `;
}

function buildClusterMap(row, item, risk) {
  const patientGroup = getPatientGroup(item);
  const profilePct = item.profile_reference_pct ?? item.cluster_risk_pct;
  const relationHTML = risk ? buildMiniRelation(profilePct, risk.pct) : '';
  const maxN = Math.max(...item.groups.map(group => group.n));

  return `
    <section class="cluster-map-card">
      <div class="cluster-map-head">
        <div>
          <strong>${row.label}</strong>
          <span>${item.group_label}</span>
        </div>
        <span class="cluster-pill ${item.level}">Perfil ${item.cluster}</span>
      </div>

      <div class="cluster-real-summary" style="--cluster-color:${row.color}">
        <div class="assigned-cluster-card">
          <div class="assigned-cluster-ring">
            <span>${item.cluster}</span>
          </div>
          <div>
            <span class="result-condition">Perfil asignado por K-Means</span>
            <strong>${item.group_label}</strong>
            <p>El modelo asigno este perfil usando la distancia real del paciente al centroide entrenado.</p>
          </div>
        </div>

        <div class="cluster-groups-list">
          ${item.groups.map(group => `
            <div class="cluster-group-row ${group.is_patient_group ? 'selected' : ''}">
              <div class="cluster-group-id">
                <span>${group.cluster}</span>
              </div>
              <div class="cluster-group-body">
                <div class="cluster-group-title">
                  <strong>${group.is_patient_group ? item.group_label : `Perfil ${group.cluster}`}</strong>
                  <span>${group.is_patient_group ? 'Paciente pertenece aqui' : 'Perfil disponible'}</span>
                </div>
                <div class="cluster-group-meter">
                  <div style="width:${Math.max(8, (group.n / maxN) * 100)}%;background:${group.is_patient_group ? row.color : 'rgba(20,96,112,0.28)'}"></div>
                </div>
              </div>
              <div class="cluster-group-values">
                <span>n=${group.n}</span>
                <strong>${formatPercent(group.profile_reference_pct ?? group.cluster_risk_pct)}</strong>
              </div>
            </div>
          `).join('')}
        </div>
      </div>

      <div class="cluster-map-stats">
        <div>
          <span>Perfil asignado</span>
          <strong>${item.group_label}</strong>
        </div>
        <div>
          <span>Referencia del perfil</span>
          <strong>${formatPercent(profilePct)}</strong>
        </div>
        <div>
          <span>Pacientes similares</span>
          <strong>${item.n}</strong>
        </div>
        <div>
          <span>Distancia al centroide</span>
          <strong>${item.distance_to_center}</strong>
        </div>
      </div>

      <div class="cluster-affinity">
        <div class="cluster-affinity-label">
          <span>Afinidad al centroide</span>
          <strong>${formatPercent(patientGroup.affinity_pct)}</strong>
        </div>
        <div class="cluster-affinity-track">
          <div style="width:${clampPercent(patientGroup.affinity_pct)}%;background:${row.color}"></div>
        </div>
      </div>

      ${relationHTML}
      <p class="cluster-footnote">k=${item.k} ? silhouette ${item.silhouette} ? distancia ${item.distance_to_center} ? exploratorio</p>
    </section>
  `;
}

function getPatientGroup(item) {
  return item.groups.find(group => group.is_patient_group) || item.groups[0];
}

function buildMiniRelation(clusterPct, modelPct) {
  const diff = Math.abs(modelPct - clusterPct);
  const relation = diff <= 15 ? 'Relacion fuerte' : diff <= 30 ? 'Relacion parcial' : 'Diferencia alta';
  return `
    <div class="cluster-relation">
      <span>Perfil ${formatPercent(clusterPct)}</span>
      <span>Supervisado ${formatPercent(modelPct)}</span>
      <strong>${relation}</strong>
    </div>
  `;
}

function buildClusterComparisonHTML(clusters, risks) {
  const items = [
    { key: 'diabetes', label: 'Diabetes' },
    { key: 'hipert', label: 'Hipertension' },
    { key: 'cardio', label: 'Paro cardiaco' },
  ];

  return `
    <div class="card comparison-card">
      <div class="card-header">
        <div class="card-header-icon amber">CMP</div>
        <div>
          <div class="card-title">Comparacion perfil vs modelo supervisado</div>
          <div class="card-subtitle">Referencia exploratoria contra prediccion individual</div>
        </div>
      </div>

      <div class="comparison-grid">
        ${items.map(item => {
          const clusterPct = clusters[item.key].profile_reference_pct ?? clusters[item.key].cluster_risk_pct;
          const modelPct = risks[item.key].pct;
          const diff = Math.abs(modelPct - clusterPct);
          const relation = diff <= 15 ? 'Relacion fuerte' : diff <= 30 ? 'Relacion parcial' : 'Diferencia alta';
          return `
            <div class="comparison-item">
              <div class="result-condition">${item.label}</div>
              <div class="comparison-values">
                <span>Perfil ${formatPercent(clusterPct)}</span>
                <span>Modelo ${formatPercent(modelPct)}</span>
              </div>
              <strong>${relation}</strong>
              <p>Diferencia descriptiva: ${formatPercent(diff).replace('%', '')} puntos porcentuales.</p>
            </div>
          `;
        }).join('')}
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
      <div class="result-percentage">${formatPercent(risk.pct)}</div>
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
  circle.style.stroke = gaugeColor(level);
  circle.style.transition = 'stroke-dashoffset 1.4s cubic-bezier(0.4,0,0.2,1)';

  setTimeout(() => {
    circle.style.strokeDashoffset = circumference - (circumference * clampPercent(pct) / 100);
    animateCounter(display, 0, pct, 1400, '%');
  }, 150);
}

function animateCounter(el, from, to, duration, suffix = '') {
  const start = performance.now();
  const update = (now) => {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = formatPercent(from + (to - from) * eased);
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function gaugeColor(level) {
  return { low: '#18a878', moderate: '#b7791f', high: '#b45309', critical: '#c2413a' }[level];
}

function clampPercent(value) {
  return Math.max(0, Math.min(100, Number(value) || 0));
}

function formatPercent(value) {
  return `${clampPercent(value).toFixed(2)}%`;
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

document.addEventListener('click', (event) => {
  if (event.target && event.target.id === 'new-analysis-btn') {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
});
