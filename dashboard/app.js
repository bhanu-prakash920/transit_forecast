/* TransitOps dashboard.
 * Renders dashboard/data.json (built by data_export.py) and live results from
 * POST /api/simulate. All values come from those two sources. */

'use strict';

const $ = (id) => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';
const COLORS = {
  accent: '#1D4ED8', accent2: '#5B82DF', accent3: '#93AEEB', accent4: '#C9D6F5', accentSoft: '#EEF2FB',
  ink: '#15171C', warn: '#C2410C', warnSoft: '#FDEEE3', amber: '#F2A93B', neutral: '#B9B5A9', grid: '#ECEAE3',
};

let DATA = null;
let SIM = null;
let API_OK = false;
let pending = 0;

// ── Small helpers ───────────────────────────────────────────────────────────
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'tip') node.dataset.tip = v;
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'text') node.textContent = v;
    else if (k === 'tip') node.dataset.tip = v;
    else node.setAttribute(k, v);
  }
  return node;
}
const hh = (h) => String(h).padStart(2, '0');
const sum = (a) => a.reduce((s, x) => s + x, 0);
const num = (v, d = 0) => Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 1) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${num(Math.abs(v), d)}%`;
const money = (v) => `${DATA.meta.currency}${num(Math.round(v))}`;
const niceMax = (v) => {
  const p = 10 ** Math.floor(Math.log10(Math.max(v, 1)));
  return [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].map((m) => m * p).find((m) => m >= v);
};
const longDate = (d) => new Date(`${d}T00:00:00`).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
const shortDate = (d) => new Date(`${d}T00:00:00`).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' });
const monthName = (d) => new Date(`${d}T00:00:00`).toLocaleDateString('en-GB', { month: 'long', year: 'numeric' });
const icon = (paths, color) => {
  const s = svg('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true' });
  paths.forEach((d) => s.append(svg('path', { d })));
  return s;
};
function download(name, rows) {
  const csv = rows.map((r) => r.map((c) => (/[",\n]/.test(String(c)) ? `"${String(c).replace(/"/g, '""')}"` : c)).join(',')).join('\n');
  const a = el('a', { href: URL.createObjectURL(new Blob([csv], { type: 'text/csv' })), download: name });
  document.body.append(a);
  a.click();
  a.remove();
}

// ── Boot ────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const res = await fetch('data.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    DATA = await res.json();
  } catch (e) {
    document.querySelector('.main').replaceChildren(el('div', { class: 'empty' },
      el('p', { text: 'Dashboard data not found.' }),
      el('p', { class: 'mono', text: 'python dashboard/data_export.py && python -m backend.app' })));
    return;
  }
  renderShell();
  renderModels();
  renderSchedule();
  setupControls();
  renderPlan(DATA.simulation);
  route();
  checkApi();
}

function renderShell() {
  const m = DATA.meta;
  $('railRoute').textContent = m.route;
  $('railStops').textContent = `${m.n_route_stops} stops`;
  const w = Object.entries(DATA.weights).sort((a, b) => b[1] - a[1]);
  const names = { xgb_direct: 'XGBoost', stgnn_route: 'STGNN', stgnn_od: 'STGNN-OD', agcrn: 'AGCRN', gru_nograph: 'GRU', stgnn_v3: 'STGNN-v3', how_mean: 'Profile' };
  $('railModel').textContent = `Ensemble · ${w.map(([k]) => names[k] || k).join(' + ')}`;
  const mon = (d) => new Date(`${d}T00:00:00`).toLocaleDateString('en-GB', { month: 'short' });
  $('railTrained').textContent = `Tuned ${mon(m.split.val_start)} · tested ${mon(m.split.test_start)}`;
}

function route() {
  const view = (location.hash || '#plan').slice(1);
  const views = { plan: 'Daily plan', schedule: 'Schedule evaluation', models: 'Model performance' };
  const active = views[view] ? view : 'plan';
  for (const v of Object.keys(views)) $(`view-${v}`).hidden = v !== active;
  document.querySelectorAll('.rail-link').forEach((a) => {
    if (a.dataset.view === active) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  $('crumb').textContent = views[active];
  document.title = `${views[active]} — TransitOps`;
}
window.addEventListener('hashchange', route);

async function checkApi() {
  const pill = $('apiPill');
  try {
    const res = await fetch('/api/health', { cache: 'no-store' });
    const body = await res.json();
    API_OK = res.ok && body.status === 'ok';
  } catch { API_OK = false; }
  pill.replaceChildren(el('span', { class: 'dot', style: { background: API_OK ? '#1D4ED8' : '#8A909C' } }),
    API_OK ? 'Live model' : 'Offline · exported day only');
  pill.className = `pill ${API_OK ? 'pill-ok' : 'pill-off'}`;
  pill.title = API_OK ? 'Plans are recomputed by the API' : 'Start the API (python -m backend.app) to change day, capacity or headway';
  ['daySel', 'capSel', 'hwSel', 'prevDay', 'nextDay'].forEach((id) => { $(id).disabled = !API_OK; });
}

// ── Controls ────────────────────────────────────────────────────────────────
function setupControls() {
  const m = DATA.meta;
  const day = $('daySel');
  m.days.forEach((d, i) => day.append(el('option', { value: i, text: shortDate(d) })));
  [40, 50, 60, 70, 80, 90, 100, 120, 150].forEach((c) => $('capSel').append(el('option', { value: c, text: `${c} passengers` })));
  [5, 6, 10, 12, 15, 20, 30].forEach((h) => $('hwSel').append(el('option', { value: h, text: `every ${h} min` })));
  $('capSel').value = m.defaults.bus_capacity;
  $('hwSel').value = m.defaults.fixed_headway;

  const step = (d) => {
    const i = Math.min(Math.max(Number(day.value) + d, 0), m.days.length - 1);
    if (i !== Number(day.value)) { day.value = i; simulate(); }
  };
  $('prevDay').addEventListener('click', () => step(-1));
  $('nextDay').addEventListener('click', () => step(1));
  ['daySel', 'capSel', 'hwSel'].forEach((id) => $(id).addEventListener('change', simulate));
  $('exportPlan').addEventListener('click', exportPlan);
  $('exportDays').addEventListener('click', exportDays);
}

async function simulate() {
  const req = { day_index: Number($('daySel').value), bus_capacity: Number($('capSel').value), fixed_headway: Number($('hwSel').value) };
  const ticket = ++pending;
  $('view-plan').classList.add('loading');
  try {
    const res = await fetch('/api/simulate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req) });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`);
    if (ticket !== pending) return;
    $('banner').hidden = true;
    renderPlan(body);
  } catch (e) {
    if (ticket !== pending) return;
    $('banner').hidden = false;
    $('banner').textContent = `Could not recompute the plan (${e.message}). Showing ${shortDate(SIM.day)}.`;
    syncControls(SIM);
  } finally {
    if (ticket === pending) $('view-plan').classList.remove('loading');
  }
}

function syncControls(sim) {
  const a = sim.optimization.assumptions;
  $('daySel').value = sim.day_index;
  $('capSel').value = a.bus_capacity;
  $('hwSel').value = a.fixed_headway;
  $('prevDay').disabled = !API_OK || sim.day_index === 0;
  $('nextDay').disabled = !API_OK || sim.day_index === DATA.meta.days.length - 1;
}

// ── Daily plan ──────────────────────────────────────────────────────────────
function renderPlan(sim) {
  if (!sim) return;
  SIM = sim;
  syncControls(sim);
  const m = DATA.meta;
  const o = sim.optimization;
  const a = o.assumptions;
  const f = o.fixed;
  const p = o.optimized;
  const hours = sim.forecast.hours;
  const cap = a.bus_capacity;

  $('dayTitle').textContent = longDate(sim.day);
  $('daySub').textContent = `Route ${m.route}${m.route_is_loop ? ' (loop)' : ''} · forecast issued 00:00 with data to the previous midnight · ${cap} passengers per bus · plan to ${Math.round(a.planning_load_factor * 100)}% of capacity`;

  // KPIs
  const fcTot = sum(sim.forecast.predicted);
  const acTot = sum(sim.forecast.actual);
  const diff = (fcTot - acTot) / acTot * 100;
  const peakIdx = o.peak_load.indexOf(Math.max(...o.peak_load));
  const best = DATA.models[0];
  const vehDiff = (p.vehicle_hours - f.vehicle_hours) / f.vehicle_hours * 100;
  const kpi = (label, value, unit, sub, tone) => el('div', { class: 'kpi' },
    el('div', { class: 'kpi-label', text: label }),
    el('div', { class: 'kpi-value' }, value, unit ? el('small', { text: ` ${unit}` }) : null),
    el('div', { class: `kpi-sub ${tone || ''}`, text: sub }));
  $('kpis').replaceChildren(
    kpi('Forecast boardings', num(fcTot), null, `Actual ${num(acTot)} · ${pct(diff)}`, Math.abs(diff) > 10 ? 'bad' : ''),
    kpi('Day forecast error', `${num(sim.forecast.day_wmape, 1)}%`, null, `WMAPE · test-month average ${num(best.wmape, 1)}%`, sim.forecast.day_wmape > best.wmape * 1.25 ? 'bad' : ''),
    kpi('Peak segment load', num(o.peak_load[peakIdx]), 'pax/h', `Forecast · ${o.peak_segment[peakIdx]} at ${hh(hours[peakIdx])}:00`),
    kpi('Planned departures', num(p.departures_total), null, `Fixed timetable ${num(f.departures_total)} · ${pct(vehDiff)} veh-h`),
    kpi('Hours over capacity', num(p.realized_overcrowded_hours), `of ${hours.length}`, `Fixed timetable: ${f.realized_overcrowded_hours} · on actual demand`, p.realized_overcrowded_hours < f.realized_overcrowded_hours ? 'good' : 'bad'),
  );

  $('fcSub').textContent = `${m.n_forecast_stops} forecast stops · forecast issued 00:00 vs what happened`;
  renderForecastChart(sim);
  renderAlerts(sim);

  $('depSub').textContent = `Sized to the busiest segment · headway ${a.min_headway}–${a.max_headway} min · ${a.round_trip_min}-min loop`;
  $('fixedLegend').textContent = `Fixed (${num(60 / a.fixed_headway, 1).replace('.0', '')}/h)`;
  renderDepartures(sim);

  const row = (label, fv, pv, better) => el('tr', {},
    el('th', { scope: 'row', text: label }),
    el('td', { class: 'num', text: fv, style: better === 'p' ? { color: '#9A3412' } : {} }),
    el('td', { class: 'num', text: pv, style: { fontWeight: 500, color: better === 'p' ? '#1E40AF' : '' } }));
  const b = (x, y) => (y < x ? 'p' : '');
  $('cmpTable').replaceChildren(
    el('thead', {}, el('tr', {}, el('th', { scope: 'col', text: 'Metric' }), el('th', { scope: 'col', class: 'num', text: 'Fixed' }), el('th', { scope: 'col', class: 'num', text: 'Plan', style: { color: '#1D4ED8' } }))),
    el('tbody', {},
      row('Vehicle-hours', num(f.vehicle_hours), num(p.vehicle_hours)),
      row('Operating cost', money(f.cost), money(p.cost)),
      row('Avg wait', `${num(f.realized_avg_wait_min, 1)} min`, `${num(p.realized_avg_wait_min, 1)} min`, b(f.realized_avg_wait_min, p.realized_avg_wait_min)),
      row('Hours over capacity', num(f.realized_overcrowded_hours), num(p.realized_overcrowded_hours), b(f.realized_overcrowded_hours, p.realized_overcrowded_hours)),
      row('Worst bus load', `${num(f.realized_max_utilisation * 100)}%`, `${num(p.realized_max_utilisation * 100)}%`, b(f.realized_max_utilisation, p.realized_max_utilisation))));
  $('cmpNote').textContent = `Assumes a ${a.round_trip_min}-min round trip and ${m.currency}${a.cost_per_vehicle_hour} per vehicle-hour; neither is in the source data.`;

  renderHeatmap(sim);
}

function renderForecastChart(sim) {
  const A = sim.forecast.actual;
  const P = sim.forecast.predicted;
  const H = sim.forecast.hours;
  const W = 860, Ht = 400, x0 = 52, x1 = 844, y0 = 16, y1 = 368;
  const max = niceMax(Math.max(...A, ...P));
  const X = (i) => x0 + i * ((x1 - x0) / (H.length - 1));
  const Y = (v) => y1 - (v / max) * (y1 - y0);
  const s = svg('svg', { viewBox: `0 0 ${W} ${Ht}`, class: 'chart-svg', role: 'img', 'aria-label': 'Forecast and actual boardings per hour' });
  const half = (x1 - x0) / (H.length - 1) / 2;
  A.forEach((a, i) => {
    if (a >= 200 && (a - P[i]) / a > 0.2) s.append(svg('rect', { x: X(i) - half, y: y0 - 4, width: half * 2, height: y1 - y0 + 4, fill: COLORS.warnSoft }));
  });
  for (let k = 0; k <= 4; k++) {
    const v = (max / 4) * k;
    s.append(svg('line', { x1: x0, x2: x1, y1: Y(v), y2: Y(v), stroke: COLORS.grid }));
    s.append(svg('text', { x: x0 - 8, y: Y(v) + 4, 'text-anchor': 'end', text: num(v) }));
  }
  H.forEach((h, i) => { if (i % 3 === 0) s.append(svg('text', { x: X(i), y: Ht - 8, 'text-anchor': 'middle', text: hh(h) })); });
  const pts = (arr) => arr.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  s.append(svg('path', { d: `M${x0},${y1} L${pts(P).replace(/ /g, ' L')} L${x1},${y1} Z`, fill: COLORS.accent, 'fill-opacity': 0.08 }));
  s.append(svg('polyline', { points: pts(A), fill: 'none', stroke: COLORS.ink, 'stroke-width': 2, 'stroke-linejoin': 'round' }));
  s.append(svg('polyline', { points: pts(P), fill: 'none', stroke: COLORS.accent, 'stroke-width': 2.5, 'stroke-linejoin': 'round' }));
  A.forEach((a, i) => {
    const err = a > 0 ? ` (${pct((P[i] - a) / a * 100, 0)})` : '';
    s.append(svg('rect', { x: X(i) - half, y: y0, width: half * 2, height: y1 - y0, fill: 'transparent', tip: `${hh(H[i])}:00 · forecast ${num(P[i])} · actual ${num(a)}${err}` }));
  });
  $('fcChart').replaceChildren(s);
}

function renderAlerts(sim) {
  const A = sim.forecast.actual, P = sim.forecast.predicted, H = sim.forecast.hours;
  const o = sim.optimization, a = o.assumptions, dep = o.optimized.departures;
  const out = [];
  const periodName = (h) => (h < 10 ? 'Morning peak' : h < 16 ? 'Midday' : h < 21 ? 'Evening peak' : 'Late evening');

  // Longest run of hours under-forecast by more than 20%
  const runs = [];
  let cur = null;
  A.forEach((act, i) => {
    const miss = act >= 200 && (act - P[i]) / act > 0.2;
    if (miss) { if (!cur) cur = { s: i, e: i }; else cur.e = i; } else if (cur) { runs.push(cur); cur = null; }
  });
  if (cur) runs.push(cur);
  if (runs.length) {
    const r = runs.map((x) => ({ ...x, gap: sum(A.slice(x.s, x.e + 1)) - sum(P.slice(x.s, x.e + 1)) })).sort((x, y) => y.gap - x.gap)[0];
    const act = sum(A.slice(r.s, r.e + 1)), fc = sum(P.slice(r.s, r.e + 1));
    const seg = o.actual_peak_load.slice(r.s, r.e + 1);
    const k = r.s + seg.indexOf(Math.max(...seg));
    out.push(['warn', ['M12 3l9 16H3z', 'M12 10v4', 'M12 17h.01'],
      `${periodName(H[r.s])} under-forecast by ${num((act - fc) / act * 100)}%`,
      `${hh(H[r.s])}:00–${hh(H[r.e])}:59 actual ${num(act)} vs forecast ${num(fc)} boardings. ${o.actual_peak_segment[k]} carried ${num(o.actual_peak_load[k])} passengers at ${hh(H[k])}:00.`]);
  }

  const over = H.filter((h, i) => o.actual_peak_load[i] > dep[i] * a.bus_capacity + 1e-9);
  const pr = o.optimized.realized_overcrowded_hours, fr = o.fixed.realized_overcrowded_hours;
  if (pr > 0) {
    out.push(['warn', ['M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z', 'M12 7v5l3 2'],
      `${pr} hour${pr > 1 ? 's' : ''} ran over capacity`,
      `${over.map((h) => `${hh(h)}:00`).join(', ')}. Plan sized to ${Math.round(a.planning_load_factor * 100)}% of forecast load; the fixed timetable would have overloaded ${fr}.`]);
  } else {
    out.push(['info', ['M5 12l5 5L20 7'], 'Every hour within capacity',
      `On actual demand the plan never exceeded ${a.bus_capacity} passengers per bus. The fixed timetable would have overloaded ${fr} hour${fr === 1 ? '' : 's'}.`]);
  }

  const maxDep = Math.max(...dep);
  const peakHours = H.filter((h, i) => dep[i] === maxDep);
  out.push(['info', ['M4 3h16v14H4z', 'M4 11h16'],
    `Peak service ${peakHours.map((h) => `${hh(h)}:00`).join(' and ')}`,
    `${num(maxDep)} departures/h (${num(60 / maxDep, 1)} min headway). Needs ${num(Math.ceil(a.round_trip_min / (60 / maxDep)))} buses in service on a ${a.round_trip_min}-min loop.`]);

  const floor = 60 / a.max_headway;
  const minHours = H.filter((h, i) => dep[i] <= floor + 1e-9);
  if (minHours.length) {
    out.push(['', ['M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z'],
      `Minimum service ${minHours.length} hour${minHours.length > 1 ? 's' : ''}`,
      `${minHours.map((h) => hh(h)).join(', ')} h: forecast demand below one bus; runs at the ${a.max_headway}-min headway floor.`]);
  }

  const tone = { warn: '#C2410C', info: '#1D4ED8', '': '#5A6170' };
  $('alerts').replaceChildren(...out.map(([kind, paths, title, body]) => el('div', { class: `alert ${kind}` },
    icon(paths, tone[kind]),
    el('div', {}, el('div', { class: 'alert-title', text: title }), el('div', { class: 'alert-body', text: body })))));
  $('alertCount').textContent = `${out.length} items`;
}

function renderDepartures(sim) {
  const o = sim.optimization, a = o.assumptions, H = sim.forecast.hours;
  const dep = o.optimized.departures, fixed = o.fixed.departures[0];
  const max = Math.max(...dep, fixed) * 1.1;
  const height = 170;
  const floor = 60 / a.max_headway;
  const bars = el('div', { class: 'bars', style: { height: `${height + 22}px` } });
  bars.append(el('div', { class: 'refline', style: { bottom: `${(fixed / max) * height}px` } }));
  dep.forEach((d, i) => {
    const vehicles = Math.ceil(a.round_trip_min / (60 / d) - 1e-9);
    bars.append(el('div', { class: 'col' },
      el('span', { class: 'val', text: num(d) }),
      el('div', {
        class: 'bar',
        tip: `${hh(H[i])}:00 · ${num(d)} departures · every ${num(60 / d, 1)} min · ${vehicles} buses · forecast peak ${num(o.peak_load[i])} pax on ${o.peak_segment[i]}`,
        style: { height: `${(d / max) * height}px`, background: d <= floor + 1e-9 ? COLORS.accent3 : COLORS.accent },
      })));
  });
  $('depChart').replaceChildren(bars, el('div', { class: 'axis', style: { marginTop: '6px' } }, ...H.map((h) => el('span', { text: hh(h) }))));
}

const LOAD_BUCKETS = [
  [0.1, '#EEF2FB', '< 10%'], [0.3, '#C9D6F5', '30%'], [0.5, '#93AEEB', '50%'],
  [0.7, '#5B82DF', '70%'], [0.85, '#1D4ED8', '85%'], [1.0, '#F2A93B', '85–100%'], [Infinity, '#C2410C', 'Over capacity'],
];
function renderHeatmap(sim) {
  const hm = sim.heatmap, o = sim.optimization, cap = o.assumptions.bus_capacity;
  const dep = o.optimized.departures;
  const grid = $('heatmap');
  grid.style.gridTemplateColumns = `minmax(90px, 130px) repeat(${hm.hours.length}, minmax(0, 1fr))`;
  const cells = [];
  hm.segments.forEach((seg, s) => {
    cells.push(el('div', { class: 'lbl', text: seg }));
    hm.actual[s].forEach((load, i) => {
      const u = load / (dep[i] * cap);
      const color = LOAD_BUCKETS.find(([t]) => u <= t)[1];
      cells.push(el('div', {
        class: 'cell',
        style: { background: color, boxShadow: u <= 0.1 ? 'inset 0 0 0 1px #E3E7F2' : '' },
        tip: `${seg} · ${hh(hm.hours[i])}:00 · actual ${num(load)} pax (forecast ${num(hm.data[s][i])}) · ${num(u * 100)}% of ${num(dep[i] * cap)} planned seats`,
      }));
    });
  });
  cells.push(el('div'));
  hm.hours.forEach((h, i) => cells.push(el('div', { class: 'hr', text: i % 2 === 0 ? hh(h) : '' })));
  grid.replaceChildren(...cells);
  $('hmLegend').replaceChildren(...LOAD_BUCKETS.map(([, c, label]) => el('span', {},
    el('i', { class: 'sw', style: { background: c, boxShadow: c === '#EEF2FB' ? 'inset 0 0 0 1px #D6DCEB' : '' } }), label)));
}

function exportPlan() {
  const o = SIM.optimization, H = SIM.forecast.hours, a = o.assumptions;
  const rows = [['date', 'hour', 'forecast_boardings', 'actual_boardings', 'forecast_peak_load', 'peak_segment', 'departures', 'headway_min', 'buses_in_service', 'fixed_departures']];
  H.forEach((h, i) => {
    const d = o.optimized.departures[i];
    rows.push([SIM.day, `${hh(h)}:00`, SIM.forecast.predicted[i], SIM.forecast.actual[i], o.peak_load[i], o.peak_segment[i],
      d, (60 / d).toFixed(1), Math.ceil(a.round_trip_min / (60 / d) - 1e-9), o.fixed.departures[i]]);
  });
  download(`plan_${DATA.meta.route}_${SIM.day}.csv`, rows);
}

// ── Schedule evaluation ─────────────────────────────────────────────────────
function renderSchedule() {
  const S = DATA.schedule, m = DATA.meta, a = S.assumptions;
  const days = S.days;
  const P = S.plans;
  $('schTitle').textContent = `${monthName(days[0].day)} · ${days.length} service days`;
  $('schSub').textContent = `Every plan is scored against the demand that actually happened. ${a.bus_capacity} passengers per bus, ${a.round_trip_min}-min loop, ${m.currency}${a.cost_per_vehicle_hour} per vehicle-hour, fixed timetable every ${a.fixed_headway} min.`;

  const lfPct = Math.round(S.load_factor * 100);
  const card = (cls, title, sub, badge, r) => el('div', { class: `plan ${cls}` },
    el('div', { class: 'card-head' },
      el('div', {}, el('div', { class: 'plan-title', text: title }), el('div', { class: 'sub', text: sub, style: { fontSize: '13px', color: 'var(--muted)', marginTop: '2px' } })),
      badge ? el('span', { class: 'pill pill-ok', text: badge, style: { padding: '4px 10px', fontSize: '12px' } }) : null),
    el('div', { class: 'plan-grid' },
      el('div', {}, el('div', { class: 'k', text: 'Hours over capacity' }), el('div', { class: 'v big', text: num(r.overcrowded_hours_total), style: { color: cls === 'active' ? '#1E40AF' : r.overcrowded_hours_total > 50 ? '#9A3412' : '' } })),
      el('div', {}, el('div', { class: 'k', text: 'Days affected' }), el('div', { class: 'v big', text: `${r.days_with_overcrowding} / ${days.length}` })),
      el('div', {}, el('div', { class: 'k', text: 'Vehicle-hours / day' }), el('div', { class: 'v' }, num(r.vehicle_hours_per_day, 1), r.cost_vs_fixed_pct ? el('span', { text: `  ${pct(r.cost_vs_fixed_pct)}`, style: { fontSize: '13px', color: r.cost_vs_fixed_pct > 0 ? '#9A3412' : 'var(--muted)' } }) : null)),
      el('div', {}, el('div', { class: 'k', text: 'Avg wait' }), el('div', { class: 'v', text: `${num(r.avg_wait_min, 1)} min` }))));
  $('plans').replaceChildren(
    card('', 'Fixed timetable', `Every ${a.fixed_headway} min, all day`, null, P.fixed),
    card('active', 'Forecast-driven plan', `Sized to ${lfPct}% of forecast load`, 'In use', P.forecast),
    card('ref', 'Perfect-foresight plan', 'Reference only · planned on actual demand', null, P.oracle));

  const maxFix = Math.max(...days.map((d) => d.fixed.overcrowded_hours), ...days.map((d) => d.forecast.overcrowded_hours), 1);
  const scale = 176 / maxFix;
  $('days').replaceChildren(...days.map((d) => {
    const label = shortDate(d.day);
    const fo = d.fixed.overcrowded_hours, po = d.forecast.overcrowded_hours;
    return el('div', { class: 'day' },
      el('div', { class: 'up' }, el('span', { text: fo }),
        el('div', { class: 'b', tip: `${label} · fixed timetable: ${fo} h over capacity · wait ${num(d.fixed.avg_wait_min, 1)} min`, style: { height: `${fo * scale}px`, background: COLORS.neutral, borderRadius: '3px 3px 0 0' } })),
      el('div', { class: 'mid' }),
      el('div', { class: 'down' },
        el('div', { class: 'b', tip: `${label} · plan: ${po} h over capacity · ${num(d.forecast.vehicle_hours)} vehicle-hours · wait ${num(d.forecast.avg_wait_min, 1)} min`, style: { height: `${Math.min(po * scale, 40)}px`, background: COLORS.warn, borderRadius: '0 0 3px 3px' } }),
        el('span', { text: po || '', style: { color: '#9A3412' } })),
      el('span', { text: d.day.slice(8).replace(/^0/, '') }));
  }));

  const curve = S.validation_curve || [];
  $('bufSub').textContent = `Validation month (${monthName(m.split.val_start)}) · target ≤ 2% of service hours over capacity`;
  const maxCost = Math.max(...curve.map((r) => r.cost_vs_fixed_pct), 1);
  const maxOc = Math.max(...curve.map((r) => r.overcrowded_hours_total), 1);
  const meter = (w, color) => el('div', { style: { width: `${Math.max(2, w)}px`, height: '8px', borderRadius: '2px', background: color, flexShrink: 0 } });
  $('bufTable').replaceChildren(
    el('thead', {}, el('tr', {}, el('th', { scope: 'col', text: 'Plan to' }), el('th', { scope: 'col', text: 'Extra vehicle-hours' }), el('th', { scope: 'col', text: 'Hours over capacity' }))),
    el('tbody', {}, ...curve.map((r) => el('tr', { class: r.load_factor === S.load_factor ? 'hl' : '' },
      el('th', { scope: 'row', class: 'mono', text: `${Math.round(r.load_factor * 100)}%` }),
      el('td', {}, el('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } }, meter(Math.max(r.cost_vs_fixed_pct, 0) / maxCost * 80, COLORS.accent), el('span', { class: 'mono', text: pct(r.cost_vs_fixed_pct) }))),
      el('td', {}, el('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } }, meter(r.overcrowded_hours_total / maxOc * 80, COLORS.warn), el('span', { class: 'mono', text: num(r.overcrowded_hours_total) })))))));
  $('bufNote').textContent = `${lfPct}% is the cheapest setting that met the target in ${monthName(m.split.val_start)}; the ${monthName(days[0].day)} results above use it unchanged.`;

  const f = P.fixed, fc = P.forecast, or = P.oracle;
  $('takeaway').replaceChildren(
    el('strong', { text: or.overcrowded_hours_total === 0 && Math.abs(or.cost_vs_fixed_pct) < 3
      ? 'The fixed timetable is not too big. It runs buses in the wrong hours.'
      : 'Forecast-driven frequencies remove most overcrowding.' }),
    el('div', { text: `With perfect foresight, ${num(or.vehicle_hours_per_day, 1)} vehicle-hours a day (fixed: ${num(f.vehicle_hours_per_day, 1)}) would leave ${num(or.overcrowded_hours_total)} hours over capacity. `
      + `The forecast plan spends ${pct(fc.cost_vs_fixed_pct)} vehicle-hours to cover forecast error, and cuts overcrowding from ${num(f.overcrowded_hours_total)} hours to ${num(fc.overcrowded_hours_total)} and average wait from ${num(f.avg_wait_min, 1)} to ${num(fc.avg_wait_min, 1)} minutes.`
      + (S.unbuffered ? ` Without the ${100 - lfPct}% buffer it would cost ${pct(S.unbuffered.cost_vs_fixed_pct)} but leave ${num(S.unbuffered.overcrowded_hours_total)} hours over capacity.` : '') }));
}

function exportDays() {
  const rows = [['date', 'plan', 'vehicle_hours', 'overcrowded_hours', 'avg_wait_min']];
  DATA.schedule.days.forEach((d) => ['fixed', 'forecast', 'oracle'].forEach((p) => rows.push([d.day, p, d[p].vehicle_hours, d[p].overcrowded_hours, d[p].avg_wait_min])));
  download(`schedule_evaluation_${DATA.meta.route}.csv`, rows);
}

// ── Models ──────────────────────────────────────────────────────────────────
function renderModels() {
  const m = DATA.meta, models = DATA.models;
  $('mdSub').textContent = `WMAPE = total absolute error ÷ total boardings. Test period ${m.test_period[0].slice(0, 10)} → ${m.test_period[1].slice(0, 10)}, ${num(m.samples.test)} hourly forecasts of the next 24 h at ${m.n_forecast_stops} stops; learned models averaged over ${m.seeds.length} seeds.`;
  $('spTrain').textContent = 'Train · Oct–Jan';
  $('spVal').textContent = `Validate · ${new Date(`${m.split.val_start}T00:00:00`).toLocaleDateString('en-GB', { month: 'short' })}`;
  $('spTest').textContent = `Test · ${new Date(`${m.split.test_start}T00:00:00`).toLocaleDateString('en-GB', { month: 'short' })}`;

  const maxW = Math.max(...models.map((x) => x.wmape));
  const base = models.find((x) => x.id === 'how_mean');
  $('lbTable').replaceChildren(
    el('thead', {}, el('tr', {},
      el('th', { scope: 'col', text: 'Model' }), el('th', { scope: 'col', text: 'WMAPE', style: { width: '44%' } }),
      el('th', { scope: 'col', class: 'num', text: 'MAE' }), el('th', { scope: 'col', class: 'num', text: 'RMSE' }),
      el('th', { scope: 'col', class: 'num', text: 'MAPE*', title: `MAPE over stop-hours with at least ${m.mape_label.match(/[\d.]+/)[0]} boardings` }))),
    el('tbody', {}, ...models.map((x, i) => {
      const bestRow = i === 0;
      const color = bestRow ? COLORS.accent : base && x.wmape < base.wmape ? COLORS.accent2 : COLORS.neutral;
      return el('tr', { class: bestRow ? 'hl' : '' },
        el('th', { scope: 'row' }, el('div', { style: { fontWeight: bestRow ? 600 : 500 }, text: x.name }), el('div', { class: 'muted', style: { fontSize: '12px' }, text: x.kind })),
        el('td', { style: { paddingRight: '16px' } }, el('div', { style: { display: 'flex', alignItems: 'center', gap: '10px' } },
          el('div', { class: 'meter', style: { flex: '1 1 auto' } },
            el('div', { style: { width: `${x.wmape / maxW * 100}%`, background: color } }),
            base ? el('div', { class: 'base', style: { left: `${base.wmape / maxW * 100}%` } }) : null),
          el('span', { class: 'mono', style: { fontSize: '13px', whiteSpace: 'nowrap', minWidth: '104px' }, text: `${x.wmape_text}%` }))),
        el('td', { class: 'num', text: num(x.mae, 2) }),
        el('td', { class: 'num', text: num(x.rmse, 2) }),
        el('td', { class: 'num', text: `${num(x.mape, 1)}%` }));
    })));

  const ab = DATA.ablation || [];
  if (ab.length) {
    const lo = Math.min(...ab.map((r) => r.wmape_mean)) - 1.5, hi = Math.max(...ab.map((r) => r.wmape_mean)) + 0.5;
    const full = ab[0].wmape_mean;
    $('ablation').replaceChildren(...ab.map((r) => {
      const worse = r.wmape_mean - full > 0.5;
      const name = r.variant === 'all features' ? 'All inputs' : r.variant.replace(/^no /, 'Without ').replace('hour-of-week profile', 'weekly profile');
      return el('div', { style: { display: 'flex', flexDirection: 'column', gap: '5px' } },
        el('div', { style: { display: 'flex', justifyContent: 'space-between', fontSize: '13px' } },
          el('span', { text: name }), el('span', { class: 'mono', text: `${r.test_WMAPE}%`, style: { color: worse ? '#9A3412' : '', fontWeight: worse ? 500 : 400 } })),
        el('div', { class: 'meter', style: { height: '8px' } }, el('div', { style: { width: `${(r.wmape_mean - lo) / (hi - lo) * 100}%`, background: worse ? COLORS.warn : COLORS.accent2 } })));
    }));
    const noise = ab.filter((r) => Math.abs(r.wmape_mean - full) < 0.2).map((r) => r.variant.replace(/^no /, '')).filter((v) => v !== 'all features');
    $('ablationNote').textContent = noise.length
      ? `Removing ${noise.join(' or ')} changes the error by less than 0.2 points, within seed noise.`
      : '';
  }

  const st = DATA.errors.by_stop;
  const maxS = niceMax(Math.max(...st.map((s) => s.wmape)));
  const sb = el('div', { class: 'bars', style: { height: '170px' } });
  st.forEach((s) => sb.append(el('div', { class: 'col' },
    el('span', { class: 'val', text: num(s.wmape) }),
    el('div', { class: 'bar', tip: `${s.stop} · WMAPE ${num(s.wmape, 1)}%`, style: { height: `${s.wmape / maxS * 145}px`, background: s.wmape > 35 ? COLORS.warn : COLORS.accent2 } }))));
  $('stopChart').replaceChildren(sb, el('div', { class: 'axis', style: { marginTop: '6px' } }, ...st.map((s) => el('span', { text: s.stop.replace('_', ''), style: { fontSize: '10px' } }))));

  const hr = DATA.errors.by_hour.filter((h) => h.hour >= 5 && h.wmape != null);
  const maxH = niceMax(Math.max(...hr.map((h) => h.wmape)));
  const hb = el('div', { class: 'bars', style: { height: '170px', gap: '3px' } });
  hr.forEach((h) => hb.append(el('div', { class: 'col' },
    el('div', { class: 'bar', tip: `${hh(h.hour)}:00 · WMAPE ${num(h.wmape, 1)}%`, style: { height: `${h.wmape / maxH * 160}px`, background: COLORS.accent2 } }))));
  $('hourChart').replaceChildren(hb, el('div', { class: 'axis', style: { marginTop: '6px', gap: '3px' } }, ...hr.map((h) => el('span', { text: h.hour % 4 === 1 || h.hour === 23 ? hh(h.hour) : '', style: { fontSize: '10px', overflow: 'visible' } }))));
}

// ── Tooltip ─────────────────────────────────────────────────────────────────
(() => {
  const tip = $('tip');
  let target = null;
  document.addEventListener('mouseover', (e) => {
    const t = e.target.closest('[data-tip]');
    if (t === target) return;
    target = t;
    if (!t) { tip.classList.remove('show'); return; }
    tip.textContent = t.dataset.tip;
    tip.classList.add('show');
  });
  document.addEventListener('mousemove', (e) => {
    if (!target) return;
    const r = tip.getBoundingClientRect();
    const x = Math.min(e.clientX + 14, window.innerWidth - r.width - 8);
    const y = e.clientY + 18 + r.height > window.innerHeight ? e.clientY - r.height - 12 : e.clientY + 18;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  });
})();

window.addEventListener('DOMContentLoaded', init);
