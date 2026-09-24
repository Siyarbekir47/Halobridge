'use strict';
const $ = id => document.getElementById(id);
let L = {}, NUM = 'en-US', currentLang = 'en', localeSequence = 0, localeLoading = false;
let liveData, currentView = 'overview';
const t = (key, params) => {
  let s = L[key];
  if (typeof s !== 'string') return key;
  if (params) for (const [k, v] of Object.entries(params)) s = s.replaceAll(`{${k}}`, String(v));
  return s;
};
const valid = n => typeof n === 'number' && Number.isFinite(n);
const integer = n => valid(n) ? n.toLocaleString(NUM) : '–';
const decimal = (n, digits = 1) => valid(n) ? n.toLocaleString(NUM, {minimumFractionDigits: digits, maximumFractionDigits: digits}) : '–';
const percent = n => valid(n) ? `${decimal(n * 100)} %` : '–';
const seconds = n => valid(n) ? `${decimal(n / 1000, 2)} s` : '–';
const rate = n => valid(n) ? `${decimal(n)} ${t('tokens_per_second')}` : '–';
const esc = s => String(s ?? '–').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const dateTime = n => new Date(n * 1000).toLocaleString(NUM, {day:'2-digit', month:'2-digit', year:'numeric', hour:'2-digit', minute:'2-digit'});
const compact = n => n.toLocaleString(NUM, {notation:'compact', maximumFractionDigits:1});
function bytes(n) {
  if (!valid(n)) return '–';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${decimal(n, 1)} ${units[i]}`;
}
const state = {period:'24h', from:null, to:null, page:1, pages:1, range:null, pinned:null};
let analyticsController, analyticsSequence = 0, liveBusy = false, liveTimer, analyticsTimer, currentAnalytics;
let updatesData, updatesTimer, updatesBusy = false, updateActionBusy = false, updateSequence = 0;
const expanded = new Set();
const clientLabel = value => ['Unknown', 'Unbekannt'].includes(value) ? t('unknown_client') : value;

async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  const language = currentLang;
  headers.set('Accept-Language', language);
  const response = await fetch(url, {...options, headers});
  if (language !== currentLang && (!options.method || options.method === 'GET')) {
    return apiFetch(url, options);
  }
  return response;
}

function showView(focus = false) {
  const requested = location.hash.slice(1);
  currentView = ['overview', 'requests', 'models', 'system'].includes(requested) ? requested : 'overview';
  for (const name of ['overview', 'requests', 'models', 'system']) $('view-' + name).hidden = name !== currentView;
  for (const link of document.querySelectorAll('[data-view]')) {
    if (link.dataset.view === currentView) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  $('viewTitle').textContent = t('nav_' + currentView);
  $('viewDescription').textContent = t('intro_' + currentView);
  document.title = `Halobridge · ${t('nav_' + currentView)}`;
  $('analytics').hidden = currentView === 'models';
  if (focus) $('viewTitle').focus({preventScroll: true});
  if (currentView === 'overview' && currentAnalytics) renderChart(currentAnalytics);
  if (currentView === 'models' && focus) { refreshDeploy(); startJobPoll(); }
}
window.addEventListener('hashchange', () => showView(true));
document.querySelector('.skip-link').addEventListener('click', event => {
  event.preventDefault();
  $('mainContent').focus();
});
function modelTab(assets) {
  $('deployError').hidden = true;
  resetDeployArm();
  $('profilesPanel').hidden = assets;
  $('assetsPanel').hidden = !assets;
  $('profilesTab').setAttribute('aria-pressed', String(!assets));
  $('assetsTab').setAttribute('aria-pressed', String(assets));
}
$('profilesTab').addEventListener('click', () => modelTab(false));
$('assetsTab').addEventListener('click', () => modelTab(true));
$('quickModel').addEventListener('change', () => {
  const uncensored = $('quickModel').value === 'uncensored';
  $('quickTokenField').hidden = !uncensored;
  $('quickHfAccessHint').hidden = !uncensored;
  $('quickOfficialBtn').hidden = uncensored;
  $('quickUncensoredBtn').hidden = !uncensored;
  resetDeployArm();
});
$('deployRefresh').addEventListener('click', () => { refreshDeploy(); startJobPoll(); });

function handleUnauthorized(response) {
  if (response.status === 401) {
    location.assign(`/dashboard/login?lang=${currentLang}`);
    return true;
  }
  return false;
}

function renderLive(data) {
  liveData = data;
  const api = data.api || {}, backend = data.backend || {}, system = data.system || {}, app = data.app || {};
  $('appVersion').textContent = app.version ? `v${app.version}` : 'v0+local';
  const online = backend.status === 'ok';
  const switching = api.status === 'switching';
  const maintenance = api.status === 'maintenance';
  $('connection').textContent = maintenance ? t('status_maintenance') : switching ? t('status_switching') : online ? t('status_ready') : t('status_backend_down');
  $('statusDot').dataset.state = maintenance || switching ? 'switching' : online ? 'ok' : 'error';
  $('activeModel').textContent = api.active_model || t('no_active_model');
  $('switchState').hidden = !api.switch_target;
  $('switchState').textContent = api.switch_target ? t('switch_to', {target: api.switch_target}) : '';
  $('activeCount').textContent = integer(backend.in_flight);
  $('queueCount').textContent = integer(backend.queued);
  const active = data.active_requests || [];
  $('activeRequests').hidden = active.length === 0;
  $('activeRequests').innerHTML = active.map(r => `<div><span>${esc(clientLabel(r.client))} · ${esc(r.model)}</span><span>${seconds(r.elapsed_ms)}</span></div>`).join('');
  const memory = system.memory || {};
  $('ram').textContent = `${bytes(memory.used_bytes)} / ${bytes(memory.total_bytes)}`;
  $('gpu').textContent = valid(system.gpu_busy_percent) ? `${integer(system.gpu_busy_percent)} %` : '–';
  $('pool').textContent = percent(data.cache?.pool?.usage_ratio);
  $('disk').textContent = `${bytes(system.disk_used_bytes)} / ${bytes(system.disk_total_bytes)}`;
  $('context').textContent = `${integer(backend.context)} / ${integer(backend.slots)}`;
  $('maxTokens').textContent = `${integer(backend.max_tokens_default)} / ${integer(backend.max_tokens_cap)}`;
  $('reasoning').textContent = backend.reasoning_effort_default || '–';
  $('version').textContent = typeof backend.version === 'string' ? backend.version : backend.version?.api || '–';
  $('storage').textContent = Object.entries(data.cache?.model_bytes || {}).map(([name, size]) => `${name}: ${bytes(size)} Cache`).join(' · ');
  $('updated').textContent = t('live_updated', {time: new Date(data.generated_at * 1000).toLocaleTimeString(NUM)});
}

const bucketLabel = (ts, unit, period) => {
  const d = new Date(ts * 1000);
  if (period === '7d') return d.toLocaleDateString(NUM, {weekday: 'short'});
  if (unit === 'month') return d.toLocaleDateString(NUM, {month: 'short'});
  if (unit === 'day' || unit === 'week') return String(d.getDate()).padStart(2, '0');
  return String(d.getHours()).padStart(2, '0');
};

function renderChart(data) {
  const timeline = data.timeline || [];
  const unit = data.bucket_unit || 'hour';
  const period = data.period || '';
  const hasUsage = data.summary.input_reported > 0 || data.summary.output_reported > 0;
  $('tokenChart').toggleAttribute('hidden', !hasUsage);
  $('chartEmpty').hidden = hasUsage;
  $('chartEmpty').textContent = data.summary.requests ? t('chart_no_verified') : t('chart_no_requests');
  $('chartNote').textContent = t('chart_note', {interval: t(`bucket_${unit}`)});
  if (!hasUsage) { $('tokenChart').innerHTML = ''; return; }
  const max = Math.max(1, ...timeline.flatMap(x => [x.input_tokens || 0, x.output_tokens || 0]));
  const chartWidth = Math.max(260, $('tokenChart').clientWidth || 1080);
  const chartHeight = $('tokenChart').clientHeight || 200;
  $('tokenChart').setAttribute('viewBox', `0 0 ${chartWidth} ${chartHeight}`);
  const left = 52, width = chartWidth - 60, top = 12, axisY = chartHeight - 30;
  const height = axisY - top;
  let svg = `<title>${esc(t('chart_svg_title'))}</title>`;
  for (const ratio of [0, .5, 1]) {
    const y = top + height * (1 - ratio);
    svg += `<path class="gridline" d="M${left} ${y}H${left + width}"/><text x="${left - 10}" y="${y + 4}" text-anchor="end">${esc(compact(max * ratio))}</text>`;
  }
  svg += `<path class="axis" d="M${left} ${axisY}H${left + width}"/>`;
  const step = width / Math.max(1, timeline.length);
  const minLabelPx = unit === 'month' ? 36 : period === '7d' ? 34 : 22;
  const labelEvery = Math.max(1, Math.ceil(minLabelPx / step));
  timeline.forEach((item, i) => {
    const bucketFrom = Math.max(item.bucket_start, data.from);
    const bucketTo = Math.min(item.bucket_end ?? item.bucket_start + data.bucket_seconds, data.to);
    const x = left + (bucketFrom - data.from) / (data.to - data.from) * width;
    const w = (bucketTo - bucketFrom) / (data.to - data.from) * width;
    if (w <= 0) return;
    const label = t('chart_tooltip', {from: dateTime(Math.max(item.bucket_start, data.from)), to: dateTime(Math.min(item.bucket_end ?? item.bucket_start + data.bucket_seconds, data.to)), input: integer(item.input_tokens), output: integer(item.output_tokens), requests: integer(item.requests)}) + (item.partial ? t('chart_tooltip_partial') : '');
    svg += `<g class="bucket"><title>${esc(label)}</title>`;
    for (const [j, key, color] of [[0, 'input_tokens', 'var(--blue)'], [1, 'output_tokens', 'var(--mint)']]) {
      if (!valid(item[key])) continue;
      const h = item[key] / max * height;
      svg += `<rect class="bar" x="${x + w * (.1 + j * .42)}" y="${axisY - h}" width="${Math.max(.5, w * .36)}" height="${h}" rx="2" fill="${color}"/>`;
    }
    if (item.partial) svg += `<circle cx="${x + w / 2}" cy="${axisY + 7}" r="2" fill="var(--amber)"/>`;
    if (i % labelEvery === 0) {
      svg += `<text class="tick" x="${x + w / 2}" y="${chartHeight - 9}" text-anchor="middle">${esc(bucketLabel(item.bucket_start, unit, period))}</text>`;
    }
    svg += `<rect x="${x}" y="${top}" width="${w}" height="${height}" fill="transparent"/></g>`;
  });
  $('tokenChart').innerHTML = svg;
}

function renderHistory(history) {
  state.page = history.page;
  state.pages = history.pages;
  $('historyCount').textContent = t('history_count', {total: integer(history.total)});
  $('pageInfo').textContent = t('page_of', {page: history.page, pages: history.pages}) + (state.pinned ? t('pinned_suffix') : '');
  $('prevPage').disabled = state.page <= 1;
  $('nextPage').disabled = state.page >= state.pages;
  const items = history.items || [];
  $('rows').innerHTML = items.length ? items.map((r, i) => {
    const legacy = r.telemetry_version < 2;
    const count = n => `${integer(n)}${legacy && valid(n) ? '*' : ''}`;
    const open = expanded.has(r.request_id);
    const details = [
      [t('detail_cache'), count(r.cached_tokens)], [t('detail_reasoning'), count(r.reasoning_tokens)],
      [t('detail_ttfb'), seconds(r.ttfb_ms)], [t('detail_generation'), legacy ? '–' : rate(r.tps)],
      [t('detail_reasoning_requested'), r.reasoning_effort], [t('detail_response_format'), r.stream ? 'Streaming' : 'JSON'],
      [t('detail_endpoint'), r.endpoint], [t('detail_capture'), legacy ? t('capture_legacy') : t('capture_api')],
    ];
    return `<tr><td><button class="row-toggle" data-request="${esc(r.request_id)}" aria-expanded="${open}" aria-controls="detail-${i}"><span aria-hidden="true">${open ? '−' : '+'}</span>${esc(dateTime(r.completed_at))}</button></td>
      <td>${esc(clientLabel(r.client))}<span class="cell-sub">${esc(r.model)}</span></td><td class="number ${legacy ? 'legacy' : ''}">${count(r.input_tokens)}</td><td class="number ${legacy ? 'legacy' : ''}">${count(r.output_tokens)}</td>
      <td class="number">${seconds(r.duration_ms)}</td><td class="number ${r.status >= 400 ? 'failure' : 'success'}">${integer(r.status)}<span class="cell-sub">${r.status === 499 ? t('status_aborted') : r.status >= 400 ? t('status_error') : t('status_ok')}</span></td></tr>
      <tr class="request-detail" id="detail-${i}" ${open ? '' : 'hidden'}><td colspan="6"><dl class="detail-grid">${details.map(([key, value]) => `<div><dt>${esc(key)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>${r.error ? `<p class="failure">${esc(r.error)}</p>` : ''}${legacy ? `<small class="legacy">${esc(t('legacy_note'))}</small>` : ''}</td></tr>`;
  }).join('') : `<tr><td colspan="6" class="empty">${esc(t('history_empty'))}</td></tr>`;
}

function renderBreakdown(id, rows) {
  $(id).innerHTML = rows.length ? rows.map(r => `<div class="breakdown-row"><span>${esc(id === 'clients' ? clientLabel(r.name) : r.name)}</span><span>${integer(r.requests)} / ${integer(r.output_tokens)}${r.output_reported < r.requests ? esc(t('partial_suffix')) : ''}</span></div>`).join('') : `<p class="muted">${esc(t('no_requests'))}</p>`;
}

function renderAnalytics(data) {
  currentAnalytics = data;
  const s = data.summary, e = data.engine;
  state.range = {from:data.from, to:data.to};
  $('rangeCaption').textContent = t('range_caption', {from: dateTime(data.from), to: dateTime(data.to), time: new Date(data.generated_at * 1000).toLocaleTimeString(NUM)});
  $('requestCount').textContent = integer(s.requests);
  $('requestNote').textContent = t('request_note', {success: integer(s.successes), errors: integer(s.errors)});
  $('inputTokens').textContent = integer(s.input_tokens);
  $('outputTokens').textContent = integer(s.output_tokens);
  $('inputNote').textContent = t('input_note', {reported: integer(s.input_reported), total: integer(s.requests)});
  $('outputNote').textContent = t('output_note', {reported: integer(s.output_reported), total: integer(s.requests)});
  $('cacheRatio').textContent = percent(s.cache_ratio);
  $('cacheNote').textContent = t('cache_note', {cached: integer(s.cached_tokens), reported: integer(s.cache_reported), total: integer(s.requests)});
  const missing = s.requests - s.usage_reported;
  const messages = [];
  if (missing > 0) messages.push(t('coverage_missing', {reported: integer(s.usage_reported), total: integer(s.requests)}));
  if (s.legacy_requests > 0) messages.push(t('coverage_legacy', {count: integer(s.legacy_requests)}));
  $('coverageNotice').hidden = messages.length === 0;
  $('coverageNotice').textContent = messages.join(' ');
  renderChart(data);
  renderHistory(data.history);
  renderBreakdown('clients', data.by_client);
  renderBreakdown('models', data.by_model);
  $('engineCaption').textContent = t('engine_caption', {count: integer(e.requests)});
  $('engineTps').textContent = rate(e.weighted_tps);
  $('prefillTps').textContent = rate(e.weighted_prefill_tps);
  $('mtp').textContent = decimal(e.avg_commit_per_round, 2);
  $('pld').textContent = decimal(e.avg_pld_accept_per_round, 2);
}

function analyticsUrl() {
  const range = state.pinned || (state.period === 'custom' ? {from:state.from, to:state.to} : null);
  const params = new URLSearchParams({period:range ? 'custom' : state.period, page:String(state.page)});
  if (range) { params.set('from', range.from); params.set('to', range.to); }
  return `/dashboard/api/analytics?${params}`;
}

async function refreshLive() {
  if (liveBusy || document.hidden) return;
  clearTimeout(liveTimer);
  liveBusy = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await apiFetch('/dashboard/api/snapshot', {cache:'no-store', signal:controller.signal});
    if (handleUnauthorized(response)) return;
    if (!response.ok) throw new Error();
    renderLive(await response.json());
  } catch {
    $('connection').textContent = t('conn_broken');
    $('statusDot').dataset.state = 'error';
    $('updated').textContent = t('live_stale');
  } finally {
    clearTimeout(timeout);
    liveBusy = false;
    if (!document.hidden) liveTimer = setTimeout(refreshLive, 5000);
  }
}

async function refreshAnalytics(clear = false) {
  clearTimeout(analyticsTimer);
  analyticsController?.abort();
  const sequence = ++analyticsSequence;
  const controller = analyticsController = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  $('analytics').setAttribute('aria-busy', 'true');
  $('prevPage').disabled = $('nextPage').disabled = true;
  if (clear) {
    currentAnalytics = null;
    for (const id of ['requestCount', 'inputTokens', 'outputTokens', 'cacheRatio']) $(id).textContent = '–';
    for (const id of ['requestNote', 'inputNote', 'outputNote', 'cacheNote']) $(id).textContent = t('loading');
    $('rangeCaption').textContent = t('range_loading');
    $('coverageNotice').hidden = true;
    $('tokenChart').setAttribute('hidden', '');
    $('chartEmpty').hidden = false;
    $('chartEmpty').textContent = t('loading');
    $('rows').innerHTML = `<tr><td colspan="6" class="empty">${esc(t('requests_loading'))}</td></tr>`;
  }
  try {
    const response = await apiFetch(analyticsUrl(), {cache:'no-store', signal:controller.signal});
    if (handleUnauthorized(response)) return;
    if (!response.ok) throw new Error();
    const data = await response.json();
    if (sequence !== analyticsSequence) return;
    renderAnalytics(data);
    $('analyticsError').hidden = true;
  } catch {
    if (sequence !== analyticsSequence) return;
    $('analyticsError').hidden = false;
    $('analyticsError').textContent = t('analytics_error');
    if (clear) {
      $('chartEmpty').textContent = t('analytics_unavailable');
      $('rows').innerHTML = `<tr><td colspan="6" class="empty">${esc(t('requests_failed'))}</td></tr>`;
    }
    $('prevPage').disabled = state.page <= 1;
    $('nextPage').disabled = state.page >= state.pages;
  } finally {
    clearTimeout(timeout);
    if (sequence === analyticsSequence) {
      $('analytics').setAttribute('aria-busy', 'false');
      if (!document.hidden) analyticsTimer = setTimeout(() => refreshAnalytics(), 30000);
    }
  }
}

function changeRange(period, from = null, to = null) {
  Object.assign(state, {period, from, to, page:1, pages:1, pinned:null});
  expanded.clear();
  refreshAnalytics(true);
}
$('period').addEventListener('change', event => {
  $('customRange').hidden = event.target.value !== 'custom';
  if (event.target.value !== 'custom') changeRange(event.target.value);
});
$('customRange').addEventListener('submit', event => {
  event.preventDefault();
  const from = new Date($('fromDate').value).getTime() / 1000;
  const to = new Date($('toDate').value).getTime() / 1000;
  if (!Number.isFinite(from) || !Number.isFinite(to) || from >= to || from >= Date.now() / 1000) {
    $('rangeHint').textContent = t('range_invalid');
    return;
  }
  $('rangeHint').textContent = t('range_applied');
  changeRange('custom', from, to);
});
function changePage(delta) {
  if (!state.range) return;
  state.page = Math.max(1, Math.min(state.pages, state.page + delta));
  state.pinned = state.page > 1 ? state.pinned || {...state.range} : null;
  expanded.clear();
  refreshAnalytics();
}
$('prevPage').addEventListener('click', () => changePage(-1));
$('nextPage').addEventListener('click', () => changePage(1));
$('rows').addEventListener('click', event => {
  const button = event.target.closest('button[data-request]');
  if (!button) return;
  const open = button.getAttribute('aria-expanded') !== 'true';
  button.setAttribute('aria-expanded', String(open));
  button.querySelector('span').textContent = open ? '−' : '+';
  $(button.getAttribute('aria-controls')).hidden = !open;
  if (open) expanded.add(button.dataset.request); else expanded.delete(button.dataset.request);
});
$('refresh').addEventListener('click', () => {
  if (!Object.keys(L).length) { switchLanguage(currentLang); return; }
  refreshLive(); refreshAnalytics();
});
$('quickHfToken').addEventListener('input', resetDeployArm);
$('endpoint').textContent = `${location.origin}/v1`;
$('copy').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('endpoint').textContent);
    $('copyStatus').textContent = t('copied');
  } catch {
    const selection = window.getSelection(), range = document.createRange();
    range.selectNodeContents($('endpoint'));
    selection.removeAllRanges(); selection.addRange(range);
    $('copyStatus').textContent = t('copy_hint');
  }
});
function localInput(date) {
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}
function renderUpdates(data) {
  const attention = data.running || data.recovery_required || data.update_available;
  $('systemAttention').hidden = !attention;
  $('operationNotice').hidden = !attention;
  $('operationText').textContent = data.recovery_required ? t('update_recovery_notice') : data.running ? t('update_running_notice') : t('update_available_notice');
  updatesData = data;
  const job = data.job;
  let title = data.current_version ? t('update_server_title', {version: data.current_version}) : t('update_default_title');
  if (data.recovery_required) title += t('update_suffix_recovery');
  else if (data.running) title += t('update_suffix_running');
  else if (data.update_available) title += t('update_suffix_available', {version: data.latest_version});
  else if (data.up_to_date && !data.check_error) title += t('update_suffix_current');
  if ($('updateTitle').textContent !== title) $('updateTitle').textContent = title;
  $('updateMessage').textContent = data.running || data.recovery_required || job?.phase === 'failed' || job?.phase === 'rolled_back' || job?.phase === 'succeeded'
    ? job?.message || t('update_msg_recovery')
    : data.check_error || data.support_error || data.blocked_reason || t('update_msg_default');
  $('updateVersions').textContent = Object.entries(data.configured_versions || {}).map(([model, version]) => `${model}: ${version}`).join(' · ') || t('update_no_quadlets');
  $('updateChecked').textContent = data.checked_at ? t('update_checked_at', {time: dateTime(data.checked_at), tag: data.latest_version}) : t('not_checked');
  $('updateBackup').hidden = !job?.backup_dir;
  $('updateBackup').textContent = job?.backup_dir ? t('update_backup', {dir: job.backup_dir}) : '';
  $('updateChangelog').hidden = !data.release_url;
  if (data.release_url) $('updateChangelog').href = data.release_url;
  $('checkUpdate').disabled = updateActionBusy || data.running || data.recovery_required;
  $('installUpdate').hidden = !data.update_available || data.recovery_required;
  $('installUpdate').disabled = updateActionBusy || !data.can_install;
  $('installUpdate').textContent = data.running ? t('update_running_btn') : t('update_install_btn', {version: data.latest_version});
  $('recoverUpdate').hidden = !data.recovery_required;
  $('recoverUpdate').disabled = updateActionBusy || !data.can_recover;
}
function scheduleUpdates() {
  clearTimeout(updatesTimer);
  if (!document.hidden) updatesTimer = setTimeout(refreshUpdates, updatesData?.running ? 3000 : updatesData?.checked_at ? 60000 : 10000);
}
async function refreshUpdates() {
  if (updatesBusy || updateActionBusy || document.hidden) return;
  updatesBusy = true;
  const sequence = ++updateSequence;
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await apiFetch('/dashboard/api/updates', {cache:'no-store', signal:controller.signal});
    if (handleUnauthorized(response)) return;
    if (!response.ok) throw new Error(t('update_unavailable'));
    const data = await response.json();
    if (sequence === updateSequence) { renderUpdates(data); $('updateError').hidden = true; }
  } catch (error) {
    if (sequence === updateSequence) {
      $('updateError').hidden = false;
      $('updateError').textContent = error.name === 'AbortError' ? t('update_timeout') : error.message;
      $('installUpdate').disabled = true;
      $('recoverUpdate').disabled = true;
    }
  } finally { clearTimeout(timeout); updatesBusy = false; scheduleUpdates(); }
}
async function updateAction(action) {
  if (updateActionBusy) return;
  updateActionBusy = true;
  setLanguageAvailability();
  ++updateSequence;
  clearTimeout(updatesTimer);
  for (const id of ['checkUpdate', 'installUpdate', 'recoverUpdate']) $(id).disabled = true;
  $('updateError').hidden = true;
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 180000);
  try {
    const response = await apiFetch(`/dashboard/api/updates/${action}`, {
      method:'POST', headers:{'Content-Type':'application/json', 'X-Halogen-Action':'update'},
      body:JSON.stringify(action === 'install' ? {version:updatesData?.latest_version} : {}), signal:controller.signal,
    });
    if (handleUnauthorized(response)) return;
    if (!response.ok) throw new Error((await response.text()) || t('update_action_failed'));
    updatesData = await response.json();
    refreshLive();
  } catch (error) {
    $('updateError').hidden = false;
    $('updateError').textContent = error.name === 'AbortError' ? t('update_action_timeout') : error.message;
  } finally {
    clearTimeout(timeout);
    updateActionBusy = false;
    setLanguageAvailability();
    if (updatesData) renderUpdates(updatesData); else $('checkUpdate').disabled = false;
    scheduleUpdates();
  }
}
$('checkUpdate').addEventListener('click', () => updateAction('check'));
$('installUpdate').addEventListener('click', () => updateAction('install'));
$('recoverUpdate').addEventListener('click', () => updateAction('recover'));
let resetArmed = false, resetBusy = false, resetRevertTimer;
function setResetUi() {
  setLanguageAvailability();
  $('resetDb').hidden = resetArmed;
  $('resetConfirm').hidden = !resetArmed;
  $('resetCancel').hidden = !resetArmed || resetBusy;
  $('resetConfirm').disabled = resetBusy;
  $('resetConfirm').textContent = resetBusy ? t('reset_running') : t('reset_confirm');
}
function disarmReset() {
  resetArmed = false;
  clearTimeout(resetRevertTimer);
  setResetUi();
}
$('resetDb').addEventListener('click', () => {
  resetArmed = true;
  $('resetStatus').textContent = t('reset_arm');
  clearTimeout(resetRevertTimer);
  resetRevertTimer = setTimeout(disarmReset, 15000);
  setResetUi();
});
$('resetCancel').addEventListener('click', () => {
  disarmReset();
  $('resetStatus').textContent = '';
});
$('resetConfirm').addEventListener('click', async () => {
  if (!resetArmed || resetBusy) return;
  resetBusy = true;
  setResetUi();
  try {
    const response = await apiFetch('/dashboard/api/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Halogen-Action': 'reset'},
      body: JSON.stringify({confirm: true}),
    });
    if (handleUnauthorized(response)) return;
    if (!response.ok) throw new Error((await response.text()) || t('reset_failed'));
    const data = await response.json();
    const count = Object.values(data.deleted || {}).reduce((total, value) => total + value, 0);
    $('resetStatus').textContent = t('reset_done', {count: integer(count)});
    refreshAnalytics(true);
    refreshLive();
  } catch (error) {
    $('resetStatus').textContent = error.message || t('reset_failed');
  } finally {
    resetBusy = false;
    disarmReset();
  }
});
// ---------- Deployment ----------
let deployData = null, deployBusy = false, deployPreviewOk = false, deployArm = null, deployRevision = 0, editorReturnFocus;
const DEPLOY_ENV_FIELDS = [
  ['HALOGEN_CHECKPOINT', 'text', 'deploy_f_checkpoint'],
  ['HALOGEN_TOKENIZER', 'text', 'deploy_f_tokenizer'],
  ['HALOGEN_VISION_TOWER', 'text', 'deploy_f_vision'],
  ['HALOGEN_CACHE_DIR', 'text', 'deploy_f_cache_dir'],
  ['HALOGEN_KV_SLOTS', 'number', 'deploy_f_kv_slots'],
  ['HALOGEN_KV_POOL_POSITIONS', 'number', 'deploy_f_kv_pool'],
  ['HALOGEN_CTX', 'number', 'deploy_f_ctx'],
  ['HALOGEN_MAX_TOK', 'number', 'deploy_f_max_tok'],
  ['HALOGEN_HOST_RESERVE_GIB', 'number', 'deploy_f_host_reserve'],
  ['HALOGEN_CACHE_DISK_GIB', 'number', 'deploy_f_cache_gib'],
  ['HALOGEN_CACHE_PRUNE_OLD', 'check', 'deploy_f_cache_prune'],
  ['HALOGEN_REASONING_EFFORT', 'select:minimal,low,medium,high,xhigh', 'deploy_f_reasoning'],
  ['HALOGEN_MAX_TOKENS_DEFAULT', 'number', 'deploy_f_out_default'],
  ['HALOGEN_MAX_TOKENS_CAP', 'number', 'deploy_f_out_cap'],
  ['HALOGEN_DOWNLOAD', 'text', 'deploy_f_download'],
];
const DEPLOY_ENV_NAMES = new Set(DEPLOY_ENV_FIELDS.map(f => f[0]));
DEPLOY_ENV_NAMES.add('HALOGEN_MODEL_ID');
function setDeployBusy(busy) {
  deployBusy = busy;
  setLanguageAvailability();
  $('view-models').setAttribute('aria-busy', String(busy));
  $('deployProgress').hidden = !busy;
  $('deployProgress').textContent = t('working');
  if (busy) $('deployNotice').hidden = true;
  for (const button of $('view-models').querySelectorAll('button')) {
    if (['profilesTab', 'assetsTab', 'jobCancelBtn'].includes(button.id)) continue;
    button.disabled = busy || (button.id === 'deployApply' && !deployPreviewOk);
  }
}
function setLanguageAvailability() {
  $('langSelect').disabled = localeLoading || deployBusy || updateActionBusy || resetBusy;
}
function volHost(profile, containerPath) {
  const v = (profile.volumes || []).find(x => x[1] === containerPath);
  return v ? v[0] : '';
}
function volRo(profile, containerPath) {
  const v = (profile.volumes || []).find(x => x[1] === containerPath);
  return v ? v[2].includes('ro') : false;
}
function renderDeployForm(profile) {
  const core = [
    ['profile_id', 'text', 'deploy_f_profile_id', profile.profile_id],
    ['model_id', 'text', 'deploy_f_model_id', (profile.env || {}).HALOGEN_MODEL_ID || ''],
    ['image', 'text', 'deploy_f_image', profile.image],
    ['host_port', 'number', 'deploy_f_port', profile.host_port],
    ['models_host', 'text', 'deploy_f_models_path', volHost(profile, '/models')],
    ['models_ro', 'check', 'deploy_f_models_ro', volRo(profile, '/models')],
    ['cache_host', 'text', 'deploy_f_cache_path', volHost(profile, '/cache')],
  ];
  const fieldHtml = (name, type, key, value) => {
    if (type === 'check') return `<label class="chk"><span data-i18n="${key}">${esc(t(key))}</span><input id="df_${name}" type="checkbox" ${value === true || value === '1' ? 'checked' : ''}></label>`;
    if (type.startsWith('select')) {
      const opts = type.split(':')[1].split(',');
      return `<label><span data-i18n="${key}">${esc(t(key))}</span><select id="df_${name}"><option value=""></option>${opts.map(c => `<option value="${c}" ${value === c ? 'selected' : ''}>${c}</option>`).join('')}</select></label>`;
    }
    return `<label><span data-i18n="${key}">${esc(t(key))}</span><input id="df_${name}" type="${type}" value="${esc(String(value ?? ''))}"></label>`;
  };
  const group = (key, html) => `<fieldset><legend data-i18n="${key}">${esc(t(key))}</legend><div class="deploy-fields">${html}</div></fieldset>`;
  const fields = items => items.map(([name, type, key]) => fieldHtml(name, type, key, (profile.env || {})[name] ?? '')).join('');
  $('deployFields').innerHTML = group('profile_identity', core.slice(0, 4).map(f => fieldHtml(...f)).join(''))
    + group('profile_storage', core.slice(4).map(f => fieldHtml(...f)).join('') + fields(DEPLOY_ENV_FIELDS.slice(0, 4)))
    + group('profile_capacity', fields(DEPLOY_ENV_FIELDS.slice(4, 11)))
    + group('profile_generation', fields(DEPLOY_ENV_FIELDS.slice(11)));
  for (const el of $('deployFields').querySelectorAll('input,select')) el.addEventListener('input', invalidateDeployPreview);
  $('deployEnvRows').innerHTML = '';
  for (const [k, v] of Object.entries(profile.env || {})) if (!DEPLOY_ENV_NAMES.has(k)) addEnvRow(k, v);
  $('deployMountRows').innerHTML = '';
  for (const v of profile.volumes || []) if (v[1] !== '/models' && v[1] !== '/cache') addMountRow(v[0], v[1], v[2].includes('ro'));
}
function addMountRow(host = '', container = '', ro = true) {
  const row = document.createElement('div');
  row.className = 'deploy-env-row';
  row.innerHTML = `<input class="mount-host" value="${esc(host)}" placeholder="/host/path" aria-label="${esc(t('mount_host'))}" data-i18n-attr="aria-label:mount_host"><input class="mount-container" value="${esc(container)}" placeholder="/container/path" aria-label="${esc(t('mount_container'))}" data-i18n-attr="aria-label:mount_container"><label class="mount-ro-label"><input type="checkbox" class="mount-ro" ${ro ? 'checked' : ''}>ro</label><button type="button" class="mount-del" aria-label="${esc(t('remove'))}" data-i18n-attr="aria-label:remove">✕</button>`;
  row.querySelector('.mount-del').addEventListener('click', () => { row.remove(); invalidateDeployPreview(); });
  for (const el of row.querySelectorAll('input')) el.addEventListener('input', invalidateDeployPreview);
  $('deployMountRows').appendChild(row);
}
function addEnvRow(key = '', value = '') {
  const row = document.createElement('div');
  row.className = 'deploy-env-row';
  row.innerHTML = `<input list="deployEnvKeys" class="env-key" value="${esc(key)}" placeholder="HALOGEN_..." aria-label="${esc(t('field_parameter'))}" data-i18n-attr="aria-label:field_parameter"><input class="env-value" value="${esc(value)}" placeholder="${esc(t('field_value'))}" aria-label="${esc(t('field_value'))}" data-i18n-attr="placeholder:field_value;aria-label:field_value"><button type="button" class="env-del" aria-label="${esc(t('remove'))}" data-i18n-attr="aria-label:remove">✕</button>`;
  row.querySelector('.env-del').addEventListener('click', () => { row.remove(); invalidateDeployPreview(); });
  for (const el of row.querySelectorAll('input')) el.addEventListener('input', invalidateDeployPreview);
  $('deployEnvRows').appendChild(row);
}
function collectDeploy() {
  const env = {};
  const set = (k, v) => { if (v !== '') env[k] = v; };
  set('HALOGEN_MODEL_ID', $('df_model_id').value.trim());
  for (const [name, type] of DEPLOY_ENV_FIELDS) {
    const el = $('df_' + name);
    set(name, type === 'check' ? (el.checked ? '1' : '') : el.value.trim());
  }
  for (const row of $('deployEnvRows').children) {
    const k = row.querySelector('.env-key').value.trim();
    if (k) env[k] = row.querySelector('.env-value').value.trim();
  }
  const mounts = [];
  for (const row of $('deployMountRows').children) {
    const host = row.querySelector('.mount-host').value.trim();
    const container = row.querySelector('.mount-container').value.trim();
    if (host && container) mounts.push([host, container, row.querySelector('.mount-ro').checked ? 'ro,Z' : 'Z']);
  }
  return {
    profile_id: $('df_profile_id').value.trim(),
    image: $('df_image').value.trim(),
    host_port: parseInt($('df_host_port').value, 10) || 0,
    volumes: [
      [$('df_models_host').value.trim(), '/models', $('df_models_ro').checked ? 'ro,Z' : 'Z'],
      [$('df_cache_host').value.trim(), '/cache', 'Z'],
      ...mounts,
    ],
    env,
  };
}
function invalidateDeployPreview() { deployRevision++; deployPreviewOk = false; $('deployApply').disabled = true; $('deployDiff').hidden = true; $('deployWarnings').hidden = true; $('previewStatus').textContent = t('preview_required'); resetDeployArm(); }
function showDeployError(message) { const el = $('deployError'); el.hidden = false; el.textContent = message; el.scrollIntoView({block: 'nearest'}); }
function resetDeployArm() { if (deployArm) { deployArm.btn.textContent = deployArm.label; deployArm = null; } }
function armDeploy(btn, run, confirmLabel) {
  if (deployArm && deployArm.btn === btn) { const fn = deployArm.run; resetDeployArm(); fn(); return; }
  resetDeployArm();
  deployArm = {btn, label: btn.textContent, run};
  btn.textContent = confirmLabel || t('deploy_confirm');
}
async function deployRequest(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.method === 'POST' ? 180000 : 15000);
  let response;
  try { response = await apiFetch(path, {cache: 'no-store', signal: controller.signal, ...options}); }
  finally { clearTimeout(timeout); }
  if (handleUnauthorized(response)) throw new Error(t('auth_required'));
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.error || data.message || `HTTP ${response.status} ${response.statusText}`;
    throw new Error(detail);
  }
  return data;
}
async function refreshDeploy() {
  try { deployData = await deployRequest('/dashboard/api/deploy'); renderDeploy(); } catch { showDeployError(t('deploy_load_failed')); }
}
function renderDeploy() {
  if (!deployData) return;
  $('deployError').hidden = true;
  if (!deployData.enabled || !deployData.posix) {
    $('deployHint').textContent = t('deploy_disabled');
    $('deployList').innerHTML = '';
    $('quickDeploy').hidden = true;
    $('deployEditor').hidden = true;
    $('deployNew').hidden = true;
    $('deployHf').hidden = true;
    return;
  }
  $('quickDeploy').hidden = false;
  $('deployNew').hidden = false;
  $('deployHf').hidden = false;
  refreshHf();
  $('deployHint').textContent = t('deploy_hint', {dir: deployData.quadlet_dir});
  const rows = (deployData.profiles || []).map(p => {
    if (p.error) return `<div class="deploy-row"><strong>${esc(p.profile_id)}</strong> <span class="failure">${esc(p.error)}</span></div>`;
    const badge = p.service_active ? `<span class="badge ok">${esc(t('deploy_running'))}</span>` : `<span class="badge">${esc(t('deploy_stopped'))}</span>`;
    const btns = [`<button type="button" data-act="edit" data-id="${esc(p.profile_id)}">${esc(t('deploy_edit'))}</button>`];
    if (!p.service_active) btns.push(`<button type="button" data-act="start" data-id="${esc(p.profile_id)}">${esc(t('deploy_start'))}</button>`);
    if (p.has_backup) btns.push(`<button type="button" data-act="rollback" data-id="${esc(p.profile_id)}">${esc(t('deploy_rollback'))}</button>`);
    if (!p.service_active) btns.push(`<button type="button" data-act="delete" data-id="${esc(p.profile_id)}">${esc(t('deploy_delete'))}</button>`);
    return `<div class="deploy-row"><div class="profile-info"><div class="profile-title"><strong>${esc(p.profile_id)}</strong>${badge}</div><span>${esc(p.model_id)} · ${esc(String(p.image).split(':').pop())}</span></div><div class="deploy-row-actions">${btns.join(' ')}</div></div>`;
  }).join('');
  $('deployList').innerHTML = rows || `<p class="muted">${esc(t('deploy_none'))}</p>`;
  $('deployEnvKeys').innerHTML = (deployData.env_keys || []).map(k => `<option value="${k}"></option>`).join('');
}
async function deployReload() {
  try { await deployRequest('/dashboard/api/deploy/reload', {method: 'POST', headers: {'X-Halogen-Action': 'deploy'}}); refreshLive(); } catch { /* reload optional */ }
}
async function runDeployAction(id, act) {
  if (deployBusy) return;
  setDeployBusy(true);
  try {
    await deployRequest(`/dashboard/api/deploy/${encodeURIComponent(id)}/${act}`, {method: 'POST', headers: {'X-Halogen-Action': 'deploy'}});
    await refreshDeploy();
    if (act !== 'start') await deployReload();
    $('deployNotice').hidden = false;
    $('deployNotice').textContent = t('profile_action_done');
  } catch (err) { showDeployError(err.message); }
  finally { setDeployBusy(false); resetDeployArm(); }
}
$('deployList').addEventListener('click', event => {
  const btn = event.target.closest('button[data-act]');
  if (!btn || deployBusy) return;
  const act = btn.dataset.act, id = btn.dataset.id;
  if (act === 'edit') {
    const p = (deployData.profiles || []).find(x => x.profile_id === id);
    if (p && !p.error) openDeployEditor(p, false);
    return;
  }
  if (act === 'rollback' || act === 'delete') { armDeploy(btn, () => runDeployAction(id, act)); return; }
  runDeployAction(id, act);
});
$('deployAddEnv').addEventListener('click', () => { addEnvRow(); invalidateDeployPreview(); });
$('deployAddMount').addEventListener('click', () => { addMountRow(); invalidateDeployPreview(); });
$('deployLoadTemplate').addEventListener('click', async () => {
  if (deployBusy) return;
  try {
    const data = await deployRequest(`/dashboard/api/deploy/template/${$('deployTemplate').value}`);
    openDeployEditor(data.profile, true);
  } catch (err) { showDeployError(err.message); }
});
function openDeployEditor(profile, isNew) {
  editorReturnFocus = document.activeElement;
  $('deployEditor').hidden = false;
  $('deployEditorTitle').textContent = (isNew ? t('deploy_new_profile') : t('deploy_edit_profile')) + ': ' + profile.profile_id;
  renderDeployForm(profile);
  $('deployDiff').hidden = true;
  $('deployWarnings').hidden = true;
  $('deployApply').disabled = true;
  deployPreviewOk = false;
  $('deployEditor').dataset.profileId = profile.profile_id;
  $('deployEditor').dataset.isNew = String(isNew);
  deployRevision++;
  $('previewStatus').textContent = t('preview_required');
  $('deployEditorTitle').focus();
}
$('deployCancel').addEventListener('click', () => {
  $('deployEditor').hidden = true;
  resetDeployArm();
  (editorReturnFocus?.isConnected ? editorReturnFocus : $('deployRefresh')).focus();
});
$('deployPreview').addEventListener('click', async () => {
  if (deployBusy) return;
  setDeployBusy(true);
  const revision = deployRevision;
  $('deployError').hidden = true;
  try {
    const data = await deployRequest('/dashboard/api/deploy/dry-run', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Halogen-Action': 'deploy'}, body: JSON.stringify(collectDeploy())});
    if (revision !== deployRevision) return;
    $('deployDiff').hidden = false;
    $('deployDiff').textContent = data.diff;
    const notes = [...(data.errors || []), ...(data.warnings || [])];
    $('deployWarnings').hidden = notes.length === 0;
    $('deployWarnings').textContent = notes.join('\n');
    $('deployWarnings').className = data.ok ? 'warning' : 'failure';
    $('previewStatus').textContent = data.ok ? t('preview_ready') : t('preview_invalid');
    deployPreviewOk = data.ok;
    $('deployApply').disabled = !data.ok;
  } catch (err) { showDeployError(err.message); }
  finally { setDeployBusy(false); }
});
$('deployApply').addEventListener('click', () => {
  if (!deployPreviewOk || deployBusy) return;
  armDeploy($('deployApply'), async () => {
    setDeployBusy(true);
    try {
      await deployRequest('/dashboard/api/deploy/apply', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Halogen-Action': 'deploy'}, body: JSON.stringify(collectDeploy())});
      $('deployEditor').hidden = true;
      await refreshDeploy();
      await deployReload();
      $('deployNotice').hidden = false;
      $('deployNotice').textContent = t('profile_saved');
    } catch (err) { showDeployError(err.message); }
    finally { setDeployBusy(false); resetDeployArm(); }
  });
});
// ---------- HF download & convert ----------
let jobTimer = null, jobBusy = false;
let hfDefaults = {};
let hfAutoGguf = false;
function syncHfGgufPath() {
  if (!hfAutoGguf || !hfDefaults.default_gguf_name) return;
  const dest = $('hfDest').value.trim().replace(/\/$/, '');
  if (dest) $('convGguf').value = dest + '/' + hfDefaults.default_gguf_name;
}
async function refreshHf() {
  try {
    const hf = await deployRequest('/dashboard/api/deploy/hf');
    hfDefaults = hf || {};
    $('hfStatusLine').textContent = hf.installed ? t('hf_installed') : t('hf_not_installed');
    $('hfInstallBtn').hidden = hf.installed;
    if (!$('hfRepo').value) $('hfRepo').value = hf.default_repo || '';
    if (!$('hfFile').value) $('hfFile').value = hf.default_file || '';
    if (!$('hfDest').value && deployData.allowed_roots && deployData.allowed_roots.length) {
      $('hfDest').value = String(deployData.allowed_roots[0]).replace(/\/$/, '') + '/halogen/models/uncensored';
    }
    if (!$('convImage').value) {
      const tag = (typeof updatesData !== 'undefined' && updatesData && updatesData.latest_version) || '';
      if (tag) $('convImage').value = 'ghcr.io/peonist-ai/halogen-flash-server:' + tag;
    }
    if (!$('convOut').value) $('convOut').value = hf.default_output || 'qwen3.8-flash-uncensored.hgn';
    if (!$('convGguf').value && hf.default_gguf_name) {
      hfAutoGguf = true;
      syncHfGgufPath();
    }
  } catch { $('hfStatusLine').textContent = t('hf_load_failed'); }
}
$('hfDest').addEventListener('input', syncHfGgufPath);
function startJobPoll() { clearTimeout(jobTimer); pollJob(); }
async function pollJob() {
  if (jobBusy || document.hidden) return;
  clearTimeout(jobTimer);
  jobBusy = true;
  try {
    const job = await deployRequest('/dashboard/api/deploy/job');
    if (!job.active) { $('jobBox').hidden = true; return; }
    $('jobBox').hidden = false;
    const step = job.total_steps ? ` (${job.step || 0}/${job.total_steps})` : '';
    $('jobKind').textContent = t('job_' + job.kind.replace(/-/g, '_')) + step;
    const stateEl = $('jobState');
    stateEl.textContent = t('job_state_' + job.state);
    stateEl.className = 'badge' + (job.state === 'running' ? ' ok' : job.state === 'error' ? ' err' : '');
    $('jobElapsed').textContent = Math.round(job.elapsed) + 's';
    const jobLog = $('jobLog');
    const followLog = jobLog.scrollHeight - jobLog.scrollTop - jobLog.clientHeight < 48;
    jobLog.textContent = (job.lines || []).join('\n') + (job.error ? '\nERROR: ' + job.error : '');
    if (followLog) jobLog.scrollTop = jobLog.scrollHeight;
    $('jobCancelBtn').hidden = job.state !== 'running';
    if (job.state === 'running') {
      jobTimer = setTimeout(pollJob, 2000);
    } else if (job.kind === 'hf-install') {
      await refreshHf();
    }
  } catch { jobTimer = setTimeout(pollJob, 5000); }
  finally { jobBusy = false; }
}
$('jobCancelBtn').addEventListener('click', () => {
  armDeploy($('jobCancelBtn'), async () => {
    try {
      await deployRequest('/dashboard/api/deploy/job/cancel', {method: 'POST', headers: {'X-Halogen-Action': 'deploy'}});
    } catch (err) { showDeployError(err.message); }
    finally { resetDeployArm(); }
  }, t('job_cancel_confirm'));
});
async function startModelJob(path, payload) {
  if (deployBusy) return;
  setDeployBusy(true);
  $('deployError').hidden = true;
  try {
    await deployRequest(path, {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-Halogen-Action': 'deploy'},
      body: JSON.stringify(payload || {}),
    });
    startJobPoll();
  } catch (err) { showDeployError(err.message); }
  finally { setDeployBusy(false); }
}
$('hfInstallBtn').addEventListener('click', () => startModelJob('/dashboard/api/deploy/hf/install'));
$('hfDownloadBtn').addEventListener('click', async () => {
  if (deployBusy) return;
  const token = $('hfToken').value.trim();
  if (!token) {
    showDeployError(t('hf_token_required'));
    return;
  }
  $('hfToken').value = '';
  await startModelJob('/dashboard/api/deploy/hf/download', {repo: $('hfRepo').value.trim(), file: $('hfFile').value.trim(), dest: $('hfDest').value.trim(), token});
});
$('quickOfficialBtn').addEventListener('click', () => {
  if (deployBusy) return;
  armDeploy($('quickOfficialBtn'), async () => {
    setDeployBusy(true);
    $('deployError').hidden = true;
    try {
      await deployRequest('/dashboard/api/deploy/quick/official', {method: 'POST', headers: {'X-Halogen-Action': 'deploy'}});
      await refreshDeploy();
      await deployReload();
      startJobPoll();
    } catch (err) { showDeployError(err.message); }
    finally { setDeployBusy(false); resetDeployArm(); }
  }, t('quick_confirm'));
});
$('quickUncensoredBtn').addEventListener('click', () => {
  if (deployBusy) return;
  const token = $('quickHfToken').value.trim();
  if (!token) {
    showDeployError(t('hf_token_required'));
    return;
  }
  armDeploy($('quickUncensoredBtn'), async () => {
    setDeployBusy(true);
    $('deployError').hidden = true;
    try {
      await deployRequest('/dashboard/api/deploy/quick/uncensored', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Halogen-Action': 'deploy'}, body: JSON.stringify({token})});
      $('quickHfToken').value = '';
      startJobPoll();
    } catch (err) { showDeployError(err.message); }
    finally { setDeployBusy(false); resetDeployArm(); }
  }, t('quick_confirm'));
});
$('convBtn').addEventListener('click', async () => {
  await startModelJob('/dashboard/api/deploy/convert', {image: $('convImage').value.trim(), gguf: $('convGguf').value.trim(), output: $('convOut').value.trim(), download_head: $('convHead').checked});
});
$('verifyBtn').addEventListener('click', async () => {
  const gguf = $('convGguf').value.trim();
  const dir = gguf.slice(0, gguf.lastIndexOf('/') + 1);
  await startModelJob('/dashboard/api/deploy/verify', {image: $('convImage').value.trim(), hgn: dir + $('convOut').value.trim()});
});
$('toDate').value = localInput(new Date());
$('fromDate').value = localInput(new Date(Date.now() - 86400000));
new ResizeObserver(() => {
  if (currentAnalytics) renderChart(currentAnalytics);
}).observe(document.querySelector('.chart-area'));
document.addEventListener('visibilitychange', () => {
  clearTimeout(liveTimer); clearTimeout(analyticsTimer); clearTimeout(updatesTimer); clearTimeout(jobTimer);
  if (!document.hidden) { refreshLive(); refreshAnalytics(); refreshUpdates(); refreshDeploy(); pollJob(); }
});
function applyLocale() {
  document.documentElement.lang = currentLang;
  document.title = t('page_title');
  for (const el of document.querySelectorAll('[data-i18n]')) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll('[data-i18n-html]')) el.innerHTML = t(el.dataset.i18nHtml);
  for (const el of document.querySelectorAll('[data-i18n-attr]')) {
    for (const pair of el.dataset.i18nAttr.split(';')) {
      const [attr, key] = pair.split(':');
      if (attr && key) el.setAttribute(attr, t(key));
    }
  }
}
async function switchLanguage(lang) {
  const sequence = ++localeSequence;
  localeLoading = true;
  setLanguageAvailability();
  try {
    const response = await fetch(`/dashboard/api/locale/${encodeURIComponent(lang)}`, {cache: 'no-store'});
    if (handleUnauthorized(response)) return;
    if (!response.ok) throw new Error('locale unavailable');
    const strings = await response.json();
    if (sequence !== localeSequence) return;
    L = strings;
    currentLang = lang;
    NUM = L.__locale || (lang === 'de' ? 'de-DE' : 'en-US');
    try { localStorage.setItem('halobridge_lang', currentLang); } catch {}
    document.cookie = `halobridge_lang=${currentLang}; Path=/dashboard; SameSite=Lax`;
    applyLocale();
    $('deployNotice').hidden = true;
    $('copyStatus').textContent = '';
    $('resetStatus').textContent = '';
    showView();
    setResetUi();
    resetDeployArm();
    if (!$('deployEditor').hidden) {
      $('deployEditorTitle').textContent = t($('deployEditor').dataset.isNew === 'true' ? 'deploy_new_profile' : 'deploy_edit_profile') + ': ' + $('deployEditor').dataset.profileId;
      invalidateDeployPreview();
    }
    if (liveData) renderLive(liveData);
    if (currentAnalytics) renderAnalytics(currentAnalytics);
    await Promise.all([refreshLive(), refreshAnalytics(), refreshUpdates(), refreshDeploy(), pollJob()]);
  } catch {
    if (sequence !== localeSequence) return;
    if (!Object.keys(L).length && lang !== 'en') { await switchLanguage('en'); return; }
    $('analyticsError').hidden = false;
    $('analyticsError').textContent = Object.keys(L).length ? t('locale_failed') : 'Could not load the dashboard. Reload the page to try again.';
  } finally {
    if (sequence === localeSequence) {
      localeLoading = false;
      $('langSelect').value = currentLang;
      setLanguageAvailability();
    }
  }
}
const initialLang = (() => {
  const cookie = document.cookie.match(/(?:^|; )halobridge_lang=(de|en)(?:;|$)/);
  if (cookie) return cookie[1];
  try {
    const stored = localStorage.getItem('halobridge_lang');
    if (stored === 'de' || stored === 'en') return stored;
  } catch {}
  return 'en';
})();
$('langSelect').value = initialLang;
$('langSelect').addEventListener('change', event => switchLanguage(event.target.value));
switchLanguage(initialLang);
