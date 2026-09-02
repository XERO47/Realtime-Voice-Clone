const $ = id => document.getElementById(id);
const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = path => `${wsProtocol}//${location.host}${path}`;
const monitor = new WebSocket(wsUrl('/ws/monitor'));
const sessions = new Map();
let selectedId = null;
let audioContext = null;
const taps = new Map();
const wavePhase = new Map();

const STATE_META = {
  safe:          { badge: 'Calm',            word: 'Natural',     tone: 'safe' },
  verify:        { badge: 'Uncertain',       word: 'Uncertain',   tone: 'verify' },
  high_risk:     { badge: 'Needs attention', word: 'High risk',   tone: 'high' },
  insufficient:  { badge: 'Listening…',      word: 'Listening…',  tone: 'neutral' },
  model_error:   { badge: 'Unavailable',     word: 'Unavailable', tone: 'neutral' },
};

const SENTENCES = {
  safe: (a, b) => `Both callers — <em>${a}</em> and <em>${b}</em> — sound human right now. We'll keep listening quietly and only interrupt if that changes.`,
  verify: name => `<em>${name}</em>'s recent audio doesn't quite match a typical human pattern. It's not conclusive yet — a quiet verification question can confirm it's really them.`,
  high_risk: name => `<em>${name}</em>'s voice has stayed consistent with a cloned or generated pattern. Verify through another channel before continuing.`,
  insufficient: name => `Still collecting enough audio from <em>${name}</em> to say anything meaningful.`,
  model_error: name => `The detector is temporarily unreachable for <em>${name}</em>. We'll keep retrying in the background.`,
};
const TONE_COLOR = { safe: '#3ecf6e', verify: '#e3a53d', high: '#f2545b', neutral: '#4a4a50' };
const AUDIO_WAVE_ON = '#4f8dff';
const AUDIO_WAVE_OFF = '#4a4a50';
const SEVERITY = { safe: 0, verify: 1, high_risk: 2 };
const peaks = new Map();

// Once a caller has ever been classified safe/verify/high_risk, the dashboard keeps showing
// the worst (highest-severity) reading for the rest of the call instead of following the live
// score back down — a brief quiet patch or a synthetic source going idle shouldn't erase a flag.
function peakFor(session, userId, state, score) {
  const key = `${session.session_id}:${userId}`;
  const severity = SEVERITY[state];
  if (severity !== undefined) {
    const current = peaks.get(key);
    if (!current || severity >= SEVERITY[current.state]) peaks.set(key, { state, score });
  }
  return peaks.get(key);
}

function toast(message) { const el = $('toast'); el.textContent = message; el.classList.add('show'); setTimeout(() => el.classList.remove('show'), 2600); }
function duration(session) { const start = (session.connected_at || session.created_at) * 1000; const sec = Math.max(0, Math.floor((Date.now() - start) / 1000)); return `${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`; }
function pct(score) { return typeof score === 'number' ? `${Math.round(score * 100)}%` : '--'; }
// Backend score is P(fake). For display, "safe" reads better as confidence-of-real
// (0.0004 P(fake) shown as "0%" reads as an unimpressive non-answer; flipped to "100%" it
// actually reads as reassurance). Verify/high_risk keep showing P(fake) as-is, since a big
// alarming number is exactly the right signal there.
function displayScore(score, state) { return typeof score === 'number' && state === 'safe' ? 1 - score : score; }
function ms(latency) { return typeof latency === 'number' ? `${Math.round(latency)}ms` : '--'; }
function windowCaption(state, pending, latency, progress) {
  if (state === 'model_error') return 'Detector unavailable';
  if (pending) return 'Running inference…';
  if (typeof latency === 'number') return `${Math.round(progress * 100)}% of the next window collected`;
  return 'Collecting the first 4-second window';
}

monitor.onopen = () => { $('monitorPill').classList.add('on'); $('monitorPill').querySelector('span').textContent = 'Monitor online'; };
monitor.onclose = () => { $('monitorPill').classList.remove('on'); $('monitorPill').querySelector('span').textContent = 'Disconnected'; };
monitor.onmessage = event => {
  const message = JSON.parse(event.data);
  if (message.type === 'sessions') { syncSessions(message.sessions); renderQueue(); renderMain(); }
  if (message.type === 'telemetry') { message.sessions.forEach(session => sessions.set(session.session_id, session)); renderMain(); }
  if (message.type === 'monitor-error') toast(message.message);
};

async function updateModelStatus() {
  try {
    const status = await fetch('/api/detector/status').then(r => r.json());
    const ready = status.connected && status.checkpoint;
    const pill = $('modelPill');
    pill.classList.toggle('on', !!ready);
    pill.classList.toggle('warn', !ready);
    const device = status.device || 'CPU';
    pill.querySelector('span').textContent = ready ? (status.loaded ? `Detector live · ${device}` : `Detector ready · ${device}`) : 'Detector offline';
  } catch (_) { $('modelPill').querySelector('span').textContent = 'Detector offline'; }
}

function syncSessions(items) {
  const ids = new Set(items.map(item => item.session_id));
  [...sessions.keys()].forEach(id => { if (!ids.has(id)) sessions.delete(id); });
  items.forEach(item => sessions.set(item.session_id, item));
  if (selectedId && !sessions.has(selectedId)) { stopAllTaps(); selectedId = null; }
  if (!selectedId && items.length) selectedId = items[0].session_id;
}

function pickSubject(session) {
  const a = session.caller_a, b = session.caller_b;
  const peakA = peakFor(session, a, session.risk_states?.[a] || 'insufficient', session.risk_scores?.[a]);
  const peakB = peakFor(session, b, session.risk_states?.[b] || 'insufficient', session.risk_scores?.[b]);
  const rank = peak => (peak ? SEVERITY[peak.state] : -1);
  if (!peakA && !peakB) return { id: a, score: null, state: session.risk_states?.[a] || 'insufficient' };
  const id = rank(peakB) > rank(peakA) ? b : a;
  const peak = id === a ? peakA : peakB;
  return { id, score: peak.score, state: peak.state };
}

function renderQueue() {
  $('queueCount').textContent = `${sessions.size} live`;
  if (!sessions.size) { $('queueList').innerHTML = '<div class="vg-empty-queue">Waiting for an active call…</div>'; return; }
  $('queueList').innerHTML = [...sessions.values()].map(session => {
    const simulated = session.caller_a_role === 'attacker' || session.caller_b_role === 'attacker';
    const subject = pickSubject(session);
    const meta = STATE_META[subject.state] || STATE_META.insufficient;
    return `<button class="vg-scard ${session.session_id === selectedId ? 'selected' : ''}" data-session="${session.session_id}">
      <div class="row1"><span class="vg-dot ${meta.tone}"></span><span class="spair">${session.caller_a} &amp; ${session.caller_b}</span></div>
      <div class="smeta"><span>${session.session_id}${simulated ? ' · red-team run' : ''}</span><span class="vg-badge-word ${meta.tone}">${meta.badge}</span></div>
    </button>`;
  }).join('');
  document.querySelectorAll('[data-session]').forEach(button => button.onclick = () => {
    if (button.dataset.session !== selectedId) stopAllTaps();
    selectedId = button.dataset.session;
    renderQueue();
    renderMain();
  });
}

function waveformPoints(level, online, voiceActive, phase) {
  const width = 300, height = 34, mid = height / 2, n = 26;
  const points = [];
  for (let i = 0; i < n; i++) {
    const x = Math.round(i * (width / (n - 1)));
    let y = mid;
    if (online) {
      const amp = voiceActive ? 3 + Math.min(1, level) * 12 : 1.1;
      y = mid + Math.sin(phase + i * 0.85) * amp;
    }
    points.push(`${x},${y.toFixed(1)}`);
  }
  return points.join(' ');
}

function buildBarChart(el, history) {
  if (!history || !history.length) { el.innerHTML = '<span class="waiting">Waiting for first inference…</span>'; return; }
  el.innerHTML = history.map(point => {
    const raw = typeof point.score === 'number' ? point.score : 0;
    const tone = (STATE_META[point.state] || STATE_META.insufficient).tone;
    const shown = displayScore(raw, point.state) ?? 0;
    const height = Math.max(6, Math.round(shown * 100));
    return `<i class="${tone}" style="height:${height}%" title="${Math.round(shown * 100)}%"></i>`;
  }).join('');
}

function renderParticipant(prefix, session, userId) {
  const online = !!session.audio_online?.[userId];
  const level = session.audio_levels?.[userId] || 0;
  const voiceActive = !!session.audio_voice_active?.[userId];
  const state = session.risk_states?.[userId] || 'insufficient';
  const score = session.risk_scores?.[userId];
  const latency = session.risk_latency_ms?.[userId];
  const progress = session.detector_progress?.[userId] || 0;
  const pending = !!session.detector_pending?.[userId];
  const peak = peakFor(session, userId, state, score);

  $(`name${prefix}`).textContent = userId;
  $(`online${prefix}`).textContent = !online ? 'Waiting for audio' : voiceActive ? 'Voice detected' : 'Quiet, audio healthy';

  const scoreEl = $(`score${prefix}`), wordEl = $(`word${prefix}`);
  if (peak) {
    scoreEl.textContent = pct(displayScore(peak.score, peak.state));
    wordEl.textContent = STATE_META[peak.state].word;
    $(`card${prefix}`).setAttribute('data-tone', STATE_META[peak.state].tone);
  } else if (state === 'model_error') {
    scoreEl.textContent = '--';
    wordEl.textContent = STATE_META.model_error.word;
    $(`card${prefix}`).setAttribute('data-tone', 'neutral');
  } else {
    scoreEl.textContent = '--';
    wordEl.textContent = '';
    $(`card${prefix}`).setAttribute('data-tone', 'neutral');
  }
  $(`latency${prefix}`).textContent = ms(latency);
  $(`window${prefix}`).textContent = windowCaption(state, pending, latency, progress);

  const key = `${session.session_id}:${userId}`;
  const phase = (wavePhase.get(key) || 0) + 0.6;
  wavePhase.set(key, phase);
  const wave = $(`wave${prefix}`);
  wave.setAttribute('points', waveformPoints(level, online, voiceActive, phase));
  wave.setAttribute('stroke', online ? AUDIO_WAVE_ON : AUDIO_WAVE_OFF);

  buildBarChart($(`bars${prefix}`), session.risk_history?.[userId] || []);
}

function renderMain() {
  const session = sessions.get(selectedId);
  $('mainEmpty').classList.toggle('hidden', !!session);
  $('mainDetail').classList.toggle('hidden', !session);
  document.querySelectorAll('.vg-topbar-actions [data-action]').forEach(button => { button.disabled = !session; button.title = session ? '' : 'Select a session first'; });
  if (!session) return;

  $('sessionSummary').textContent = `${session.caller_a} ↔ ${session.caller_b} · ${session.status} · ${duration(session)}`;

  const subject = pickSubject(session);
  $('heroSentence').innerHTML = subject.state === 'safe' ? SENTENCES.safe(session.caller_a, session.caller_b) : (SENTENCES[subject.state] || SENTENCES.insufficient)(subject.id);

  renderParticipant('A', session, session.caller_a);
  renderParticipant('B', session, session.caller_b);
}

async function ensureAudioContext() { if (!audioContext) audioContext = new AudioContext(); await audioContext.resume(); return audioContext; }
async function openTap(userId, label) {
  if (taps.has(userId)) return;
  const context = await ensureAudioContext();
  const state = { socket: null, sampleRate: 48000, nextTime: context.currentTime + 0.08 };
  const socket = new WebSocket(wsUrl(`/ws/tap/${selectedId}/${encodeURIComponent(userId)}`)); socket.binaryType = 'arraybuffer'; state.socket = socket; taps.set(userId, state);
  socket.onmessage = event => {
    if (typeof event.data === 'string') { const meta = JSON.parse(event.data); state.sampleRate = meta.sample_rate || 48000; return; }
    const samples = new Float32Array(event.data); if (!samples.length) return;
    const buffer = context.createBuffer(1, samples.length, state.sampleRate); buffer.copyToChannel(samples, 0);
    const source = context.createBufferSource(); source.buffer = buffer; source.connect(context.destination);
    state.nextTime = Math.max(state.nextTime, context.currentTime + 0.06); source.start(state.nextTime); state.nextTime += buffer.duration;
  };
  socket.onclose = () => taps.delete(userId);
  $('tapState').classList.remove('hidden'); $('tapLabel').textContent = label;
}
function stopAllTaps() { taps.forEach(state => state.socket?.close()); taps.clear(); $('tapState').classList.add('hidden'); document.querySelectorAll('[data-tap]').forEach(button => button.classList.remove('selected')); }

document.querySelectorAll('[data-tap]').forEach(button => button.onclick = async () => {
  const session = sessions.get(selectedId); if (!session) return;
  stopAllTaps(); button.classList.add('selected');
  if (button.dataset.tap === 'caller_a') await openTap(session.caller_a, `Caller A · ${session.caller_a}`);
  if (button.dataset.tap === 'caller_b') await openTap(session.caller_b, `Caller B · ${session.caller_b}`);
  if (button.dataset.tap === 'both') { await openTap(session.caller_a, 'Both participants'); await openTap(session.caller_b, 'Both participants'); }
});
$('stopTap').onclick = () => { stopAllTaps(); toast('Live audio tap stopped.'); };

document.querySelectorAll('[data-action]').forEach(button => button.onclick = () => {
  if (!selectedId) return;
  const action = button.dataset.action;
  if (action === 'end' && !confirm('End this call for both participants?')) return;
  monitor.send(JSON.stringify({ type: 'intervention', session_id: selectedId, action, target: 'both' }));
  const labels = { verify: 'Verification requested.', hold: 'Verification hold placed.', end: 'Session termination requested.' };
  $('eventLog').innerHTML = `<b>${new Date().toLocaleTimeString()}</b> — ${labels[action]}`;
  toast(labels[action]);
});

updateModelStatus();
setInterval(updateModelStatus, 5000);
setInterval(() => { if (sessions.size) { renderQueue(); if (selectedId) renderMain(); } }, 1000);
