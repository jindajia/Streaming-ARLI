'use strict';
/* SCH-01 fixed_delay10_v2 viewer.  All data are precomputed (data/); no GPU or server logic is needed.
   Action index: 0 = KEEP; 1 + (R-1)*8 + (N-3) = REPLAN(R, N).  Probabilities are formed exactly as in
   sch01.learner.policy_probs: softmax over feasible actions of residual + prior, prior = strength * log(p0 (M-1)/(1-p0))
   on the fixed-baseline action (M = number of feasible actions, p0 = 0.999). */

const NA = 201, P0 = 0.999, NV = [3, 4, 5, 6, 7, 8, 9, 10], RMIN_SHOWN = 6, RMAX = 25;
const DELAY = { 3: 6, 4: 7, 5: 7, 6: 8, 7: 9, 8: 9, 9: 10, 10: 10 };
const GAMMA = 0.999, HZ = 50;
const COL = { accent: '#2a63c4', accent2: '#d9632b', keep: '#7a5cc4', good: '#1a9a63', bad: '#c43a3a', ink2: '#5b5a55',
  grid: '#ebeae5', q: '#0f8a8a', obs: '#9a6b00', across: '#f3f0fa' };
const enc = (r, n) => 1 + (r - 1) * 8 + (n - 3);
const decA = (a) => (a === 0 ? { keep: true } : { keep: false, r: Math.floor((a - 1) / 8) + 1, n: 3 + ((a - 1) % 8) });
const actName = (a) => { if (a === null || a === undefined || a < 0) return '–'; const d = decA(a); return d.keep ? 'KEEP' : `REPLAN(R=${d.r}, N=${d.n})`; };
const actShort = (a) => { const d = decA(a); return d.keep ? 'KEEP' : `R${d.r}·N${d.n}`; };
const $ = (id) => document.getElementById(id);
const pct = (p) => (p === null || p === undefined ? '–' : p >= 0.9995 ? (100 * p).toFixed(2) + '%' : p >= 1e-3 ? (100 * p).toFixed(1) + '%' : p.toExponential(1));
const fmtSteps = (s) => s.toLocaleString('en-US');

function priorStrength(step) { if (step <= 5000) return 1; if (step >= 50000) return 0; return 1 - (step - 5000) / 45000; }

function feasibleMask(q, ticksLeft) {
  const m = new Uint8Array(NA);
  for (const n of NV) for (let r = DELAY[n]; r <= Math.min(q, RMAX); r++) if (r + 25 <= 50) m[enc(r, n)] = 1;
  if (q - 1 >= 6 && ticksLeft > 1) m[0] = 1;
  return m;
}

function probsFrom(res, mask, ref, strength) {
  let M = 0;
  for (let a = 0; a < NA; a++) M += mask[a];
  const big = M > 1 ? strength * Math.log(P0 * (M - 1) / (1 - P0)) : 0;
  const lg = new Float64Array(NA);
  let mx = -Infinity;
  for (let a = 0; a < NA; a++) {
    if (!mask[a]) { lg[a] = -Infinity; continue; }
    lg[a] = res[a] + (a === ref ? big : 0);
    if (lg[a] > mx) mx = lg[a];
  }
  const p = new Float64Array(NA);
  let z = 0;
  for (let a = 0; a < NA; a++) if (mask[a]) { p[a] = Math.exp(lg[a] - mx); z += p[a]; }
  for (let a = 0; a < NA; a++) p[a] /= z;
  return p;
}

function priorBig(mask, strength) { let M = 0; for (let a = 0; a < NA; a++) M += mask[a]; return M > 1 ? strength * Math.log(P0 * (M - 1) / (1 - P0)) : 0; }

function onehot(a) { const p = new Float64Array(NA); p[a] = 1; return p; }

function distStats(p, mask) {
  const margN = {}, margR = {};
  NV.forEach((n) => { margN[n] = 0; });
  for (let r = 1; r <= RMAX; r++) margR[r] = 0;
  let rep = 0, en = 0, er = 0, ent = 0, am = 0;
  for (let a = 0; a < NA; a++) {
    const v = p[a];
    if (v > 0) ent -= v * Math.log(v);
    if (v > p[am]) am = a;
    if (a === 0) continue;
    const d = decA(a);
    rep += v; margN[d.n] += v; margR[d.r] += v; en += v * d.n; er += v * d.r;
  }
  const ok = rep > 1e-9;
  return { pKeep: p[0], rep, margN, margR, EN: ok ? en / rep : null, ER: ok ? er / rep : null, ent, argmax: am, negligible: rep < 1e-6 };
}

/* ------------------------------------------------------------------ data */
const App = { idx: null, learning: null, eps: new Map(), policies: [], ckById: {}, charts: {} };
const S = { pol: null, seed: null, tick: 0, playing: false, speed: 1, lock: null, probView: 'joint', scale: 'log',
  critView: 'qmin', showQ: true, smooth: 25 };

async function fetchJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); }
async function fetchBin(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.arrayBuffer(); }

const EP = {
  mask(i) { return this.masks.subarray(i * NA, (i + 1) * NA); },
  pUsed(i) { return this.learned ? Float64Array.from(this.rec.subarray(i * 2 * NA, i * 2 * NA + NA)) : onehot(this.meta.decisions.action[i]); },
  resid(i) { return this.rec.subarray(i * 2 * NA + NA, (i + 1) * 2 * NA); },
  input(i) { return this.inputs.subarray(i * 408, (i + 1) * 408); },
};
async function loadEpisode(eid) {
  if (App.eps.has(eid)) return App.eps.get(eid);
  return _loadEpisodeInto(eid, Object.create(EP));
}
async function _loadEpisodeInto(eid, proto) {
  const base = `data/episodes/${eid}`;
  const [meta, masks, rec, inputs, frames] = await Promise.all([fetchJSON(`${base}/meta.json`), fetchBin(`${base}/masks.bin`),
    fetchBin(`${base}/rec.bin`), fetchBin(`${base}/inputs.bin`), fetchBin(`${base}/frames.bin`)]);
  const E = Object.assign(proto, { id: eid, meta, n: meta.n_decisions, T: meta.t_end, masks: new Uint8Array(masks),
    rec: new Float32Array(rec), inputs: new Float32Array(inputs), frames, bitmaps: new Map(), probes: new Map(), netimg: null });
  derive(E);
  if (E.learned) await loadProbe(E, meta.policy.id);
  App.eps.set(eid, E);
  if (App.eps.size > 6) { const k = App.eps.keys().next().value; if (k !== eid && (!S.lock || S.lock.eid !== k)) App.eps.delete(k); }
  return E;
}

function derive(E) {
  const meta = E.meta, T = E.T, D = meta.decisions;
  E.decAt = new Int32Array(T + 1).fill(-1);
  E.lastDecAt = new Int32Array(T + 1).fill(-1);
  D.t.forEach((t, i) => { E.decAt[t] = i; });
  let last = -1;
  for (let t = 0; t <= T; t++) { if (E.decAt[t] >= 0) last = E.decAt[t]; E.lastDecAt[t] = last; }
  E.reqAt = new Int32Array(T + 1).fill(-1);
  E.lastReqAt = new Int32Array(T + 1).fill(-1);
  meta.requests.forEach((q, j) => {
    for (let t = q.obs_t; t <= Math.min(T, q.handoff_t); t++) {
      if (t > q.obs_t && t < q.handoff_t) E.reqAt[t] = j;
    }
    for (let t = q.obs_t; t <= T; t++) E.lastReqAt[t] = j;
  });
  const tk = meta.ticks;
  E.handoffs = []; E.chunkSpans = [];
  let c0 = tk.chunk[0], s0 = 0;
  for (let t = 1; t < tk.t.length; t++) {
    if (tk.chunk[t] !== tk.chunk[t - 1]) { E.handoffs.push({ t, chunk: tk.chunk[t] }); E.chunkSpans.push({ chunk: c0, t0: s0, t1: t }); c0 = tk.chunk[t]; s0 = t; }
  }
  E.chunkSpans.push({ chunk: c0, t0: s0, t1: tk.t.length });
  E.chunkOrigin = { 1: 0 }; E.chunkReq = {};
  meta.requests.forEach((q, j) => { if (!q.abandoned) { E.chunkOrigin[q.chunk_at_launch + 1] = q.obs_t; E.chunkReq[q.chunk_at_launch + 1] = j; } });
  const rw = tk.reward;
  E.cumRet = []; let c = 0; rw.forEach((r) => { c += r; E.cumRet.push(c); });
  E.rtg = new Array(rw.length).fill(0); let g = 0;
  for (let t = rw.length - 1; t >= 0; t--) { g = rw[t] + GAMMA * g; E.rtg[t] = g; }
  E.learned = meta.policy.kind === 'learned';
  E.strength = E.learned ? meta.policy.prior_strength : null;
  E.stats = [];
  for (let i = 0; i < E.n; i++) E.stats.push(distStats(E.pUsed(i), E.mask(i)));
}

async function loadProbe(E, ckId) {
  if (E.probes.has(ckId)) return E.probes.get(ckId);
  const arr = new Float32Array(await fetchBin(`data/episodes/${E.id}/probes/${ckId}.bin`));
  if (arr.length !== E.n * 3 * NA) throw new Error(`probe size mismatch ${E.id}/${ckId}`);
  E.probes.set(ckId, arr);
  return arr;
}
function probeRow(arr, i) { const o = i * 3 * NA; return { res: arr.subarray(o, o + NA), q1: arr.subarray(o + NA, o + 2 * NA), q2: arr.subarray(o + 2 * NA, o + 3 * NA) }; }

async function loadNetimg(E) {
  if (!E.netimg) E.netimg = new Uint8Array(await fetchBin(`data/episodes/${E.id}/netimg.bin`));
  return E.netimg;
}

/* ------------------------------------------------------------------ frames */
function frameBitmap(E, t) {
  if (E.bitmaps.has(t)) return E.bitmaps.get(t);
  const o = E.meta.frame_offsets;
  const blob = new Blob([E.frames.slice(o[t], o[t + 1])], { type: 'image/jpeg' });
  const pr = createImageBitmap(blob);
  E.bitmaps.set(t, pr);
  if (E.bitmaps.size > 160) { const k = E.bitmaps.keys().next().value; const old = E.bitmaps.get(k); E.bitmaps.delete(k); old.then((b) => b.close && b.close()); }
  return pr;
}

let _drawToken = 0;
async function drawVideo() {
  const E = curEp(); if (!E) return;
  const t = viewTick();
  const tok = ++_drawToken;
  const bm = await frameBitmap(E, t);
  if (tok !== _drawToken) return;
  const cv = $('video'), g = cv.getContext('2d');
  g.drawImage(bm, 0, 0, cv.width, cv.height);
  for (let k = 1; k <= 20 && t + k <= E.T; k++) frameBitmap(E, t + k);
  const dec = E.decAt[t] >= 0;
  $('videoTag').textContent = `${E.meta.policy.id === 'baseline' ? 'Fixed baseline' : 'Checkpoint ' + fmtSteps(E.meta.policy.train_steps)} · seed ${E.meta.seed} · frame o_${t} (before action ${t})${dec ? ' · decision tick' : ''}`;
}

/* ------------------------------------------------------------------ state helpers */
const curEid = () => (S.lock ? S.lock.eid : `${S.pol}_s${S.seed}`);
const curEp = () => App.eps.get(curEid());
const viewTick = () => (S.lock ? App.eps.get(S.lock.eid).meta.decisions.t[S.lock.i] : S.tick);
const polById = (id) => App.policies.find((p) => p.id === id);

function writeHash() {
  const h = new URLSearchParams({ pol: S.pol, seed: S.seed, t: viewTick() });
  if (S.lock) { h.set('lock', `${S.lock.eid}:${S.lock.i}`); h.set('probe', S.lock.probe); }
  history.replaceState(null, '', '#' + h.toString());
}
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  return { pol: h.get('pol'), seed: h.get('seed') ? +h.get('seed') : null, t: h.get('t') ? +h.get('t') : 0, lock: h.get('lock'), probe: h.get('probe') };
}

/* ------------------------------------------------------------------ checkpoint slider */
function ckSliderDraw() {
  const el = $('ckSlider');
  let cv = el.querySelector('canvas');
  if (!cv) {
    cv = document.createElement('canvas'); el.appendChild(cv);
    cv.addEventListener('click', (e) => { const id = ckAtX(e); if (id) selectPolicy(id); });
    cv.addEventListener('mousemove', (e) => {
      const id = ckAtX(e); const c = id && App.ckById[id];
      showTip(c ? `<b>${fmtSteps(c.train_steps)}</b> training env steps<br>${fmtSteps(c.grad_steps)} gradient updates<br>prior strength ${c.prior_strength.toFixed(3)}<br>held-out eval ${c.eval_success}/${c.eval_episodes}` : null, e);
    });
    cv.addEventListener('mouseleave', () => showTip(null));
  }
  const dpr = window.devicePixelRatio || 1, W = el.clientWidth, H = 54;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = '100%'; cv.style.height = H + 'px';
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const L = 14, R = W - 14, maxS = 310000, X = (s) => L + s / maxS * (R - L);
  App._ckX = X;
  g.strokeStyle = '#a3a29c'; g.lineWidth = 2; g.beginPath(); g.moveTo(L, 22); g.lineTo(R, 22); g.stroke(); g.lineWidth = 1;
  g.fillStyle = '#5b5a55'; g.font = '10px system-ui'; g.textAlign = 'center';
  for (let s = 0; s <= 300000; s += 50000) { g.fillText(s === 0 ? '0' : s / 1000 + 'k', X(s), 50); g.fillRect(X(s), 26, 1, 4); }
  g.textAlign = 'left'; g.fillText('Training environment steps', L, 10);
  // annotations
  [[5000, 'warm-up end'], [50000, 'prior removed']].forEach(([s, lab]) => { g.fillStyle = '#b07a00'; g.fillRect(X(s), 14, 1, 16); g.fillText(lab, X(s) + 3, 40); });
  const sel = S.lock ? S.lock.probe : S.pol;
  App.idx.checkpoints.forEach((c) => {
    const x = X(c.train_steps), on = c.id === sel;
    g.beginPath(); g.arc(x, 22, on ? 7 : 4.5, 0, 7);
    g.fillStyle = on ? (S.lock ? COL.accent2 : COL.accent) : '#ffffff'; g.fill();
    g.strokeStyle = on ? '#141413' : '#5b5a55'; g.stroke();
  });
  $('btnBaseline').classList.toggle('on', sel === 'baseline');
}
function ckAtX(e) {
  const r = e.target.getBoundingClientRect(), x = e.clientX - r.left;
  let best = null, bd = 1e9;
  App.idx.checkpoints.forEach((c) => { const d = Math.abs(App._ckX(c.train_steps) - x); if (d < bd) { bd = d; best = c.id; } });
  return bd < 30 ? best : null;
}
function ckReadout() {
  const sel = S.lock ? S.lock.probe : S.pol;
  const c = App.ckById[sel];
  $('ckTitle').textContent = S.lock ? 'Probed checkpoint' : 'Checkpoint (rollout policy)';
  if (sel === 'baseline') {
    const r = App.learning ? App.learning.reference_eval[0] : null;
    $('ckRead').innerHTML = `<b>Fixed baseline schedule</b> (rule, not a checkpoint): KEEP while q &gt; 10, REPLAN(R=10, N=10) at q = 10.<br>` +
      (r ? `Held-out eval ${r.n_success}/${r.n_episodes}` : '');
  } else {
    $('ckRead').innerHTML = `<b>${fmtSteps(c.train_steps)}</b> training environment steps · <b>${fmtSteps(c.grad_steps)}</b> gradient updates<br>` +
      `training-time prior strength ${c.prior_strength.toFixed(3)} · held-out eval ${c.eval_success}/${c.eval_episodes} (deterministic)<br>` +
      `<span class="note">file ckpt/learner_${String(c.train_steps).padStart(7, '0')}.msgpack · sha256 ${c.sha256.slice(0, 12)}…</span>`;
  }
}

/* ------------------------------------------------------------------ selection */
async function selectPolicy(id) {
  if (S.lock) { S.lock.probe = id; await ensureLockProbes(); renderAll(); return; }
  const keep = S.tick;
  S.pol = id;
  pause();
  await loadEpisode(curEid());
  S.tick = Math.min(keep, curEp().T);
  App._switchNote = true;
  renderAll(true);
}
async function selectSeed(seed) {
  if (S.lock) unlock(false);
  S.seed = seed; pause();
  await loadEpisode(curEid());
  S.tick = Math.min(S.tick, curEp().T);
  App._switchNote = true;
  renderAll(true);
}
function setTick(t, opts = {}) {
  const E = curEp(); if (!E) return;
  if (S.lock) {
    if (!opts.relock) return;
  }
  S.tick = Math.max(0, Math.min(E.T, Math.round(t)));
  if (!opts.fromPlay) pause();
  renderTick();
}
function seekFromChart(x) {
  if (S.lock) { // moving the cursor in locked mode re-locks at the nearest decision tick of the source episode
    const E = curEp(); const t = Math.round(x); const i = nearestDecision(E, t);
    if (i >= 0) { S.lock.i = i; renderAll(); }
    return;
  }
  pause(); setTick(x);
}
function nearestDecision(E, t) {
  const ts = E.meta.decisions.t; let best = -1, bd = 1e9;
  ts.forEach((u, i) => { const d = Math.abs(u - t); if (d < bd || (d === bd && u <= t)) { bd = d; best = i; } });
  return best;
}
function stepDecision(dir) {
  const E = curEp(); const ts = E.meta.decisions.t;
  if (S.lock) {
    const i = S.lock.i + dir;
    if (i >= 0 && i < E.n) { S.lock.i = i; renderAll(); }
    return;
  }
  const t = S.tick;
  let tgt = null;
  if (dir > 0) { for (const u of ts) if (u > t) { tgt = u; break; } }
  else { for (let k = ts.length - 1; k >= 0; k--) if (ts[k] < t) { tgt = ts[k]; break; } }
  if (tgt !== null) setTick(tgt);
}

/* ------------------------------------------------------------------ playback */
let _raf = null, _last = null, _acc = 0;
function play() {
  if (S.lock) return;
  const E = curEp(); if (S.tick >= E.T) S.tick = 0;
  S.playing = true; _last = null; _acc = 0; $('bPlay').textContent = '❚❚ Pause';
  _raf = requestAnimationFrame(loop);
}
function pause() { S.playing = false; if (_raf) cancelAnimationFrame(_raf); _raf = null; $('bPlay').textContent = '▶ Play'; }
function loop(ts) {
  if (!S.playing) return;
  if (_last === null) _last = ts;
  _acc += (ts - _last) / 1000 * HZ * S.speed; _last = ts;
  const adv = Math.floor(_acc);
  if (adv > 0) {
    _acc -= adv;
    const E = curEp();
    S.tick = Math.min(E.T, S.tick + adv);
    renderTick();
    if (S.tick >= E.T) { pause(); return; }
  }
  _raf = requestAnimationFrame(loop);
}

/* ------------------------------------------------------------------ lock mode */
async function lockHere() {
  const E = curEp(); const i = E.decAt[S.tick];
  if (i < 0) return;
  pause();
  S.lock = { eid: E.id, i, probe: S.pol };
  await ensureLockProbes();
  renderAll(true);
}
function unlock(render = true) {
  if (!S.lock) return;
  const E = App.eps.get(S.lock.eid);
  S.tick = E.meta.decisions.t[S.lock.i];
  S.pol = E.meta.policy.id; S.seed = E.meta.seed;
  S.lock = null;
  if (render) renderAll(true);
}
async function ensureLockProbes() {
  const E = App.eps.get(S.lock.eid);
  $('lockInfo').innerHTML = '<span class="note">loading probes for all checkpoints…</span>';
  await Promise.all(App.idx.checkpoints.map((c) => loadProbe(E, c.id)));
  await loadNetimg(E);
}

/* The distribution and critic values displayed for the current view.
   Mode A (playback): the episode's own policy at the decision shown (recorded probabilities, own checkpoint probes).
   Mode B (locked): the probed checkpoint applied to the frozen decision state of the source episode. */
function currentView() {
  const E = curEp(); if (!E) return null;
  const t = viewTick();
  const i = S.lock ? S.lock.i : (E.decAt[t] >= 0 ? E.decAt[t] : E.lastDecAt[t]);
  if (i < 0) return null;
  const D = E.meta.decisions, mask = E.mask(i), ref = D.ref[i];
  const v = { E, i, t: D.t[i], mask, ref, srcAct: D.action[i], stale: !S.lock && E.decAt[t] < 0, q1: null, q2: null };
  const pid = S.lock ? S.lock.probe : E.meta.policy.id;
  v.probeId = pid;
  if (pid === 'baseline') {
    v.kind = 'rule'; v.pOn = onehot(ref); v.pOff = null; v.strength = null; v.res = null;
    v.selAct = S.lock ? ref : D.action[i];
  } else {
    const c = App.ckById[pid];
    v.kind = 'learned'; v.strength = c.prior_strength;
    const pr = E.probes.get(pid);
    const row = pr ? probeRow(pr, i) : null;
    if (!S.lock) { v.pOn = E.pUsed(i); v.res = E.resid(i); }
    else { v.res = row.res; v.pOn = probsFrom(row.res, mask, ref, v.strength); }
    v.pOff = probsFrom(v.res, mask, ref, 0);
    if (row) { v.q1 = row.q1; v.q2 = row.q2; }
    v.selAct = S.lock ? argmaxOf(v.pOn) : D.action[i];
  }
  v.stOn = distStats(v.pOn, mask);
  v.stOff = v.pOff ? distStats(v.pOff, mask) : null;
  return v;
}
const argmaxOf = (p) => { let a = 0; for (let k = 1; k < NA; k++) if (p[k] > p[a]) a = k; return a; };
const qmin = (v, a) => (v.q1 ? Math.min(v.q1[a], v.q2[a]) : null);

/* ------------------------------------------------------------------ heatmaps */
const VIR = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37]];
function seqColor(u) { u = Math.max(0, Math.min(1, u)); const x = u * (VIR.length - 1), k = Math.min(VIR.length - 2, Math.floor(x)), f = x - k; const c = VIR[k].map((v, j) => Math.round(v + (VIR[k + 1][j] - v) * f)); return `rgb(${c})`; }
function divColor(u) { // u in [-1, 1]; blue (negative) - white - orange (positive)
  u = Math.max(-1, Math.min(1, u));
  const a = u < 0 ? [42, 99, 196] : [217, 99, 43], w = [247, 247, 245], f = Math.abs(u);
  return `rgb(${w.map((v, j) => Math.round(v + (a[j] - v) * f))})`;
}
let _hatch = null;
function hatchPattern(g) {
  if (_hatch) return _hatch;
  const c = document.createElement('canvas'); c.width = c.height = 6;
  const h = c.getContext('2d'); h.fillStyle = '#efeee9'; h.fillRect(0, 0, 6, 6); h.strokeStyle = '#c9c7bf'; h.beginPath(); h.moveTo(0, 6); h.lineTo(6, 0); h.stroke();
  _hatch = g.createPattern(c, 'repeat'); return _hatch;
}

/* spec: {title, vals Float64Array(201) | null, mask, mode:'prob'|'seq'|'div', dom:[lo,hi], log, cond, marks:{sel,ref,arg}, stale, note} */
function drawHeat(el, spec) {
  let cv = el.querySelector('canvas');
  if (!cv) {
    el.innerHTML = '';
    cv = document.createElement('canvas'); el.appendChild(cv);
    cv.addEventListener('mousemove', (e) => heatHover(el, e));
    cv.addEventListener('mouseleave', () => showTip(null));
  }
  el._spec = spec;
  if (!spec) { el.style.display = 'none'; return; }
  el.style.display = '';
  const dpr = window.devicePixelRatio || 1, W = Math.max(320, el.clientWidth);
  const lw = 62, kw = 40, gap = 8, top = 20, cols = RMAX - RMIN_SHOWN + 1, rows = NV.length;
  const cw = Math.max(9, Math.min(22, Math.floor((W - lw - kw - gap - 8) / cols))), ch = Math.max(12, Math.min(18, cw));
  const H = top + rows * ch + 16 + 30;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + 'px';
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const x0 = lw + kw + gap;
  el._geo = { lw, kw, gap, top, cw, ch, x0, rows, cols };
  g.font = '600 11.5px system-ui'; g.fillStyle = '#141413'; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
  g.fillText(spec.title, 0, 13);
  if (spec.stale) { g.fillStyle = COL.accent2; g.font = '600 11px system-ui'; g.textAlign = 'right'; g.fillText(spec.staleText || 'from last decision (stale)', W - 4, 13); g.textAlign = 'left'; }
  const color = (v) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return '#ffffff';
    if (spec.mode === 'div') return divColor(v / Math.max(1e-12, spec.dom[1]));
    let u;
    if (spec.log) { const lo = Math.log10(spec.dom[0]), hi = Math.log10(spec.dom[1]); u = (Math.log10(Math.max(v, spec.dom[0])) - lo) / (hi - lo); }
    else u = (v - spec.dom[0]) / Math.max(1e-12, spec.dom[1] - spec.dom[0]);
    return seqColor(u);
  };
  g.globalAlpha = spec.stale ? 0.55 : 1;
  const cellXY = (a) => { if (a === 0) return [lw, top, kw - 2, rows * ch]; const d = decA(a); return [x0 + (d.r - RMIN_SHOWN) * cw, top + (10 - d.n) * ch, cw, ch]; };
  el._cellXY = cellXY;
  const vals = spec.vals;
  for (let a = 0; a < NA; a++) {
    const d = decA(a);
    if (!d.keep && d.r < RMIN_SHOWN) continue;
    const [x, y, w, h] = cellXY(a);
    if (!spec.mask[a]) { g.fillStyle = hatchPattern(g); g.fillRect(x, y, w - 1, h - 1); continue; }
    if (a === 0 && spec.cond) { g.fillStyle = '#ffffff'; g.fillRect(x, y, w, h); g.strokeStyle = '#c9c7bf'; g.strokeRect(x + .5, y + .5, w - 1, h - 1); continue; }
    g.fillStyle = vals ? color(vals[a]) : '#ffffff';
    g.fillRect(x, y, w - 1, h - 1);
  }
  g.globalAlpha = 1;
  // labels
  g.fillStyle = '#5b5a55'; g.font = '10px system-ui'; g.textAlign = 'right'; g.textBaseline = 'middle';
  NV.forEach((n) => g.fillText(`N=${n} (d=${DELAY[n]})`, lw - 4, top + (10 - n) * ch + ch / 2));
  g.textAlign = 'center';
  g.save(); g.translate(lw + kw / 2 - 1, top + rows * ch / 2); g.rotate(-Math.PI / 2); g.fillStyle = spec.mask[0] ? '#ffffff' : '#5b5a55';
  if (spec.mask[0] && vals && !spec.cond) { const u = spec.mode === 'prob' ? vals[0] : 0.5; g.fillStyle = (spec.mode !== 'div' && u > 0.55 && spec.log) || (spec.mode === 'prob' && !spec.log && vals[0] > 0.6) ? '#141413' : '#ffffff'; }
  g.font = '600 10.5px system-ui';
  g.fillText(spec.cond && spec.mask[0] ? 'KEEP (excluded)' : spec.mask[0] ? 'KEEP' : 'KEEP infeasible', 0, 0); g.restore();
  g.fillStyle = '#5b5a55'; g.font = '9.5px system-ui'; g.textBaseline = 'top';
  for (let r = RMIN_SHOWN; r <= RMAX; r++) if (cw >= 14 || r % 2 === 0) g.fillText(String(r), x0 + (r - RMIN_SHOWN) * cw + cw / 2, top + rows * ch + 2);
  g.textAlign = 'left'; g.fillText('R = old actions executed after the observation (handoff R steps later) →', x0, top + rows * ch + 14);
  // marks
  const mark = (a, style) => {
    if (a === null || a === undefined || a < 0) return;
    const d = decA(a); if (!d.keep && d.r < RMIN_SHOWN) return;
    const [x, y, w, h] = cellXY(a);
    g.save(); g.lineWidth = style.w; g.strokeStyle = style.c; if (style.dash) g.setLineDash(style.dash);
    g.strokeRect(x + 1, y + 1, w - 3, h - 3); g.restore();
  };
  if (spec.marks) { mark(spec.marks.ref, { c: '#ffffff', w: 2, dash: [3, 2] }); mark(spec.marks.ref, { c: '#141413', w: 1, dash: [3, 2] }); mark(spec.marks.sel, { c: COL.accent2, w: 2.5 }); if (spec.marks.arg !== spec.marks.sel) mark(spec.marks.arg, { c: '#e0218a', w: 1.5 }); }
  // colour bar
  const by = top + rows * ch + 30, bx = x0, bw = Math.min(260, cols * cw);
  for (let k = 0; k < bw; k++) {
    const u = k / (bw - 1);
    let v;
    if (spec.mode === 'div') v = (u * 2 - 1) * spec.dom[1];
    else if (spec.log) v = Math.pow(10, Math.log10(spec.dom[0]) + u * (Math.log10(spec.dom[1]) - Math.log10(spec.dom[0])));
    else v = spec.dom[0] + u * (spec.dom[1] - spec.dom[0]);
    g.fillStyle = color(v); g.fillRect(bx + k, by, 1, 8);
  }
  g.fillStyle = '#5b5a55'; g.font = '9.5px system-ui'; g.textBaseline = 'top'; g.textAlign = 'left';
  const lo = spec.mode === 'div' ? -spec.dom[1] : spec.dom[0], hi = spec.dom[1];
  g.fillText(fmtNum(lo) + (spec.log ? ' (and below)' : ''), bx, by + 9); g.textAlign = 'right'; g.fillText(fmtNum(hi), bx + bw, by + 9);
  g.textAlign = 'left'; g.fillText(spec.note || '', bx + bw + 10, by);
  g.fillStyle = hatchPattern(g); g.fillRect(4, by, 12, 10); g.fillStyle = '#5b5a55'; g.fillText('infeasible', 18, by);
}
function heatHover(el, e) {
  const sp = el._spec, geo = el._geo; if (!sp || !geo) return;
  const r = e.target.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  let a = null;
  if (x >= geo.lw && x < geo.lw + geo.kw && y >= geo.top && y < geo.top + geo.rows * geo.ch) a = 0;
  else if (x >= geo.x0 && y >= geo.top && y < geo.top + geo.rows * geo.ch) {
    const r_ = RMIN_SHOWN + Math.floor((x - geo.x0) / geo.cw), n = 10 - Math.floor((y - geo.top) / geo.ch);
    if (r_ <= RMAX && n >= 3 && n <= 10) a = enc(r_, n);
  }
  if (a === null) { showTip(null); return; }
  const v = App._view;
  let html = `<b>${actName(a)}</b>`;
  if (!sp.mask[a]) {
    const d = decA(a);
    html += d.keep ? '<br>infeasible: KEEP needs q − 1 ≥ 6 and more than one tick left' : `<br>infeasible: needs d(N)=${DELAY[d.n]} ≤ R ≤ q=${v ? v.E.meta.decisions.q[v.i] : '?'}`;
  } else if (v) {
    html += `<br>P incl. prior: ${pct(v.pOn[a])}`;
    if (v.pOff) html += `<br>P network only: ${pct(v.pOff[a])}`;
    if (sp.cond && a > 0) html += `<br>P(· | REPLAN): ${pct(v.pOn[a] / Math.max(1e-300, v.stOn.rep))}`;
    if (v.q1) html += `<br>min(Q1,Q2) ${fmtNum(qmin(v, a))} · |Q1−Q2| ${fmtNum(Math.abs(v.q1[a] - v.q2[a]))}<br>Q adv. vs baseline ${fmtNum(qmin(v, a) - qmin(v, v.ref))}`;
    if (a === v.ref) html += '<br><i>fixed-baseline action</i>';
    if (a === v.selAct) html += `<br><i>${S.lock ? 'argmax of probed policy' : 'selected action'}</i>`;
  }
  showTip(html, e);
}

function drawBars(el, spec) {
  let cv = el.querySelector('canvas');
  if (!cv) { cv = document.createElement('canvas'); el.appendChild(cv); }
  const dpr = window.devicePixelRatio || 1, W = Math.max(120, el.clientWidth), H = 86;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = '100%'; cv.style.height = H + 'px';
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  g.font = '600 10.5px system-ui'; g.fillStyle = '#141413'; g.textAlign = 'left'; g.fillText(spec.title, 0, 11);
  const keys = spec.keys, n = keys.length, bw = (W - 6) / n, top = 18, ph = H - top - 16;
  const mx = Math.max(1e-12, spec.max === undefined ? Math.max(...keys.map((k) => spec.vals[k])) : spec.max);
  keys.forEach((k, j) => {
    const v = spec.vals[k], h = Math.max(0, v / mx * ph);
    g.fillStyle = k === spec.hl ? COL.accent2 : k === spec.hl2 ? '#141413' : COL.accent;
    g.globalAlpha = spec.stale ? 0.5 : 1;
    g.fillRect(3 + j * bw + 1, top + ph - h, bw - 2, h); g.globalAlpha = 1;
    g.fillStyle = '#5b5a55'; g.font = '9px system-ui'; g.textAlign = 'center';
    if (bw >= 12 || j % 2 === 0) g.fillText(String(k), 3 + j * bw + bw / 2, H - 4);
  });
  g.textAlign = 'right'; g.fillStyle = '#5b5a55'; g.font = '9.5px system-ui'; g.fillText(`max ${pct(mx)}`, W - 2, 11);
}

/* ------------------------------------------------------------------ scheduler panel */
function renderSched() {
  const E = curEp(); const t = viewTick(); const v = currentView(); App._view = v;
  const D = E.meta.decisions;
  // status
  const sb = $('statusBox');
  const reqIdx = S.lock ? -1 : E.reqAt[t];
  if (S.lock) {
    sb.className = 'statusbox dec';
    const pc = v.probeId === 'baseline' ? 'the fixed baseline rule' : `checkpoint ${fmtSteps(App.ckById[v.probeId].train_steps)}`;
    sb.innerHTML = `<b>Locked state</b> · probing <b>${pc}</b> on the frozen input of decision #${v.i} (tick ${v.t}).<br>` +
      `Source policy chose <b>${actName(v.srcAct)}</b>; probed ${v.kind === 'rule' ? 'rule' : 'policy argmax'}: <b>${actName(v.selAct)}</b>; fixed-baseline action: ${actName(v.ref)}.`;
  } else if (E.decAt[t] >= 0) {
    const i = E.decAt[t];
    sb.className = 'statusbox dec';
    const dep = D.action[i] !== D.ref[i];
    sb.innerHTML = `<b>Scheduler decision</b> at tick ${t} (decision #${i}, q = ${D.q[i]}, p = ${D.p[i]}, ${D.nfeas[i]} feasible actions).<br>` +
      `Selected: <b>${actName(D.action[i])}</b> · fixed baseline would choose: ${actName(D.ref[i])}${dep ? ' <span class="warn">(departure)</span>' : ''}`;
  } else if (reqIdx >= 0) {
    const q = E.meta.requests[reqIdx];
    sb.className = 'statusbox fly';
    const toReady = Math.max(0, q.ready_t - t), toHand = q.handoff_t - t;
    sb.innerHTML = `<b>No scheduler decision — request in flight.</b> Active request: N = ${q.n}, selected R = ${q.r} (observation at tick ${q.obs_t}).<br>` +
      `Elapsed ${t - q.obs_t} steps · ${toReady > 0 ? `logically ready in ${toReady}` : `ready since tick ${q.ready_t} (waiting for handoff)`} · ` +
      `${q.abandoned ? `<span class="warn">planned handoff at ${q.handoff_t} is after the episode end (abandoned)</span>` : `handoff in <b>${toHand}</b> control steps (tick ${q.handoff_t})`}.<br>` +
      `<span class="note">Selected R counts old actions executed after the observation; the handoff countdown is what remains of it.</span>`;
  } else {
    sb.className = 'statusbox none';
    sb.innerHTML = t >= E.T ? `<b>Episode ended</b> at tick ${E.T} (${E.meta.success ? 'success' : 'timeout / no success'}). Showing the last decision.` : 'No decision at this tick.';
  }
  $('bLock').disabled = !S.lock && E.decAt[t] < 0;
  $('bLock').classList.toggle('on', !!S.lock);
  $('bLock').textContent = S.lock ? 'Unlock (return to playback)' : (E.decAt[t] >= 0 ? 'Lock state across checkpoints' : 'Lock (only at decision ticks)');
  if (!v) { ['heatA', 'heatB', 'heatC'].forEach((k) => drawHeat($(k), null)); $('pkeepBox').innerHTML = ''; return; }
  const st = v.stOn, stale = v.stale, staleText = `from last decision at tick ${v.t} (stale)`;
  // P(KEEP)
  $('pkeepBox').innerHTML = `<span class="ctl-title">P(KEEP)</span><span class="big">${v.mask[0] ? pct(st.pKeep) : 'n/a'}</span>` +
    `<div class="bar"><div style="width:${(100 * st.pKeep).toFixed(2)}%"></div></div>` +
    `<span class="note">P(any REPLAN) = ${pct(st.rep)}${v.kind === 'rule' ? ' · deterministic rule' : ''}${v.pOff ? ` · network only: P(KEEP) = ${pct(v.stOff.pKeep)}` : ''}${stale ? ` · <span class="warn">${staleText}</span>` : ''}${v.mask[0] ? '' : ' · KEEP infeasible here (q − 1 < 6)'}</span>`;
  const cond = S.probView === 'cond', log = S.scale === 'log';
  const mkVals = (p, s) => { if (!cond) return p; const o = new Float64Array(NA); for (let a = 1; a < NA; a++) o[a] = p[a] / Math.max(1e-300, s.rep); return o; };
  const dom = log ? [1e-4, 1] : [0, 1];
  const condNote = cond ? (st.negligible ? 'REPLAN mass < 1e-6: conditional view numerically unreliable' : 'conditional on REPLAN; sums to 1 over REPLAN cells') : 'unconditional; REPLAN cells sum to 1 − P(KEEP)';
  const sname = v.kind === 'rule' ? 'Fixed baseline rule (deterministic, no network)' : `Policy incl. training-time prior (strength ${v.strength.toFixed(3)})`;
  drawHeat($('heatA'), { title: sname + (S.lock ? '' : ' — as used in this rollout'), vals: mkVals(v.pOn, st), mask: v.mask, mode: 'prob', dom, log, cond, stale, staleText,
    marks: { sel: v.selAct, ref: v.ref, arg: st.argmax }, note: condNote });
  if (v.pOff && v.strength > 0) {
    drawHeat($('heatB'), { title: 'Network only / prior disabled (same residual logits, strength 0)', vals: mkVals(v.pOff, v.stOff), mask: v.mask, mode: 'prob', dom, log, cond, stale, staleText,
      marks: { sel: v.selAct, ref: v.ref, arg: v.stOff.argmax }, note: cond ? '' : 'unconditional' });
  } else {
    drawHeat($('heatB'), null);
  }
  const hb = $('heatB');
  if (v.pOff && v.strength === 0) { hb.style.display = ''; hb.innerHTML = '<div class="note">Network only / prior disabled: identical to the distribution above — this checkpoint\'s training-time prior strength is 0 (from 50k steps on).</div>'; }
  if (v.kind === 'rule') { hb.style.display = ''; hb.innerHTML = '<div class="note">The fixed baseline is a rule; it has no network or prior. ' + (S.lock ? '' : 'Lock this state and select a checkpoint to probe learned preferences on it.') + '</div>'; }
  // marginals
  const mN = {}, mR = {};
  NV.forEach((n) => { mN[n] = cond ? st.margN[n] / Math.max(1e-300, st.rep) : st.margN[n]; });
  for (let r = RMIN_SHOWN; r <= RMAX; r++) mR[r] = cond ? st.margR[r] / Math.max(1e-300, st.rep) : st.margR[r];
  const selD = decA(v.selAct), refD = decA(v.ref);
  drawBars($('margN'), { title: `${cond ? 'P(N | REPLAN)' : 'P(REPLAN with N)'} · E[N|REPLAN] ${st.EN === null ? '–' : st.EN.toFixed(2)}`, keys: NV, vals: mN, hl: selD.keep ? null : selD.n, hl2: refD.keep ? null : refD.n, stale });
  drawBars($('margR'), { title: `${cond ? 'P(R | REPLAN)' : 'P(REPLAN with R)'} · E[R|REPLAN] ${st.ER === null ? '–' : st.ER.toFixed(2)} · entropy ${st.ent.toFixed(3)} nats`, keys: Array.from({ length: RMAX - RMIN_SHOWN + 1 }, (_, k) => k + RMIN_SHOWN), vals: mR, hl: selD.keep ? null : selD.r, hl2: refD.keep ? null : refD.r, stale });
  // top candidates
  const order = Array.from({ length: NA }, (_, a) => a).filter((a) => v.mask[a]).sort((a, b) => v.pOn[b] - v.pOn[a]);
  const show = order.slice(0, 6);
  [v.selAct, v.ref].forEach((a) => { if (a >= 0 && !show.includes(a) && v.mask[a]) show.push(a); });
  const qref = qmin(v, v.ref);
  let h = `<table class="cand"><tr><th>top candidates (full joint incl. KEEP)</th><th>P incl. prior</th>${v.pOff ? '<th>P network only</th>' : ''}${v.q1 ? '<th>min Q</th><th>adv. vs baseline</th><th>|Q1−Q2|</th>' : ''}</tr>`;
  show.forEach((a) => {
    const tags = [a === v.selAct ? (S.lock ? 'argmax' : 'selected') : '', a === v.ref ? 'baseline' : ''].filter(Boolean).join(', ');
    h += `<tr class="${a === v.selAct ? 'sel' : ''}"><td>${actName(a)}${tags ? ` <span class="badge">${tags}</span>` : ''}</td><td>${pct(v.pOn[a])}</td>${v.pOff ? `<td>${pct(v.pOff[a])}</td>` : ''}` +
      (v.q1 ? `<td>${fmtNum(qmin(v, a))}</td><td>${fmtNum(qmin(v, a) - qref)}</td><td>${fmtNum(Math.abs(v.q1[a] - v.q2[a]))}</td>` : '') + '</tr>';
  });
  h += '</table>';
  if (!S.lock && E.learned) h += `<div class="note">Winner = argmax of the full joint distribution (KEEP included). Deterministic evaluation: the selected action is that argmax${D.action[v.i] === st.argmax ? ' ✓' : ' <span class="warn">(mismatch!)</span>'}.</div>`;
  $('topCands').innerHTML = h;
  // logits
  if (v.res) {
    const big = priorBig(v.mask, v.strength);
    let lt = `<table class="cand"><tr><th>action</th><th>residual logit</th><th>prior contribution</th><th>total logit</th><th>P incl. prior</th></tr>`;
    const rows = [...new Set([0, ...order.slice(0, 10), v.selAct, v.ref])].filter((a) => v.mask[a]);
    rows.forEach((a) => {
      const pc = a === v.ref ? big : 0;
      lt += `<tr><td>${actName(a)}</td><td>${fmtNum(v.res[a])}</td><td>${fmtNum(pc)}</td><td>${fmtNum(v.res[a] + pc)}</td><td>${pct(v.pOn[a])}</td></tr>`;
    });
    lt += `</table><div class="note">Prior: the fixed-baseline action gets strength × log(p0 (M−1)/(1−p0)) with p0 = 0.999, M = ${v.mask.reduce((s, x) => s + x, 0)} feasible actions; other feasible logits get 0. Probabilities are the softmax over feasible actions.</div>`;
    $('logitTable').innerHTML = lt;
  } else $('logitTable').innerHTML = '<div class="note">No residual logits: fixed-baseline rule.</div>';
  // critic heatmap
  renderCritic(v);
}

function criticWarning(ckId) {
  // From the run's own updates.jsonl: the last logged update row at or before this checkpoint.
  const c = App.ckById[ckId], U = App.learning && App.learning.updates;
  if (!c || !U) return '';
  let k = -1;
  for (let j = 0; j < U.steps.length; j++) if (U.steps[j] <= c.train_steps) k = j;
  if (k < 0 || !(U.q_taken[k] > 1)) return '';
  return `<div class="warn">Critic divergence: at this checkpoint the logged mean Q of replay actions is ${fmtNum(U.q_taken[k])} and critic loss ${fmtNum(U.critic_loss[k])} (updates.jsonl, ${fmtSteps(U.steps[k])} steps). The return cannot exceed 1, so these Q values are not meaningful estimates; see Learning diagnostics.</div>`;
}

function renderCritic(v) {
  $('critWarn').innerHTML = v.probeId ? criticWarning(v.probeId) : '';
  const el = $('heatC');
  if (!v.q1) {
    drawHeat(el, null); el.style.display = '';
    el.innerHTML = `<div class="note">No critic for the fixed-baseline rule. ${S.lock ? 'Select a checkpoint to see its critics on this frozen state.' : 'Lock this state and select a checkpoint to see learned critics on it.'}</div>`;
    return;
  }
  const mode = S.critView, mask = v.mask, E = v.E;
  // consistent colour ranges: Mode A over every decision of this episode (own checkpoint); Mode B over all probed checkpoints for the locked state
  const key = `${mode}|${S.lock ? `L|${E.id}|${v.i}` : `A|${E.id}`}`;
  if (!App._qdom || App._qdom.key !== key) App._qdom = { key, dom: qDomain(v, mode) };
  const dom = App._qdom.dom;
  let vals = new Float64Array(NA), spec;
  if (mode === 'prob') {
    spec = { title: 'Actor probability (incl. prior), log scale', vals: v.pOn, mode: 'prob', dom: [1e-4, 1], log: true };
  } else {
    const qr = qmin(v, v.ref);
    for (let a = 0; a < NA; a++) {
      if (!mask[a]) continue;
      const m = Math.min(v.q1[a], v.q2[a]);
      vals[a] = mode === 'qmin' ? m : mode === 'qdis' ? Math.abs(v.q1[a] - v.q2[a]) : m - qr;
    }
    const titles = { qmin: 'min(Q1, Q2) — learned soft return estimate', qdis: '|Q1 − Q2| — critic disagreement', adv: 'min Q(a) − min Q(fixed-baseline action)' };
    spec = { title: titles[mode], vals, mode: mode === 'adv' ? 'div' : 'seq', dom, log: false,
      note: S.lock ? 'colour range fixed across all checkpoints for this locked state' : 'colour range fixed over this episode' };
  }
  spec.mask = mask; spec.marks = { sel: v.selAct, ref: v.ref, arg: argmaxOf(v.pOn) }; spec.stale = v.stale; spec.staleText = `from last decision at tick ${v.t} (stale)`;
  drawHeat(el, spec);
}
function qDomain(v, mode) {
  const E = v.E;
  const arrs = S.lock ? App.idx.checkpoints.map((c) => [E.probes.get(c.id), v.i]) : [[E.probes.get(E.meta.policy.id), null]];
  let lo = Infinity, hi = -Infinity, amax = 0;
  for (const [arr, only] of arrs) {
    if (!arr) continue;
    const idxs = only === null ? Array.from({ length: E.n }, (_, k) => k) : [only];
    for (const i of idxs) {
      const m = E.mask(i), r = probeRow(arr, i), qr = Math.min(r.q1[E.meta.decisions.ref[i]], r.q2[E.meta.decisions.ref[i]]);
      for (let a = 0; a < NA; a++) {
        if (!m[a]) continue;
        const q = Math.min(r.q1[a], r.q2[a]);
        const val = mode === 'qmin' ? q : mode === 'qdis' ? Math.abs(r.q1[a] - r.q2[a]) : q - qr;
        if (mode === 'adv') amax = Math.max(amax, Math.abs(val));
        else { lo = Math.min(lo, val); hi = Math.max(hi, val); }
      }
    }
  }
  if (mode === 'adv') return [0, amax || 1e-6];
  if (mode === 'qdis') return [0, hi > 0 ? hi : 1e-6];
  return [lo, hi > lo ? hi : lo + 1e-6];
}

/* ------------------------------------------------------------------ readouts + lock info */
function renderReadouts() {
  const E = curEp(), t = viewTick(), tk = E.meta.ticks, D = E.meta.decisions;
  const tt = Math.min(t, tk.t.length - 1);
  const chunk = t < tk.t.length ? tk.chunk[t] : tk.chunk[tk.t.length - 1] + (E.handoffs.length && E.handoffs[E.handoffs.length - 1].t === t ? 1 : 0);
  const p = t < tk.t.length ? tk.p[t] : null;
  const q = t < tk.t.length ? tk.qlen_after[t] + (1 - tk.underrun[t]) : null;
  const origin = E.chunkOrigin[chunk];
  const j = S.lock ? -1 : (E.reqAt[t] >= 0 ? E.reqAt[t] : E.lastReqAt[t]);
  const rq = j >= 0 ? E.meta.requests[j] : null;
  const inflight = !S.lock && E.reqAt[t] >= 0;
  const rows = [
    ['control step t', `${t} / ${E.T}`], ['simulated time', `${(t / HZ).toFixed(2)} s`],
    ['executed chunk ID', chunk], ['p (executed from chunk)', p ?? '–'], ['q (queue at step start)', q ?? '–'],
    ['plan observation age', origin !== undefined ? `${t - origin} steps (obs at ${origin})` : '–'],
    [inflight ? 'active request N / R' : 'last request N / R', rq ? `N=${rq.n} / R=${rq.r}` : '–'],
    ['request obs → ready → handoff', rq ? `${rq.obs_t} → ${rq.ready_t} → ${rq.handoff_t}${rq.abandoned ? ' (abandoned)' : ''}` : '–'],
    ['old actions committed before handoff', rq ? rq.r : '–'], ['old actions discarded', rq ? rq.discarded : '–'],
    ['aligned start index of new chunk', rq ? `new[${rq.r} : ${rq.r + 25}]` : '–'], ['old actions executed in chunk', rq ? rq.old_executed : '–'],
    ['reward at this step', t < tk.reward.length ? fmtNum(tk.reward[t]) : '–'], ['cumulative return', fmtNum(t > 0 ? E.cumRet[Math.min(t, E.cumRet.length) - 1] : 0)],
  ];
  if (t < tk.t.length && tk.underrun[t]) rows.push(['underrun', 'YES (queue empty, last action repeated)']);
  $('readouts').innerHTML = rows.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
  $('tickRead').textContent = `step ${t} of ${E.T} · ${(t / HZ).toFixed(2)} s simulated · ${E.decAt[t] >= 0 ? 'decision tick' : (E.reqAt[t] >= 0 ? 'request in flight' : '')}`;
  $('tickRange').max = E.T; $('tickRange').value = t;
  ['bPrev', 'bNext', 'bPlay', 'tickRange', 'speedSel', 'bNearest'].forEach((k) => { $(k).disabled = !!S.lock; });
  $('bNearest').disabled = !!S.lock || E.decAt[t] >= 0;
}

function renderLockInfo() {
  const el = $('lockInfo');
  if (!S.lock) { el.innerHTML = ''; return; }
  const E = App.eps.get(S.lock.eid), i = S.lock.i, D = E.meta.decisions, inp = E.input(i);
  const rt = inp.subarray(403, 408);
  const srcT = D.t[i] - D.plan_age[i];
  const j = D.t.indexOf(srcT);
  el.innerHTML = `<b>Frozen scheduler input</b> (source: ${E.meta.policy.id === 'baseline' ? 'fixed baseline' : 'checkpoint ' + fmtSteps(E.meta.policy.train_steps)} rollout, seed ${E.meta.seed}, decision #${i} at tick ${D.t[i]}; ` +
    `<span class="badge new">new recording</span>). The slider now changes ONLY the actor/critic checkpoint applied to this input.<br>` +
    `q = ${D.q[i]} · p = ${D.p[i]} · chunk ${D.chunk[i]} · plan age ${D.plan_age[i]} · source N = ${D.plan_n[i]} · remaining ${D.ticks_left[i]} steps · ` +
    `${D.nfeas[i]} feasible actions (same mask for every checkpoint) · runtime features [${Array.from(rt).map((x) => x.toFixed(3)).join(', ')}]` +
    `<div class="imgs"><div><canvas id="niNow" width="64" height="64"></canvas><div class="note">img_now (o_${D.t[i]}, 64×64 network input)</div></div>` +
    `<div><canvas id="niSrc" width="64" height="64"></canvas><div class="note">img_src (o_${srcT}, observation of the executing plan)</div></div></div>`;
  const put = (id, k) => {
    if (k < 0 || !E.netimg) return;
    const g = $(id).getContext('2d'), im = g.createImageData(64, 64), src = E.netimg.subarray(k * 12288, (k + 1) * 12288);
    for (let p = 0; p < 4096; p++) { im.data[4 * p] = src[3 * p]; im.data[4 * p + 1] = src[3 * p + 1]; im.data[4 * p + 2] = src[3 * p + 2]; im.data[4 * p + 3] = 255; }
    g.putImageData(im, 0, 0);
  };
  put('niNow', i); put('niSrc', j);
}

/* ------------------------------------------------------------------ banner / episode readout */
function renderBanner() {
  const b = $('modeBanner'), E = curEp();
  const idxE = App.idx.episodes[E.id] || {};
  const match = E.meta.outcome_matches_log;
  const lg = E.meta.logged;
  $('epRead').innerHTML = `this recording: <b>${E.meta.success ? 'success' : 'no success'}</b> at step ${E.T} · return ${fmtNum(E.meta.ret)} · ${E.meta.requests.length} requests · ` +
    (lg ? `logged eval (training time): ${lg.success ? 'success' : 'no success'} at ${lg.t_end} <span class="badge ${match ? 'ok' : 'no'}">${match ? 'matches log' : 'differs from log'}</span>` : '') +
    ` <span class="badge new">new recording</span>`;
  if (S.lock) {
    b.className = 'mode b';
    b.innerHTML = `<b>Mode B — locked state across checkpoints.</b> Source: ${E.meta.policy.id === 'baseline' ? 'fixed-baseline' : 'checkpoint ' + fmtSteps(E.meta.policy.train_steps)} rollout, seed ${E.meta.seed}, tick ${E.meta.decisions.t[S.lock.i]}. ` +
      `Probed: ${S.lock.probe === 'baseline' ? 'fixed baseline rule' : 'checkpoint ' + fmtSteps(App.ckById[S.lock.probe].train_steps)}. These are counterfactual probes of one frozen input, not rollouts. ` +
      `Use the checkpoint slider to change the probed checkpoint; « dec / dec » move the lock to another decision of the source episode.`;
  } else {
    b.className = 'mode a';
    b.innerHTML = `<b>Mode A — checkpoint rollout playback.</b> Newly recorded evaluation rollout of the frozen ${S.pol === 'baseline' ? 'fixed baseline' : 'checkpoint'} on held-out seed ${S.seed} (deterministic argmax, noise stream default_rng([seed, 1])). ` +
      (App._switchNote ? '<span class="warn">Switching checkpoint keeps the seed and control step, but the physical state at this step may differ (different rollout).</span>' : '');
  }
}

/* ------------------------------------------------------------------ timeline */
function renderTimeline() {
  const el = $('timeline'), E = curEp(), t = viewTick();
  let cv = el.querySelector('canvas');
  if (!cv) {
    cv = document.createElement('canvas'); el.appendChild(cv);
    cv.addEventListener('click', (e) => { const x = tlX(e); if (x !== null) seekFromChart(x); });
    cv.addEventListener('mousemove', (e) => { const x = tlX(e); showTip(x === null ? null : tlHover(Math.round(x)), e); });
    cv.addEventListener('mouseleave', () => showTip(null));
    $('tlLegend').innerHTML = [
      ['#d9e3f5', 'executed chunk (ID)'], ['#141413', 'actual handoff'], ['#4a7bd0', 'inference: observation → logically ready (colour = N)'],
      ['repeating-linear-gradient(45deg,#f6c99f 0 3px,#fff 3px 6px)', 'ready, waiting for handoff'], ['#ffffff;border:1.5px solid #d9632b', 'planned handoff (▽)'],
      ['#7a5cc4', 'REPLAN decision'], ['#c9c7bf', 'KEEP decision'], ['#c43a3a', 'underrun'],
    ].map(([c, l]) => `<span><i style="background:${c}"></i>${l}</span>`).join('');
  }
  const dpr = window.devicePixelRatio || 1, W = Math.max(400, el.clientWidth), H = 132, m = { l: 64, r: 12 };
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + 'px';
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const T = E.T, pw = W - m.l - m.r, X = (x) => m.l + x / Math.max(1, T) * pw;
  App._tl = { X, m, pw, T };
  const rows = { chunk: [6, 22], req: [34, 26], dec: [66, 16], und: [88, 6] };
  g.font = '10px system-ui'; g.fillStyle = '#5b5a55'; g.textAlign = 'right'; g.textBaseline = 'middle';
  g.fillText('chunk', m.l - 6, rows.chunk[0] + rows.chunk[1] / 2); g.fillText('requests', m.l - 6, rows.req[0] + rows.req[1] / 2);
  g.fillText('decisions', m.l - 6, rows.dec[0] + rows.dec[1] / 2); g.fillText('underrun', m.l - 6, rows.und[0] + 3);
  // chunks
  E.chunkSpans.forEach((c, k) => {
    g.fillStyle = k % 2 ? '#c6d5f0' : '#e3eaf8'; g.fillRect(X(c.t0), rows.chunk[0], X(c.t1) - X(c.t0), rows.chunk[1]);
    if (X(c.t1) - X(c.t0) > 18) { g.fillStyle = '#28406e'; g.textAlign = 'center'; g.fillText(String(c.chunk), (X(c.t0) + X(c.t1)) / 2, rows.chunk[0] + rows.chunk[1] / 2); }
  });
  // requests
  const nCol = (n) => seqColor((n - 3) / 7 * 0.85 + 0.05);
  E.meta.requests.forEach((q) => {
    const y = rows.req[0], h = rows.req[1];
    g.fillStyle = q.abandoned ? '#bdbcb6' : nCol(q.n);
    g.fillRect(X(q.obs_t), y + 3, Math.max(1, X(Math.min(q.ready_t, T)) - X(q.obs_t)), h - 6);
    if (q.handoff_t > q.ready_t) {
      const x0 = X(q.ready_t), x1 = X(Math.min(q.handoff_t, T));
      g.save(); g.beginPath(); g.rect(x0, y + 3, x1 - x0, h - 6); g.clip(); g.strokeStyle = '#e59a5c';
      for (let k = x0 - h; k < x1; k += 4) { g.beginPath(); g.moveTo(k, y + h); g.lineTo(k + h, y); g.stroke(); }
      g.restore();
    }
    // planned handoff marker (open triangle)
    const xh = X(q.handoff_t);
    if (q.handoff_t <= T) {
      g.strokeStyle = COL.accent2; g.beginPath(); g.moveTo(xh - 4, y - 1); g.lineTo(xh + 4, y - 1); g.lineTo(xh, y + 5); g.closePath(); g.stroke();
    }
    if (X(Math.min(q.ready_t, T)) - X(q.obs_t) > 22) { g.fillStyle = '#fff'; g.textAlign = 'left'; g.fillText(`N${q.n}`, X(q.obs_t) + 2, y + h / 2); }
    if (q.abandoned) { g.fillStyle = '#5b5a55'; g.textAlign = 'left'; g.fillText('abandoned (episode ended)', X(q.obs_t) + 2, y - 3); }
  });
  // actual handoffs
  g.strokeStyle = '#141413'; g.lineWidth = 1.5;
  E.handoffs.forEach((hdf) => { const x = X(hdf.t); g.beginPath(); g.moveTo(x, rows.chunk[0]); g.lineTo(x, rows.req[0] + rows.req[1]); g.stroke(); });
  g.lineWidth = 1;
  // decisions
  const D = E.meta.decisions;
  D.t.forEach((u, i) => {
    const a = D.action[i], x = X(u);
    if (a === 0) { g.fillStyle = '#c9c7bf'; g.fillRect(x - 0.5, rows.dec[0] + 5, 1.5, 6); }
    else { g.fillStyle = COL.keep; g.beginPath(); g.arc(x, rows.dec[0] + 8, 3.5, 0, 7); g.fill(); }
  });
  // underrun
  g.fillStyle = COL.bad;
  E.meta.ticks.underrun.forEach((u, k) => { if (u) g.fillRect(X(k), rows.und[0], Math.max(1, X(k + 1) - X(k)), rows.und[1]); });
  // axis
  g.fillStyle = '#5b5a55'; g.textAlign = 'center'; g.textBaseline = 'top'; g.strokeStyle = '#ebeae5';
  niceTicks(0, T, Math.floor(pw / 60)).forEach((v) => { const x = X(v); g.fillText(`${v}`, x, 100); g.fillRect(x, 96, 1, 3); });
  g.textAlign = 'right'; g.fillText('control step (50 Hz)', m.l + pw, 116);
  if (E.meta.success) { g.fillStyle = COL.good; g.textAlign = 'right'; g.fillText(`success at step ${T}`, m.l + pw, 86); }
  // cursor
  g.strokeStyle = S.lock ? COL.accent2 : '#d9632b'; g.lineWidth = 2; g.beginPath(); g.moveTo(X(t), 2); g.lineTo(X(t), 96); g.stroke(); g.lineWidth = 1;
}
function tlX(e) { const tl = App._tl; if (!tl) return null; const r = e.target.getBoundingClientRect(), x = e.clientX - r.left; if (x < tl.m.l - 4 || x > tl.m.l + tl.pw + 4) return null; return Math.max(0, Math.min(tl.T, (x - tl.m.l) / tl.pw * tl.T)); }
function tlHover(t) {
  const E = curEp(), tk = E.meta.ticks;
  let s = `<b>step ${t}</b>`;
  if (t < tk.t.length) s += ` · chunk ${tk.chunk[t]} · p ${tk.p[t]} · queue after pop ${tk.qlen_after[t]}`;
  const i = E.decAt[t]; if (i >= 0) s += `<br>decision: ${actName(E.meta.decisions.action[i])}`;
  E.meta.requests.forEach((q) => {
    if (t >= q.obs_t && t <= q.handoff_t) s += `<br>request N=${q.n}, R=${q.r}: obs ${q.obs_t}, ready ${q.ready_t}, planned handoff ${q.handoff_t}${q.abandoned ? ' (abandoned — not an actual handoff)' : ''}; discarded ${q.discarded}`;
  });
  if (E.handoffs.some((h) => h.t === t)) s += '<br><b>actual handoff</b> (new chunk installed)';
  return s;
}

/* ------------------------------------------------------------------ episode-time plots */
function ensureEpCharts() {
  if (App.charts.ep) return;
  const host = $('epPlots');
  App.charts.ep = {
    nr: new Chart(host, { h: 140 }), pk: new Chart(host, { h: 140 }), ent: new Chart(host, { h: 140 }), cond: new Chart(host, { h: 140 }),
    queue: new Chart(host, { h: 140 }), rew: new Chart(host, { h: 140 }), ret: new Chart(host, { h: 140 }), q: new Chart(host, { h: 150 }), qd: new Chart(host, { h: 140 }),
  };
}
function renderEpPlots() {
  ensureEpCharts();
  const E = curEp(), t = viewTick(), C = App.charts.ep, D = E.meta.decisions, R = E.meta.requests, tk = E.meta.ticks;
  const xd = [0, E.T], base = { xDomain: xd, cursor: t, click: (x) => seekFromChart(x), xLabel: 'episode control step' };
  const ho = { type: 'vlines', x: E.handoffs.map((h) => h.t), color: '#141413', alpha: 0.18, width: 1 };
  const hov = (fn) => (x) => fn(Math.round(x));
  const reqNear = (x) => { let b = null, bd = 1e9; R.forEach((q) => { const d = Math.abs(q.obs_t - x); if (d < bd) { bd = d; b = q; } }); return bd <= 3 ? b : null; };
  const decNear = (x) => { const i = nearestDecision(E, x); return i >= 0 && Math.abs(D.t[i] - x) <= 3 ? i : -1; };
  C.nr.set({ ...base, title: 'Selected N and R at request events', yDomain: [0, 26], series: [ho,
    { type: 'hline', y: 10, color: '#a3a29c', dash: [4, 3], width: 1 },
    { type: 'points', x: R.map((q) => q.obs_t), y: R.map((q) => q.r), color: COL.accent, label: 'R (old actions after obs.)', r: 3 },
    { type: 'points', x: R.map((q) => q.obs_t), y: R.map((q) => q.n), color: COL.accent2, label: 'N (denoising steps)', shape: 'diamond', r: 3 }],
  hover: hov((x) => { const q = reqNear(x); return q ? `request at ${q.obs_t}: N=${q.n}, R=${q.r}, d(N)=${q.d}, discarded ${q.discarded}${q.abandoned ? ', abandoned' : ''}` : `step ${x}`; }) });
  const st = E.stats;
  C.pk.set({ ...base, title: `P(KEEP) at decision ticks${E.learned ? ' (policy as used, incl. prior)' : ' (deterministic rule)'}`, yDomain: [-0.04, 1.04], series: [ho,
    { type: 'points', x: D.t, y: st.map((s, i) => (E.mask(i)[0] ? s.pKeep : null)), color: COL.keep, r: 2.2, label: 'P(KEEP)' },
    { type: 'points', x: D.t, y: st.map((s, i) => (E.mask(i)[0] ? null : 0)), color: '#c9c7bf', r: 2, label: 'KEEP infeasible' }],
  hover: hov((x) => { const i = decNear(x); return i >= 0 ? `decision at ${D.t[i]}: P(KEEP) ${pct(st[i].pKeep)}, chose ${actName(D.action[i])}` : `step ${x}`; }) });
  C.ent.set({ ...base, title: 'Policy entropy at decision ticks (nats)', series: [ho, { type: 'points', x: D.t, y: st.map((s) => s.ent), color: '#3d3c38', r: 2.2 }], yMin0: true,
    hover: hov((x) => { const i = decNear(x); return i >= 0 ? `decision at ${D.t[i]}: entropy ${st[i].ent.toFixed(4)}` : `step ${x}`; }) });
  C.cond.set({ ...base, title: 'Conditional E[N | REPLAN] and E[R | REPLAN] at decisions', yDomain: [0, 26], series: [ho,
    { type: 'points', x: D.t, y: st.map((s) => (s.negligible ? null : s.ER)), color: COL.accent, r: 2.2, label: 'E[R | REPLAN]' },
    { type: 'points', x: D.t, y: st.map((s) => (s.negligible ? null : s.EN)), color: COL.accent2, r: 2.2, label: 'E[N | REPLAN]', shape: 'diamond' }],
  hover: hov((x) => { const i = decNear(x); return i >= 0 ? (st[i].negligible ? `decision at ${D.t[i]}: REPLAN mass < 1e-6 (omitted)` : `decision at ${D.t[i]}: E[N|R] ${st[i].EN.toFixed(2)}, E[R|R] ${st[i].ER.toFixed(2)}, P(REPLAN) ${pct(st[i].rep)}`) : `step ${x}`; }) });
  const qStart = tk.t.map((_, k) => tk.qlen_after[k] + (1 - tk.underrun[k]));
  const age = tk.t.map((_, k) => k - (E.chunkOrigin[tk.chunk[k]] ?? 0));
  C.queue.set({ ...base, title: 'Queue length q and plan observation age (steps)', yMin0: true, series: [ho,
    { type: 'step', x: tk.t, y: qStart, color: COL.accent, label: 'q (start of step)', xEnd: E.T },
    { type: 'step', x: tk.t, y: age, color: COL.obs, label: 'plan obs. age', xEnd: E.T },
    { type: 'vlines', x: [], color: '#000', label: 'actual handoff' }],
  hover: hov((x) => (x < tk.t.length ? `step ${x}: q ${qStart[x]}, plan age ${age[x]}, chunk ${tk.chunk[x]}` : `step ${x}`)) });
  C.rew.set({ ...base, title: 'Immediate reward per control step (observed)', series: [
    { type: 'points', x: tk.t, y: tk.reward, color: '#3d3c38', r: 1.6, label: 'reward' }],
  hover: hov((x) => (x < tk.reward.length ? `step ${x}: reward ${tk.reward[x].toFixed(5)}` : `step ${x}`)) });
  C.ret.set({ ...base, title: 'Cumulative undiscounted return (observed)', series: [
    { type: 'step', x: tk.t.map((k) => k + 1), y: E.cumRet, color: COL.good, label: 'cumulative return', xEnd: E.T }],
  hover: hov((x) => (x > 0 && x <= E.cumRet.length ? `after step ${x - 1}: ${E.cumRet[x - 1].toFixed(5)}` : `step ${x}`)) });
  const showQ = S.showQ && E.learned;
  C.q.box.style.display = showQ ? '' : 'none'; C.qd.box.style.display = showQ ? '' : 'none';
  if (showQ) {
    const pr = E.probes.get(E.meta.policy.id);
    const qs = D.t.map((_, i) => { const r = probeRow(pr, i), a = D.action[i]; return Math.min(r.q1[a], r.q2[a]); });
    const qb = D.t.map((_, i) => { const r = probeRow(pr, i), a = D.ref[i]; return Math.min(r.q1[a], r.q2[a]); });
    const qd = D.t.map((_, i) => { const r = probeRow(pr, i), a = D.action[i]; return Math.abs(r.q1[a] - r.q2[a]); });
    C.q.set({ ...base, title: 'Predicted min(Q1,Q2) vs observed discounted return-to-go', series: [ho,
      { type: 'points', x: D.t, y: qs, color: COL.q, r: 2.2, label: 'Q(selected) [predicted]' },
      { type: 'points', x: D.t, y: qb, color: '#a3a29c', r: 1.8, label: 'Q(baseline action)', hollow: true },
      { type: 'step', x: tk.t, y: E.rtg, color: COL.good, label: 'observed return-to-go (rewards, γ=0.999)', xEnd: E.T, dash: [4, 2] }],
    hover: hov((x) => { const i = decNear(x); const o = x < E.rtg.length ? `observed return-to-go ${E.rtg[x].toFixed(4)}` : ''; return i >= 0 ? `decision at ${D.t[i]}: Q(selected) ${qs[i].toFixed(4)}, Q(baseline) ${qb[i].toFixed(4)}<br>${o}` : `step ${x}<br>${o}`; }) });
    C.qd.set({ ...base, title: 'Critic disagreement |Q1 − Q2| at the selected action', yMin0: true, series: [ho, { type: 'points', x: D.t, y: qd, color: '#8a3d9a', r: 2.2 }],
      hover: hov((x) => { const i = decNear(x); return i >= 0 ? `decision at ${D.t[i]}: |Q1−Q2| ${qd[i].toExponential(2)}` : `step ${x}`; }) });
  }
}

/* ------------------------------------------------------------------ across checkpoints */
function renderAcross() {
  const sec = $('acrossSec'), host = $('acrossPlots');
  if (!S.lock) {
    sec.style.display = '';
    if (App.charts.ac) { Object.values(App.charts.ac).forEach((c) => { c.box.style.display = 'none'; }); }
    let msg = host.querySelector('.note.lockmsg');
    if (!msg) { msg = document.createElement('div'); msg.className = 'note lockmsg'; host.appendChild(msg); }
    msg.style.display = '';
    msg.textContent = 'Lock a decision state (button under the video, available at decision ticks) to compare every checkpoint on the identical frozen input.';
    return;
  }
  const msg = host.querySelector('.note.lockmsg'); if (msg) msg.style.display = 'none';
  if (!App.charts.ac) {
    const o = { h: 175, m: { l: 64, r: 12, t: 34, b: 26 } };
    App.charts.ac = { p: new Chart(host, o), arg: new Chart(host, o), cond: new Chart(host, o), q: new Chart(host, o), qd: new Chart(host, o) };
  }
  Object.values(App.charts.ac).forEach((c) => { c.box.style.display = ''; });
  const E = App.eps.get(S.lock.eid), i = S.lock.i, D = E.meta.decisions, mask = E.mask(i), ref = D.ref[i], src = D.action[i];
  const cks = App.idx.checkpoints;
  const xs = cks.map((c) => c.train_steps);
  const rows = cks.map((c) => {
    const r = probeRow(E.probes.get(c.id), i);
    const on = probsFrom(r.res, mask, ref, c.prior_strength), off = probsFrom(r.res, mask, ref, 0);
    const so = distStats(on, mask), sf = distStats(off, mask);
    let qmx = -Infinity, dis = 0, nf = 0;
    for (let a = 0; a < NA; a++) if (mask[a]) { const q = Math.min(r.q1[a], r.q2[a]); if (q > qmx) qmx = q; dis += Math.abs(r.q1[a] - r.q2[a]); nf++; }
    const qa = (a) => Math.min(r.q1[a], r.q2[a]);
    return { c, on, off, so, sf, qkeep: mask[0] ? qa(0) : null, qsrc: qa(src), qref: qa(ref), qarg: qa(so.argmax), qmx, dis: dis / nf, disArg: Math.abs(r.q1[so.argmax] - r.q2[so.argmax]) };
  });
  const sel = S.lock.probe;
  const cur = App.ckById[sel] ? App.ckById[sel].train_steps : null;
  const pick = (x) => { let b = cks[0], bd = 1e18; cks.forEach((c) => { const d = Math.abs(c.train_steps - x); if (d < bd) { bd = d; b = c; } }); return b; };
  const base = { xDomain: [-8000, 308000], cursor: cur, cursorColor: COL.accent2, bg: COL.across, xFmt: (v) => (v === 0 ? '0' : v / 1000 + 'k'),
    xLabel: 'training environment steps of the probed checkpoint', click: (x) => selectPolicy(pick(x).id),
    annotations: [{ x: 5000, label: '' }, { x: 50000, label: 'prior → 0' }],
    hover: (x) => { const c = pick(x), r = rows[cks.indexOf(c)]; return `<b>checkpoint ${fmtSteps(c.train_steps)}</b> (prior strength ${c.prior_strength.toFixed(3)})<br>argmax ${actName(r.so.argmax)}<br>P(KEEP) incl. prior ${pct(r.so.pKeep)} · network only ${pct(r.sf.pKeep)}<br>P(source action ${actShort(src)}) ${pct(r.on[src])} · P(baseline ${actShort(ref)}) ${pct(r.on[ref])}<br>min Q: KEEP ${fmtNum(r.qkeep)}, source ${fmtNum(r.qsrc)}, baseline ${fmtNum(r.qref)}, max ${fmtNum(r.qmx)}`; } };
  const AC = App.charts.ac;
  AC.p.set({ ...base, title: 'Action probabilities on the frozen state', yDomain: [-0.04, 1.08], series: [
    { type: 'line', x: xs, y: rows.map((r) => (mask[0] ? r.so.pKeep : null)), color: COL.keep, label: 'P(KEEP) incl. prior' },
    { type: 'points', x: xs, y: rows.map((r) => (mask[0] ? r.so.pKeep : null)), color: COL.keep, r: 3 },
    { type: 'line', x: xs, y: rows.map((r) => (mask[0] ? r.sf.pKeep : null)), color: COL.keep, dash: [4, 3], label: 'P(KEEP) network only' },
    { type: 'line', x: xs, y: rows.map((r) => r.on[src]), color: COL.accent2, label: `P(source action ${actShort(src)})` },
    { type: 'points', x: xs, y: rows.map((r) => r.on[src]), color: COL.accent2, r: 2.5 },
    { type: 'line', x: xs, y: rows.map((r) => r.on[ref]), color: '#3d3c38', dash: [2, 2], label: `P(baseline ${actShort(ref)})` }] });
  AC.arg.set({ ...base, title: 'Argmax (selected) joint action per checkpoint — KEEP plotted at 0', yDomain: [-1.5, 27], series: [
    { type: 'points', x: xs, y: rows.map((r) => (r.so.argmax === 0 ? null : decA(r.so.argmax).r)), color: COL.accent, r: 3.5, label: 'R' },
    { type: 'points', x: xs, y: rows.map((r) => (r.so.argmax === 0 ? null : decA(r.so.argmax).n)), color: COL.accent2, r: 3.5, shape: 'diamond', label: 'N' },
    { type: 'points', x: xs, y: rows.map((r) => (r.so.argmax === 0 ? 0 : null)), color: COL.keep, r: 4, shape: 'square', label: 'KEEP' },
    { type: 'labels', x: xs, y: rows.map((r) => (r.so.argmax === 0 ? 0 : decA(r.so.argmax).r)), text: rows.map((r) => actShort(r.so.argmax)), color: '#3d3c38' }] });
  AC.cond.set({ ...base, title: 'Conditional E[N | REPLAN], E[R | REPLAN] (incl. prior)', yDomain: [0, 26], series: [
    { type: 'line', x: xs, y: rows.map((r) => (r.so.negligible ? null : r.so.ER)), color: COL.accent, label: 'E[R | REPLAN]' },
    { type: 'points', x: xs, y: rows.map((r) => (r.so.negligible ? null : r.so.ER)), color: COL.accent, r: 2.5 },
    { type: 'line', x: xs, y: rows.map((r) => (r.so.negligible ? null : r.so.EN)), color: COL.accent2, label: 'E[N | REPLAN]' },
    { type: 'points', x: xs, y: rows.map((r) => (r.so.negligible ? null : r.so.EN)), color: COL.accent2, r: 2.5, shape: 'diamond' }] });
  AC.q.set({ ...base, title: 'Critic estimates for the frozen state (min(Q1,Q2); not success probability)', series: [
    { type: 'line', x: xs, y: rows.map((r) => r.qmx), color: COL.q, label: 'max over feasible' },
    { type: 'line', x: xs, y: rows.map((r) => r.qkeep), color: COL.keep, label: 'KEEP' },
    { type: 'line', x: xs, y: rows.map((r) => r.qsrc), color: COL.accent2, label: 'source action' },
    { type: 'line', x: xs, y: rows.map((r) => r.qref), color: '#3d3c38', dash: [2, 2], label: 'baseline action' },
    { type: 'points', x: xs, y: rows.map((r) => r.qarg), color: COL.accent, r: 2.5, label: 'argmax action' }] });
  AC.qd.set({ ...base, title: 'Critic disagreement |Q1 − Q2| on the frozen state', yMin0: true, series: [
    { type: 'line', x: xs, y: rows.map((r) => r.dis), color: '#8a3d9a', label: 'mean over feasible actions' },
    { type: 'points', x: xs, y: rows.map((r) => r.disArg), color: COL.accent, r: 2.5, label: 'at argmax action' }] });
}

/* ------------------------------------------------------------------ learning diagnostics */
function movMean(y, w) {
  if (w <= 1) return y;
  const o = []; let s = 0, n = 0; const q = [];
  for (const v of y) { const x = v === null ? null : +v; q.push(x); if (x !== null) { s += x; n++; } if (q.length > w) { const r = q.shift(); if (r !== null) { s -= r; n--; } } o.push(n ? s / n : null); }
  return o;
}
function renderLearning() {
  const L = App.learning; if (!L) return;
  const host = $('learnPlots');
  if (!App.charts.learn) {
    const o = { h: 170 };
    const names = ['evals', 'trsucc', 'ret', 'tend', 'closs', 'qest', 'qdis', 'ent', 'alpha', 'prior', 'resid', 'aloss', 'calls', 'ms', 'exec', 'nr', 'phase'];
    App.charts.learn = {}; names.forEach((k) => { App.charts.learn[k] = new Chart(host, o); });
  }
  const C = App.charts.learn, ep = L.episodes, up = L.updates, w = S.smooth;
  const cks = App.idx.checkpoints;
  const sel = S.lock ? S.lock.probe : S.pol;
  const cur = App.ckById[sel] ? App.ckById[sel].train_steps : null;
  const nearest = (x) => { let b = cks[0], bd = 1e18; cks.forEach((c) => { const d = Math.abs(c.train_steps - x); if (d < bd) { bd = d; b = c; } }); return b; };
  const base = { xDomain: [-5000, 305000], xFmt: (v) => (v === 0 ? '0' : v / 1000 + 'k'), xLabel: 'training environment steps', cursor: cur,
    annotations: L.annotations.map((a) => ({ x: a.steps, label: a.steps === 5000 ? '5k' : '50k prior → 0', color: '#b07a00' })),
    click: (x) => { selectPolicy(nearest(x).id); }, };
  const ckl = { type: 'vlines', x: cks.map((c) => c.train_steps), color: '#7a5cc4', alpha: 0.15, width: 1 };
  const hovEp = (keys) => (x) => { let k = 0; while (k < ep.steps.length - 1 && ep.steps[k] < x) k++; return `training episode ending at ${fmtSteps(ep.steps[k])} steps<br>` + keys.map(([lab, arr]) => `${lab}: ${fmtNum(arr[k])}`).join('<br>') + `<br><i>click: select checkpoint ${fmtSteps(nearest(x).train_steps)}</i>`; };
  const hovUp = (keys) => (x) => { let k = 0; while (k < up.steps.length - 1 && up.steps[k] < x) k++; return `update row at ${fmtSteps(up.steps[k])} steps (${fmtSteps(up.grad_steps[k])} grad updates)<br>` + keys.map((kk) => `${kk}: ${fmtNum(up[kk][k])}`).join('<br>'); };
  const ev = L.evals, ref = L.reference_eval[0], sup = L.supplemental;
  const smLab = w > 1 ? ` (moving mean of ${w} episodes)` : ' (raw)';
  C.evals.set({ ...base, title: 'Held-out evaluation success (50 seeds, deterministic)', yDomain: [0, 1], series: [ckl,
    { type: 'hline', y: ref.success_rate, y2: ref.wilson95, color: '#5b5a55', dash: [5, 4], label: `fixed baseline ${ref.n_success}/50` },
    { type: 'band', x: ev.map((e) => e.train_steps), y: ev.map((e) => e.wilson95[0]), y2: ev.map((e) => e.wilson95[1]), color: COL.accent, label: '95% Wilson' },
    { type: 'line', x: ev.map((e) => e.train_steps), y: ev.map((e) => e.success_rate), color: COL.accent, label: 'learned (checkpoint prior)' },
    { type: 'points', x: ev.map((e) => e.train_steps), y: ev.map((e) => e.success_rate), color: COL.accent, r: 3 },
    { type: 'points', x: sup.map((e) => e.train_steps), y: sup.map((e) => e.success_rate), color: COL.accent2, r: 4.5, shape: 'diamond', hollow: true, label: 'prior disabled (25k)' }],
  hover: (x) => { const e = ev.reduce((b, e2) => (Math.abs(e2.train_steps - x) < Math.abs(b.train_steps - x) ? e2 : b)); return `eval at ${fmtSteps(e.train_steps)} steps: ${e.n_success}/${e.n_episodes} (95% ${e.wilson95.map((v) => v.toFixed(2)).join('–')})<br>prior strength ${e.prior_strength.toFixed(3)} · fixed baseline ${ref.n_success}/50`; } });
  const succ = ep.success.map((v) => +v);
  C.trsucc.set({ ...base, title: 'Training-episode success (stochastic behaviour policy)' + smLab, yDomain: [-0.05, 1.05], series: [ckl,
    { type: 'points', x: ep.steps, y: succ, color: '#a3a29c', r: 1.3, alpha: 0.5, label: 'episode outcome' },
    ...(w > 1 ? [{ type: 'line', x: ep.steps, y: movMean(succ, w), color: COL.accent2, label: 'moving mean' }] : [])], hover: hovEp([['success', succ]]) });
  C.ret.set({ ...base, title: 'Training-episode return' + smLab, series: [ckl,
    { type: w > 1 ? 'line' : 'points', r: 1.3, x: ep.steps, y: movMean(ep.ret, w), color: COL.good, label: 'undiscounted return (logged)' },
    { type: w > 1 ? 'line' : 'points', r: 1.3, x: ep.steps, y: movMean(ep.discounted_return_reconstructed, w), color: '#0f6b4a', dash: [4, 2], label: 'discounted γ=0.999 (reconstructed)' }],
  hover: hovEp([['return', ep.ret], ['discounted (reconstructed)', ep.discounted_return_reconstructed]]) });
  C.tend.set({ ...base, title: 'Training-episode completion time (control steps; 400 = timeout)' + smLab, series: [ckl,
    { type: w > 1 ? 'line' : 'points', r: 1.3, x: ep.steps, y: movMean(ep.t_end, w), color: '#3d3c38', label: 't_end' },
    { type: 'points', x: ev.map((e) => e.train_steps), y: ev.map((e) => { const s = e.episodes; const v = s.t_end.filter((_, k) => s.success[k]); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; }), color: COL.accent, r: 3, label: 'held-out: mean t_end of successes' }],
  hover: hovEp([['t_end', ep.t_end]]) });
  const ul = (k, col, lab, extra = {}) => ({ type: 'line', x: up.steps, y: up[k], color: col, label: lab || k, ...extra });
  C.closs.set({ ...base, title: 'Critic loss (mean of both critics; log scale)', yLog: true, series: [ckl, ul('critic_loss', '#8a3d9a', 'critic loss')], hover: hovUp(['critic_loss']) });
  C.qest.set({ ...base, title: 'Q estimates and targets (replay batches; log scale, values ≤ 0 not drawn)', yLog: true, series: [ckl, ul('q_taken', COL.q, 'Q(s,a) taken'), ul('target', COL.good, 'target y', { dash: [4, 2] }), ul('q_max', '#a3a29c', 'max Q(s,a) in batch')], hover: hovUp(['q_taken', 'target', 'q_max']) });
  C.qdis.set({ ...base, title: 'Critic disagreement and feasible Q gap', series: [ckl, ul('q_disagreement', '#8a3d9a', '|Q1−Q2| at replay action'), ul('q_gap_feasible', COL.accent2, 'max−min min-Q over feasible')], hover: hovUp(['q_disagreement', 'q_gap_feasible']) });
  C.ent.set({ ...base, title: 'Policy entropy vs target entropy (replay states)', series: [ckl, ul('entropy', '#3d3c38', 'entropy'), ul('target_entropy', COL.accent2, 'target entropy', { dash: [4, 2] })], hover: hovUp(['entropy', 'target_entropy']) });
  C.alpha.set({ ...base, title: 'Temperature α (log scale)', yLog: true, series: [ckl, ul('alpha', COL.keep, 'α')], hover: hovUp(['alpha']) });
  C.prior.set({ ...base, title: 'Prior strength and probability of the baseline action', yDomain: [-0.03, 1.03], series: [ckl, ul('prior_strength', '#b07a00', 'prior strength'), ul('ref_prob', COL.accent, 'P(baseline action) on replay states')], hover: hovUp(['prior_strength', 'ref_prob']) });
  C.resid.set({ ...base, title: 'Residual logit magnitude (max |residual| over feasible)', series: [ckl, ul('residual_max', '#3d3c38', 'residual max')], hover: hovUp(['residual_max']) });
  C.aloss.set({ ...base, title: 'Actor loss', series: [ckl, ul('actor_loss', '#3d3c38', 'actor loss')], hover: hovUp(['actor_loss']) });
  const evp = (k) => ({ type: 'points', x: ev.map((e) => e.train_steps), y: ev.map((e) => e[k]), color: COL.accent, r: 3, label: 'held-out eval' });
  C.calls.set({ ...base, title: 'Inference requests per 1,000 control steps' + smLab, series: [ckl, { type: 'hline', y: ref.calls_per_1k, color: '#5b5a55', dash: [5, 4], label: 'fixed baseline' },
    { type: w > 1 ? 'line' : 'points', r: 1.3, x: ep.steps, y: movMean(ep.calls_per_1k, w), color: COL.accent2, label: 'training' }, evp('calls_per_1k')], hover: hovEp([['calls / 1k', ep.calls_per_1k]]) });
  C.ms.set({ ...base, title: 'Modeled inference time per 1,000 steps (ms, d(N)·20 ms; not GPU time)' + smLab, series: [ckl, { type: 'hline', y: ref.modeled_ms_per_1k, color: '#5b5a55', dash: [5, 4], label: 'fixed baseline' },
    { type: w > 1 ? 'line' : 'points', r: 1.3, x: ep.steps, y: movMean(ep.modeled_ms_per_1k, w), color: COL.accent2, label: 'training' }, evp('modeled_ms_per_1k')], hover: hovEp([['modeled ms / 1k', ep.modeled_ms_per_1k]]) });
  C.exec.set({ ...base, title: 'Chunk execution length = old actions executed per chunk' + smLab, series: [ckl, { type: 'hline', y: 25, color: '#5b5a55', dash: [5, 4], label: 'fixed baseline 25' },
    { type: w > 1 ? 'line' : 'points', r: 1.3, x: ep.steps, y: movMean(ep.mean_exec_len, w), color: COL.accent2, label: 'training' }, evp('mean_exec_len')], hover: hovEp([['exec length', ep.mean_exec_len]]) });
  C.nr.set({ ...base, title: 'Selected N and R (mean per episode)' + smLab, yDomain: [0, 26], series: [ckl, { type: 'hline', y: 10, color: '#5b5a55', dash: [5, 4] },
    { type: 'line', x: ep.steps, y: movMean(ep.mean_N, w), color: COL.accent2, label: 'N (training)' }, { type: 'line', x: ep.steps, y: movMean(ep.mean_R, w), color: COL.accent, label: 'R (training)' }],
  hover: hovEp([['mean N', ep.mean_N], ['mean R', ep.mean_R]]) });
  C.phase.set({ ...base, title: 'Launch phase p and observation age at handoff' + smLab, yMin0: true, series: [ckl, { type: 'hline', y: 15, color: '#5b5a55', dash: [5, 4], label: 'baseline p = 15' },
    { type: 'line', x: ep.steps, y: movMean(ep.mean_launch_p, w), color: '#3d3c38', label: 'launch p' }, { type: 'line', x: ep.steps, y: movMean(ep.mean_obs_age_handoff, w), color: COL.obs, label: 'obs. age at handoff' }],
  hover: hovEp([['launch p', ep.mean_launch_p], ['obs age at handoff', ep.mean_obs_age_handoff]]) });
  $('learnNotes').innerHTML = `<h4>Notes</h4><ul>${Object.values(L.notes).map((n) => `<li>${n}</li>`).join('')}</ul><h4>Metrics not logged by the run (marked missing)</h4><ul>${L.missing.map((n) => `<li>${n}</li>`).join('')}</ul>` +
    `<p>Reward: ${L.reward.formula}; GPU cost is logged but not part of the reward. Discount γ = ${L.reward.gamma} per control step.</p>`;
}

/* ------------------------------------------------------------------ about */
function renderAbout() {
  const I = App.idx, L = App.learning;
  const seedRows = I.seeds.map((s) => `<tr><td>${s.seed}</td><td>${s.category.replace(/_/g, ' ')}</td><td>${s.logged_baseline_success ? 'success' : 'fail'}</td><td>${s.logged_learned_successes}/14</td><td>${s.logged_outcome_flips}</td></tr>`).join('');
  const pols = App.policies.map((p) => p.id);
  let grid = `<table><tr><th>policy \\ seed</th>${I.seeds.map((s) => `<th>${s.seed}</th>`).join('')}</tr>`;
  pols.forEach((pid) => {
    grid += `<tr><td>${pid === 'baseline' ? 'fixed baseline' : fmtSteps(App.ckById[pid].train_steps)}</td>`;
    I.seeds.forEach((s) => { const e = I.episodes[`${pid}_s${s.seed}`]; grid += e ? `<td><span class="badge ${e.success ? 'ok' : 'no'}">${e.success ? 'S' : 'F'} ${e.t_end}</span>${e.outcome_matches_log === false ? ' ≠log' : ''}</td>` : '<td>missing</td>'; });
    grid += '</tr>';
  });
  grid += '</table>';
  const pc = I.probe_check || [];
  const pcOk = pc.length ? `${pc.reduce((s, r) => s + r.argmax_agree, 0)}/${pc.reduce((s, r) => s + r.decisions, 0)} own-checkpoint decisions: batched probe argmax equals the recorded action; max |Δ residual| ${Math.max(...pc.map((r) => r.max_abs_residual_diff)).toExponential(1)}, max |Δ prob| ${Math.max(...pc.map((r) => r.max_abs_prob_diff)).toExponential(1)}` : 'not available';
  const nm = Object.values(I.episodes).filter((e) => e.outcome_matches_log === false).length;
  $('about').innerHTML = `
<h2>What this viewer shows</h2>
<p>Run <b>${L.run.run_id}</b> of SCH-01 <code>fixed_delay10_v2</code> (one training seed; ${L.run.done}). Frozen π0 (${L.run.pi0_checkpoint}) on ALOHA transfer-cube, RTC on,
deterministic delay table d(N) = ${JSON.stringify(L.run.delay_table)} control steps (${L.run.timing_label}).</p>
<p><b>Mode A — checkpoint rollout playback.</b> Every rollout here is a <span class="badge new">new recording</span> made after training with a frozen checkpoint
(or the fixed baseline), using the held-out evaluation protocol: deterministic masked argmax, the checkpoint's own training-time prior strength, noise stream
<code>default_rng([seed, 1])</code>. Rendering is not bit-deterministic, so a recording can differ from the episode logged at training time
(${nm} of ${Object.keys(I.episodes).length} recordings differ in outcome or end step; each is labelled).</p>
<p><b>Mode B — lock state across checkpoints.</b> At any decision tick, the complete scheduler input (current and source 64×64 images, robot state, queue and
mask, plan age, source N, remaining time, runtime features, feasibility mask) is frozen. The slider then changes only the actor/critic checkpoint applied to it.
Both the policy including that checkpoint's training-time prior and the network-only distribution (prior disabled) are shown; for checkpoints at ≥ 50k steps
the prior strength is 0 and both are identical. All probes are precomputed for every recorded decision state and every checkpoint.</p>
<h3>How to</h3>
<ol><li><b>Inspect an episode:</b> pick a checkpoint on the slider (or "Fixed baseline") and a seed; press Play (Space) or step with ←/→, Shift+←/→ for the
previous/next decision; click any timeline or chart to pause and seek.</li>
<li><b>Lock a state:</b> stop on a decision tick (use "Jump to nearest decision" if a request is in flight) and press "Lock state across checkpoints". Move the
checkpoint slider: the heatmaps and the "Across training checkpoints" charts show what each checkpoint would do on the identical input. "Unlock" returns to playback.</li>
<li><b>Learning diagnostics:</b> the second tab plots the run's own logs; clicking a chart selects the nearest checkpoint.</li></ol>
<p>Keyboard: Space play/pause · ←/→ step · Shift+←/→ decision · [ / ] previous/next checkpoint · L lock/unlock. The URL hash stores the current view.</p>
<h3>Conventions</h3>
<ul><li>Frame t = observation o_t <b>before</b> the action of control step t (frame 0 = reset). The decision at tick t is made on o_t.</li>
<li>Action space: KEEP, or REPLAN(R, N): observe now, run N denoising steps (ready after d(N)), execute R more old actions, hand off at t + R to new[R : R + 25].
Feasible iff d(N) ≤ R ≤ q and R + 25 ≤ 50; KEEP iff q − 1 ≥ 6 and more than one step remains. R &lt; 6 is never feasible and is not drawn.</li>
<li>Fixed baseline: KEEP while q &gt; 10; REPLAN(10, 10) at q = 10 (p = 15); 25 actions per chunk.</li>
<li>P(KEEP) + Σ P(REPLAN(R,N)) = 1 (unconditional view). The conditional view P(R,N | REPLAN) is labelled as such.</li>
<li>Q = online critics' learned soft return estimate (γ = 0.999 per step), not a success probability; no calibration is claimed. Observed rewards and returns are plotted separately.</li>
<li>Planned handoff (▽) vs actual handoff (black line): a request whose planned handoff falls after the episode end is marked abandoned and has no actual handoff.</li></ul>
<h3>Curated inspection seeds</h3>
<p>${I.seeds_note}</p><p class="note">Rule: ${I.seeds_rule.replace(/\n/g, '<br>')}</p>
<table><tr><th>seed</th><th>category</th><th>logged baseline</th><th>logged learned successes</th><th>outcome flips</th></tr>${seedRows}</table>
<h3>Recorded rollouts (S/F and end step of the NEW recording; ≠log = differs from the training-time log)</h3>${grid}
<h3>Validation</h3><ul>${(I.validation || []).map((v) => `<li>${v}</li>`).join('')}<li>Probe consistency: ${pcOk}.</li></ul>
<h3>Provenance</h3><p class="note">${JSON.stringify(I.provenance || {})}</p>`;
}

/* ------------------------------------------------------------------ render orchestration */
function renderTick() {
  drawVideo(); renderSched(); renderReadouts(); renderTimeline(); renderEpPlots();
  if (!S.playing) writeHash();
}
function renderAll() {
  ckSliderDraw(); ckReadout(); renderBanner(); renderLockInfo(); renderTick(); renderAcross();
  if ($('tab-learning').classList.contains('active')) renderLearning();
  App._switchNote = false;
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  App.idx = await fetchJSON('data/index.json');
  App.learning = await fetchJSON('data/learning.json');
  App.policies = [{ id: 'baseline', kind: 'fixed_baseline' }].concat(App.idx.checkpoints.map((c) => ({ ...c, kind: 'learned' })));
  App.idx.checkpoints.forEach((c) => { App.ckById[c.id] = c; });
  App.ckById.baseline = null;
  $('runLabel').textContent = `fixed_delay10_v2 · ${App.learning.run.run_id} · ${App.idx.checkpoints.length} checkpoints + fixed baseline · ${App.idx.seeds.length} curated seeds`;
  const sel = $('seedSel');
  App.idx.seeds.forEach((s) => { const o = document.createElement('option'); o.value = s.seed; o.textContent = `${s.seed} — ${s.category.replace(/_/g, ' ')}`; sel.appendChild(o); });
  const h = readHash();
  S.pol = h.pol && (h.pol === 'baseline' || App.ckById[h.pol]) ? h.pol : App.idx.checkpoints[App.idx.checkpoints.length - 1].id;
  S.seed = h.seed && App.idx.seeds.some((s) => s.seed === h.seed) ? h.seed : App.idx.seeds[0].seed;
  sel.value = S.seed;
  await loadEpisode(curEid());
  S.tick = Math.min(h.t || 0, curEp().T);
  if (h.lock) {
    const [eid, ii] = h.lock.split(':');
    try { await loadEpisode(eid); S.lock = { eid, i: +ii, probe: h.probe || S.pol }; await ensureLockProbes(); } catch (e) { S.lock = null; }
  }
  wire();
  renderAll();
  renderAbout();
}

function wire() {
  document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x === b));
    document.querySelectorAll('.tabpane').forEach((p) => p.classList.toggle('active', p.id === 'tab-' + b.dataset.tab));
    if (b.dataset.tab === 'learning') renderLearning();
    if (b.dataset.tab === 'playback') renderAll();
  }));
  $('btnBaseline').addEventListener('click', () => selectPolicy('baseline'));
  $('seedSel').addEventListener('change', (e) => selectSeed(+e.target.value));
  $('bPlay').addEventListener('click', () => (S.playing ? pause() : play()));
  $('bPrev').addEventListener('click', () => setTick(S.tick - 1));
  $('bNext').addEventListener('click', () => setTick(S.tick + 1));
  $('bPrevDec').addEventListener('click', () => stepDecision(-1));
  $('bNextDec').addEventListener('click', () => stepDecision(1));
  $('speedSel').addEventListener('change', (e) => { S.speed = +e.target.value; });
  $('tickRange').addEventListener('input', (e) => setTick(+e.target.value));
  $('bLock').addEventListener('click', () => (S.lock ? unlock() : lockHere()));
  $('bNearest').addEventListener('click', () => { const E = curEp(); const i = nearestDecision(E, S.tick); if (i >= 0) setTick(E.meta.decisions.t[i]); });
  $('chkQ').addEventListener('change', (e) => { S.showQ = e.target.checked; renderEpPlots(); });
  $('smoothSel').addEventListener('change', (e) => { S.smooth = +e.target.value; renderLearning(); });
  const seg = (id, key, fn) => $(id).querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    $(id).querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b)); S[key] = b.dataset.v; fn();
  }));
  seg('probView', 'probView', renderSched); seg('scaleView', 'scale', renderSched); seg('critView', 'critView', renderSched);
  const ckStep = (d) => {
    const ids = App.policies.map((p) => p.id), cur = S.lock ? S.lock.probe : S.pol;
    const k = Math.max(0, Math.min(ids.length - 1, ids.indexOf(cur) + d));
    selectPolicy(ids[k]);
  };
  $('ckSlider').addEventListener('keydown', (e) => { if (e.key === 'ArrowLeft') { ckStep(-1); e.preventDefault(); e.stopPropagation(); } if (e.key === 'ArrowRight') { ckStep(1); e.preventDefault(); e.stopPropagation(); } });
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (!$('tab-playback').classList.contains('active')) return;
    if (e.key === ' ') { e.preventDefault(); S.playing ? pause() : play(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); e.shiftKey ? stepDecision(-1) : setTick(S.tick - 1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); e.shiftKey ? stepDecision(1) : setTick(S.tick + 1); }
    else if (e.key === '[') ckStep(-1);
    else if (e.key === ']') ckStep(1);
    else if (e.key === 'l' || e.key === 'L') { S.lock ? unlock() : lockHere(); }
  });
  window.addEventListener('resize', () => { ckSliderDraw(); renderTick(); });
}

boot().catch((e) => { document.body.insertAdjacentHTML('afterbegin', `<pre style="color:#c43a3a;padding:10px">Failed to load viewer data: ${e.message}</pre>`); console.error(e); });
