'use strict';
/* Minimal canvas chart: step / line / point / vertical-line / band / horizontal-line / label series on one y-axis.
   No interpolation is implied by 'points' and 'step' series (step = value held until the next event). */

const TIP = () => document.getElementById('tip');

function niceTicks(a, b, n) {
  if (!(b > a)) { b = a + 1; }
  const span = b - a, raw = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / mag, step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function showTip(html, ev) {
  const t = TIP();
  if (!html) { t.style.display = 'none'; return; }
  t.innerHTML = html;
  t.style.display = 'block';
  const w = t.offsetWidth, h = t.offsetHeight;
  let x = ev.clientX + 14, y = ev.clientY + 12;
  if (x + w > window.innerWidth - 4) x = ev.clientX - w - 12;
  if (y + h > window.innerHeight - 4) y = ev.clientY - h - 10;
  t.style.left = x + 'px'; t.style.top = y + 'px';
}

class Chart {
  constructor(parent, opts = {}) {
    this.o = Object.assign({ h: 150, m: { l: 64, r: 12, t: 34, b: 26 } }, opts);
    this.box = document.createElement('div');
    this.box.className = 'chartbox';
    parent.appendChild(this.box);
    this.cv = document.createElement('canvas');
    this.box.appendChild(this.cv);
    this.spec = null;
    this.cv.addEventListener('mousemove', (e) => this._hover(e));
    this.cv.addEventListener('mouseleave', () => showTip(null));
    this.cv.addEventListener('click', (e) => {
      if (!this.spec || !this.spec.click) return;
      const x = this._xval(e);
      if (x !== null) this.spec.click(x, e);
    });
    if (window.ResizeObserver) new ResizeObserver(() => this.draw()).observe(this.box);
  }
  set(spec) { this.spec = spec; this.draw(); }
  _xval(e) {
    const r = this.cv.getBoundingClientRect(), m = this.o.m;
    const px = e.clientX - r.left;
    if (px < m.l - 4 || px > r.width - m.r + 4) return null;
    const [a, b] = this.spec.xDomain;
    return a + (px - m.l) / Math.max(1, r.width - m.l - m.r) * (b - a);
  }
  _hover(e) {
    if (!this.spec || !this.spec.hover) return;
    const x = this._xval(e);
    showTip(x === null ? null : this.spec.hover(x), e);
  }
  _ydomain(s) {
    if (s.yDomain) return s.yDomain;
    let lo = Infinity, hi = -Infinity;
    for (const se of s.series) {
      if (se.type === 'vlines' || se.type === 'labels' || se.noAuto) continue;
      const ys = se.type === 'hline' ? [se.y] : se.type === 'band' ? se.y.concat(se.y2) : se.y;
      for (const v of ys) {
        if (v === null || v === undefined || !Number.isFinite(v)) continue;
        if (s.yLog && v <= 0) continue;
        if (v < lo) lo = v; if (v > hi) hi = v;
      }
    }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (s.yLog) { lo = Math.log10(lo); hi = Math.log10(hi); }
    if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.08;
    lo -= pad; hi += pad;
    if (!s.yLog && s.yMin0 && lo < 0 && lo > -pad * 1.5) lo = 0;
    return [lo, hi];
  }
  draw() {
    const s = this.spec;
    if (!s) return;
    const dpr = window.devicePixelRatio || 1, W = Math.max(200, this.box.clientWidth), H = s.h || this.o.h, m = this.o.m;
    this.cv.width = W * dpr; this.cv.height = H * dpr; this.cv.style.height = H + 'px';
    const g = this.cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    g.fillStyle = s.bg || '#ffffff';
    g.fillRect(m.l, m.t, W - m.l - m.r, H - m.t - m.b);
    const [xa, xb] = s.xDomain, [ya, yb] = this._ydomain(s);
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const X = (x) => m.l + (x - xa) / (xb - xa) * pw;
    const Y = (y) => { const v = s.yLog ? Math.log10(Math.max(y, 1e-300)) : y; return m.t + ph - (v - ya) / (yb - ya) * ph; };
    this.X = X; this.Y = Y;
    // grid + ticks
    g.font = '10.5px system-ui, sans-serif';
    g.strokeStyle = '#ebeae5'; g.fillStyle = '#5b5a55'; g.lineWidth = 1;
    const yt = s.yLog ? niceTicks(Math.ceil(ya), Math.floor(yb), Math.min(5, Math.max(1, Math.floor(yb) - Math.ceil(ya)))).map((v) => Math.pow(10, v))
      : (s.yTicks || niceTicks(ya, yb, Math.max(2, Math.floor(ph / 28))));
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const v of yt) {
      const y = Y(v);
      if (y < m.t - 1 || y > m.t + ph + 1) continue;
      g.beginPath(); g.moveTo(m.l, y); g.lineTo(m.l + pw, y); g.stroke();
      g.fillText(s.yFmt ? s.yFmt(v) : fmtNum(v), m.l - 4, y);
    }
    const xt = s.xTicks || niceTicks(xa, xb, Math.max(2, Math.floor(pw / 70)));
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const v of xt) {
      const x = X(v);
      if (x < m.l - 1 || x > m.l + pw + 1) continue;
      g.beginPath(); g.moveTo(x, m.t); g.lineTo(x, m.t + ph); g.stroke();
      g.fillText(s.xFmt ? s.xFmt(v) : fmtNum(v), x, m.t + ph + 3);
    }
    g.strokeStyle = '#a3a29c';
    g.strokeRect(m.l + 0.5, m.t + 0.5, pw, ph);
    // annotations (vertical dashed with label)
    for (const an of (s.annotations || [])) {
      const x = X(an.x);
      if (x < m.l || x > m.l + pw) continue;
      g.save(); g.setLineDash([3, 3]); g.strokeStyle = an.color || '#5b5a55';
      g.beginPath(); g.moveTo(x, m.t); g.lineTo(x, m.t + ph); g.stroke(); g.restore();
      if (an.label) { g.fillStyle = an.color || '#5b5a55'; g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText(an.label, x + 3, m.t + 2); }
    }
    g.save();
    g.beginPath(); g.rect(m.l, m.t, pw, ph); g.clip();
    for (const se of s.series) this._series(g, se, X, Y, m, ph, pw);
    g.restore();
    // cursor
    if (s.cursor !== undefined && s.cursor !== null) {
      const x = X(s.cursor);
      g.strokeStyle = s.cursorColor || '#d9632b'; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(x, m.t); g.lineTo(x, m.t + ph); g.stroke(); g.lineWidth = 1;
    }
    // title + legend
    g.textAlign = 'left'; g.textBaseline = 'alphabetic'; g.fillStyle = '#141413'; g.font = '600 11.5px system-ui, sans-serif';
    g.fillText(s.title || '', m.l, 13);
    let lx = m.l, ly = 26;
    g.font = '10.5px system-ui, sans-serif';
    for (const se of s.series) {
      if (!se.label) continue;
      const w = g.measureText(se.label).width;
      if (lx + w + 22 > W - 2) break;
      g.strokeStyle = se.color; g.fillStyle = se.color; g.lineWidth = 2;
      if (se.dash) g.setLineDash(se.dash);
      if (se.type === 'points') { g.beginPath(); g.arc(lx + 6, ly - 4, 3, 0, 7); g.fill(); }
      else if (se.type === 'band') { g.globalAlpha = 0.25; g.fillRect(lx, ly - 8, 12, 8); g.globalAlpha = 1; }
      else { g.beginPath(); g.moveTo(lx, ly - 4); g.lineTo(lx + 12, ly - 4); g.stroke(); }
      g.setLineDash([]); g.lineWidth = 1;
      g.fillStyle = '#5b5a55'; g.fillText(se.label, lx + 16, ly);
      lx += w + 30;
    }
    if (s.yLabel) {
      g.save(); g.translate(11, m.t + ph / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillStyle = '#5b5a55';
      g.fillText(s.yLabel, 0, 0); g.restore();
    }
    if (s.xLabel) { g.textAlign = 'right'; g.fillStyle = '#5b5a55'; g.fillText(s.xLabel, m.l + pw, H - 2); }
  }
  _series(g, se, X, Y, m, ph, pw) {
    g.strokeStyle = se.color || '#2a63c4'; g.fillStyle = se.color || '#2a63c4'; g.lineWidth = se.width || 1.5;
    g.globalAlpha = se.alpha === undefined ? 1 : se.alpha;
    if (se.dash) g.setLineDash(se.dash);
    const n = se.x ? se.x.length : 0;
    if (se.type === 'line' || se.type === 'step') {
      g.beginPath();
      let pen = false, lastY = null;
      for (let i = 0; i < n; i++) {
        const v = se.y[i];
        if (v === null || v === undefined || !Number.isFinite(v)) { pen = false; continue; }
        const x = X(se.x[i]), y = Y(v);
        if (!pen) { g.moveTo(x, y); pen = true; }
        else if (se.type === 'step') { g.lineTo(x, lastY); g.lineTo(x, y); }
        else g.lineTo(x, y);
        lastY = y;
        if (se.type === 'step' && se.xEnd !== undefined && i === n - 1) g.lineTo(X(se.xEnd), y);
      }
      g.stroke();
    } else if (se.type === 'points') {
      const r = se.r || 2.5;
      for (let i = 0; i < n; i++) {
        const v = se.y[i];
        if (v === null || v === undefined || !Number.isFinite(v)) continue;
        const x = X(se.x[i]), y = Y(v);
        g.beginPath();
        if (se.shape === 'diamond') { g.moveTo(x, y - r - 1); g.lineTo(x + r + 1, y); g.lineTo(x, y + r + 1); g.lineTo(x - r - 1, y); g.closePath(); }
        else if (se.shape === 'square') g.rect(x - r, y - r, 2 * r, 2 * r);
        else g.arc(x, y, r, 0, 7);
        if (se.hollow) { g.lineWidth = 1.3; g.stroke(); } else g.fill();
      }
    } else if (se.type === 'vlines') {
      for (let i = 0; i < n; i++) {
        const x = X(se.x[i]);
        g.beginPath(); g.moveTo(x, m.t); g.lineTo(x, m.t + ph); g.stroke();
      }
    } else if (se.type === 'band') {
      g.globalAlpha = se.alpha === undefined ? 0.15 : se.alpha;
      g.beginPath();
      for (let i = 0; i < n; i++) { const x = X(se.x[i]); i ? g.lineTo(x, Y(se.y2[i])) : g.moveTo(x, Y(se.y2[i])); }
      for (let i = n - 1; i >= 0; i--) g.lineTo(X(se.x[i]), Y(se.y[i]));
      g.closePath(); g.fill();
    } else if (se.type === 'hline') {
      const y = Y(se.y);
      g.beginPath(); g.moveTo(m.l, y); g.lineTo(m.l + pw, y); g.stroke();
      if (se.y2 !== undefined) { g.globalAlpha = 0.08; g.fillRect(m.l, Y(se.y2[1]), pw, Y(se.y2[0]) - Y(se.y2[1])); }
    } else if (se.type === 'labels') {
      g.font = se.font || '9.5px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'bottom';
      for (let i = 0; i < n; i++) if (se.text[i]) g.fillText(se.text[i], X(se.x[i]), Y(se.y[i]) - 4);
    }
    g.setLineDash([]); g.globalAlpha = 1; g.lineWidth = 1;
  }
}

window.Chart = Chart; window.fmtNum = fmtNum; window.showTip = showTip; window.niceTicks = niceTicks;
