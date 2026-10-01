/**
 * Scanning-probe image view.
 *
 * The default presentation is a grayscale/silver measurement image with the
 * instrument's own noise, exactly as it comes out of the simulated feedback
 * loop. Atomic identities are not drawn on it. They appear only when the user
 * hovers or selects, and the hover card states what the model believes the
 * feature is and how confident that attribution is.
 * @module probe-view
 */

import { api } from './api.js';
import { set, state } from './state.js';
import { graticule, setHud } from './hud.js';
import { $, clear, cssVar, decodeFloat32, el, fitCanvas, niceNumber, num } from './util.js';

const PALETTES = {
  silver: [[0, [11, 12, 14]], [0.18, [38, 40, 44]], [0.42, [94, 98, 104]],
           [0.68, [158, 162, 168]], [0.86, [206, 210, 216]], [1, [244, 246, 250]]],
  gray: [[0, [0, 0, 0]], [1, [255, 255, 255]]],
  gold: [[0, [12, 6, 0]], [0.3, [92, 42, 6]], [0.62, [196, 118, 28]],
         [0.85, [240, 190, 96]], [1, [255, 242, 200]]],
  copper: [[0, [8, 4, 2]], [0.5, [150, 82, 50]], [1, [255, 218, 185]]],
  viridis: [[0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]],
            [0.75, [94, 201, 98]], [1, [253, 231, 37]]],
  'blue-white-red': [[0, [33, 66, 160]], [0.5, [247, 247, 247]], [1, [178, 34, 34]]],
  'high-contrast': [[0, [0, 0, 0]], [0.5, [255, 255, 0]], [1, [255, 255, 255]]],
};

function buildLut(name) {
  const stops = PALETTES[name] || PALETTES.silver;
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let a = stops[0], b = stops[stops.length - 1];
    for (let k = 0; k < stops.length - 1; k++) {
      if (t >= stops[k][0] && t <= stops[k + 1][0]) { a = stops[k]; b = stops[k + 1]; break; }
    }
    const f = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = a[1][c] + (b[1][c] - a[1][c]) * f;
  }
  return lut;
}

export class ProbeView {
  constructor(ids, { onIdentify, onProfile } = {}) {
    this.canvas = $(ids.canvas);
    this.overlay = $(ids.overlay);
    this.controls = $(ids.controls);
    this.scaleEl = $(ids.scale);
    this.colorbar = $(ids.colorbar);
    this.hoverCard = $(ids.hover);
    this.ctx = this.canvas.getContext('2d');
    this.onIdentify = onIdentify || (() => {});
    this.onProfile = onProfile || (() => {});
    this.zoom = 1;
    this.offset = { x: 0, y: 0 };
    this.profilePoints = [];
    this.lut = buildLut('silver');
    this._bind();
    this._renderControls();
  }

  _bind() {
    const c = this.canvas;
    c.addEventListener('mousemove', (e) => this._onMove(e));
    c.addEventListener('mouseleave', () => { this.hoverCard.hidden = true; });
    c.addEventListener('click', (e) => this._onClick(e));
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const before = this._toWorld(e);
      this.zoom = Math.max(0.5, Math.min(60, this.zoom * Math.exp(-e.deltaY * 0.0015)));
      const after = this._toWorld(e);
      this.offset.x += before.x - after.x;
      this.offset.y += before.y - after.y;
      this.draw();
    }, { passive: false });
    let drag = null;
    c.addEventListener('mousedown', (e) => {
      if (e.shiftKey || e.button === 1) drag = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('mouseup', () => { drag = null; });
    window.addEventListener('mousemove', (e) => {
      if (!drag) return;
      const s = this._pixelScale();
      this.offset.x -= (e.clientX - drag.x) / s;
      this.offset.y -= (e.clientY - drag.y) / s;
      drag = { x: e.clientX, y: e.clientY };
      this.draw();
    });
  }

  setScan(payload) {
    if (!payload || !payload.data_b64) { this.scan = null; this.draw(); return; }
    this.scan = payload;
    this.values = decodeFloat32(payload.data_b64);
    this.shape = payload.shape;
    this.zoom = 1;
    this.offset = { x: 0, y: 0 };
    this.profilePoints = [];
    this._computeLimits();
    this._renderControls();
    this.draw();
  }

  _computeLimits() {
    const v = this.values;
    const sorted = Float32Array.from(v).sort();
    const lo = sorted[Math.floor(sorted.length * 0.005)];
    const hi = sorted[Math.floor(sorted.length * 0.995)];
    this.limits = { lo, hi: hi > lo ? hi : lo + 1e-9 };
  }

  _pixelScale() {
    if (!this.scan) return 1;
    const [ny, nx] = this.shape;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = this.canvas.width / rect.width;
    return Math.min(this.canvas.width / nx, this.canvas.height / ny) * this.zoom / dpr * dpr;
  }

  _imageRect() {
    const [ny, nx] = this.shape;
    const s = Math.min(this.canvas.width / nx, this.canvas.height / ny) * this.zoom;
    const w = nx * s, h = ny * s;
    const x = (this.canvas.width - w) / 2 - this.offset.x * s;
    const y = (this.canvas.height - h) / 2 - this.offset.y * s;
    return { x, y, w, h, s };
  }

  _toWorld(e) {
    if (!this.scan) return { x: 0, y: 0 };
    const rect = this.canvas.getBoundingClientRect();
    const dpr = this.canvas.width / rect.width;
    const px = (e.clientX - rect.left) * dpr;
    const py = (e.clientY - rect.top) * dpr;
    const r = this._imageRect();
    return { x: (px - r.x) / r.s, y: (py - r.y) / r.s };
  }

  _toAngstrom(col, row) {
    const [x0, y0, x1, y1] = this.scan.extent_A;
    const [ny, nx] = this.shape;
    return [x0 + (x1 - x0) * (col / Math.max(nx - 1, 1)),
            y0 + (y1 - y0) * (row / Math.max(ny - 1, 1))];
  }

  async _onMove(e) {
    if (!this.scan) return;
    const { x: col, y: row } = this._toWorld(e);
    const [ny, nx] = this.shape;
    if (col < 0 || row < 0 || col >= nx || row >= ny) { this.hoverCard.hidden = true; return; }
    const value = this.values[Math.floor(row) * nx + Math.floor(col)];
    const [xa, ya] = this._toAngstrom(col, row);
    const unit = this.scan.units[this.scan.channel] || '';
    $('#status-coords').textContent = `x ${xa.toFixed(3)} Å   y ${ya.toFixed(3)} Å`;
    $('#status-value').textContent = `${this.scan.channel} ${num(value, 5)} ${unit}`;
    set({ tip: { x: xa, y: ya } });

    if (this._hoverTimer) clearTimeout(this._hoverTimer);
    this._hoverTimer = setTimeout(async () => {
      try {
        const info = await api.scanIdentify(xa, ya, this.scan.key);
        this._showHoverCard(e, info, value, unit);
        set({ hover: info });
      } catch { /* ignore transient */ }
    }, 90);
    this.draw();
  }

  _showHoverCard(e, info, value, unit) {
    const rect = this.canvas.getBoundingClientRect();
    const card = this.hoverCard;
    const kindLabel = {
      'atomic-site': 'predicted atomic site',
      'electronic-feature': 'electronic feature (between sites)',
      adsorbate: 'adsorbate / adatom',
      'dopant-site': 'substitutional dopant site',
      defect: 'defect neighbourhood',
      uncertain: 'uncertain feature',
    }[info.kind] || info.kind;
    const pct = Math.round((info.confidence || 0) * 100);
    const rows = [['signal', `${num(value, 5)} ${unit}`]];
    if (info.nearest_element) {
      rows.push(['element', `${info.nearest_element}   Z ${info.atomic_number || '?'}`]);
      rows.push(['name', info.element_name || '--']);
      rows.push(['atom id', `#${info.nearest_atom_id}`]);
      rows.push(['offset', `${num(info.lateral_offset_A, 3)} A`]);
      if (info.predicted_isotope) rows.push(['isotope', info.predicted_isotope]);
      if (info.charge_state_e) {
        rows.push(['charge',
          `${info.charge_state_e > 0 ? '+' : ''}${num(info.charge_state_e, 3)} e`]);
      }
      rows.push(['site', info.lattice_site || '--']);
      rows.push(['role', info.role || '--']);
      rows.push(['coordination', String(info.coordination)]);
    }
    rows.push('rule');
    rows.push(['confidence', `${pct} %`, 'confidence']);
    setHud(card, kindLabel, rows, { foot: info.caveat || '' });
    card.hidden = false;
    const x = e.clientX - rect.left + 16;
    const y = e.clientY - rect.top + 16;
    card.style.left = `${Math.min(x, rect.width - 240)}px`;
    card.style.top = `${Math.min(y, rect.height - 190)}px`;
  }

  _onClick(e) {
    if (!this.scan || e.shiftKey) return;
    const { x: col, y: row } = this._toWorld(e);
    const [xa, ya] = this._toAngstrom(col, row);
    if (state.tool === 'measure') {
      this.profilePoints.push([xa, ya]);
      if (this.profilePoints.length > 2) this.profilePoints = [[xa, ya]];
      if (this.profilePoints.length === 2) this.onProfile(this.profilePoints);
      this.draw();
      return;
    }
    this.onIdentify(xa, ya, e.shiftKey);
  }

  draw() {
    const [w, h] = fitCanvas(this.canvas);
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = cssVar('--viewport');
    ctx.fillRect(0, 0, w, h);
    if (!this.scan) {
      ctx.fillStyle = '#5a6168';
      ctx.font = '13px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No scan acquired - press Scan on the toolbar.', w / 2, h / 2);
      ctx.restore();
      this.overlay.innerHTML = '<b>No probe data</b>';
      return;
    }
    const [ny, nx] = this.shape;
    if (!this._buffer || this._buffer.width !== nx || this._buffer.height !== ny) {
      this._buffer = document.createElement('canvas');
      this._buffer.width = nx; this._buffer.height = ny;
      this._bufferCtx = this._buffer.getContext('2d');
      this._imageData = this._bufferCtx.createImageData(nx, ny);
    }
    const img = this._imageData.data;
    const { lo, hi } = this.limits;
    const span = hi - lo;
    const gamma = state.gamma, contrast = state.contrast, brightness = state.brightness;
    for (let i = 0; i < this.values.length; i++) {
      let t = (this.values[i] - lo) / span;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      if (gamma !== 1) t = Math.pow(t, gamma);
      if (contrast !== 1 || brightness !== 0) {
        t = (t - 0.5) * contrast + 0.5 + brightness;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
      }
      const k = (t * 255) | 0;
      img[i * 4] = this.lut[k * 3];
      img[i * 4 + 1] = this.lut[k * 3 + 1];
      img[i * 4 + 2] = this.lut[k * 3 + 2];
      img[i * 4 + 3] = 255;
    }
    this._bufferCtx.putImageData(this._imageData, 0, 0);

    const r = this._imageRect();
    ctx.imageSmoothingEnabled = r.s < 2.2;
    ctx.drawImage(this._buffer, r.x, r.y, r.w, r.h);
    ctx.strokeStyle = '#3c4247';
    ctx.lineWidth = 1;
    ctx.strokeRect(r.x - 0.5, r.y - 0.5, r.w + 1, r.h + 1);

    if (state.showFeatures && state.scanFeatures) this._drawFeatures(ctx, r);
    if (state.showOverlay && state.render) this._drawAtomOverlay(ctx, r);
    if (state.showTip) this._drawTip(ctx, r);
    this._drawProfileLine(ctx, r);

    ctx.restore();
    this._renderOverlay();
    this._drawScaleBar(r);
    this._drawColorbar();
  }

  _xyToCanvas(xa, ya, r) {
    const [x0, y0, x1, y1] = this.scan.extent_A;
    const [ny, nx] = this.shape;
    const col = ((xa - x0) / (x1 - x0)) * (nx - 1);
    const row = ((ya - y0) / (y1 - y0)) * (ny - 1);
    return [r.x + col * r.s, r.y + row * r.s];
  }

  _drawFeatures(ctx, r) {
    ctx.lineWidth = 1;
    for (const f of state.scanFeatures.features || []) {
      const [x, y] = this._xyToCanvas(f.x_A, f.y_A, r);
      const colour = { 'atomic-site': '#6aa9dd', 'electronic-feature': '#ffb020',
        adsorbate: '#2fb6c4', 'dopant-site': '#e879f9', defect: '#ff6b58',
        uncertain: '#9aa2a8' }[f.kind] || '#9aa2a8';
      ctx.strokeStyle = colour;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 0.35 + 0.65 * (f.confidence || 0);
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  _drawAtomOverlay(ctx, r) {
    const d = state.render;
    if (!d) return;
    const [x0, y0, x1, y1] = this.scan.extent_A;
    const topZ = d.bounds[1][2];
    ctx.strokeStyle = 'rgba(47,182,196,.85)';
    ctx.lineWidth = 1;
    for (let i = 0; i < d.ids.length; i++) {
      const x = d.positions[i * 3], y = d.positions[i * 3 + 1], z = d.positions[i * 3 + 2];
      if (z < topZ - 2.2) continue;
      if (x < x0 || x > x1 || y < y0 || y > y1) continue;
      const [px, py] = this._xyToCanvas(x, y, r);
      ctx.beginPath();
      ctx.arc(px, py, Math.max(2, r.s * 0.5), 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  _drawTip(ctx, r) {
    const [px, py] = this._xyToCanvas(state.tip.x, state.tip.y, r);
    ctx.strokeStyle = '#ffb020';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px - 14, py); ctx.lineTo(px - 4, py);
    ctx.moveTo(px + 4, py); ctx.lineTo(px + 14, py);
    ctx.moveTo(px, py - 14); ctx.lineTo(px, py - 4);
    ctx.moveTo(px, py + 4); ctx.lineTo(px, py + 14);
    ctx.stroke();
  }

  _drawProfileLine(ctx, r) {
    if (this.profilePoints.length === 0) return;
    ctx.strokeStyle = '#6aa9dd';
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    this.profilePoints.forEach((p, i) => {
      const [x, y] = this._xyToCanvas(p[0], p[1], r);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      ctx.rect(x - 2, y - 2, 4, 4);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _renderOverlay() {
    const s = this.scan;
    const p = s.provenance || {};
    const conv = s.convergence || {};
    const unit = s.units[s.channel] || '';
    const notConverged = conv.converged === false;
    const rows = [
      ['technique', `${s.technique}  ${s.mode}`],
      ['channel', `${s.channel}${unit ? '  [' + unit + ']' : ''}`],
      ['pixels', `${s.shape[1]} x ${s.shape[0]}`],
      ['pitch', `${num(s.pixel_size_A[0], 4)} A/px`],
      ['min', num(s.statistics.min, 5)],
      ['max', num(s.statistics.max, 5)],
      ['p-p', num(s.statistics.max - s.statistics.min, 5)],
      ['rms', num(s.statistics.std, 5)],
      'rule',
      ['model', String(p.model || '').replace('microscopy/', '')],
      ['seed', p.seed === null || p.seed === undefined ? '--' : String(p.seed)],
      ['acq time', `${num(s.wall_time_s, 4)} s`],
    ];
    if (conv.residual !== null && conv.residual !== undefined) {
      rows.push(['feedback', Number(conv.residual).toExponential(1),
        notConverged ? 'fault' : '']);
    }
    setHud(this.overlay, 'probe', rows, {
      id: s.technique,
      foot: 'Bright = signal maximum, not necessarily a nucleus.',
      footAlert: true,
    });
  }

  _renderControls() {
    const wrap = this.controls;
    wrap.classList.add('interactive');
    clear(wrap);
    if (!this.scan) { wrap.innerHTML = ''; return; }
    const row = (label, control) => {
      const d = el('div', { style: { display: 'flex', alignItems: 'center', gap: '5px',
        justifyContent: 'space-between' } });
      d.append(el('span.k', { text: label }), control);
      return d;
    };
    const channelSel = el('select', {
      style: { fontSize: '10px', height: '18px' },
      onchange: async (e) => {
        const payload = await api.scanPayload(this.scan.key, e.target.value);
        set({ scan: payload, channel: e.target.value });
        this.setScan(payload);
      },
    }, this.scan.channels.map((c) => el('option', { value: c, selected: c === this.scan.channel }, c)));
    const paletteSel = el('select', {
      style: { fontSize: '10px', height: '18px' },
      onchange: (e) => { this.lut = buildLut(e.target.value); set({ palette: e.target.value }); this.draw(); },
    }, Object.keys(PALETTES).map((p) => el('option', { value: p, selected: p === state.palette }, p)));
    const filterSel = el('select', {
      style: { fontSize: '10px', height: '18px' },
      onchange: async (e) => {
        const method = e.target.value;
        set({ filterMethod: method });
        if (method === 'none') {
          const payload = await api.scanPayload(this.scan.key, this.scan.channel);
          this.setScan(payload);
          return;
        }
        const filtered = await api.scanFiltered(this.scan.key, this.scan.channel, method);
        this.values = decodeFloat32(filtered.data_b64);
        this._computeLimits();
        this.draw();
      },
    }, ['none', 'plane', 'line', 'median', 'plane+median', 'plane+line', 'plane+line+median']
      .map((m) => el('option', { value: m, selected: m === state.filterMethod }, m)));

    const slider = (label, key, min, max, step) => {
      const input = el('input', {
        type: 'range', min, max, step, value: state[key],
        style: { width: '86px' },
        oninput: (e) => { set({ [key]: parseFloat(e.target.value) }); this.draw(); },
      });
      return row(label, input);
    };
    const check = (label, key, fn) => {
      const input = el('input', {
        type: 'checkbox', checked: state[key],
        onchange: (e) => { set({ [key]: e.target.checked }); if (fn) fn(e.target.checked); this.draw(); },
      });
      return row(label, input);
    };
    wrap.append(
      row('channel', channelSel),
      row('palette', paletteSel),
      row('filter', filterSel),
      slider('contrast', 'contrast', 0.2, 3, 0.05),
      slider('brightness', 'brightness', -0.5, 0.5, 0.02),
      slider('gamma', 'gamma', 0.3, 3, 0.05),
      check('features', 'showFeatures', async (on) => {
        if (on && !state.scanFeatures) {
          set({ scanFeatures: await api.scanFeatures(this.scan.key) });
        }
        this.draw();
      }),
      check('atom overlay', 'showOverlay'),
      check('tip marker', 'showTip'),
    );
  }

  _drawScaleBar(r) {
    const [x0, , x1] = this.scan.extent_A;
    const angstromPerPixel = (x1 - x0) / Math.max(this.shape[1] - 1, 1);
    const rect = this.canvas.getBoundingClientRect();
    const dpr = this.canvas.width / rect.width;
    const pxPerAngstrom = r.s / angstromPerPixel / dpr;
    const a = niceNumber(110 / pxPerAngstrom);
    this.scaleEl.innerHTML =
      `<span class="bar" style="width:${(a * pxPerAngstrom).toFixed(0)}px"></span>` +
      `<span>${a >= 10 ? `${num(a / 10, 3)} nm` : `${num(a, 3)} Å`}</span>`;
  }

  _drawColorbar() {
    const unit = this.scan.units[this.scan.channel] || '';
    const n = 48;
    const cells = [];
    for (let i = 0; i < n; i++) {
      const k = Math.round((i / (n - 1)) * 255);
      cells.push(`<span style="display:inline-block;width:4px;height:10px;background:rgb(` +
        `${this.lut[k * 3]},${this.lut[k * 3 + 1]},${this.lut[k * 3 + 2]})"></span>`);
    }
    setHud(this.colorbar, 'scale', [
      ['map', state.palette],
      ['min', `${num(this.limits.lo, 5)} ${unit}`],
      ['max', `${num(this.limits.hi, 5)} ${unit}`]]);
    const bar = el('div', { style: { padding: '0 7px 4px 7px', lineHeight: '0',
      display: 'flex', border: '0' } });
    bar.innerHTML = cells.join('');
    this.colorbar.append(bar);
  }

  resize() { this.draw(); }
}
