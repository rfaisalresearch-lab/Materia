/**
 * Wafer and microstructure views.
 *
 * Both are 2-D canvas renderings of the *procedural* wafer description. They
 * draw a synthetic microstructure generated from the wafer seed, and label it
 * as such: nothing here is measured data.
 * @module wafer-view
 */

import { api } from './api.js';
import { set, state, subscribe } from './state.js';
import { graticule, setHud } from './hud.js';
import { drawCallout } from './locator.js';
import { $, clear, cssVar, el, fitCanvas, fmtLength, niceNumber, num } from './util.js';

const WAFER_SCALE_INDEX = 0;

export class WaferView {
  constructor(canvasId, overlayId, scaleId, { microstructure = false } = {}) {
    this.canvas = $(canvasId);
    this.overlay = $(overlayId);
    this.scaleEl = $(scaleId);
    this.micro = microstructure;
    this.ctx = this.canvas.getContext('2d');
    this.zoom = 1;
    this.centre = { x: 0, y: 0 };
    this.roi = { x: 0, y: 0 };
    this.hover = null;
    this._bind();
  }

  _bind() {
    const c = this.canvas;
    c.addEventListener('mousemove', (e) => this._onMove(e));
    c.addEventListener('mouseleave', () => { this.hover = null; this.draw(); });
    c.addEventListener('click', (e) => this._onClick(e));
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const f = Math.exp(-e.deltaY * 0.0015);
      this.zoom = Math.max(0.4, Math.min(4.0e6, this.zoom * f));
      this._zoomTouched = true;
      this.draw();
    }, { passive: false });
    let drag = null;
    c.addEventListener('mousedown', (e) => {
      if (e.button === 1 || e.shiftKey) drag = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('mouseup', () => { drag = null; });
    window.addEventListener('mousemove', (e) => {
      if (!drag) return;
      const s = this._scale();
      this.centre.x -= (e.clientX - drag.x) / s;
      this.centre.y += (e.clientY - drag.y) / s;
      drag = { x: e.clientX, y: e.clientY };
      this.draw();
    });
    c.addEventListener('keydown', (e) => {
      const step = 1 / this._scale() * 12;
      if (e.key === 'ArrowLeft') { this.roi.x -= step; this._moved(); }
      else if (e.key === 'ArrowRight') { this.roi.x += step; this._moved(); }
      else if (e.key === 'ArrowUp') { this.roi.y += step; this._moved(); }
      else if (e.key === 'ArrowDown') { this.roi.y -= step; this._moved(); }
      else return;
      e.preventDefault();
    });
  }

  _moved() {
    set({ roi: { x: this.roi.x, y: this.roi.y } });
    this.draw();
    this._probe();
  }

  _scale() {
    const [w, h] = [this.canvas.width, this.canvas.height];
    const wafer = state.server && state.server.wafer;
    const radius = wafer ? wafer.radius_mm : 150;
    const base = Math.min(w, h) / (2.2 * radius);
    return base * this.zoom;
  }

  _toScreen(xmm, ymm) {
    const s = this._scale();
    return [this.canvas.width / 2 + (xmm - this.centre.x) * s,
            this.canvas.height / 2 - (ymm - this.centre.y) * s];
  }

  _toWorld(px, py) {
    const s = this._scale();
    const dpr = this.canvas.width / this.canvas.getBoundingClientRect().width;
    return [(px * dpr - this.canvas.width / 2) / s + this.centre.x,
            -(py * dpr - this.canvas.height / 2) / s + this.centre.y];
  }

  _onMove(e) {
    const r = this.canvas.getBoundingClientRect();
    const [x, y] = this._toWorld(e.clientX - r.left, e.clientY - r.top);
    this.hover = { x, y };
    $('#status-coords').textContent =
      `x ${x.toFixed(3)} mm   y ${y.toFixed(3)} mm`;
    this.draw();
  }

  async _onClick(e) {
    if (e.shiftKey) return;
    const r = this.canvas.getBoundingClientRect();
    const [x, y] = this._toWorld(e.clientX - r.left, e.clientY - r.top);
    this.roi = { x, y };
    set({ roi: { x, y } });
    this.draw();
    await this._probe();
  }

  async _probe() {
    if (!state.server || !state.server.wafer) return;
    try {
      const info = await api.waferProbe(this.roi.x, this.roi.y);
      set({ waferProbe: info });
      this.renderOverlay(info);
    } catch (err) { /* reported by the global handler */ }
  }

  async loadMap(field = 'roughness') {
    if (!state.server || !state.server.wafer) return;
    const map = await api.waferMap(field, 110);
    set({ waferMap: map });
    this.draw();
  }

  renderOverlay(info) {
    const wafer = state.server && state.server.wafer;
    if (!wafer) {
      setHud(this.overlay, this.micro ? 'microstructure' : 'specimen',
        [['status', 'no specimen'], ['stage', 'empty'], ['signal', 'none']],
        { foot: 'Build \u203a Wafer loads a specimen.' });
      return;
    }
    const w = wafer.spec;
    const rows = [
      ['material', w.material_id],
      ['diameter', `${num(w.diameter_mm, 5)} mm`],
      ['thickness', `${num(w.thickness_um, 5)} um`],
      ['orientation', `(${w.orientation.join('')})`],
      ['edge', w.edge_feature],
      ['miscut', `${num(w.miscut_deg, 3)} deg`],
      ['dopant', w.dopant ? `${w.dopant}  ${w.dopant_concentration_cm3.toExponential(1)} cm-3`
        : 'none'],
      ['temp', `${num(w.temperature_K, 4)} K`],
      ['seed', String(w.seed)],
    ];
    if (info && info.on_wafer) {
      rows.push('rule');
      rows.push(['cursor x', `${num(this.roi.x, 5)} mm`]);
      rows.push(['cursor y', `${num(this.roi.y, 5)} mm`]);
      rows.push(['height', `${num(info.local_height_A, 3)} A`]);
      if (info.terrace_width_nm && Number.isFinite(info.terrace_width_nm)) {
        rows.push(['terrace', `${num(info.terrace_width_nm, 4)} nm`]);
      }
      if (info.grain) rows.push(['grain', `#${info.grain.grain_id}`]);
      if (info.device_region) rows.push(['die', info.device_region]);
    }
    setHud(this.overlay, this.micro ? 'microstructure' : 'specimen', rows, {
      id: `${w.material_id}(${w.orientation.join('')})`,
      foot: 'Procedural microstructure, seeded. Synthetic, not measured.',
      footAlert: true,
    });
  }

  draw() {
    if (this.micro && state.roi) {
      const key = `${state.roi.x},${state.roi.y}`;
      if (key !== this._lastRoiKey) {
        this._lastRoiKey = key;
        this.syncRoi();
      }
    }
    const [w, h] = fitCanvas(this.canvas);
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = cssVar('--viewport');
    ctx.fillRect(0, 0, w, h);

    const wafer = state.server && state.server.wafer;
    if (!wafer) {
      graticule(ctx, w, h, { step: 54, label: 'NO SPECIMEN' });
      ctx.restore();
      this.renderOverlay(null);
      this.scaleEl.innerHTML = '';
      return;
    }

    if (state.waferMap) this._drawField(ctx);
    this._drawOutline(ctx, wafer);
    this._drawDeviceRegions(ctx, wafer);
    this._drawRoi(ctx);
    this._drawScaleBar();
    this.renderOverlay(state.waferProbe);
    this._drawLocator(wafer);
    ctx.restore();
  }

  _drawGrid(ctx, w, h) {
    const s = this._scale();
    const stepMm = niceNumber(60 / s);
    ctx.strokeStyle = cssVar('--viewport-grid');
    ctx.lineWidth = 1;
    ctx.beginPath();
    const [x0, y0] = this._toWorld(0, h / (this.canvas.height / this.canvas.getBoundingClientRect().height));
    const left = this.centre.x - (w / 2) / s, right = this.centre.x + (w / 2) / s;
    const bottom = this.centre.y - (h / 2) / s, top = this.centre.y + (h / 2) / s;
    for (let x = Math.ceil(left / stepMm) * stepMm; x <= right; x += stepMm) {
      const [sx] = this._toScreen(x, 0);
      ctx.moveTo(sx, 0); ctx.lineTo(sx, h);
    }
    for (let y = Math.ceil(bottom / stepMm) * stepMm; y <= top; y += stepMm) {
      const [, sy] = this._toScreen(0, y);
      ctx.moveTo(0, sy); ctx.lineTo(w, sy);
    }
    ctx.stroke();
  }

  _drawField(ctx) {
    if (this.micro) { this._drawMicrostructure(ctx); return; }
    const map = state.waferMap;
    const n = map.data.length;
    const xs = map.x_mm, ys = map.y_mm;
    const dx = (xs[1] - xs[0]);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const v = map.data[j][i];
        if (v === null) continue;
        const t = map.field === 'grain' ? (v % 32) / 32 : (v + 1) / 2;
        const g = Math.round(28 + t * 120);
        ctx.fillStyle = map.field === 'grain'
          ? `hsl(${Math.round(t * 360)}, 22%, 34%)`
          : `rgb(${g},${g + 2},${g + 5})`;
        const [px, py] = this._toScreen(xs[i], ys[j]);
        const s = this._scale() * dx;
        ctx.fillRect(px - s / 2, py - s / 2, s + 1, s + 1);
      }
    }
  }

  _drawMicrostructure(ctx) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const step = 3;
    const wafer = state.server && state.server.wafer;
    const rough = (wafer && wafer.spec.roughness_rms_A) || 0.8;
    const correlation = (wafer && wafer.spec.roughness_correlation_nm) || 40;
    const lambdaMm = correlation * 1e-6;
    const s = this._scale();
    const image = ctx.createImageData(Math.ceil(w / step), Math.ceil(h / step));
    const data = image.data;
    let k = 0;
    for (let py = 0; py < h; py += step) {
      for (let px = 0; px < w; px += step) {
        const x = this.centre.x + (px - w / 2) / s;
        const y = this.centre.y - (py - h / 2) / s;
        const v = valueNoise(x, y, lambdaMm, wafer ? wafer.spec.seed : 0);
        const level = Math.round(34 + 96 * (v + 1) / 2);
        data[k++] = level;
        data[k++] = level + 2;
        data[k++] = level + 5;
        data[k++] = 255;
      }
    }
    const buffer = document.createElement('canvas');
    buffer.width = image.width; buffer.height = image.height;
    buffer.getContext('2d').putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(buffer, 0, 0, w, h);
    void rough;
  }

  _drawOutline(ctx, wafer) {
    if (this.micro) return;
    const pts = wafer.outline;
    ctx.beginPath();
    pts.forEach(([x, y], i) => {
      const [sx, sy] = this._toScreen(x, y);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    });
    ctx.closePath();
    ctx.strokeStyle = '#8d959c';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = 'rgba(120,132,142,.06)';
    ctx.fill();

    const [cx, cy] = this._toScreen(0, 0);
    ctx.strokeStyle = '#4a525a';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, cy - 14); ctx.lineTo(cx, cy + 14);
    ctx.moveTo(cx - 14, cy); ctx.lineTo(cx + 14, cy);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#6d757c';
    ctx.font = "10px 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Tahoma, sans-serif";
    ctx.textAlign = 'left';
    ctx.fillText(`(${wafer.spec.orientation.join('')})`, cx + 6, cy - 6);
  }

  _drawDeviceRegions(ctx, wafer) {
    if (this.micro) return;
    for (const r of wafer.spec.device_regions || []) {
      const [x0, y0] = this._toScreen(r.x_mm - r.width_mm / 2, r.y_mm + r.height_mm / 2);
      const s = this._scale();
      ctx.strokeStyle = '#7d8e9d';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, r.width_mm * s, r.height_mm * s);
      if (r.width_mm * s > 40) {
        ctx.fillStyle = '#93a2af';
        ctx.font = "9px 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Tahoma, sans-serif";
        ctx.fillText(r.name, x0 + 3, y0 + 11);
      }
    }
  }

  _drawRoi(ctx) {
    const [x, y] = this._toScreen(this.roi.x, this.roi.y);
    ctx.strokeStyle = '#ffb020';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x - 11, y); ctx.lineTo(x - 3, y);
    ctx.moveTo(x + 3, y); ctx.lineTo(x + 11, y);
    ctx.moveTo(x, y - 11); ctx.lineTo(x, y - 3);
    ctx.moveTo(x, y + 3); ctx.lineTo(x, y + 11);
    ctx.stroke();
    ctx.strokeRect(x - 7, y - 7, 14, 14);
    ctx.fillStyle = '#ffb020';
    ctx.font = "9.5px 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Tahoma, sans-serif";
    ctx.textAlign = 'left';
    ctx.fillText('ROI', x + 10, y - 9);
  }

  _drawScaleBar() {
    const s = this._scale();
    const targetPx = 110;
    const mm = niceNumber(targetPx / s);
    const px = mm * s;
    const dpr = this.canvas.width / this.canvas.getBoundingClientRect().width;
    this.scaleEl.innerHTML =
      `<span class="bar" style="width:${(px / dpr).toFixed(0)}px"></span>` +
      `<span>${mm >= 1 ? `${num(mm, 3)} mm` : `${num(mm * 1000, 3)} µm`}</span>`;
  }

  _drawLocator(wafer) {
    if (!this.micro) return;
    const canvas = document.getElementById('canvas-region-callout');
    if (!canvas) return;
    const spanMm = this.canvas.width / this._scale();
    drawCallout(canvas, wafer, this.roi, {
      spanMm, label: `field of view ${spanMm < 1 ? (spanMm * 1000).toFixed(1) + ' um'
        : spanMm.toFixed(3) + ' mm'}` });
  }

  syncRoi() {
    if (!state.roi) return;
    this.roi = { x: state.roi.x, y: state.roi.y };
    if (this.micro) {
      // The microstructure view exists to show the micrometre scale, so it
      // follows the region of interest and opens at a 100 um field of view
      // rather than repeating the whole-wafer picture.
      this.centre = { x: this.roi.x, y: this.roi.y };
      if (!this._zoomTouched) {
        const wafer = state.server && state.server.wafer;
        if (wafer) {
          const base = Math.min(this.canvas.width, this.canvas.height)
            / (2.2 * wafer.radius_mm);
          this.zoom = (this.canvas.width / 0.1) / base;
        }
      }
    }
  }

  resize() { this.draw(); }
}


/**
 * Band-limited value noise matching the procedural roughness field the core
 * generates, so the microstructure view and the extracted region agree.
 */
function valueNoise(xMm, yMm, lambdaMm, seed) {
  let total = 0;
  let amplitude = 1;
  let norm = 0;
  for (let octave = 0; octave < 3; octave++) {
    const scale = lambdaMm / Math.pow(2, octave);
    const gx = Math.floor(xMm / scale);
    const gy = Math.floor(yMm / scale);
    const fx = xMm / scale - gx;
    const fy = yMm / scale - gy;
    const corner = (ix, iy) => {
      let h = (ix * 374761393 + iy * 668265263 + octave * 2246822519
               + seed * 3266489917) | 0;
      h = Math.imul(h ^ (h >>> 13), 1274126177);
      return (((h ^ (h >>> 16)) >>> 0) % 2000003) / 2000003 * 2 - 1;
    };
    const sx = fx * fx * (3 - 2 * fx);
    const sy = fy * fy * (3 - 2 * fy);
    const top = corner(gx, gy) * (1 - sy) + corner(gx, gy + 1) * sy;
    const bottom = corner(gx + 1, gy) * (1 - sy) + corner(gx + 1, gy + 1) * sy;
    total += amplitude * (top * (1 - sx) + bottom * sx);
    norm += amplitude;
    amplitude *= 0.5;
  }
  return total / Math.max(norm, 1e-12);
}
