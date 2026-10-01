/**
 * Atomic-model viewport (WebGL2, with a 2-D canvas fallback).
 *
 * Atoms are drawn as GPU-instanced ray-traced sphere impostors, which keeps
 * a hundred thousand atoms interactive without tessellating geometry. Bonds
 * are line segments. Picking is done on the CPU against the projected
 * positions, which is exact and needs no extra render pass.
 *
 * Element colours here are a *display convention* for the structural model.
 * They are never applied to scanning-probe data: that view stays grayscale
 * until the user asks for a labelled colour map.
 * @module atoms-view
 */

import { graticule, setHud } from './hud.js';
import { drawCallout } from './locator.js';
import { $, cssVar, el, fitCanvas, niceNumber, num } from './util.js';
import { set, state } from './state.js';

/** Muted, print-safe element colours (CPK-derived, desaturated). */
export const ELEMENT_COLOURS = {
  H: [0.88, 0.88, 0.86], C: [0.36, 0.38, 0.40], N: [0.36, 0.48, 0.74],
  O: [0.80, 0.34, 0.30], F: [0.56, 0.78, 0.56], Si: [0.70, 0.66, 0.52],
  P: [0.85, 0.60, 0.28], S: [0.84, 0.78, 0.36], Cl: [0.48, 0.76, 0.46],
  Ga: [0.66, 0.50, 0.44], As: [0.58, 0.46, 0.66], Ge: [0.56, 0.60, 0.62],
  In: [0.62, 0.50, 0.58], Sb: [0.56, 0.50, 0.66], Se: [0.70, 0.58, 0.28],
  Mo: [0.44, 0.58, 0.66], W: [0.40, 0.50, 0.58], Au: [0.84, 0.70, 0.34],
  Ag: [0.78, 0.80, 0.82], Cu: [0.78, 0.52, 0.36], Pt: [0.74, 0.76, 0.78],
  Ni: [0.56, 0.68, 0.56], Ti: [0.66, 0.68, 0.70], Al: [0.72, 0.72, 0.76],
  B: [0.78, 0.62, 0.58], Zn: [0.62, 0.66, 0.70], Sn: [0.60, 0.62, 0.64],
};
const DEFAULT_COLOUR = [0.62, 0.64, 0.66];

const VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 corner;
layout(location=1) in vec3 centre;
layout(location=2) in float radius;
layout(location=3) in vec3 colour;
layout(location=4) in float flags;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vColour;
out vec2 vCorner;
out float vRadius;
out vec3 vEyePos;
out float vFlags;
void main() {
  vec4 eye = uView * vec4(centre, 1.0);
  vEyePos = eye.xyz;
  vCorner = corner;
  vRadius = radius;
  vColour = colour;
  vFlags = flags;
  eye.xy += corner * radius;
  gl_Position = uProj * eye;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vColour;
in vec2 vCorner;
in float vRadius;
in vec3 vEyePos;
in float vFlags;
uniform mat4 uProj;
uniform vec3 uLight;
out vec4 fragColour;
void main() {
  float r2 = dot(vCorner, vCorner);
  if (r2 > 1.0) discard;
  float z = sqrt(1.0 - r2);
  vec3 normal = vec3(vCorner, z);
  vec3 pos = vEyePos + vec3(vCorner * vRadius, z * vRadius);
  vec4 clip = uProj * vec4(pos, 1.0);
  gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;

  float diff = max(dot(normal, normalize(uLight)), 0.0);
  vec3 h = normalize(normalize(uLight) + vec3(0.0, 0.0, 1.0));
  float spec = pow(max(dot(normal, h), 0.0), 28.0);
  vec3 base = vColour;
  if (vFlags > 1.5) base = mix(base, vec3(1.0, 0.69, 0.13), 0.80);
  else if (vFlags > 0.5) base = mix(base, vec3(0.18, 0.71, 0.77), 0.45);
  vec3 lit = base * (0.26 + 0.74 * diff) + vec3(0.9) * spec * 0.28;
  float rim = pow(1.0 - z, 3.0) * 0.22;
  fragColour = vec4(lit + rim, 1.0);
}`;

const LINE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 position;
uniform mat4 uView;
uniform mat4 uProj;
void main() { gl_Position = uProj * uView * vec4(position, 1.0); }`;

const LINE_FS = `#version 300 es
precision highp float;
uniform vec4 uColour;
out vec4 fragColour;
void main() { fragColour = uColour; }`;

function compile(gl, type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error('Shader compilation failed: ' + gl.getShaderInfoLog(s));
  }
  return s;
}
function link(gl, vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('Program link failed: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0]);
}
function multiply(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
    o[i * 4 + j] = s;
  }
  return o;
}
function identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}
function translate(m, x, y, z) {
  const t = identity(); t[12] = x; t[13] = y; t[14] = z; return multiply(m, t);
}
function rotateX(m, a) {
  const c = Math.cos(a), s = Math.sin(a);
  const r = identity(); r[5] = c; r[6] = s; r[9] = -s; r[10] = c;
  return multiply(m, r);
}
function rotateY(m, a) {
  const c = Math.cos(a), s = Math.sin(a);
  const r = identity(); r[0] = c; r[2] = -s; r[8] = s; r[10] = c;
  return multiply(m, r);
}

export class AtomsView {
  constructor(canvasId, overlayId, legendId, scaleId, { onPick } = {}) {
    this.canvas = $(canvasId);
    this.overlay = $(overlayId);
    this.legend = $(legendId);
    this.scaleEl = $(scaleId);
    this.onPick = onPick || (() => {});
    this.rotX = -0.38;
    this.rotY = 0.55;
    this.distance = 40;
    this.pan = { x: 0, y: 0 };
    this.styleScale = 0.36;
    this.showBonds = true;
    this.showCell = true;
    this.sliceZ = null;
    this.gl = null;
    this.data = null;
    this._boxStart = null;
    this._init();
    this._bind();
  }

  _init() {
    try {
      this.gl = this.canvas.getContext('webgl2', { antialias: true, alpha: false });
    } catch { this.gl = null; }
    if (!this.gl) {
      this.fallback = this.canvas.getContext('2d');
      return;
    }
    const gl = this.gl;
    this.prog = link(gl, VS, FS);
    this.lineProg = link(gl, LINE_VS, LINE_FS);
    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.centreBuf = gl.createBuffer();
    this.radiusBuf = gl.createBuffer();
    this.colourBuf = gl.createBuffer();
    this.flagBuf = gl.createBuffer();
    this.lineBuf = gl.createBuffer();
    this.cellBuf = gl.createBuffer();
    gl.enable(gl.DEPTH_TEST);
  }

  _bind() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener('mousedown', (e) => {
      c.focus();
      if (state.tool === 'box' && e.button === 0 && !e.shiftKey) {
        const r = c.getBoundingClientRect();
        this._boxStart = { x: e.clientX - r.left, y: e.clientY - r.top };
        return;
      }
      drag = { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey,
               moved: false };
    });
    window.addEventListener('mousemove', (e) => {
      if (this._boxStart) {
        const r = c.getBoundingClientRect();
        this._boxNow = { x: e.clientX - r.left, y: e.clientY - r.top };
        this.draw();
        return;
      }
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
      if (drag.button === 0 && !drag.shift && state.tool === 'navigate') {
        this.rotY += dx * 0.008;
        this.rotX += dy * 0.008;
        this.rotX = Math.max(-1.55, Math.min(1.55, this.rotX));
      } else if (drag.button === 1 || drag.shift) {
        const s = this.distance * 0.0016;
        this.pan.x -= dx * s;
        this.pan.y += dy * s;
      }
      drag.x = e.clientX; drag.y = e.clientY;
      this.draw();
    });
    window.addEventListener('mouseup', (e) => {
      if (this._boxStart && this._boxNow) {
        this._finishBox();
      } else if (drag && !drag.moved && drag.button === 0) {
        this._pick(e);
      }
      drag = null; this._boxStart = null; this._boxNow = null;
      this.draw();
    });
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.distance *= Math.exp(e.deltaY * 0.0012);
      this.distance = Math.max(0.15, Math.min(4000, this.distance));
      this.draw();
    }, { passive: false });
    c.addEventListener('mousemove', (e) => this._hover(e));
    c.addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      if (k === 'b') { this.showBonds = !this.showBonds; this.draw(); }
      else if (k === 'c') { this.showCell = !this.showCell; this.draw(); }
      else if (k === '+' || k === '=') { this.styleScale = Math.min(1.0, this.styleScale + 0.04); this.draw(); }
      else if (k === '-') { this.styleScale = Math.max(0.08, this.styleScale - 0.04); this.draw(); }
      else if (k === 'r') { this.frame(); }
      else return;
      e.preventDefault();
    });
  }

  /** Install a render payload from /api/structure/render. */
  setData(payload) {
    this.data = payload ? { ...payload,
      bounds: payload.bounds ? payload.bounds.map((row) => [...row]) : payload.bounds } : null;
    if (!payload) { this.draw(); return; }
    const n = payload.ids.length;
    this.positions = new Float32Array(payload.positions);
    this.radii = new Float32Array(n);
    this.colours = new Float32Array(n * 3);
    this.flags = new Float32Array(n);
    const symbols = {};
    for (const [sym, z] of Object.entries(payload.elements)) symbols[z] = sym;
    for (let i = 0; i < n; i++) {
      const sym = symbols[payload.numbers[i]] || 'X';
      const c = ELEMENT_COLOURS[sym] || DEFAULT_COLOUR;
      this.colours[i * 3] = c[0]; this.colours[i * 3 + 1] = c[1]; this.colours[i * 3 + 2] = c[2];
      this.radii[i] = payload.radii[i];
      const role = payload.roles[i];
      this.flags[i] = payload.selected[i] ? 2 : (role === 'dopant' || role === 'adatom'
        || role === 'interstitial' || role === 'adsorbate' ? 1 : 0);
    }
    this._uploadBuffers();
    this.frame();
    this._renderLegend();
  }

  updateSelection(ids) {
    if (!this.data) return;
    const sel = new Set(ids);
    for (let i = 0; i < this.data.ids.length; i++) {
      const role = this.data.roles[i];
      this.flags[i] = sel.has(this.data.ids[i]) ? 2
        : (role === 'dopant' || role === 'adatom' || role === 'interstitial'
           || role === 'adsorbate' ? 1 : 0);
    }
    if (this.gl) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.flagBuf);
      gl.bufferData(gl.ARRAY_BUFFER, this.flags, gl.DYNAMIC_DRAW);
    }
    this.draw();
  }

  setTrajectoryFrame(payload) {
    if (!this.data || !payload || !payload.ids) return false;
    const lookup = new Map(payload.ids.map((id, index) => [id, index]));
    const palette = [
      [0.10, 0.43, 0.90], [0.20, 0.58, 1.00], [0.08, 0.31, 0.68],
      [0.36, 0.68, 1.00], [0.16, 0.50, 0.78], [0.48, 0.77, 1.00],
    ];
    for (let i = 0; i < this.data.ids.length; i++) {
      const source = lookup.get(this.data.ids[i]);
      if (source === undefined) return false;
      this.positions[i * 3] = payload.positions[source * 3];
      this.positions[i * 3 + 1] = payload.positions[source * 3 + 1];
      this.positions[i * 3 + 2] = payload.positions[source * 3 + 2];
      const colour = palette[(payload.fragment_labels[source] || 0) % palette.length];
      this.colours[i * 3] = colour[0];
      this.colours[i * 3 + 1] = colour[1];
      this.colours[i * 3 + 2] = colour[2];
    }
    this.data.bounds = payload.bounds;
    if (this.gl) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.centreBuf);
      gl.bufferData(gl.ARRAY_BUFFER, this.positions, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.colourBuf);
      gl.bufferData(gl.ARRAY_BUFFER, this.colours, gl.DYNAMIC_DRAW);
    }
    this.lineCount = 0;
    this.draw();
    return true;
  }

  _uploadBuffers() {
    if (!this.gl || !this.data) return;
    const gl = this.gl;
    const upload = (buf, arr) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
    };
    upload(this.centreBuf, this.positions);
    upload(this.radiusBuf, this.radii);
    upload(this.colourBuf, this.colours);
    upload(this.flagBuf, this.flags);

    const bonds = this.data.bonds || [];
    const lines = new Float32Array(bonds.length * 3);
    for (let i = 0; i < bonds.length; i++) {
      const a = bonds[i];
      lines[i * 3] = this.positions[a * 3];
      lines[i * 3 + 1] = this.positions[a * 3 + 1];
      lines[i * 3 + 2] = this.positions[a * 3 + 2];
    }
    upload(this.lineBuf, lines);
    this.lineCount = bonds.length;

    const m = this.data.cell;
    const corners = [];
    const add = (a, b) => { corners.push(...a, ...b); };
    const V = (i, j, k) => [
      i * m[0][0] + j * m[1][0] + k * m[2][0],
      i * m[0][1] + j * m[1][1] + k * m[2][1],
      i * m[0][2] + j * m[1][2] + k * m[2][2]];
    for (const [a, b] of [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]],
      [[1, 0, 0], [1, 1, 0]], [[1, 0, 0], [1, 0, 1]], [[0, 1, 0], [1, 1, 0]],
      [[0, 1, 0], [0, 1, 1]], [[0, 0, 1], [1, 0, 1]], [[0, 0, 1], [0, 1, 1]],
      [[1, 1, 0], [1, 1, 1]], [[1, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]]) {
      add(V(...a), V(...b));
    }
    upload(this.cellBuf, new Float32Array(corners));
    this.cellCount = corners.length / 3;
  }

  frame() {
    if (!this.data) return;
    const [lo, hi] = this.data.bounds;
    this.centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 4);
    this.distance = span * 1.8;
    this.pan = { x: 0, y: 0 };
    this.draw();
  }

  _viewMatrix() {
    let m = identity();
    m = translate(m, -this.pan.x, -this.pan.y, -this.distance);
    m = rotateX(m, this.rotX);
    m = rotateY(m, this.rotY);
    const c = this.centre || [0, 0, 0];
    m = translate(m, -c[0], -c[1], -c[2]);
    return m;
  }

  _projMatrix(w, h) {
    return perspective(0.7, w / h, Math.max(0.2, this.distance * 0.02),
      this.distance * 8 + 200);
  }

  draw() {
    const [w, h] = fitCanvas(this.canvas);
    if (!this.gl) return this._drawFallback(w, h);
    const gl = this.gl;
    gl.viewport(0, 0, w, h);
    const bg = cssVar('--viewport') || '#17181a';
    const rgb = bg.startsWith('#')
      ? [parseInt(bg.slice(1, 3), 16) / 255, parseInt(bg.slice(3, 5), 16) / 255,
         parseInt(bg.slice(5, 7), 16) / 255] : [0.09, 0.09, 0.10];
    gl.clearColor(rgb[0], rgb[1], rgb[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.data) {
      this._renderOverlay();
      this._renderEmptyGraticule();
      this.scaleEl.innerHTML = '';
      return;
    }

    const view = this._viewMatrix();
    const proj = this._projMatrix(w, h);

    if (this.showCell && this.cellCount) {
      gl.useProgram(this.lineProg);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, 'uView'), false, view);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, 'uProj'), false, proj);
      gl.uniform4f(gl.getUniformLocation(this.lineProg, 'uColour'), 0.36, 0.40, 0.44, 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cellBuf);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(0, 0);
      gl.drawArrays(gl.LINES, 0, this.cellCount);
    }

    if (this.showBonds && this.lineCount) {
      gl.useProgram(this.lineProg);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, 'uView'), false, view);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, 'uProj'), false, proj);
      gl.uniform4f(gl.getUniformLocation(this.lineProg, 'uColour'), 0.58, 0.60, 0.62, 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(0, 0);
      gl.drawArrays(gl.LINES, 0, this.lineCount);
    }

    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, 'uView'), false, view);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, 'uProj'), false, proj);
    gl.uniform3f(gl.getUniformLocation(this.prog, 'uLight'), -0.4, 0.55, 0.75);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(0, 0);

    const bindInstanced = (loc, buf, size) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(loc, 1);
    };
    bindInstanced(1, this.centreBuf, 3);
    bindInstanced(2, this.radiusBuf, 1);
    bindInstanced(3, this.colourBuf, 3);
    bindInstanced(4, this.flagBuf, 1);

    // radius scaling is applied by re-uploading a scaled buffer only when needed
    if (this._lastStyleScale !== this.styleScale) {
      const scaled = new Float32Array(this.radii.length);
      for (let i = 0; i < scaled.length; i++) scaled[i] = this.radii[i] * this.styleScale * 2.2;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.radiusBuf);
      gl.bufferData(gl.ARRAY_BUFFER, scaled, gl.DYNAMIC_DRAW);
      this._lastStyleScale = this.styleScale;
    }
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.data.ids.length);

    this._hideEmptyGraticule();
    this._drawBoxOverlay();
    this._renderOverlay();
    this._drawScaleBar();
    this._drawLocator();
  }

  _drawBoxOverlay() {
    if (!this._boxStart || !this._boxNow) return;
    const ov = this.canvas.parentElement.querySelector('.box-overlay')
      || this.canvas.parentElement.appendChild(el('div.box-overlay', {
        style: { position: 'absolute', border: '1px dashed #ffb020',
                 background: 'rgba(255,176,32,.12)', pointerEvents: 'none' } }));
    const x = Math.min(this._boxStart.x, this._boxNow.x);
    const y = Math.min(this._boxStart.y, this._boxNow.y);
    ov.style.left = `${x}px`; ov.style.top = `${y}px`;
    ov.style.width = `${Math.abs(this._boxNow.x - this._boxStart.x)}px`;
    ov.style.height = `${Math.abs(this._boxNow.y - this._boxStart.y)}px`;
    ov.hidden = false;
  }

  _clearBoxOverlay() {
    const ov = this.canvas.parentElement.querySelector('.box-overlay');
    if (ov) ov.hidden = true;
  }

  _project(i, view, proj, w, h) {
    const x = this.positions[i * 3], y = this.positions[i * 3 + 1], z = this.positions[i * 3 + 2];
    const ex = view[0] * x + view[4] * y + view[8] * z + view[12];
    const ey = view[1] * x + view[5] * y + view[9] * z + view[13];
    const ez = view[2] * x + view[6] * y + view[10] * z + view[14];
    const cw = proj[3] * ex + proj[7] * ey + proj[11] * ez + proj[15];
    if (cw <= 0) return null;
    const cx = (proj[0] * ex + proj[8] * ez) / cw;
    const cy = (proj[5] * ey + proj[9] * ez) / cw;
    return [(cx * 0.5 + 0.5) * w, (0.5 - cy * 0.5) * h, ez];
  }

  _pick(e) {
    if (!this.data) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (this.canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (this.canvas.height / rect.height);
    const view = this._viewMatrix();
    const proj = this._projMatrix(this.canvas.width, this.canvas.height);
    let best = -1, bestDist = 26 * (this.canvas.width / rect.width), bestZ = -1e9;
    for (let i = 0; i < this.data.ids.length; i++) {
      const p = this._project(i, view, proj, this.canvas.width, this.canvas.height);
      if (!p) continue;
      const d = Math.hypot(p[0] - px, p[1] - py);
      if (d < bestDist && p[2] > bestZ) { best = i; bestZ = p[2]; bestDist = Math.max(d, 6); }
    }
    if (best >= 0) this.onPick([this.data.ids[best]], e.shiftKey);
    else this.onPick([], e.shiftKey);
  }

  _finishBox() {
    if (!this.data) return;
    const rect = this.canvas.getBoundingClientRect();
    const sx = this.canvas.width / rect.width;
    const x0 = Math.min(this._boxStart.x, this._boxNow.x) * sx;
    const x1 = Math.max(this._boxStart.x, this._boxNow.x) * sx;
    const y0 = Math.min(this._boxStart.y, this._boxNow.y) * sx;
    const y1 = Math.max(this._boxStart.y, this._boxNow.y) * sx;
    const view = this._viewMatrix();
    const proj = this._projMatrix(this.canvas.width, this.canvas.height);
    const ids = [];
    for (let i = 0; i < this.data.ids.length; i++) {
      const p = this._project(i, view, proj, this.canvas.width, this.canvas.height);
      if (!p) continue;
      if (p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1) ids.push(this.data.ids[i]);
    }
    this._clearBoxOverlay();
    this.onPick(ids, true);
  }

  _hover(e) {
    if (!this.data) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (this.canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (this.canvas.height / rect.height);
    const view = this._viewMatrix();
    const proj = this._projMatrix(this.canvas.width, this.canvas.height);
    let best = -1, bestD = 22 * (this.canvas.width / rect.width);
    for (let i = 0; i < this.data.ids.length; i++) {
      const p = this._project(i, view, proj, this.canvas.width, this.canvas.height);
      if (!p) continue;
      const d = Math.hypot(p[0] - px, p[1] - py);
      if (d < bestD) { bestD = d; best = i; }
    }
    const symbols = {};
    for (const [sym, z] of Object.entries(this.data.elements)) symbols[z] = sym;
    if (best >= 0) {
      const i = best;
      $('#status-coords').textContent =
        `x ${this.positions[i * 3].toFixed(3)}  y ${this.positions[i * 3 + 1].toFixed(3)}  ` +
        `z ${this.positions[i * 3 + 2].toFixed(3)} Å`;
      $('#status-value').textContent =
        `#${this.data.ids[i]} ${symbols[this.data.numbers[i]]} · ${this.data.roles[i]}`;
    } else {
      $('#status-value').textContent = '-';
    }
  }

  _renderOverlay() {
    const d = this.data;
    if (!d) {
      setHud(this.overlay, 'atomic model',
        [['status', 'no region'], ['atoms', '0'], ['bonds', '0']],
        { foot: 'Extract a region from the specimen, or Build \u203a Surface.' });
      return;
    }
    const [lo, hi] = d.bounds;
    const rows = [
      ['atoms', d.n_total.toLocaleString()],
      ['shown', (d.ids.length).toLocaleString() + (d.truncated ? '  LOD' : ''),
        d.truncated ? 'alert' : ''],
      ['bonds', (d.bonds.length / 2).toLocaleString()],
      ['box x', `${num(hi[0] - lo[0], 5)} A`],
      ['box y', `${num(hi[1] - lo[1], 5)} A`],
      ['box z', `${num(hi[2] - lo[2], 5)} A`],
      ['pbc', d.pbc.map((v) => (v ? 'T' : 'F')).join(' ')],
      'rule',
      ['camera', `${num(this.distance, 4)} A`],
      ['elev', `${num(this.rotX * 57.2958, 4)} deg`],
      ['azim', `${num(this.rotY * 57.2958, 4)} deg`],
    ];
    setHud(this.overlay, 'atomic model', rows, {
      id: Object.keys(d.elements).sort().join(' '),
      foot: d.truncated ? d.note : 'Bond lines are a covalent-radius criterion.',
      footAlert: !!d.truncated,
    });
  }

  _renderLegend() {
    if (!this.data) { this.legend.innerHTML = ''; return; }
    const rows = Object.keys(this.data.elements).sort().map((sym) => {
      const c = ELEMENT_COLOURS[sym] || DEFAULT_COLOUR;
      const hex = '#' + c.map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
      return [`\u25a0 ${sym}`, `Z ${this.data.elements[sym]}`, '', hex];
    });
    setHud(this.legend, 'species',
      rows.map(([k, v]) => [k, v]),
      { foot: 'Display convention. Probe data stays greyscale.' });
    const cells = this.legend.querySelectorAll('table.hud td.k');
    rows.forEach((r, i) => { if (cells[i]) cells[i].style.color = r[3]; });
  }

  _drawScaleBar() {
    const rect = this.canvas.getBoundingClientRect();
    const proj = this._projMatrix(this.canvas.width, this.canvas.height);
    const pxPerAngstrom = (proj[5] * 0.5 * rect.height) / Math.max(this.distance, 1e-6);
    const target = 110;
    const a = niceNumber(target / Math.max(pxPerAngstrom, 1e-9));
    this.scaleEl.innerHTML =
      `<span class="bar" style="width:${(a * pxPerAngstrom).toFixed(0)}px"></span>` +
      `<span>${a >= 10 ? `${num(a / 10, 3)} nm` : `${num(a, 3)} Å`}</span>`;
  }

  _renderEmptyGraticule() {
    let layer = this.canvas.parentElement.querySelector('canvas.empty-layer');
    if (!layer) {
      layer = document.createElement('canvas');
      layer.className = 'empty-layer';
      layer.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;' +
        'pointer-events:none';
      this.canvas.parentElement.append(layer);
    }
    layer.hidden = false;
    const [w, h] = fitCanvas(layer);
    const ctx = layer.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    graticule(ctx, w, h, { step: 54, label: 'NO REGION' });
  }

  _hideEmptyGraticule() {
    const layer = this.canvas.parentElement.querySelector('canvas.empty-layer');
    if (layer) layer.hidden = true;
  }

  _drawLocator() {
    const canvas = document.getElementById('canvas-atoms-callout');
    if (!canvas) return;
    const wafer = state.server && state.server.wafer;
    if (!wafer || !state.roi || !this.data) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const [lo, hi] = this.data.bounds;
    const spanA = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
    drawCallout(canvas, wafer, state.roi, {
      spanMm: spanA * 1e-7,
      label: `region ${(spanA / 10).toFixed(2)} nm across`,
    });
  }

  _drawFallback(w, h) {
    const ctx = this.fallback;
    ctx.fillStyle = cssVar('--viewport');
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#9aa2a8';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('WebGL2 is not available in this browser.', w / 2, h / 2 - 10);
    ctx.fillText('The atomic model viewport needs WebGL2; everything else still works.',
      w / 2, h / 2 + 12);
  }

  resize() { this.draw(); }
}
