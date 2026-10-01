/**
 * Small DOM and formatting helpers.
 * @module util
 */

/** @param {string} sel @param {ParentNode} [root] @returns {HTMLElement} */
export const $ = (sel, root = document) => /** @type {HTMLElement} */ (root.querySelector(sel));
/** @param {string} sel @param {ParentNode} [root] @returns {HTMLElement[]} */
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/**
 * Create an element.
 * @param {string} tag  e.g. "div.row.active" or "button#go"
 * @param {Object<string,any>} [attrs]
 * @param {(Node|string)[]|Node|string} [children]
 */
export function el(tag, attrs = {}, children = []) {
  const m = tag.match(/^([a-zA-Z0-9-]+)((?:[.#][\w-]+)*)$/);
  const node = document.createElement(m ? m[1] : tag);
  if (m && m[2]) {
    for (const token of m[2].match(/[.#][\w-]+/g) || []) {
      if (token[0] === '.') node.classList.add(token.slice(1));
      else node.id = token.slice(1);
    }
  }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'text') node.textContent = String(v);
    else if (k === 'html') node.innerHTML = String(v);
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else node.setAttribute(k, v === true ? '' : String(v));
  }
  const list = Array.isArray(children) ? children : [children];
  for (const c of list) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }

/** Number formatting that never lies about precision. */
export function num(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  if (typeof value !== 'number') return String(value);
  if (value === 0) return '0';
  const a = Math.abs(value);
  if (a >= 1e5 || a < 1e-3) return value.toExponential(Math.max(digits - 1, 1));
  return value.toFixed(Math.max(0, digits - Math.max(0, Math.floor(Math.log10(a)) + 1)));
}

export function fmtLength(angstrom) {
  const a = Math.abs(angstrom);
  if (a >= 1e7) return `${num(angstrom / 1e7, 4)} mm`;
  if (a >= 1e4) return `${num(angstrom / 1e4, 4)} µm`;
  if (a >= 10) return `${num(angstrom / 10, 4)} nm`;
  return `${num(angstrom, 4)} Å`;
}

export function fmtTime(seconds) {
  if (seconds === null || seconds === undefined) return '-';
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 60) return `${seconds.toFixed(2)} s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
}

/** Decode a base64 float32 payload from the service. */
export function decodeFloat32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** Size a canvas to its CSS box at device pixel ratio. Returns [w, h, dpr]. */
export function fitCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return [w, h, dpr];
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** A "nice" round number at or below `target`, for scale bars and axes. */
export function niceNumber(target) {
  if (target <= 0) return 1;
  const exp = Math.floor(Math.log10(target));
  const base = Math.pow(10, exp);
  const frac = target / base;
  const nice = frac >= 5 ? 5 : frac >= 2 ? 2 : 1;
  return nice * base;
}

export function debounce(fn, ms = 120) {
  let t = 0;
  return (...args) => { clearTimeout(t); t = window.setTimeout(() => fn(...args), ms); };
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** Provenance origin -> CSS tag class. */
export function originTag(origin) {
  return el('span.tag.' + (origin || 'reference'), { text: origin || 'reference',
    title: originHelp(origin) });
}

export function originHelp(origin) {
  return ({
    calculated: 'Produced by a solver in this session from the current state.',
    interpolated: 'Interpolated or extrapolated from calculated samples.',
    estimated: 'Produced by a heuristic or empirical rule, not a variational solution.',
    illustrative: 'Teaching visualisation only. Not a physical prediction.',
    reference: 'Literature value shipped with the program.',
    imported: 'Read from a file or an external solver.',
    measured: 'Experimental data supplied by the user.',
    unsupported: 'The active model cannot produce this quantity.',
  })[origin] || '';
}

export function tierTag(fidelity) {
  const short = { 'tier0-structural': 'Tier 0', 'tier1-classical': 'Tier 1',
    'tier2-semi-empirical': 'Tier 2', 'tier3-external-first-principles': 'Tier 3',
    'non-physical': 'non-physical' }[fidelity] || fidelity;
  const cls = (fidelity || '').startsWith('tier') ? fidelity.slice(0, 5) : '';
  return el('span.tag' + (cls ? '.' + cls : ''), { text: short, title: fidelity });
}
