/**
 * Instrument head-up display primitives.
 *
 * Viewport overlays follow the conventions of laboratory instrument screens:
 * a titled block flush to the corner, hairline rules, uppercase field labels
 * and right-hand values in a fixed-width face. Empty viewports show a
 * graticule and a terse status legend rather than prose.
 */

import { el } from './util.js';

/**
 * Build a HUD block.
 * @param {string} title
 * @param {Array<[string, string|null]|'rule'>} rows
 * @param {{id?:string, foot?:string, footAlert?:boolean}} [opts]
 */
export function hud(title, rows, opts = {}) {
  const frag = document.createDocumentFragment();
  const head = el('div.hud-title', {}, [el('span', { text: title })]);
  if (opts.id) head.append(el('span.id', { text: opts.id }));
  frag.append(head);
  const table = el('table.hud');
  for (const row of rows) {
    if (row === 'rule') {
      table.append(el('tr.rule', {}, [el('td', { colspan: '2' })]));
      continue;
    }
    const [k, v, cls] = row;
    if (v === null || v === undefined) continue;
    table.append(el('tr', {}, [
      el('td.k', { text: k }),
      el('td', { class: 'v' + (cls ? ' ' + cls : ''), text: String(v) })]));
  }
  frag.append(table);
  if (opts.foot) {
    frag.append(el('div', { class: 'hud-foot' + (opts.footAlert ? ' alert' : ''),
      text: opts.foot }));
  }
  return frag;
}

export function setHud(node, title, rows, opts) {
  while (node.firstChild) node.removeChild(node.firstChild);
  node.append(hud(title, rows, opts));
}

/**
 * Draw an instrument graticule: a fine grid, centre crosshair and edge ticks.
 * @param {CanvasRenderingContext2D} ctx
 */
export function graticule(ctx, w, h, { label = '' } = {}) {
  if (!label) return;
  ctx.save();
  ctx.font = 'bold 11px "Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const metrics = ctx.measureText(label);
  const pad = 12;
  const bw = metrics.width + pad * 2;
  ctx.fillStyle = '#f4f3ef';
  ctx.fillRect(w / 2 - bw / 2, h / 2 - 11, bw, 22);
  ctx.strokeStyle = '#6d6b64';
  ctx.lineWidth = 1;
  ctx.strokeRect(w / 2 - bw / 2 + 0.5, h / 2 - 10.5, bw - 1, 21);
  ctx.fillStyle = '#14130f';
  ctx.fillText(label, w / 2, h / 2 + 1);
  ctx.restore();
}
