/**
 * Modal dialogs. Kept deliberately plain: a title bar, a form, OK/Cancel.
 * @module dialogs
 */

import { $, clear, el } from './util.js';

let resolver = null;

function close(value) {
  $('#modal-backdrop').hidden = true;
  const r = resolver; resolver = null;
  if (r) r(value);
}

/**
 * Show a modal form.
 * @param {string} title
 * @param {{name:string,label:string,type?:string,value?:any,options?:string[],
 *          step?:number,min?:number,max?:number,hint?:string,wide?:boolean}[]} fields
 * @param {{okLabel?:string, note?:string, danger?:boolean}} [opts]
 * @returns {Promise<Object|null>}
 */
export function form(title, fields, opts = {}) {
  const body = $('#modal-body');
  const foot = $('#modal-foot');
  $('#modal-title').textContent = title;
  clear(body); clear(foot);

  if (opts.note) body.append(el('div.note' + (opts.danger ? '.warn' : '.info'),
    { html: opts.note }));

  const grid = el('div.field-grid');
  const inputs = {};
  for (const f of fields) {
    if (f.type === 'section') {
      grid.append(el('div.span2.section-title', {}, el('span', { text: f.label })));
      continue;
    }
    const id = `dlg-${f.name}`;
    let input;
    if (f.type === 'select') {
      input = el('select', { id }, (f.options || []).map((o) =>
        el('option', { value: typeof o === 'string' ? o : o.value,
          selected: (typeof o === 'string' ? o : o.value) === f.value },
        typeof o === 'string' ? o : o.label)));
    } else if (f.type === 'checkbox') {
      input = el('input', { type: 'checkbox', id, checked: !!f.value });
    } else if (f.type === 'textarea') {
      input = el('textarea', { id, rows: f.rows || 4,
        style: { width: '100%', height: 'auto', fontFamily: 'var(--code)' } });
      input.value = f.value ?? '';
    } else {
      input = el('input', { type: f.type || 'text', id, value: f.value ?? '',
        step: f.step, min: f.min, max: f.max,
        class: f.type === 'number' ? 'num mono' + (f.wide ? ' wide' : '') : '' });
      if (f.type !== 'number') input.style.width = '100%';
    }
    inputs[f.name] = input;
    grid.append(el('label', { for: id, text: f.label }), input);
    if (f.hint) grid.append(el('div.span2.hint', { text: f.hint }));
  }
  body.append(grid);

  const ok = el('button.tool' + (opts.danger ? '.danger' : '.primary'),
    { text: opts.okLabel || 'OK' });
  const cancel = el('button.tool', { text: 'Cancel' });
  ok.addEventListener('click', () => {
    const out = {};
    for (const f of fields) {
      if (f.type === 'section') continue;
      const input = inputs[f.name];
      out[f.name] = f.type === 'checkbox' ? input.checked
        : f.type === 'number' ? parseFloat(input.value) : input.value;
    }
    close(out);
  });
  cancel.addEventListener('click', () => close(null));
  foot.append(cancel, ok);

  $('#modal-backdrop').hidden = false;
  const first = Object.values(inputs)[0];
  if (first) setTimeout(() => first.focus(), 0);
  return new Promise((res) => { resolver = res; });
}

/** Show read-only content with a single Close button. */
export function info(title, node, { wide = false } = {}) {
  const body = $('#modal-body');
  const foot = $('#modal-foot');
  $('#modal-title').textContent = title;
  clear(body); clear(foot);
  body.append(node instanceof Node ? node : el('div', { html: String(node) }));
  if (wide) $('#modal-backdrop').querySelector('.modal').style.minWidth = '620px';
  const ok = el('button.tool.primary', { text: 'Close', onclick: () => close(null) });
  foot.append(ok);
  $('#modal-backdrop').hidden = false;
  setTimeout(() => ok.focus(), 0);
  return new Promise((res) => { resolver = res; });
}

/** Confirm with an explicit consequence statement. */
export async function confirm(title, message, { okLabel = 'Continue', danger = false } = {}) {
  const body = $('#modal-body');
  const foot = $('#modal-foot');
  $('#modal-title').textContent = title;
  clear(body); clear(foot);
  body.append(el('div', { html: message, style: { lineHeight: '1.6' } }));
  const ok = el('button.tool' + (danger ? '.danger' : '.primary'),
    { text: okLabel, onclick: () => close(true) });
  const cancel = el('button.tool', { text: 'Cancel', onclick: () => close(false) });
  foot.append(cancel, ok);
  $('#modal-backdrop').hidden = false;
  setTimeout(() => ok.focus(), 0);
  return new Promise((res) => { resolver = res; });
}

export function choice(title, message, actions, cancelLabel = 'Not now') {
  const body = $('#modal-body');
  const foot = $('#modal-foot');
  $('#modal-title').textContent = title;
  clear(body); clear(foot);
  body.append(message instanceof Node ? message : el('div', {
    text: String(message), style: { lineHeight: '1.6' },
  }));
  const cancel = el('button.tool', { text: cancelLabel, onclick: () => close(null) });
  foot.append(cancel);
  for (const action of actions) {
    const button = el('button.tool' + (action.danger ? '.danger' : '.primary'), {
      text: action.label,
      onclick: () => close(action.value),
    });
    foot.append(button);
  }
  $('#modal-backdrop').hidden = false;
  const preferred = foot.querySelector('.primary') || cancel;
  setTimeout(() => preferred.focus(), 0);
  return new Promise((res) => { resolver = res; });
}

document.addEventListener('keydown', (e) => {
  if (!$('#modal-backdrop').hidden && e.key === 'Escape') close(null);
});


/** True when running inside the native application window. */
export function isNative() {
  return typeof window !== 'undefined' && !!window.pywebview && !!window.pywebview.api;
}

/**
 * Ask the operating system for a path to open, falling back to an in-window
 * field when the native bridge is unavailable.
 * @param {{title?:string, filters?:string[], placeholder?:string}} opts
 * @returns {Promise<string|null>}
 */
export async function choosePathToOpen(opts = {}) {
  if (isNative()) {
    const path = await window.pywebview.api.open_file(
      opts.title || 'Open', opts.filters || ['All files (*.*)']);
    return path || null;
  }
  const v = await form(opts.title || 'Open', [
    { name: 'path', label: 'path', value: opts.placeholder || '' }],
    { note: opts.note });
  return v ? v.path : null;
}

/**
 * Ask the operating system for a path to write to.
 * @param {{title?:string, filename?:string, note?:string}} opts
 * @returns {Promise<string|null>}
 */
export async function choosePathToSave(opts = {}) {
  if (isNative()) {
    const path = await window.pywebview.api.save_file(
      opts.title || 'Save', opts.filename || '');
    return path || null;
  }
  const v = await form(opts.title || 'Save', [
    { name: 'path', label: 'path', value: opts.filename || '' }],
    { note: opts.note });
  return v ? v.path : null;
}
