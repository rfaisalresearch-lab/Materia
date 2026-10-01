/**
 * Menu bar definition and dispatch.
 * @module menus
 */

import { $, $$, el } from './util.js';

/** @typedef {{label:string, key?:string, action?:string, arg?:any, sep?:boolean,
 *             title?:string, enabled?:(s:any)=>boolean}} MenuEntry */

let cachedSpec = null;

/**
 * Load the shared menu specification. The same file drives the native
 * platform menu bar, so the two can never drift apart.
 * @returns {Promise<Object<string, MenuEntry[]>>}
 */
export async function loadMenuSpec() {
  if (cachedSpec) return cachedSpec;
  const response = await fetch('menus.json');
  cachedSpec = await response.json();
  return cachedSpec;
}

export function menuSpec() { return cachedSpec || {}; }

export function installMenus(dispatch, spec) {
  const popup = $('#menu-popup');
  let openName = null;

  function close() {
    popup.hidden = true;
    openName = null;
    $$('.menu-root').forEach((b) => b.setAttribute('aria-expanded', 'false'));
  }

  function open(name, anchor) {
    const entries = spec[name];
    if (!entries) return;
    popup.innerHTML = '';
    for (const entry of entries) {
      if (entry.sep) { popup.append(el('div.menu-sep')); continue; }
      const item = el('div.menu-item', { role: 'menuitem', tabindex: '0',
        title: entry.title || '' }, [
        el('span', { text: entry.label }),
        el('span.key', { text: entry.key || '' })]);
      item.addEventListener('click', () => { close(); dispatch(entry.action, entry.arg); });
      item.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { close(); dispatch(entry.action, entry.arg); }
      });
      popup.append(item);
    }
    const r = anchor.getBoundingClientRect();
    popup.style.left = `${r.left}px`;
    popup.style.top = `${r.bottom}px`;
    popup.hidden = false;
    openName = name;
    anchor.setAttribute('aria-expanded', 'true');
  }

  const roots = $$('.menu-root');
  const names = Object.keys(spec);
  roots.forEach((btn, i) => {
    if (names[i]) { btn.dataset.menu = names[i]; btn.textContent = names[i]; }
  });
  roots.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const name = btn.dataset.menu;
      if (openName === name) close(); else { close(); open(name, btn); }
    });
    btn.addEventListener('mouseenter', () => {
      if (openName && openName !== btn.dataset.menu) { close(); open(btn.dataset.menu, btn); }
    });
  });
  document.addEventListener('click', close);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
}
