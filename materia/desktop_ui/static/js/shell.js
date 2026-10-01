/**
 * Electron-shell and orbital drawings for the atom inspector.
 *
 * Scientific labelling matters here and is enforced in the markup:
 *
 *  - The shell diagram is an **educational electron-shell diagram**. It is
 *    explicitly not a picture of electrons orbiting a nucleus, and the panel
 *    says so. Its only physical content is the shell occupancy.
 *  - The orbital plots are hydrogen-like solutions evaluated with a Slater
 *    effective nuclear charge. They are the correct shapes and the correct
 *    radial nodal structure for a one-electron ion, and an approximation for
 *    a many-electron atom. They are tagged "estimated".
 *  - The Pauli/Hund grid shows an occupancy convention, not a measured
 *    assignment of electrons to particular m_l values.
 *
 * @module shell
 */

import { el, num } from './util.js';

const ORBITAL_LETTERS = 'spdfghi';

/**
 * Slater's rules for the effective nuclear charge seen by an electron in
 * shell n with orbital letter (s/p share a group; d and f are separate).
 * J. C. Slater, Phys. Rev. 36 (1930) 57.
 */
export function slaterZeff(Z, subshells, n, l) {
  let shielding = 0;
  const isSP = l <= 1;
  for (const sh of subshells) {
    const same = sh.n === n && ((sh.l <= 1) === isSP);
    let count = sh.electrons;
    if (same) count -= 1;
    if (count <= 0 && same) continue;
    if (isSP) {
      if (sh.n === n && (sh.l <= 1)) shielding += 0.35 * Math.max(count, 0);
      else if (sh.n === n - 1) shielding += 0.85 * sh.electrons;
      else if (sh.n < n - 1) shielding += 1.0 * sh.electrons;
      else if (sh.n === n && sh.l > 1) shielding += 0;
    } else {
      if (sh.n === n && sh.l === l) shielding += 0.35 * Math.max(count, 0);
      else if (sh.n < n || (sh.n === n && sh.l < l)) shielding += 1.0 * sh.electrons;
    }
  }
  if (n === 1 && isSP) {
    shielding = 0.30 * Math.max(subshells.find((s) => s.n === 1)?.electrons - 1 || 0, 0);
  }
  return Math.max(Z - shielding, 1.0);
}

function laguerre(nMinusLMinus1, twoLPlus1, x) {
  const k = nMinusLMinus1, alpha = twoLPlus1;
  let lPrev = 1.0;
  if (k === 0) return lPrev;
  let lCurr = 1.0 + alpha - x;
  for (let i = 1; i < k; i++) {
    const next = ((2 * i + 1 + alpha - x) * lCurr - (i + alpha) * lPrev) / (i + 1);
    lPrev = lCurr; lCurr = next;
  }
  return lCurr;
}

function factorial(n) { let r = 1; for (let i = 2; i <= n; i++) r *= i; return r; }

/**
 * Hydrogen-like radial function R_{nl}(r) in units where a0 = 1.
 * @param {number} n @param {number} l @param {number} Zeff @param {number} r
 */
export function radialWavefunction(n, l, Zeff, r) {
  const rho = (2 * Zeff * r) / n;
  const norm = Math.sqrt(
    Math.pow(2 * Zeff / n, 3) * factorial(n - l - 1) / (2 * n * factorial(n + l)));
  return norm * Math.exp(-rho / 2) * Math.pow(rho, l) * laguerre(n - l - 1, 2 * l + 1, rho);
}

/** Radial probability density 4 pi r^2 |R|^2 (per bohr). */
export function radialProbability(n, l, Zeff, r) {
  const R = radialWavefunction(n, l, Zeff, r);
  return R * R * r * r;
}

/**
 * Educational electron-shell diagram.
 * @param {{symbol:string, protons:number, neutrons:number|null,
 *          occupancy:number[], charge:number, valence:number}} spec
 */
export function shellDiagram(spec) {
  const NS = 'http://www.w3.org/2000/svg';
  const size = 268, cx = size / 2, cy = size / 2;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'shell-diagram');
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', size);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label',
    `Educational electron-shell diagram of ${spec.symbol}: ` +
    spec.occupancy.map((n, i) => `shell ${i + 1} holds ${n} electrons`).join(', ') + '.');

  const mk = (tag, attrs) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    return n;
  };

  const shells = spec.occupancy.length || 1;
  const rMax = size / 2 - 16;
  const rMin = 34;

  svg.append(mk('circle', { cx, cy, r: 17, fill: '#3a3f45', stroke: '#8d949b', 'stroke-width': 1 }));
  const nucLabel = mk('text', { x: cx, y: cy + 1, 'text-anchor': 'middle',
    'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif', 'font-size': 11, fill: '#e7ebee' });
  nucLabel.textContent = spec.symbol;
  svg.append(nucLabel);
  const nucSub = mk('text', { x: cx, y: cy + 12, 'text-anchor': 'middle',
    'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif', 'font-size': 8, fill: '#9aa2a8' });
  nucSub.textContent = spec.neutrons === null || spec.neutrons === undefined
    ? `${spec.protons}p` : `${spec.protons}p ${spec.neutrons}n`;
  svg.append(nucSub);

  for (let i = 0; i < shells; i++) {
    const r = shells === 1 ? rMin : rMin + (rMax - rMin) * (i / (shells - 1));
    svg.append(mk('circle', { cx, cy, r, fill: 'none', stroke: '#394046',
      'stroke-width': 1, 'stroke-dasharray': '2 4' }));
    const count = spec.occupancy[i];
    const isValence = i === shells - 1;
    for (let k = 0; k < count; k++) {
      const a = (-Math.PI / 2) + (2 * Math.PI * k) / count + (i * 0.22);
      const x = cx + r * Math.cos(a);
      const y = cy + r * Math.sin(a);
      svg.append(mk('circle', { cx: x, cy: y, r: 3.1,
        fill: isValence ? '#ffb020' : '#7dd3fc',
        stroke: '#11161a', 'stroke-width': 0.8 }));
    }
    const lbl = mk('text', { x: cx + r + 3, y: cy - 3, 'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif',
      'font-size': 8.5, fill: '#77808a' });
    lbl.textContent = `${'KLMNOPQ'[i] || i + 1}:${count}`;
    svg.append(lbl);
  }

  if (spec.charge) {
    const t = mk('text', { x: size - 8, y: 14, 'text-anchor': 'end',
      'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif', 'font-size': 11, fill: '#ffb020' });
    t.textContent = `charge ${spec.charge > 0 ? '+' : ''}${spec.charge} e`;
    svg.append(t);
  }
  const legend = mk('text', { x: 8, y: size - 8, 'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif',
    'font-size': 8.5, fill: '#77808a' });
  legend.textContent = `valence electrons: ${spec.valence} (amber)`;
  svg.append(legend);
  return svg;
}

/**
 * Radial probability distribution for every occupied subshell.
 * @param {{n:number,l:number,label:string,electrons:number}[]} subshells
 * @param {number} Z
 */
export function radialPlot(subshells, Z, width = 300, height = 170) {
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'orbital-plot');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', height);
  const mk = (tag, attrs, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const pad = { l: 34, r: 8, t: 10, b: 22 };
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b;

  const nMax = Math.max(...subshells.map((s) => s.n));
  const rMax = Math.max(4, (nMax * nMax * 3.2) / Math.max(1, slaterZeff(Z, subshells, nMax, 0)) * 1.6);
  const samples = 220;
  const colours = ['#6aa9dd', '#ffb020', '#8fc3eb', '#ff6b58', '#c084fc', '#7dd3fc', '#facc15'];

  let peak = 0;
  const curves = subshells.map((sh) => {
    const Zeff = slaterZeff(Z, subshells, sh.n, sh.l);
    const pts = [];
    for (let i = 0; i <= samples; i++) {
      const r = (i / samples) * rMax;
      const p = radialProbability(sh.n, sh.l, Zeff, r) * sh.electrons;
      pts.push([r, p]);
      if (p > peak) peak = p;
    }
    return { sh, Zeff, pts };
  });
  if (peak <= 0) peak = 1;

  svg.append(mk('rect', { x: pad.l, y: pad.t, width: w, height: h, fill: 'none',
    stroke: '#3a4046', 'stroke-width': 1 }));
  for (let i = 1; i < 5; i++) {
    const x = pad.l + (w * i) / 5;
    svg.append(mk('line', { x1: x, y1: pad.t, x2: x, y2: pad.t + h, stroke: '#242a2f' }));
  }
  curves.forEach((c, idx) => {
    const d = c.pts.map(([r, p], i) => {
      const x = pad.l + (r / rMax) * w;
      const y = pad.t + h - (p / peak) * h;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(' ');
    svg.append(mk('path', { d, fill: 'none', stroke: colours[idx % colours.length],
      'stroke-width': 1.4 }));
    svg.append(mk('text', { x: pad.l + 6 + idx * 34, y: pad.t + 11,
      'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif', 'font-size': 9,
      fill: colours[idx % colours.length] }, `${c.sh.label} Z*=${c.Zeff.toFixed(2)}`));
  });
  svg.append(mk('text', { x: pad.l + w / 2, y: height - 6, 'text-anchor': 'middle',
    'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif', 'font-size': 9, fill: '#77808a' },
    `r / a₀   (0 … ${rMax.toFixed(1)})`));
  const yl = mk('text', { x: 10, y: pad.t + h / 2, 'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif',
    'font-size': 9, fill: '#77808a', transform: `rotate(-90 10 ${pad.t + h / 2})`,
    'text-anchor': 'middle' }, '4πr²|R|²');
  svg.append(yl);
  return svg;
}

/**
 * Angular probability density |Y_lm|^2 cross-section in the xz plane.
 * @param {number} l @param {number} m
 */
export function angularPlot(l, m, size = 150) {
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'orbital-plot');
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.setAttribute('width', size);
  svg.setAttribute('height', size);
  const mk = (tag, attrs, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const cx = size / 2, cy = size / 2, R = size / 2 - 12;
  const amp = (theta) => {
    const c = Math.cos(theta), s = Math.sin(theta);
    if (l === 0) return 1;
    if (l === 1) return Math.abs(m) === 0 ? c : s;
    if (l === 2) {
      if (m === 0) return 0.5 * (3 * c * c - 1);
      if (Math.abs(m) === 1) return s * c;
      return s * s;
    }
    if (l === 3) {
      if (m === 0) return 0.5 * c * (5 * c * c - 3);
      if (Math.abs(m) === 1) return s * (5 * c * c - 1);
      if (Math.abs(m) === 2) return s * s * c;
      return s * s * s;
    }
    return 1;
  };
  const pts = [];
  for (let i = 0; i <= 360; i++) {
    const th = (i / 360) * 2 * Math.PI;
    const r = Math.abs(amp(th));
    pts.push([cx + R * r * Math.sin(th), cy - R * r * Math.cos(th), amp(th) >= 0]);
  }
  svg.append(mk('circle', { cx, cy, r: R, fill: 'none', stroke: '#242a2f', 'stroke-dasharray': '2 3' }));
  let d = '';
  pts.forEach(([x, y], i) => { d += `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`; });
  svg.append(mk('path', { d: d + 'Z', fill: 'rgba(47,182,196,.30)', stroke: '#2fb6c4',
    'stroke-width': 1.3 }));
  svg.append(mk('line', { x1: cx, y1: 6, x2: cx, y2: size - 6, stroke: '#333a40' }));
  svg.append(mk('line', { x1: 6, y1: cy, x2: size - 6, y2: cy, stroke: '#333a40' }));
  svg.append(mk('text', { x: cx + 4, y: 12, 'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif', 'font-size': 8.5,
    fill: '#77808a' }, 'z'));
  svg.append(mk('text', { x: size - 12, y: cy - 4, 'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif',
    'font-size': 8.5, fill: '#77808a' }, 'x'));
  svg.append(mk('text', { x: 6, y: size - 5, 'font-family': '"Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif', 'font-size': 8.5,
    fill: '#9aa2a8' }, `l=${l} |m|=${Math.abs(m)}`));
  return svg;
}

/** Pauli / Hund occupancy boxes for each subshell. */
export function pauliGrid(subshells, spinOrbitals) {
  const wrap = el('div.pauli-grid');
  for (const sh of subshells) {
    const box = el('div.pauli-sub');
    box.append(el('div.lbl', { text: `${sh.label} (${sh.electrons}/${sh.capacity})` }));
    const row = el('div.pauli-boxes');
    for (const ml of sh.m_l) {
      const cell = el('div.pauli-box', { title: `n=${sh.n}, l=${sh.l}, m_l=${ml}` });
      const up = spinOrbitals.some((o) => o.n === sh.n && o.l === sh.l && o.m_l === ml
        && o.m_s > 0 && o.occupied);
      const dn = spinOrbitals.some((o) => o.n === sh.n && o.l === sh.l && o.m_l === ml
        && o.m_s < 0 && o.occupied);
      if (up) cell.append(el('span.up', { text: '↑' }));
      if (dn) cell.append(el('span.dn', { text: '↓' }));
      row.append(cell);
    }
    box.append(row);
    box.append(el('div.pauli-ml', { text: sh.m_l.join('  ') }));
    wrap.append(box);
  }
  return wrap;
}
