/**
 * Secondary plot pane: line profiles, DOS, LDOS, band structure, dI/dV,
 * force curves and relaxation history.
 *
 * Every plot labels its axes with units and states which model produced the
 * data, so a curve can never be read without knowing where it came from.
 * @module plots
 */

import { state } from './state.js';
import { $, cssVar, fitCanvas, niceNumber, num } from './util.js';

const COLOURS = ['#6aa9dd', '#ffb020', '#8fc3eb', '#ff6b58', '#c084fc', '#7dd3fc'];

export class PlotPane {
  constructor(canvasId, noteId) {
    this.canvas = $(canvasId);
    this.note = $(noteId);
    this.ctx = this.canvas.getContext('2d');
  }

  draw() {
    const [w, h] = fitCanvas(this.canvas);
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    const kind = state.plot;
    const spec = this[`_${kind}`] ? this[`_${kind}`]() : null;
    if (!spec) {
      ctx.fillStyle = '#5a6168';
      ctx.font = `${12 * (w / this.canvas.getBoundingClientRect().width)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(this._emptyMessage(kind), w / 2, h / 2);
      this.note.textContent = '';
      ctx.restore();
      return;
    }
    this._render(ctx, w, h, spec);
    this.note.textContent = spec.note || '';
    ctx.restore();
  }

  _emptyMessage(kind) {
    return ({
      profile: 'Choose the Measure tool and click two points on the probe image.',
      dos: 'Run Solve › Electronic structure to compute the density of states.',
      ldos: 'Select an atom after an electronic-structure run to see its local DOS.',
      bands: 'Run Solve › Band structure on a periodic cell.',
      sts: 'Use Instrument › Point spectroscopy to simulate a dI/dV curve.',
      force: 'Use Instrument › Force curve to simulate an approach curve.',
      relax: 'Run Solve › Relax to record a minimisation history.',
    })[kind] || 'No data.';
  }

  _profile() {
    const p = state.profile;
    if (!p) return null;
    return {
      series: [{ x: p.distance_A, y: p.values, label: p.channel }],
      xlabel: 'distance / Å', ylabel: `${p.channel} / ${p.unit}`,
      note: 'Line profile sampled from the displayed channel (nearest pixel).',
    };
  }

  _dos() {
    const e = state.electronic;
    if (!e || !e.dos) return null;
    return {
      series: [{ x: e.dos.energy_eV, y: e.dos.dos, label: 'total DOS' }],
      xlabel: 'energy / eV', ylabel: 'DOS / states eV⁻¹',
      vline: e.fermi_level_eV, vlabel: 'E_F',
      note: `${e.solver} · Gaussian broadening · supercell Γ point only`,
    };
  }

  _ldos() {
    const l = state.atomLdos;
    if (!l || !l.supported) return null;
    return {
      series: [{ x: l.energy_eV, y: l.ldos, label: `atom #${l.atom_id}` }],
      xlabel: 'energy / eV', ylabel: 'LDOS / states eV⁻¹',
      vline: state.electronic ? state.electronic.fermi_level_eV : null, vlabel: 'E_F',
      note: 'Site-projected density of states from the last electronic run.',
    };
  }

  _bands() {
    const b = state.bands;
    if (!b || !b.supported) return null;
    const nb = b.bands_eV[0].length;
    const series = [];
    for (let j = 0; j < nb; j++) {
      series.push({ x: b.k_coord, y: b.bands_eV.map((row) => row[j]),
        label: j === 0 ? 'bands' : '', colour: '#2fb6c4', thin: true });
    }
    return {
      series, xlabel: 'k path', ylabel: 'energy / eV',
      hline: b.fermi_level_eV, hlabel: 'E_F',
      ticks: b.ticks, tickLabels: b.labels,
      note: `${b.solver} · gap ${num(b.band_gap_eV, 4)} eV ` +
        `(${b.band_gap_note && b.band_gap_note.direct ? 'direct' : 'indirect'} ` +
        'along the sampled path)',
    };
  }

  _sts() {
    const s = state.sts;
    if (!s || !s.supported) return null;
    return {
      series: [{ x: s.bias_V, y: s.dIdV, label: 'dI/dV' }],
      xlabel: 'sample bias / V', ylabel: 'dI/dV / arb. units',
      vline: 0, vlabel: 'E_F',
      note: 'Tersoff-Hamann: dI/dV ∝ LDOS at the tip position. Scale is arbitrary.',
    };
  }

  _force() {
    const f = state.forceCurve;
    if (!f || !f.supported) return null;
    return {
      series: [{ x: f.height_A, y: f.force_nN, label: 'F_z / nN' },
               { x: f.height_A, y: f.frequency_shift_Hz.map((v) => v / 40),
                 label: 'Δf / 40 Hz' }],
      xlabel: 'tip height above the topmost atom / Å', ylabel: 'force / nN',
      hline: 0,
      note: 'Lennard-Jones + Hamaker; rigid tip and sample; no chemical bonding.',
    };
  }

  _relax() {
    const r = state.relaxHistory;
    if (!r) return null;
    return {
      series: [{ x: r.step, y: r.fmax_eV_A, label: 'max |F| / eV Å⁻¹' }],
      xlabel: 'FIRE step', ylabel: 'max |F| / eV Å⁻¹', log: true,
      note: 'FIRE minimisation; convergence criterion is the maximum force on free atoms.',
    };
  }

  _render(ctx, w, h, spec) {
    const dpr = this.canvas.width / this.canvas.getBoundingClientRect().width;
    const pad = { l: 56 * dpr, r: 14 * dpr, t: 12 * dpr, b: 30 * dpr };
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of spec.series) {
      for (let i = 0; i < s.x.length; i++) {
        const xv = s.x[i], yv = spec.log ? Math.log10(Math.max(s.y[i], 1e-12)) : s.y[i];
        if (xv < xmin) xmin = xv; if (xv > xmax) xmax = xv;
        if (yv < ymin) ymin = yv; if (yv > ymax) ymax = yv;
      }
    }
    if (!Number.isFinite(xmin)) return;
    if (xmax === xmin) xmax = xmin + 1;
    const padY = (ymax - ymin) * 0.08 || 1;
    ymin -= padY; ymax += padY;

    const X = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * pw;
    const Y = (v) => {
      const val = spec.log ? Math.log10(Math.max(v, 1e-12)) : v;
      return pad.t + ph - ((val - ymin) / (ymax - ymin)) * ph;
    };

    ctx.strokeStyle = '#20252a';
    ctx.lineWidth = 1 * dpr;
    ctx.font = `${10 * dpr}px "Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif`;
    ctx.fillStyle = '#7d868e';
    ctx.textAlign = 'right';
    const yStep = niceNumber((ymax - ymin) / 4);
    for (let v = Math.ceil(ymin / yStep) * yStep; v <= ymax; v += yStep) {
      const y = spec.log ? pad.t + ph - ((v - ymin) / (ymax - ymin)) * ph : Y(v);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
      ctx.fillText(spec.log ? `1e${v.toFixed(0)}` : num(v, 4), pad.l - 5 * dpr, y + 3 * dpr);
    }
    ctx.textAlign = 'center';
    if (spec.ticks && spec.tickLabels) {
      spec.ticks.forEach((t, i) => {
        const x = X(t);
        ctx.strokeStyle = '#3c454d';
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
        ctx.fillText(spec.tickLabels[i] || '', x, pad.t + ph + 14 * dpr);
      });
    } else {
      const xStep = niceNumber((xmax - xmin) / 5);
      for (let v = Math.ceil(xmin / xStep) * xStep; v <= xmax; v += xStep) {
        const x = X(v);
        ctx.strokeStyle = '#2a3036';
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
        ctx.fillText(num(v, 4), x, pad.t + ph + 14 * dpr);
      }
    }

    if (spec.vline !== null && spec.vline !== undefined && spec.vline >= xmin && spec.vline <= xmax) {
      ctx.strokeStyle = '#ffb020';
      ctx.setLineDash([4 * dpr, 3 * dpr]);
      ctx.beginPath(); ctx.moveTo(X(spec.vline), pad.t); ctx.lineTo(X(spec.vline), pad.t + ph);
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#ffb020';
      ctx.textAlign = 'left';
      ctx.fillText(spec.vlabel || '', X(spec.vline) + 4 * dpr, pad.t + 11 * dpr);
    }
    if (spec.hline !== null && spec.hline !== undefined) {
      ctx.strokeStyle = '#ffb020';
      ctx.setLineDash([4 * dpr, 3 * dpr]);
      ctx.beginPath(); ctx.moveTo(pad.l, Y(spec.hline)); ctx.lineTo(pad.l + pw, Y(spec.hline));
      ctx.stroke(); ctx.setLineDash([]);
      if (spec.hlabel) {
        ctx.fillStyle = '#ffb020'; ctx.textAlign = 'left';
        ctx.fillText(spec.hlabel, pad.l + 4 * dpr, Y(spec.hline) - 4 * dpr);
      }
    }

    spec.series.forEach((s, idx) => {
      ctx.strokeStyle = s.colour || COLOURS[idx % COLOURS.length];
      ctx.lineWidth = (s.thin ? 0.9 : 1.5) * dpr;
      ctx.beginPath();
      for (let i = 0; i < s.x.length; i++) {
        const x = X(s.x[i]), y = Y(s.y[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    ctx.fillStyle = '#7d868e';
    ctx.textAlign = 'center';
    ctx.fillText(spec.xlabel, pad.l + pw / 2, h - 6 * dpr);
    ctx.save();
    ctx.translate(12 * dpr, pad.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(spec.ylabel, 0, 0);
    ctx.restore();

    let lx = pad.l + 8 * dpr;
    ctx.textAlign = 'left';
    spec.series.forEach((s, idx) => {
      if (!s.label) return;
      ctx.fillStyle = s.colour || COLOURS[idx % COLOURS.length];
      ctx.fillText(s.label, lx, pad.t + 11 * dpr);
      lx += ctx.measureText(s.label).width + 14 * dpr;
    });

    ctx.strokeStyle = '#3c454d';
    ctx.lineWidth = 1 * dpr;
    ctx.strokeRect(pad.l, pad.t, pw, ph);
  }

  resize() { this.draw(); }
}
