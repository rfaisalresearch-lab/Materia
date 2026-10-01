/**
 * Right-dock panels: properties, instrument control, solvers, measurement;
 * plus the left-dock project tree, material browser and selection tools.
 * @module panels
 */

import { api } from './api.js';
import { set, state } from './state.js';
import { $, clear, el, fmtLength, fmtTime, num, originTag, tierTag } from './util.js';
import { renderDft } from './dft-panel.js';

export function renderProperties(body, actions) {
  clear(body);
  const s = state.server;
  if (!s) return;
  const st = s.structure;
  if (!st) {
    body.append(el('p.empty', { text:
      'No structure. Create a wafer and extract a region, or build a surface.' }));
    return;
  }
  const grid = (rows) => {
    const t = el('table.pgrid');
    for (const [k, v, cls] of rows) {
      if (v === undefined) continue;
      t.append(el('tr', {}, [el('th', { text: k }),
        el('td', cls ? { class: cls } : {}, v === null ? '-' : String(v))]));
    }
    return t;
  };
  body.append(el('div.section-title', {}, [el('span', { text: 'Structure' }),
    originTag('calculated')]));
  body.append(grid([
    ['atoms', st.n_atoms.toLocaleString()],
    ['formula', st.formula],
    ['elements', st.elements.join(', ')],
    ['net charge', `${st.total_charge > 0 ? '+' : ''}${num(st.total_charge, 4)} e`],
    ['total electrons (Z − q)', num(st.total_electrons, 8)],
    ['total magnetic moment', `${num(st.total_moment, 4)} µʙ`],
    ['cell a, b, c', st.cell_lengths.map((v) => num(v, 5)).join(', ') + ' Å'],
    ['cell α, β, γ', st.cell_angles.map((v) => num(v, 4)).join(', ') + '°'],
    ['periodicity', st.cell.pbc.map((p) => (p ? 'T' : 'F')).join(' ')],
    ['bounding box', st.bounds[1].map((v, i) => num(v - st.bounds[0][i], 4)).join(' × ') + ' Å'],
  ]));

  if (st.surface && st.surface.miller) {
    body.append(el('div.section-title', {}, [el('span', { text: 'Surface' }),
      originTag('calculated')]));
    const sf = st.surface;
    body.append(grid([
      ['material', sf.material_id],
      ['orientation', `(${sf.miller.join('')})`],
      ['termination', `${sf.termination} (${sf.termination_mode})`],
      ['cut offset', num(sf.shift, 4)],
      ['in-plane cell', `${num(sf.in_plane_a_A, 5)} × ${num(sf.in_plane_b_A, 5)} Å, ` +
        `${num(sf.in_plane_angle_deg, 3)}°`],
      ['interplanar spacing', `${num(sf.interplanar_spacing_A, 5)} Å`],
      ['conventional d(hkl)', `${num(sf.conventional_d_hkl_A, 5)} Å`],
      ['slab thickness', `${num(sf.slab_thickness_A, 4)} Å`],
      ['atomic planes', sf.n_atomic_planes],
      ['vacuum', `${num(sf.vacuum_A, 3)} Å`],
      ['supercell matrix', JSON.stringify(sf.supercell_matrix)],
    ]));
    if (sf.termination_candidates && sf.termination_candidates.length) {
      body.append(el('div.note', { text:
        'Termination chosen automatically as the cut that breaks the fewest bonds: ' +
        sf.termination_candidates.map((c) =>
          `${num(c.shift, 2)}→${c.broken_bonds}`).join('  ') }));
    }
  }

  if (st.reconstruction) {
    const rc = st.reconstruction;
    const small = (v, digits = 4) => {
      if (v === undefined || v === null) return '-';
      const floor = 1e-6;
      return Math.abs(v) < floor ? `< ${floor} Å` : `${num(v, digits)} Å`;
    };
    const m = rc.measured || {};
    const pc = rc.periodicity_check || {};
    const rx = rc.relaxation || {};
    const geometryStatus = rc.geometry_status
      || (!rx.performed ? 'unrelaxed' : rx.converged ? 'converged' : 'not-converged');
    body.append(el('div.section-title', {}, [el('span', { text: 'Reconstruction' }),
      originTag(rc.provenance ? rc.provenance.origin : 'estimated')]));
    body.append(grid([
      ['reconstruction', rc.id],
      ['periodicity', `${rc.periodicity.join(' × ')} of the 1×1 cell`],
      ['dimers', rc.n_dimers],
      ['rows', rc.n_rows],
      ['dimer axis', `[${rc.dimer_axis.map((v) => num(v, 3)).join(' ')}]`],
      ['1×1 spacing', `${num(rc.unreconstructed_spacing_A, 4)} Å`],
      ['geometry', {
        converged: 'converged energy minimum',
        'not-converged': 'NOT CONVERGED - not an energy minimum',
        unrelaxed: 'construction guess, not relaxed',
      }[geometryStatus] || '-'],
      ['relaxed with', rx.performed ? rx.model : 'not relaxed'],
      ['relaxation converged', rx.performed ? (rx.converged ? `yes, ${rx.steps} steps`
        : `no, stopped at ${rx.steps} steps`) : '-'],
      ['residual force', rx.max_force_eV_A === null || rx.max_force_eV_A === undefined
        ? '-' : `${num(rx.max_force_eV_A, 4)} eV/Å`],
      ['dimer bond length', m.bond_length_A === undefined ? '-'
        : `${num(m.bond_length_A, 4)} Å`],
      ['spread over dimers', small(m.bond_length_spread_A, 5)],
      ['buckling', small(m.buckling_A)],
      ['contraction from 1×1', m.contraction_A === undefined ? '-'
        : `${num(m.contraction_A, 4)} Å`],
      ['1×1 symmetry broken', pc.invariant_under_1x1_shift === undefined ? '-'
        : (pc.invariant_under_1x1_shift ? 'no' : 'yes')],
      ['2×1 symmetry held', pc.invariant_under_2x1_shift === undefined ? '-'
        : (pc.invariant_under_2x1_shift
          ? (pc['2x1_shift_is_lattice_vector'] ? 'yes (cell is one 2×1 repeat)' : 'yes')
          : 'no')],
    ]));
    if (geometryStatus !== 'converged') {
      body.append(el('div.note.blocked', { text:
        'The measured values above are not a prediction of the model. '
        + (geometryStatus === 'not-converged'
          ? 'The relaxation stopped before reaching its force target.'
          : 'No relaxation was run.') }));
    }
    if (rc.provenance) {
      for (const a of rc.provenance.approximations || []) {
        body.append(el('div.note', { text: a }));
      }
    }
    body.append(el('div.field-row', {}, [
      el('button.tool.sm', { text: 'Compare to published geometry',
        onclick: () => actions.reconstructionReport() })]));
  }

  if (st.defects && st.defects.length) {
    body.append(el('div.section-title', {}, [el('span', { text: 'Defect history' })]));
    const t = el('table.grid');
    t.append(el('thead', {}, el('tr', {}, [el('th', { text: 'type' }),
      el('th', { text: 'atom' }), el('th', { text: 'detail' })])));
    const tb = el('tbody');
    for (const d of st.defects.slice(-60)) {
      tb.append(el('tr', {}, [
        el('td', { text: d.type }),
        el('td', { text: d.atom_id ?? d.removed_id ?? '-' }),
        el('td', { text: d.to ? `${d.from} → ${d.to}` : (d.element || '') })]));
    }
    t.append(tb);
    body.append(el('div.table-wrap', { style: { maxHeight: '180px' } }, t));
  }

  const wafer = s.wafer;
  if (wafer) {
    body.append(el('div.section-title', {}, [el('span', { text: 'Wafer' }),
      originTag('estimated')]));
    const w = wafer.spec;
    body.append(grid([
      ['material', w.material_id],
      ['diameter', `${w.diameter_mm} mm`],
      ['thickness', `${w.thickness_um} µm`],
      ['orientation', `(${w.orientation.join('')})`],
      ['edge feature', w.edge_feature],
      ['miscut', `${num(w.miscut_deg, 4)}°`],
      ['terrace width', wafer.step_spacing_nm ? `${num(wafer.step_spacing_nm, 4)} nm` : 'singular'],
      ['dopant', w.dopant ? `${w.dopant} at ${w.dopant_concentration_cm3.toExponential(2)} cm⁻³` : 'undoped'],
      ['surface roughness', `${num(w.roughness_rms_A, 3)} Å rms`],
      ['vacancy density', `${w.vacancy_density_cm2.toExponential(2)} cm⁻²`],
      ['temperature', `${w.temperature_K} K`],
      ['procedural seed', w.seed],
    ]));
    body.append(el('div.note.warn', { text:
      'The wafer microstructure is generated procedurally from the seed. It is ' +
      'reproducible and statistically plausible, but it is not a measurement of ' +
      'any real wafer.' }));
  }
}

export function renderInstrument(body, actions) {
  clear(body);
  const opts = state.instrumentOptions;
  if (!opts) { body.append(el('p.empty', { text: 'Loading instrument options…' })); return; }
  const technique = $('#tb-technique').value;

  const field = (label, control, hint) => {
    const wrap = el('div.field-grid', { style: { padding: '2px 8px' } });
    wrap.append(el('label', { text: label }), control);
    if (hint) wrap.append(el('div.span2.hint', { text: hint }));
    return wrap;
  };
  const number = (id, value, step, min, max, unit) => {
    const inp = el('input.num.mono', { type: 'number', id, value, step, min, max });
    const wrap = el('div', { style: { display: 'flex', alignItems: 'center' } },
      [inp, el('span.unit', { text: unit || '' })]);
    return wrap;
  };
  const select = (id, values, current) =>
    el('select', { id }, values.map((v) => el('option', { value: v, selected: v === current }, v)));

  body.append(el('div.section-title', {}, [el('span', { text: 'Probe tip' })]));
  body.append(field('material', select('inst-tip', opts.tips, 'W'),
    'Sets the work function used in the tunnelling barrier.'));
  body.append(field('apex state', select('inst-apex', ['s', 'pz', 'dz2'], 's'),
    'Tersoff-Hamann assumes an s-wave apex; the others are a directional weighting only.'));
  body.append(field('apex radius', number('inst-radius', 50, 5, 1, 2000, 'Å'),
    'Used by the van der Waals background in AFM, not by STM.'));

  if (technique === 'stm') {
    body.append(el('div.section-title', {}, [el('span', { text: 'STM' }), tierTag('tier2-semi-empirical')]));
    body.append(field('mode', select('inst-mode', opts.stm_modes, 'constant-current')));
    body.append(field('bias', number('inst-bias', 1.0, 0.05, -4, 4, 'V'),
      'Positive bias probes empty states, negative bias probes filled states.'));
    body.append(field('setpoint', number('inst-setpoint', 0.5, 0.05, 0.001, 100, 'nA'),
      'Calibrated against the computed signal: absolute currents are not predicted.'));
    body.append(field('height (const-height)', number('inst-height', 5.0, 0.1, 1, 20, 'Å')));
    body.append(field('resolution', el('div', { style: { display: 'flex', gap: '3px' } }, [
      el('input.num.mono', { type: 'number', id: 'inst-resx', value: 192, step: 16, min: 32, max: 1024 }),
      el('input.num.mono', { type: 'number', id: 'inst-resy', value: 192, step: 16, min: 32, max: 1024 })])));
    body.append(field('k-grid', el('div', { style: { display: 'flex', gap: '3px' } }, [
      el('input.num.mono', { type: 'number', id: 'inst-kx', value: 2, step: 1, min: 1, max: 8 }),
      el('input.num.mono', { type: 'number', id: 'inst-ky', value: 2, step: 1, min: 1, max: 8 })]),
      'Surface Brillouin-zone sampling of the slab.'));
    body.append(field('broadening', number('inst-broadening', 0.10, 0.01, 0.005, 1, 'eV')));
    body.append(field('feedback gain', number('inst-gain', 1.0, 0.05, 0.05, 1.5, '')));
  } else {
    body.append(el('div.section-title', {}, [el('span', { text: 'AFM' }), tierTag('tier1-classical')]));
    body.append(field('mode', select('inst-mode', opts.afm_modes, 'fm-afm')));
    body.append(field('height', number('inst-height', 4.0, 0.1, 1, 20, 'Å')));
    body.append(field('force setpoint', number('inst-force', 0.05, 0.01, -5, 5, 'eV/Å')));
    body.append(field('Δf setpoint', number('inst-df', -20, 1, -500, 0, 'Hz')));
    body.append(field('resonance f₀', number('inst-f0', 30000, 500, 100, 5e6, 'Hz')));
    body.append(field('stiffness k', number('inst-k', 1800, 50, 0.1, 1e5, 'N/m')));
    body.append(field('amplitude A', number('inst-amp', 1.0, 0.1, 0.05, 50, 'Å')));
    body.append(field('resolution', el('div', { style: { display: 'flex', gap: '3px' } }, [
      el('input.num.mono', { type: 'number', id: 'inst-resx', value: 160, step: 16, min: 32, max: 1024 }),
      el('input.num.mono', { type: 'number', id: 'inst-resy', value: 160, step: 16, min: 32, max: 1024 })])));
  }

  body.append(el('div.section-title', {}, [el('span', { text: 'Noise and artefacts' })]));
  body.append(field('preset', select('inst-noise', opts.noise_presets, 'realistic'),
    'Instrumental noise only: it describes the measurement chain, not the sample.'));
  body.append(field('seed', number('inst-seed', 0, 1, 0, 1e9, ''),
    'The same seed reproduces the same noise exactly.'));

  const actionsRow = el('div.field-row');
  actionsRow.append(
    el('button.tool.primary', { text: 'Acquire scan', onclick: actions.scan }),
    el('button.tool', { text: 'Point spectroscopy', onclick: actions.spectroscopy,
      title: 'Simulate dI/dV at the tip position' }),
    el('button.tool', { text: 'Force curve', onclick: actions.forceCurve }),
    el('button.tool', { text: 'Detect features', onclick: actions.features }));
  body.append(actionsRow);
  body.append(el('div.note', { text:
    'The tip position used by point spectroscopy and the force curve is the ' +
    'crosshair on the probe image. Move it with the Tip tool.' }));
}

export function readInstrumentSettings(technique) {
  const val = (id, d) => {
    const n = document.getElementById(id);
    return n ? (n.type === 'number' ? parseFloat(n.value) : n.value) : d;
  };
  const base = {
    tip: val('inst-tip', 'W'),
    apex_state: val('inst-apex', 's'),
    tip_radius_A: val('inst-radius', 50),
    noise: val('inst-noise', 'realistic'),
    seed: val('inst-seed', 0),
    resolution: [val('inst-resx', 192), val('inst-resy', 192)],
    mode: val('inst-mode', technique === 'stm' ? 'constant-current' : 'fm-afm'),
  };
  if (technique === 'stm') {
    return { ...base,
      bias_V: parseFloat($('#tb-bias').value),
      setpoint_nA: parseFloat($('#tb-current').value),
      height_A: val('inst-height', 5),
      kgrid: [val('inst-kx', 2), val('inst-ky', 2)],
      broadening_eV: val('inst-broadening', 0.1),
      feedback_gain: val('inst-gain', 1.0) };
  }
  return { ...base,
    height_A: val('inst-height', 4),
    force_setpoint_eV_A: val('inst-force', 0.05),
    frequency_shift_setpoint_Hz: val('inst-df', -20),
    resonance_frequency_Hz: val('inst-f0', 30000),
    stiffness_N_m: val('inst-k', 1800),
    oscillation_amplitude_A: val('inst-amp', 1.0) };
}

export function renderSolvers(body, actions) {
  clear(body);
  const s = state.solvers;
  if (!s) { body.append(el('p.empty', { text: 'Loading solvers…' })); return; }

  body.append(el('div.section-title', {}, [el('span', { text: 'Run' })]));
  const modelSelect = el('select', { id: 'solver-model', style: { width: '100%' } },
    [el('option', { value: 'recommended' }, 'recommended for this material')]
      .concat(s.registered.filter((d) => d.available !== false)
        .map((d) => el('option', { value: d.key, title: d.name }, d.key))));
  const row = el('div.field-grid');
  row.append(el('label', { text: 'model' }), modelSelect);
  body.append(row);

  const buttons = el('div.field-row');
  buttons.append(
    el('button.tool', { text: 'Energy', onclick: () => actions.solve('energy') }),
    el('button.tool.primary', { text: 'Relax', onclick: () => actions.solve('relax') }),
    el('button.tool', { text: 'Dynamics', onclick: () => actions.solve('md') }),
    el('button.tool', { text: 'Electronic structure', onclick: () => actions.solve('electronic') }),
    el('button.tool', { text: 'Band structure', onclick: () => actions.solve('band_structure') }));
  body.append(buttons);

  const params = el('div.field-grid');
  params.append(
    el('label', { text: 'fₘₐₓ' }),
    el('input.num.mono', { type: 'number', id: 'solver-fmax', value: 0.02, step: 0.005, min: 0.001 }),
    el('label', { text: 'max steps' }),
    el('input.num.mono', { type: 'number', id: 'solver-steps', value: 300, step: 50, min: 1 }),
    el('label', { text: 'MD steps' }),
    el('input.num.mono', { type: 'number', id: 'solver-mdsteps', value: 200, step: 50, min: 1 }),
    el('label', { text: 'temperature' }),
    el('input.num.mono', { type: 'number', id: 'solver-temp', value: 300, step: 25, min: 0 }),
    el('label', { text: 'thermostat' }),
    el('select', { id: 'solver-thermostat' },
      [el('option', { value: 'none' }, 'none (NVE)'), el('option', { value: 'langevin' }, 'langevin')]),
    el('label', { text: 'seed' }),
    el('input.num.mono', { type: 'number', id: 'solver-seed', value: 0, step: 1, min: 0 }));
  body.append(params);

  const scLabel = el('label', { style: { display: 'flex', alignItems: 'center', gap: '6px',
    padding: '4px 8px', fontSize: 'var(--fs-sm)' } }, [
    el('input', { type: 'checkbox', id: 'solver-sc' }),
    'request a self-consistent solution']);
  body.append(scLabel);
  body.append(el('div.note', { text:
    'A non-self-consistent model will decline this request and name the solvers ' +
    'that could satisfy it, rather than returning an approximate answer.' }));

  body.append(el('div.section-title', {}, [el('span', { text: 'Registered models' })]));
  const table = el('table.grid');
  table.append(el('thead', {}, el('tr', {}, [el('th', { text: 'solver' }),
    el('th', { text: 'tier' }), el('th', { text: 'capabilities' })])));
  const tb = el('tbody');
  for (const d of s.registered) {
    const tr = el('tr', { title: d.description || '' }, [
      el('td', { text: d.name }),
      el('td', {}, tierTag(d.fidelity)),
      el('td', { text: `${d.capabilities.length}` })]);
    tr.addEventListener('click', () => showCapabilities(body, d));
    tb.append(tr);
  }
  table.append(tb);
  body.append(el('div.table-wrap', { style: { maxHeight: '200px' } }, table));

  const eamBox = el('div', { id: 'eam-section' });
  body.append(eamBox);
  renderEam(eamBox, actions);

  const es = el('div', { id: 'es-section' });
  body.append(es);
  renderElectrostatics(es, actions);

  const dftBox = el('div', { id: 'dft-section' });
  body.append(dftBox);
  renderDft(dftBox, actions);

  body.append(el('div.section-title', {}, [el('span', { text: 'External solvers' })]));
  for (const e of s.external) {
    const box = el('div', { style: { padding: '4px 8px', borderBottom: '1px solid var(--border)' } });
    box.append(el('div', {}, [
      el('b', { text: e.name }), ' ',
      el('span.tag' + (e.available ? '.calculated' : '.unsupported'),
        { text: e.available ? 'available' : 'not installed' })]));
    box.append(el('div.hint', { text: e.description }));
    if (!e.available) box.append(el('div.readout', { text: e.install_hint }));
    box.append(el('div.hint', { text: e.license_note }));
    body.append(box);
  }

  if (s.plugins && s.plugins.length) {
    body.append(el('div.section-title', {}, [el('span', { text: 'Plug-ins' })]));
    for (const p of s.plugins) {
      const box = el('div', { style: { padding: '4px 8px', borderBottom: '1px solid var(--border)' } });
      box.append(el('div', {}, [el('b', { text: `${p.name} ${p.version}` }), ' ',
        el('span.tag' + (p.ok ? '.calculated' : '.unsupported'),
          { text: p.ok ? 'loaded' : 'failed' })]));
      if (p.error) box.append(el('div.readout', { text: p.error }));
      for (const item of p.provided || []) box.append(el('div.hint', { text: item }));
      body.append(box);
    }
  }
}

export function eamSpec() {
  const task = $('#eam-task') ? $('#eam-task').value : 'energy';
  const backend = $('#eam-backend') ? $('#eam-backend').value
    : ((state.eamSpec && state.eamSpec.backend) || 'materia');
  const val = (id, fallback) => ($(id) && $(id).value !== '' ? Number($(id).value) : fallback);
  const settings = {};
  if (backend === 'lammps') {
    if (task === 'relax') {
      settings.ftol_eV_A = val('#eam-fmax', 0.01);
      settings.max_iterations = val('#eam-maxsteps', 1000);
      settings.max_evaluations = 10 * settings.max_iterations;
      settings.min_style = $('#eam-min-style') ? $('#eam-min-style').value : 'cg';
    } else if (task === 'md') {
      const thermostat = $('#eam-thermostat') ? $('#eam-thermostat').value : 'none';
      settings.steps = val('#eam-mdsteps', 200);
      settings.timestep_fs = val('#eam-dt', 1.0);
      settings.temperature_K = val('#eam-temp', 300);
      settings.ensemble = thermostat === 'none' ? 'nve' : thermostat;
      settings.damping_fs = val('#eam-damping', 100);
      settings.seed = Math.max(1, val('#eam-seed', 12345));
      settings.sample_every = val('#eam-sample-every', 10);
      settings.initial_velocities = $('#eam-initial-velocities')
        ? $('#eam-initial-velocities').value : 'keep';
    }
    return { backend, task, potential: $('#eam-potential') ? $('#eam-potential').value : '',
      settings };
  }
  if (task === 'relax') {
    settings.fmax_eV_A = val('#eam-fmax', 0.01);
    settings.max_steps = val('#eam-maxsteps', 1000);
  } else if (task === 'md') {
    settings.steps = val('#eam-mdsteps', 200);
    settings.dt_fs = val('#eam-dt', 2.0);
    settings.temperature_K = val('#eam-temp', 300);
    settings.thermostat = $('#eam-thermostat') ? $('#eam-thermostat').value : 'langevin';
    settings.seed = val('#eam-seed', 0);
    settings.sample_every = val('#eam-sample-every', 1);
  }
  return { backend, task, potential: $('#eam-potential') ? $('#eam-potential').value : '',
    settings };
}

export function renderEam(body, actions) {
  clear(body);
  const st = state.eamStatus;
  body.append(el('div.section-title', {}, [
    el('span', { text: 'Embedded-atom metals (EAM)' }), tierTag('tier1-classical')]));
  if (!st) {
    body.append(el('div.field-row', {}, [
      el('button.tool.sm', { text: 'Load', onclick: () => actions.eamRefresh() })]));
    return;
  }
  if (!st.structure) {
    body.append(el('p.empty', { text: 'No structure.' }));
    return;
  }
  const kept = state.eamSpec || {};
  const potentials = st.potentials || [];
  const select = el('select', { id: 'eam-potential', 'aria-label': 'EAM potential' }, [
    el('option', { value: '' }, st.structure.selected
      ? `automatic: ${st.structure.selected}` : 'automatic: none covers this structure')]
    .concat(potentials.map((p) => el('option', { value: p.id },
      `${p.id} (${p.elements.join(', ')})`)))
    .concat((st.user_files || []).map((f) => el('option', { value: f },
      `user file: ${f.split('/').pop()}`))));
  select.value = kept.potential || '';
  const chosenId = select.value || st.structure.selected;
  const chosen = potentials.find((p) => p.id === chosenId);

  const grid = el('table.pgrid');
  const add = (k, v, cls, title) => grid.append(el('tr', {}, [el('th', { text: k }),
    el('td.mono' + (cls ? '.' + cls : ''), { text: v === null || v === undefined ? '-' : String(v),
      title: title || null })]));
  add('structure elements', st.structure.elements.join(', '));
  add('potential', chosenId || 'none', chosenId ? '' : 'blocked');
  body.append(grid);
  if (!st.structure.selected && !select.value) {
    body.append(el('div.note.blocked', { text: st.structure.reason }));
  }
  if (st.structure.blocking) {
    body.append(el('div.note.blocked', { id: 'eam-blocking', text: st.structure.blocking }));
  }
  const lmp = state.lammpsStatus;
  const lmpEnv = lmp && lmp.environment ? lmp.environment : null;
  const backendSelect = el('select', { id: 'eam-backend', 'aria-label': 'EAM backend' }, [
    el('option', { value: 'materia' }, 'Materia EAM (built in)'),
    el('option', { value: 'lammps' }, lmpEnv && lmpEnv.available
      ? `LAMMPS ${lmpEnv.version} (external)` : 'LAMMPS (not installed)')]);
  backendSelect.value = kept.backend || 'materia';
  const lammps = backendSelect.value === 'lammps';
  const fields = el('div.field-grid');
  fields.append(el('label', { text: 'backend', for: 'eam-backend' }), backendSelect,
    el('label', { text: 'potential', for: 'eam-potential' }), select);
  body.append(fields);
  select.addEventListener('change', () => actions.eamChanged());
  backendSelect.addEventListener('change', () => actions.eamChanged());
  let lammpsBlocked = false;
  if (lammps) {
    if (!lmpEnv || !lmpEnv.available) {
      lammpsBlocked = true;
      body.append(el('div.note.blocked', { id: 'lammps-blocking', text:
        `${lmpEnv ? lmpEnv.blocking_reason : 'LAMMPS status is unavailable.'} `
        + `${lmpEnv ? lmpEnv.install_hint : ''}` }));
    } else {
      const info = el('table.pgrid');
      const irow = (k, v) => info.append(el('tr', {}, [el('th', { text: k }),
        el('td.mono', { text: v === null || v === undefined || v === '' ? '-' : String(v) })]));
      irow('LAMMPS', lmpEnv.version);
      irow('route', `${lmpEnv.kind}: ${lmpEnv.path}`);
      irow('packages', (lmpEnv.packages || []).join(' '));
      body.append(info);
      const blocking = (lmp.structure && lmp.structure.blocking) || [];
      if (blocking.length) {
        lammpsBlocked = true;
        for (const b of blocking) body.append(el('div.note.blocked', { text: b }));
      }
    }
  }

  if (chosen) {
    const d = el('table.pgrid');
    const row = (k, v, title) => d.append(el('tr', {}, [el('th', { text: k }),
      el('td', { text: v === null || v === undefined ? '-' : String(v), title: title || null })]));
    row('family', chosen.family);
    row('elements', chosen.elements.join(', '));
    row('intended structure', Object.entries(chosen.intended_structures || {})
      .map(([e, s]) => `${e} ${s}`).join(', '));
    row('file', chosen.file);
    row('SHA-256', `${chosen.sha256.slice(0, 16)}...`, chosen.sha256);
    row('cutoff', `${num(chosen.cutoff_A, 7)} Å`);
    row('source', chosen.openkim_id, chosen.source_url);
    row('origin', chosen.content_origin);
    row('licence', `${chosen.license}: "${chosen.license_text}"`);
    row('retrieved', chosen.retrieved);
    body.append(d);
    for (const c of chosen.citations || []) body.append(el('div.hint', { text: c }));
    for (const n of chosen.notes || []) body.append(el('div.note', { text: n }));
  }

  const task = el('select', { id: 'eam-task', 'aria-label': 'EAM task' }, [
    el('option', { value: 'energy' }, lammps ? 'energy, forces and stress' : 'energy and forces'),
    el('option', { value: 'relax' }, lammps ? 'relax positions (LAMMPS minimize, fixed cell)'
      : 'relax positions (FIRE, fixed cell)'),
    el('option', { value: 'md' }, 'molecular dynamics')]);
  task.value = kept.task || 'energy';
  const ks = kept.settings || {};
  const params = el('div.field-grid');
  params.append(el('label', { text: 'task', for: 'eam-task' }), task);
  if (task.value === 'relax') {
    params.append(
      el('label', { text: 'fₘₐₓ / eV Å⁻¹', for: 'eam-fmax' }),
      el('input.num.mono', { type: 'number', id: 'eam-fmax', value: ks.fmax_eV_A ?? 0.01, step: 0.001, min: 0.0001 }),
      el('label', { text: 'max steps', for: 'eam-maxsteps' }),
      el('input.num.mono', { type: 'number', id: 'eam-maxsteps',
        value: ks.max_steps ?? ks.max_iterations ?? 1000, step: 100, min: 1 }));
    if (lammps) {
      const minStyle = el('select', { id: 'eam-min-style', 'aria-label': 'minimizer' }, [
        el('option', { value: 'cg' }, 'conjugate gradient (cg)'),
        el('option', { value: 'fire' }, 'FIRE'),
        el('option', { value: 'sd' }, 'steepest descent (sd)')]);
      minStyle.value = ks.min_style || 'cg';
      params.append(el('label', { text: 'minimizer', for: 'eam-min-style' }), minStyle);
    }
  } else if (task.value === 'md') {
    const thermostat = el('select', { id: 'eam-thermostat', 'aria-label': 'thermostat' }, [
      el('option', { value: 'langevin' }, 'Langevin (NVT)')]
      .concat(lammps ? [el('option', { value: 'nvt' }, 'Nose-Hoover (NVT)')] : [])
      .concat([el('option', { value: 'none' }, 'none (NVE)')]));
    thermostat.value = ks.thermostat
      || (ks.ensemble ? (ks.ensemble === 'nve' ? 'none' : ks.ensemble) : 'langevin');
    params.append(
      el('label', { text: 'steps', for: 'eam-mdsteps' }),
      el('input.num.mono', { type: 'number', id: 'eam-mdsteps', value: ks.steps ?? 200, step: 50, min: 1 }),
      el('label', { text: 'time step / fs', for: 'eam-dt' }),
      el('input.num.mono', { type: 'number', id: 'eam-dt', value: ks.dt_fs ?? 2.0, step: 0.5, min: 0.1, max: 10 }),
      el('label', { text: 'temperature / K', for: 'eam-temp' }),
      el('input.num.mono', { type: 'number', id: 'eam-temp', value: ks.temperature_K ?? 300, step: 25, min: 0 }),
      el('label', { text: 'thermostat', for: 'eam-thermostat' }), thermostat,
      el('label', { text: 'seed', for: 'eam-seed' }),
      el('input.num.mono', { type: 'number', id: 'eam-seed', value: ks.seed ?? 0, step: 1, min: 0 }),
      el('label', { text: 'store every', for: 'eam-sample-every' }),
      el('input.num.mono', { type: 'number', id: 'eam-sample-every',
        value: ks.sample_every ?? (lammps ? 10 : 1), step: 1, min: 1 }));
    if (lammps) {
      const initial = el('select', { id: 'eam-initial-velocities',
        'aria-label': 'initial velocities' }, [
        el('option', { value: 'keep' }, 'keep the structure velocities'),
        el('option', { value: 'create' }, 'create at the temperature')]);
      initial.value = ks.initial_velocities || 'keep';
      params.append(
        el('label', { text: 'damping / fs', for: 'eam-damping' }),
        el('input.num.mono', { type: 'number', id: 'eam-damping', value: ks.damping_fs ?? 100,
          step: 10, min: 1 }),
        el('label', { text: 'initial velocities', for: 'eam-initial-velocities' }), initial);
    }
  }
  body.append(params);
  task.addEventListener('change', () => actions.eamChanged());
  const runnable = Boolean(chosenId) && (lammps ? !lammpsBlocked : !st.structure.blocking);
  body.append(el('div.field-row', {}, [
    el('button.tool.primary', { text: lammps ? 'Run LAMMPS' : 'Run EAM', disabled: !runnable,
      onclick: () => actions.eamRun(eamSpec()) })]));
  body.append(el('div.hint', { text: lammps
    ? 'LAMMPS runs in its own process from a frozen specification with its exact input, '
      + 'version and file hashes recorded. Nothing is kept if it is refused, fails or is '
      + 'cancelled. A finished run is applied as one undoable change only if the structure '
      + 'has not changed meanwhile, and a relaxation only if it converged. Materia never '
      + 'substitutes its own EAM solver for LAMMPS.'
    : 'The run works on a copy. Its forces, relaxed positions or final dynamics state are '
      + 'written back as one undoable change only if the structure has not changed '
      + 'meanwhile, and a relaxation only if it converged. Cancel from the job log.' }));

  renderEamResult(body, state.eamResult, actions);
}

function renderEamResult(body, r, actions) {
  if (!r) return;
  const external = r.backend === 'lammps';
  body.append(el('div.section-title', {}, [el('span', { text: external ? 'Last LAMMPS run'
    : 'Last EAM run' }),
    r.status === 'cancelled' ? el('span.tag', { text: 'cancelled' })
      : originTag(r.ok ? r.origin : 'unsupported')]));
  if (!r.ok) {
    body.append(el(r.status === 'cancelled' ? 'div.note.warn' : 'div.note.blocked',
      { text: r.reason || r.error || '' }));
    return;
  }
  if (!r.current) {
    body.append(el('div.note.warn', { text:
      'The structure changed after this run, or its result was not applied. The values '
      + 'below describe the geometry the run produced, not the structure now on screen.' }));
  }
  if (r.convergence && r.convergence.converged === false) {
    body.append(el('div.note.warn', { text: r.convergence.message }));
  }
  if (r.potential_file_matches === false) {
    body.append(el('div.note.blocked', { text:
      'The potential file installed now differs from the one that produced this result.' }));
  }
  const t = el('table.pgrid');
  const row = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td.mono', { text: v === null || v === undefined ? '-' : String(v) })]));
  row('task', r.task);
  row('structure', r.structure_key);
  row('potential', `${r.potential.id}, ${r.potential.sha256.slice(0, 12)}...`);
  row('energy', `${num(r.energy_eV, 10)} eV`);
  row('energy per atom', `${num(r.energy_per_atom_eV, 8)} eV`);
  row('max |F|', `${num(r.max_force_eV_A, 5)} eV/Å`);
  if (r.energy_change_eV !== null && r.energy_change_eV !== undefined) {
    row('energy change', `${num(r.energy_change_eV, 6)} eV`);
  }
  if (r.steps !== undefined) row('steps', r.steps);
  if (r.mean_temperature_K !== undefined) {
    row('mean temperature', `${num(r.mean_temperature_K, 4)} ± ${num(r.temperature_stddev_K, 3)} K`);
  }
  if (r.energy_drift_eV !== undefined && r.energy_drift_eV !== null) {
    row('total-energy drift', `${num(r.energy_drift_eV, 4)} eV`);
  }
  if (r.convergence) row('convergence', r.convergence.message);
  row('applied', r.applied ? 'yes' : 'no');
  row('current', r.current ? 'yes' : 'no, stale');
  row('atoms', r.n_atoms);
  if (external) {
    if (r.stress_eV_A3) {
      row('pressure', `${num(r.pressure_bar, 6)} bar`);
      row('stress xx, yy, zz', r.stress_eV_A3.map((line, i) => num(line[i], 5)).join(', ')
        + ' eV/Å³');
    }
    if (r.stopping_criterion) row('LAMMPS stopped on', r.stopping_criterion);
    row('state', r.run_state);
    row('LAMMPS', r.lammps ? r.lammps.version : '-');
    row('command', r.command_line);
    const script = r.inputs ? r.inputs['in.lammps'] : null;
    row('input SHA-256', script ? `${script.sha256.slice(0, 16)}...` : '-');
    row('specification', r.spec_digest ? `${r.spec_digest.slice(0, 16)}...` : '-');
  } else {
    row('density beyond table', r.rho_extrapolated ? `${r.rho_extrapolated} atom(s)` : 'none');
  }
  row('wall time', `${num(r.wall_time_s, 4)} s`);
  row('run', r.run_id);
  body.append(t);
  if (r.apply_reason) body.append(el('div.hint', { text: r.apply_reason }));
  if (external && r.state_reason) body.append(el('div.hint', { text: r.state_reason }));
  if (external && r.applicable) {
    body.append(el('div.field-row', {}, [
      el('button.tool.sm', { id: 'lammps-apply', text: 'Apply to structure',
        onclick: () => actions.lammpsApply(r.run_id) })]));
  }
  if (r.rho_extrapolated) {
    body.append(el('div.note.warn', { text:
      'Some atoms reached a density beyond the tabulated range and were extrapolated '
      + 'linearly, as LAMMPS does. The structure is far outside the fitted conditions.' }));
  }
  if (r.trajectory_available) {
    const meta = state.trajectory && state.trajectory.run_id === r.run_id
      ? state.trajectory : null;
    const last = Math.max(0, (meta ? meta.frames : r.frames) - 1);
    const index = Math.min(state.trajectoryFrame || 0, last);
    const time = meta && meta.times_fs ? meta.times_fs[index] : null;
    const controls = el('div.trajectory-controls', {}, [
      el('button.tool.sm', { id: 'eam-playback-toggle',
        text: state.trajectoryPlaying ? 'Pause' : 'Play',
        onclick: () => actions.eamTogglePlayback(r.run_id) }),
      el('input.trajectory-range', { id: 'eam-frame', type: 'range', min: 0, max: last,
        step: 1, value: index, 'aria-label': 'trajectory frame',
        oninput: (event) => actions.eamFrame(r.run_id, Number(event.target.value)) }),
      el('select', { id: 'eam-playback-speed', 'aria-label': 'playback speed',
        onchange: (event) => actions.eamPlaybackSpeed(Number(event.target.value)) },
      [0.25, 0.5, 1, 2, 4].map((speed) => el('option', {
        value: speed, text: `${speed}x`, selected: speed === state.trajectorySpeed })))
    ]);
    body.append(el('div.section-title', {}, [el('span', { text: 'Trajectory' }),
      el('span.tag.calculated', { text: `${r.frames} frames` })]));
    body.append(controls);
    body.append(el('div.readout.mono', { id: 'eam-frame-readout',
      text: `frame ${index + 1} / ${last + 1}` +
        (time === null || time === undefined ? '' : `, ${num(time, 6)} fs`) }));
    body.append(el('div.readout', { id: 'eam-fragment-readout', text:
      'Select a frame to inspect fragments and velocities.' }));
    body.append(el('div.field-row', {}, [
      el('button.tool.sm', { text: 'Show final structure', onclick: actions.eamPlaybackExit })]));
    body.append(el('div.hint', { text:
      `Playback shows classical nuclei under the selected EAM potential${external
        ? ', as computed by LAMMPS' : ''}. Fragment groups `
      + 'are a covalent-radius contact diagnostic, not a quantum bond-order calculation.' }));
  }
  for (const a of r.approximations || []) body.append(el('div.hint', { text: a }));
}

const ES_COMPONENT_LABELS = {
  real: 'real space', reciprocal: 'reciprocal space', self: 'self interaction',
  background: 'neutralising background', surface: 'vacuum surface term',
  slab_correction: 'slab dipole correction', direct: 'direct pair sum',
};

export function electrostaticsSpec() {
  const kind = $('#es-kind') ? $('#es-kind').value : 'per-element';
  const byElement = {};
  for (const input of document.querySelectorAll('#es-section input[data-element]')) {
    if (input.value.trim() !== '') byElement[input.dataset.element] = Number(input.value);
  }
  return {
    kind,
    by_element: byElement,
    source: $('#es-source') ? $('#es-source').value : '',
    settings: {
      accuracy: Number($('#es-accuracy') ? $('#es-accuracy').value : 1e-8),
      surrounding: $('#es-surrounding') ? $('#es-surrounding').value : 'tinfoil',
      background: $('#es-background') ? $('#es-background').value : 'none',
      check_convergence: $('#es-check') ? $('#es-check').checked : true,
    },
  };
}

export function renderElectrostatics(body, actions) {
  clear(body);
  const st = state.esStatus;
  body.append(el('div.section-title', {}, [
    el('span', { text: 'Electrostatics (point charges)' }), tierTag('tier1-classical')]));
  if (!st) {
    body.append(el('div.field-row', {}, [
      el('button.tool.sm', { text: 'Load', onclick: () => actions.esRefresh() })]));
    return;
  }
  if (!st.structure) {
    body.append(el('p.empty', { text: 'No structure.' }));
    return;
  }
  const kept = state.esSpec || {};
  const keptSettings = kept.settings || st.settings || {};
  const model = st.charge_model;
  const acct = st.charge_accounting;
  const grid = el('table.pgrid');
  const add = (k, v, cls) => grid.append(el('tr', {}, [el('th', { text: k }),
    el('td.mono' + (cls ? '.' + cls : ''), { text: v === null || v === undefined ? '-' : String(v) })]));
  add('geometry', st.geometry.description, st.geometry.supported ? '' : 'blocked');
  add('charge model', model ? model.kind : 'none assigned', model ? '' : 'blocked');
  if (model && model.source) add('stated source', model.source);
  if (acct) {
    add('net point charge', `${num(acct.total_charge_e, 6)} e`);
    add('sum of |q|', `${num(acct.sum_abs_charge_e, 6)} e`);
  }
  add('net formal charge', `${num(st.formal_total_e, 6)} e`);
  body.append(grid);

  if (acct && acct.per_element) {
    const t = el('table.grid');
    t.append(el('thead', {}, el('tr', {}, ['element', 'atoms', 'q / e', 'total / e']
      .map((h) => el('th', { text: h })))));
    const tb = el('tbody');
    for (const [sym, e] of Object.entries(acct.per_element)) {
      const q = e.min_e === e.max_e ? num(e.min_e, 5) : `${num(e.min_e, 4)} to ${num(e.max_e, 4)}`;
      tb.append(el('tr', {}, [el('td', { text: sym }), el('td.mono', { text: String(e.count) }),
        el('td.mono', { text: q }), el('td.mono', { text: num(e.total_e, 6) })]));
    }
    t.append(tb);
    body.append(el('div.table-wrap', { style: { maxHeight: '140px' } }, t));
  }

  const kind = el('select', { id: 'es-kind', 'aria-label': 'charge model' }, [
    el('option', { value: 'per-element' }, 'per element (stated charges)'),
    el('option', { value: 'formal-point-ion' }, 'formal point ion (oxidation states)')]);
  kind.value = kept.kind || (model && model.kind !== 'per-atom' ? model.kind : 'per-element');
  const fields = el('div.field-grid');
  fields.append(el('label', { text: 'assign' }), kind);
  const current = model && model.kind === 'per-element' ? model.by_element : {};
  const keptQ = kept.by_element || {};
  for (const sym of st.elements || []) {
    const value = keptQ[sym] ?? current[sym];
    fields.append(
      el('label', { text: `q(${sym}) / e`, for: `es-q-${sym}` }),
      el('input.num.mono', { type: 'number', id: `es-q-${sym}`, step: 0.1,
        'data-element': sym, value: value === undefined ? '' : value,
        disabled: kind.value !== 'per-element' }));
  }
  const source = el('input', { type: 'text', id: 'es-source',
    value: kept.source ?? (model ? model.source : ''), 'aria-label': 'charge source' });
  fields.append(el('label', { text: 'source', for: 'es-source' }), source);
  body.append(fields);
  kind.addEventListener('change', () => {
    for (const input of body.querySelectorAll('input[data-element]')) {
      input.disabled = kind.value !== 'per-element';
    }
  });
  body.append(el('div.field-row', {}, [
    el('button.tool', { text: 'Assign charges', onclick: () => actions.esAssign(electrostaticsSpec()) }),
    el('button.tool', { text: 'Remove charges', disabled: !model,
      onclick: () => actions.esClear() })]));
  body.append(el('div.hint', { text:
    'Charges are never inferred. An element left blank is refused, not assumed neutral. '
    + 'Per-atom charges are set from Python: electrostatics.assign(by_atom_id={...}).' }));

  const bulk = st.geometry.kind === 'bulk-3d';
  const accuracy = el('select', { id: 'es-accuracy', 'aria-label': 'accuracy' },
    [['1e-6', '10⁻⁶'], ['1e-8', '10⁻⁸'], ['1e-10', '10⁻¹⁰']].map(([v, t]) =>
      el('option', { value: v }, t)));
  accuracy.value = String(keptSettings.accuracy ?? '1e-8').replace('1e-08', '1e-8')
    .replace('1e-06', '1e-6');
  const surrounding = el('select', { id: 'es-surrounding', disabled: !bulk,
    'aria-label': 'surrounding' }, [
    el('option', { value: 'tinfoil' }, 'conductor (tin foil)'),
    el('option', { value: 'vacuum' }, 'vacuum (spherical crystal)')]);
  surrounding.value = bulk ? (keptSettings.surrounding || 'tinfoil') : 'tinfoil';
  const background = el('select', { id: 'es-background', disabled: !bulk,
    'aria-label': 'background' }, [
    el('option', { value: 'none' }, 'none'),
    el('option', { value: 'uniform' }, 'uniform neutralising')]);
  background.value = bulk ? (keptSettings.background || 'none') : 'none';
  const check = el('input', { type: 'checkbox', id: 'es-check' });
  check.checked = keptSettings.check_convergence ?? true;
  const opts = el('div.field-grid');
  opts.append(
    el('label', { text: 'accuracy', for: 'es-accuracy' }), accuracy,
    el('label', { text: 'surrounding', for: 'es-surrounding' }), surrounding,
    el('label', { text: 'background', for: 'es-background' }), background,
    el('label', { text: 'convergence check', for: 'es-check' }), check);
  body.append(opts);
  for (const reason of st.blocking || []) body.append(el('div.note.blocked', { text: reason }));
  for (const w of st.warnings || []) body.append(el('div.note.warn', { text: w }));
  body.append(el('div.field-row', {}, [
    el('button.tool.primary', { text: 'Compute electrostatics', disabled: !st.ready,
      onclick: () => actions.esRun(electrostaticsSpec().settings) })]));

  renderElectrostaticsResult(body, state.esResult);
}

function renderElectrostaticsResult(body, r) {
  if (!r) return;
  body.append(el('div.section-title', {}, [el('span', { text: 'Last electrostatics run' }),
    originTag(r.ok ? r.origin : 'unsupported')]));
  if (!r.ok) {
    body.append(el(r.status === 'cancelled' ? 'div.note.warn' : 'div.note.blocked',
      { text: r.reason || r.error || '' }));
    if (r.status) body.append(el('div.hint', { text: `status: ${r.status}` }));
    return;
  }
  if (!r.current) {
    body.append(el('div.note.warn', { text:
      'The structure or its charges changed after this run. The values below describe '
      + 'the earlier state; the inspector no longer shows them per atom.' }));
  }
  if (r.convergence && !r.convergence.converged) {
    body.append(el('div.note.warn', { text: r.convergence.message }));
  }
  const t = el('table.pgrid');
  const row = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td.mono', { text: v === null || v === undefined ? '-' : String(v) })]));
  row('method', r.parameters.method);
  row('boundary', r.boundary_conditions);
  row('energy', `${num(r.energy_eV, 10)} eV`);
  if (r.uncertainty_eV !== null && r.uncertainty_eV !== undefined) {
    row('split difference', `${num(r.uncertainty_eV, 3)} eV`);
  }
  row('energy per atom', `${num(r.energy_per_atom_eV, 8)} eV`);
  for (const [k, v] of Object.entries(r.components_eV || {})) {
    row(ES_COMPONENT_LABELS[k] || k, `${num(v, 10)} eV`);
  }
  row('max |F|', `${num(r.max_force_eV_A, 5)} eV/Å`);
  row('max |E| at a site', `${num(r.max_field_V_A, 5)} V/Å`);
  if (r.net_force_eV_A) row('net force', r.net_force_eV_A.map((v) => num(v, 3)).join(', ') + ' eV/Å');
  const p = r.parameters;
  if (p.alpha_per_A !== undefined) {
    row('α', `${num(p.alpha_per_A, 5)} Å⁻¹`);
    row('real cutoff', `${num(p.real_cutoff_A, 5)} Å`);
    row('reciprocal cutoff', `${num(p.kspace_cutoff_per_A, 5)} Å⁻¹`);
    row('k vectors (half space)', p.n_kvectors_half_space);
    row('real-space pairs', p.real_space_pairs);
  }
  if (p.internal_vacuum_gap_A !== undefined) {
    row('slab thickness', `${num(p.slab_thickness_A, 5)} Å`);
    row('internal vacuum gap', `${num(p.internal_vacuum_gap_A, 5)} Å`);
    row('slab dipole', `${num(p.slab_dipole_e_A, 5)} e·Å`);
  }
  if (r.convergence) row('convergence', r.convergence.message);
  row('run', r.run_id);
  row('input digest', r.inputs_digest);
  body.append(t);

  if (r.site_summary && r.site_summary.length) {
    const s = el('table.grid');
    s.append(el('thead', {}, el('tr', {}, ['site group', 'n', 'mean φ / V', 'min / V', 'max / V']
      .map((h) => el('th', { text: h })))));
    const tb = el('tbody');
    for (const g of r.site_summary) {
      tb.append(el('tr', {}, [el('td', { text: g.group }), el('td.mono', { text: String(g.count) }),
        el('td.mono', { text: num(g.potential_mean_V, 6) }),
        el('td.mono', { text: num(g.potential_min_V, 6) }),
        el('td.mono', { text: num(g.potential_max_V, 6) })]));
    }
    s.append(tb);
    body.append(el('div.table-wrap', { style: { maxHeight: '140px' } }, s));
  } else if (!r.per_atom_stored) {
    body.append(el('div.hint', { text:
      'Per-atom values were too large to store in the project file.' }));
  }
  for (const a of r.approximations || []) body.append(el('div.hint', { text: a }));
  for (const line of r.log || []) body.append(el('div.note.info', { text: line }));
}

function showCapabilities(body, d) {
  const existing = body.querySelector('.cap-detail');
  if (existing) existing.remove();
  const box = el('div.cap-detail', { style: { padding: '6px 8px',
    background: 'var(--panel-alt)', borderTop: '1px solid var(--border-strong)' } });
  box.append(el('div', {}, [el('b', { text: d.name }), ' ', tierTag(d.fidelity)]));
  box.append(el('div.hint', { text: d.description || '' }));
  box.append(el('div', { style: { marginTop: '4px' } }, [
    el('b', { text: 'provides: ' }),
    el('span.readout', { text: d.capabilities.join(', ') || 'nothing' })]));
  box.append(el('div', { style: { marginTop: '3px' } }, [
    el('b', { text: 'does not provide: ' }),
    el('span.readout', { text: (d.missing_capabilities || []).join(', ') })]));
  body.append(box);
}

export function renderMeasure(body, actions) {
  clear(body);
  const sel = state.server ? state.server.selection.ids : [];
  body.append(el('div.section-title', {}, [el('span', { text: 'Selection' })]));
  body.append(el('div.field-row', {}, [
    el('span.readout', { text: `${sel.length} atom(s) selected` }),
    el('button.tool.sm', { text: 'Clear', onclick: () => actions.select('none') })]));

  const results = el('div', { id: 'measure-results' });
  body.append(el('div.section-title', {}, [el('span', { text: 'Geometry' })]));
  const btns = el('div.field-row');
  btns.append(
    el('button.tool.sm', { text: 'Distance (2)', onclick: () => actions.measure('distance') }),
    el('button.tool.sm', { text: 'Angle (3)', onclick: () => actions.measure('angle') }),
    el('button.tool.sm', { text: 'Dihedral (4)', onclick: () => actions.measure('dihedral') }),
    el('button.tool.sm', { text: 'Coordination', onclick: () => actions.measure('coordination') }),
    el('button.tool.sm', { text: 'g(r)', onclick: () => actions.measure('rdf') }));
  body.append(btns);
  body.append(results);
  body.append(el('div.note', { text:
    'Distances and angles use the minimum-image convention for periodic cells. ' +
    'Values are exact geometry of the current model, with no experimental uncertainty.' }));

  body.append(el('div.section-title', {}, [el('span', { text: 'Edit selected atoms' })]));
  const editGrid = el('div.field-grid');
  const elementInput = el('input', { type: 'text', id: 'edit-element', value: 'P',
    style: { width: '70px' }, 'aria-label': 'Element symbol' });
  const chargeInput = el('input.num.mono', { type: 'number', id: 'edit-charge', value: 0, step: 1 });
  const spinInput = el('input.num.mono', { type: 'number', id: 'edit-spin', value: 0, step: 0.5 });
  editGrid.append(
    el('label', { text: 'element' }), elementInput,
    el('label', { text: 'charge / e' }), chargeInput,
    el('label', { text: 'spin S' }), spinInput);
  body.append(editGrid);
  const editBtns = el('div.field-row');
  editBtns.append(
    el('button.tool.sm', { text: 'Substitute', onclick: () => actions.edit('substitute',
      { element: elementInput.value }) }),
    el('button.tool.sm.danger', { text: 'Vacancy', onclick: () => actions.edit('vacancy', {}) }),
    el('button.tool.sm', { text: 'Set charge', onclick: () => actions.edit('set_charge',
      { charge: parseFloat(chargeInput.value) }) }),
    el('button.tool.sm', { text: 'Set spin', onclick: () => actions.edit('set_spin',
      { spin: parseFloat(spinInput.value) }) }),
    el('button.tool.sm', { text: 'Fix', onclick: () => actions.edit('fix', { fixed: true }) }),
    el('button.tool.sm', { text: 'Release', onclick: () => actions.edit('fix', { fixed: false }) }));
  body.append(editBtns);
  body.append(el('div.note.warn', { text:
    'Changing charge or spin changes the electron bookkeeping. Classical potentials ' +
    'ignore charge entirely and the tight-binding model is restricted to neutral ' +
    'systems; the warnings panel will say so when it matters.' }));
}

export function renderTree(container, actions) {
  clear(container);
  const s = state.server;
  if (!s) return;
  const node = (label, depth, value, opts = {}) => {
    const n = el('div.node', { style: { paddingLeft: `${6 + depth * 13}px` },
      title: opts.title || '', role: 'treeitem' });
    n.append(el('span.twisty', { text: opts.leaf ? '·' : '▾' }));
    n.append(el('span', { text: label }));
    if (value !== undefined && value !== null) n.append(el('span.val', { text: String(value) }));
    if (opts.onclick) n.addEventListener('click', opts.onclick);
    if (opts.active) n.classList.add('active');
    container.append(n);
    return n;
  };

  node(s.project.name, 0, '', { title: 'Project' });
  if (s.wafer) {
    const w = s.wafer.spec;
    node(`Wafer - ${w.material_id}`, 1, `${w.diameter_mm} mm`,
      { onclick: () => actions.setView('wafer') });
    node(`orientation (${w.orientation.join('')})`, 2, '', { leaf: true });
    node(`thickness`, 2, `${w.thickness_um} µm`, { leaf: true });
    if (w.dopant) node(`dopant ${w.dopant}`, 2,
      `${w.dopant_concentration_cm3.toExponential(1)} cm⁻³`, { leaf: true });
    if ((w.layers || []).length) {
      node('Layer stack', 2, `${w.layers.length}`);
      for (const l of w.layers) node(l.name, 3, `${l.thickness_nm} nm`, { leaf: true });
    }
    if ((w.device_regions || []).length) {
      node('Device regions', 2, `${w.device_regions.length}`);
      for (const r of w.device_regions) node(r.name, 3, r.kind, { leaf: true });
    }
  } else {
    node('No wafer', 1, '', { leaf: true });
  }

  const regions = Object.entries(s.project.regions || {});
  node('Atomistic regions', 1, `${regions.length}`);
  for (const [id, r] of regions) {
    node(id, 2, `${r.n_atoms} atoms`, {
      active: id === s.project.active_region,
      onclick: () => actions.setView('atoms'),
      title: `at (${num(r.spec.x_mm, 3)}, ${num(r.spec.y_mm, 3)}) mm`,
    });
  }

  node('Structures', 1, `${s.project.structures.length}`);
  for (const k of s.project.structures) {
    node(k, 2, '', { leaf: true, active: k === s.project.active_structure });
  }

  node('Checkpoints', 1, `${s.project.checkpoints.length}`);
  for (const c of s.project.checkpoints) {
    node(c, 2, 'restore', { leaf: true, onclick: () => actions.restoreCheckpoint(c) });
  }

  node('Results', 1, `${s.project.results.length}`);
  for (const r of s.project.results.slice(0, 40)) node(r, 2, '', { leaf: true });

  node('Scans', 1, `${s.project.scans.length}`);
  for (const k of s.project.scans) {
    node(k, 2, 'show', { leaf: true, onclick: () => actions.showScan(k) });
  }

  node('Selection', 1, `${s.selection.ids.length}`);
  if (s.selection.query) node(s.selection.query, 2, '', { leaf: true });
}

export function renderMaterialList(container, detail, actions) {
  const filter = ($('#material-search').value || '').toLowerCase();
  clear(container);
  for (const m of state.materialList || []) {
    const hay = `${m.id} ${m.name} ${m.formula} ${m.category} ${(m.aliases || []).join(' ')}`.toLowerCase();
    if (filter && !hay.includes(filter)) continue;
    const row = el('div.row', { onclick: () => actions.selectMaterial(m.id) }, [
      el('span', { text: m.name }),
      el('span.sub', { text: m.formula })]);
    if (state.material && state.material.summary.id === m.id) row.classList.add('active');
    container.append(row);
  }
}

export function renderMaterialDetail(container, actions) {
  clear(container);
  const d = state.material;
  if (!d) { container.append(el('p.empty', { text: 'Select a material.' })); return; }
  const s = d.summary;
  const grid = el('table.pgrid');
  const add = (k, v, cls) => grid.append(el('tr', {}, [el('th', { text: k }),
    el('td', cls ? { class: cls } : {}, v === null || v === undefined ? '-' : String(v))]));
  add('id', s.id);
  add('formula', s.formula);
  add('category', s.category);
  add('prototype', s.prototype);
  add('space group', s.space_group);
  add('lattice a, b, c', s.lattice_A.map((v) => num(v, 6)).join(', ') + ' Å');
  add('angles', s.angles_deg.map((v) => num(v, 3)).join(', ') + '°');
  add('basis sites', s.n_basis);
  add('elements', s.elements.join(', '));
  add('orientations', s.orientations.map((o) => `(${o.join('')})`).join(' '));
  add('licence', s.license, 'txt');
  container.append(grid);

  container.append(el('div.section-title', {}, [el('span', { text: 'Properties' }),
    originTag('reference')]));
  const pt = el('table.pgrid');
  for (const [k, p] of Object.entries(d.properties)) {
    const td = el('td', { text: `${p.value}${p.unit ? ' ' + p.unit : ''}` });
    const tr = el('tr', { title: `${p.conditions || ''}\n${p.source || ''}\n${p.note || ''}` },
      [el('th', { text: k.replace(/_/g, ' ') }), td]);
    pt.append(tr);
  }
  container.append(pt);

  if (d.reconstructions.length) {
    container.append(el('div.section-title', {}, [el('span', { text: 'Known reconstructions' })]));
    for (const r of d.reconstructions) {
      const box = el('div', { style: { padding: '4px 8px', borderBottom: '1px solid var(--border)' } });
      box.append(el('div', {}, [el('b', { text: r.id }), ' ',
        el('span.readout', { text: `(${r.orientation.join('')})` }), ' ',
        el('span.tag' + (r.supported ? '.calculated' : '.unsupported'),
          { text: r.supported ? 'available' : 'not implemented' })]));
      box.append(el('div.hint', { text: r.description }));
      if (r.reason) box.append(el('div.hint', { text: r.reason }));
      if (r.note) box.append(el('div.hint', { text: r.note }));
      if (r.supported && r.generator) {
        box.append(el('div.hint', { text: `Method: ${r.generator.method}` }));
      }
      const geom = Object.entries(r.reference_geometry || {});
      if (geom.length) {
        const t = el('table.pgrid');
        for (const [k, g] of geom) {
          const unc = g.uncertainty ? ` ± ${num(g.uncertainty, 3)}` : '';
          t.append(el('tr', { title: `${g.method || ''}\n${g.source || ''}\n${g.note || ''}` }, [
            el('th', { text: k.replace(/_A$|_deg$/, '').replace(/_/g, ' ') }),
            el('td', {}, [el('span', { text: `${num(g.value, 3)}${unc} ${g.unit}` }), ' ',
              originTag('reference')])]));
        }
        box.append(el('div.hint', { text: 'Published geometry, for comparison only:' }));
        box.append(t);
      }
      if (r.reference) box.append(el('div.readout', { text: r.reference }));
      container.append(box);
    }
  }

  container.append(el('div.section-title', {}, [el('span', { text: 'Build' })]));
  const orient = el('input', { type: 'text', id: 'mat-miller', value:
    (s.orientations[0] || [0, 0, 1]).join(''), style: { width: '54px' } });
  const nx = el('input.num.mono', { type: 'number', id: 'mat-nx', value: 4, min: 1, max: 40 });
  const ny = el('input.num.mono', { type: 'number', id: 'mat-ny', value: 4, min: 1, max: 40 });
  const nz = el('input.num.mono', { type: 'number', id: 'mat-nz', value: 4, min: 1, max: 40 });
  const vac = el('input.num.mono', { type: 'number', id: 'mat-vac', value: 14, min: 0, max: 60 });
  const fix = el('input.num.mono', { type: 'number', id: 'mat-fix', value: 0, min: 0, max: 20 });
  const rec = el('select', { id: 'mat-recon' }, [el('option', { value: '' }, 'none (ideal truncation)')]);
  for (const r of d.reconstructions) {
    rec.append(el('option', { value: r.supported ? r.id : '',
      disabled: !r.supported },
    `${r.id} (${r.orientation.join('')})${r.supported ? '' : ' - not implemented'}`));
  }
  const recNote = el('div.hint', { text: '' });
  const syncNote = () => {
    const chosen = d.reconstructions.find((r) => r.id === rec.value);
    recNote.textContent = chosen
      ? `Applied to the upper face, then relaxed with the recommended model. ${chosen.note || ''}`
      : 'Ideal bulk truncation: no reconstruction applied.';
    if (chosen) orient.value = chosen.orientation.join('');
  };
  rec.addEventListener('change', syncNote);
  const g = el('div.field-grid');
  g.append(el('label', { text: '(hkl)' }), orient,
    el('label', { text: 'repeats' }), el('div', { style: { display: 'flex', gap: '3px' } },
      [nx, ny, nz]),
    el('label', { text: 'vacuum / Å' }), vac,
    el('label', { text: 'fixed layers' }), fix,
    el('label', { text: 'reconstruction' }), rec);
  container.append(g);
  container.append(recNote);
  syncNote();
  container.append(el('div.field-row', {}, [
    el('button.tool.primary', { text: 'Build surface', onclick: () => actions.buildSurface({
      material: s.id,
      miller: orient.value.split('').map(Number),
      size: [Number(nx.value), Number(ny.value), Number(nz.value)],
      vacuum_A: Number(vac.value),
      fix_bottom_layers: Number(fix.value),
      reconstruction: rec.value || null,
    }) }),
    el('button.tool', { text: 'Use for wafer', onclick: () => actions.waferDialog(s.id) })]));
  container.append(el('div.note', { text: d.provenance_note || '' }));
  if (d.references.length) {
    container.append(el('div.section-title', {}, [el('span', { text: 'References' })]));
    const ul = el('ul', { style: { margin: '4px 10px', paddingLeft: '16px',
      fontSize: 'var(--fs-sm)', lineHeight: '1.5' } });
    for (const r of d.references) ul.append(el('li', { text: r }));
    container.append(ul);
  }
}

export function renderSelectionPanel(container, table, actions) {
  clear(container);
  const mk = (label, fn) => el('button.tool.sm', { text: label, onclick: fn });
  container.append(
    el('label', { text: 'by element' }),
    el('div', { style: { display: 'flex', gap: '3px' } }, [
      el('input', { type: 'text', id: 'sel-element', value: 'Si', style: { width: '54px' } }),
      mk('Select', () => actions.select('element',
        { elements: $('#sel-element').value.split(/[,\s]+/).filter(Boolean) }))]),
    el('label', { text: 'by role' }),
    el('div', { style: { display: 'flex', gap: '3px' } }, [
      el('select', { id: 'sel-role' }, ['surface', 'dimer', 'subsurface', 'bulk', 'dopant',
        'adatom', 'interstitial', 'adsorbate', 'vacancy-neighbour']
        .map((r) => el('option', { value: r }, r))),
      mk('Select', () => actions.select('role', { roles: [$('#sel-role').value] }))]),
    el('label', { text: 'radius / Å' }),
    el('div', { style: { display: 'flex', gap: '3px' } }, [
      el('input.num.mono', { type: 'number', id: 'sel-radius', value: 5, step: 0.5, min: 0.1 }),
      mk('Neighbours', () => actions.selectNeighbours(parseFloat($('#sel-radius').value)))]),
    el('div.span2', {}, el('div', { style: { display: 'flex', gap: '3px' } }, [
      mk('All', () => actions.select('all')),
      mk('None', () => actions.select('none')),
      mk('Invert', () => actions.select('invert'))])));

  const tbody = table.querySelector('tbody');
  clear(tbody);
  const render = state.render;
  const ids = state.server ? state.server.selection.ids : [];
  if (!render) return;
  const index = new Map(render.ids.map((v, i) => [v, i]));
  const symbols = {};
  for (const [sym, z] of Object.entries(render.elements)) symbols[z] = sym;
  for (const id of ids.slice(0, 400)) {
    const i = index.get(id);
    if (i === undefined) continue;
    tbody.append(el('tr', { onclick: () => actions.inspect(id) }, [
      el('td', { text: `#${id}` }),
      el('td', { text: symbols[render.numbers[i]] }),
      el('td', { text: num(render.positions[i * 3], 4) }),
      el('td', { text: num(render.positions[i * 3 + 1], 4) }),
      el('td', { text: num(render.positions[i * 3 + 2], 4) }),
      el('td', { text: render.roles[i] })]));
  }
}
