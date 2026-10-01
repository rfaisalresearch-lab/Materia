/**
 * Ground-state DFT experiment panel: physical system, electronic state,
 * numerical accuracy, requested observables, the convergence laboratory and
 * the last run with its provenance.
 * Every control maps to one variable of the experiment specification; the
 * service checks the whole specification as it is edited and every refusal is
 * shown beside the control it concerns.
 * @module dft-panel
 */

import { state } from './state.js';
import { $, clear, el, num, originTag, tierTag } from './util.js';

const SVG = 'http://www.w3.org/2000/svg';

const OBSERVABLE_LABELS = {
  energy: 'total energy', forces: 'forces', stress: 'stress tensor',
  density: 'electron density', spin_density: 'spin density',
  electrostatic_potential: 'electrostatic potential', eigenvalues: 'eigenvalues',
  occupations: 'occupations', fermi_level: 'Fermi level', magnetic_moment: 'magnetic moment',
};

const STUDY_DEFAULTS = {
  grid_spacing_A: '0.24, 0.20, 0.16',
  cutoff_eV: '300, 400, 500',
  kpoint_density_A: '10, 15, 20, 25',
  vacuum_A: '4, 5, 6',
  supercell: '1, 2',
  n_bands: '',
  smearing_eV: '0.2, 0.1, 0.05',
};

const TOLERANCE_DEFAULTS = {
  energy_per_atom_eV: 0.001, energy_eV: 0.001, max_force_eV_A: 0.01,
  fermi_level_eV: 0.01, magnetic_moment_muB: 0.01, pressure_GPa: 0.5,
};

function field(spec, name) {
  for (const section of spec.sections) {
    for (const f of section.fields) if (f.name === name) return f;
  }
  return null;
}

function problemsFor(check, name) {
  return ((check && check.by_field) || {})[name] || [];
}

function explain(spec, name) {
  const f = field(spec, name);
  return f ? f.explanation : '';
}

function label(spec, name) {
  const f = field(spec, name);
  if (!f) return name;
  return f.unit && !['', 'SHA-256'].includes(f.unit) ? `${f.label} / ${f.unit}` : f.label;
}

function numberInput(id, value, attrs = {}) {
  return el('input.num.mono', { type: 'number', id, value: value === null || value === undefined ? '' : value,
    'aria-label': attrs.aria || id, ...attrs });
}

function group(title, open, children, key = title) {
  const remembered = (state.dftOpen || {})[key];
  const box = el('details.dft-group', { open: (remembered ?? open) ? true : null });
  box.append(el('summary', { text: title }));
  for (const c of children) if (c) box.append(c);
  box.addEventListener('toggle', () => {
    state.dftOpen = { ...(state.dftOpen || {}), [key]: box.open };
  });
  return box;
}

function row(grid, spec, check, name, control, extra) {
  grid.append(el('label', { text: label(spec, name), for: control.id || null,
    title: explain(spec, name) }), control);
  const hint = explain(spec, name);
  if (hint) grid.append(el('div.hint.span2', { text: hint }));
  for (const p of problemsFor(check, name)) {
    grid.append(el('div.note.blocked.span2', { text: p, dataset: { field: name } }));
  }
  if (extra) grid.append(extra);
}

/** Read every control into the variables sent to the service. */
export function dftVariables() {
  const root = $('#dft-section');
  if (!root || !$('#dft-xc')) return (state.dftVars || null);
  const value = (id) => ($(id) ? $(id).value : '');
  const number = (id) => (value(id) === '' ? null : Number(value(id)));
  const vars = {};
  vars.xc = value('#dft-xc');
  vars.representation = value('#dft-mode');
  if (vars.representation === 'pw') vars.cutoff_eV = number('#dft-cutoff');
  else vars.grid_spacing_A = number('#dft-h');
  if (vars.representation === 'lcao') vars.basis = value('#dft-basis');
  vars.kpoints = ['#dft-kx', '#dft-ky', '#dft-kz'].map((id) => Number(value(id) || 1));
  vars.kpoints_gamma_centered = $('#dft-gamma') ? $('#dft-gamma').checked : true;
  vars.charge_e = number('#dft-charge') ?? 0;
  if ($('#dft-background')) vars.charged_periodic_policy = value('#dft-background');
  vars.poisson = value('#dft-poisson');
  if ($('#dft-field-x')) {
    vars.external_field_V_per_A = ['#dft-field-x', '#dft-field-y', '#dft-field-z']
      .map((id) => Number(value(id) || 0));
  }
  vars.spin_polarized = $('#dft-spin').checked;
  const nAtoms = (state.dftStatus && state.dftStatus.structure) ? state.dftStatus.structure.n_atoms : 0;
  const total = value('#dft-moment');
  if (!vars.spin_polarized) {
    vars.initial_magnetic_moments_muB = new Array(nAtoms).fill(0);
  } else if (total !== '') {
    vars.initial_magnetic_moments_muB = new Array(nAtoms).fill(Number(total) / Math.max(1, nAtoms));
  }
  vars.occupations = value('#dft-occupations');
  vars.smearing_eV = vars.occupations === 'fixed' ? 0 : (number('#dft-smearing') ?? 0.1);
  const bands = number('#dft-bands');
  vars.n_bands = bands === null ? null : Math.round(bands);
  vars.energy_tol_eV_per_electron = number('#dft-etol');
  vars.density_tol_electrons_per_electron = number('#dft-dtol');
  vars.eigenstates_tol_eV2_per_electron = number('#dft-eigtol');
  vars.forces_tol_eV_A = number('#dft-ftol');
  vars.max_scf_iterations = Math.round(number('#dft-maxiter') || 333);
  vars.observables = Array.from(root.querySelectorAll('input[data-observable]'))
    .filter((box) => box.checked).map((box) => box.dataset.observable);
  return vars;
}

const VOIGT = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy'];

/** Read the relaxation controls into the relaxation variables. */
export function dftRelaxVariables() {
  const value = (id) => ($(id) ? $(id).value : '');
  const number = (id) => (value(id) === '' ? null : Number(value(id)));
  const kept = state.dftRelaxVars || {};
  if (!$('#dft-relax-mode')) return { ...kept };
  const vars = {
    mode: value('#dft-relax-mode'),
    optimizer: value('#dft-relax-optimizer'),
    fmax_eV_A: number('#dft-relax-fmax'),
    max_steps: Math.round(number('#dft-relax-steps') || 100),
    maxstep_A: number('#dft-relax-maxstep'),
    symmetry: value('#dft-relax-symmetry'),
  };
  if (vars.mode === 'variable-cell') {
    vars.stress_tol_eV_A3 = number('#dft-relax-stress-tol');
    vars.target_pressure_GPa = number('#dft-relax-pressure') ?? 0;
    vars.hydrostatic_strain = $('#dft-relax-hydrostatic') ? $('#dft-relax-hydrostatic').checked : false;
    vars.cell_mask = VOIGT.map((c) => ($(`#dft-relax-mask-${c}`) ? $(`#dft-relax-mask-${c}`).checked : true));
  }
  return vars;
}

/** Read the DOS controls into the DOS variables and the geometry source. */
export function dftDosRequest() {
  const value = (id) => ($(id) ? $(id).value : '');
  const number = (id) => (value(id) === '' ? null : Number(value(id)));
  const kept = state.dftDosVars || {};
  if (!$('#dft-dos-emin')) return { variables: { ...kept }, source: state.dftDosSource || null };
  const vars = {
    energy_reference: value('#dft-dos-reference'),
    energy_min_eV: number('#dft-dos-emin'),
    energy_max_eV: number('#dft-dos-emax'),
    energy_step_eV: number('#dft-dos-step'),
    broadening: value('#dft-dos-broadening'),
    spin_channels: value('#dft-dos-spin'),
    dos_kpoints: ['#dft-dos-kx', '#dft-dos-ky', '#dft-dos-kz'].map((id) => Math.round(Number(value(id) || 1))),
    n_bands: Math.round(number('#dft-dos-bands') || 1),
  };
  vars.width_eV = vars.broadening === 'gaussian' ? number('#dft-dos-width') : null;
  const boxes = Array.from(document.querySelectorAll('input[data-dos-projection]'));
  if (boxes.length) {
    vars.projections = boxes.filter((b) => b.checked).map((b) => JSON.parse(b.dataset.dosProjection));
  }
  const chosen = value('#dft-dos-source');
  const source = chosen && chosen !== 'structure' ? JSON.parse(chosen) : null;
  return { variables: vars, source };
}

export function dftStudyRequest() {
  const values = ($('#dft-study-values') ? $('#dft-study-values').value : '')
    .split(',').map((v) => v.trim()).filter((v) => v !== '').map(Number);
  return {
    parameter: $('#dft-study-parameter').value,
    values,
    observable: $('#dft-study-observable').value,
    tolerance: Number($('#dft-study-tolerance').value),
  };
}

export function renderDft(body, actions) {
  clear(body);
  const st = state.dftStatus;
  body.append(el('div.section-title', {}, [
    el('span', { text: 'First principles: ground-state DFT (GPAW)' }),
    tierTag('tier3-external-first-principles')]));
  if (!st) {
    body.append(el('div.field-row', {}, [
      el('button.tool.sm', { text: 'Check for GPAW', onclick: () => actions.dftRefresh(true) })]));
    return;
  }
  const env = st.environment;
  const grid = el('table.pgrid');
  const add = (k, v, cls) => grid.append(el('tr', {}, [el('th', { text: k }),
    el('td', cls ? { class: cls } : {}, v === null || v === undefined ? '-' : String(v))]));
  add('code', env.code_available ? `GPAW ${env.gpaw_version}` : 'not found',
    env.code_available ? '' : 'blocked');
  add('PAW datasets', env.datasets_available
    ? `${env.dataset_file_count} files, ${env.dataset_elements.length} elements`
    : 'not found', env.datasets_available ? '' : 'blocked');
  add('interpreter', env.interpreter || '-');
  add('found via', env.source || '-');
  add('ASE', env.ase_version || '-');
  add('parallel', env.mpi ? `MPI, ${env.mpi_world_size} ranks` : 'serial');
  body.append(grid);
  if (!env.operational) {
    body.append(el('div.note.blocked', { id: 'dft-unavailable', text: env.blocking_reason }));
    body.append(el('div.readout', { text: env.install_hint }));
    body.append(el('div.field-row', {}, [
      el('button.tool.sm', { text: 'Re-check', onclick: () => actions.dftRefresh(true) })]));
    return;
  }
  if (!st.structure) {
    body.append(el('p.empty', { text: 'No structure. Build or import one to set up a ground-state experiment.' }));
    renderRuns(body, actions);
    return;
  }
  const check = state.dftCheck || { ...st.structure.report, spec: st.structure.spec };
  const spec = check.spec || st.structure.spec;
  const settings = spec.settings;
  const report = check;
  const options = (report.options || st.structure.report.options);
  const boundary = report.boundary || st.structure.report.boundary;
  const electrons = report.electrons || st.structure.report.electrons;

  const system = el('div.field-grid');
  const b = el('table.pgrid');
  const brow = (k, v, title) => b.append(el('tr', {}, [el('th', { text: k }),
    el('td', { text: v === null || v === undefined ? '-' : String(v), title: title || null })]));
  brow('structure', `${st.structure.key || 'detached'}, ${st.structure.formula}, ${st.structure.n_atoms} atoms`);
  brow('boundary class', boundary.boundary, boundary.note);
  brow('periodic axes', boundary.periodic_axes.join(', ') || 'none');
  brow('open axes', boundary.open_axes.join(', ') || 'none');
  if (boundary.min_vacuum_A !== null && boundary.min_vacuum_A !== undefined) {
    brow('smallest vacuum', `${num(boundary.min_vacuum_A, 3)} Å per side`);
  }
  brow('electrons', `${num(electrons.total, 6)} total${electrons.valence !== null ? `, ${num(electrons.valence, 6)} valence` : ''}${electrons.odd ? ', odd' : ''}`);
  const isotopes = (field(spec, 'mass_numbers') || {}).value || [];
  const set = isotopes.filter((a) => a > 0).length;
  brow('isotopes', set ? `${set} atom(s) with a chosen isotope` : 'natural abundance');
  system.append(el('div.span2', {}, b));
  system.append(el('div.note.info.span2', { id: 'dft-isotope-note', text:
    'Isotopes set the nuclear masses. The Born-Oppenheimer electronic ground state does not '
    + 'depend on nuclear mass, so changing an isotope changes no value this calculation '
    + 'produces; it matters for vibrations and dynamics, and a result stays current across it.' }));
  for (const p of problemsFor(check, 'boundary')) {
    system.append(el('div.note.blocked.span2', { text: p, dataset: { field: 'boundary' } }));
  }
  row(system, spec, check, 'charge_e', numberInput('dft-charge', settings.charge_e, { step: 1, aria: 'net charge' }));
  if (boundary.boundary === 'bulk') {
    const background = el('select', { id: 'dft-background', 'aria-label': 'charged cell treatment' },
      options.charged_periodic_policy.map((p) => el('option', { value: p }, p)));
    background.value = settings.charged_periodic_policy;
    row(system, spec, check, 'charged_periodic_policy', background);
  }
  const poisson = el('select', { id: 'dft-poisson', 'aria-label': 'electrostatic boundary' },
    options.poisson.map((p) => el('option', { value: p }, p)));
  poisson.value = settings.poisson;
  row(system, spec, check, 'poisson', poisson);
  let fieldBox = null;
  if (options.field_allowed) {
    const f = settings.external_field_V_per_A || [0, 0, 0];
    const fields = el('div', { style: { display: 'flex', gap: '3px' } }, [
      numberInput('dft-field-x', f[0], { step: 0.01, aria: 'field x' }),
      numberInput('dft-field-y', f[1], { step: 0.01, aria: 'field y' }),
      numberInput('dft-field-z', f[2], { step: 0.01, aria: 'field z' })]);
    fieldBox = el('div.field-grid');
    row(fieldBox, spec, check, 'external_field_V_per_A', fields);
  } else {
    system.append(el('div.hint.span2', { text: settings.representation === 'pw'
      ? 'No external field is offered: plane waves are periodic in every direction, and a '
        + 'uniform field needs an open one.'
      : `No external field is offered: a uniform field needs an open direction, and a ${boundary.boundary} cell has none.` }));
  }

  const electronic = el('div.field-grid');
  const spin = el('input', { type: 'checkbox', id: 'dft-spin', checked: settings.spin_polarized ? true : null,
    'aria-label': 'spin polarisation' });
  row(electronic, spec, check, 'spin_polarized', spin);
  const moments = settings.initial_magnetic_moments_muB || [];
  const totalMoment = moments.reduce((a, m) => a + m, 0);
  row(electronic, spec, check, 'initial_magnetic_moments_muB',
    numberInput('dft-moment', '', { step: 0.5, placeholder: `as set: total ${num(totalMoment, 4)}`,
      disabled: settings.spin_polarized ? null : true, aria: 'total initial moment spread over all atoms' }),
    el('div.hint.span2', { text: 'Leave empty to keep the moments set on the atoms; a value is '
      + 'spread evenly over every atom as a starting guess. Set per-atom moments with Edit > Spin.' }));
  const occupations = el('select', { id: 'dft-occupations', 'aria-label': 'occupations' },
    options.occupations.map((o) => el('option', { value: o }, o)));
  occupations.value = settings.occupations;
  row(electronic, spec, check, 'occupations', occupations);
  row(electronic, spec, check, 'smearing_eV', numberInput('dft-smearing', settings.smearing_eV,
    { step: 0.01, min: 0, disabled: settings.occupations === 'fixed' ? true : null, aria: 'smearing width' }));
  const advancedElectronic = el('div.field-grid');
  row(advancedElectronic, spec, check, 'n_bands', numberInput('dft-bands', settings.n_bands,
    { step: 1, min: 1, placeholder: 'GPAW default', aria: 'number of bands' }));

  const numerical = el('div.field-grid');
  const xc = el('select', { id: 'dft-xc', 'aria-label': 'functional' },
    (options.functionals.length ? options.functionals : [settings.xc]).map((f) => el('option', { value: f }, f)));
  xc.value = settings.xc;
  row(numerical, spec, check, 'xc', xc,
    el('div.hint.span2', { text: (options.functional_notes || {})[settings.xc] || '' }));
  const mode = el('select', { id: 'dft-mode', 'aria-label': 'representation' },
    options.representations.map((m) => el('option', { value: m }, m)));
  mode.value = settings.representation;
  row(numerical, spec, check, 'representation', mode,
    el('div.hint.span2', { text: (options.representation_notes || {})[settings.representation] || '' }));
  if (settings.representation === 'pw') {
    row(numerical, spec, check, 'cutoff_eV', numberInput('dft-cutoff', settings.cutoff_eV,
      { step: 50, min: 100, aria: 'plane-wave cutoff' }));
  } else {
    row(numerical, spec, check, 'grid_spacing_A', numberInput('dft-h', settings.grid_spacing_A,
      { step: 0.01, min: 0.05, max: 0.35, aria: 'grid spacing' }));
  }
  if (settings.representation === 'lcao') {
    const basis = el('select', { id: 'dft-basis', 'aria-label': 'LCAO basis' },
      (options.bases.length ? options.bases : [settings.basis || 'dzp']).map((n) => el('option', { value: n }, n)));
    basis.value = settings.basis || 'dzp';
    row(numerical, spec, check, 'basis', basis);
  }
  const k = settings.kpoints;
  const pbc = boundary.pbc;
  const kbox = el('div', { style: { display: 'flex', gap: '3px' } }, ['x', 'y', 'z'].map((axis, i) =>
    numberInput(`dft-k${axis}`, k[i], { step: 1, min: 1, max: 32, disabled: pbc[i] ? null : true,
      aria: `k-points along ${'abc'[i]}` })));
  row(numerical, spec, check, 'kpoints', kbox);
  row(numerical, spec, check, 'energy_tol_eV_per_electron', numberInput('dft-etol',
    settings.energy_tol_eV_per_electron, { step: 0.0001, min: 0, aria: 'energy tolerance' }));
  row(numerical, spec, check, 'max_scf_iterations', numberInput('dft-maxiter',
    settings.max_scf_iterations, { step: 10, min: 1, max: 2000, aria: 'SCF iteration limit' }));
  const advancedNumerical = el('div.field-grid');
  row(advancedNumerical, spec, check, 'kpoints_gamma_centered', el('input', { type: 'checkbox',
    id: 'dft-gamma', checked: settings.kpoints_gamma_centered ? true : null, 'aria-label': 'Gamma-centred grid' }));
  row(advancedNumerical, spec, check, 'density_tol_electrons_per_electron', numberInput('dft-dtol',
    settings.density_tol_electrons_per_electron, { step: 0.00001, min: 0, aria: 'density tolerance' }));
  row(advancedNumerical, spec, check, 'eigenstates_tol_eV2_per_electron', numberInput('dft-eigtol',
    settings.eigenstates_tol_eV2_per_electron, { step: 1e-9, min: 0, aria: 'eigenstate tolerance' }));
  row(advancedNumerical, spec, check, 'forces_tol_eV_A', numberInput('dft-ftol', settings.forces_tol_eV_A,
    { step: 0.001, min: 0, placeholder: 'none', aria: 'force tolerance' }));

  const observables = el('div.field-grid');
  const chosen = new Set(settings.observables);
  for (const name of options.observables) {
    const box = el('input', { type: 'checkbox', id: `dft-obs-${name}`, dataset: { observable: name },
      checked: chosen.has(name) ? true : null, disabled: name === 'energy' ? true : null,
      'aria-label': OBSERVABLE_LABELS[name] || name });
    observables.append(el('label', { text: OBSERVABLE_LABELS[name] || name, for: `dft-obs-${name}`,
      title: (options.observable_notes || {})[name] || '' }), box);
  }
  observables.append(el('div.hint.span2', { text:
    'Always recorded: charge and electron accounting, the SCF history and the convergence '
    + 'record. Grids are stored in chunks in the project file.' }));
  for (const p of problemsFor(check, 'observables')) {
    observables.append(el('div.note.blocked.span2', { text: p, dataset: { field: 'observables' } }));
  }

  body.append(group('Physical system', true, [system, fieldBox && group('External field', false, [fieldBox])]));
  body.append(group('Electronic state', true, [electronic, group('Advanced', false, [advancedElectronic], 'electronic-advanced')]));
  body.append(group('Numerical accuracy', true, [numerical, group('Advanced', false, [advancedNumerical], 'numerical-advanced')]));
  body.append(group('Requested observables', false, [observables]));

  for (const input of body.querySelectorAll('input, select')) {
    input.addEventListener('change', () => actions.dftChanged());
  }

  const shown = new Set();
  for (const texts of Object.values(report.by_field || {})) for (const t of texts) shown.add(t);
  const general = (report.blocking || []).filter((t) => !shown.has(t));
  const runBox = el('div', { id: 'dft-run-box' });
  for (const text of general) runBox.append(el('div.note.blocked', { text }));
  for (const text of report.warnings || []) runBox.append(el('div.note.warn', { text }));
  const reuse = el('input', { type: 'checkbox', id: 'dft-reuse', checked: state.dftReuse ? true : null,
    'aria-label': 'keep and reuse compatible restart data' });
  reuse.addEventListener('change', () => { state.dftReuse = reuse.checked; });
  runBox.append(el('label', { style: { display: 'flex', alignItems: 'center', gap: '6px',
    padding: '4px 8px', fontSize: 'var(--fs-sm)' } }, [reuse,
    'keep converged wavefunctions and reuse them when a later run has the same atoms, grid, '
    + 'k-points, electron count and spin setup']));
  runBox.append(el('div.field-row', {}, [
    el('button.tool.primary', { id: 'dft-run', text: 'Run ground state',
      disabled: report.ok ? null : true, onclick: () => actions.dftRun(dftVariables()) }),
    el('button.tool', { text: 'Check', onclick: () => actions.dftChanged() })]));
  runBox.append(el('div.hint', { text:
    'Runs in the background on a frozen copy of the geometry and never changes the structure. '
    + 'A run that does not converge, fails or is cancelled keeps no value. Cancel from the job log.' }));
  runBox.append(el('div.hint.mono', { text: `specification ${spec.short_digest}, version ${spec.version}` }));
  body.append(runBox);

  body.append(renderRelaxControls(st, settings, boundary, actions));
  body.append(renderDosControls(st, actions));
  body.append(renderBandsControls(st, actions));
  body.append(renderEosControls(st, actions));
  body.append(renderLdosControls(st, actions));
  body.append(renderStudyControls(st, settings, boundary, actions));
  renderStudyResult(body, state.dftStudy);
  renderRuns(body, actions);
  renderResult(body, state.dftResult, actions);
  renderRelaxRuns(body, actions);
  renderRelaxResult(body, state.dftRelaxResult, actions);
  renderDosRuns(body, actions);
  renderDosResult(body, state.dftDosResult, actions);
  renderBandsRuns(body, actions);
  renderBandsResult(body, state.dftBandsResult, actions);
  renderEosRuns(body, actions);
  renderEosResult(body, state.dftEosResult, actions);
  renderLdosRuns(body, actions);
  renderLdosResult(body, state.dftLdosResult, actions);
}

function dosField(name) {
  return ((state.dftStatus && state.dftStatus.dos_fields) || []).find((f) => f.name === name) || null;
}

function dosRow(grid, check, name, control) {
  const info = dosField(name);
  const text = info ? (info.unit ? `${info.label} / ${info.unit}` : info.label) : name;
  grid.append(el('label', { text, for: control.id || null, title: info ? info.explanation : '' }), control);
  if (info && info.explanation) grid.append(el('div.hint.span2', { text: info.explanation }));
  for (const p of problemsFor(check, name)) {
    grid.append(el('div.note.blocked.span2', { text: p, dataset: { field: name } }));
  }
}

function renderDosControls(st, actions) {
  const check = state.dftDosCheck;
  const spec = check && check.spec ? check.spec : null;
  const settings = spec ? spec.settings : (state.dftDosVars || {});
  const options = (check && check.options) || {};
  const grid = el('div.field-grid');
  const sources = st.dos_sources || [];
  const source = el('select', { id: 'dft-dos-source', 'aria-label': 'geometry source' },
    [el('option', { value: 'structure' }, 'active structure, settings above')]
      .concat(sources.map((x) => el('option', { value: JSON.stringify({ kind: x.kind, run_id: x.run_id }) }, x.label))));
  const kept = state.dftDosSource;
  if (kept) source.value = JSON.stringify({ kind: kept.kind, run_id: kept.run_id });
  if (!source.value) source.value = 'structure';
  dosRow(grid, check, 'source', source);
  const reference = el('select', { id: 'dft-dos-reference', 'aria-label': 'energy zero' },
    (options.references || ['fermi-level']).map((r) => el('option', { value: r }, r)));
  reference.value = settings.energy_reference || 'fermi-level';
  dosRow(grid, check, 'energy_reference', reference);
  dosRow(grid, check, 'energy_min_eV', numberInput('dft-dos-emin', settings.energy_min_eV ?? -10,
    { step: 0.5, aria: 'window start' }));
  dosRow(grid, check, 'energy_max_eV', numberInput('dft-dos-emax', settings.energy_max_eV ?? 5,
    { step: 0.5, aria: 'window end' }));
  dosRow(grid, check, 'energy_step_eV', numberInput('dft-dos-step', settings.energy_step_eV ?? 0.01,
    { step: 0.005, min: 0.001, aria: 'grid spacing' }));
  const broadening = el('select', { id: 'dft-dos-broadening', 'aria-label': 'broadening' },
    (options.broadenings || ['gaussian']).map((b) => el('option', { value: b }, b)));
  broadening.value = settings.broadening || 'gaussian';
  dosRow(grid, check, 'broadening', broadening);
  if (broadening.value === 'gaussian') {
    dosRow(grid, check, 'width_eV', numberInput('dft-dos-width', settings.width_eV ?? 0.1,
      { step: 0.01, min: 0.001, aria: 'Gaussian width' }));
  }
  const spin = el('select', { id: 'dft-dos-spin', 'aria-label': 'spin channels' },
    (options.spin_channels || ['total']).map((c) => el('option', { value: c }, c)));
  spin.value = settings.spin_channels || 'total';
  dosRow(grid, check, 'spin_channels', spin);
  const k = settings.dos_kpoints || [1, 1, 1];
  const pbc = (check && check.boundary && check.boundary.pbc) || [true, true, true];
  dosRow(grid, check, 'dos_kpoints', el('div', { style: { display: 'flex', gap: '3px' } },
    ['x', 'y', 'z'].map((axis, i) => numberInput(`dft-dos-k${axis}`, k[i], { step: 1, min: 1, max: 32,
      disabled: pbc[i] ? null : true, aria: `DOS k-points along ${'abc'[i]}` }))));
  dosRow(grid, check, 'n_bands', numberInput('dft-dos-bands', settings.n_bands ?? '',
    { step: 1, min: 1, aria: 'bands in the DOS step' }));
  const projections = el('div.field-grid');
  const offered = spec ? (settings.projections || []) : [];
  const channels = options.projection_channels || {};
  projections.append(el('div.hint.span2', { text: `Projectable channels: ${Object.entries(channels)
    .map(([el_, c]) => `${el_} ${c.join(', ') || 'none'}`).join('; ') || 'unknown until checked'}` }));
  const all = [...(state.dftDosProjections || [])];
  for (const p of offered) if (!all.some((q) => q.label === p.label)) all.push(p);
  state.dftDosProjections = all;
  const chosen = new Set(offered.map((p) => p.label));
  all.forEach((p, i) => {
    projections.append(el('label', { text: `${p.label} (${p.atom_ids.length} atom${p.atom_ids.length === 1 ? '' : 's'})`,
      for: `dft-dos-proj-${i}` }),
    el('input', { type: 'checkbox', id: `dft-dos-proj-${i}`, checked: chosen.has(p.label) ? true : null,
      'aria-label': `project on ${p.label}`, dataset: { dosProjection: JSON.stringify(p) } }));
  });
  for (const p of problemsFor(check, 'projections')) {
    projections.append(el('div.note.blocked.span2', { text: p, dataset: { field: 'projections' } }));
  }
  const box = el('div', { id: 'dft-dos-box' });
  if (check) {
    const shown = new Set();
    for (const texts of Object.values(check.by_field || {})) for (const t of texts) shown.add(t);
    const known = new Set(((state.dftStatus && state.dftStatus.dos_fields) || []).map((f) => f.name));
    for (const [name, texts] of Object.entries(check.by_field || {})) {
      if (known.has(name)) continue;
      for (const t of texts) box.append(el('div.note.blocked', { text: `${name}: ${t}` }));
    }
    for (const t of (check.blocking || []).filter((t) => !shown.has(t))) box.append(el('div.note.blocked', { text: t }));
    for (const t of check.warnings || []) box.append(el('div.note.warn', { text: t }));
  }
  box.append(el('div.field-row', {}, [
    el('button.tool.primary', { id: 'dft-dos-run', text: 'Compute DOS',
      disabled: check && !check.ok ? true : null, onclick: () => actions.dftDos(dftVariables(), dftDosRequest()) }),
    el('button.tool', { id: 'dft-dos-check', text: 'Check DOS',
      onclick: () => actions.dftDosChanged(dftVariables(), dftDosRequest()) })]));
  box.append(el('div.hint', { text: 'Converges the ground state, then a non-self-consistent step on the '
    + 'DOS k-points with every band converged, and evaluates GPAW\'s DOS and projected DOS on the '
    + 'grid. The DOS observes the geometry and never changes it. A run that is cancelled, times '
    + 'out or fails keeps nothing.' }));
  if (spec) box.append(el('div.hint.mono', { text: `DOS specification ${spec.short_digest}, version ${spec.version}, ${spec.npoints} points` }));
  const wrapper = group('Density of states', false, [grid, group('Projections', true, [projections], 'dos-projections'), box], 'dos');
  for (const input of wrapper.querySelectorAll('input, select')) {
    input.addEventListener('change', () => actions.dftDosChanged(dftVariables(), dftDosRequest()));
  }
  return wrapper;
}

function renderDosRuns(body, actions) {
  const runs = (state.dftStatus && state.dftStatus.dos_runs) || [];
  if (!runs.length) return;
  const select = el('select', { id: 'dft-dos-select', 'aria-label': 'stored DOS runs' },
    runs.map((r) => el('option', { value: r.run_id },
      `${r.run_id.slice(-6)} ${r.formula || ''} ${r.xc || ''} ${r.broadening || ''}${r.state === 'stale' ? ', stale' : ''}`)));
  if (state.dftDosResult && state.dftDosResult.run_id) select.value = state.dftDosResult.run_id;
  select.addEventListener('change', () => actions.dftDosShow(select.value));
  const grid = el('div.field-grid');
  grid.append(el('label', { text: `stored DOS (${runs.length})`, for: 'dft-dos-select' }), select);
  body.append(grid);
}

const PROJECTION_DASHES = ['4 3', '1 3', '6 2 1 2', '2 2', '8 3', '3 1'];

function renderDosResult(body, r, actions) {
  if (!r) return;
  body.append(el('div.section-title', {}, [el('span', { text: 'Density of states' }),
    r.ok ? originTag((r.provenance || {}).origin) : el('span.tag', { text: r.status || 'refused' })]));
  if (!r.ok) {
    body.append(el(r.status === 'cancelled' ? 'div.note.warn' : 'div.note.blocked',
      { id: 'dft-dos-reason', text: r.reason || r.error || '' }));
    return;
  }
  if (!r.current) body.append(el('div.note.warn', { id: 'dft-dos-stale', text: r.state_reason }));
  const view = state.dftDosView || {};
  const curves = r.curves || {};
  const spinResolved = Array.isArray(curves.up);
  const channel = el('select', { id: 'dft-dos-channel', 'aria-label': 'spin channel to plot' },
    [el('option', { value: 'total' }, 'total')].concat(spinResolved ? [
      el('option', { value: 'mirror' }, 'up above, down below'),
      el('option', { value: 'up' }, 'spin up'), el('option', { value: 'down' }, 'spin down')] : []));
  channel.value = spinResolved ? (view.channel || 'total') : 'total';
  channel.addEventListener('change', () => actions.dftDosView({ ...view, channel: channel.value }));
  const hidden = new Set(view.hidden || []);
  const toggles = el('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '8px', padding: '2px 8px' } },
    (r.projections || []).map((p, i) => el('label', { style: { display: 'flex', gap: '3px', alignItems: 'center' } }, [
      el('input', { type: 'checkbox', id: `dft-dos-show-${i}`, checked: hidden.has(p.label) ? null : true,
        'aria-label': `show ${p.label}`, onchange: (e) => {
          const next = new Set(hidden);
          if (e.target.checked) next.delete(p.label); else next.add(p.label);
          actions.dftDosView({ ...view, hidden: [...next] });
        } }), p.label])));
  const x = curves.energies_eV || [];
  const pick = (item) => {
    if (channel.value === 'up') return item.up;
    if (channel.value === 'down') return item.down;
    return item.total;
  };
  const series = [];
  if (channel.value === 'mirror') {
    series.push({ name: 'up', points: x.map((e, i) => [e, curves.up[i]]) });
    series.push({ name: 'down', points: x.map((e, i) => [e, -curves.down[i]]) });
  } else {
    series.push({ name: channel.value === 'total' ? 'total' : channel.value, points: x.map((e, i) => [e, pick(curves)[i]]) });
  }
  (r.projections || []).forEach((p, i) => {
    if (hidden.has(p.label)) return;
    const dash = PROJECTION_DASHES[i % PROJECTION_DASHES.length];
    if (channel.value === 'mirror') {
      series.push({ name: `${p.label} up`, dash, points: x.map((e, j) => [e, p.up[j]]) });
      series.push({ name: `${p.label} down`, dash, points: x.map((e, j) => [e, -p.down[j]]) });
    } else {
      series.push({ name: p.label, dash, points: x.map((e, j) => [e, pick(p)[j]]) });
    }
  });
  const ys = series.flatMap((s) => s.points.map(([, y]) => y)).filter((y) => Number.isFinite(y));
  const fermi = r.reference === 'fermi-level' ? 0 : r.fermi_level_eV;
  if (ys.length) {
    series.push({ name: 'Fermi level', dash: '1 2', points: [[fermi, Math.min(...ys)], [fermi, Math.max(...ys)]] });
  }
  const controls = el('div.field-grid');
  controls.append(el('label', { text: 'plot', for: 'dft-dos-channel' }), channel);
  body.append(controls);
  if ((r.projections || []).length) body.append(toggles);
  body.append(el('div', { style: { padding: '4px 8px' } }, linePlot(series, {
    xlabel: r.reference === 'fermi-level' ? 'E - E_F / eV' : 'E / eV',
    ylabel: 'states / eV', id: 'dft-dos-plot', dots: false })));
  if (r.display_stride > 1) {
    body.append(el('div.hint', { text: `Plotted every ${r.display_stride} points of ${r.npoints}; the export and the stored arrays are at full resolution.` }));
  }
  const t = el('table.pgrid', { id: 'dft-dos-table' });
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td', { text: v === null || v === undefined ? '-' : String(v) })]));
  const c = r.checks || {};
  const s = r.settings || {};
  add('state', r.run_state);
  add('structure', `${r.structure_key || 'detached'}, ${r.formula}, ${r.boundary}`);
  add('source', r.source && r.source.run_id ? `${r.source.kind} ${r.source.run_id}` : 'structure');
  add('Fermi level', `${num(r.fermi_level_eV, 6)} eV`);
  add('energy zero', r.reference === 'fermi-level' ? 'Fermi level' : 'absolute (vacuum of the box)');
  add('window', `${num(s.energy_min_eV, 4)} to ${num(s.energy_max_eV, 4)} eV, step ${num(s.energy_step_eV, 4)} eV, ${r.npoints} points`);
  add('broadening', s.broadening === 'gaussian'
    ? `Gaussian, GPAW width ${num(s.width_eV, 4)} eV (standard deviation ${num(s.width_eV / Math.SQRT2, 4)} eV)`
    : 'linear tetrahedron');
  add('k-points', `${(s.dos_kpoints || []).join('x')}, ${r.n_ibz_kpoints} irreducible`);
  add('bands', s.n_bands);
  add('spin channels', `${s.spin_channels}, ${r.n_spins} spin(s)`);
  add('unit', r.unit);
  add('recomputed from eigenvalues', `agrees to ${num(c.recompute_max_relative_error, 3)} of the maximum`);
  if (c.window_integral_exact_states !== undefined) {
    add('window integral', `${num(c.window_integral_states, 7)} states (exact ${num(c.window_integral_exact_states, 7)})`);
  } else {
    add('window integral', `${num(c.window_integral_states, 7)} states`);
  }
  add('electrons below E_F', c.electron_count_difference_e !== undefined
    ? `${num(c.integral_to_fermi_level_e, 7)} (valence ${num(c.expected_valence_electrons, 6)})`
    : 'window does not reach the lowest band');
  if (c.projected_fraction_of_window !== undefined && c.projected_fraction_of_window !== null) {
    add('projected fraction', num(c.projected_fraction_of_window, 4));
  }
  add('bands above window', `${num(c.highest_band_above_window_eV, 4)} eV`);
  if (c.source_energy_difference_eV !== undefined) add('energy against source', `${num(c.source_energy_difference_eV, 3)} eV`);
  add('specification', (r.spec_digest || '').slice(0, 16));
  add('run', r.run_id);
  body.append(t);
  for (const w of r.warnings || []) body.append(el('div.note.warn', { text: w }));
  body.append(el('div.field-row', {}, [
    el('button.tool.sm', { id: 'dft-dos-export', text: 'Export CSV', onclick: () => actions.dftDosExport(r.run_id) }),
    el('button.tool.sm', { id: 'dft-dos-provenance', text: 'Provenance', onclick: () => actions.dftDosProvenance(r) })]));
}

/** The provenance of one DOS, for the inspector dialog. */
export function dosProvenanceView(r) {
  const p = r.provenance || {};
  const box = el('div', { id: 'dft-dos-provenance-view' });
  const t = el('table.pgrid');
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td.txt', { text: v === null || v === undefined ? '-' : String(v) })]));
  add('model', p.model);
  add('fidelity', p.fidelity);
  add('origin', p.origin);
  add('DOS specification (SHA-256)', p.inputs_digest);
  add('source', JSON.stringify(r.source));
  add('state', `${r.run_state}: ${r.state_reason}`);
  add('boundary conditions', p.boundary_conditions);
  add('software pinned', Object.entries(p.software_pinned || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('software reported', Object.entries(p.software_reported || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('datasets', (p.paw_datasets || []).map((d) => `${d.symbol} ${String(d.sha256 || '').slice(0, 12)}`).join('; '));
  box.append(t);
  box.append(el('div.section-title', {}, el('span', { text: 'Approximations' })));
  for (const a of p.approximations || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: a }));
  box.append(el('div.section-title', {}, el('span', { text: 'DOS step sent and used' })));
  box.append(el('pre.readout', { style: { whiteSpace: 'pre-wrap', padding: '4px 8px' },
    text: JSON.stringify({ nscf_sent: p.nscf_parameters, dos_sent: p.dos_settings, dos_used: p.dos_used }, null, 1) }));
  box.append(el('div.section-title', {}, el('span', { text: 'Exact GPAW ground-state input' })));
  box.append(el('pre.readout', { style: { whiteSpace: 'pre-wrap', padding: '4px 8px' },
    text: JSON.stringify(p.gpaw_parameters || {}, null, 1) }));
  return box;
}

function bandsField(name) {
  return ((state.dftStatus && state.dftStatus.bands_fields) || []).find((f) => f.name === name) || null;
}

function bandsRow(grid, check, name, control) {
  const info = bandsField(name);
  const text = info ? (info.unit && info.unit !== 'fractional' ? `${info.label} / ${info.unit}` : info.label) : name;
  grid.append(el('label', { text, for: control.id || null, title: info ? info.explanation : '' }), control);
  if (info && info.explanation) grid.append(el('div.hint.span2', { text: info.explanation }));
  for (const p of problemsFor(check, name)) {
    grid.append(el('div.note.blocked.span2', { text: p, dataset: { field: name } }));
  }
}

/** Symmetry labels as physicists write them: G is the zone centre, Gamma. */
export function kLabel(label) {
  return String(label || '').split('|').map((l) => (l === 'G' ? 'Γ' : l)).join('|');
}

/** Read the band-structure controls into the variables and source sent to the service. */
export function dftBandsRequest() {
  const kept = state.dftBandsVars || {};
  if (!$('#dft-bands-path')) return { variables: { ...kept }, source: state.dftBandsSource || null };
  const value = (id) => ($(id) ? $(id).value : null);
  const number = (id) => (value(id) === '' || value(id) === null ? null : Number(value(id)));
  const vars = {
    energy_reference: value('#dft-bands-reference') || 'fermi-level',
    scf_symmetry: value('#dft-bands-symmetry') || 'preserve',
    path: (value('#dft-bands-path') || '').trim() || 'standard',
    sampling_density_per_invA: number('#dft-bands-density'),
    n_bands: Math.round(number('#dft-bands-bands') || 1),
    extra_bands: Math.round(number('#dft-bands-extra') ?? 4),
  };
  if (vars.sampling_density_per_invA === null) delete vars.sampling_density_per_invA;
  const chosen = value('#dft-bands-source');
  const source = chosen && chosen !== 'structure' ? JSON.parse(chosen) : null;
  return { variables: vars, source };
}

function renderBandsControls(st, actions) {
  const check = state.dftBandsCheck;
  const spec = check && check.spec ? check.spec : null;
  const settings = spec ? spec.settings : (state.dftBandsVars || {});
  const options = (check && check.options) || {};
  const grid = el('div.field-grid');
  const sources = st.dos_sources || [];
  const source = el('select', { id: 'dft-bands-source', 'aria-label': 'geometry source' },
    [el('option', { value: 'structure' }, 'active structure, settings above')]
      .concat(sources.map((x) => el('option', { value: JSON.stringify({ kind: x.kind, run_id: x.run_id }) }, x.label))));
  const kept = state.dftBandsSource;
  if (kept) source.value = JSON.stringify({ kind: kept.kind, run_id: kept.run_id });
  if (!source.value) source.value = 'structure';
  bandsRow(grid, check, 'source', source);
  const reference = el('select', { id: 'dft-bands-reference', 'aria-label': 'energy zero' },
    (options.references || ['fermi-level']).map((r) => el('option', { value: r }, r === 'fermi-level' ? 'Fermi level of the ground state' : r)));
  reference.value = settings.energy_reference || 'fermi-level';
  bandsRow(grid, check, 'energy_reference', reference);
  const symmetry = el('select', { id: 'dft-bands-symmetry', 'aria-label': 'ground-state symmetry' },
    (options.scf_symmetry || ['preserve', 'off']).map((c) => el('option', { value: c }, c)));
  symmetry.value = settings.scf_symmetry || 'preserve';
  bandsRow(grid, check, 'scf_symmetry', symmetry);
  const path = el('input.mono', { type: 'text', id: 'dft-bands-path', value: settings.path || '',
    'aria-label': 'reciprocal-space path', spellcheck: 'false', style: { width: '100%' } });
  bandsRow(grid, check, 'path', path);
  grid.append(el('span'), el('div.field-row', {}, [
    el('button.tool.sm', { id: 'dft-bands-standard', text: 'Standard path',
      title: 'Regenerate the standard path of this lattice from ASE, visibly, as a new specification',
      onclick: () => { path.value = 'standard'; actions.dftBandsChanged(dftVariables(), dftBandsRequest()); } })]));
  bandsRow(grid, check, 'sampling_density_per_invA', numberInput('dft-bands-density',
    settings.sampling_density_per_invA ?? 10, { step: 1, min: 0.5, max: 200, aria: 'sampling density' }));
  bandsRow(grid, check, 'n_bands', numberInput('dft-bands-bands', settings.n_bands ?? '',
    { step: 1, min: 1, aria: 'bands returned' }));
  bandsRow(grid, check, 'extra_bands', numberInput('dft-bands-extra', settings.extra_bands ?? 4,
    { step: 1, min: 0, aria: 'buffer bands' }));
  for (const name of ['special_points', 'segment_intervals', 'kpoints_frac', 'path_origin']) {
    for (const p of problemsFor(check, name)) grid.append(el('div.note.blocked.span2', { text: p, dataset: { field: name } }));
  }
  const points = el('table.pgrid', { id: 'dft-bands-points' });
  points.append(el('tr', {}, [el('th', { text: 'label' }), el('th', { text: 'k1' }), el('th', { text: 'k2' }), el('th', { text: 'k3' })]));
  for (const [labelText, coords] of Object.entries((settings.special_points) || {})) {
    points.append(el('tr', {}, [el('td', { text: kLabel(labelText) })].concat(coords.map((c) => el('td.mono', { text: num(c, 6) })))));
  }
  const origin = spec ? spec.path_origin || {} : {};
  const originText = origin.generator === 'ase'
    ? `Generated once by ASE ${origin.ase_version} for the ${origin.lattice} lattice of this cell (standard path ${origin.path}); stored as exact fractional coordinates.`
    : (origin.generator === 'user' ? 'Special points given by hand for this cell.' : (origin.reason || 'No path yet.'));
  const segments = spec ? (spec.segments || []) : [];
  const layout = el('div.hint', { id: 'dft-bands-layout', text: spec
    ? `${spec.n_kpoints} k-points in ${segments.length} segment${segments.length === 1 ? '' : 's'}: ${segments.map((s) => `${kLabel(s.from)} to ${kLabel(s.to)} ${s.intervals}`).join(', ')}`
    : 'Check to see the k-points.' });
  const box = el('div', { id: 'dft-bands-box' });
  if (check) {
    const shown = new Set();
    for (const texts of Object.values(check.by_field || {})) for (const t of texts) shown.add(t);
    const known = new Set(((state.dftStatus && state.dftStatus.bands_fields) || []).map((f) => f.name));
    for (const [name, texts] of Object.entries(check.by_field || {})) {
      if (known.has(name)) continue;
      for (const t of texts) box.append(el('div.note.blocked', { text: `${name}: ${t}` }));
    }
    for (const t of (check.blocking || []).filter((t) => !shown.has(t))) box.append(el('div.note.blocked', { text: t }));
    for (const t of check.warnings || []) box.append(el('div.note.warn', { text: t }));
  }
  box.append(el('div.field-row', {}, [
    el('button.tool.primary', { id: 'dft-bands-run', text: 'Compute band structure',
      disabled: check && !check.ok ? true : null, onclick: () => actions.dftBands(dftVariables(), dftBandsRequest()) }),
    el('button.tool', { id: 'dft-bands-check', text: 'Check band structure',
      onclick: () => actions.dftBandsChanged(dftVariables(), dftBandsRequest()) })]));
  box.append(el('div.hint', { text: 'Converges the ground state, then computes the Kohn-Sham bands at exactly '
    + 'the listed k-points with symmetry off, every requested band converged. The k-points, labels and '
    + 'path distance GPAW reports are verified before anything is kept. Band edges are read along the '
    + 'path only. The calculation observes the geometry and never changes it; a run that is cancelled, '
    + 'times out, fails or does not verify keeps nothing.' }));
  box.append(el('div.hint', { id: 'dft-bands-character', text: 'Orbital character (fat bands) is not offered: '
    + 'PAW projector weights are not normalised orbital populations, and an approximate version would be misleading.' }));
  if (spec) box.append(el('div.hint.mono', { text: `band-structure specification ${spec.short_digest}, version ${spec.version}` }));
  const wrapper = group('Band structure', false, [grid,
    group('Special points', false, [el('div.hint', { text: originText }), points, layout], 'bands-points'), box], 'bands');
  for (const input of wrapper.querySelectorAll('input, select')) {
    input.addEventListener('change', () => actions.dftBandsChanged(dftVariables(), dftBandsRequest()));
  }
  return wrapper;
}

function renderBandsRuns(body, actions) {
  const runs = (state.dftStatus && state.dftStatus.bands_runs) || [];
  if (!runs.length) return;
  const select = el('select', { id: 'dft-bands-select', 'aria-label': 'stored band structures' },
    runs.map((r) => el('option', { value: r.run_id },
      `${r.run_id.slice(-6)} ${r.formula || ''} ${r.xc || ''} ${r.path || ''}${r.state && r.state !== 'current' ? `, ${r.state}` : ''}`)));
  if (state.dftBandsResult && state.dftBandsResult.run_id) select.value = state.dftBandsResult.run_id;
  select.addEventListener('change', () => actions.dftBandsShow(select.value));
  const grid = el('div.field-grid');
  grid.append(el('label', { text: `stored band structures (${runs.length})`, for: 'dft-bands-select' }), select);
  body.append(grid);
}

/** Visible ranges of the band plot: k from one tick to another, energy between two limits. */
export function bandWindow(r, view = {}) {
  const x = r.distance_invA || [];
  const ticks = r.ticks || [];
  const from = Math.max(0, Math.min(ticks.length - 1, view.from ?? 0));
  const to = Math.max(from, Math.min(ticks.length - 1, view.to ?? ticks.length - 1));
  let x0 = ticks.length ? ticks[from].distance_invA : 0;
  let x1 = ticks.length ? ticks[to].distance_invA : (x[x.length - 1] || 1);
  if (x1 <= x0) { x0 = x[0] || 0; x1 = x[x.length - 1] || 1; }
  const all = (r.bands_relative_eV || []).flat(2).filter((v) => Number.isFinite(v));
  const lo = all.length ? Math.min(...all) : -1;
  const hi = all.length ? Math.max(...all) : 1;
  const e0 = Number.isFinite(view.emin) ? view.emin : lo - 0.02 * (hi - lo);
  const e1 = Number.isFinite(view.emax) && view.emax > e0 ? view.emax : hi + 0.02 * (hi - lo);
  return { x0, x1, e0, e1, from, to };
}

/**
 * The band plot: bands against cumulative path distance, symmetry labels and
 * segment dividers at the special points, a double divider at a break, and
 * the Fermi level at zero. Clicking selects the nearest k-point.
 */
export function bandPlot(r, view = {}, onPick = null, { id = 'dft-bands-plot', w = 420, h = 280 } = {}) {
  const left = 44; const right = 10; const top = 10; const bottom = 30;
  const svg = svgNode('svg', { viewBox: `0 0 ${w} ${h}`, class: 'dft-plot band-plot', role: 'img',
    id, 'aria-label': 'Kohn-Sham bands, E - E_F against distance along the path' });
  const x = r.distance_invA || [];
  if (!x.length) return svg;
  const { x0, x1, e0, e1 } = bandWindow(r, view);
  const px = (v) => left + ((v - x0) / (x1 - x0)) * (w - left - right);
  const py = (v) => top + (1 - (v - e0) / (e1 - e0)) * (h - top - bottom);
  const clip = svgNode('clipPath', { id: `${id}-clip` });
  clip.append(svgNode('rect', { x: left, y: top, width: w - left - right, height: h - top - bottom }));
  const defs = svgNode('defs');
  defs.append(clip);
  svg.append(defs);
  svg.append(svgNode('rect', { x: left, y: top, width: w - left - right, height: h - top - bottom, class: 'frame' }));
  const text = (tx, ty, t, anchor = 'middle', cls = 'axis') => {
    const n = svgNode('text', { x: tx, y: ty, 'text-anchor': anchor, class: cls });
    n.textContent = t; svg.append(n);
  };
  const layer = svgNode('g', { 'clip-path': `url(#${id}-clip)` });
  svg.append(layer);
  const span = e1 - e0;
  const step = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20].find((v) => span / v <= 8) || 50;
  for (let v = Math.ceil(e0 / step) * step; v <= e1 + 1e-9; v += step) {
    const Y = py(v);
    if (Math.abs(v) > 1e-9) layer.append(svgNode('line', { x1: left, x2: w - right, y1: Y, y2: Y, class: 'grid' }));
    text(left - 3, Y + 3, num(Math.abs(v) < 1e-9 ? 0 : v, 3), 'end');
  }
  for (const tick of r.ticks || []) {
    if (tick.distance_invA < x0 - 1e-9 || tick.distance_invA > x1 + 1e-9) continue;
    const X = px(tick.distance_invA);
    layer.append(svgNode('line', { x1: X, x2: X, y1: top, y2: h - bottom, class: tick.break ? 'divider break' : 'divider' }));
    text(X, h - bottom + 11, kLabel(tick.label), 'middle', 'axis klabel');
  }
  const zero = py(0);
  if (zero >= top && zero <= h - bottom) {
    layer.append(svgNode('line', { x1: left, x2: w - right, y1: zero, y2: zero, class: 'fermi' }));
    text(w - right - 2, zero - 3, 'E_F', 'end', 'axis');
  }
  const breaks = new Set(r.breaks || []);
  const spins = (r.bands_relative_eV || []).map((_, s) => s)
    .filter((s) => view.spin === undefined || view.spin === 'both' || (view.spin === 'up' ? s === 0 : s === 1));
  for (const s of spins) {
    for (const band of r.bands_relative_eV[s]) {
      let run = [];
      const flush = () => {
        if (run.length > 1) layer.append(svgNode('polyline', { points: run.join(' '), class: s === 1 ? 'line down' : 'line' }));
        run = [];
      };
      band.forEach((e, k) => {
        if (breaks.has(k)) flush();
        if (x[k] < x0 - 1e-9 || x[k] > x1 + 1e-9) { flush(); return; }
        run.push(`${px(x[k]).toFixed(2)},${py(e).toFixed(2)}`);
      });
      flush();
    }
  }
  if (Number.isInteger(view.k) && x[view.k] >= x0 - 1e-9 && x[view.k] <= x1 + 1e-9) {
    layer.append(svgNode('line', { x1: px(x[view.k]), x2: px(x[view.k]), y1: top, y2: h - bottom, class: 'cursor' }));
  }
  text(left + (w - left - right) / 2, h - 4, 'distance along the path / (1/A)');
  const yl = svgNode('text', { x: 10, y: top + (h - top - bottom) / 2, 'text-anchor': 'middle',
    class: 'axis', transform: `rotate(-90 10 ${top + (h - top - bottom) / 2})` });
  yl.textContent = 'E - E_F / eV'; svg.append(yl);
  if (spins.length === 2) {
    svg.append(svgNode('line', { x1: w - right - 70, x2: w - right - 56, y1: top + 9, y2: top + 9, class: 'line' }));
    text(w - right - 52, top + 12, 'up', 'start');
    svg.append(svgNode('line', { x1: w - right - 34, x2: w - right - 20, y1: top + 9, y2: top + 9, class: 'line down' }));
    text(w - right - 16, top + 12, 'down', 'start');
  }
  if (onPick) {
    svg.addEventListener('click', (event) => {
      const box = svg.getBoundingClientRect();
      if (!box.width) return;
      const sx = ((event.clientX - box.left) / box.width) * w;
      const d = x0 + ((sx - left) / (w - left - right)) * (x1 - x0);
      let best = 0;
      x.forEach((v, k) => { if (Math.abs(v - d) < Math.abs(x[best] - d)) best = k; });
      onPick(best);
    });
  }
  return svg;
}

function renderBandsResult(body, r, actions) {
  if (!r) return;
  body.append(el('div.section-title', {}, [el('span', { text: 'Band structure' }),
    r.ok ? originTag((r.provenance || {}).origin) : el('span.tag', { text: r.status || 'refused' })]));
  if (!r.ok) {
    body.append(el(r.status === 'cancelled' ? 'div.note.warn' : 'div.note.blocked',
      { id: 'dft-bands-reason', text: r.reason || r.error || '' }));
    return;
  }
  if (!r.current) body.append(el('div.note.warn', { id: 'dft-bands-stale', text: r.state_reason }));
  const view = { spin: 'both', ...(state.dftBandsView || {}) };
  const ticks = r.ticks || [];
  const controls = el('div.field-grid');
  if (r.n_spins === 2) {
    const spin = el('select', { id: 'dft-bands-spin-view', 'aria-label': 'spin channel to plot' }, [
      el('option', { value: 'both' }, 'both, down dashed'), el('option', { value: 'up' }, 'spin up'),
      el('option', { value: 'down' }, 'spin down')]);
    spin.value = view.spin;
    spin.addEventListener('change', () => actions.dftBandsView({ ...view, spin: spin.value }));
    controls.append(el('label', { text: 'spin', for: 'dft-bands-spin-view' }), spin);
  }
  const tickOptions = () => ticks.map((t, i) => el('option', { value: String(i) }, `${kLabel(t.label)} (${num(t.distance_invA, 4)})`));
  const from = el('select', { id: 'dft-bands-from', 'aria-label': 'plot from special point' }, tickOptions());
  const to = el('select', { id: 'dft-bands-to', 'aria-label': 'plot to special point' }, tickOptions());
  const win = bandWindow(r, view);
  from.value = String(win.from); to.value = String(win.to);
  from.addEventListener('change', () => actions.dftBandsView({ ...view, from: Number(from.value) }));
  to.addEventListener('change', () => actions.dftBandsView({ ...view, to: Number(to.value) }));
  controls.append(el('label', { text: 'k from', for: 'dft-bands-from' }), from,
    el('label', { text: 'k to', for: 'dft-bands-to' }), to);
  const emin = numberInput('dft-bands-zoom-min', Number(win.e0.toFixed(3)), { step: 0.5, aria: 'lowest energy shown' });
  const emax = numberInput('dft-bands-zoom-max', Number(win.e1.toFixed(3)), { step: 0.5, aria: 'highest energy shown' });
  const applyZoom = () => actions.dftBandsView({ ...view, emin: Number(emin.value), emax: Number(emax.value) });
  emin.addEventListener('change', applyZoom); emax.addEventListener('change', applyZoom);
  controls.append(el('label', { text: 'E - E_F from / eV', for: 'dft-bands-zoom-min' }), emin,
    el('label', { text: 'E - E_F to / eV', for: 'dft-bands-zoom-max' }), emax);
  body.append(controls);
  const edges = r.band_edges || {};
  body.append(el('div.field-row', {}, [
    el('button.tool.sm', { id: 'dft-bands-zoom-gap', text: 'Zoom near E_F',
      onclick: () => actions.dftBandsView({ ...view, emin: -3, emax: 3 }) }),
    el('button.tool.sm', { id: 'dft-bands-zoom-reset', text: 'Full range',
      onclick: () => actions.dftBandsView({ spin: view.spin, k: view.k }) })]));
  body.append(el('div', { style: { padding: '4px 8px' } },
    bandPlot(r, view, (k) => actions.dftBandsView({ ...view, k }))));
  body.append(el('div.field-row', {}, [el('button.tool.sm', { id: 'dft-bands-enlarge', text: 'Enlarge plot',
    onclick: () => actions.dftBandsEnlarge(r, view) })]));
  body.append(el('div.hint', { text: 'Click the plot to list the eigenvalues at the nearest k-point. '
    + 'Lines join bands of equal index; at a crossing that is an ordering, not a character.' }));
  const kIndex = Number.isInteger(view.k) ? view.k : null;
  if (kIndex !== null) {
    const kt = el('table.pgrid', { id: 'dft-bands-kpoint-table' });
    const kf = (r.kpoints_frac || [])[kIndex] || [];
    kt.append(el('tr', {}, [el('th', { text: 'k-point' }), el('td', { colspan: String(r.n_spins),
      text: `${kIndex}${(r.labels || [])[kIndex] ? ` ${kLabel(r.labels[kIndex])}` : ''}, (${kf.map((v) => num(v, 5)).join(', ')}), ${num((r.distance_invA || [])[kIndex], 5)} 1/A` })]));
    kt.append(el('tr', {}, [el('th', { text: 'band' })].concat(
      (r.bands_relative_eV || []).map((_, s) => el('th', { text: r.n_spins === 2 ? (s ? 'down / eV' : 'up / eV') : 'E - E_F / eV' })))));
    for (let n = 0; n < r.n_bands; n += 1) {
      kt.append(el('tr', {}, [el('td', { text: String(n + 1) })].concat(
        (r.bands_relative_eV || []).map((spin) => el('td.mono', { text: num(spin[n][kIndex], 6) })))));
    }
    body.append(kt);
  }
  const t = el('table.pgrid', { id: 'dft-bands-table' });
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td', { text: v === null || v === undefined ? '-' : String(v) })]));
  const c = r.checks || {};
  const s = r.settings || {};
  const at = (p) => (p ? `${p.label ? kLabel(p.label) : `k-point ${p.index}`} (${(p.kpoint_frac || []).map((v) => num(v, 4)).join(', ')})` : '-');
  add('state', r.run_state);
  add('structure', `${r.structure_key || 'detached'}, ${r.formula}, ${r.boundary}, ${r.xc}`);
  add('source', r.source && r.source.run_id ? `${r.source.kind} ${r.source.run_id}` : 'structure');
  add('Fermi level', `${num(r.fermi_level_eV, 6)} eV (absolute GPAW, ground-state grid)`);
  add('path', `${kLabel(r.path).replace(/,/g, ' | ')}, ${r.n_kpoints} k-points, ${num(r.path_length_invA, 5)} 1/A`);
  add('bands', `${r.n_bands} per spin, ${s.extra_bands} buffer bands discarded, ${r.n_spins} spin channel(s)`);
  add('ground-state symmetry', s.scf_symmetry);
  if (edges.gap_eV !== undefined && edges.gap_eV !== null) {
    add('gap along the path', `${num(edges.gap_eV, 5)} eV, ${edges.gap_kind}`);
    add('valence band top', `${num(edges.vbm_eV, 5)} eV at ${at(edges.vbm)}`);
    add('conduction band bottom', `${num(edges.cbm_eV, 5)} eV at ${at(edges.cbm)}`);
    for (const p of edges.per_spin || []) {
      if (p.direct_gap_eV !== undefined) add(`smallest direct gap${r.n_spins === 2 ? (p.spin ? ', down' : ', up') : ''}`, `${num(p.direct_gap_eV, 5)} eV at ${at(p.direct_gap_at)}`);
    }
  } else {
    add('gap along the path', edges.metallic_on_path ? 'none: a band crosses the Fermi level' : 'not determined');
  }
  add('path distance check', `agrees with GPAW's reciprocal cell to ${num(c.distance_max_relative_error, 3)} (tolerance ${num(c.distance_tolerance, 2)})`);
  add('path step', `${c.nscf_iterations} iterations, ${c.nscf_symmetry_operations} symmetry operation`);
  if (c.source_energy_difference_eV !== undefined) add('energy against source', `${num(c.source_energy_difference_eV, 3)} eV`);
  add('occupations', r.occupations);
  add('orbital character', r.orbital_character);
  add('experimental comparison', r.experimental_comparison ? JSON.stringify(r.experimental_comparison) : 'none recorded');
  add('specification', (r.spec_digest || '').slice(0, 16));
  add('run', r.run_id);
  body.append(t);
  for (const w of r.warnings || []) body.append(el('div.note.warn', { text: w }));
  body.append(el('div.field-row', {}, [
    el('button.tool.sm', { id: 'dft-bands-export', text: 'Export CSV', onclick: () => actions.dftBandsExport(r.run_id) }),
    el('button.tool.sm', { id: 'dft-bands-provenance', text: 'Provenance', onclick: () => actions.dftBandsProvenance(r) })]));
}

/** The provenance of one band structure, for the inspector dialog. */
export function bandsProvenanceView(r) {
  const p = r.provenance || {};
  const box = el('div', { id: 'dft-bands-provenance-view' });
  const t = el('table.pgrid');
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td.txt', { text: v === null || v === undefined || v === '' ? '-' : String(v) })]));
  const origin = p.path_origin || {};
  add('model', p.model);
  add('fidelity', p.fidelity);
  add('origin', p.origin);
  add('specification (SHA-256)', p.inputs_digest);
  add('source', `${(r.source || {}).kind} ${(r.source || {}).run_id || ''}`);
  add('source geometry fingerprint', (r.source || {}).geometry_digest);
  add('source electronic fingerprint', (r.source || {}).electronic_digest);
  add('state', `${r.run_state}: ${r.state_reason}`);
  add('boundary conditions', p.boundary_conditions);
  add('path generator', origin.generator === 'ase'
    ? `ASE ${origin.ase_version}, ${origin.lattice_description || origin.lattice}, ${origin.convention}`
    : origin.generator);
  add('standard path', origin.path);
  add('path used', r.path);
  add('special points', Object.entries(r.special_points || {}).map(([k, v]) => `${kLabel(k)} (${v.map((x) => num(x, 6)).join(', ')})`).join('; '));
  add('segment intervals', (r.segment_intervals || []).join(', '));
  add('reciprocal cell / (1/A)', JSON.stringify(r.reciprocal_cell_invA));
  add('software pinned', Object.entries(p.software_pinned || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('software reported', Object.entries(p.software_reported || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('datasets', (p.paw_datasets || []).map((d) => `${d.symbol} ${String(d.sha256 || '').slice(0, 12)}`).join('; '));
  add('SCF iterations', p.scf_iterations);
  add('tolerances', Object.entries(p.tolerances || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('started', r.started_unix ? new Date(r.started_unix * 1000).toISOString() : null);
  add('finished', r.finished_unix ? new Date(r.finished_unix * 1000).toISOString() : null);
  add('wall time / s', num(r.wall_time_s, 4));
  add('units', Object.entries(r.units || {}).map(([k, v]) => `${k}: ${v}`).join('; '));
  add('array checksums (SHA-256)', Object.entries(r.array_sha256 || {}).map(([k, v]) => `${k} ${String(v).slice(0, 16)}`).join('; '));
  add('experimental comparison', r.experimental_comparison ? JSON.stringify(r.experimental_comparison)
    : 'none: no validated comparison with experiment is recorded, so none is claimed');
  box.append(t);
  box.append(el('div.section-title', {}, el('span', { text: 'Citations' })));
  for (const c of p.references || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: c }));
  box.append(el('div.section-title', {}, el('span', { text: 'Approximations and limitations' })));
  for (const a of p.approximations || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: a }));
  box.append(el('div.section-title', {}, el('span', { text: 'Convergence evidence' })));
  box.append(el('div.hint', { style: { padding: '2px 8px' }, text: (r.convergence || {}).message || '-' }));
  box.append(el('div.section-title', {}, el('span', { text: 'Path step sent and used' })));
  const sent = { ...(p.nscf_parameters || {}) };
  const used = { ...(p.nscf_parameters_used || {}) };
  if (Array.isArray(sent.kpts)) sent.kpts = `${sent.kpts.length} explicit k-points`;
  if (Array.isArray(used.kpts)) used.kpts = `${used.kpts.length} explicit k-points`;
  box.append(el('pre.readout', { style: { whiteSpace: 'pre-wrap', padding: '4px 8px' },
    text: JSON.stringify({ nscf_sent: sent, bands_used: p.bands_used ? { ...p.bands_used, labels: undefined } : null }, null, 1) }));
  box.append(el('div.section-title', {}, el('span', { text: 'Exact GPAW ground-state input' })));
  box.append(el('pre.readout', { style: { whiteSpace: 'pre-wrap', padding: '4px 8px' },
    text: JSON.stringify(p.gpaw_parameters || {}, null, 1) }));
  return box;
}

function eosField(name) {
  return ((state.dftStatus && state.dftStatus.eos_fields) || []).find((f) => f.name === name) || null;
}

function eosRow(grid, check, name, control) {
  const info = eosField(name);
  const text = info ? (info.unit ? `${info.label} / ${info.unit}` : info.label) : name;
  grid.append(el('label', { text, for: control.id || null, title: info ? info.explanation : '' }), control);
  if (info && info.explanation) grid.append(el('div.hint.span2', { text: info.explanation }));
  for (const p of problemsFor(check, name)) {
    grid.append(el('div.note.blocked.span2', { text: p, dataset: { field: name } }));
  }
}

/** Read the equation-of-state controls into the variables and source sent to the service. */
export function dftEosRequest() {
  const kept = state.dftEosVars || {};
  if (!$('#dft-eos-min')) return { variables: { ...kept }, source: state.dftEosSource || null };
  const number = (id) => Number($(id).value);
  const vars = {
    volume_min_scale: number('#dft-eos-min'),
    volume_max_scale: number('#dft-eos-max'),
    n_points: Math.round(number('#dft-eos-points')),
  };
  const chosen = $('#dft-eos-source') ? $('#dft-eos-source').value : 'structure';
  const source = chosen && chosen !== 'structure' ? JSON.parse(chosen) : null;
  return { variables: vars, source };
}

function renderEosControls(st, actions) {
  const check = state.dftEosCheck;
  const spec = check && check.spec ? check.spec : null;
  const settings = spec ? spec.settings : (state.dftEosVars || {});
  const grid = el('div.field-grid');
  const sources = st.dos_sources || [];
  const source = el('select', { id: 'dft-eos-source', 'aria-label': 'reference crystal' },
    [el('option', { value: 'structure' }, 'active structure, settings above')]
      .concat(sources.map((x) => el('option', { value: JSON.stringify({ kind: x.kind, run_id: x.run_id }) }, x.label))));
  const kept = state.dftEosSource;
  if (kept) source.value = JSON.stringify({ kind: kept.kind, run_id: kept.run_id });
  if (!source.value) source.value = 'structure';
  eosRow(grid, check, 'source', source);
  eosRow(grid, check, 'volume_min_scale', numberInput('dft-eos-min', settings.volume_min_scale ?? 0.94,
    { step: 0.01, min: 0.7, max: 1.3, aria: 'smallest volume' }));
  eosRow(grid, check, 'volume_max_scale', numberInput('dft-eos-max', settings.volume_max_scale ?? 1.06,
    { step: 0.01, min: 0.7, max: 1.3, aria: 'largest volume' }));
  eosRow(grid, check, 'n_points', numberInput('dft-eos-points', settings.n_points ?? 7,
    { step: 1, min: 5, max: 15, aria: 'number of volumes' }));
  for (const name of ['volume_scales', 'fit', 'representation', 'charge_e']) {
    for (const p of problemsFor(check, name)) grid.append(el('div.note.blocked.span2', { text: p, dataset: { field: name } }));
  }
  const box = el('div', { id: 'dft-eos-box' });
  if (spec) {
    box.append(el('div.hint', { id: 'dft-eos-fixed', text: `Every volume uses the ${(settings.kpoints || []).join('x')} k-point grid `
      + `and the ${settings.cutoff_eV} eV cutoff of the reference, fixed across the scan. Volumes: `
      + `${(spec.volumes_A3 || []).map((v) => num(v, 5)).join(', ')} A^3 per cell.` }));
  }
  if (check) {
    const shown = new Set();
    for (const texts of Object.values(check.by_field || {})) for (const t of texts) shown.add(t);
    const known = new Set(((state.dftStatus && state.dftStatus.eos_fields) || []).map((f) => f.name)
      .concat(['representation', 'charge_e']));
    for (const [name, texts] of Object.entries(check.by_field || {})) {
      if (known.has(name)) continue;
      for (const t of texts) box.append(el('div.note.blocked', { text: `${name}: ${t}` }));
    }
    for (const t of (check.blocking || []).filter((t) => !shown.has(t))) box.append(el('div.note.blocked', { text: t }));
    for (const t of check.warnings || []) box.append(el('div.note.warn', { text: t }));
  }
  box.append(el('div.field-row', {}, [
    el('button.tool.primary', { id: 'dft-eos-run', text: 'Compute equation of state',
      disabled: check && !check.ok ? true : null, onclick: () => actions.dftEos(dftVariables(), dftEosRequest()) }),
    el('button.tool', { id: 'dft-eos-check', text: 'Check equation of state',
      onclick: () => actions.dftEosChanged(dftVariables(), dftEosRequest()) })]));
  box.append(el('div.hint', { text: 'Runs one ground state per volume, scaling the cell with the atoms at fixed '
    + 'fractional coordinates, and fits the Birch-Murnaghan equation of state. Nothing is kept unless every '
    + 'volume converged with the same settings and the fitted minimum lies inside the sampled range. The '
    + 'structure is not changed until you apply the equilibrium volume.' }));
  if (spec) box.append(el('div.hint.mono', { text: `equation-of-state specification ${spec.short_digest}, version ${spec.version}` }));
  const wrapper = group('Equation of state', false, [grid, box], 'eos');
  for (const input of wrapper.querySelectorAll('input, select')) {
    input.addEventListener('change', () => actions.dftEosChanged(dftVariables(), dftEosRequest()));
  }
  return wrapper;
}

function renderEosRuns(body, actions) {
  const runs = (state.dftStatus && state.dftStatus.eos_runs) || [];
  if (!runs.length) return;
  const select = el('select', { id: 'dft-eos-select', 'aria-label': 'stored equations of state' },
    runs.map((r) => el('option', { value: r.run_id },
      `${r.run_id.slice(-6)} ${r.formula || ''} ${r.xc || ''}${r.state && r.state !== 'current' ? `, ${r.state}` : ''}`)));
  if (state.dftEosResult && state.dftEosResult.run_id) select.value = state.dftEosResult.run_id;
  select.addEventListener('change', () => actions.dftEosShow(select.value));
  const grid = el('div.field-grid');
  grid.append(el('label', { text: `stored equations of state (${runs.length})`, for: 'dft-eos-select' }), select);
  body.append(grid);
}

/** Computed points as markers and the fitted curve as a line, on shared axes. */
export function eosPlot(points, curve, { id, ylabel, xlabel = 'volume / A^3 per cell' }) {
  const w = 300; const h = 160; const left = 50; const right = 8; const top = 8; const bottom = 30;
  const svg = svgNode('svg', { viewBox: `0 0 ${w} ${h}`, class: 'dft-plot eos-plot', role: 'img', id,
    'aria-label': `${ylabel} against ${xlabel}` });
  const all = points.concat(curve).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (!all.length) return svg;
  let x0 = Math.min(...all.map(([x]) => x)); let x1 = Math.max(...all.map(([x]) => x));
  let y0 = Math.min(...all.map(([, y]) => y)); let y1 = Math.max(...all.map(([, y]) => y));
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const pad = 0.06 * (y1 - y0); y0 -= pad; y1 += pad;
  const px = (x) => left + ((x - x0) / (x1 - x0)) * (w - left - right);
  const py = (y) => top + (1 - (y - y0) / (y1 - y0)) * (h - top - bottom);
  svg.append(svgNode('rect', { x: left, y: top, width: w - left - right, height: h - top - bottom, class: 'frame' }));
  if (curve.length) {
    svg.append(svgNode('polyline', { class: 'line', points: curve.filter(([x, y]) => Number.isFinite(y))
      .map(([x, y]) => `${px(x).toFixed(2)},${py(y).toFixed(2)}`).join(' ') }));
  }
  for (const [x, y] of points) {
    if (!Number.isFinite(y)) continue;
    svg.append(svgNode('rect', { x: px(x) - 2.2, y: py(y) - 2.2, width: 4.4, height: 4.4, class: 'marker' }));
  }
  const text = (tx, ty, t, anchor = 'middle') => {
    const n = svgNode('text', { x: tx, y: ty, 'text-anchor': anchor, class: 'axis' });
    n.textContent = t; svg.append(n);
  };
  text(left - 3, top + 8, num(y1, 5), 'end');
  text(left - 3, h - bottom, num(y0, 5), 'end');
  text(left, h - bottom + 11, num(x0, 5), 'start');
  text(w - right, h - bottom + 11, num(x1, 5), 'end');
  text(left + (w - left - right) / 2, h - 4, xlabel);
  const yl = svgNode('text', { x: 10, y: top + (h - top - bottom) / 2, 'text-anchor': 'middle',
    class: 'axis', transform: `rotate(-90 10 ${top + (h - top - bottom) / 2})` });
  yl.textContent = ylabel; svg.append(yl);
  return svg;
}

function renderEosResult(body, r, actions) {
  if (!r) return;
  body.append(el('div.section-title', {}, [el('span', { text: 'Equation of state' }),
    r.ok ? originTag((r.provenance || {}).origin) : el('span.tag', { text: r.status || 'refused' })]));
  if (!r.ok) {
    body.append(el(r.status === 'cancelled' ? 'div.note.warn' : 'div.note.blocked',
      { id: 'dft-eos-reason', text: r.reason || r.error || '' }));
    return;
  }
  if (r.run_state !== 'current') {
    body.append(el(r.run_state === 'applied' ? 'div.note.info' : 'div.note.warn', { id: 'dft-eos-stale', text: r.state_reason }));
  }
  const points = r.points || [];
  const curve = r.fit_curve || {};
  body.append(el('div', { style: { padding: '4px 8px' } }, eosPlot(
    points.map((p) => [p.volume_A3, p.energy_eV]),
    (curve.volumes_A3 || []).map((v, i) => [v, curve.energies_eV[i]]),
    { id: 'dft-eos-energy-plot', ylabel: 'zero-width energy / eV per cell' })));
  if (points.some((p) => p.pressure_GPa !== null && p.pressure_GPa !== undefined)) {
    body.append(el('div', { style: { padding: '4px 8px' } }, eosPlot(
      points.map((p) => [p.volume_A3, p.pressure_GPa]),
      (curve.volumes_A3 || []).map((v, i) => [v, curve.free_pressures_GPa[i]]),
      { id: 'dft-eos-pressure-plot', ylabel: 'pressure / GPa' })));
    body.append(el('div.hint', { id: 'dft-eos-energy-note', text: 'Energy plot: squares are GPAW energies extrapolated to zero '
      + 'smearing width, the line their Birch-Murnaghan fit, the 0 K static-lattice equation of state. Pressure plot: squares '
      + 'are GPAW stress, which is the derivative of the free energy E - TS at the smearing width, and the line is -dF/dV of '
      + 'the separate free-energy fit. Disagreement there is Pulay stress.' }));
  }
  const t = el('table.pgrid', { id: 'dft-eos-table' });
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td', { text: v === null || v === undefined ? '-' : String(v) })]));
  const c = r.checks || {};
  add('state', r.run_state);
  add('structure', `${r.structure_key || 'detached'}, ${r.formula}, ${r.xc}`);
  add('source', r.source && r.source.run_id ? `${r.source.kind} ${r.source.run_id}` : 'structure');
  add('equilibrium volume', `${num(r.V0_A3_per_atom, 6)} A^3 per atom (${num(r.V0_A3, 6)} per cell)`);
  add('equilibrium cell lengths', `${(r.equilibrium_cell_lengths_A || []).map((v) => num(v, 6)).join(', ')} A`);
  add('scale of every lattice vector', num(r.linear_scale, 6));
  add('bulk modulus B0', `${num(r.B0_GPa, 5)} GPa`);
  add("pressure derivative B'", num(r.B1, 4));
  add('minimum energy', `${num(r.E0_eV_per_atom, 7)} eV per atom`);
  add('fitted energy', `${r.energy_definition}, 0 K static lattice`);
  const occ = r.occupations || {};
  const free = r.free_energy_fit || {};
  add('smearing', occ.width_eV ? `${occ.name}, ${num(occ.width_eV, 4)} eV` : 'none (fixed occupations): free and zero-width energies are equal');
  if (occ.width_eV) {
    add('free-energy fit (E - TS)', `V0 ${num(free.V0_A3_per_atom, 6)} A^3 per atom, B0 ${num(free.B0_GPa, 5)} GPa, B' ${num(free.B1, 4)}`);
    add('free minus zero-width V0', `${num(100 * (c.free_minus_zero_width_V0_relative || 0), 3)} %`);
    add('largest entropy term', `${num(c.largest_entropy_term_meV_per_atom, 4)} meV per atom`);
  }
  add('fit residual', `${num(c.fit_rms_meV_per_atom, 3)} meV per atom rms, ${num(c.fit_max_meV_per_atom, 3)} largest`);
  if (c.stress_pressure_max_difference_GPa !== undefined) add('stress against -dF/dV', `${num(c.stress_pressure_max_difference_GPa, 3)} GPa largest difference`);
  if (c.largest_force_eV_A !== undefined) add('largest force at any volume', `${num(c.largest_force_eV_A, 3)} eV/A`);
  add('fixed numerical settings', `${(r.kpoints || []).join('x')} k-points, ${r.cutoff_eV} eV, ${points.length} volumes`);
  add('experimental comparison', r.experimental_comparison ? JSON.stringify(r.experimental_comparison) : 'none recorded');
  add('specification', (r.spec_digest || '').slice(0, 16));
  add('run', r.run_id);
  body.append(t);
  for (const w of r.warnings || []) body.append(el('div.note.warn', { text: w }));
  body.append(el('div.field-row', {}, [
    el('button.tool.sm', { id: 'dft-eos-apply', text: 'Apply equilibrium volume', disabled: r.applicable ? null : true,
      title: 'Scale the structure to the fitted equilibrium volume as one undoable change',
      onclick: () => actions.dftEosApply(r.run_id) }),
    el('button.tool.sm', { id: 'dft-eos-export', text: 'Export CSV', onclick: () => actions.dftEosExport(r.run_id) }),
    el('button.tool.sm', { id: 'dft-eos-provenance', text: 'Provenance', onclick: () => actions.dftEosProvenance(r) })]));
}

/** The provenance of one equation of state, for the inspector dialog. */
export function eosProvenanceView(r) {
  const p = r.provenance || {};
  const box = el('div', { id: 'dft-eos-provenance-view' });
  const t = el('table.pgrid');
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td.txt', { text: v === null || v === undefined || v === '' ? '-' : String(v) })]));
  add('model', p.model);
  add('fidelity', p.fidelity);
  add('origin', p.origin);
  add('specification (SHA-256)', p.inputs_digest);
  add('source', `${(r.source || {}).kind} ${(r.source || {}).run_id || ''}`);
  add('source geometry fingerprint', (r.source || {}).geometry_digest);
  add('state', `${r.run_state}: ${r.state_reason}`);
  add('software pinned', Object.entries(p.software_pinned || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('software reported', Object.entries(p.software_reported || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('datasets', (p.paw_datasets || []).map((d) => `${d.symbol} ${String(d.sha256 || '').slice(0, 12)}`).join('; '));
  add('point specifications', (p.point_spec_digests || []).map((d) => String(d).slice(0, 12)).join(', '));
  add('started', r.started_unix ? new Date(r.started_unix * 1000).toISOString() : null);
  add('finished', r.finished_unix ? new Date(r.finished_unix * 1000).toISOString() : null);
  add('wall time / s', num(r.wall_time_s, 4));
  add('array checksums (SHA-256)', Object.entries(r.array_sha256 || {}).map(([k, v]) => `${k} ${String(v).slice(0, 16)}`).join('; '));
  add('experimental comparison', r.experimental_comparison ? JSON.stringify(r.experimental_comparison)
    : 'none: no validated comparison with experiment is recorded, so none is claimed');
  box.append(t);
  box.append(el('div.section-title', {}, el('span', { text: 'Citations' })));
  for (const c of p.references || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: c }));
  box.append(el('div.section-title', {}, el('span', { text: 'Approximations and limitations' })));
  for (const a of p.approximations || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: a }));
  box.append(el('div.section-title', {}, el('span', { text: 'Convergence evidence' })));
  box.append(el('div.hint', { style: { padding: '2px 8px' }, text: (r.convergence || {}).message || '-' }));
  box.append(el('div.section-title', {}, el('span', { text: 'Exact GPAW input at every volume' })));
  box.append(el('pre.readout', { style: { whiteSpace: 'pre-wrap', padding: '4px 8px' },
    text: JSON.stringify(p.gpaw_parameters || {}, null, 1) }));
  return box;
}

function ldosField(name) {
  return ((state.dftStatus && state.dftStatus.ldos_fields) || []).find((f) => f.name === name) || null;
}

function ldosRow(grid, check, name, control) {
  const info = ldosField(name);
  const text = info ? (info.unit ? `${info.label} / ${info.unit}` : info.label) : name;
  grid.append(el('label', { text, for: control.id || null, title: info ? info.explanation : '' }), control);
  if (info && info.explanation) grid.append(el('div.hint.span2', { text: info.explanation }));
  for (const p of problemsFor(check, name)) {
    grid.append(el('div.note.blocked.span2', { text: p, dataset: { field: name } }));
  }
}

/** Read the LDOS controls into the variables and source sent to the service. */
export function dftLdosRequest() {
  const kept = state.dftLdosVars || {};
  if (!$('#dft-ldos-emin')) return { variables: { ...kept }, source: state.dftLdosSource || null };
  const vars = {
    energy_min_eV: Number($('#dft-ldos-emin').value),
    energy_max_eV: Number($('#dft-ldos-emax').value),
    spin_channels: $('#dft-ldos-spin').value,
    n_bands: Math.round(Number($('#dft-ldos-bands').value)),
  };
  const chosen = $('#dft-ldos-source') ? $('#dft-ldos-source').value : 'structure';
  const source = chosen && chosen !== 'structure' ? JSON.parse(chosen) : null;
  return { variables: vars, source };
}

function renderLdosControls(st, actions) {
  const check = state.dftLdosCheck;
  const spec = check && check.spec ? check.spec : null;
  const settings = spec ? spec.settings : (state.dftLdosVars || {});
  const options = (check && check.options) || {};
  const grid = el('div.field-grid');
  const sources = st.dos_sources || [];
  const source = el('select', { id: 'dft-ldos-source', 'aria-label': 'geometry source' },
    [el('option', { value: 'structure' }, 'active structure, settings above')]
      .concat(sources.map((x) => el('option', { value: JSON.stringify({ kind: x.kind, run_id: x.run_id }) }, x.label))));
  const kept = state.dftLdosSource;
  if (kept) source.value = JSON.stringify({ kind: kept.kind, run_id: kept.run_id });
  if (!source.value) source.value = 'structure';
  ldosRow(grid, check, 'source', source);
  const bias = el('div.field-row', {}, [
    el('button.tool.sm', { id: 'dft-ldos-bias-filled', text: 'Filled states, -1 V',
      title: 'Window -1 to 0 eV: negative sample bias images occupied states',
      onclick: () => { $('#dft-ldos-emin').value = -1; $('#dft-ldos-emax').value = 0; actions.dftLdosChanged(dftVariables(), dftLdosRequest()); } }),
    el('button.tool.sm', { id: 'dft-ldos-bias-empty', text: 'Empty states, +1 V',
      title: 'Window 0 to 1 eV: positive sample bias images unoccupied states',
      onclick: () => { $('#dft-ldos-emin').value = 0; $('#dft-ldos-emax').value = 1; actions.dftLdosChanged(dftVariables(), dftLdosRequest()); } })]);
  grid.append(el('label', { text: 'sample bias preset' }), bias);
  ldosRow(grid, check, 'energy_min_eV', numberInput('dft-ldos-emin', settings.energy_min_eV ?? -1,
    { step: 0.1, aria: 'window start' }));
  ldosRow(grid, check, 'energy_max_eV', numberInput('dft-ldos-emax', settings.energy_max_eV ?? 0,
    { step: 0.1, aria: 'window end' }));
  const spin = el('select', { id: 'dft-ldos-spin', 'aria-label': 'spin channels' },
    (options.spin_channels || ['total']).map((c) => el('option', { value: c }, c)));
  spin.value = settings.spin_channels || 'total';
  ldosRow(grid, check, 'spin_channels', spin);
  ldosRow(grid, check, 'n_bands', numberInput('dft-ldos-bands', settings.n_bands ?? '',
    { step: 1, min: 1, aria: 'bands in the LDOS step' }));
  const box = el('div', { id: 'dft-ldos-box' });
  if (check) {
    const shown = new Set();
    for (const texts of Object.values(check.by_field || {})) for (const t of texts) shown.add(t);
    const known = new Set(((state.dftStatus && state.dftStatus.ldos_fields) || []).map((f) => f.name));
    for (const [name, texts] of Object.entries(check.by_field || {})) {
      if (known.has(name)) continue;
      for (const t of texts) box.append(el('div.note.blocked', { text: `${name}: ${t}` }));
    }
    for (const t of (check.blocking || []).filter((t) => !shown.has(t))) box.append(el('div.note.blocked', { text: t }));
    for (const t of check.warnings || []) box.append(el('div.note.warn', { text: t }));
    if (options.stm_images === false) box.append(el('div.hint', { text: 'This geometry is not a slab with vacuum along c, so STM images cannot be taken from its LDOS; the map itself is still computed.' }));
  }
  box.append(el('div.field-row', {}, [
    el('button.tool.primary', { id: 'dft-ldos-run', text: 'Compute LDOS',
      disabled: check && !check.ok ? true : null, onclick: () => actions.dftLdos(dftVariables(), dftLdosRequest()) }),
    el('button.tool', { id: 'dft-ldos-check', text: 'Check LDOS',
      onclick: () => actions.dftLdosChanged(dftVariables(), dftLdosRequest()) })]));
  box.append(el('div.hint', { text: 'Converges the ground state, then a non-self-consistent step on the same k-points with '
    + 'point-group symmetry off, and sums |psi|^2 of every Kohn-Sham state in the window. The map is exact in the vacuum and '
    + 'differs from the all-electron LDOS inside the PAW spheres. It observes the geometry and never changes it; a run that '
    + 'is cancelled, fails or does not verify keeps nothing.' }));
  if (spec) box.append(el('div.hint.mono', { text: `LDOS specification ${spec.short_digest}, version ${spec.version}` }));
  const wrapper = group('Local density of states and STM image', false, [grid, box], 'ldos');
  for (const input of wrapper.querySelectorAll('input, select')) {
    input.addEventListener('change', () => actions.dftLdosChanged(dftVariables(), dftLdosRequest()));
  }
  return wrapper;
}

function renderLdosRuns(body, actions) {
  const runs = (state.dftStatus && state.dftStatus.ldos_runs) || [];
  if (!runs.length) return;
  const select = el('select', { id: 'dft-ldos-select', 'aria-label': 'stored LDOS runs' },
    runs.map((r) => el('option', { value: r.run_id },
      `${r.run_id.slice(-6)} ${r.formula || ''} ${(r.window_eV || []).join(' to ')} eV${r.state && r.state !== 'current' ? `, ${r.state}` : ''}`)));
  if (state.dftLdosResult && state.dftLdosResult.run_id) select.value = state.dftLdosResult.run_id;
  select.addEventListener('change', () => actions.dftLdosShow(select.value));
  const grid = el('div.field-grid');
  grid.append(el('label', { text: `stored LDOS (${runs.length})`, for: 'dft-ldos-select' }), select);
  body.append(grid);
}

/**
 * A map drawn with the scanning-probe palette, one pixel per grid column,
 * scaled without smoothing, with its value range beside it.
 */
export function heatmap(image, { id, label }) {
  const [nx, ny] = image.shape;
  const canvas = el('canvas', { id, width: nx, height: ny, role: 'img', 'aria-label': label,
    style: { width: '100%', maxWidth: '320px', imageRendering: 'pixelated', display: 'block',
      border: '1px solid var(--border-strong)' } });
  const ctx = canvas.getContext('2d');
  const data = ctx.createImageData(nx, ny);
  const span = (image.max - image.min) || 1;
  const lut = image.lut || [];
  for (let j = 0; j < ny; j += 1) {
    for (let i = 0; i < nx; i += 1) {
      const t = Math.max(0, Math.min(1, (image.values[j * nx + i] - image.min) / span));
      const colour = lut[Math.round(t * (lut.length - 1))] || [0, 0, 0];
      const k = ((ny - 1 - j) * nx + i) * 4;
      data.data[k] = colour[0]; data.data[k + 1] = colour[1]; data.data[k + 2] = colour[2]; data.data[k + 3] = 255;
    }
  }
  ctx.putImageData(data, 0, 0);
  return el('div', { style: { padding: '4px 8px' } }, [canvas,
    el('div.hint.mono', { text: `${label}: ${num(image.min, 4)} to ${num(image.max, 4)} ${image.unit}, ${nx} x ${ny} columns, a along x, b along y` })]);
}

function renderLdosResult(body, r, actions) {
  if (!r) return;
  body.append(el('div.section-title', {}, [el('span', { text: 'Local density of states' }),
    r.ok ? originTag((r.provenance || {}).origin) : el('span.tag', { text: r.status || 'refused' })]));
  if (!r.ok) {
    body.append(el(r.status === 'cancelled' ? 'div.note.warn' : 'div.note.blocked',
      { id: 'dft-ldos-reason', text: r.reason || r.error || '' }));
    return;
  }
  if (!r.current) body.append(el('div.note.warn', { id: 'dft-ldos-stale', text: r.state_reason }));
  const profile = r.profile || {};
  body.append(el('div', { style: { padding: '4px 8px' } }, linePlot(
    [{ name: 'planar average', points: (profile.c_A || []).map((z, i) => [z, profile.planar_average[i]]) }],
    { xlabel: 'height along c / A', ylabel: 'LDOS / states A^-3', id: 'dft-ldos-profile', logY: true, dots: false })));
  body.append(el('div.hint', { text: 'Planar average on a logarithmic scale: in the vacuum it falls as a straight line, '
    + 'the exponential decay that the tunnelling current follows.' }));
  const slice = state.dftLdosSlice;
  const sliceRow = el('div.field-grid');
  const nPlanes = (r.grid_shape || [0, 0, 1])[2];
  const planeIndex = numberInput('dft-ldos-plane', slice ? slice.index : '', { step: 1, min: 0, max: nPlanes - 1, aria: 'plane index along c' });
  planeIndex.addEventListener('change', () => actions.dftLdosSlice(r.run_id, Math.round(Number(planeIndex.value))));
  sliceRow.append(el('label', { text: `plane along c (0 to ${nPlanes - 1})`, for: 'dft-ldos-plane' }), planeIndex);
  body.append(sliceRow);
  if (slice && slice.ok) {
    body.append(heatmap(slice, { id: 'dft-ldos-slice', label: `plane ${slice.index} at c = ${num(slice.c_A, 4)} A` }));
  } else if (slice && !slice.ok) {
    body.append(el('div.note.blocked', { text: slice.error }));
  }
  const stm = r.stm || {};
  if (stm.supported) {
    const request = state.dftLdosImageRequest || { mode: 'constant-height', height_A: 4.0 };
    const imageRow = el('div.field-grid', { id: 'dft-ldos-stm-controls' });
    const mode = el('select', { id: 'dft-ldos-stm-mode', 'aria-label': 'STM mode' }, [
      el('option', { value: 'constant-height' }, 'constant height'),
      el('option', { value: 'constant-current' }, 'constant current (constant LDOS)')]);
    mode.value = request.mode;
    const height = numberInput('dft-ldos-stm-height', request.height_A ?? 4.0, { step: 0.5, aria: 'tip height above the topmost atom' });
    const iso = numberInput('dft-ldos-stm-iso', request.isovalue ?? '', { step: 'any', aria: 'LDOS value to follow' });
    imageRow.append(el('label', { text: 'STM mode', for: 'dft-ldos-stm-mode' }), mode);
    imageRow.append(el('label', { text: 'tip height above topmost atom / A', for: 'dft-ldos-stm-height' }), height);
    imageRow.append(el('label', { text: 'LDOS to follow / states A^-3', for: 'dft-ldos-stm-iso' }), iso);
    const readRequest = () => (mode.value === 'constant-height'
      ? { mode: mode.value, height_A: Number(height.value) }
      : { mode: mode.value, isovalue: Number(iso.value) });
    body.append(imageRow);
    body.append(el('div.field-row', {}, [
      el('button.tool.sm', { id: 'dft-ldos-stm-run', text: 'Take STM image', onclick: () => actions.dftLdosImage(r.run_id, readRequest()) }),
      el('button.tool.sm', { id: 'dft-ldos-stm-export', text: 'Export image CSV', onclick: () => actions.dftLdosImageExport(r.run_id, readRequest()) })]));
    body.append(el('div.hint', { text: 'Tersoff-Hamann: the current is proportional to the LDOS at the tip centre. Constant '
      + 'current is shown as the height at which the LDOS equals the chosen value. No conversion to amperes is made. '
      + 'Heights are limited to the vacuum above the PAW spheres and away from the cell face.' }));
    const image = state.dftLdosImage;
    if (image && image.ok) {
      body.append(heatmap(image, { id: 'dft-ldos-stm-image',
        label: image.mode === 'constant-height' ? `LDOS at ${num(image.height_A, 3)} A` : `height at LDOS ${num(image.isovalue, 3)}` }));
      body.append(el('div.hint', { text: `valid tip heights ${num(image.valid_heights_A[0], 3)} to ${num(image.valid_heights_A[1], 3)} A above the topmost atom` }));
    } else if (image && !image.ok) {
      body.append(el('div.note.blocked', { id: 'dft-ldos-stm-reason', text: image.error }));
    }
  } else if (stm.reason) {
    body.append(el('div.hint', { id: 'dft-ldos-stm-unsupported', text: stm.reason }));
  }
  const t = el('table.pgrid', { id: 'dft-ldos-table' });
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td', { text: v === null || v === undefined ? '-' : String(v) })]));
  add('state', r.run_state);
  add('structure', `${r.structure_key || 'detached'}, ${r.formula}, ${r.boundary}, ${r.xc}`);
  add('source', r.source && r.source.run_id ? `${r.source.kind} ${r.source.run_id}` : 'structure');
  add('window', `${num(r.window_eV[0], 4)} to ${num(r.window_eV[1], 4)} eV about E_F (${num(r.fermi_level_eV, 6)} eV)`);
  add('states in window', `${num(r.states_in_window, 6)} per cell, both spins`);
  add('map integral', `${num(r.map_integral_states, 6)} (pseudo-norm ratio ${num(r.pseudo_norm_ratio, 4)})`);
  add('PAW sphere radii', Object.entries(r.augmentation_radii_A || {}).map(([k, v]) => `${k} ${num(v, 3)} A`).join(', '));
  add('grid', `${(r.grid_shape || []).join(' x ')} points, ${r.n_kpoints} k-points, ${r.n_bands} converged bands`);
  add('spin', `${r.spin_channels}, ${r.n_spins} channel(s)`);
  add('experimental comparison', r.experimental_comparison ? JSON.stringify(r.experimental_comparison) : 'none recorded');
  add('specification', (r.spec_digest || '').slice(0, 16));
  add('run', r.run_id);
  body.append(t);
  for (const w of r.warnings || []) body.append(el('div.note.warn', { text: w }));
  body.append(el('div.field-row', {}, [
    el('button.tool.sm', { id: 'dft-ldos-export', text: 'Export map as Cube', onclick: () => actions.dftLdosExport(r.run_id) }),
    el('button.tool.sm', { id: 'dft-ldos-provenance', text: 'Provenance', onclick: () => actions.dftLdosProvenance(r) })]));
}

/** The provenance of one LDOS, for the inspector dialog. */
export function ldosProvenanceView(r) {
  const p = r.provenance || {};
  const box = el('div', { id: 'dft-ldos-provenance-view' });
  const t = el('table.pgrid');
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td.txt', { text: v === null || v === undefined || v === '' ? '-' : String(v) })]));
  add('model', p.model);
  add('fidelity', p.fidelity);
  add('origin', p.origin);
  add('specification (SHA-256)', p.inputs_digest);
  add('source', `${(r.source || {}).kind} ${(r.source || {}).run_id || ''}`);
  add('source geometry fingerprint', (r.source || {}).geometry_digest);
  add('state', `${r.run_state}: ${r.state_reason}`);
  add('software pinned', Object.entries(p.software_pinned || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('software reported', Object.entries(p.software_reported || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('datasets', (p.paw_datasets || []).map((d) => `${d.symbol} ${String(d.sha256 || '').slice(0, 12)}`).join('; '));
  add('started', r.started_unix ? new Date(r.started_unix * 1000).toISOString() : null);
  add('finished', r.finished_unix ? new Date(r.finished_unix * 1000).toISOString() : null);
  add('array checksums (SHA-256)', Object.entries(r.array_sha256 || {}).map(([k, v]) => `${k} ${String(v).slice(0, 16)}`).join('; '));
  add('experimental comparison', r.experimental_comparison ? JSON.stringify(r.experimental_comparison)
    : 'none: no validated comparison with experiment is recorded, so none is claimed');
  box.append(t);
  box.append(el('div.section-title', {}, el('span', { text: 'Citations' })));
  for (const c of p.references || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: c }));
  box.append(el('div.section-title', {}, el('span', { text: 'Approximations and limitations' })));
  for (const a of p.approximations || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: a }));
  box.append(el('div.section-title', {}, el('span', { text: 'Convergence evidence' })));
  box.append(el('div.hint', { style: { padding: '2px 8px' }, text: (r.convergence || {}).message || '-' }));
  box.append(el('div.section-title', {}, el('span', { text: 'LDOS step sent and used' })));
  box.append(el('pre.readout', { style: { whiteSpace: 'pre-wrap', padding: '4px 8px' },
    text: JSON.stringify({ nscf_sent: p.nscf_parameters, ldos_sent: p.ldos_settings, ldos_used: p.ldos_used }, null, 1) }));
  box.append(el('div.section-title', {}, el('span', { text: 'Exact GPAW ground-state input' })));
  box.append(el('pre.readout', { style: { whiteSpace: 'pre-wrap', padding: '4px 8px' },
    text: JSON.stringify(p.gpaw_parameters || {}, null, 1) }));
  return box;
}

function relaxField(name) {
  return ((state.dftStatus && state.dftStatus.relaxation_fields) || []).find((f) => f.name === name) || null;
}

function relaxRow(grid, check, name, control) {
  const info = relaxField(name);
  const text = info ? (info.unit ? `${info.label} / ${info.unit}` : info.label) : name;
  grid.append(el('label', { text, for: control.id || null, title: info ? info.explanation : '' }), control);
  if (info && info.explanation) grid.append(el('div.hint.span2', { text: info.explanation }));
  for (const p of problemsFor(check, name)) {
    grid.append(el('div.note.blocked.span2', { text: p, dataset: { field: name } }));
  }
}

function renderRelaxControls(st, settings, boundary, actions) {
  const kept = state.dftRelaxVars || {};
  const check = state.dftRelaxCheck;
  const variableAllowed = boundary.boundary === 'bulk' && settings.representation === 'pw';
  const grid = el('div.field-grid');
  const mode = el('select', { id: 'dft-relax-mode', 'aria-label': 'relaxation mode' }, [
    el('option', { value: 'fixed-cell' }, 'fixed cell: move the nuclei'),
    el('option', { value: 'variable-cell', disabled: variableAllowed ? null : true },
      'variable cell: move the nuclei and the cell')]);
  mode.value = kept.mode === 'variable-cell' && variableAllowed ? 'variable-cell' : 'fixed-cell';
  relaxRow(grid, check, 'mode', mode);
  if (!variableAllowed) {
    grid.append(el('div.hint.span2', { text: 'A variable cell needs the plane-wave stress, so it is '
      + 'offered only for a bulk crystal in pw mode.' }));
  }
  const optimizer = el('select', { id: 'dft-relax-optimizer', 'aria-label': 'optimiser' },
    ['BFGS', 'FIRE'].map((o) => el('option', { value: o }, o)));
  optimizer.value = kept.optimizer || 'BFGS';
  relaxRow(grid, check, 'optimizer', optimizer);
  relaxRow(grid, check, 'fmax_eV_A', numberInput('dft-relax-fmax', kept.fmax_eV_A ?? 0.05,
    { step: 0.005, min: 0.001, aria: 'force criterion' }));
  relaxRow(grid, check, 'max_steps', numberInput('dft-relax-steps', kept.max_steps ?? 100,
    { step: 10, min: 1, max: 1000, aria: 'ionic step limit' }));
  const symmetry = el('select', { id: 'dft-relax-symmetry', 'aria-label': 'symmetry' }, [
    el('option', { value: 'preserve' }, 'preserve the starting symmetry'),
    el('option', { value: 'off' }, 'off: allow lower symmetry')]);
  symmetry.value = kept.symmetry || 'preserve';
  relaxRow(grid, check, 'symmetry', symmetry);
  const advanced = el('div.field-grid');
  relaxRow(advanced, check, 'maxstep_A', numberInput('dft-relax-maxstep', kept.maxstep_A ?? 0.2,
    { step: 0.05, min: 0.01, max: 0.5, aria: 'largest step' }));
  let cellBox = null;
  if (mode.value === 'variable-cell') {
    cellBox = el('div.field-grid');
    relaxRow(cellBox, check, 'stress_tol_eV_A3', numberInput('dft-relax-stress-tol',
      kept.stress_tol_eV_A3 ?? 0.005, { step: 0.001, min: 0.0001, aria: 'stress criterion' }));
    relaxRow(cellBox, check, 'target_pressure_GPa', numberInput('dft-relax-pressure',
      kept.target_pressure_GPa ?? 0, { step: 0.5, aria: 'target pressure' }));
    relaxRow(cellBox, check, 'hydrostatic_strain', el('input', { type: 'checkbox',
      id: 'dft-relax-hydrostatic', checked: kept.hydrostatic_strain ? true : null,
      'aria-label': 'hydrostatic strain only' }));
    const mask = kept.cell_mask || [true, true, true, true, true, true];
    relaxRow(cellBox, check, 'cell_mask', el('div', { style: { display: 'flex', gap: '6px' } },
      VOIGT.map((c, i) => el('label', { style: { display: 'flex', gap: '2px', alignItems: 'center' } }, [
        el('input', { type: 'checkbox', id: `dft-relax-mask-${c}`, checked: mask[i] ? true : null,
          'aria-label': `strain ${c} free` }), c]))));
  }
  const fixed = st.structure ? (st.structure.fixed_atoms || 0) : 0;
  const box = el('div', { id: 'dft-relax-box' });
  if (check) {
    const shown = new Set();
    for (const texts of Object.values(check.by_field || {})) for (const t of texts) shown.add(t);
    const known = new Set(((state.dftStatus && state.dftStatus.relaxation_fields) || []).map((f) => f.name));
    for (const [name, texts] of Object.entries(check.by_field || {})) {
      if (known.has(name)) continue;
      for (const t of texts) box.append(el('div.note.blocked', { text: `${name}: ${t}` }));
    }
    for (const t of (check.blocking || []).filter((t) => !shown.has(t))) box.append(el('div.note.blocked', { text: t }));
    for (const t of check.warnings || []) box.append(el('div.note.warn', { text: t }));
  }
  box.append(el('div.field-row', {}, [
    el('button.tool.primary', { id: 'dft-relax-run', text: 'Relax geometry',
      disabled: check && !check.ok ? true : null,
      onclick: () => actions.dftRelax(dftVariables(), dftRelaxVariables()) }),
    el('button.tool', { id: 'dft-relax-check', text: 'Check relaxation',
      onclick: () => actions.dftRelaxChanged(dftVariables(), dftRelaxVariables()) })]));
  box.append(el('div.hint', { text:
    'Uses every ground-state setting above at each ionic step. Atoms marked fixed stay put. '
    + 'A relaxation that is cancelled, times out or fails keeps nothing. A converged one is '
    + 'applied as one undoable change if the structure has not changed meanwhile; one that '
    + 'reaches its step limit is kept as a record and never applied.' }));
  if (check && check.spec) {
    box.append(el('div.hint.mono', { text: `relaxation specification ${check.spec.short_digest}, version ${check.spec.version}` }));
  }
  const children = [grid, group('Advanced', false, [advanced], 'relax-advanced')];
  if (cellBox) children.push(group('Cell', true, [cellBox], 'relax-cell'));
  if (fixed) children.push(el('div.note.info', { text: `${fixed} atom(s) are marked fixed on the structure.` }));
  children.push(box);
  const wrapper = group('Geometry relaxation', false, children, 'relaxation');
  for (const input of wrapper.querySelectorAll('input, select')) {
    input.addEventListener('change', () => actions.dftRelaxChanged(dftVariables(), dftRelaxVariables()));
  }
  return wrapper;
}

function renderRelaxRuns(body, actions) {
  const runs = (state.dftStatus && state.dftStatus.relaxations) || [];
  if (!runs.length) return;
  const select = el('select', { id: 'dft-relax-select', 'aria-label': 'stored relaxations' },
    runs.map((r) => el('option', { value: r.run_id },
      `${r.run_id.slice(-6)} ${r.formula || ''} ${r.mode || ''} ${r.status}${r.state === 'stale' ? ', stale' : ''}`)));
  if (state.dftRelaxResult && state.dftRelaxResult.run_id) select.value = state.dftRelaxResult.run_id;
  select.addEventListener('change', () => actions.dftRelaxShow(select.value));
  const grid = el('div.field-grid');
  grid.append(el('label', { text: `stored relaxations (${runs.length})`, for: 'dft-relax-select' }), select);
  body.append(grid);
}

function renderRelaxResult(body, r, actions) {
  if (!r) return;
  body.append(el('div.section-title', {}, [el('span', { text: 'Relaxation' }),
    r.ok ? originTag((r.provenance || {}).origin) : el('span.tag', { text: r.status || 'refused' })]));
  if (!r.ok) {
    body.append(el(r.status === 'cancelled' ? 'div.note.warn' : 'div.note.blocked',
      { id: 'dft-relax-reason', text: r.reason || r.error || '' }));
    return;
  }
  if (r.convergence && r.convergence.converged === false) {
    body.append(el('div.note.warn', { id: 'dft-relax-unconverged', text: r.convergence.message }));
  } else if (!r.current) {
    body.append(el('div.note.warn', { id: 'dft-relax-stale', text: r.state_reason }));
  }
  const t = el('table.pgrid', { id: 'dft-relax-table' });
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td', { text: v === null || v === undefined ? '-' : String(v) })]));
  add('status', r.status);
  add('state', r.run_state + (r.applied ? ', applied' : ''));
  add('mode', `${r.mode}, ${r.optimizer}`);
  add('structure', `${r.structure_key || 'detached'}, ${r.formula}, ${r.boundary}`);
  add('optimiser steps', `${r.optimizer_steps} (${r.scf_iterations_total} SCF iterations)`);
  add('energy', `${num(r.initial_energy_eV, 10)} → ${num(r.final_energy_eV, 10)} eV`);
  add('energy change', `${num(r.energy_change_eV, 6)} eV`);
  add('max |F| on mobile atoms', `${num(r.initial_max_force_eV_A, 4)} → ${num(r.final_max_force_eV_A, 4)} eV/Å (criterion ${num(r.fmax_eV_A, 4)})`);
  if (r.mode === 'variable-cell') {
    add('stress deviation', `${num(r.final_stress_residual_eV_A3, 4)} eV/Å³ (criterion ${num(r.stress_tol_eV_A3, 4)})`);
  }
  if (r.pressure_GPa !== undefined) add('pressure', `${num(r.pressure_GPa, 5)} GPa`);
  const d = r.displacement || {};
  add('largest displacement', `${num(d.max_A, 5)} Å (atom #${d.max_atom_id}), rms ${num(d.rms_A, 5)} Å`);
  const c = r.cell || {};
  if (c.final_volume_A3 !== undefined) {
    add('volume', `${num(c.initial_volume_A3, 7)} → ${num(c.final_volume_A3, 7)} Å³ (${num(c.volume_change_percent, 4)} %)`);
    add('cell lengths', `${(c.final_lengths_A || []).map((x) => num(x, 6)).join(', ')} Å`);
  }
  if ((r.fixed_atom_ids || []).length) add('fixed atoms', r.fixed_atom_ids.length);
  if (r.convergence) add('convergence', r.convergence.message);
  add('wall time', r.wall_time_s === undefined ? '-' : `${num(r.wall_time_s, 4)} s`);
  add('specification', (r.spec_digest || '').slice(0, 16));
  add('run', r.run_id);
  body.append(t);
  if (r.apply_reason) body.append(el('div.hint', { text: r.apply_reason }));
  const history = r.history || [];
  if (history.length) {
    body.append(el('div', { style: { padding: '4px 8px' } }, linePlot([
      { name: 'free energy', points: history.map((h) => [h.step, h.energy_free_eV]) },
    ], { xlabel: 'optimiser step', ylabel: 'eV', id: 'dft-relax-energy-plot' })));
    const series = [{ name: 'max |F|', points: history.map((h) => [h.step, h.max_force_eV_A]) },
      { name: 'criterion', dash: '4 3', points: history.map((h) => [h.step, r.fmax_eV_A]) }];
    if (r.mode === 'variable-cell') {
      series.push({ name: 'stress', dash: '1 3', points: history.map((h) => [h.step, h.stress_residual_eV_A3]) });
    }
    body.append(el('div', { style: { padding: '4px 8px' } }, linePlot(series,
      { xlabel: 'optimiser step', ylabel: 'residual', logY: true, id: 'dft-relax-force-plot' })));
  }
  for (const w of r.warnings || []) body.append(el('div.note.warn', { text: w }));
  const buttons = [el('button.tool.sm', { id: 'dft-relax-provenance', text: 'Provenance',
    onclick: () => actions.dftRelaxProvenance(r) })];
  if (r.applicable) {
    buttons.unshift(el('button.tool.sm.primary', { id: 'dft-relax-apply', text: 'Apply relaxed geometry',
      onclick: () => actions.dftRelaxApply(r.run_id) }));
  }
  body.append(el('div.field-row', {}, buttons));
}

/** The provenance of one relaxation, for the inspector dialog. */
export function relaxProvenanceView(r) {
  const p = r.provenance || {};
  const box = el('div', { id: 'dft-relax-provenance-view' });
  const t = el('table.pgrid');
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td.txt', { text: v === null || v === undefined ? '-' : String(v) })]));
  add('model', p.model);
  add('fidelity', p.fidelity);
  add('origin', p.origin);
  add('relaxation specification (SHA-256)', p.inputs_digest);
  add('final ground state (SHA-256)', r.final_spec_digest);
  add('geometry before', r.input_geometry_digest);
  add('geometry after', r.output_geometry_digest);
  add('state', `${r.run_state}: ${r.state_reason}`);
  add('boundary conditions', p.boundary_conditions);
  add('software pinned', Object.entries(p.software_pinned || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('software reported', Object.entries(p.software_reported || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  box.append(t);
  box.append(el('div.section-title', {}, el('span', { text: 'Approximations' })));
  for (const a of p.approximations || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: a }));
  box.append(el('div.section-title', {}, el('span', { text: 'Optimiser sent and used' })));
  box.append(el('pre.readout', { style: { whiteSpace: 'pre-wrap', padding: '4px 8px' },
    text: JSON.stringify({ sent: p.relaxation_settings, used: p.relaxation_used }, null, 1) }));
  box.append(el('div.section-title', {}, el('span', { text: 'Exact GPAW input' })));
  box.append(el('pre.readout', { style: { whiteSpace: 'pre-wrap', padding: '4px 8px' },
    text: JSON.stringify(p.gpaw_parameters || {}, null, 1) }));
  return box;
}

function studyParameters(st, settings, boundary) {
  const out = [];
  for (const [name, info] of Object.entries(st.study_parameters || {})) {
    if (name === 'grid_spacing_A' && settings.representation === 'pw') continue;
    if (name === 'cutoff_eV' && settings.representation !== 'pw') continue;
    if (name === 'kpoint_density_A' && boundary.boundary === 'cluster') continue;
    if (name === 'vacuum_A' && boundary.boundary === 'bulk') continue;
    if (name === 'supercell' && boundary.boundary === 'cluster') continue;
    if (name === 'smearing_eV' && settings.occupations === 'fixed') continue;
    out.push([name, info]);
  }
  return out;
}

function renderStudyControls(st, settings, boundary, actions) {
  const kept = state.dftStudyRequest || {};
  const params = studyParameters(st, settings, boundary);
  const parameter = el('select', { id: 'dft-study-parameter', 'aria-label': 'convergence parameter' },
    params.map(([name, info]) => el('option', { value: name }, `${info.label} / ${info.unit}`)));
  if (kept.parameter && params.some(([n]) => n === kept.parameter)) parameter.value = kept.parameter;
  const values = el('input.mono', { type: 'text', id: 'dft-study-values', 'aria-label': 'values to try',
    value: kept.parameter === parameter.value && kept.values ? kept.values.join(', ')
      : (STUDY_DEFAULTS[parameter.value] || '') });
  const observable = el('select', { id: 'dft-study-observable', 'aria-label': 'observable' },
    Object.entries(st.study_observables || {}).map(([name, info]) =>
      el('option', { value: name }, `${info.label} / ${info.unit}`)));
  observable.value = kept.observable || 'energy_per_atom_eV';
  const tolerance = numberInput('dft-study-tolerance',
    kept.tolerance ?? TOLERANCE_DEFAULTS[observable.value], { step: 0.0001, min: 0, aria: 'tolerance' });
  parameter.addEventListener('change', () => {
    values.value = STUDY_DEFAULTS[parameter.value] || '';
    const info = (st.study_parameters || {})[parameter.value];
    help.textContent = info ? info.explanation : '';
  });
  observable.addEventListener('change', () => {
    tolerance.value = TOLERANCE_DEFAULTS[observable.value];
  });
  const help = el('div.hint.span2', { text: ((st.study_parameters || {})[parameter.value] || {}).explanation || '' });
  const grid = el('div.field-grid');
  grid.append(el('label', { text: 'parameter', for: 'dft-study-parameter' }), parameter, help,
    el('label', { text: 'values', for: 'dft-study-values' }), values,
    el('label', { text: 'observable', for: 'dft-study-observable' }), observable,
    el('label', { text: 'tolerance', for: 'dft-study-tolerance' }), tolerance);
  return group('Convergence laboratory', false, [
    el('div.hint', { style: { padding: '4px 8px' }, text:
      'Runs one full ground-state calculation per value, keeps every run, and reports how much '
      + 'the observable still changes between the two most accurate values. That residual '
      + 'measures numerical convergence only; the functional\'s own error is not included.' }),
    grid,
    el('div.field-row', {}, [el('button.tool', { id: 'dft-study-run', text: 'Run convergence study',
      onclick: () => actions.dftStudy(dftStudyRequest(), dftVariables()) })])]);
}

function svgNode(tag, attrs = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

/**
 * A small line plot. `series` is [{points: [[x, y]], dash, name}].
 */
export function linePlot(series, { xlabel = '', ylabel = '', id = null, logY = false, dots = true } = {}) {
  const w = 300; const h = 150; const left = 46; const right = 8; const top = 8; const bottom = 30;
  const svg = svgNode('svg', { viewBox: `0 0 ${w} ${h}`, class: 'dft-plot', role: 'img',
    'aria-label': `${ylabel} against ${xlabel}` });
  if (id) svg.setAttribute('id', id);
  const all = series.flatMap((s) => s.points).filter(([, y]) => y !== null && Number.isFinite(y)
    && (!logY || y > 0));
  if (!all.length) return svg;
  const tx = (x) => x;
  const ty = (y) => (logY ? Math.log10(y) : y);
  let x0 = Math.min(...all.map(([x]) => tx(x))); let x1 = Math.max(...all.map(([x]) => tx(x)));
  let y0 = Math.min(...all.map(([, y]) => ty(y))); let y1 = Math.max(...all.map(([, y]) => ty(y)));
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  if (y0 === y1) { const d = Math.abs(y0) * 0.1 || 1; y0 -= d; y1 += d; }
  const px = (x) => left + ((tx(x) - x0) / (x1 - x0)) * (w - left - right);
  const py = (y) => top + (1 - (ty(y) - y0) / (y1 - y0)) * (h - top - bottom);
  svg.append(svgNode('rect', { x: left, y: top, width: w - left - right, height: h - top - bottom,
    class: 'frame' }));
  for (const s of series) {
    const pts = s.points.filter(([, y]) => y !== null && Number.isFinite(y) && (!logY || y > 0));
    if (!pts.length) continue;
    svg.append(svgNode('polyline', { points: pts.map(([x, y]) => `${px(x)},${py(y)}`).join(' '),
      class: 'line', 'stroke-dasharray': s.dash || '' }));
    if (dots) for (const [x, y] of pts) svg.append(svgNode('circle', { cx: px(x), cy: py(y), r: 2.2, class: 'dot' }));
  }
  const text = (x, y, t, anchor = 'middle') => {
    const n = svgNode('text', { x, y, 'text-anchor': anchor, class: 'axis' });
    n.textContent = t; svg.append(n);
  };
  const fmt = (v) => (logY ? `1e${Math.round(v)}` : num(v, 4));
  text(left - 3, top + 8, fmt(y1), 'end');
  text(left - 3, h - bottom, fmt(y0), 'end');
  text(left, h - bottom + 11, num(x0, 4), 'start');
  text(w - right, h - bottom + 11, num(x1, 4), 'end');
  text(left + (w - left - right) / 2, h - 4, xlabel);
  const yl = svgNode('text', { x: 10, y: top + (h - top - bottom) / 2, 'text-anchor': 'middle',
    class: 'axis', transform: `rotate(-90 10 ${top + (h - top - bottom) / 2})` });
  yl.textContent = ylabel; svg.append(yl);
  if (series.length > 1) {
    const x = w - right - 4;
    series.forEach((s, i) => {
      svg.append(svgNode('line', { x1: x - 16, x2: x, y1: top + 8 + i * 11, y2: top + 8 + i * 11,
        class: 'line', 'stroke-dasharray': s.dash || '' }));
      text(x - 20, top + 11 + i * 11, s.name, 'end');
    });
  }
  return svg;
}

function renderStudyResult(body, r) {
  if (!r) return;
  body.append(el('div.section-title', {}, [el('span', { text: 'Convergence study' }),
    el('span.tag', { text: r.status || '' })]));
  if (!r.ok) {
    body.append(el('div.note.blocked', { text: r.error || r.reason || '' }));
    return;
  }
  const info = r.parameter_info || {};
  const points = (r.points || []).map((p) => [Number(p.value), p.observable]);
  body.append(el('div', { style: { padding: '4px 8px' } },
    linePlot([{ points, name: r.observable }], { xlabel: `${info.label || r.parameter} / ${info.unit || ''}`,
      ylabel: r.unit, id: 'dft-study-plot' })));
  const note = r.tolerance_met ? 'div.note.info' : 'div.note.warn';
  body.append(el(note, { id: 'dft-study-verdict', text: r.verdict }));
  const table = el('table.grid');
  table.append(el('thead', {}, el('tr', {}, ['value', r.unit, 'change', 'from last', 'run'].map((t) =>
    el('th', { text: t })))));
  const tb = el('tbody');
  (r.points || []).forEach((p, i) => {
    tb.append(el('tr', {}, [
      el('td', { text: String(p.value) }),
      el('td', { text: p.observable === null ? p.status : num(p.observable, 8) }),
      el('td', { text: num((r.changes || [])[i], 3) }),
      el('td', { text: num((r.deviations_from_last || [])[i], 3) }),
      el('td', { text: p.run_id, title: p.restart ? `restart: ${p.restart.reason}` : '' })]));
  });
  table.append(tb);
  body.append(el('div.table-wrap', { style: { maxHeight: '160px' } }, table));
  body.append(el('div.hint', { text: r.restart_policy || '' }));
  body.append(el('div.hint', { text: r.note || '' }));
}

function renderRuns(body, actions) {
  const runs = (state.dftStatus && state.dftStatus.runs) || [];
  if (!runs.length) return;
  const select = el('select', { id: 'dft-run-select', 'aria-label': 'stored runs' },
    runs.map((r) => el('option', { value: r.run_id },
      `${r.run_id.slice(-6)} ${r.formula || ''} ${r.xc || ''} ${r.status}${r.state === 'stale' ? ', stale' : ''}`)));
  if (state.dftResult && state.dftResult.run_id) select.value = state.dftResult.run_id;
  select.addEventListener('change', () => actions.dftShow(select.value));
  const grid = el('div.field-grid');
  grid.append(el('label', { text: `stored runs (${runs.length})`, for: 'dft-run-select' }), select);
  body.append(grid);
}

function renderResult(body, r, actions) {
  if (!r) return;
  const cancelled = r.status === 'cancelled';
  body.append(el('div.section-title', {}, [el('span', { text: 'Ground-state run' }),
    cancelled ? el('span.tag', { text: 'cancelled' })
      : originTag(r.ok ? (r.provenance || {}).origin : 'unsupported')]));
  if (!r.ok) {
    body.append(el(cancelled ? 'div.note.warn' : 'div.note.blocked', { id: 'dft-result-reason',
      text: r.reason || r.error || '' }));
  } else if (!r.current) {
    body.append(el('div.note.warn', { id: 'dft-stale', text: r.state_reason }));
  } else if (r.state_reason && r.state_reason.includes('isotopes')) {
    body.append(el('div.note.info', { text: r.state_reason }));
  }
  if (r.owner_replaced) {
    body.append(el('div.note.warn', { text:
      'This run belonged to a project that has since been replaced; its record stayed there.' }));
  }
  const t = el('table.pgrid', { id: 'dft-result-table' });
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td', { text: v === null || v === undefined ? '-' : String(v) })]));
  add('status', r.status);
  add('state', r.current ? 'current' : r.state);
  add('structure', `${r.structure_key || 'detached'}, ${r.formula}, ${r.boundary}`);
  add('classification', r.classification);
  if (r.ok) {
    add('total energy (free)', `${num(r.energy_eV, 10)} eV`);
    add('zero-width extrapolation', `${num(r.extrapolated_energy_eV, 10)} eV`);
    add('energy per atom', `${num(r.energy_per_atom_eV, 8)} eV`);
    if (r.max_force_eV_A !== undefined) add('max |F|', `${num(r.max_force_eV_A, 5)} eV/Å`);
    if (r.pressure_GPa !== undefined) add('pressure', `${num(r.pressure_GPa, 5)} GPa`);
    if (r.fermi_level_eV !== undefined) add('Fermi level', `${num(r.fermi_level_eV, 6)} eV`);
    if (r.band_edges && r.band_edges.gap_eV !== undefined) add('Kohn-Sham gap', `${num(r.band_edges.gap_eV, 5)} eV`);
    if (r.magnetic_moment_muB !== undefined) add('magnetic moment', `${num(r.magnetic_moment_muB, 5)} μB`);
  }
  if (r.convergence) add('convergence', r.convergence.message);
  add('SCF iterations', r.iterations);
  add('wall time', r.wall_time_s === undefined ? '-' : `${num(r.wall_time_s, 4)} s`);
  add('input digest', (r.spec_digest || '').slice(0, 16));
  const restart = (r.provenance || {}).restart;
  if (restart) add('restart', restart.reused_from ? `reused ${restart.reused_from}` : restart.reason);
  add('run', r.run_id);
  body.append(t);

  if (r.ok && r.charge_accounting) {
    const a = r.charge_accounting;
    body.append(el(a.balanced ? 'div.note.info' : 'div.note.warn', { id: 'dft-accounting', text: a.message }));
    const acc = el('table.pgrid');
    const arow = (k, v) => acc.append(el('tr', {}, [el('th', { text: k }), el('td', { text: v })]));
    arow('nuclear charge', `${num(a.nuclear_charge_e, 6)} e`);
    arow('frozen core electrons', num(a.frozen_core_electrons, 6));
    arow('net charge', `${num(a.net_charge_e, 4)} e`);
    arow('valence electrons', `${num(a.gpaw_valence_electrons, 8)} (expected ${num(a.expected_valence_electrons, 8)})`);
    arow('density integral', `${num(a.density_integral_e, 8)} e (expected ${num(a.expected_total_electrons, 8)})`);
    if (a.spin_up_e !== undefined) arow('spin up / down', `${num(a.spin_up_e, 6)} / ${num(a.spin_down_e, 6)} e`);
    if (a.background_charge_e) arow('uniform background', `${num(a.background_charge_e, 4)} e`);
    body.append(acc);
  }
  const history = r.scf_history || [];
  if (history.length) {
    const it = history.map((h) => h.iteration);
    body.append(el('div', { style: { padding: '4px 8px' } }, linePlot([
      { name: 'energy change', points: history.map((h, i) => [it[i], h.energy_change_eV_per_electron]) },
      { name: 'density', dash: '4 3', points: history.map((h, i) => [it[i], h.density_error_electrons_per_electron]) },
      { name: 'eigenstates', dash: '1 3', points: history.map((h, i) => [it[i], h.eigenstate_error_eV2_per_electron]) },
    ], { xlabel: 'SCF iteration', ylabel: 'residual', logY: true, id: 'dft-scf-plot' })));
  }
  if (r.ok && r.arrays) {
    const grids = Object.keys(r.arrays).filter((n) => r.arrays[n].kind === 'volumetric');
    if (grids.length) {
      const which = el('select', { id: 'dft-profile-name', 'aria-label': 'grid to profile' },
        grids.map((n) => el('option', { value: n }, OBSERVABLE_LABELS[n] || n)));
      const axis = el('select', { id: 'dft-profile-axis', 'aria-label': 'profile axis' },
        ['a', 'b', 'c'].map((a, i) => el('option', { value: String(i) }, `along ${a}`)));
      const kept = state.dftProfile || {};
      if (kept.name && grids.includes(kept.name)) which.value = kept.name;
      axis.value = String(kept.axis ?? 2);
      const show = () => actions.dftProfile(r.run_id, which.value, Number(axis.value));
      which.addEventListener('change', show);
      axis.addEventListener('change', show);
      const grid = el('div.field-grid');
      grid.append(el('label', { text: 'planar average', for: 'dft-profile-name' }),
        el('div', { style: { display: 'flex', gap: '3px' } }, [which, axis]));
      body.append(grid);
      if (kept.values && kept.run_id === r.run_id) {
        body.append(el('div', { style: { padding: '4px 8px' } }, linePlot(
          [{ points: kept.coordinate_A.map((x, i) => [x, kept.values[i]]), name: kept.name }],
          { xlabel: `position along ${kept.axis_name} / Å`, ylabel: kept.unit, id: 'dft-profile-plot' })));
      } else {
        body.append(el('div.field-row', {}, [el('button.tool.sm', { text: 'Show profile', onclick: show })]));
      }
    }
  }
  for (const w of r.warnings || []) body.append(el('div.note.warn', { text: w }));
  if (r.evidence) {
    body.append(el('div.hint', { id: 'dft-evidence', text: r.evidence.length
      ? `Convergence evidence: ${r.evidence.map((e) => `${e.parameter} (residual ${num(e.residual, 3)}, tolerance ${num(e.tolerance, 3)})`).join('; ')}`
      : 'No convergence evidence yet: run a convergence study for the parameters this result depends on before treating it as a finding.' }));
  }
  body.append(el('div.field-row', {}, [
    el('button.tool.sm', { id: 'dft-provenance', text: 'Provenance', onclick: () => actions.dftProvenance(r) })]));
}

/** The full provenance of one run, for the inspector dialog. */
export function provenanceView(r) {
  const p = r.provenance || {};
  const box = el('div', { id: 'dft-provenance-view' });
  const t = el('table.pgrid');
  const add = (k, v) => t.append(el('tr', {}, [el('th', { text: k }),
    el('td.txt', { text: v === null || v === undefined ? '-' : String(v) })]));
  add('model', p.model);
  add('fidelity', p.fidelity);
  add('origin', p.origin);
  add('classification', r.classification);
  add('state', r.current ? 'current' : `${r.state}: ${r.state_reason}`);
  add('boundary conditions', p.boundary_conditions);
  add('input digest (SHA-256)', p.inputs_digest);
  add('convergence', r.convergence ? r.convergence.message : '-');
  add('tolerances', Object.entries(p.tolerances || {}).map(([k, v]) => `${k} ${v === null ? 'none' : v}`).join(', '));
  add('grid points', p.grid ? `coarse ${(p.grid.coarse || []).join('x')}, fine ${(p.grid.fine || []).join('x')}` : '-');
  add('bands, IBZ k-points, symmetry operations', `${p.n_bands}, ${p.n_ibz_kpoints}, ${p.symmetry_operations}`);
  add('software pinned', Object.entries(p.software_pinned || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('software reported', Object.entries(p.software_reported || {}).map(([k, v]) => `${k} ${v}`).join('; '));
  add('restart', p.restart ? `${p.restart.reused_from ? `reused ${p.restart.reused_from}` : 'fresh start'}: ${p.restart.reason}` : '-');
  add('dataset licence', p.dataset_license);
  box.append(t);
  box.append(el('div.section-title', {}, el('span', { text: 'PAW datasets' })));
  const d = el('table.grid');
  d.append(el('thead', {}, el('tr', {}, ['element', 'file', 'SHA-256', 'valence', 'core'].map((x) => el('th', { text: x })))));
  const tb = el('tbody');
  for (const ds of p.paw_datasets || []) {
    tb.append(el('tr', {}, [el('td', { text: ds.symbol }), el('td', { text: ds.file || '' }),
      el('td', { text: String(ds.sha256 || '').slice(0, 16), title: ds.sha256 || '' }),
      el('td', { text: num(ds.valence_electrons, 3) }), el('td', { text: num(ds.core_electrons, 3) })]));
  }
  d.append(tb);
  box.append(d);
  box.append(el('div.section-title', {}, el('span', { text: 'Approximations' })));
  for (const a of p.approximations || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: a }));
  box.append(el('div.section-title', {}, el('span', { text: 'Warnings' })));
  if (!(r.warnings || []).length) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: 'none' }));
  for (const w of r.warnings || []) box.append(el('div.note.warn', { text: w }));
  for (const w of r.software_warnings || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: `software: ${w}` }));
  box.append(el('div.section-title', {}, el('span', { text: 'Exact GPAW input' })));
  box.append(el('pre.readout', { style: { whiteSpace: 'pre-wrap', padding: '4px 8px' },
    text: JSON.stringify(p.gpaw_parameters || {}, null, 1) }));
  box.append(el('div.section-title', {}, el('span', { text: 'Units' })));
  box.append(el('div.hint', { style: { padding: '2px 8px' },
    text: Object.entries(p.units || {}).map(([k, v]) => `${k}: ${v}`).join('; ') }));
  for (const ref of p.references || []) box.append(el('div.hint', { style: { padding: '2px 8px' }, text: ref }));
  return box;
}
