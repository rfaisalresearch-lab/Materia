/**
 * Application controller: layout, menus, tools, keyboard, job polling and
 * the dispatch table that connects the interface to the service.
 * @module main
 */

import { ApiError, api, onApiEvent } from './api.js';
import { AtomsView } from './atoms-view.js';
import * as dialogs from './dialogs.js';
import { EXAMPLES, FEATURE_AREAS, exampleCoverage } from './examples.js';
import { Inspector } from './inspector.js';
import { installMenus, loadMenuSpec } from './menus.js';
import * as dftPanel from './dft-panel.js';
import * as panels from './panels.js';
import { PlotPane } from './plots.js';
import { ProbeView } from './probe-view.js';
import { set, state, subscribe } from './state.js';
import { $, $$, clear, debounce, el, fmtTime, num, originTag, tierTag } from './util.js';
import { WaferView } from './wafer-view.js';

let waferView, regionView, atomsView, probeView, plotPane, inspector;
let currentJob = null;
let trajectoryTimer = null;
let trajectoryRequest = 0;
let dismissedTrajectoryRunId = null;
let selectedExampleId = EXAMPLES[0].id;
const runtimeIssues = [];

window.addEventListener('error', (event) => {
  runtimeIssues.push(event.message || 'Unspecified window error');
});
window.addEventListener('unhandledrejection', (event) => {
  runtimeIssues.push(String(event.reason && event.reason.message
    ? event.reason.message : event.reason));
});

async function boot() {
  $('#status-version').textContent = 'Materia';
  window.__materia = { dispatch, state, api, selfCheck: runInterfaceSelfCheck };

  const native = !!(window.pywebview && window.pywebview.api);
  document.documentElement.dataset.native = native ? '1' : '0';
  if (native) {
    document.body.classList.add('native');
    try {
      const info = await window.pywebview.api.platform();
      $('#status-version').textContent = `Materia ${info.version}`;
    } catch { /* the bridge answers once the window is ready */ }
  }

  installMenus(dispatch, await loadMenuSpec());
  installTabs();
  installTools();
  installSplitters();
  installKeyboard();
  installConsole();
  installTrajectoryStage();
  installWorkflowBar();

  waferView = new WaferView('#canvas-wafer', '#wafer-overlay', '#wafer-scale');
  regionView = new WaferView('#canvas-region', '#region-overlay', '#region-scale',
    { microstructure: true });
  atomsView = new AtomsView('#canvas-atoms', '#atoms-overlay', '#atoms-legend',
    '#atoms-scale', { onPick: onAtomPick });
  probeView = new ProbeView({
    canvas: '#canvas-probe', overlay: '#probe-overlay', controls: '#probe-controls',
    scale: '#probe-scale', colorbar: '#probe-colorbar', hover: '#hover-card',
  }, { onIdentify: onProbeIdentify, onProfile: onProfile });
  plotPane = new PlotPane('#canvas-plot', '#plot-note');
  inspector = new Inspector('#atom-body', '#atom-head-hint');

  onApiEvent((e) => {
    if (e.type === 'error') pushWarning(e.detail.message, 'error', e.detail.route);
    if (e.type === 'offline') setStatusJob('service unreachable', true);
  });

  window.addEventListener('resize', debounce(() => {
    waferView.resize(); regionView.resize(); atomsView.resize();
    probeView.resize(); plotPane.resize();
  }, 80));

  await refreshState();
  await offerRecovery();
  await loadMaterials();
  await loadInstrumentOptions();
  await loadSolvers();
  $('#status-version').textContent = `Materia ${state.server.version}`;
  setInterval(pollJobs, 1200);
  render();
}

async function offerRecovery() {
  const recovery = state.server && state.server.recovery;
  if (!recovery || !recovery.available) return;
  const message = el('div', { style: { lineHeight: '1.6' } }, [
    el('p', {}, [
      'Materia found an autosave for ',
      el('b', { text: recovery.project_name }),
      ' from ',
      el('span.mono', { text: recovery.saved_at_text }),
      '.',
    ]),
    el('p', { text: `${recovery.structures} structure(s), ${recovery.scans} scan(s), ` +
      `and ${recovery.history_entries} history entries are available.` }),
  ]);
  const choice = await dialogs.choice('Recover unsaved project', message, [
      { label: 'Discard autosave', value: 'discard', danger: true },
      { label: 'Restore autosave', value: 'restore' },
    ]);
  if (choice === 'restore') {
    set({ server: await api.recoverProject() });
    if (state.server.structure) set({ render: await api.render({}) });
  } else if (choice === 'discard') {
    const result = await api.discardRecovery();
    set({ server: result.state });
  }
}

async function runInterfaceSelfCheck() {
  const checks = [];
  const record = (name, passed, detail = '') => {
    checks.push({ name, passed: !!passed, detail: String(detail || '') });
  };
  const required = [
    '#menubar', '#toolbar', '#project-tree', '#material-list', '#canvas-wafer',
    '#canvas-region', '#canvas-atoms', '#canvas-probe', '#canvas-plot',
    '#script-editor', '#example-list', '#example-detail', '#statusbar', '#modal-backdrop',
  ];
  for (const selector of required) record(`element ${selector}`, !!$(selector));
  let serverState = null;
  try {
    serverState = await api.state();
    record('local service', !!serverState.version, serverState.version);
    record('project state', !!serverState.project, serverState.project && serverState.project.name);
    record('material library', Array.isArray(state.materialList) && state.materialList.length >= 21,
      `${(state.materialList || []).length} definitions`);
  } catch (error) {
    record('local service', false, error.message);
  }
  const original = state.view || 'wafer';
  for (const name of ['wafer', 'region', 'atoms', 'probe']) {
    setView(name);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const view = $(`#view-${name}`);
    const tab = document.querySelector(`#center .view-tab[data-view="${name}"]`);
    const canvas = view && view.querySelector('canvas');
    const rect = canvas ? canvas.getBoundingClientRect() : { width: 0, height: 0 };
    record(`view ${name}`, !!view && !view.hidden && !!tab && tab.getAttribute('aria-selected') === 'true');
    record(`canvas ${name}`, !!canvas && rect.width > 0 && rect.height > 0,
      `${Math.round(rect.width)}x${Math.round(rect.height)}`);
  }
  setView(original);
  try {
    const detail = await api.material('silicon');
    const entries = detail.reconstructions || [];
    const dimer = entries.find((r) => r.id === '2x1-dimer');
    const seven = entries.find((r) => r.id === '7x7-DAS');
    record('reconstruction registry', !!dimer && dimer.supported === true
      && !!seven && seven.supported === false,
    `${entries.filter((r) => r.supported).length} of ${entries.length} available`);
    record('unsupported reconstruction states a reason', !!(seven && seven.reason),
      seven ? seven.reason.slice(0, 60) : '');
  } catch (error) {
    record('reconstruction registry', false, error.message);
  }
  record('runtime errors', runtimeIssues.length === 0, runtimeIssues.join(' | '));
  return {
    ok: checks.every((item) => item.passed),
    checks,
    runtimeIssues: [...runtimeIssues],
    version: serverState ? serverState.version : null,
    checkedAt: new Date().toISOString(),
  };
}

async function refreshDft(reprobe = false) {
  const section = $('#dft-section');
  try {
    if (reprobe) await api.gpawStatus(true);
    const status = await api.dftStatus();
    const geometry = (entry) => {
      if (!entry) return null;
      for (const section of entry.spec.sections) {
        const found = section.fields.find((f) => f.name === 'geometry_digest');
        if (found) return `${entry.key}:${found.value}`;
      }
      return null;
    };
    const previous = geometry(state.dftStatus && state.dftStatus.structure);
    const sameStructure = previous !== null && previous === geometry(status.structure);
    let check = null;
    if (sameStructure && state.dftVars) {
      check = await api.dftSpec(state.dftVars);
    }
    let result = state.dftResult;
    const ids = (status.runs || []).map((r) => r.run_id);
    if (ids.length) {
      const wanted = result && ids.includes(result.run_id) ? result.run_id : ids[0];
      result = await api.dftResult(wanted);
    } else {
      result = null;
    }
    let relaxResult = state.dftRelaxResult;
    const relaxIds = (status.relaxations || []).map((r) => r.run_id);
    if (relaxIds.length) {
      const wanted = relaxResult && relaxIds.includes(relaxResult.run_id) ? relaxResult.run_id : relaxIds[0];
      relaxResult = await api.dftRelaxResult(wanted);
    } else if (relaxResult && relaxResult.ok) {
      relaxResult = null;
    }
    let dosResult = state.dftDosResult;
    const dosIds = (status.dos_runs || []).map((r) => r.run_id);
    if (dosIds.length) {
      const wanted = dosResult && dosIds.includes(dosResult.run_id) ? dosResult.run_id : dosIds[0];
      dosResult = await api.dftDosResult(wanted);
    } else if (dosResult && dosResult.ok) {
      dosResult = null;
    }
    if (!sameStructure) set({ dftDosProjections: null, dftDosSource: null });
    let bandsResult = state.dftBandsResult;
    const bandsIds = (status.bands_runs || []).map((r) => r.run_id);
    if (bandsIds.length) {
      const wanted = bandsResult && bandsIds.includes(bandsResult.run_id) ? bandsResult.run_id : bandsIds[0];
      bandsResult = await api.dftBandsResult(wanted);
    } else if (bandsResult && bandsResult.ok) {
      bandsResult = null;
    }
    if (!sameStructure) set({ dftBandsSource: null, dftBandsView: null });
    let eosResult = state.dftEosResult;
    const eosIds = (status.eos_runs || []).map((r) => r.run_id);
    if (eosIds.length) {
      const wanted = eosResult && eosIds.includes(eosResult.run_id) ? eosResult.run_id : eosIds[0];
      eosResult = await api.dftEosResult(wanted);
    } else if (eosResult && eosResult.ok) {
      eosResult = null;
    }
    if (!sameStructure) set({ dftEosSource: null });
    let ldosResult = state.dftLdosResult;
    let ldosSlice = state.dftLdosSlice;
    const ldosIds = (status.ldos_runs || []).map((r) => r.run_id);
    if (ldosIds.length) {
      const wanted = ldosResult && ldosIds.includes(ldosResult.run_id) ? ldosResult.run_id : ldosIds[0];
      if (!ldosResult || ldosResult.run_id !== wanted) ldosSlice = null;
      ldosResult = await api.dftLdosResult(wanted);
      if (ldosResult.ok && !ldosSlice) ldosSlice = await api.dftLdosSlice(wanted, null);
    } else if (ldosResult && ldosResult.ok) {
      ldosResult = null;
      ldosSlice = null;
    }
    if (!sameStructure) set({ dftLdosSource: null, dftLdosImage: null });
    let ldosCheck = null;
    if (status.structure && status.environment && status.environment.operational) {
      const source = sameStructure ? state.dftLdosSource : null;
      const vars = source ? (state.dftLdosVars || {}) : { ...(sameStructure ? state.dftVars || {} : {}),
        ...(sameStructure ? state.dftLdosVars || {} : {}) };
      ldosCheck = await api.dftLdosSpec(vars, source).catch(() => null);
    }
    let eosCheck = null;
    if (status.structure && status.environment && status.environment.operational) {
      const source = sameStructure ? state.dftEosSource : null;
      const vars = source ? (state.dftEosVars || {}) : { ...(sameStructure ? state.dftVars || {} : {}),
        ...(sameStructure ? state.dftEosVars || {} : {}) };
      eosCheck = await api.dftEosSpec(vars, source).catch(() => null);
    }
    let bandsCheck = null;
    if (status.structure && status.environment && status.environment.operational) {
      const source = sameStructure ? state.dftBandsSource : null;
      const vars = source ? (state.dftBandsVars || {}) : { ...(sameStructure ? state.dftVars || {} : {}),
        ...(sameStructure ? state.dftBandsVars || {} : {}) };
      bandsCheck = await api.dftBandsSpec(vars, source).catch(() => null);
    }
    let dosCheck = null;
    if (status.structure && status.environment && status.environment.operational) {
      const source = sameStructure ? state.dftDosSource : null;
      const vars = source ? (state.dftDosVars || {}) : { ...(sameStructure ? state.dftVars || {} : {}),
        ...(sameStructure ? state.dftDosVars || {} : {}) };
      dosCheck = await api.dftDosSpec(vars, source).catch(() => null);
    }
    set({ dftStatus: status, dftCheck: check, dftVars: check ? state.dftVars : null,
      dftResult: result, dftRelaxResult: relaxResult,
      dftRelaxCheck: sameStructure ? state.dftRelaxCheck : null,
      dftDosResult: dosResult, dftDosCheck: dosCheck,
      dftBandsResult: bandsResult, dftBandsCheck: bandsCheck,
      dftEosResult: eosResult, dftEosCheck: eosCheck,
      dftLdosResult: ldosResult, dftLdosCheck: ldosCheck, dftLdosSlice: ldosSlice });
  } catch {
    set({ dftStatus: null, dftCheck: null });
  }
  if (section) dftPanel.renderDft(section, solverActions);
}

const dftCheckNow = debounce(async () => {
  const vars = dftPanel.dftVariables();
  if (!vars) return;
  set({ dftVars: vars });
  try {
    set({ dftCheck: await api.dftSpec(vars) });
  } catch (err) {
    set({ dftCheck: { ok: false, blocking: [err.message], by_field: {}, warnings: [],
      spec: state.dftStatus.structure.spec } });
  }
  const section = $('#dft-section');
  if (section) dftPanel.renderDft(section, solverActions);
}, 150);

async function refreshState() {
  const s = await api.state();
  set({ server: s });
  if (s.structure) {
    try { set({ render: await api.render({}) }); } catch { /* reported */ }
  } else {
    set({ render: null });
  }
  await refreshElectrostatics();
  await refreshEam();
  await refreshDft();
  return s;
}

async function refreshEam() {
  const section = $('#eam-section');
  if (section && section.childElementCount) set({ eamSpec: panels.eamSpec() });
  let result = null;
  try {
    const status = await api.eamStatus();
    let lammps = null;
    try { lammps = await api.lammpsStatus(); } catch { lammps = null; }
    const backend = (state.eamSpec && state.eamSpec.backend) || 'materia';
    if (backend === 'lammps') {
      if (lammps && lammps.runs && lammps.runs.length) {
        result = await api.lammpsResult(lammps.runs[0].run_id);
      }
    } else if (status.runs && status.runs.length) {
      result = await api.eamResult(status.runs[0].run_id);
    }
    let trajectory = null;
    if (result && result.trajectory_available) {
      trajectory = await api.eamTrajectory(result.run_id);
    }
    set({ eamStatus: status, lammpsStatus: lammps, eamResult: result, trajectory });
    syncTrajectoryStage();
  } catch {
    set({ eamStatus: null, lammpsStatus: null, eamResult: null, trajectory: null });
    syncTrajectoryStage();
  }
  if (section) panels.renderEam(section, solverActions);
}

async function refreshElectrostatics() {
  const section = $('#es-section');
  if (section && section.childElementCount) set({ esSpec: panels.electrostaticsSpec() });
  let result = null;
  try {
    const status = await api.electrostaticsStatus((state.esSpec || {}).settings);
    if (status.runs && status.runs.length) result = await api.electrostaticsResult(status.runs[0].run_id);
    set({ esStatus: status, esResult: result });
  } catch {
    set({ esStatus: null, esResult: null });
  }
  if (section) panels.renderElectrostatics(section, solverActions);
}

function render() {
  const s = state.server;
  if (!s) return;
  $('#project-name').textContent = s.project.name;
  $('#status-selection').textContent = `Selection: ${s.selection.ids.length}`;
  $('#status-model').textContent = 'Model: ' +
    (state.electronic ? state.electronic.solver : (state.lastSolver || '-'));
  $('#console-mode-readout').textContent = s.script_mode;

  const region = s.project.regions[s.project.active_region];
  if (region && !state.roi) {
    set({ roi: { x: region.spec.x_mm, y: region.spec.y_mm } });
  }
  waferView.syncRoi();
  regionView.syncRoi();
  panels.renderTree($('#project-tree'), treeActions);
  panels.renderProperties($('#props-body'), materialActions);
  panels.renderMeasure($('#measure-body'), measureActions);
  panels.renderSelectionPanel($('#selection-tools'), $('#selection-table'), selectionActions);
  atomsView.setData(state.render);
  if (state.render) atomsView.updateSelection(s.selection.ids);
  waferView.draw(); regionView.draw();
  renderWarnings();
  renderHistory();
  plotPane.draw();
  updateScaleReadout();
  const lamp = $('#lamp-warn');
  const n = (s.warnings || []).length;
  lamp.classList.toggle('on-amber', n > 0);
  $('#warn-badge').hidden = n === 0;
  $('#warn-badge').textContent = String(n);
}

async function loadMaterials() {
  const { materials, errors } = await api.materials();
  set({ materialList: materials });
  for (const [path, message] of Object.entries(errors || {})) {
    pushWarning(`Material file rejected: ${path} - ${message}`, 'warning', 'materials');
  }
  panels.renderMaterialList($('#material-list'), $('#material-detail'), materialActions);
  $('#material-search').addEventListener('input', () =>
    panels.renderMaterialList($('#material-list'), $('#material-detail'), materialActions));
}

async function loadInstrumentOptions() {
  set({ instrumentOptions: await api.instrumentOptions() });
  panels.renderInstrument($('#instrument-body'), instrumentActions);
}

async function loadSolvers() {
  set({ solvers: await api.solvers() });
  panels.renderSolvers($('#solver-body'), solverActions);
}

function installTabs() {
  const wire = (root, attr, onChange) => {
    $$(`${root} [${attr}]`).forEach((btn) => {
      btn.addEventListener('click', () => {
        $$(`${root} [${attr}]`).forEach((b) => {
          b.classList.remove('active');
          b.setAttribute('aria-selected', 'false');
        });
        btn.classList.add('active');
        btn.setAttribute('aria-selected', 'true');
        onChange(btn.getAttribute(attr));
      });
    });
  };
  wire('#left-dock .dock-tabs', 'data-panel', (name) => {
    for (const p of ['tree', 'materials', 'examples', 'selection']) {
      $(`#panel-${p}`).hidden = p !== name;
    }
  });
  wire('#right-dock .dock-tabs', 'data-panel', (name) => {
    for (const p of ['atom', 'props', 'instrument', 'solver', 'measure']) {
      $(`#panel-${p}`).hidden = p !== name;
    }
  });
  wire('#bottom-dock .dock-tabs', 'data-panel', (name) => {
    for (const p of ['console', 'log', 'warnings', 'output', 'history']) {
      $(`#panel-${p}`).hidden = p !== name;
    }
  });
  wire('#center .view-tabs:first-of-type', 'data-view', setView);
  wire('#secondary .view-tabs', 'data-plot', (name) => { set({ plot: name }); plotPane.draw(); });
  $$('#panel-atom .subtab').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('#panel-atom .subtab').forEach((b) => {
        b.classList.remove('active'); b.setAttribute('aria-selected', 'false');
      });
      btn.classList.add('active'); btn.setAttribute('aria-selected', 'true');
      inspector.setSection(btn.dataset.sub);
      if (btn.dataset.sub === 'ldos') plotPane.draw();
    });
  });
}

function setView(name) {
  set({ view: name });
  for (const v of ['wafer', 'region', 'atoms', 'probe']) $(`#view-${v}`).hidden = v !== name;
  $$('#center .view-tabs:first-of-type .view-tab').forEach((b) => {
    const on = b.dataset.view === name;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', String(on));
  });
  const hints = {
    wafer: 'click to place the region of interest · wheel zooms · shift-drag pans',
    region: 'procedural microstructure · click to move the region of interest',
    atoms: 'drag rotates · shift-drag pans · wheel zooms · B bonds · C cell · R frame',
    probe: 'hover identifies a feature · measure tool draws a line profile',
  };
  $('#view-hint').textContent = hints[name] || '';
  const idx = { wafer: 0, region: 1, atoms: 3, probe: 4 }[name];
  if (idx !== undefined) { $('#scale-slider').value = String(idx); updateScaleReadout(); }
  if (name === 'region') regionView.syncRoi();
  const view = { wafer: waferView, region: regionView, atoms: atomsView,
                 probe: probeView }[name];
  if (view) view.resize();
}

function updateScaleReadout() {
  const s = state.server;
  const idx = Number($('#scale-slider').value);
  const levels = (s && s.scales) || [];
  const level = levels[idx];
  let span = '-';
  if (level) {
    if (idx === 0 && s.wafer) span = `${s.wafer.spec.diameter_mm} mm`;
    else if (idx === 1) span = '50 µm';
    else if (idx === 2) span = '100 nm';
    else if (idx === 3 && s.structure) span = `${num(s.structure.cell_lengths[0], 4)} Å`;
    else if (idx === 4) span = 'atomic';
    $('#scale-readout').innerHTML =
      `${level.level} · ${span}`;
    $('#scale-readout').title = level.description;
  }
}

function installTools() {
  $$('#toolbar .tool[data-tool]').forEach((btn) => {
    btn.addEventListener('click', () => selectTool(btn.dataset.tool));
  });
  $('#btn-scan').addEventListener('click', () => dispatch('scan.run'));
  $('#btn-abort').addEventListener('click', () => dispatch('scan.abort'));
  $('#tb-technique').addEventListener('change', () =>
    panels.renderInstrument($('#instrument-body'), instrumentActions));
  $('#scale-slider').addEventListener('input', (e) => {
    const idx = Number(e.target.value);
    setView(['wafer', 'region', 'atoms', 'atoms', 'probe'][idx]);
    $('#scale-slider').value = String(idx);
    updateScaleReadout();
  });
}

function selectTool(tool) {
  set({ tool });
  $$('#toolbar .tool[data-tool]').forEach((b) =>
    b.setAttribute('aria-pressed', String(b.dataset.tool === tool)));
  $('#status-tool').textContent = `Tool: ${tool}`;
}

function installSplitters() {
  $$('.splitter').forEach((sp) => {
    const target = document.getElementById(sp.dataset.target);
    if (!target) return;
    const vertical = sp.classList.contains('vertical');
    let start = null;
    const begin = (e) => {
      start = { pos: vertical ? e.clientX : e.clientY,
                size: vertical ? target.offsetWidth : target.offsetHeight };
      document.body.style.cursor = vertical ? 'col-resize' : 'row-resize';
      e.preventDefault();
    };
    sp.addEventListener('mousedown', begin);
    window.addEventListener('mousemove', (e) => {
      if (!start) return;
      const delta = (vertical ? e.clientX : e.clientY) - start.pos;
      const sign = (target.id === 'right-dock') ? -1 : (vertical ? 1 : -1);
      const size = Math.max(120, start.size + sign * delta);
      if (vertical) target.style.width = `${size}px`;
      else target.style.height = `${size}px`;
      waferView && waferView.resize();
      atomsView && atomsView.resize();
      probeView && probeView.resize();
      plotPane && plotPane.resize();
    });
    window.addEventListener('mouseup', () => { start = null; document.body.style.cursor = ''; });
    sp.addEventListener('keydown', (e) => {
      const step = e.shiftKey ? 40 : 12;
      const cur = vertical ? target.offsetWidth : target.offsetHeight;
      let next = cur;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = cur - step;
      else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = cur + step;
      else return;
      if (vertical) target.style.width = `${Math.max(120, next)}px`;
      else target.style.height = `${Math.max(90, next)}px`;
      e.preventDefault();
    });
  });
}

const SHORTCUTS = [
  ['v', 'Navigate tool'], ['s', 'Select tool'], ['b', 'Box select / toggle bonds in 3-D'],
  ['l', 'Lasso select'], ['m', 'Measure tool'], ['t', 'Tip tool'], ['g', 'Move atom tool'],
  ['1-4', 'Switch viewport'], ['F5', 'Acquire scan'], ['F6', 'Relax'],
  ['F7', 'Electronic structure'], ['Ctrl+Z / Ctrl+Shift+Z', 'Undo / redo'],
  ['Ctrl+Enter', 'Run the console script'], ['Ctrl+S', 'Save project as'],
  ['Ctrl+A / Esc', 'Select all / none'], ['R', 'Frame the structure'],
  ['C', 'Toggle the cell box'], ['Delete', 'Create a vacancy at the selection'],
];

function installKeyboard() {
  document.addEventListener('keydown', (e) => {
    const inField = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (e.ctrlKey || e.metaKey) {
      const k = e.key.toLowerCase();
      if (k === 'z' && !e.shiftKey) { e.preventDefault(); dispatch('edit.undo'); return; }
      if ((k === 'z' && e.shiftKey) || k === 'y') { e.preventDefault(); dispatch('edit.redo'); return; }
      if (k === 's') { e.preventDefault(); dispatch('project.save'); return; }
      if (k === 'o') { e.preventDefault(); dispatch('project.open'); return; }
      if (k === 'n') { e.preventDefault(); dispatch('project.new'); return; }
      if (k === 'a' && !inField) { e.preventDefault(); dispatch('select.all'); return; }
      if (k === 'enter') {
        e.preventDefault();
        if (document.activeElement === $('#script-editor')) runScript();
        return;
      }
    }
    if (inField) return;
    const map = { v: 'navigate', s: 'select', b: 'box', l: 'lasso', m: 'measure',
      t: 'tip', g: 'manipulate' };
    const k = e.key.toLowerCase();
    if (map[k] && state.view !== 'atoms') { selectTool(map[k]); e.preventDefault(); return; }
    if (map[k] && !'bc r'.includes(k)) { selectTool(map[k]); e.preventDefault(); return; }
    if (e.key === 'F5') { e.preventDefault(); dispatch('scan.run'); }
    else if (e.key === 'F6') { e.preventDefault(); dispatch('solve.relax'); }
    else if (e.key === 'F7') { e.preventDefault(); dispatch('solve.electronic'); }
    else if (e.key === 'Escape') dispatch('select.none');
    else if (e.key === 'Delete' || e.key === 'Backspace') dispatch('edit.vacancy');
    else if (['1', '2', '3', '4'].includes(e.key)) {
      setView(['wafer', 'region', 'atoms', 'probe'][Number(e.key) - 1]);
    }
  });
}

function installConsole() {
  const sel = $('#script-examples');
  for (const ex of EXAMPLES) sel.append(el('option', { value: ex.id }, ex.label));
  sel.addEventListener('change', () => {
    const ex = EXAMPLES.find((x) => x.id === sel.value);
    if (ex) { loadExample(ex); sel.value = ''; }
  });
  $('#script-editor').value = EXAMPLES[1].code;
  $('#btn-run-script').addEventListener('click', runScript);
  $('#btn-clear-console').addEventListener('click', () => clear($('#script-output')));
  $('#script-mode').addEventListener('change', async (e) => {
    const mode = e.target.value;
    const result = await api.scriptMode(mode, false);
    if (result.requires_confirmation) {
      const ok = await dialogs.confirm('Switch the Python console to trusted mode?',
        `<p>${result.message}</p>`, { okLabel: 'Enable trusted mode', danger: true });
      if (!ok) { e.target.value = state.server.script_mode; return; }
      await api.scriptMode(mode, true);
    }
    await refreshState();
    $('#console-mode-readout').textContent = mode;
  });
  $('#script-editor').addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const t = e.target;
      const s = t.selectionStart;
      t.value = t.value.slice(0, s) + '    ' + t.value.slice(t.selectionEnd);
      t.selectionStart = t.selectionEnd = s + 4;
    }
  });
  installExamples();
}

function loadExample(example, showConsole = true) {
  $('#script-editor').value = example.code.trimStart();
  selectedExampleId = example.id;
  renderExamples();
  if (example.view) setView(example.view);
  if (showConsole) $('#bottom-dock [data-panel="console"]').click();
}

function installTrajectoryStage() {
  $('#trajectory-stage-first').addEventListener('click', () => seekTrajectoryStage(0));
  $('#trajectory-stage-back').addEventListener('click', () =>
    seekTrajectoryStage(Math.max(0, state.trajectoryFrame - 1)));
  $('#trajectory-stage-forward').addEventListener('click', () =>
    seekTrajectoryStage(Math.min((state.trajectory?.frames || 1) - 1,
      state.trajectoryFrame + 1)));
  $('#trajectory-stage-last').addEventListener('click', () =>
    seekTrajectoryStage(Math.max(0, (state.trajectory?.frames || 1) - 1)));
  $('#trajectory-stage-play').addEventListener('click', async () => {
    if (state.trajectory) await solverActions.eamTogglePlayback(state.trajectory.run_id);
  });
  $('#trajectory-stage-range').addEventListener('input', (event) =>
    seekTrajectoryStage(Number(event.target.value)));
  $('#trajectory-stage-speed').addEventListener('change', (event) =>
    solverActions.eamPlaybackSpeed(Number(event.target.value)));
  $('#trajectory-stage-close').addEventListener('click', () => {
    stopTrajectoryPlayback();
    dismissedTrajectoryRunId = state.trajectory ? state.trajectory.run_id : null;
    $('#trajectory-stage').hidden = true;
  });
}

function openExample(category, exampleId) {
  document.querySelector('#left-dock .dock-tab[data-panel="examples"]').click();
  $('#example-category').value = category || '';
  selectedExampleId = exampleId || EXAMPLES.find((item) =>
    !category || item.category === category)?.id || EXAMPLES[0].id;
  renderExamples();
}

function installWorkflowBar() {
  $('#workflow-bar').addEventListener('click', async (event) => {
    const button = event.target.closest('[data-workflow]');
    if (!button) return;
    const action = button.dataset.workflow;
    if (action === 'build') {
      document.querySelector('#left-dock .dock-tab[data-panel="materials"]').click();
    } else if (action === 'simulate') {
      document.querySelector('#right-dock .dock-tab[data-panel="solver"]').click();
    } else if (action === 'collide') {
      openExample('Collisions', 'two-atom-head-on');
    } else if (action === 'quantum') {
      openExample('Quantum', 'quantum-state-explorer');
    } else if (action === 'microscopy') {
      document.querySelector('#right-dock .dock-tab[data-panel="instrument"]').click();
    } else if (action === 'measure') {
      document.querySelector('#right-dock .dock-tab[data-panel="measure"]').click();
    } else if (action === 'results') {
      document.querySelector('#bottom-dock .dock-tab[data-panel="output"]').click();
    } else if (action === 'export') {
      await dispatch('export.npz');
    }
  });
}

async function seekTrajectoryStage(frame) {
  if (!state.trajectory) return;
  stopTrajectoryPlayback();
  await showTrajectoryFrame(state.trajectory.run_id, frame);
}

function syncTrajectoryStage(payload = null) {
  const meta = payload || state.trajectory;
  const stage = $('#trajectory-stage');
  if (!meta || !meta.frames || meta.run_id === dismissedTrajectoryRunId) {
    stage.hidden = true;
    return;
  }
  stage.hidden = false;
  const frame = payload ? payload.frame : Math.min(state.trajectoryFrame || 0, meta.frames - 1);
  const range = $('#trajectory-stage-range');
  range.max = String(meta.frames - 1);
  range.value = String(frame);
  const time = meta.times_fs && meta.times_fs[frame] !== undefined ? meta.times_fs[frame] : null;
  $('#trajectory-stage-time').textContent = `frame ${frame + 1} / ${meta.frames}`
    + (time === null ? '' : ` · ${num(time, 6)} fs`);
  $('#trajectory-stage-first').disabled = frame <= 0;
  $('#trajectory-stage-back').disabled = frame <= 0;
  $('#trajectory-stage-forward').disabled = frame >= meta.frames - 1;
  $('#trajectory-stage-last').disabled = frame >= meta.frames - 1;
  $('#trajectory-stage-speed').value = String(state.trajectorySpeed || 1);
  if (payload && payload.fragments) {
    const largest = payload.fragments.length ? payload.fragments[0] : null;
    let maxSpeed = 0;
    for (let i = 0; i < payload.velocities.length; i += 3) {
      maxSpeed = Math.max(maxSpeed, Math.hypot(payload.velocities[i],
        payload.velocities[i + 1], payload.velocities[i + 2]));
    }
    $('#trajectory-stage-readout').textContent = `${payload.fragments.length} fragment(s)`
      + (largest ? ` · largest ${largest.formula}, ${largest.atoms} atom(s)` : '')
      + ` · maximum speed ${num(maxSpeed, 6)} Å/fs`;
  } else {
    $('#trajectory-stage-readout').textContent =
      'Scrub to any saved instant before, during or after impact.';
  }
}

function installExamples() {
  const categories = [...new Set(EXAMPLES.map((example) => example.category))];
  for (const category of categories) {
    $('#example-category').append(el('option', { value: category }, category));
  }
  $('#example-search').addEventListener('input', renderExamples);
  $('#example-category').addEventListener('change', renderExamples);
  const coverage = exampleCoverage();
  $('#example-coverage').textContent = `${EXAMPLES.length} examples`;
  $('#example-coverage').title =
    `${coverage.covered} declared example topics represented. This is not product completion.`;
  renderExamples();
}

function renderExamples() {
  const query = $('#example-search').value.trim().toLowerCase();
  const category = $('#example-category').value;
  const visible = EXAMPLES.filter((example) => {
    const featureText = example.features.map((id) =>
      FEATURE_AREAS.find((feature) => feature.id === id)?.label || id).join(' ');
    const text = `${example.label} ${example.description} ${example.category} ${featureText}`.toLowerCase();
    return (!category || example.category === category) && (!query || text.includes(query));
  });
  if (!visible.some((example) => example.id === selectedExampleId) && visible.length) {
    selectedExampleId = visible[0].id;
  }
  const list = $('#example-list');
  clear(list);
  for (const example of visible) {
    const row = el('div.row', {
      role: 'option', tabindex: '0',
      'aria-selected': String(example.id === selectedExampleId),
      dataset: { exampleId: example.id },
      onclick: () => { selectedExampleId = example.id; renderExamples(); },
      onkeydown: (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault(); selectedExampleId = example.id; renderExamples();
        }
      },
    }, [
      el('span', { text: example.label }),
      el('span.sub', { text: example.runtime }),
    ]);
    row.classList.toggle('active', example.id === selectedExampleId);
    list.append(row);
  }
  if (!visible.length) {
    list.append(el('div.note', { text: 'No examples match this filter.' }));
    clear($('#example-detail'));
    return;
  }
  const example = EXAMPLES.find((item) => item.id === selectedExampleId) || visible[0];
  const detail = $('#example-detail');
  clear(detail);
  const features = el('div.example-features');
  for (const id of example.features) {
    const feature = FEATURE_AREAS.find((item) => item.id === id);
    features.append(el('span.tag', { text: feature ? feature.label : id }));
  }
  const load = el('button.btn', { type: 'button', text: 'Load in Python',
    onclick: () => loadExample(example) });
  const run = el('button.btn.primary', { type: 'button', text: 'Run example',
    onclick: async () => {
      loadExample(example);
      run.disabled = true;
      try {
        const result = await runScript();
        if (result && result.ok && example.features.includes('collision-playback')
            && state.trajectory) {
          setView('atoms');
          syncTrajectoryStage();
          await showTrajectoryFrame(state.trajectory.run_id, 0);
        }
      } finally { run.disabled = false; }
    } });
  detail.append(
    el('div.section-title', { text: example.label }),
    el('div.note.info', { text: example.description }),
    el('table.pgrid', {}, el('tbody', {}, [
      el('tr', {}, [el('th', { text: 'Area' }), el('td.txt', { text: example.category })]),
      el('tr', {}, [el('th', { text: 'Runtime' }), el('td.txt', { text: example.runtime })]),
      el('tr', {}, [el('th', { text: 'Needs' }), el('td.txt', { text: example.requires })]),
    ])),
    el('div.panel-head', { text: 'Features demonstrated' }),
    features,
    el('div.field-row.example-actions', {}, [load, run]),
    el('div.panel-head', { text: 'Python source' }),
    el('pre.example-source', { text: example.code.trim() }),
  );
}

async function runScript() {
  const out = $('#script-output');
  const code = $('#script-editor').value;
  const block = el('div');
  block.append(el('div.meta', { text: `▸ run · ${new Date().toLocaleTimeString()}` }));
  out.append(block);
  out.scrollTop = out.scrollHeight;
  setStatusJob('script running', true);
  let result = null;
  try {
    result = await api.runScript(code, {
      onJobStart: (job) => { currentJob = job; },
      onProgress: (job) => setStatusJob(`script · ${job.status}`, true),
    });
    if (result.stdout) block.append(el('pre', { style: { margin: 0 }, text: result.stdout }));
    if (result.stderr) block.append(el('pre.err', { style: { margin: 0 }, text: result.stderr }));
    if (!result.ok) {
      block.append(el('pre.err', { style: { margin: 0 },
        text: result.traceback || result.error }));
    }
    block.append(el('div.meta', { text:
      `${result.ok ? 'completed' : 'failed'} in ${fmtTime(result.duration_s)} ` +
      `· mode ${result.mode}` }));
    for (const item of result.view_items || []) await handleViewItem(item);
  } catch (err) {
    block.append(el('pre.err', { style: { margin: 0 }, text: err.message }));
    if (err.payload && err.payload.traceback) {
      block.append(el('pre.err', { style: { margin: 0 }, text: err.payload.traceback }));
    }
  } finally {
    block.append(el('div.sep'));
    out.scrollTop = out.scrollHeight;
    setStatusJob('Idle', false);
    currentJob = null;
    await refreshState();
    render();
  }
  return result;
}

async function handleViewItem(item) {
  const out = $('#numeric-output');
  if (item.kind === 'scan') {
    const payload = await api.scanPayload(item.key, item.channel);
    set({ scan: payload, scanFeatures: null });
    probeView.setScan(payload);
    setView('probe');
  } else if (item.kind === 'plot') {
    set({ profile: { distance_A: item.x, values: item.y, unit: item.ylabel,
      channel: item.title || 'script plot' }, plot: 'profile' });
    plotPane.draw();
  } else if (item.kind === 'structure') {
    await refreshState();
  } else if (item.kind === 'table') {
    out.append(el('div', { text: item.title }));
    out.append(el('pre', { text: JSON.stringify(item.columns, null, 1).slice(0, 4000) }));
  } else if (item.kind === 'message') {
    pushWarning(item.text, item.level, 'script');
  } else {
    out.append(el('div', { text: `${item.title}: ${item.repr || ''}` }));
  }
}

async function pollJobs() {
  try {
    const { jobs } = await api.jobs();
    const tbody = $('#job-table tbody');
    clear(tbody);
    for (const j of jobs) {
      const cancel = el('button.tool.sm', { text: 'cancel',
        disabled: !['queued', 'running'].includes(j.status),
        onclick: () => api.cancelJob(j.id) });
      tbody.append(el('tr', {}, [
        el('td', { text: j.id }),
        el('td', { text: j.kind }),
        el('td', { text: j.status }),
        el('td', { text: `${Math.round((j.progress || 0) * 100)} %` }),
        el('td', { text: fmtTime(j.elapsed_s) }),
        el('td', { text: (j.message || j.error || '').slice(0, 120) }),
        el('td', {}, cancel)]));
    }
  } catch { /* server may be busy */ }
}

function setStatusJob(text, busy) {
  const cell = $('#status-job');
  cell.textContent = text;
  cell.classList.toggle('busy', !!busy);
  $('#btn-abort').disabled = !busy;
  $('#lamp-solver').classList.toggle('on-amber', !!busy);
}

function showUnsupported(out, reconstruction) {
  const box = el('div');
  if (out.unsupported) {
    const u = out.unsupported;
    box.append(el('div.note', { text: u.reason }));
    if (u.reference) box.append(el('div.readout', { text: `Reference: ${u.reference}` }));
    if (u.suggested && u.suggested.length) {
      box.append(el('div.hint', { text: `Models that could answer it: ${u.suggested.join(', ')}` }));
    }
    box.append(el('div.hint', { text:
      'No approximate reconstructed geometry was produced, and no structure was added.' }));
  } else {
    box.append(el('div.note', { text: out.error || 'The request was refused.' }));
  }
  dialogs.info(`Cannot build ${reconstruction || 'that surface'}`, box);
}

function showComparison(comparison) {
  if (!comparison || !comparison.rows || !comparison.rows.length) return;
  const comparable = comparison.comparable !== false;
  const box = el('div');
  if (!comparable) {
    box.append(el('div.note.blocked', { text: comparison.blocked_reason
      || 'This geometry is not a converged result and cannot be compared.' }));
  }
  box.append(el('div.hint', { text: comparable
    ? `${comparison.reconstruction} relaxed with ${comparison.model}, measured against `
      + 'published structural data.'
    : `${comparison.reconstruction}, geometry ${comparison.geometry_status}. The values `
      + 'below are shown so the run can be judged, not as a result.' }));
  const t = el('table.grid');
  t.append(el('thead', {}, el('tr', {}, ['quantity', 'measured', 'published', 'deviation', 'within ±']
    .map((h) => el('th', { text: h })))));
  const tb = el('tbody');
  for (const r of comparison.rows) {
    const unc = r.reference_uncertainty ? ` ± ${r.reference_uncertainty}` : '';
    const verdict = r.within_stated_uncertainty === null
      || r.within_stated_uncertainty === undefined
      ? el('span.tag.unsupported', { text: 'not comparable' })
      : el('span.tag' + (r.within_stated_uncertainty ? '.calculated' : '.unsupported'),
        { text: r.within_stated_uncertainty ? 'yes' : 'no' });
    tb.append(el('tr', { title: `${r.method || ''}\n${r.source || ''}` }, [
      el('td', { text: r.quantity.replace(/_A$|_deg$/, '').replace(/_/g, ' ') }),
      el('td.mono', { text: `${r.measured.toFixed(4)} ${r.unit}` }),
      el('td.mono', { text: `${r.reference}${unc} ${r.unit}` }),
      el('td.mono', { text: comparable
        ? `${r.deviation >= 0 ? '+' : ''}${r.deviation.toFixed(4)} ${r.unit}` : '-' }),
      el('td', {}, verdict)]));
  }
  t.append(tb);
  box.append(t);
  if (comparable) box.append(el('div.note', { text: comparison.note }));
  const sources = [...new Set(comparison.rows.map((r) => r.source).filter(Boolean))];
  for (const src of sources) box.append(el('div.readout', { text: src }));
  dialogs.info(comparable ? 'Measured geometry against published data'
    : 'Geometry is not a comparable result', box, { wide: true });
}

function pushWarning(text, level = 'warning', source = '') {
  const log = $('#warning-log');
  log.prepend(el('div.entry', {}, [
    el('span.lvl.' + level, { text: level }),
    el('span', { text }),
    el('span.src', { text: source })]));
  $('#lamp-warn').classList.add('on-amber');
  const badge = $('#warn-badge');
  badge.hidden = false;
  badge.textContent = String((parseInt(badge.textContent, 10) || 0) + 1);
}

function renderWarnings() {
  const s = state.server;
  if (!s) return;
  const log = $('#warning-log');
  clear(log);
  for (const w of [...(s.warnings || [])].reverse()) {
    log.append(el('div.entry', {}, [
      el('span.lvl.' + (w.level || 'warning'), { text: w.level || 'warning' }),
      el('span', { text: w.text }),
      el('span.src', { text: w.source || '' })]));
  }
}

function renderHistory() {
  const s = state.server;
  const tbody = $('#history-table tbody');
  clear(tbody);
  for (const e of [...(s.project.history || [])].reverse()) {
    tbody.append(el('tr', {}, [
      el('td', { text: new Date(e.timestamp * 1000).toLocaleTimeString() }),
      el('td', { text: e.operation }),
      el('td.txt', { text: e.label }),
      el('td', { text: e.undone ? 'undone' : 'applied' })]));
  }
}

async function onAtomPick(ids, add) {
  if (!ids.length && !add) {
    await api.select('none', {});
    inspector.clearAtom();
  } else {
    await api.select('ids', { ids, add });
    if (ids.length === 1) await inspector.load(ids[0]);
  }
  const s = await refreshState();
  atomsView.updateSelection(s.selection.ids);
  panels.renderSelectionPanel($('#selection-tools'), $('#selection-table'), selectionActions);
  $('#status-selection').textContent = `Selection: ${s.selection.ids.length}`;
}

async function onProbeIdentify(x, y) {
  const info = await api.scanIdentify(x, y, state.scan ? state.scan.key : null);
  if (info.nearest_atom_id) {
    await api.select('ids', { ids: [info.nearest_atom_id] });
    await inspector.load(info.nearest_atom_id);
    const s = await refreshState();
    atomsView.updateSelection(s.selection.ids);
    document.querySelector('#right-dock .dock-tab[data-panel="atom"]').click();
  }
}

async function onProfile(points) {
  const [a, b] = points;
  const profile = await api.scanProfile({ x0: a[0], y0: a[1], x1: b[0], y1: b[1],
    key: state.scan ? state.scan.key : null,
    channel: state.scan ? state.scan.channel : null, n: 300 });
  set({ profile, plot: 'profile' });
  $$('#secondary .view-tab').forEach((t) => {
    const on = t.dataset.plot === 'profile';
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', String(on));
  });
  plotPane.draw();
}

const treeActions = {
  setView,
  restoreCheckpoint: async (name) => { await api.restoreCheckpoint(name); await refreshState(); render(); },
  showScan: async (key) => {
    const payload = await api.scanPayload(key, null);
    set({ scan: payload, scanFeatures: null });
    probeView.setScan(payload);
    setView('probe');
  },
};

const materialActions = {
  selectMaterial: async (id) => {
    set({ material: await api.material(id) });
    panels.renderMaterialList($('#material-list'), $('#material-detail'), materialActions);
    panels.renderMaterialDetail($('#material-detail'), materialActions);
  },
  buildSurface: async (spec) => {
    setStatusJob(spec.reconstruction ? 'building reconstructed surface' : 'building surface', true);
    try {
      const out = await api.buildSurface(spec);
      await refreshState();
      if (out && out.ok === false) { showUnsupported(out, spec.reconstruction); render(); return; }
      setView('atoms');
      render();
      if (out && out.comparison) showComparison(out.comparison);
    } finally { setStatusJob('Idle', false); }
  },
  reconstructionReport: async () => {
    const out = await api.reconstructionReport();
    if (!out.reconstructed) {
      dialogs.info('Reconstruction', el('div.note', { text: out.note }));
      return;
    }
    showComparison(out.comparison);
  },
  waferDialog: (materialId) => dispatch('build.wafer', materialId),
};

const selectionActions = {
  select: async (mode, opts) => {
    await api.select(mode, opts || {});
    const s = await refreshState();
    atomsView.updateSelection(s.selection.ids);
    panels.renderSelectionPanel($('#selection-tools'), $('#selection-table'), selectionActions);
    $('#status-selection').textContent = `Selection: ${s.selection.ids.length}`;
  },
  selectNeighbours: async (radius) => {
    const ids = state.server.selection.ids;
    if (!ids.length) { pushWarning('Select one atom first.', 'info', 'selection'); return; }
    await api.select('neighbors', { atom_id: ids[0], radius_A: radius });
    const s = await refreshState();
    atomsView.updateSelection(s.selection.ids);
    panels.renderSelectionPanel($('#selection-tools'), $('#selection-table'), selectionActions);
  },
  inspect: (id) => inspector.load(id),
};

const measureActions = {
  select: selectionActions.select,
  inspect: selectionActions.inspect,
  measure: async (kind) => {
    const ids = state.server.selection.ids;
    const need = { distance: 2, angle: 3, dihedral: 4, coordination: 1, rdf: 0 }[kind];
    if (ids.length < need) {
      pushWarning(`Select ${need} atom(s) for a ${kind} measurement.`, 'info', 'measure');
      return;
    }
    const code = {
      distance: `print(measure.distance(${ids[0]}, ${ids[1]}), "A")`,
      angle: `print(measure.angle(${ids[0]}, ${ids[1]}, ${ids[2]}), "deg")`,
      dihedral: `print(measure.dihedral(${ids[0]}, ${ids[1]}, ${ids[2]}, ${ids[3]}), "deg")`,
      coordination: `print(measure.coordination(${ids[0]}))`,
      rdf: 'r, g = measure.radial_distribution(8.0)\nview.plot(r, g, title="g(r)", ' +
        'xlabel="r / A", ylabel="g(r)")',
    }[kind];
    $('#script-editor').value = code;
    document.querySelector('#bottom-dock .dock-tab[data-panel="console"]').click();
    await runScript();
  },
  edit: async (op, opts) => {
    const ids = state.server.selection.ids;
    if (!ids.length) { pushWarning('Select at least one atom first.', 'info', 'edit'); return; }
    await api.edit(op, { ids, ...opts });
    await refreshState();
    render();
    if (ids.length === 1) await inspector.load(ids[0]);
  },
};

const instrumentActions = {
  scan: () => dispatch('scan.run'),
  spectroscopy: () => dispatch('scan.sts'),
  forceCurve: () => dispatch('scan.force'),
  features: () => dispatch('scan.features'),
};

function stopTrajectoryPlayback() {
  if (trajectoryTimer !== null) window.clearTimeout(trajectoryTimer);
  trajectoryTimer = null;
  set({ trajectoryPlaying: false });
  const button = $('#eam-playback-toggle');
  if (button) button.textContent = 'Play';
  const stageButton = $('#trajectory-stage-play');
  if (stageButton) stageButton.textContent = 'Play';
}

async function showTrajectoryFrame(runId, frame) {
  const request = ++trajectoryRequest;
  const payload = await api.eamTrajectory(runId, frame);
  if (request !== trajectoryRequest) return;
  if (!atomsView.setTrajectoryFrame(payload)) {
    stopTrajectoryPlayback();
    pushWarning('This trajectory does not match the atoms currently in the 3D view.',
      'warning', 'eam/trajectory');
    return;
  }
  set({ trajectoryFrame: payload.frame });
  const range = $('#eam-frame');
  if (range) range.value = String(payload.frame);
  const time = payload.times_fs[payload.frame];
  const readout = $('#eam-frame-readout');
  if (readout) readout.textContent =
    `frame ${payload.frame + 1} / ${payload.frames}, ${num(time, 6)} fs`;
  const fragmentReadout = $('#eam-fragment-readout');
  if (fragmentReadout) {
    const largest = payload.fragments.length ? payload.fragments[0] : null;
    let maxSpeed = 0;
    for (let i = 0; i < payload.velocities.length; i += 3) {
      maxSpeed = Math.max(maxSpeed, Math.hypot(payload.velocities[i],
        payload.velocities[i + 1], payload.velocities[i + 2]));
    }
    fragmentReadout.textContent = `${payload.fragments.length} fragment(s)`
      + (largest ? `, largest ${largest.formula} with ${largest.atoms} atom(s)` : '')
      + `, maximum speed ${num(maxSpeed, 6)} Å/fs`;
  }
  syncTrajectoryStage(payload);
}

function scheduleTrajectoryFrame(runId) {
  if (!state.trajectoryPlaying || !state.trajectory) return;
  const delay = Math.max(25, 140 / (state.trajectorySpeed || 1));
  trajectoryTimer = window.setTimeout(async () => {
    if (!state.trajectoryPlaying || !state.trajectory) return;
    const next = state.trajectoryFrame + 1;
    if (next >= state.trajectory.frames) {
      stopTrajectoryPlayback();
      return;
    }
    try {
      await showTrajectoryFrame(runId, next);
    } catch (error) {
      stopTrajectoryPlayback();
      pushWarning(error.message, 'error', 'eam/trajectory');
      return;
    }
    scheduleTrajectoryFrame(runId);
  }, delay);
}

const solverActions = {
  solve: (task) => dispatch(`solve.${task}`),
  esRefresh: () => refreshElectrostatics(),
  eamRefresh: () => refreshEam(),
  eamChanged: () => {
    const previous = (state.eamSpec && state.eamSpec.backend) || 'materia';
    set({ eamSpec: panels.eamSpec() });
    if (state.eamSpec.backend !== previous) {
      stopTrajectoryPlayback();
      refreshEam();
      return;
    }
    panels.renderEam($('#eam-section'), solverActions);
  },
  eamRun: async (spec) => {
    stopTrajectoryPlayback();
    set({ eamSpec: spec });
    const external = spec.backend === 'lammps';
    const label = external ? 'LAMMPS' : 'EAM';
    setStatusJob(`${label} running`, true);
    $('#lamp-feedback').classList.add('on-blue');
    try {
      let out;
      try {
        const request = { task: spec.task, potential: spec.potential || null,
          settings: spec.settings };
        const options = {
          onJobStart: (job) => { currentJob = job; },
          onProgress: (job) => setStatusJob(`${label} · ${Math.round((job.progress || 0) * 100)} % · ${job.message || job.status}`, true),
        };
        out = external ? await api.lammpsRun(request, options)
          : await api.eamRun(request, options);
      } catch (err) {
        if (!currentJob) throw err;
        const finished = await api.job(currentJob.id);
        out = finished.result;
        if (!out) throw err;
      }
      await refreshState();
      render();
      if (out && out.ok === false) {
        const title = out.status === 'cancelled' ? `${label} run cancelled`
          : out.status === 'failed' ? `${label} run failed` : `${label} run refused`;
        await dialogs.info(title,
          el(out.status === 'cancelled' ? 'div.note.warn' : 'div.note.blocked', {
            text: out.reason || out.error || '' }));
      }
    } finally {
      $('#lamp-feedback').classList.remove('on-blue');
      setStatusJob('Idle', false);
      currentJob = null;
    }
  },
  eamFrame: async (runId, frame) => showTrajectoryFrame(runId, frame),
  lammpsApply: async (runId) => {
    try {
      const out = await api.lammpsApply(runId);
      if (out && out.ok === false) {
        await dialogs.info('LAMMPS run not applied', el('div.note.blocked', { text: out.error }));
      }
    } finally {
      await refreshState();
      render();
    }
  },
  eamTogglePlayback: async (runId) => {
    if (state.trajectoryPlaying) {
      stopTrajectoryPlayback();
      return;
    }
    if (!state.trajectory || state.trajectory.run_id !== runId) {
      set({ trajectory: await api.eamTrajectory(runId), trajectoryFrame: 0 });
    }
    if (!state.trajectory || !state.trajectory.frames) return;
    if (state.trajectoryFrame >= state.trajectory.frames - 1) {
      set({ trajectoryFrame: 0 });
    }
    set({ trajectoryPlaying: true });
    const button = $('#eam-playback-toggle');
    if (button) button.textContent = 'Pause';
    $('#trajectory-stage-play').textContent = 'Pause';
    await showTrajectoryFrame(runId, state.trajectoryFrame);
    scheduleTrajectoryFrame(runId);
  },
  eamPlaybackSpeed: (speed) => {
    set({ trajectorySpeed: Number.isFinite(speed) && speed > 0 ? speed : 1 });
  },
  eamPlaybackExit: () => {
    stopTrajectoryPlayback();
    set({ trajectoryFrame: 0 });
    atomsView.setData(state.render);
    if (state.server && state.server.selection) {
      atomsView.updateSelection(state.server.selection.ids);
    }
  },
  esAssign: async (spec) => {
    set({ esSpec: spec });
    const out = await api.electrostaticsAssign({ kind: spec.kind, by_element: spec.by_element,
      source: spec.source });
    if (!out.ok) {
      await dialogs.info('Charges not assigned', el('div.note.blocked', { text: out.error }));
    }
    await refreshState();
    render();
  },
  esClear: async () => {
    await api.electrostaticsClear();
    await refreshState();
    render();
  },
  esRun: async (settings) => {
    set({ esSpec: { ...(state.esSpec || {}), ...panels.electrostaticsSpec(), settings } });
    setStatusJob('Electrostatics running', true);
    $('#lamp-feedback').classList.add('on-blue');
    try {
      let out;
      try {
        out = await api.electrostaticsRun(settings, {
          onJobStart: (job) => { currentJob = job; },
          onProgress: (job) => setStatusJob(`Electrostatics · ${Math.round((job.progress || 0) * 100)} % · ${job.message || job.status}`, true),
        });
      } catch (err) {
        if (!currentJob) throw err;
        const finished = await api.job(currentJob.id);
        out = finished.result;
        if (!out) throw err;
      }
      await refreshState();
      render();
      if (out && out.ok === false) {
        await dialogs.info('No electrostatic energy', el('div.note.blocked', {
          text: out.reason || out.error || '' }));
      }
    } finally {
      $('#lamp-feedback').classList.remove('on-blue');
      setStatusJob('Idle', false);
      currentJob = null;
    }
  },
  dftRefresh: (reprobe) => refreshDft(reprobe),
  dftChanged: () => dftCheckNow(),
  dftRun: async (vars) => {
    if (!state.server || !state.server.structure) {
      pushWarning('No structure to run DFT on.', 'warning', 'dft');
      return;
    }
    set({ dftVars: vars });
    setStatusJob('DFT running', true);
    $('#lamp-feedback').classList.add('on-blue');
    try {
      let out;
      try {
        out = await api.dftRun(vars, { reuse_restart: !!state.dftReuse, keep_restart: !!state.dftReuse }, {
          onJobStart: (job) => { currentJob = job; },
          onProgress: (job) => setStatusJob(`DFT · ${Math.round((job.progress || 0) * 100)} % · ${job.message || job.status}`, true),
        });
      } catch (err) {
        if (!currentJob) throw err;
        const finished = await api.job(currentJob.id);
        out = finished.result;
        if (!out) {
          out = { ok: false, status: finished.status, reason: finished.message || err.message };
        }
      }
      if (out && out.run_id && out.ok !== undefined && !out.refused) set({ dftResult: out });
      await refreshState();
      render();
      if (out && out.ok === false) {
        const cancelled = out.status === 'cancelled';
        await dialogs.info(cancelled ? 'DFT run cancelled' : (out.refused ? 'DFT run refused' : 'DFT produced no result'),
          el(cancelled ? 'div.note.warn' : 'div.note.blocked', { id: 'dft-dialog-reason',
            text: out.reason || out.error || '' }));
      }
    } finally {
      $('#lamp-feedback').classList.remove('on-blue');
      setStatusJob('Idle', false);
      currentJob = null;
    }
  },
  dftRelaxChanged: async (vars, relaxVars) => {
    set({ dftVars: vars, dftRelaxVars: relaxVars });
    try {
      set({ dftRelaxCheck: await api.dftRelaxSpec({ ...vars, ...relaxVars }) });
    } catch (err) {
      set({ dftRelaxCheck: { ok: false, blocking: [err.message], by_field: {}, warnings: [] } });
    }
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftRelax: async (vars, relaxVars) => {
    if (!state.server || !state.server.structure) {
      pushWarning('No structure to relax.', 'warning', 'dft-relax');
      return;
    }
    set({ dftVars: vars, dftRelaxVars: relaxVars });
    setStatusJob('DFT relaxation running', true);
    $('#lamp-feedback').classList.add('on-blue');
    try {
      let out;
      try {
        out = await api.dftRelaxRun({ ...vars, ...relaxVars }, {
          onJobStart: (job) => { currentJob = job; },
          onProgress: (job) => setStatusJob(`Relaxation · ${Math.round((job.progress || 0) * 100)} % · ${job.message || job.status}`, true),
        });
      } catch (err) {
        if (!currentJob) throw err;
        const finished = await api.job(currentJob.id);
        out = finished.result;
        if (!out) out = { ok: false, status: finished.status, reason: finished.message || err.message };
      }
      if (out && !out.refused) set({ dftRelaxResult: out });
      if (out && out.refused) set({ dftRelaxCheck: { ...out, ok: false } });
      await refreshState();
      render();
      if (out && out.ok === false) {
        const cancelled = out.status === 'cancelled';
        await dialogs.info(cancelled ? 'Relaxation cancelled'
          : (out.refused ? 'Relaxation refused' : 'Relaxation produced no result'),
        el(cancelled ? 'div.note.warn' : 'div.note.blocked', { id: 'dft-relax-dialog-reason',
          text: out.reason || out.error || '' }));
      }
    } finally {
      $('#lamp-feedback').classList.remove('on-blue');
      setStatusJob('Idle', false);
      currentJob = null;
    }
  },
  dftDosChanged: async (vars, request) => {
    const previous = JSON.stringify(state.dftDosSource || null);
    set({ dftVars: vars, dftDosVars: request.variables, dftDosSource: request.source });
    const sourceChanged = previous !== JSON.stringify(request.source || null);
    const dosVars = sourceChanged
      ? {} : { ...request.variables };
    if (sourceChanged) set({ dftDosProjections: null });
    const merged = request.source ? dosVars : { ...vars, ...dosVars };
    try {
      set({ dftDosCheck: await api.dftDosSpec(merged, request.source) });
    } catch (err) {
      set({ dftDosCheck: { ok: false, blocking: [err.message], by_field: {}, warnings: [] } });
    }
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftDos: async (vars, request) => {
    if (!state.server || (!state.server.structure && !request.source)) {
      pushWarning('No structure or stored run to take a DOS of.', 'warning', 'dft-dos');
      return;
    }
    set({ dftVars: vars, dftDosVars: request.variables, dftDosSource: request.source });
    const merged = request.source ? request.variables : { ...vars, ...request.variables };
    setStatusJob('DOS running', true);
    $('#lamp-feedback').classList.add('on-blue');
    try {
      let out;
      try {
        out = await api.dftDosRun(merged, request.source, {
          onJobStart: (job) => { currentJob = job; },
          onProgress: (job) => setStatusJob(`DOS · ${Math.round((job.progress || 0) * 100)} % · ${job.message || job.status}`, true),
        });
      } catch (err) {
        if (!currentJob) throw err;
        const finished = await api.job(currentJob.id);
        out = finished.result;
        if (!out) out = { ok: false, status: finished.status, reason: finished.message || err.message };
      }
      if (out && !out.refused) set({ dftDosResult: out, dftDosView: null });
      if (out && out.refused) set({ dftDosCheck: { ...(state.dftDosCheck || {}), ...out, ok: false } });
      await refreshState();
      render();
      if (out && out.ok === false) {
        const cancelled = out.status === 'cancelled';
        await dialogs.info(cancelled ? 'DOS cancelled' : (out.refused ? 'DOS refused' : 'DOS produced no result'),
          el(cancelled ? 'div.note.warn' : 'div.note.blocked', { id: 'dft-dos-dialog-reason',
            text: out.reason || out.error || '' }));
      }
    } finally {
      $('#lamp-feedback').classList.remove('on-blue');
      setStatusJob('Idle', false);
      currentJob = null;
    }
  },
  dftDosShow: async (runId) => {
    set({ dftDosResult: await api.dftDosResult(runId), dftDosView: null });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftDosView: (view) => {
    set({ dftDosView: view });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftDosExport: async (runId) => {
    const path = await dialogs.choosePathToSave({ title: 'Export density of states',
      filename: `dos_${String(runId).slice(-6)}.csv` });
    if (!path) return;
    const r = await api.dftDosExport(path, runId);
    pushWarning(`DOS written to ${r.path} (${r.rows} rows)`, 'info', 'export');
  },
  dftDosProvenance: (r) => dialogs.info(`Provenance of DOS ${r.run_id}`,
    dftPanel.dosProvenanceView(r), { wide: true }),
  dftBandsChanged: async (vars, request) => {
    const previous = JSON.stringify(state.dftBandsSource || null);
    const sourceChanged = previous !== JSON.stringify(request.source || null);
    const bandsVars = sourceChanged ? {} : { ...request.variables };
    set({ dftVars: vars, dftBandsVars: bandsVars, dftBandsSource: request.source });
    const merged = request.source ? bandsVars : { ...vars, ...bandsVars };
    try {
      set({ dftBandsCheck: await api.dftBandsSpec(merged, request.source) });
    } catch (err) {
      set({ dftBandsCheck: { ok: false, blocking: [err.message], by_field: {}, warnings: [] } });
    }
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftBands: async (vars, request) => {
    if (!state.server || (!state.server.structure && !request.source)) {
      pushWarning('No structure or stored run to compute bands for.', 'warning', 'dft-bands');
      return;
    }
    set({ dftVars: vars, dftBandsVars: request.variables, dftBandsSource: request.source });
    const merged = request.source ? request.variables : { ...vars, ...request.variables };
    setStatusJob('Band structure running', true);
    $('#lamp-feedback').classList.add('on-blue');
    try {
      let out;
      try {
        out = await api.dftBandsRun(merged, request.source, {
          onJobStart: (job) => { currentJob = job; },
          onProgress: (job) => setStatusJob(`Bands, ${Math.round((job.progress || 0) * 100)} %, ${job.message || job.status}`, true),
        });
      } catch (err) {
        if (!currentJob) throw err;
        const finished = await api.job(currentJob.id);
        out = finished.result;
        if (!out) out = { ok: false, status: finished.status, reason: finished.message || err.message };
      }
      if (out && !out.refused) set({ dftBandsResult: out, dftBandsView: null });
      if (out && out.refused) set({ dftBandsCheck: { ...(state.dftBandsCheck || {}), ...out, ok: false } });
      await refreshState();
      render();
      if (out && out.ok === false) {
        const cancelled = out.status === 'cancelled';
        await dialogs.info(cancelled ? 'Band structure cancelled' : (out.refused ? 'Band structure refused' : 'Band structure produced no result'),
          el(cancelled ? 'div.note.warn' : 'div.note.blocked', { id: 'dft-bands-dialog-reason',
            text: out.reason || out.error || '' }));
      }
    } finally {
      $('#lamp-feedback').classList.remove('on-blue');
      setStatusJob('Idle', false);
      currentJob = null;
    }
  },
  dftBandsShow: async (runId) => {
    set({ dftBandsResult: await api.dftBandsResult(runId), dftBandsView: null });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftBandsView: (view) => {
    set({ dftBandsView: view });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftBandsExport: async (runId) => {
    const path = await dialogs.choosePathToSave({ title: 'Export band structure',
      filename: `bands_${String(runId).slice(-6)}.csv` });
    if (!path) return;
    try {
      const r = await api.dftBandsExport(path, runId);
      pushWarning(`Band structure written to ${r.path} (${r.rows} rows)`, 'info', 'export');
    } catch (err) {
      pushWarning(err.message, 'error', 'export');
    }
  },
  dftEosChanged: async (vars, request) => {
    const previous = JSON.stringify(state.dftEosSource || null);
    const sourceChanged = previous !== JSON.stringify(request.source || null);
    set({ dftVars: vars, dftEosVars: request.variables, dftEosSource: request.source });
    const merged = request.source || sourceChanged ? request.variables : { ...vars, ...request.variables };
    try {
      set({ dftEosCheck: await api.dftEosSpec(merged, request.source) });
    } catch (err) {
      set({ dftEosCheck: { ok: false, blocking: [err.message], by_field: {}, warnings: [] } });
    }
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftEos: async (vars, request) => {
    if (!state.server || (!state.server.structure && !request.source)) {
      pushWarning('No structure or stored run to compute an equation of state for.', 'warning', 'dft-eos');
      return;
    }
    set({ dftVars: vars, dftEosVars: request.variables, dftEosSource: request.source });
    const merged = request.source ? request.variables : { ...vars, ...request.variables };
    setStatusJob('Equation of state running', true);
    $('#lamp-feedback').classList.add('on-blue');
    try {
      let out;
      try {
        out = await api.dftEosRun(merged, request.source, {
          onJobStart: (job) => { currentJob = job; },
          onProgress: (job) => setStatusJob(`Equation of state, ${Math.round((job.progress || 0) * 100)} %, ${job.message || job.status}`, true),
        });
      } catch (err) {
        if (!currentJob) throw err;
        const finished = await api.job(currentJob.id);
        out = finished.result;
        if (!out) out = { ok: false, status: finished.status, reason: finished.message || err.message };
      }
      if (out && !out.refused) set({ dftEosResult: out });
      if (out && out.refused) set({ dftEosCheck: { ...(state.dftEosCheck || {}), ...out, ok: false } });
      await refreshState();
      render();
      if (out && out.ok === false) {
        const cancelled = out.status === 'cancelled';
        await dialogs.info(cancelled ? 'Equation of state cancelled'
          : (out.refused ? 'Equation of state refused' : 'Equation of state produced no result'),
        el(cancelled ? 'div.note.warn' : 'div.note.blocked', { id: 'dft-eos-dialog-reason',
          text: out.reason || out.error || '' }));
      }
    } finally {
      $('#lamp-feedback').classList.remove('on-blue');
      setStatusJob('Idle', false);
      currentJob = null;
    }
  },
  dftEosShow: async (runId) => {
    set({ dftEosResult: await api.dftEosResult(runId) });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftEosApply: async (runId) => {
    const out = await api.dftEosApply(runId);
    if (!out.ok) {
      await dialogs.info('Equilibrium volume not applied', el('div.note.blocked', { text: out.error || '' }));
      return;
    }
    pushWarning(out.message, 'info', 'dft-eos');
    await refreshState();
    render();
  },
  dftEosExport: async (runId) => {
    const path = await dialogs.choosePathToSave({ title: 'Export equation of state',
      filename: `eos_${String(runId).slice(-6)}.csv` });
    if (!path) return;
    try {
      const r = await api.dftEosExport(path, runId);
      pushWarning(`Equation of state written to ${r.path} (${r.rows} rows)`, 'info', 'export');
    } catch (err) {
      pushWarning(err.message, 'error', 'export');
    }
  },
  dftEosProvenance: (r) => dialogs.info(`Provenance of equation of state ${r.run_id}`,
    dftPanel.eosProvenanceView(r), { wide: true }),
  dftLdosChanged: async (vars, request) => {
    const previous = JSON.stringify(state.dftLdosSource || null);
    const sourceChanged = previous !== JSON.stringify(request.source || null);
    const own = sourceChanged ? {} : { ...request.variables };
    set({ dftVars: vars, dftLdosVars: own, dftLdosSource: request.source });
    const merged = request.source ? own : { ...vars, ...own };
    try {
      set({ dftLdosCheck: await api.dftLdosSpec(merged, request.source) });
    } catch (err) {
      set({ dftLdosCheck: { ok: false, blocking: [err.message], by_field: {}, warnings: [] } });
    }
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftLdos: async (vars, request) => {
    if (!state.server || (!state.server.structure && !request.source)) {
      pushWarning('No structure or stored run to take an LDOS of.', 'warning', 'dft-ldos');
      return;
    }
    set({ dftVars: vars, dftLdosVars: request.variables, dftLdosSource: request.source });
    const merged = request.source ? request.variables : { ...vars, ...request.variables };
    setStatusJob('LDOS running', true);
    $('#lamp-feedback').classList.add('on-blue');
    try {
      let out;
      try {
        out = await api.dftLdosRun(merged, request.source, {
          onJobStart: (job) => { currentJob = job; },
          onProgress: (job) => setStatusJob(`LDOS, ${Math.round((job.progress || 0) * 100)} %, ${job.message || job.status}`, true),
        });
      } catch (err) {
        if (!currentJob) throw err;
        const finished = await api.job(currentJob.id);
        out = finished.result;
        if (!out) out = { ok: false, status: finished.status, reason: finished.message || err.message };
      }
      if (out && !out.refused) {
        set({ dftLdosResult: out, dftLdosImage: null,
          dftLdosSlice: out.ok ? await api.dftLdosSlice(out.run_id, null) : null });
      }
      if (out && out.refused) set({ dftLdosCheck: { ...(state.dftLdosCheck || {}), ...out, ok: false } });
      await refreshState();
      render();
      if (out && out.ok === false) {
        const cancelled = out.status === 'cancelled';
        await dialogs.info(cancelled ? 'LDOS cancelled' : (out.refused ? 'LDOS refused' : 'LDOS produced no result'),
          el(cancelled ? 'div.note.warn' : 'div.note.blocked', { id: 'dft-ldos-dialog-reason',
            text: out.reason || out.error || '' }));
      }
    } finally {
      $('#lamp-feedback').classList.remove('on-blue');
      setStatusJob('Idle', false);
      currentJob = null;
    }
  },
  dftLdosShow: async (runId) => {
    set({ dftLdosResult: await api.dftLdosResult(runId), dftLdosSlice: await api.dftLdosSlice(runId, null),
      dftLdosImage: null });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftLdosSlice: async (runId, index) => {
    set({ dftLdosSlice: await api.dftLdosSlice(runId, index) });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftLdosImage: async (runId, request) => {
    set({ dftLdosImageRequest: request, dftLdosImage: await api.dftLdosImage(runId, request) });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftLdosExport: async (runId) => {
    const path = await dialogs.choosePathToSave({ title: 'Export LDOS map',
      filename: `ldos_${String(runId).slice(-6)}.cube` });
    if (!path) return;
    try {
      const r = await api.dftLdosExport(path, runId);
      pushWarning(`LDOS map written to ${r.path} (${r.shape.join(' x ')} points)`, 'info', 'export');
    } catch (err) {
      pushWarning(err.message, 'error', 'export');
    }
  },
  dftLdosImageExport: async (runId, request) => {
    const path = await dialogs.choosePathToSave({ title: 'Export STM image',
      filename: `stm_${String(runId).slice(-6)}.csv` });
    if (!path) return;
    try {
      const r = await api.dftLdosImageExport(path, runId, request);
      pushWarning(`STM image written to ${r.path} (${r.rows} rows)`, 'info', 'export');
    } catch (err) {
      pushWarning(err.message, 'error', 'export');
    }
  },
  dftLdosProvenance: (r) => dialogs.info(`Provenance of LDOS ${r.run_id}`,
    dftPanel.ldosProvenanceView(r), { wide: true }),
  dftBandsEnlarge: (r, view) => dialogs.info(`Band structure ${r.run_id}, ${r.formula}, ${r.xc}`,
    el('div', { style: { padding: '4px 8px' } }, dftPanel.bandPlot(r, view, null,
      { id: 'dft-bands-plot-large', w: 560, h: 330 })), { wide: true }),
  dftBandsProvenance: (r) => dialogs.info(`Provenance of band structure ${r.run_id}`,
    dftPanel.bandsProvenanceView(r), { wide: true }),
  dftRelaxShow: async (runId) => {
    set({ dftRelaxResult: await api.dftRelaxResult(runId) });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftRelaxApply: async (runId) => {
    try {
      const out = await api.dftRelaxApply(runId);
      if (out && out.ok === false) {
        await dialogs.info('Relaxation not applied', el('div.note.blocked', { text: out.error }));
      }
    } finally {
      await refreshState();
      render();
    }
  },
  dftRelaxProvenance: (r) => dialogs.info(`Provenance of relaxation ${r.run_id}`,
    dftPanel.relaxProvenanceView(r), { wide: true }),
  dftShow: async (runId) => {
    set({ dftResult: await api.dftResult(runId), dftProfile: null });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftProfile: async (runId, name, axis) => {
    const out = await api.dftArray(runId, name, axis);
    if (!out.ok) {
      pushWarning(out.error, 'warning', 'dft');
      return;
    }
    set({ dftProfile: { ...out, axis, axis_name: out.axis } });
    const section = $('#dft-section');
    if (section) dftPanel.renderDft(section, solverActions);
  },
  dftProvenance: (r) => dialogs.info(`Provenance of DFT run ${r.run_id}`,
    dftPanel.provenanceView(r), { wide: true }),
  dftStudy: async (request, vars) => {
    set({ dftStudyRequest: request, dftVars: vars });
    setStatusJob('Convergence study running', true);
    $('#lamp-feedback').classList.add('on-blue');
    try {
      let out;
      try {
        out = await api.dftStudy(request, vars, {
          onJobStart: (job) => { currentJob = job; },
          onProgress: (job) => setStatusJob(`Convergence · ${Math.round((job.progress || 0) * 100)} % · ${job.message || job.status}`, true),
        });
      } catch (err) {
        if (!currentJob) throw err;
        const finished = await api.job(currentJob.id);
        out = finished.result;
        if (!out) out = await api.dftStudyResult().catch(() => ({ ok: false, error: err.message }));
      }
      set({ dftStudy: out });
      await refreshState();
      render();
      if (out && out.ok === false) {
        await dialogs.info('Convergence study refused', el('div.note.blocked', { text: out.error || '' }));
      }
    } finally {
      $('#lamp-feedback').classList.remove('on-blue');
      setStatusJob('Idle', false);
      currentJob = null;
    }
  },
};

async function dispatch(action, arg) {
  if (!action) return;
  try {
    await handle(action, arg);
  } catch (err) {
    if (err instanceof ApiError) {
      pushWarning(err.message, 'error', action);
      await dialogs.info('Operation failed', el('div', {}, [
        el('div.note.blocked', { text: err.message }),
        err.payload && err.payload.traceback
          ? el('pre', { style: { fontSize: '10px', overflow: 'auto', maxHeight: '260px' },
              text: err.payload.traceback })
          : null]));
    } else {
      pushWarning(String(err && err.message ? err.message : err), 'error', action);
      throw err;
    }
  } finally {
    setStatusJob('Idle', false);
  }
}

async function handle(action, arg) {
  const s = state.server;
  switch (action) {
    case 'project.new': {
      const v = await dialogs.form('New project',
        [{ name: 'name', label: 'name', value: 'Untitled project' }]);
      if (!v) return;
      await api.newProject(v.name);
      set({ scan: null, atom: null, electronic: null, bands: null });
      probeView.setScan(null);
      await refreshState(); render();
      return;
    }
    case 'project.open': {
      const path = await dialogs.choosePathToOpen({
        title: 'Open project', filters: ['Materia project (*.materia)', 'All files (*.*)'],
        note: 'The file is read from this machine; Materia never contacts the network.' });
      if (!path) return;
      await api.openProject(path);
      await refreshState(); render();
      return;
    }
    case 'project.save': {
      const path = await dialogs.choosePathToSave({
        title: 'Save project as',
        filename: `${(s.project.name || 'project').replace(/\s+/g, '_')}.materia` });
      if (!path) return;
      const r = await api.saveProject(path);
      pushWarning(`Project saved to ${r.path}`, 'info', 'file');
      if (window.pywebview && window.pywebview.api) {
        window.pywebview.api.set_title(s.project.name);
      }
      await refreshState(); render();
      return;
    }
    case 'project.checkpoint': {
      const v = await dialogs.form('Save checkpoint', [
        { name: 'name', label: 'name', value: `checkpoint-${s.project.checkpoints.length + 1}` },
        { name: 'note', label: 'note', value: '' }]);
      if (!v) return;
      await api.checkpoint(v.name, v.note);
      await refreshState(); render();
      return;
    }
    case 'project.provenance': {
      const { text } = await api.provenance();
      await dialogs.info('Provenance report',
        el('pre', { style: { fontFamily: 'var(--code)', fontSize: '11px',
          whiteSpace: 'pre-wrap', margin: 0 }, text }), { wide: true });
      return;
    }
    case 'import.structure': {
      const path = await dialogs.choosePathToOpen({
        title: 'Import structure',
        filters: ['Structures (*.xyz;*.extxyz;*.cif;*.pdb;*.data;POSCAR;CONTCAR)',
                  'All files (*.*)'],
        note: 'XYZ, extended XYZ, CIF, POSCAR/CONTCAR, PDB or LAMMPS data.' });
      if (!path) return;
      await api.importStructure(path);
      await refreshState(); setView('atoms'); render();
      return;
    }
    case 'export.structure': {
      const v = await dialogs.form('Export structure', [
        { name: 'format', label: 'format', type: 'select', value: 'xyz',
          options: ['xyz', 'extxyz', 'cif', 'poscar', 'pdb', 'lammps-data'] }]);
      if (!v) return;
      const path = await dialogs.choosePathToSave({
        title: 'Export structure',
        filename: `structure.${v.format === 'lammps-data' ? 'data'
          : v.format === 'poscar' ? 'vasp' : v.format}` });
      if (!path) return;
      const r = await api.exportStructure(path, v.format);
      pushWarning(`Structure written to ${r.path}`, 'info', 'export');
      return;
    }
    case 'export.image': {
      if (!state.scan) { pushWarning('No scan to export.', 'warning', 'export'); return; }
      const v = await dialogs.form('Export scan image', [
        { name: 'channel', label: 'channel', type: 'select',
          value: state.scan.channel, options: state.scan.channels },
        { name: 'palette', label: 'palette', type: 'select', value: state.palette,
          options: ['silver', 'gray', 'gold', 'copper', 'viridis', 'high-contrast'] }],
        { note: 'PNG carries the provenance summary in a text chunk. Use a .tif ' +
          'extension for a 16-bit export that preserves the full dynamic range.' });
      if (!v) return;
      const path = await dialogs.choosePathToSave({
        title: 'Export scan image', filename: `${state.scan.technique.toLowerCase()}_scan.png` });
      if (!path) return;
      const r = await api.exportImage(path, v.channel, v.palette, state.scan.key);
      pushWarning(`Image written to ${r.path}`, 'info', 'export');
      return;
    }
    case 'export.csv':
    case 'export.csvscan': {
      const what = action === 'export.csv' ? 'atoms' : 'scan';
      const path = await dialogs.choosePathToSave({
        title: `Export ${what} table`, filename: `${what}.csv` });
      if (!path) return;
      const r = await api.exportCsv(path, what);
      pushWarning(`CSV written to ${r.path}`, 'info', 'export');
      return;
    }

    case 'export.npz': {
      const answer = await dialogs.form('Export arrays (NumPy)', [
        { name: 'structure', label: 'active structure', type: 'checkbox', value: true },
        { name: 'arrays', label: 'stored arrays (grids, spectra, maps)', type: 'checkbox',
          value: true },
        { name: 'results', label: 'numeric results and histories', type: 'checkbox',
          value: true },
        { name: 'scans', label: 'scan channels', type: 'checkbox', value: true },
      ], { okLabel: 'Choose file', note: 'One .npz archive that loads with '
        + 'numpy.load(path, allow_pickle=False). A manifest entry records every unit, '
        + 'origin and SHA-256; values that are not numeric are listed as skipped.' });
      if (!answer) return;
      const parts = ['structure', 'arrays', 'results', 'scans'].filter((k) => answer[k]);
      if (!parts.length) { pushWarning('Nothing selected to export.', 'warning', 'export'); return; }
      const path = await dialogs.choosePathToSave({
        title: 'Export arrays (NumPy .npz)', filename: 'materia-arrays.npz' });
      if (!path) return;
      const r = await api.exportNpz(path, parts);
      if (r.ok === false) { pushWarning(r.error, 'error', 'export'); return; }
      const skipped = Object.keys(r.skipped || {}).length;
      pushWarning(`${r.entries} arrays written to ${r.path} and verified`
        + (skipped ? `; ${skipped} non-numeric values skipped, listed in the manifest` : ''),
        'info', 'export');
      return;
    }

    case 'edit.undo': {
      const r = await api.undo();
      pushWarning(`Undone: ${r.label}`, 'info', 'edit');
      await refreshState(); render();
      return;
    }
    case 'edit.redo': {
      const r = await api.redo();
      pushWarning(`Redone: ${r.label}`, 'info', 'edit');
      await refreshState(); render();
      return;
    }
    case 'select.all': case 'select.none': case 'select.invert':
      await selectionActions.select(action.split('.')[1]);
      return;
    case 'edit.substitute': {
      const v = await dialogs.form('Substitute element',
        [{ name: 'element', label: 'element', value: 'P' }],
        { note: 'Applied to every selected atom. Substitutional dopants keep the ' +
          'host site and its id.' });
      if (!v) return;
      await measureActions.edit('substitute', { element: v.element });
      return;
    }
    case 'edit.vacancy':
      await measureActions.edit('vacancy', {});
      return;
    case 'edit.adatom': {
      const v = await dialogs.form('Add adatom', [
        { name: 'element', label: 'element', value: 'Si' },
        { name: 'x', label: 'x / Å', type: 'number', value: num(state.tip.x, 4) },
        { name: 'y', label: 'y / Å', type: 'number', value: num(state.tip.y, 4) },
        { name: 'height_A', label: 'height / Å', type: 'number', value: '', step: 0.1,
          hint: 'Leave empty to place the adatom at the covalent bond length to the nearest surface atom.' }]);
      if (!v) return;
      await api.edit('adatom', { element: v.element, xy: [v.x, v.y],
        height_A: Number.isFinite(v.height_A) ? v.height_A : null });
      await refreshState(); render();
      return;
    }
    case 'edit.interstitial': {
      const v = await dialogs.form('Add interstitial', [
        { name: 'element', label: 'element', value: 'Si' },
        { name: 'x', label: 'x / Å', type: 'number', value: 0 },
        { name: 'y', label: 'y / Å', type: 'number', value: 0 },
        { name: 'z', label: 'z / Å', type: 'number', value: 0 }],
        { note: 'The insertion is refused if it overlaps an existing atom, which ' +
          'would be an unphysical configuration rather than a defect.' });
      if (!v) return;
      await api.edit('interstitial', { element: v.element, position: [v.x, v.y, v.z] });
      await refreshState(); render();
      return;
    }
    case 'edit.charge': {
      const v = await dialogs.form('Set formal charge',
        [{ name: 'charge', label: 'charge / e', type: 'number', value: 1, step: 1 }],
        { note: 'Classical potentials ignore charge entirely, and the tight-binding ' +
          'model is restricted to neutral systems. A charged calculation needs a ' +
          'self-consistent solver with a compensating background.', danger: true });
      if (!v) return;
      await measureActions.edit('set_charge', { charge: v.charge });
      return;
    }
    case 'edit.spin': {
      const v = await dialogs.form('Set spin',
        [{ name: 'spin', label: 'spin S', type: 'number', value: 0.5, step: 0.5 }],
        { note: 'Stored as a magnetic moment of 2S µʙ (spin-only). The ' +
          'shipped tight-binding model is spin-restricted and will ignore it.' });
      if (!v) return;
      await measureActions.edit('set_spin', { spin: v.spin });
      return;
    }
    case 'edit.strain': {
      const v = await dialogs.form('Apply homogeneous strain', [
        { name: 'exx', label: 'εₓₓ', type: 'number', value: 0.01, step: 0.005 },
        { name: 'eyy', label: 'εᵧᵧ', type: 'number', value: 0.01, step: 0.005 },
        { name: 'ezz', label: 'εᶻᶻ', type: 'number', value: 0, step: 0.005 }]);
      if (!v) return;
      await api.edit('strain', { strain: [v.exx, v.eyy, v.ezz] });
      await refreshState(); render();
      return;
    }
    case 'edit.passivate':
      await api.edit('passivate', { element: 'H', side: 'bottom' });
      await refreshState(); render();
      return;

    case 'build.wafer': {
      const mats = (state.materialList || []).map((m) => ({ value: m.id, label: m.name }));
      const v = await dialogs.form('Create wafer', [
        { name: 'material_id', label: 'material', type: 'select',
          value: arg || 'silicon', options: mats },
        { name: 'diameter_mm', label: 'diameter / mm', type: 'number', value: 300, step: 25 },
        { name: 'orientation', label: 'orientation (hkl)', value: '111' },
        { name: 'edge_feature', label: 'edge', type: 'select', value: 'notch',
          options: ['notch', 'flat', 'none'] },
        { name: 'miscut_deg', label: 'miscut / °', type: 'number', value: 0, step: 0.1 },
        { type: 'section', label: 'Doping and defects' },
        { name: 'dopant', label: 'dopant', value: 'P' },
        { name: 'dopant_concentration_cm3', label: 'concentration / cm⁻³',
          type: 'number', value: 1e19, step: 1e18, wide: true },
        { name: 'roughness_rms_A', label: 'roughness / Å rms', type: 'number',
          value: 0.8, step: 0.1 },
        { name: 'vacancy_density_cm2', label: 'vacancies / cm⁻²', type: 'number',
          value: 1e12, step: 1e11, wide: true },
        { name: 'temperature_K', label: 'temperature / K', type: 'number', value: 300, step: 10 },
        { name: 'seed', label: 'procedural seed', type: 'number', value: 20260920, step: 1,
          wide: true }],
        { note: 'The wafer is defined procedurally: a 300 mm wafer holds of order ' +
          '10<sup>24</sup> atoms and is never instantiated. Atoms are generated only ' +
          'for the region you select.' });
      if (!v) return;
      setStatusJob('creating wafer', true);
      await api.createWafer({ ...v,
        orientation: String(v.orientation).split('').map(Number) });
      await refreshState();
      setView('wafer');
      await waferView.loadMap('roughness');
      await regionView.loadMap('roughness');
      render();
      return;
    }
    case 'build.region': {
      if (!s.wafer) { pushWarning('Create a wafer first.', 'warning', 'build'); return; }
      const v = await dialogs.form('Extract atomistic region', [
        { name: 'x_mm', label: 'x / mm', type: 'number', value: num(waferView.roi.x, 5), step: 0.1 },
        { name: 'y_mm', label: 'y / mm', type: 'number', value: num(waferView.roi.y, 5), step: 0.1 },
        { name: 'size_x', label: 'width / nm', type: 'number', value: 3.0, step: 0.5 },
        { name: 'size_y', label: 'depth / nm', type: 'number', value: 3.0, step: 0.5 },
        { name: 'depth_layers', label: 'stacking repeats', type: 'number', value: 5, step: 1 },
        { name: 'vacuum_A', label: 'vacuum / Å', type: 'number', value: 14, step: 1 },
        { name: 'include_defects', label: 'include defects', type: 'checkbox', value: true },
        { name: 'include_dopants', label: 'include dopants', type: 'checkbox', value: true },
        { name: 'max_atoms', label: 'atom budget', type: 'number', value: 8000, step: 500,
          wide: true }],
        { note: 'Defects and dopants are drawn from a Poisson process at the wafer’s ' +
          'densities, seeded by the wafer seed and this coordinate, so the same ' +
          'coordinate always regenerates the same region.' });
      if (!v) return;
      setStatusJob('generating atoms', true);
      set({ roi: { x: v.x_mm, y: v.y_mm } });
      const r = await api.extractRegion({
        x_mm: v.x_mm, y_mm: v.y_mm, size_nm: [v.size_x, v.size_y],
        depth_layers: v.depth_layers, vacuum_A: v.vacuum_A,
        include_defects: v.include_defects, include_dopants: v.include_dopants,
        max_atoms: v.max_atoms });
      await refreshState();
      setView('atoms');
      atomsView.frame();
      render();
      pushWarning(`Region ${r.region_id}: ${r.n_atoms} atoms instantiated.`, 'info', 'build');
      return;
    }
    case 'build.surface': {
      document.querySelector('#left-dock .dock-tab[data-panel="materials"]').click();
      return;
    }

    case 'structure.reconstruct': {
      if (!s.structure) { pushWarning('No structure to reconstruct.', 'warning', 'structure'); return; }
      if (s.structure.reconstruction) {
        await dialogs.info('Already reconstructed', el('div.note', { text:
          `This slab carries the ${s.structure.reconstruction.id} reconstruction. `
          + 'Build a new surface to apply a different one.' }));
        return;
      }
      const report = await api.reconstructionReport();
      const declared = report.declared || [];
      if (!declared.length) {
        await dialogs.info('No reconstructions declared', el('div.note', { text:
          'The material definition for this surface declares no reconstruction for '
          + 'this orientation.' }));
        return;
      }
      const options = declared.map((r) => ({ value: r.id,
        label: `${r.id} (${r.orientation.join('')})${r.supported ? '' : ' - not implemented'}` }));
      const answer = await dialogs.form('Reconstruct surface', [
        { name: 'reconstruction', label: 'reconstruction', type: 'select', options,
          value: (declared.find((r) => r.supported) || declared[0]).id },
        { name: 'relax', label: 'relax after generating', type: 'checkbox', value: true,
          hint: 'The generator sets the pairing only. Without relaxation the geometry '
                + 'is an unrelaxed construction guess and is labelled estimated.' },
      ], { okLabel: 'Apply', note:
        'Applied to the upper face. A reconstruction this build cannot generate is '
        + 'refused rather than approximated.' });
      if (!answer) return;
      setStatusJob('reconstructing surface', true);
      try {
        const out = await api.reconstruct({ reconstruction: answer.reconstruction,
          relax: answer.relax });
        await refreshState();
        render();
        if (out.ok === false) { showUnsupported(out, answer.reconstruction); return; }
        showComparison(out.comparison);
      } finally { setStatusJob('Idle', false); }
      return;
    }

    case 'structure.reconstruction': {
      if (!s.structure) { pushWarning('No structure loaded.', 'warning', 'structure'); return; }
      await materialActions.reconstructionReport();
      return;
    }

    case 'scan.run': {
      if (!s.structure) { pushWarning('No structure to scan.', 'warning', 'scan'); return; }
      const technique = $('#tb-technique').value;
      const settings = panels.readInstrumentSettings(technique);
      setStatusJob(`${technique.toUpperCase()} scanning`, true);
      $('#lamp-feedback').classList.add('on-blue');
      try {
        const result = await api.scan(technique, settings, {
          onJobStart: (job) => { currentJob = job; },
          onProgress: (job) => setStatusJob(
            `${technique.toUpperCase()} · ${job.message || job.status}`, true),
        });
        if (!result.supported) {
          $('#lamp-feedback').classList.remove('on-blue');
          await dialogs.info('This measurement is not supported', el('div', {}, [
            el('div.note.blocked', { text: result.reason }),
            el('div.note', { html: '<b>Solvers that could produce it:</b><br>' +
              (result.suggested_models || []).join('<br>') }),
            el('div.note', { text:
              'The requested configuration has been kept, so it can be run unchanged ' +
              'once one of those solvers is installed.' })]));
          return;
        }
        set({ scan: result, scanFeatures: null, channel: result.channel });
        probeView.setScan(result);
        setView('probe');
        await refreshState();
        render();
      } finally {
        $('#lamp-feedback').classList.remove('on-blue');
        currentJob = null;
      }
      return;
    }
    case 'scan.abort': {
      if (currentJob) { await api.cancelJob(currentJob.id); pushWarning('Abort requested.', 'info', 'scan'); }
      return;
    }
    case 'scan.features': {
      if (!state.scan) { pushWarning('Acquire a scan first.', 'warning', 'scan'); return; }
      const f = await api.scanFeatures(state.scan.key);
      set({ scanFeatures: f, showFeatures: true });
      probeView.draw();
      const counts = {};
      for (const feat of f.features) counts[feat.kind] = (counts[feat.kind] || 0) + 1;
      pushWarning(`${f.features.length} features: ` +
        Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(', '), 'info', 'scan');
      return;
    }
    case 'scan.sts': {
      setStatusJob('point spectroscopy', true);
      const r = await api.spectroscopy({ x_A: state.tip.x, y_A: state.tip.y, height_A: 5.0 });
      if (!r.supported) { pushWarning(r.reason, 'error', 'sts'); return; }
      set({ sts: r, plot: 'sts' });
      selectPlot('sts');
      return;
    }
    case 'scan.force': {
      setStatusJob('force curve', true);
      const r = await api.forceCurve(state.tip.x, state.tip.y);
      if (!r.supported) { pushWarning(r.reason, 'error', 'force'); return; }
      set({ forceCurve: r, plot: 'force' });
      selectPlot('force');
      return;
    }

    case 'solve.energy': case 'solve.relax': case 'solve.md':
    case 'solve.electronic': case 'solve.band_structure': {
      const task = action.split('.')[1];
      if (!s.structure) { pushWarning('No structure.', 'warning', 'solve'); return; }
      const model = (document.getElementById('solver-model') || {}).value || 'recommended';
      const extra = {};
      if (task === 'relax') {
        extra.fmax = parseFloat(($('#solver-fmax') || {}).value || 0.02);
        extra.steps = parseInt(($('#solver-steps') || {}).value || 300, 10);
      } else if (task === 'md') {
        extra.steps = parseInt(($('#solver-mdsteps') || {}).value || 200, 10);
        extra.temperature_K = parseFloat(($('#solver-temp') || {}).value || 300);
        extra.thermostat = ($('#solver-thermostat') || {}).value || 'none';
        extra.seed = parseInt(($('#solver-seed') || {}).value || 0, 10);
      } else if (task === 'electronic') {
        const sc = document.getElementById('solver-sc');
        extra.self_consistent = !!(sc && sc.checked);
      }
      setStatusJob(`${task} running`, true);
      const result = await api.solve(task, model, extra, {
        onJobStart: (job) => { currentJob = job; },
        onProgress: (job) => setStatusJob(`${task} · ${job.message || job.status}`, true),
      });
      currentJob = null;
      if (!result.supported) {
        await dialogs.info('This calculation is not supported', el('div', {}, [
          el('div.note.blocked', { text: result.reason }),
          (result.suggested_models || []).length
            ? el('div.note', { html: '<b>Solvers that could produce it:</b><br>' +
                result.suggested_models.join('<br>') })
            : null,
          el('div.note', { text: 'Nothing has been fabricated: the requested ' +
            'configuration is preserved and can be run when a capable solver is ' +
            'available.' })]));
        return;
      }
      set({ lastSolver: result.solver });
      await applySolverResult(task, result);
      await refreshState();
      render();
      return;
    }

    case 'view.wafer': setView('wafer'); return;
    case 'view.region': setView('region'); return;
    case 'view.atoms': setView('atoms'); return;
    case 'view.probe': setView('probe'); return;
    case 'view.frame': atomsView.frame(); return;
    case 'view.bonds': atomsView.showBonds = !atomsView.showBonds; atomsView.draw(); return;
    case 'view.cell': atomsView.showCell = !atomsView.showCell; atomsView.draw(); return;
    case 'theme.light':
      document.documentElement.dataset.theme = 'light';
      document.documentElement.removeAttribute('data-contrast');
      redrawAll(); return;
    case 'theme.dark':
      document.documentElement.dataset.theme = 'dark';
      document.documentElement.removeAttribute('data-contrast');
      redrawAll(); return;
    case 'theme.contrast':
      document.documentElement.dataset.contrast =
        document.documentElement.dataset.contrast === 'high' ? '' : 'high';
      redrawAll(); return;
    case 'density.compact': case 'density.normal': case 'density.comfortable':
      document.documentElement.dataset.density = action.split('.')[1];
      redrawAll(); return;

    case 'panel.materials':
      document.querySelector('#left-dock .dock-tab[data-panel="materials"]').click(); return;
    case 'panel.instrument':
      document.querySelector('#right-dock .dock-tab[data-panel="instrument"]').click(); return;
    case 'panel.solver':
      document.querySelector('#right-dock .dock-tab[data-panel="solver"]').click(); return;
    case 'panel.console': case 'panel.log': case 'panel.warnings': case 'panel.history':
      document.querySelector(
        `#bottom-dock .dock-tab[data-panel="${action.split('.')[1]}"]`).click(); return;

    case 'tools.periodic': {
      const { elements } = await api.periodicTable();
      await dialogs.info('Periodic table', periodicTableNode(elements), { wide: true });
      return;
    }
    case 'tools.environment': {
      const env = await api.scriptEnvironment();
      await dialogs.info('Python environment', el('div', {}, [
        el('div.note', { html: `<b>Mode:</b> ${env.mode}. Restricted mode limits ` +
          'imports and builtins. It is a guard rail against accidents, not a ' +
          'security boundary: CPython cannot be sandboxed from inside.' }),
        el('div.section-title', {}, el('span', { text: 'Names in the namespace' })),
        el('pre', { style: { whiteSpace: 'pre-wrap', fontSize: '11px' },
          text: env.namespace.join('  ') }),
        el('div.section-title', {}, el('span', { text: 'Importable modules' })),
        el('pre', { style: { whiteSpace: 'pre-wrap', fontSize: '11px' },
          text: env.allowed_modules.join('  ') })]), { wide: true });
      return;
    }

    case 'help.keys': {
      const t = el('table.pgrid');
      for (const [k, d] of SHORTCUTS) {
        t.append(el('tr', {}, [el('th', { text: k }), el('td.txt', { text: d })]));
      }
      await dialogs.info('Keyboard shortcuts', t);
      return;
    }
    case 'help.selfcheck': {
      const report = await runInterfaceSelfCheck();
      const table = el('table.pgrid');
      for (const check of report.checks) {
        table.append(el('tr', {}, [
          el('th', { text: check.passed ? 'PASS' : 'FAIL' }),
          el('td', { text: check.name }),
          el('td.mono', { text: check.detail }),
        ]));
      }
      await dialogs.info(report.ok ? 'Interface self-check passed' : 'Interface self-check failed',
        table, { wide: true });
      return;
    }
    case 'help.tiers': {
      await dialogs.info('Fidelity tiers', el('div', {}, [
        tierNote('Tier 0 - structural', 'tier0-structural',
          'Crystallography and geometry: lattices, surfaces, neighbour lists, bond ' +
          'perception, coordinate transforms. Exact within its own definitions; ' +
          'contains no physics beyond geometry.'),
        tierNote('Tier 1 - classical', 'tier1-classical',
          'Empirical interatomic potentials: Stillinger-Weber, Lennard-Jones. ' +
          'Energies, forces, relaxation and molecular dynamics with no electrons. ' +
          'Good for geometry and thermal motion; not for anything electronic.'),
        tierNote('Tier 2 - semi-empirical', 'tier2-semi-empirical',
          'Orthogonal tight binding fitted to experimental band structures. Gives ' +
          'band structures, densities of states, site-projected LDOS and the ' +
          'Tersoff-Hamann STM signal. Not self-consistent: no charge transfer, no ' +
          'band bending, no response to an applied field.'),
        tierNote('Tier 3 - external first principles', 'tier3-external-first-principles',
          'Adapters for GPAW, Quantum ESPRESSO, LAMMPS, CP2K, PySCF, Psi4, ' +
          'Wannier90 and OpenMX. Materia does not bundle them; when one is ' +
          'installed it appears as a solver, and when it is not, requests that ' +
          'need it are declined with installation instructions.'),
        el('div.note.warn', { text:
          'A general exact simulation of an arbitrary many-body quantum system is ' +
          'not computationally possible at useful scales. Materia is explicit ' +
          'about which approximation produced every number it shows.' })]), { wide: true });
      return;
    }
    case 'help.limits': {
      await dialogs.info('Known limitations', el('div', { html: `
        <ul style="line-height:1.7;padding-left:18px">
        <li>No solver here is a first-principles calculation. Tier 2 tight binding
            is fitted to experiment; it reproduces band structures it was fitted to
            and is not predictive for new systems.</li>
        <li>The STM model reports current in arbitrary units. The Tersoff-Hamann
            prefactor contains the unknown tip density of states, so absolute
            currents are not predicted; the setpoint is calibrated against the
            computed signal, as a real feedback loop is.</li>
        <li>One surface reconstruction is generated end to end: Si(100)-2×1,
            whose dimer pairing is derived from the slab and whose geometry is
            the relaxed minimum of the recommended potential. It comes out
            <b>symmetric</b> at 2.4035 Å, against a buckled 2.24 ± 0.08 Å in
            LEED, because a classical potential has no mechanism for the
            charge-transfer buckling. The rest - Si(111)-7×7, the Au(111)
            herringbone - are declared, <b>not generated</b>, and are
            refused with a citation rather than approximated.</li>
        <li>AFM contrast comes from a Lennard-Jones plus Hamaker force model. It has
            no covalent tip-sample bonding, which dominates real atomic contrast on
            semiconductors.</li>
        <li>The wafer microstructure is synthetic and seeded. It is reproducible and
            statistically plausible; it is not a measurement.</li>
        <li>Restricted Python mode is a guard rail, not a security sandbox.</li>
        <li>Charged and spin-polarised systems are accepted as inputs but the shipped
            solvers decline to compute them.</li>
        </ul>` }), { wide: true });
      return;
    }
    case 'help.about': {
      await dialogs.info('About Materia', el('div', { html: `
        <p><b>Materia ${s.version}</b></p>
        <p>Atomic-scale wafer exploration and scanning-probe simulation.
        MIT. Runs entirely on this machine: there is no cloud service and
        no telemetry.</p>
        <p>${(state.materialList || []).length} material definitions ·
        ${state.solvers ? state.solvers.registered.length : 0} registered solvers.</p>
        <p class="hint">Every number this program shows carries a provenance record
        naming the model that produced it and the approximations it makes.</p>` }));
      return;
    }
    default:
      pushWarning(`No handler for action ${action}`, 'warning', 'ui');
  }
}

function tierNote(title, cls, text) {
  return el('div', { style: { margin: '6px 0' } }, [
    el('div', {}, [tierTag(cls), ' ', el('b', { text: title })]),
    el('div.hint', { text, style: { marginTop: '3px' } })]);
}

function periodicTableNode(elements) {
  const wrap = el('div', { style: { display: 'grid',
    gridTemplateColumns: 'repeat(18, 1fr)', gap: '2px', fontSize: '9px' } });
  const byPos = new Map();
  for (const e of elements) {
    let row = e.period, col = e.group;
    if (e.group === 0) { row = e.number >= 89 ? 10 : 9; col = ((e.number - (e.number >= 89 ? 89 : 57)) % 15) + 3; }
    byPos.set(`${row}:${col}`, e);
  }
  for (let row = 1; row <= 10; row++) {
    for (let col = 1; col <= 18; col++) {
      const e = byPos.get(`${row}:${col}`);
      if (!e) { wrap.append(el('div')); continue; }
      wrap.append(el('div', {
        title: `${e.name} · Z=${e.number} · ${e.category}`,
        style: { border: '1px solid var(--border)', padding: '2px',
          textAlign: 'center', background: 'var(--panel-alt)', cursor: 'default' },
      }, [el('div.readout', { text: String(e.number), style: { fontSize: '7px' } }),
          el('div', { text: e.symbol, style: { fontWeight: '600' } })]));
    }
  }
  return wrap;
}

function selectPlot(name) {
  $$('#secondary .view-tab').forEach((t) => {
    const on = t.dataset.plot === name;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', String(on));
  });
  set({ plot: name });
  plotPane.draw();
}

function potentialCheckLines(checks) {
  const lines = [];
  const parts = checks.components_eV || {};
  if (parts.short_range !== undefined && parts.coulomb !== undefined) {
    lines.push(`  short range = ${num(parts.short_range, 8)} eV · ` +
      `Coulomb = ${num(parts.coulomb, 8)} eV (${checks.geometry})`);
  }
  if (checks.ewald_message) {
    lines.push(`  Coulomb sum ${checks.ewald_converged ? 'confirmed' : 'NOT confirmed'}: ` +
      checks.ewald_message);
  }
  const margins = checks.barrier_margin_A || {};
  for (const [pair, distance] of Object.entries(checks.closest_pair_A || {})) {
    const margin = margins[pair];
    lines.push(`  closest ${pair} = ${num(distance, 5)} A` + (margin === undefined ? '' :
      ` · ${num(margin, 4)} A outside the collapse barrier`));
  }
  return lines;
}

async function applySolverResult(task, result) {
  const out = $('#numeric-output');
  const stamp = new Date().toLocaleTimeString();
  const lines = [`[${stamp}] ${task} · ${result.solver}`];
  if (task === 'energy') {
    lines.push(`  E = ${num(result.energy_eV, 8)} eV ` +
      `(${num(result.energy_per_atom_eV, 6)} eV/atom)`);
    lines.push(`  max |F| = ${num(result.max_force_eV_A, 6)} eV/A`);
  } else if (task === 'relax') {
    lines.push(`  converged: ${result.converged} · ${result.convergence.message}`);
    lines.push(`  E = ${num(result.energy_eV, 8)} eV (change ${num(result.energy_change_eV, 6)} eV)`);
    lines.push(`  max displacement = ${num(result.max_displacement_A, 5)} A`);
    set({ relaxHistory: result.history });
    selectPlot('relax');
  } else if (task === 'md') {
    lines.push(`  mean T = ${num(result.mean_temperature_K, 5)} K`);
    const t = result.trajectory;
    set({ relaxHistory: { step: t.step, fmax_eV_A: t.temperature_K } });
  } else if (task === 'electronic') {
    set({ electronic: result });
    lines.push(`  E_F = ${num(result.fermi_level_eV, 6)} eV · ` +
      `${result.eigenvalues.length} states`);
    if (result.hl_gap_eV !== undefined) {
      lines.push(`  HOMO-LUMO gap = ${num(result.hl_gap_eV, 6)} eV`);
    }
    for (const w of result.log || []) pushWarning(w, 'info', result.solver);
    if (result.self_consistency && !result.self_consistency.supported) {
      await dialogs.info('Self-consistency is not available in this model',
        el('div', {}, [
          el('div.note.blocked', { text: result.self_consistency.reason }),
          el('div.note', { html: '<b>Solvers that could do it:</b><br>' +
            (result.self_consistency.suggested_models || []).join('<br>') })]));
    }
    selectPlot('dos');
  } else if (task === 'band_structure') {
    set({ bands: result });
    lines.push(`  gap = ${num(result.band_gap_eV, 6)} eV ` +
      `(${result.band_gap_note.direct ? 'direct' : 'indirect'})`);
    lines.push(`  VBM = ${num(result.band_gap_note.vbm_eV, 6)} eV, ` +
      `CBM = ${num(result.band_gap_note.cbm_eV, 6)} eV`);
    selectPlot('bands');
  }
  if (result.checks) lines.push(...potentialCheckLines(result.checks));
  for (const w of (task === 'electronic' ? [] : result.log || [])) {
    pushWarning(w, 'info', result.solver);
  }
  const p = result.provenance || {};
  lines.push(`  model: ${p.model} [${p.fidelity}, ${p.origin}]`);
  for (const a of (p.approximations || []).slice(0, 4)) lines.push(`    - ${a}`);
  out.append(el('pre', { style: { margin: '0 0 6px 0' }, text: lines.join('\n') }));
  out.scrollTop = out.scrollHeight;
  if (state.atom) await inspector.load(state.atom.identity.id);
}

function redrawAll() {
  waferView.resize(); regionView.resize(); atomsView.resize();
  probeView.resize(); plotPane.resize();
}

boot().catch((err) => {
  document.body.innerHTML =
    `<pre style="padding:20px;font-family:monospace;white-space:pre-wrap">` +
    `Materia failed to start:\n\n${err && err.stack ? err.stack : err}</pre>`;
});
