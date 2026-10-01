/**
 * Client for the local Materia service.
 * Every call returns the `data` payload or throws an Error carrying the
 * server's message, which callers surface verbatim rather than paraphrasing.
 * @module api
 */

const BASE = '/api/';

/** @type {Set<(e:{type:string,detail:any})=>void>} */
const listeners = new Set();
export function onApiEvent(fn) { listeners.add(fn); return () => listeners.delete(fn); }
function emit(type, detail) { for (const fn of listeners) fn({ type, detail }); }

export class ApiError extends Error {
  constructor(message, payload, status) {
    super(message);
    this.name = 'ApiError';
    this.payload = payload || {};
    this.status = status;
  }
}

/**
 * POST a JSON body to an endpoint.
 * @param {string} route
 * @param {object} [body]
 */
export async function call(route, body = {}) {
  let response;
  try {
    response = await fetch(BASE + route, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    emit('offline', { route });
    throw new ApiError(`Cannot reach the Materia service (${err.message}). ` +
      'Is the server still running?', {}, 0);
  }
  let payload;
  try { payload = await response.json(); }
  catch { throw new ApiError(`Malformed response from /${route}`, {}, response.status); }
  if (!response.ok || payload.ok === false) {
    const message = payload.error || `Request to /${route} failed (${response.status})`;
    emit('error', { route, message, payload });
    throw new ApiError(message, payload, response.status);
  }
  return payload.data;
}

/** Poll a background job until it finishes. */
export async function waitForJob(jobId, { onProgress, intervalMs = 220, signal } = {}) {
  for (;;) {
    if (signal && signal.aborted) throw new ApiError('Cancelled', {}, 0);
    const job = await call('job', { id: jobId });
    if (onProgress) onProgress(job);
    if (job.status === 'done') return job.result;
    if (job.status === 'failed') {
      throw new ApiError(job.error || 'Job failed', { traceback: job.traceback }, 500);
    }
    if (job.status === 'cancelled') throw new ApiError('Job cancelled by the user', {}, 0);
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

/**
 * Run a request that may be dispatched as a background job and wait for it.
 * @param {string} route @param {object} body @param {{onProgress?:Function}} [opts]
 */
export async function callJob(route, body = {}, opts = {}) {
  const result = await call(route, body);
  if (result && result.job && result.job.id) {
    if (opts.onJobStart) opts.onJobStart(result.job);
    return waitForJob(result.job.id, opts);
  }
  return result;
}

export const api = {
  state: () => call('state'),
  materials: () => call('materials'),
  material: (id) => call('material', { id }),
  element: (symbol) => call('element', { symbol }),
  periodicTable: () => call('periodic_table'),

  createWafer: (spec) => call('wafer/create', spec),
  waferProbe: (x_mm, y_mm) => call('wafer/probe', { x_mm, y_mm }),
  waferMap: (field, n) => call('wafer/map', { field, n }),
  extractRegion: (spec) => call('region/extract', spec),
  buildSurface: (spec) => call('structure/build', spec),
  gpawStatus: (refresh) => call('gpaw/status', { refresh: !!refresh }),
  dftStatus: () => call('dft/status', {}),
  dftSpec: (variables) => call('dft/spec', { variables: variables || {} }),
  dftRun: (variables, options, opts) => callJob('dft/run',
    { variables: variables || {}, background: true, ...(options || {}) }, opts),
  dftResult: (runId) => call('dft/result', runId ? { run_id: runId } : {}),
  dftArray: (runId, name, axis) => call('dft/array', { run_id: runId, name, axis }),
  dftStudy: (request, variables, opts) => callJob('dft/study',
    { ...(request || {}), variables: variables || {}, background: true }, opts),
  dftStudyResult: (studyId) => call('dft/study/result', studyId ? { study_id: studyId } : {}),
  dftRelaxSpec: (variables) => call('dft/relax/spec', { variables: variables || {} }),
  dftRelaxRun: (variables, opts) => callJob('dft/relax/run',
    { variables: variables || {}, background: true }, opts),
  dftRelaxResult: (runId) => call('dft/relax/result', runId ? { run_id: runId } : {}),
  dftRelaxApply: (runId) => call('dft/relax/apply', runId ? { run_id: runId } : {}),
  dftDosSpec: (variables, source) => call('dft/dos/spec', { variables: variables || {}, source: source || null }),
  dftDosRun: (variables, source, opts) => callJob('dft/dos/run',
    { variables: variables || {}, source: source || null, background: true }, opts),
  dftDosResult: (runId) => call('dft/dos/result', runId ? { run_id: runId } : {}),
  dftDosExport: (path, runId) => call('dft/dos/export', { path, run_id: runId }),
  dftBandsSpec: (variables, source) => call('dft/bands/spec', { variables: variables || {}, source: source || null }),
  dftBandsRun: (variables, source, opts) => callJob('dft/bands/run',
    { variables: variables || {}, source: source || null, background: true }, opts),
  dftBandsResult: (runId) => call('dft/bands/result', runId ? { run_id: runId } : {}),
  dftBandsExport: (path, runId) => call('dft/bands/export', { path, run_id: runId }),
  dftEosSpec: (variables, source) => call('dft/eos/spec', { variables: variables || {}, source: source || null }),
  dftEosRun: (variables, source, opts) => callJob('dft/eos/run',
    { variables: variables || {}, source: source || null, background: true }, opts),
  dftEosResult: (runId) => call('dft/eos/result', runId ? { run_id: runId } : {}),
  dftEosExport: (path, runId) => call('dft/eos/export', { path, run_id: runId }),
  dftEosApply: (runId) => call('dft/eos/apply', runId ? { run_id: runId } : {}),
  dftLdosSpec: (variables, source) => call('dft/ldos/spec', { variables: variables || {}, source: source || null }),
  dftLdosRun: (variables, source, opts) => callJob('dft/ldos/run',
    { variables: variables || {}, source: source || null, background: true }, opts),
  dftLdosResult: (runId) => call('dft/ldos/result', runId ? { run_id: runId } : {}),
  dftLdosSlice: (runId, index) => call('dft/ldos/slice', { run_id: runId, index }),
  dftLdosImage: (runId, request) => call('dft/ldos/image', { run_id: runId, ...(request || {}) }),
  dftLdosExport: (path, runId) => call('dft/ldos/export', { path, run_id: runId }),
  dftLdosImageExport: (path, runId, request) => call('dft/ldos/image/export', { path, run_id: runId, ...(request || {}) }),
  claims: () => call('claims/list', {}),
  eamStatus: () => call('eam/status', {}),
  eamRun: (spec, opts) => callJob('eam/run', { background: true, ...(spec || {}) }, opts),
  eamResult: (runId) => call('eam/result', runId ? { run_id: runId } : {}),
  eamTrajectory: (runId, frame) => call('eam/trajectory', {
    ...(runId ? { run_id: runId } : {}), ...(frame === undefined ? {} : { frame }) }),
  lammpsStatus: (refresh) => call('lammps/status', { refresh: !!refresh }),
  lammpsSpec: (spec) => call('lammps/spec', spec || {}),
  lammpsRun: (spec, opts) => callJob('lammps/run', { background: true, ...(spec || {}) }, opts),
  lammpsResult: (runId) => call('lammps/result', runId ? { run_id: runId } : {}),
  lammpsApply: (runId) => call('lammps/apply', runId ? { run_id: runId } : {}),
  electrostaticsStatus: (settings) => call('electrostatics/status', { settings: settings || {} }),
  electrostaticsAssign: (spec) => call('electrostatics/assign', spec),
  electrostaticsClear: () => call('electrostatics/clear', {}),
  electrostaticsRun: (settings, opts) =>
    callJob('electrostatics/run', { settings: settings || {}, background: true }, opts),
  electrostaticsResult: (runId) => call('electrostatics/result', runId ? { run_id: runId } : {}),
  reconstruct: (spec) => call('structure/reconstruct', spec),
  reconstructionReport: () => call('structure/reconstruction', {}),
  render: (opts) => call('structure/render', opts || {}),
  importStructure: (path) => call('structure/import', { path }),

  select: (mode, opts) => call('select', { mode, ...(opts || {}) }),
  inspectAtom: (atom_id) => call('atom/inspect', { atom_id }),
  atomLdos: (atom_id) => call('atom/ldos', { atom_id }),
  edit: (op, opts) => call('edit', { op, ...(opts || {}) }),

  scan: (technique, settings, opts) =>
    callJob('scan', { technique, settings, background: true }, opts),
  scanPayload: (key, channel) => call('scan/payload', { key, channel }),
  scanFiltered: (key, channel, method) => call('scan/filtered', { key, channel, method }),
  scanFeatures: (key) => call('scan/features', { key }),
  scanIdentify: (x_A, y_A, key) => call('scan/identify', { x_A, y_A, key }),
  scanProfile: (p) => call('scan/profile', p),
  spectroscopy: (p) => call('spectroscopy', p),
  forceCurve: (x_A, y_A) => call('force_curve', { x_A, y_A }),
  instrumentOptions: () => call('instrument/options'),

  solvers: () => call('solvers'),
  solve: (task, model, extra, opts) =>
    callJob('solve', { task, model, background: true, ...(extra || {}) }, opts),

  runScript: (code, opts) => callJob('script/run', { code, background: true }, opts),
  scriptMode: (mode, confirm) => call('script/mode', { mode, confirm }),
  scriptEnvironment: () => call('script/environment'),

  jobs: () => call('jobs'),
  job: (id) => call('job', { id }),
  cancelJob: (id) => call('job/cancel', { id }),

  undo: () => call('project/undo'),
  redo: () => call('project/redo'),
  saveProject: (path) => call('project/save', { path }),
  openProject: (path) => call('project/open', { path }),
  newProject: (name) => call('project/new', { name }),
  checkpoint: (name, note) => call('project/checkpoint', { name, note }),
  restoreCheckpoint: (name) => call('project/restore', { name }),
  provenance: () => call('project/provenance'),
  recoverProject: () => call('project/recover'),
  discardRecovery: () => call('project/recovery/discard'),

  exportStructure: (path, format) => call('export/structure', { path, format }),
  exportImage: (path, channel, palette, key) =>
    call('export/image', { path, channel, palette, key }),
  exportCsv: (path, what) => call('export/csv', { path, what }),
  exportNpz: (path, parts) => call('export/npz', { path, parts }),
};
