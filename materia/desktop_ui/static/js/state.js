/**
 * Application state store.
 * A single observable object; views subscribe to the keys they care about.
 * @module state
 */

const listeners = new Map();

export const state = {
  /** @type {any} */ server: null,
  /** @type {any} */ render: null,
  /** @type {any} */ scan: null,
  /** @type {Float32Array|null} */ scanData: null,
  /** @type {any} */ scanFeatures: null,
  /** @type {any} */ atom: null,
  /** @type {any} */ material: null,
  /** @type {any} */ solvers: null,
  /** @type {any} */ instrumentOptions: null,
  /** @type {any} */ electronic: null,
  /** @type {any} */ bands: null,
  /** @type {any} */ sts: null,
  /** @type {any} */ forceCurve: null,
  /** @type {any} */ relaxHistory: null,
  /** @type {any} */ profile: null,
  /** @type {any} */ waferMap: null,
  /** @type {any} */ waferProbe: null,
  /** @type {any} */ lammpsStatus: null,
  /** @type {any} */ dftRelaxVars: null,
  /** @type {any} */ dftRelaxCheck: null,
  /** @type {any} */ dftRelaxResult: null,
  /** @type {any} */ dftDosVars: null,
  /** @type {any} */ dftDosSource: null,
  /** @type {any} */ dftDosCheck: null,
  /** @type {any} */ dftDosResult: null,
  /** @type {any} */ dftDosView: null,
  /** @type {any} */ dftDosProjections: null,
  /** @type {any} */ dftBandsVars: null,
  /** @type {any} */ dftBandsSource: null,
  /** @type {any} */ dftBandsCheck: null,
  /** @type {any} */ dftBandsResult: null,
  /** @type {any} */ dftBandsView: null,
  /** @type {any} */ dftEosVars: null,
  /** @type {any} */ dftEosSource: null,
  /** @type {any} */ dftEosCheck: null,
  /** @type {any} */ dftEosResult: null,
  /** @type {any} */ dftLdosVars: null,
  /** @type {any} */ dftLdosSource: null,
  /** @type {any} */ dftLdosCheck: null,
  /** @type {any} */ dftLdosResult: null,
  /** @type {any} */ dftLdosSlice: null,
  /** @type {any} */ dftLdosImage: null,
  /** @type {any} */ dftLdosImageRequest: null,
  /** @type {any} */ trajectory: null,
  /** @type {number} */ trajectoryFrame: 0,
  /** @type {boolean} */ trajectoryPlaying: false,
  /** @type {number} */ trajectorySpeed: 1,
  /** @type {{x:number,y:number}|null} */ roi: null,

  view: 'wafer',
  plot: 'profile',
  tool: 'navigate',
  scalePosition: 0,
  palette: 'silver',
  contrast: 1.0,
  brightness: 0.0,
  gamma: 1.0,
  channel: null,
  filterMethod: 'none',
  showOverlay: false,
  showFeatures: false,
  showTip: true,
  tip: { x: 0, y: 0 },
  measurePoints: [],
  hover: null,
  busy: false,
  lastError: null,
};

/** Subscribe to a key (or '*'). Returns an unsubscribe function. */
export function subscribe(key, fn) {
  if (!listeners.has(key)) listeners.set(key, new Set());
  listeners.get(key).add(fn);
  return () => listeners.get(key).delete(fn);
}

/** Merge a patch into the state and notify subscribers. */
export function set(patch) {
  const keys = Object.keys(patch);
  Object.assign(state, patch);
  for (const k of keys) {
    for (const fn of listeners.get(k) || []) fn(state[k], k);
  }
  for (const fn of listeners.get('*') || []) fn(state, keys);
}

export function get(key) { return state[key]; }
