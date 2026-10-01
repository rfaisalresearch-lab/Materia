"""Electronic density of states and projected density of states through GPAW.

A :class:`DOSSpec` is a frozen ground-state specification, the geometry and
electronic state the DOS belongs to, together with how the DOS is sampled:
the energy window and grid, the broadening, the spin channels, the
non-self-consistent k-point grid and band count, and the atom and angular
momentum projections.  It is immutable and versioned (:data:`SCHEMA`,
:data:`VERSION`).  The geometry comes from a structure, from a stored
converged ground-state run, or from a stored converged relaxation, and the
source is recorded; a specification built from a run keeps that run's
electronic settings exactly and refuses to change them.

The worker converges the ground state with the frozen settings, runs GPAW's
``fixed_density`` on the DOS k-point grid with the requested bands, and
evaluates ``gpaw.dos.DOSCalculator.raw_dos`` and ``raw_pdos`` on the energy
grid.  The Gaussian of ``raw_dos`` is ``exp(-((E - e) / w)^2) / (sqrt(pi) w)``,
so its standard deviation is ``w / sqrt(2)``; ``w = 0`` selects the linear
tetrahedron method.  Projections are onto the PAW projectors of bound partial
waves, which is what ``raw_pdos`` uses, so a channel with no bound partial
wave in the pinned dataset cannot be projected and is refused.

Every DOS variable reaches the worker; the worker reports the non-self-
consistent GPAW parameters and the grid, method, width, reference, spin and
projections it used, and every one is compared.  The total DOS is then
recomputed in the Materia process from the eigenvalues and k-point weights
the worker returned, with the same formula, and must agree; that checks the
grid, the width and the method independently of the echo.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...project_format.arrays import StoredArray
from ...provenance import Convergence, Fidelity, Origin, Result
from ...solvers.gpaw_driver import runner
from . import run as ground
from . import spec as specs
from .datasets import ANGULAR, DatasetError, bound_channels, select

SCHEMA = "materia.dft.dos"
VERSION = "1.0"
MODEL = "external:gpaw/density-of-states"

REFERENCES: Tuple[str, ...] = ("fermi-level", "absolute")
BROADENINGS: Tuple[str, ...] = ("gaussian", "tetrahedron")
SPIN_CHANNELS: Tuple[str, ...] = ("total", "resolved")
SOURCES: Tuple[str, ...] = ("structure", "ground-state", "relaxation")

MAX_POINTS = 20_001
MAX_PROJECTIONS = 64
MAX_WINDOW_EV = 100.0
MAX_WIDTH_EV = 2.0
MAX_BANDS = 4000
MAX_DIVISIONS = 32
RECOMPUTE_TOLERANCE = 1e-6
SCF_OBSERVABLES: Tuple[str, ...] = ("energy", "fermi_level")

STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

FIELDS: Tuple[specs.FieldInfo, ...] = (
    specs.FieldInfo("source", "dos", "geometry source", "",
                    "Where the geometry and electronic settings come from: a structure, a "
                    "stored converged ground-state run, or a stored converged relaxation.",
                    settable=False),
    specs.FieldInfo("energy_reference", "dos", "energy zero", "",
                    "fermi-level puts zero at the self-consistent Fermi level. absolute uses "
                    "GPAW's Kohn-Sham energies, whose zero is the vacuum only for a neutral or "
                    "charged cluster without a field."),
    specs.FieldInfo("energy_min_eV", "dos", "window start", "eV",
                    "Lowest energy of the grid, relative to the zero above."),
    specs.FieldInfo("energy_max_eV", "dos", "window end", "eV",
                    "Highest energy of the grid; the window must be a whole number of steps."),
    specs.FieldInfo("energy_step_eV", "dos", "grid spacing", "eV",
                    "Spacing of the energy grid. For Gaussian broadening it must be at most "
                    "half the width, so every peak is sampled."),
    specs.FieldInfo("broadening", "dos", "broadening", "",
                    "gaussian broadens each state; tetrahedron integrates linearly between "
                    "k-points and needs a bulk crystal with at least two divisions along "
                    "every axis."),
    specs.FieldInfo("width_eV", "dos", "Gaussian width", "eV",
                    "GPAW's width w in exp(-((E - e) / w)^2) / (sqrt(pi) w); the standard "
                    "deviation is w / sqrt(2). Empty for the tetrahedron method.",
                    applies="gaussian"),
    specs.FieldInfo("spin_channels", "dos", "spin channels", "",
                    "total sums both spins; resolved also stores spin up and spin down, "
                    "which needs a spin-polarised calculation."),
    specs.FieldInfo("dos_kpoints", "dos", "DOS k-point grid", "divisions",
                    "Brillouin-zone sampling of the non-self-consistent DOS step. Must be 1 "
                    "along every open direction."),
    specs.FieldInfo("dos_kpoints_gamma_centered", "dos", "Gamma-centred DOS grid", "",
                    "Include the Gamma point in the DOS grid.", advanced=True),
    specs.FieldInfo("n_bands", "dos", "bands", "bands",
                    "Kohn-Sham states per spin and k-point in the DOS step, all converged. "
                    "They must reach above the top of the window or the run is refused as "
                    "incomplete."),
    specs.FieldInfo("projections", "dos", "projections", "",
                    "Atom sets and angular channels (s, p, d, f) to project on. Only "
                    "channels with a bound partial wave in the pinned PAW dataset."),
    specs.FieldInfo("projection_channels", "dos", "projectable channels", "",
                    "Bound partial-wave channels of each pinned dataset.", settable=False),
)
FIELD_INDEX = {f.name: f for f in FIELDS}
SETTABLE: Tuple[str, ...] = tuple(f.name for f in FIELDS if f.settable)


@dataclass(frozen=True)
class DOSSpec:
    """One DOS and PDOS calculation, frozen.  See :data:`FIELDS`."""

    ground_state: specs.GroundStateSpec
    source: Dict[str, Any]
    energy_reference: str
    energy_min_eV: float
    energy_max_eV: float
    energy_step_eV: float
    broadening: str
    width_eV: Optional[float]
    spin_channels: str
    dos_kpoints: Tuple[int, int, int]
    dos_kpoints_gamma_centered: bool
    n_bands: int
    projections: Tuple[Tuple[str, Tuple[int, ...], str], ...]
    projection_channels: Tuple[Tuple[str, Tuple[str, ...]], ...]
    schema: str = SCHEMA
    version: str = VERSION

    def as_dict(self) -> dict:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "ground_state":
                out[f.name] = value.as_dict()
            elif f.name == "source":
                out[f.name] = json.loads(json.dumps(value))
            else:
                out[f.name] = specs._plain(value)
        return out

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "DOSSpec":
        if data.get("schema") != SCHEMA:
            raise specs.SpecError(f"Not a DOS specification: {data.get('schema')!r}.")
        if str(data.get("version")) != VERSION:
            raise specs.SpecError(f"DOS specification version {data.get('version')} is not "
                                  f"understood by this Materia (it reads {VERSION}).")
        names = {f.name for f in fields(DOSSpec)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise specs.SpecError(f"Unknown DOS field(s): {', '.join(unknown)}.")
        missing = sorted(n for n in names - set(data) if n not in ("schema", "version"))
        if missing:
            raise specs.SpecError(f"Missing DOS field(s): {', '.join(missing)}.")
        values = dict(data)
        values["ground_state"] = specs.GroundStateSpec.from_dict(data["ground_state"])
        values["source"] = dict(data["source"])
        values["dos_kpoints"] = tuple(int(k) for k in data["dos_kpoints"])
        values["projections"] = tuple((str(p[0]), tuple(int(i) for i in p[1]), str(p[2]))
                                      for p in data["projections"])
        values["projection_channels"] = tuple((str(s), tuple(str(c) for c in channels))
                                              for s, channels in data["projection_channels"])
        values["schema"] = SCHEMA
        values["version"] = VERSION
        return DOSSpec(**values)

    @property
    def digest(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def short_digest(self) -> str:
        return self.digest[:16]

    @property
    def structure_key(self) -> Optional[str]:
        return self.ground_state.structure_key

    @property
    def geometry_digest(self) -> str:
        return self.ground_state.geometry_digest

    @property
    def npoints(self) -> int:
        return int(round((self.energy_max_eV - self.energy_min_eV) / self.energy_step_eV)) + 1

    def energies(self) -> np.ndarray:
        return self.energy_min_eV + self.energy_step_eV * np.arange(self.npoints)

    @property
    def gpaw_width(self) -> float:
        return 0.0 if self.broadening == "tetrahedron" else float(self.width_eV or 0.0)

    def settings(self) -> Dict[str, Any]:
        out = {name: specs._plain(getattr(self, name)) for name in SETTABLE}
        out["projections"] = [{"label": p[0], "atom_ids": list(p[1]), "angular": p[2]}
                              for p in self.projections]
        return out

    def atom_indices(self, atom_ids: Sequence[int]) -> List[int]:
        position = {int(i): k for k, i in enumerate(self.ground_state.atom_ids)}
        return [position[int(i)] for i in atom_ids]


def electronic_digest(gs: specs.GroundStateSpec) -> str:
    """SHA-256 of a ground-state specification without its observables."""
    data = gs.as_dict()
    data.pop("observables", None)
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _channels_of(gs: specs.GroundStateSpec) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    out = []
    for symbol, path, _ in gs.paw_datasets:
        try:
            out.append((symbol, tuple(bound_channels(path))))
        except (OSError, DatasetError):
            out.append((symbol, ()))
    return tuple(sorted(out))


def _occupied_bands(gs: specs.GroundStateSpec, environment) -> Optional[int]:
    paths = list(getattr(environment, "setup_paths", []) or [])
    if not paths:
        return None
    try:
        chosen = select(gs.symbols, gs.xc, paths)
    except DatasetError:
        return None
    valence = chosen.valence_electrons(gs.symbols) - gs.charge_e
    moment = abs(float(sum(gs.initial_magnetic_moments_muB))) if gs.spin_polarized else 0.0
    per_channel = (valence + moment) / 2.0 if gs.spin_polarized else valence / 2.0
    return int(math.ceil(per_channel - 1e-9))


def _default_projections(gs: specs.GroundStateSpec,
                         channels: Tuple[Tuple[str, Tuple[str, ...]], ...]):
    table = dict(channels)
    order: List[str] = []
    for symbol in gs.symbols:
        if symbol not in order:
            order.append(symbol)
    out = []
    for symbol in order:
        ids = tuple(int(i) for i, s in zip(gs.atom_ids, gs.symbols) if s == symbol)
        for channel in table.get(symbol, ()):
            out.append((f"{symbol} {channel}", ids, channel))
    return tuple(out)


def _coerce_projections(gs: specs.GroundStateSpec, value: Any):
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise specs.SpecError("projections must be a list of selections such as "
                              "{'element': 'Si', 'angular': 'p'} or "
                              "{'atom_ids': [1, 2], 'angular': 's'}.")
    out = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            item = {"label": item[0], "atom_ids": item[1], "angular": item[2]}
        if not isinstance(item, Mapping):
            raise specs.SpecError(f"A projection must be a mapping, got {item!r}.")
        unknown = sorted(set(item) - {"label", "atom_ids", "element", "angular"})
        if unknown:
            raise specs.SpecError(f"Unknown projection key(s): {', '.join(unknown)}.")
        angular = item.get("angular")
        if angular not in ANGULAR:
            raise specs.SpecError(f"A projection's angular channel must be one of "
                                  f"{', '.join(ANGULAR)}, got {angular!r}.")
        if ("element" in item) == ("atom_ids" in item):
            raise specs.SpecError("A projection selects atoms by 'element' or by "
                                  "'atom_ids', exactly one of the two.")
        if "element" in item:
            element = str(item["element"])
            ids = tuple(int(i) for i, s in zip(gs.atom_ids, gs.symbols) if s == element)
            default_label = f"{element} {angular}"
        else:
            try:
                ids = tuple(specs._integer("atom_ids", i) for i in item["atom_ids"])
            except TypeError:
                raise specs.SpecError("atom_ids must be a list of atom ids.") from None
            default_label = f"atoms {','.join(str(i) for i in ids)} {angular}"
        label = str(item.get("label") or default_label).strip()
        out.append((label, ids, angular))
    return tuple(out)


def _coerce(name: str, value: Any, gs: specs.GroundStateSpec) -> Any:
    if name == "energy_reference":
        return specs._choice(name, value, REFERENCES)
    if name == "broadening":
        return specs._choice(name, value, BROADENINGS)
    if name == "spin_channels":
        return specs._choice(name, value, SPIN_CHANNELS)
    if name in ("energy_min_eV", "energy_max_eV", "energy_step_eV"):
        return specs._number(name, value, "eV")
    if name == "width_eV":
        return specs._optional_number(name, value, "eV")
    if name == "n_bands":
        return specs._integer(name, value)
    if name == "dos_kpoints":
        return specs._coerce("kpoints", value, gs.n_atoms)
    if name == "dos_kpoints_gamma_centered":
        return specs._boolean(name, value)
    if name == "projections":
        return _coerce_projections(gs, value)
    raise specs.SpecError(f"{name} is not a DOS variable.")


def _split(variables: Mapping[str, Any]):
    dos_vars = {k: v for k, v in variables.items() if k in SETTABLE}
    rest = {k: v for k, v in variables.items() if k not in SETTABLE}
    for name in ("source", "projection_channels"):
        if name in rest:
            raise specs.SpecError(f"{name} is not settable; it is fixed by the geometry "
                                  "source and the pinned datasets.")
    return dos_vars, rest


def _finish(gs: specs.GroundStateSpec, source: Dict[str, Any], dos_vars: Dict[str, Any],
            environment) -> DOSSpec:
    values: Dict[str, Any] = {name: _coerce(name, value, gs)
                              for name, value in dos_vars.items()}
    channels = _channels_of(gs)
    broadening = values.get("broadening", "gaussian")
    occupied = _occupied_bands(gs, environment)
    if occupied is None:
        occupied = int(math.ceil(sum(gs.numbers) / 2.0))
    defaults = {
        "energy_reference": "fermi-level", "energy_min_eV": -10.0, "energy_max_eV": 5.0,
        "energy_step_eV": 0.01, "broadening": "gaussian",
        "width_eV": None if broadening == "tetrahedron" else 0.1,
        "spin_channels": "resolved" if gs.spin_polarized else "total",
        "dos_kpoints": tuple(min(MAX_DIVISIONS, 2 * k) if gs.pbc[i] else 1
                             for i, k in enumerate(gs.kpoints)),
        "dos_kpoints_gamma_centered": gs.kpoints_gamma_centered,
        "n_bands": max(occupied + 8, int(math.ceil(1.5 * occupied))),
        "projections": _default_projections(gs, channels),
    }
    for name, value in defaults.items():
        if name not in values or (name == "projections" and values[name] is None):
            values[name] = value
    return DOSSpec(ground_state=gs, source=dict(source), projection_channels=channels,
                   **values)


def _minimal(gs: specs.GroundStateSpec) -> specs.GroundStateSpec:
    return replace(gs, observables=SCF_OBSERVABLES)


def build(structure, environment=None, structure_key: Optional[str] = None,
          **variables: Any) -> DOSSpec:
    """A DOS specification for ``structure``: its own ground state, then the DOS.

    Ground-state variables are those of
    :func:`materia.experiments.dft.spec.build`; DOS variables are
    :data:`SETTABLE`.  The ground state records only the energy and the Fermi
    level, which is what the DOS needs.  Not validated here; :func:`check`
    does that.
    """
    dos_vars, rest = _split(variables)
    environment = specs._environment(environment)
    rest.pop("observables", None)
    gs = _minimal(specs.build(structure, environment, structure_key=structure_key, **rest))
    source = {"kind": "structure", "run_id": None, "spec_digest": None,
              "geometry_digest": gs.geometry_digest,
              "electronic_digest": electronic_digest(gs), "energy_eV": None}
    return _finish(gs, source, dos_vars, environment)


def source_spec(project, kind: str, run_id: str):
    """The converged ground state a stored run describes, and its energy."""
    from . import records

    if kind == "ground-state":
        record = records.record(project, run_id)
        if record is None:
            raise specs.SpecError(f"There is no ground-state run {run_id}.")
        if not record.supported:
            raise specs.SpecError(f"Ground-state run {run_id} did not converge, so it has "
                                  "no ground state to take a DOS of.")
        gs = records.spec_of(project, run_id)
        energy = project.results.get(records.key(run_id, "energy"))
    elif kind == "relaxation":
        record = records.relax_record(project, run_id)
        if record is None:
            raise specs.SpecError(f"There is no relaxation {run_id}.")
        if record.extra.get("status") != "converged":
            raise specs.SpecError(f"Relaxation {run_id} did not meet its criteria, so its "
                                  "final geometry is not a minimum. Take the DOS of the "
                                  "structure instead if that geometry is really wanted.")
        energy = project.results.get(records.relax_key(run_id, "energy"))
        data = energy.provenance.parameters.get("experiment_spec") if energy else None
        gs = specs.GroundStateSpec.from_dict(data) if data else None
    else:
        raise specs.SpecError(f"A DOS source must be one of ground-state or relaxation, "
                              f"got {kind!r}.")
    if gs is None:
        raise specs.SpecError(f"The {kind} run {run_id} has no readable specification.")
    return gs, (None if energy is None else float(energy.value))


def build_from_run(project, kind: str, run_id: str, environment=None,
                   **variables: Any) -> DOSSpec:
    """A DOS specification on the exact geometry and settings of a stored run."""
    dos_vars, rest = _split(variables)
    if rest:
        raise specs.SpecError(
            f"{', '.join(sorted(rest))}: a DOS built from a stored {kind} run keeps that "
            "run's geometry and electronic settings exactly. Build it from the structure "
            "to change them.")
    environment = specs._environment(environment)
    original, energy = source_spec(project, kind, run_id)
    gs = _minimal(original)
    source = {"kind": kind, "run_id": run_id, "spec_digest": original.digest,
              "geometry_digest": original.geometry_digest,
              "electronic_digest": electronic_digest(original), "energy_eV": energy}
    return _finish(gs, source, dos_vars, environment)


def changed(spec: DOSSpec, environment=None, **variables: Any) -> DOSSpec:
    """A new specification with some DOS or, for a structure source, ground-state variables."""
    dos_vars, rest = _split(variables)
    values = {name: _coerce(name, value, spec.ground_state)
              for name, value in dos_vars.items()}
    if "projections" in values and values["projections"] is None:
        values["projections"] = _default_projections(spec.ground_state,
                                                     spec.projection_channels)
    if values.get("broadening") == "tetrahedron" and "width_eV" not in values:
        values["width_eV"] = None
    if values.get("broadening") == "gaussian" and "width_eV" not in values \
            and spec.width_eV is None:
        values["width_eV"] = 0.1
    updated = replace(spec, **values)
    if rest:
        if spec.source.get("kind") != "structure":
            raise specs.SpecError(f"{', '.join(sorted(rest))}: this DOS keeps the electronic "
                                  f"settings of {spec.source.get('kind')} run "
                                  f"{spec.source.get('run_id')}.")
        rest.pop("observables", None)
        gs = _minimal(specs.changed(spec.ground_state, environment, **rest))
        source = dict(spec.source, electronic_digest=electronic_digest(gs))
        updated = replace(updated, ground_state=gs, source=source,
                          projection_channels=_channels_of(gs))
    return updated


def check(spec: DOSSpec, environment=None) -> specs.SpecReport:
    """Every ground-state rule plus every DOS rule, keyed by variable."""
    environment = specs._environment(environment)
    gs = spec.ground_state
    report = specs.check(gs, environment)
    by_field = {k: list(v) for k, v in report.by_field.items()}
    warnings = list(report.warnings)
    general = [t for t in report.blocking
               if not any(t in texts for texts in report.by_field.values())]

    def refuse(name: str, text: str) -> None:
        by_field.setdefault(name, []).append(text)

    if spec.schema != SCHEMA or spec.version != VERSION:
        general.append(f"DOS specification {spec.schema} {spec.version} is not {SCHEMA} "
                       f"{VERSION}.")
    source = spec.source
    if source.get("kind") not in SOURCES:
        refuse("source", f"The geometry source must be one of {', '.join(SOURCES)}.")
    if source.get("geometry_digest") != gs.geometry_digest:
        refuse("source", "The geometry differs from the source's geometry.")
    if source.get("electronic_digest") != electronic_digest(gs):
        refuse("source", f"The electronic settings differ from those of the "
                         f"{source.get('kind')} source, so the DOS would not describe it.")
    if spec.energy_reference not in REFERENCES:
        refuse("energy_reference", f"energy_reference must be one of {', '.join(REFERENCES)}.")
    field_on = bool(np.any(np.asarray(gs.external_field_V_per_A, dtype=float)))
    if spec.energy_reference == "absolute":
        if gs.boundary != "cluster":
            refuse("energy_reference",
                   f"In a {gs.boundary} cell GPAW's Kohn-Sham energies are measured from "
                   "an arbitrary zero set by the average electrostatic potential, not the "
                   "vacuum. Use fermi-level.")
        elif field_on:
            refuse("energy_reference",
                   "A uniform field has no unique zero of potential, so absolute "
                   "Kohn-Sham energies depend on where the origin is. Use fermi-level.")
    low, high, step = spec.energy_min_eV, spec.energy_max_eV, spec.energy_step_eV
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        refuse("energy_max_eV", "The window must end above where it starts.")
    elif high - low > MAX_WINDOW_EV:
        refuse("energy_max_eV", f"A {high - low:g} eV window is wider than the "
                                f"{MAX_WINDOW_EV:g} eV allowed.")
    if not (math.isfinite(step) and step > 0):
        refuse("energy_step_eV", "The grid spacing must be a positive energy.")
    elif low < high:
        steps = (high - low) / step
        if abs(steps - round(steps)) > 1e-6 * max(1.0, steps):
            refuse("energy_step_eV", f"The window of {high - low:g} eV is not a whole number "
                                     f"of {step:g} eV steps, so its end would not be on the "
                                     "grid. Adjust the window or the spacing.")
        elif round(steps) + 1 > MAX_POINTS:
            refuse("energy_step_eV", f"{int(round(steps)) + 1} grid points exceed the "
                                     f"{MAX_POINTS} allowed; use a coarser spacing.")
    if spec.broadening not in BROADENINGS:
        refuse("broadening", f"broadening must be one of {', '.join(BROADENINGS)}.")
    if spec.broadening == "gaussian":
        width = spec.width_eV
        if width is None or not (0.0 < width <= MAX_WIDTH_EV):
            refuse("width_eV", f"Gaussian broadening needs a width between 0 and "
                               f"{MAX_WIDTH_EV:g} eV.")
        elif math.isfinite(step) and step > width / 2.0:
            refuse("energy_step_eV", f"A {step:g} eV spacing samples a {width:g} eV Gaussian "
                                     "too coarsely: peaks would be missed or distorted. Use "
                                     f"at most {width / 2.0:g} eV.")
    if spec.broadening == "tetrahedron":
        if spec.width_eV not in (None, 0.0):
            refuse("width_eV", "The tetrahedron method has no width; leave it empty.")
        if gs.boundary != "bulk":
            refuse("broadening", f"The tetrahedron method interpolates between k-points in "
                                 f"three dimensions and needs a bulk crystal; this is a "
                                 f"{gs.boundary}. Use gaussian.")
        elif any(k < 2 for k in spec.dos_kpoints):
            refuse("dos_kpoints", "The tetrahedron method needs at least two divisions "
                                  "along every axis.")
    if spec.spin_channels not in SPIN_CHANNELS:
        refuse("spin_channels", f"spin_channels must be one of {', '.join(SPIN_CHANNELS)}.")
    elif spec.spin_channels == "resolved" and not gs.spin_polarized:
        refuse("spin_channels", "The calculation is spin-paired, so spin up and spin down "
                                "are identical by construction. Use total, or turn spin "
                                "polarisation on.")
    for axis, divisions in enumerate(spec.dos_kpoints):
        if divisions < 1 or divisions > MAX_DIVISIONS:
            refuse("dos_kpoints", f"The DOS grid needs 1 to {MAX_DIVISIONS} divisions along "
                                  f"{'abc'[axis]}, got {divisions}.")
        elif divisions > 1 and not gs.pbc[axis]:
            refuse("dos_kpoints", f"{divisions} divisions along {'abc'[axis]}, which is open. "
                                  "There is no Brillouin zone along an open direction; use 1.")
    if gs.boundary != "cluster" and all(k == 1 for k in spec.dos_kpoints):
        warnings.append("The DOS of a periodic system sampled at the Gamma point alone is a "
                        "set of broadened levels, not a converged DOS.")
    occupied = _occupied_bands(gs, environment)
    if not (1 <= spec.n_bands <= MAX_BANDS):
        refuse("n_bands", f"n_bands must be 1 to {MAX_BANDS}.")
    elif occupied is not None and spec.n_bands < occupied:
        refuse("n_bands", f"{spec.n_bands} bands cannot hold the occupied states; at least "
                          f"{occupied} are occupied.")
    elif occupied is not None and spec.n_bands == occupied and spec.energy_max_eV > 0:
        warnings.append("No empty band is computed, so the DOS above the Fermi level will be "
                        "empty; the run is refused if the window reaches above the bands.")
    if abs(gs.charge_e) > 1e-9 and gs.boundary == "bulk":
        warnings.append("A charged cell with a uniform background shifts every level by a "
                        "finite-size term; only energies relative to the Fermi level are "
                        "comparable.")
    pinned = dict(spec.projection_channels)
    if spec.projection_channels != _channels_of(gs):
        refuse("projection_channels", "The bound channels of the datasets on disk differ "
                                      "from the ones this specification was pinned to.")
    projections = spec.projections
    if len(projections) > MAX_PROJECTIONS:
        refuse("projections", f"{len(projections)} projections exceed the "
                              f"{MAX_PROJECTIONS} allowed.")
    labels = [p[0] for p in projections]
    if len(set(labels)) != len(labels):
        refuse("projections", "Two projections share a label.")
    atoms = {int(i): s for i, s in zip(gs.atom_ids, gs.symbols)}
    for label, ids, angular in projections:
        if not ids:
            refuse("projections", f"Projection {label!r} selects no atoms.")
            continue
        missing = [i for i in ids if i not in atoms]
        if missing:
            refuse("projections", f"Projection {label!r} names atom(s) {missing} that are "
                                  "not in the structure.")
            continue
        if len(set(ids)) != len(ids):
            refuse("projections", f"Projection {label!r} names an atom twice.")
        if angular not in ANGULAR:
            refuse("projections", f"Projection {label!r} has angular channel {angular!r}.")
            continue
        for symbol in sorted({atoms[i] for i in ids}):
            available = pinned.get(symbol, ())
            if angular not in available:
                refuse("projections",
                       f"Projection {label!r}: the {symbol} dataset has no bound {angular} "
                       f"partial wave, so GPAW cannot project on {angular}. Projectable "
                       f"channels for {symbol}: {', '.join(available) or 'none'}.")
    if projections:
        warnings.append("Projections are onto the PAW projectors of bound partial waves "
                        "inside the augmentation spheres. They are not normalised atomic "
                        "orbitals, and their sum need not equal the total DOS.")
    blocking = list(general)
    for texts in by_field.values():
        for text in texts:
            if text not in blocking:
                blocking.append(text)
    options = dict(report.options)
    options.update({
        "references": ["fermi-level", "absolute"] if (
            gs.boundary == "cluster" and not field_on) else ["fermi-level"],
        "broadenings": ["gaussian", "tetrahedron"] if gs.boundary == "bulk" else ["gaussian"],
        "spin_channels": ["total", "resolved"] if gs.spin_polarized else ["total"],
        "projection_channels": {k: list(v) for k, v in pinned.items()},
        "occupied_bands": occupied,
    })
    return specs.SpecReport(blocking, warnings, by_field, report.boundary, report.electrons,
                            options)


def require(spec: DOSSpec, environment=None) -> specs.SpecReport:
    report = check(spec, environment)
    if not report.ok:
        raise specs.SpecRefused(report)
    return report


def nscf_parameters(spec: DOSSpec) -> Dict[str, Any]:
    """The keyword arguments handed to ``GPAW.fixed_density``."""
    return {"kpts": {"size": list(spec.dos_kpoints),
                     "gamma": bool(spec.dos_kpoints_gamma_centered)},
            "nbands": int(spec.n_bands), "convergence": {"bands": "all"}}


def dos_settings(spec: DOSSpec) -> Dict[str, Any]:
    """The DOS block of the worker job."""
    return {"nscf": nscf_parameters(spec), "reference": spec.energy_reference,
            "energy_min": float(spec.energy_min_eV), "energy_step": float(spec.energy_step_eV),
            "npoints": int(spec.npoints), "width": spec.gpaw_width,
            "method": spec.broadening, "spin": spec.spin_channels,
            "projections": [{"label": label, "indices": spec.atom_indices(ids),
                             "angular": angular}
                            for label, ids, angular in spec.projections],
            "channels": {symbol: list(channels)
                         for symbol, channels in spec.projection_channels}}


def worker_job(spec: DOSSpec) -> Dict[str, Any]:
    gs = spec.ground_state
    return {"structure": specs.worker_structure(gs),
            "parameters": specs.gpaw_parameters(gs),
            "observables": list(gs.observables),
            "expected_datasets": {s: {"path": p, "sha256": h} for s, p, h in gs.paw_datasets},
            "dos": dos_settings(spec)}


def describe(spec: DOSSpec, report: Optional[specs.SpecReport] = None) -> dict:
    report = report if report is not None else check(spec)
    items = []
    for info in FIELDS:
        item = info.as_dict()
        value = getattr(spec, info.name)
        item["value"] = (json.loads(json.dumps(value)) if info.name == "source"
                         else specs._plain(value))
        item["problems"] = list(report.by_field.get(info.name, []))
        items.append(item)
    return {"schema": spec.schema, "version": spec.version, "digest": spec.digest,
            "short_digest": spec.short_digest, "dos": items, "settings": spec.settings(),
            "source": dict(spec.source), "npoints": spec.npoints,
            "ground_state": specs.describe(spec.ground_state, report),
            "gpaw_parameters": specs.gpaw_parameters(spec.ground_state),
            "nscf_parameters": nscf_parameters(spec), "dos_settings": dos_settings(spec)}


def gaussian_dos(eigenvalues: np.ndarray, weights: np.ndarray, energies: np.ndarray,
                 width: float) -> np.ndarray:
    """GPAW's Gaussian DOS for one spin channel, from eigenvalues[k, n] and weights[k]."""
    out = np.zeros_like(energies, dtype=float)
    for weight, row in zip(weights, eigenvalues):
        delta = (energies[None, :] - np.asarray(row, dtype=float)[:, None]) / width
        out += weight * np.exp(-delta * delta).sum(axis=0)
    return out / (math.sqrt(math.pi) * width)


def gaussian_window_count(eigenvalues: np.ndarray, weights: np.ndarray, low: float,
                          high: float, width: float) -> float:
    """Exact integral of the Gaussian DOS of one spin channel between two energies."""
    from math import erf

    total = 0.0
    for k, row in enumerate(eigenvalues):
        total += weights[k] * sum(0.5 * (erf((high - e) / width) - erf((low - e) / width))
                                  for e in row)
    return float(total)


def tetrahedron_dos(eigenvalues: np.ndarray, energies: np.ndarray, cell: np.ndarray,
                    size: Sequence[int], bz2ibz: Sequence[int]) -> np.ndarray:
    """The linear-tetrahedron DOS of one spin channel, as GPAW computes it (via ASE)."""
    from ase.dft.dos import linear_tetrahedron_integration

    eig = np.asarray(eigenvalues, dtype=float)
    if len(eig) != int(np.prod(size)):
        eig = eig[np.asarray(bz2ibz, dtype=int)]
    eig = eig.reshape(tuple(int(n) for n in size) + (-1,))
    return np.asarray(linear_tetrahedron_integration(np.asarray(cell, dtype=float), eig,
                                                     energies), dtype=float)


@dataclass
class DOSOutcome:
    run_id: str
    spec: DOSSpec
    status: str
    reason: str = ""
    results: Dict[str, Result] = field(default_factory=dict)
    arrays: Dict[str, StoredArray] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    wall_time_s: float = 0.0
    audit: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_COMPLETE


class _Invalid(Exception):
    pass


def _approximations(spec: DOSSpec) -> List[str]:
    k = "x".join(str(n) for n in spec.dos_kpoints)
    out = [
        f"Kohn-Sham eigenvalues from a non-self-consistent GPAW calculation on a {k} "
        f"{'Gamma-centred' if spec.dos_kpoints_gamma_centered else 'Monkhorst-Pack'} grid "
        f"with {spec.n_bands} converged bands, at the self-consistent density.",
        "Kohn-Sham states are not quasiparticles: semilocal functionals place empty states "
        "too low and underestimate gaps.",
    ]
    if spec.broadening == "gaussian":
        out.append(f"Gaussian broadening, GPAW width {spec.width_eV:g} eV (standard deviation "
                   f"{spec.width_eV / math.sqrt(2.0):.4g} eV); the broadening, not the "
                   "physics, sets the width of every peak.")
    else:
        out.append("Linear tetrahedron integration between k-points; the energy grid, not a "
                   "width, limits the resolution.")
    out.append("Energies relative to the self-consistent Fermi level." if
               spec.energy_reference == "fermi-level" else
               "Absolute Kohn-Sham energies, zero at the vacuum of the finite box.")
    if spec.projections:
        out.append("Projected DOS onto the PAW projectors of bound partial waves inside the "
                   "augmentation spheres; not normalised atomic orbitals, and spilling "
                   "between spheres is not assigned to any projection.")
    return out


def _validate(spec: DOSSpec, run: runner.GPAWRun, job: Dict[str, Any],
              arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    result = run.result
    dos = result.get("dos")
    if not isinstance(dos, dict):
        raise _Invalid("The worker reported no DOS.")
    mismatch = ground.parameter_mismatch(job["parameters"], result.get("parameters_used"))
    if mismatch:
        raise _Invalid("GPAW did not run the ground state with the parameters Materia sent: "
                       f"{mismatch}")
    mismatch = ground.parameter_mismatch(job["dos"]["nscf"], dos.get("nscf_parameters_used"),
                                         "nscf.")
    if mismatch:
        raise _Invalid("GPAW did not run the non-self-consistent step with the parameters "
                       f"Materia sent: {mismatch}")
    sent = {k: v for k, v in job["dos"].items() if k not in ("nscf", "channels")}
    mismatch = ground.parameter_mismatch(sent, dos.get("used"), "dos.")
    if mismatch:
        raise _Invalid(f"The worker evaluated a different DOS from the one specified: "
                       f"{mismatch}")
    channels = {k: list(v) for k, v in (dos.get("bound_channels") or {}).items()}
    if channels != job["dos"]["channels"]:
        raise _Invalid("The datasets GPAW loaded have different bound channels "
                       f"({channels}) from the pinned ones ({job['dos']['channels']}).")
    if not dos.get("nscf_converged"):
        raise _Invalid("The non-self-consistent step did not converge every band.")
    fermi_scf = dos.get("fermi_level_scf_eV")
    fermi_nscf = dos.get("fermi_level_nscf_eV")
    if fermi_scf is None or not math.isfinite(float(fermi_scf)):
        raise _Invalid("The worker reported no finite Fermi level.")
    if fermi_nscf is None or abs(float(fermi_nscf) - float(fermi_scf)) > 1e-9:
        raise _Invalid("The non-self-consistent step moved the Fermi level; the energy zero "
                       "would not be the ground state's.")
    n = spec.npoints
    energies = np.asarray(arrays.get("dos_energies"), dtype=float)
    if energies.shape != (n,) or not np.allclose(energies, spec.energies(), rtol=0,
                                                 atol=1e-9):
        raise _Invalid("The energy grid the worker used is not the specified one.")
    total = np.asarray(arrays.get("dos_total"), dtype=float)
    if total.shape != (n,):
        raise _Invalid("The total DOS has the wrong shape.")
    eigen = np.asarray(arrays.get("dos_eigenvalues"), dtype=float)
    weights = np.asarray(arrays.get("dos_kweights"), dtype=float)
    if eigen.ndim != 3 or eigen.shape[2] != spec.n_bands or weights.shape != (eigen.shape[1],):
        raise _Invalid("The eigenvalues or k-point weights have the wrong shape.")
    nspins = eigen.shape[0]
    if nspins != (2 if spec.ground_state.spin_polarized else 1):
        raise _Invalid("The number of spin channels is not the specified one.")
    for name in ("dos_total", "dos_spin", "pdos_total", "pdos_spin", "dos_eigenvalues"):
        if name in arrays and not np.all(np.isfinite(np.asarray(arrays[name], dtype=float))):
            raise _Invalid(f"The worker returned a non-finite value in {name}.")
    scale = max(1.0, float(np.abs(total).max()))
    if float(total.min()) < -1e-9 * scale:
        raise _Invalid("The total DOS is negative somewhere.")
    reference = float(fermi_scf) if spec.energy_reference == "fermi-level" else 0.0
    absolute = spec.energies() + reference
    degeneracy = 2.0 / nspins
    if spec.broadening == "gaussian":
        channels_dos = [gaussian_dos(eigen[s], weights, absolute, spec.gpaw_width)
                        for s in range(nspins)]
    else:
        cell = np.asarray(dos.get("cell"), dtype=float)
        size = [int(v) for v in dos.get("size") or []]
        bz2ibz = [int(v) for v in dos.get("bz2ibz") or []]
        channels_dos = [tetrahedron_dos(eigen[s], absolute, cell, size, bz2ibz)
                        for s in range(nspins)]
    recomputed = degeneracy * sum(channels_dos)
    error = float(np.abs(recomputed - total).max() / scale)
    if error > RECOMPUTE_TOLERANCE:
        raise _Invalid(f"The DOS recomputed from the returned eigenvalues differs from GPAW's "
                       f"by {error:.2e} of its maximum; grid, width or method did not match.")
    if spec.spin_channels == "resolved":
        spin = np.asarray(arrays.get("dos_spin"), dtype=float)
        if spin.shape != (2, n):
            raise _Invalid("The spin-resolved DOS has the wrong shape.")
        if float(np.abs(spin.sum(axis=0) - total).max()) > 1e-9 * scale:
            raise _Invalid("Spin up and spin down do not add up to the total DOS.")
        for s in range(2):
            if float(np.abs(spin[s] - channels_dos[s]).max()) > RECOMPUTE_TOLERANCE * scale:
                raise _Invalid("A spin channel differs from its recomputation.")
    elif "dos_spin" in arrays:
        raise _Invalid("A spin-resolved DOS was returned but not requested.")
    p = len(spec.projections)
    if p:
        pdos = np.asarray(arrays.get("pdos_total"), dtype=float)
        if pdos.shape != (p, n):
            raise _Invalid("The projected DOS has the wrong shape.")
        if float(pdos.min()) < -1e-9 * max(1.0, float(np.abs(pdos).max())):
            raise _Invalid("A projected DOS is negative somewhere.")
        counts = dos.get("projector_counts") or []
        if len(counts) != p or any(not c or min(c) < 1 for c in counts):
            raise _Invalid("A projection has an atom without a projector in its channel.")
        if spec.spin_channels == "resolved":
            pspin = np.asarray(arrays.get("pdos_spin"), dtype=float)
            if pspin.shape != (p, 2, n):
                raise _Invalid("The spin-resolved projected DOS has the wrong shape.")
            if float(np.abs(pspin.sum(axis=1) - pdos).max()) > 1e-9 * max(
                    1.0, float(np.abs(pdos).max())):
                raise _Invalid("Projected spin channels do not add up.")
    elif "pdos_total" in arrays:
        raise _Invalid("A projected DOS was returned but none was requested.")
    top = float(absolute[-1])
    margin = 3.0 * spec.gpaw_width
    highest = float(eigen.max(axis=2).min())
    if highest < top + margin:
        raise _Invalid(f"The highest computed band reaches only {highest - reference:.3f} eV "
                       f"at some k-point, below the top of the window "
                       f"({spec.energy_max_eV:g} eV{' plus three widths' if margin else ''}); "
                       "states above it would be missing from the DOS. Raise n_bands or "
                       "lower energy_max_eV.")
    return {"fermi_level_eV": float(fermi_scf), "reference_eV": reference,
            "absolute": absolute, "channels": channels_dos, "recompute_error": error,
            "eigen": eigen, "weights": weights, "degeneracy": degeneracy,
            "highest_band_eV": highest}


def _checks(spec: DOSSpec, arrays: Dict[str, np.ndarray], data: Dict[str, Any],
            accounting: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    energies = spec.energies()
    step = spec.energy_step_eV
    total = np.asarray(arrays["dos_total"], dtype=float)
    window = float(np.trapezoid(total, dx=step))
    eigen, weights, degeneracy = data["eigen"], data["weights"], data["degeneracy"]
    absolute = data["absolute"]
    width = spec.gpaw_width
    out: Dict[str, Any] = {"window_integral_states": window,
                           "recompute_max_relative_error": data["recompute_error"]}
    if spec.broadening == "gaussian":
        exact = degeneracy * sum(gaussian_window_count(eigen[s], weights, absolute[0],
                                                       absolute[-1], width)
                                 for s in range(len(eigen)))
        out["window_integral_exact_states"] = exact
        out["window_integral_error_states"] = window - exact
    lowest = float(eigen.min())
    covers = absolute[0] <= lowest - 3.0 * width
    out["window_covers_lowest_band"] = bool(covers)
    below = energies <= 0.0 if spec.energy_reference == "fermi-level" else \
        absolute <= data["fermi_level_eV"]
    occupied = float(np.trapezoid(total[below], dx=step)) if below.sum() > 1 else 0.0
    out["integral_to_fermi_level_e"] = occupied
    expected = None
    if accounting:
        expected = accounting.get("expected_valence_electrons")
    out["expected_valence_electrons"] = expected
    if covers and expected is not None:
        out["electron_count_difference_e"] = occupied - float(expected)
        out["electron_count_note"] = (
            "Integral of the total DOS from the bottom of the window to the Fermi level "
            "against the valence electrons. Broadening that straddles the Fermi level "
            "(metals, or a width comparable to the gap) moves part of a state across it.")
    else:
        out["electron_count_note"] = ("The window does not reach below the lowest band, so "
                                      "the occupied states cannot be counted.")
    if spec.projections:
        pdos = np.asarray(arrays["pdos_total"], dtype=float)
        integrals = [float(np.trapezoid(row, dx=step)) for row in pdos]
        out["projection_integrals_states"] = dict(zip([p[0] for p in spec.projections],
                                                      integrals))
        out["projected_fraction_of_window"] = (float(sum(integrals)) / window
                                               if window > 0 else None)
    out["highest_band_above_window_eV"] = data["highest_band_eV"] - float(absolute[-1])
    return out


def execute(spec: DOSSpec, environment=None, *, run_id: Optional[str] = None,
            progress: Optional[Callable[[dict], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None) -> DOSOutcome:
    """Run one DOS.  Raises :class:`~.spec.SpecRefused` if it may not run."""
    environment = specs._environment(environment)
    report = require(spec, environment)
    run_id = run_id or ground.new_run_id()
    started = time.perf_counter()
    warnings = list(report.warnings)
    job = worker_job(spec)
    outcome = DOSOutcome(run_id=run_id, spec=spec, status=STATUS_FAILED, warnings=warnings)
    if cancelled is not None and cancelled():
        outcome.status = STATUS_CANCELLED
        outcome.reason = "Cancelled before GPAW was started. Nothing was kept."
        return outcome
    run = runner.run_job(job, environment, worker=ground.WORKER, progress=progress,
                         cancelled=cancelled, timeout_s=timeout_s)
    outcome.wall_time_s = time.perf_counter() - started
    outcome.audit = {"worker_status": run.status, "returncode": run.returncode,
                     "command": list(run.command), "stderr_tail": run.stderr[-2000:],
                     "log_tail": str(run.result.get("log_tail") or "")[-3000:]}
    if run.status == runner.STATUS_CANCELLED or (cancelled is not None and cancelled()):
        outcome.status = STATUS_CANCELLED
        outcome.reason = "The DOS calculation was cancelled. Nothing was kept."
        return outcome
    if run.status == runner.STATUS_NOT_CONVERGED:
        outcome.reason = ("The ground state did not converge, so there is no DOS to take. "
                          "Nothing was kept.")
        return outcome
    if run.status != runner.STATUS_CONVERGED:
        detail = run.error or "no error message was produced"
        if timeout_s is not None and "stopped the calculation after" in run.stderr:
            detail = f"GPAW was stopped after the {timeout_s:g} s time limit"
        outcome.reason = f"The DOS calculation failed: {detail.rstrip('.')}. Nothing was kept."
        return outcome
    arrays = {k: np.asarray(v) for k, v in run.arrays.items()}
    try:
        data = _validate(spec, run, job, arrays)
    except _Invalid as exc:
        outcome.reason = f"{exc} Nothing was kept."
        return outcome

    gs = spec.ground_state
    restart = {"key": None, "reused_from": None, "loaded": False, "kept": False,
               "reason": "the DOS converges its own ground state; no restart data is used"}
    history = run.result.get("scf_history") or []
    scf = ground._convergence(gs, run)
    base_extra = {
        "run_id": run_id, "structure_key": gs.structure_key,
        "geometry_digest": gs.geometry_digest, "spec_digest": gs.digest,
        "dos_spec_digest": spec.digest, "mass_numbers": list(gs.mass_numbers),
        "status": STATUS_COMPLETE, "classification": ground.CLASSIFICATION,
        "warnings": warnings,
        "software_warnings": [str(w) for w in (run.result.get("warnings") or [])],
        "study_id": None, "n_atoms": gs.n_atoms, "formula": gs.formula,
        "boundary": gs.boundary, "kind": "dos", "source": dict(spec.source),
    }
    staged = ground.GroundStateOutcome(run_id=run_id, spec=gs,
                                       status=runner.STATUS_CONVERGED, warnings=warnings)
    scf_run = runner.GPAWRun(status=runner.STATUS_CONVERGED, result=run.result,
                             stderr=run.stderr, returncode=run.returncode,
                             wall_time_s=run.wall_time_s, iterations=run.iterations,
                             command=run.command, arrays={}, events=run.events)
    ground.collect(staged, gs, scf_run, report, restart, base_extra, scf, history)
    accounting = staged.results.get("charge_accounting")
    checks = _checks(spec, arrays, data, accounting.value if accounting else None)
    source_energy = spec.source.get("energy_eV")
    energy = staged.results["energy"].value
    if source_energy is not None:
        checks["source_energy_difference_eV"] = float(energy) - float(source_energy)
        if abs(checks["source_energy_difference_eV"]) > 1e-3:
            warnings.append(f"The DOS run's ground-state energy differs from its source's by "
                            f"{checks['source_energy_difference_eV']:.2e} eV.")
    if spec.broadening == "gaussian" and abs(checks["window_integral_error_states"]) > \
            1e-3 * max(1.0, abs(checks["window_integral_exact_states"])):
        warnings.append("The integral of the gridded DOS differs from the exact integral of "
                        "its Gaussians by more than 0.1 percent; refine the grid.")
    dos = run.result["dos"]
    labels = [p[0] for p in spec.projections]
    meta = {"reference": spec.energy_reference, "reference_eV": data["reference_eV"],
            "fermi_level_eV": data["fermi_level_eV"], "broadening": spec.broadening,
            "width_eV": spec.width_eV, "dos_spec_digest": spec.digest}
    energies = spec.energies()
    outcome.arrays = {
        "energies": StoredArray(energies, "eV", "DOS energy grid, relative to the reference",
                                "table", dict(meta)),
        "dos_total": StoredArray(np.asarray(arrays["dos_total"], dtype=float), "states/eV",
                                 "Total DOS per cell, both spins", "table", dict(meta)),
        "eigenvalues": StoredArray(data["eigen"], "eV",
                                   "Kohn-Sham eigenvalues of the DOS grid [spin, IBZ k, band]",
                                   "table", {"axes": ["spin", "kpoint", "band"],
                                             "absolute": True}),
        "kpoint_weights": StoredArray(data["weights"], "", "IBZ k-point weights, sum 1",
                                      "table", {}),
    }
    if spec.spin_channels == "resolved":
        outcome.arrays["dos_spin"] = StoredArray(
            np.asarray(arrays["dos_spin"], dtype=float), "states/eV",
            "DOS per spin channel [up, down]", "table", dict(meta, axes=["spin", "energy"]))
    if spec.projections:
        outcome.arrays["pdos"] = StoredArray(
            np.asarray(arrays["pdos_total"], dtype=float), "states/eV",
            "Projected DOS per selection, both spins", "table",
            dict(meta, labels=labels, axes=["projection", "energy"]))
        if spec.spin_channels == "resolved":
            outcome.arrays["pdos_spin"] = StoredArray(
                np.asarray(arrays["pdos_spin"], dtype=float), "states/eV",
                "Projected DOS per selection and spin [projection, spin, energy]", "table",
                dict(meta, labels=labels, axes=["projection", "spin", "energy"]))
    results = dict(staged.results)
    template = results["energy"].provenance
    summary = {
        "status": STATUS_COMPLETE, "energy_reference": spec.energy_reference,
        "reference_eV": data["reference_eV"], "fermi_level_eV": data["fermi_level_eV"],
        "energy_min_eV": spec.energy_min_eV, "energy_max_eV": spec.energy_max_eV,
        "energy_step_eV": spec.energy_step_eV, "npoints": spec.npoints,
        "broadening": spec.broadening, "width_eV": spec.width_eV,
        "spin_channels": spec.spin_channels, "n_spins": int(len(data["eigen"])),
        "dos_kpoints": list(spec.dos_kpoints), "n_bands": spec.n_bands,
        "n_ibz_kpoints": int(len(data["weights"])),
        "n_bz_kpoints": int(dos.get("n_bz_kpoints") or 0),
        "projections": [{"label": p[0], "atom_ids": list(p[1]), "angular": p[2],
                         "projectors_per_atom": c}
                        for p, c in zip(spec.projections, dos.get("projector_counts") or [])],
        "checks": checks, "source": dict(spec.source),
        "arrays": {name: f"dftdos::{run_id}::{name}" for name in outcome.arrays},
        "unit": "states/eV per cell", "used": dos.get("used"),
        "nscf_parameters_used": dos.get("nscf_parameters_used"),
    }
    convergence = Convergence(
        converged=True, iterations=int(run.iterations),
        residual=data["recompute_error"],
        residual_metric="largest relative difference between GPAW's DOS and its "
                        "recomputation from the returned eigenvalues",
        tolerance=RECOMPUTE_TOLERANCE,
        message=(f"Ground state converged in {run.iterations} SCF iterations; every band of "
                 f"the non-self-consistent step converged; the DOS recomputed from the "
                 f"eigenvalues agrees to {data['recompute_error']:.1e}."))
    record = Result("dft_dos", summary, "states/eV", ground._copy_provenance(template),
                    convergence=convergence)
    record.extra.update(base_extra)
    results["dos"] = record
    results["fermi_level"].value = data["fermi_level_eV"]
    parameters = {"dos_spec": spec.as_dict(), "dos_spec_digest": spec.digest,
                  "nscf_parameters": job["dos"]["nscf"],
                  "nscf_parameters_used": dos.get("nscf_parameters_used"),
                  "dos_settings": job["dos"], "dos_used": dos.get("used"),
                  "source": dict(spec.source)}
    approximations = _approximations(spec)
    for item in results.values():
        item.provenance.model = MODEL
        item.provenance.fidelity = Fidelity.TIER3_EXTERNAL
        item.provenance.origin = Origin.CALCULATED
        item.provenance.inputs_digest = spec.digest
        item.provenance.parameters.update(parameters)
        item.provenance.approximations = list(item.provenance.approximations) + [
            a for a in approximations if a not in item.provenance.approximations]
        item.extra["warnings"] = list(warnings)
    outcome.results = results
    outcome.status = STATUS_COMPLETE
    outcome.warnings = warnings
    return outcome
