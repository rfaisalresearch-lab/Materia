"""DFT geometry relaxation: a ground-state experiment whose nuclei, and cell, move.

A :class:`RelaxationSpec` is a frozen
:class:`~materia.experiments.dft.spec.GroundStateSpec` together with the
variables of the ionic search: whether the cell is fixed or variable, the
optimiser and its step length, the force and stress criteria, the step limit,
the strain components and pressure of a variable-cell search, whether GPAW
keeps the symmetry of the starting geometry, and which atoms are fixed.  It is
immutable and versioned (:data:`SCHEMA`, :data:`VERSION`).  Every electronic
variable, and every refusal for it, comes from the ground-state specification
unchanged; :func:`check` adds the rules that belong to moving the nuclei.

Every variable reaches the job the GPAW worker runs.  The electronic ones are
echoed back by GPAW and compared with
:func:`~materia.experiments.dft.run.parameter_mismatch`; the optimiser, its
step length, the fixed atoms and the cell filter are reported back by the
worker from the objects it built and compared the same way, so a variable
that was accepted and then ignored fails the run.

:func:`execute` never touches a project.  A run that is cancelled, times out,
fails or reports anything inconsistent yields no results at all; a finished
run yields the relaxation record, its history and the ground-state
observables at the final geometry, computed by the same code as a
ground-state run.  Storing and applying are in
:mod:`~materia.experiments.dft.records`.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ...project_format.arrays import StoredArray
from ...provenance import Convergence, Fidelity, Origin, Result
from ...solvers.gpaw_driver import runner
from . import boundary as bc
from . import run as ground
from . import spec as specs

SCHEMA = "materia.dft.relaxation"
VERSION = "1.0"

MODEL = "external:gpaw/dft-relaxation"

MODES: Tuple[str, ...] = ("fixed-cell", "variable-cell")
OPTIMIZERS: Tuple[str, ...] = ("BFGS", "FIRE")
SYMMETRY: Tuple[str, ...] = ("preserve", "off")
VOIGT: Tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")

EV_PER_A3_TO_GPA = ground.EV_PER_A3_TO_GPA

STATUS_CONVERGED = "converged"
STATUS_NOT_CONVERGED = "not-converged"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
SUCCESS = (STATUS_CONVERGED, STATUS_NOT_CONVERGED)

MAX_STEPS = 1000
PULAY_CUTOFF_EV = 500.0
POSITION_TOLERANCE_A = 1e-8

SYMMETRY_PARAMETERS = {
    "preserve": {"point_group": True, "time_reversal": True},
    "off": {"point_group": False, "time_reversal": False},
}

FIELDS: Tuple[specs.FieldInfo, ...] = (
    specs.FieldInfo("mode", "relaxation", "relaxation", "",
                    "fixed-cell moves the nuclei inside a fixed cell. variable-cell also "
                    "changes the cell until the stress matches the target pressure; it needs "
                    "the plane-wave stress, so only a bulk crystal in pw mode."),
    specs.FieldInfo("optimizer", "relaxation", "optimiser", "",
                    "BFGS (quasi-Newton, ASE) or FIRE (damped dynamics, ASE)."),
    specs.FieldInfo("fmax_eV_A", "relaxation", "force criterion", "eV/A",
                    "The relaxation has converged when the largest force on any mobile atom "
                    "is at most this."),
    specs.FieldInfo("max_steps", "relaxation", "ionic step limit", "steps",
                    "Optimiser steps allowed. A relaxation that reaches it has not converged "
                    "and cannot be applied."),
    specs.FieldInfo("maxstep_A", "relaxation", "largest step", "A",
                    "Largest move of any atom in one optimiser step.", advanced=True),
    specs.FieldInfo("symmetry", "relaxation", "symmetry", "",
                    "preserve keeps GPAW's point-group and time-reversal symmetry: forces are "
                    "symmetrised, so the structure cannot lower the symmetry it starts with. "
                    "off lets it lower its symmetry, at the cost of more k-points. Defaults to "
                    "off when atoms are fixed, because a symmetry that maps a fixed atom onto "
                    "a mobile one would be broken by the first step."),
    specs.FieldInfo("stress_tol_eV_A3", "relaxation", "stress criterion", "eV/A^3",
                    "Largest deviation of a free stress component from minus the target "
                    "pressure.", applies="variable-cell"),
    specs.FieldInfo("cell_mask", "relaxation", "free strain components", "",
                    "Which of xx, yy, zz, yz, xz, xy may change.", applies="variable-cell",
                    advanced=True),
    specs.FieldInfo("hydrostatic_strain", "relaxation", "hydrostatic strain only", "",
                    "Scale the cell uniformly, keeping its shape.", applies="variable-cell",
                    advanced=True),
    specs.FieldInfo("target_pressure_GPa", "relaxation", "target pressure", "GPa",
                    "External hydrostatic pressure; the cell relaxes to stress = -p.",
                    applies="variable-cell"),
    specs.FieldInfo("fixed_atom_ids", "relaxation", "fixed atoms", "",
                    "Atoms marked fixed on the structure. They do not move and are excluded "
                    "from the force criterion.", settable=False),
)
FIELD_INDEX = {f.name: f for f in FIELDS}
SETTABLE: Tuple[str, ...] = tuple(f.name for f in FIELDS if f.settable)
VARIABLE_CELL_ONLY = ("stress_tol_eV_A3", "cell_mask", "hydrostatic_strain",
                      "target_pressure_GPa")

DEFAULTS: Dict[str, Any] = {"mode": "fixed-cell", "optimizer": "BFGS", "fmax_eV_A": 0.05,
                            "max_steps": 100, "maxstep_A": 0.2, "symmetry": "preserve"}
VARIABLE_CELL_DEFAULTS: Dict[str, Any] = {"stress_tol_eV_A3": 0.005,
                                          "cell_mask": (True,) * 6,
                                          "hydrostatic_strain": False,
                                          "target_pressure_GPa": 0.0}


@dataclass(frozen=True)
class RelaxationSpec:
    """One relaxation, frozen.  See :data:`FIELDS` and the ground-state fields."""

    ground_state: specs.GroundStateSpec
    mode: str
    optimizer: str
    fmax_eV_A: float
    max_steps: int
    maxstep_A: float
    symmetry: str
    stress_tol_eV_A3: Optional[float]
    cell_mask: Optional[Tuple[bool, bool, bool, bool, bool, bool]]
    hydrostatic_strain: Optional[bool]
    target_pressure_GPa: Optional[float]
    fixed_atom_ids: Tuple[int, ...]
    structure_magnetic_moments: Tuple[float, ...]
    schema: str = SCHEMA
    version: str = VERSION

    def as_dict(self) -> dict:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = value.as_dict() if f.name == "ground_state" else specs._plain(value)
        return out

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RelaxationSpec":
        if data.get("schema") != SCHEMA:
            raise specs.SpecError(f"Not a relaxation specification: {data.get('schema')!r}.")
        if str(data.get("version")) != VERSION:
            raise specs.SpecError(f"Relaxation specification version {data.get('version')} "
                                  f"is not understood by this Materia (it reads {VERSION}).")
        names = {f.name for f in fields(RelaxationSpec)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise specs.SpecError(f"Unknown relaxation field(s): {', '.join(unknown)}.")
        missing = sorted(n for n in names - set(data) if n not in ("schema", "version"))
        if missing:
            raise specs.SpecError(f"Missing relaxation field(s): {', '.join(missing)}.")
        values = dict(data)
        values["ground_state"] = specs.GroundStateSpec.from_dict(data["ground_state"])
        values["fixed_atom_ids"] = tuple(int(i) for i in data["fixed_atom_ids"])
        values["structure_magnetic_moments"] = tuple(
            float(m) for m in data["structure_magnetic_moments"])
        if data.get("cell_mask") is not None:
            values["cell_mask"] = tuple(bool(v) for v in data["cell_mask"])
        values["schema"] = SCHEMA
        values["version"] = VERSION
        return RelaxationSpec(**values)

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
    def variable_cell(self) -> bool:
        return self.mode == "variable-cell"

    def settings(self) -> Dict[str, Any]:
        """Relaxation variables and the ground-state variables, as plain data."""
        out = {name: specs._plain(getattr(self, name)) for name in SETTABLE}
        out.update(self.ground_state.settings())
        return out

    def fixed_indices(self) -> List[int]:
        position = {int(i): k for k, i in enumerate(self.ground_state.atom_ids)}
        return sorted(position[i] for i in self.fixed_atom_ids if i in position)


def _coerce(name: str, value: Any) -> Any:
    if name == "mode":
        return specs._choice(name, value, MODES)
    if name == "optimizer":
        return specs._choice(name, value, OPTIMIZERS)
    if name == "symmetry":
        return specs._choice(name, value, SYMMETRY)
    if name in ("fmax_eV_A", "maxstep_A"):
        return specs._number(name, value, FIELD_INDEX[name].unit)
    if name in ("stress_tol_eV_A3", "target_pressure_GPa"):
        return specs._optional_number(name, value, FIELD_INDEX[name].unit)
    if name == "max_steps":
        return specs._integer(name, value)
    if name == "hydrostatic_strain":
        return None if value is None else specs._boolean(name, value)
    if name == "cell_mask":
        if value is None:
            return None
        if isinstance(value, str):
            raise specs.SpecError("cell_mask must list six true or false values, one per "
                                  "strain component xx, yy, zz, yz, xz, xy.")
        try:
            items = list(value)
        except TypeError:
            raise specs.SpecError(f"cell_mask must be a list of six booleans, got "
                                  f"{value!r}.") from None
        if len(items) != 6:
            raise specs.SpecError(f"cell_mask must have six entries (xx, yy, zz, yz, xz, "
                                  f"xy), got {len(items)}.")
        return tuple(specs._boolean("cell_mask", v) for v in items)
    raise specs.SpecError(f"{name} is not a relaxation variable.")


def _split(variables: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    relax = {k: v for k, v in variables.items() if k in SETTABLE}
    rest = {k: v for k, v in variables.items() if k not in SETTABLE}
    if "fixed_atom_ids" in rest:
        raise specs.SpecError("fixed_atom_ids comes from the structure; mark atoms fixed on "
                              "the structure instead.")
    return relax, rest


def build(structure, environment=None, structure_key: Optional[str] = None,
          **variables: Any) -> RelaxationSpec:
    """A relaxation specification for ``structure``: defaults, then ``variables``.

    Relaxation variables are those in :data:`SETTABLE`; every other variable
    goes to :func:`materia.experiments.dft.spec.build` unchanged.  When the
    observables are not given, forces are requested, and the stress too for a
    variable-cell relaxation.  Not validated here; :func:`check` does that.
    """
    relax, rest = _split(variables)
    values = dict(DEFAULTS)
    for name, value in relax.items():
        values[name] = _coerce(name, value)
    mode = values["mode"]
    for name in VARIABLE_CELL_ONLY:
        if name not in relax:
            values[name] = VARIABLE_CELL_DEFAULTS[name] if mode == "variable-cell" else None
    environment = specs._environment(environment)
    gs = specs.build(structure, environment, structure_key=structure_key, **rest)
    if "observables" not in rest:
        wanted = set(gs.observables) | {"energy", "forces"}
        if mode == "variable-cell":
            wanted.add("stress")
        gs = replace(gs, observables=tuple(sorted(wanted)))
    fixed = tuple(int(i) for i, flag in zip(structure.ids, structure.fixed) if flag)
    if fixed and "symmetry" not in relax:
        values["symmetry"] = "off"
    return RelaxationSpec(
        ground_state=gs, mode=mode, optimizer=values["optimizer"],
        fmax_eV_A=float(values["fmax_eV_A"]), max_steps=int(values["max_steps"]),
        maxstep_A=float(values["maxstep_A"]), symmetry=values["symmetry"],
        stress_tol_eV_A3=values["stress_tol_eV_A3"], cell_mask=values["cell_mask"],
        hydrostatic_strain=values["hydrostatic_strain"],
        target_pressure_GPa=values["target_pressure_GPa"], fixed_atom_ids=fixed,
        structure_magnetic_moments=tuple(float(m) for m in structure.magnetic_moments))


def changed(spec: RelaxationSpec, environment=None, **variables: Any) -> RelaxationSpec:
    """A new specification with some relaxation or ground-state variables changed."""
    relax, rest = _split(variables)
    unknown = sorted(set(rest) - set(specs.SETTABLE))
    if unknown:
        raise specs.SpecError(f"Cannot change {', '.join(unknown)}: not a settable variable.")
    values = {name: _coerce(name, value) for name, value in relax.items()}
    updated = replace(spec, **values)
    if "mode" in values and values["mode"] != spec.mode:
        for name in VARIABLE_CELL_ONLY:
            if name not in values:
                updated = replace(updated, **{name: VARIABLE_CELL_DEFAULTS[name]
                                              if values["mode"] == "variable-cell" else None})
    if rest:
        updated = replace(updated, ground_state=specs.changed(spec.ground_state, environment,
                                                              **rest))
    return updated


def check(spec: RelaxationSpec, environment=None) -> specs.SpecReport:
    """Every ground-state rule plus every relaxation rule, keyed by variable."""
    environment = specs._environment(environment)
    report = specs.check(spec.ground_state, environment)
    by_field = {k: list(v) for k, v in report.by_field.items()}
    warnings = list(report.warnings)
    general = [t for t in report.blocking
               if not any(t in texts for texts in report.by_field.values())]

    def refuse(name: str, text: str) -> None:
        by_field.setdefault(name, []).append(text)

    gs = spec.ground_state
    if spec.schema != SCHEMA or spec.version != VERSION:
        general.append(f"Relaxation specification {spec.schema} {spec.version} is not "
                       f"{SCHEMA} {VERSION}.")
    if spec.mode not in MODES:
        refuse("mode", f"mode must be one of {', '.join(MODES)}.")
    if spec.optimizer not in OPTIMIZERS:
        refuse("optimizer", f"optimizer must be one of {', '.join(OPTIMIZERS)}.")
    if spec.symmetry not in SYMMETRY:
        refuse("symmetry", f"symmetry must be one of {', '.join(SYMMETRY)}.")
    if not (0.0 < spec.fmax_eV_A <= 1.0):
        refuse("fmax_eV_A", f"fmax_eV_A = {spec.fmax_eV_A:g} eV/A must be positive and at "
                            "most 1 eV/A; a looser criterion does not locate a minimum.")
    elif spec.fmax_eV_A < 0.005:
        warnings.append(f"A {spec.fmax_eV_A:g} eV/A force criterion is below the numerical "
                        "noise of most DFT forces; tighten the SCF force tolerance too.")
    if not (1 <= spec.max_steps <= MAX_STEPS):
        refuse("max_steps", f"max_steps must be 1 to {MAX_STEPS}, got {spec.max_steps}.")
    if not (0.0 < spec.maxstep_A <= 0.5):
        refuse("maxstep_A", f"maxstep_A = {spec.maxstep_A:g} A must be positive and at most "
                            "0.5 A; larger steps skip over minima.")
    observables = set(gs.observables)
    if "forces" not in observables:
        refuse("observables", "A relaxation follows the forces, so 'forces' must be among "
                              "the observables.")
    tolerance = gs.forces_tol_eV_A
    if tolerance is None:
        warnings.append("No SCF force tolerance is set: the SCF stops on energy, density "
                        "and eigenstates, and the forces the optimiser follows may carry "
                        f"noise comparable to the {spec.fmax_eV_A:g} eV/A criterion. Set "
                        "forces_tol_eV_A to a fraction of it.")
    elif tolerance > spec.fmax_eV_A:
        refuse("forces_tol_eV_A", f"The SCF force tolerance {tolerance:g} eV/A is looser "
                                  f"than the {spec.fmax_eV_A:g} eV/A relaxation criterion, "
                                  "so the relaxation could stop on forces that are not "
                                  "converged. Make it smaller than fmax_eV_A.")
    ids = set(gs.atom_ids)
    if not set(spec.fixed_atom_ids) <= ids:
        refuse("fixed_atom_ids", "A fixed atom is not in the structure.")
    if len(spec.structure_magnetic_moments) != gs.n_atoms:
        refuse("structure_magnetic_moments", "The recorded magnetic moments do not match "
                                             "the atom count.")
    field_on = bool(np.any(np.asarray(gs.external_field_V_per_A, dtype=float)))
    if field_on and abs(gs.charge_e) > 1e-9:
        refuse("external_field_V_per_A",
               f"A system with net charge {gs.charge_e:+g} e in a uniform field feels a net "
               "force qE and has no equilibrium geometry; the relaxation would drift "
               "towards the box edge. Relax the neutral system, or remove the field.")
    if spec.mode == "fixed-cell":
        for name in VARIABLE_CELL_ONLY:
            if getattr(spec, name) is not None:
                refuse(name, f"{name} applies only to a variable-cell relaxation; for a "
                             "fixed cell leave it empty.")
        if gs.n_atoms and len(spec.fixed_atom_ids) >= gs.n_atoms:
            refuse("fixed_atom_ids", "Every atom is fixed and the cell is fixed, so there is "
                                     "nothing to relax.")
    if spec.mode == "variable-cell":
        if gs.boundary != "bulk":
            refuse("mode", f"A variable-cell relaxation needs the stress tensor, which GPAW "
                           f"computes only in plane-wave mode for a bulk crystal; this is a "
                           f"{gs.boundary}. Relax it at fixed cell.")
        if gs.representation != "pw":
            refuse("mode", f"A variable-cell relaxation needs the stress tensor, which GPAW "
                           f"computes only in plane-wave mode, not {gs.representation}.")
        if "stress" not in observables:
            refuse("observables", "A variable-cell relaxation follows the stress, so "
                                  "'stress' must be among the observables.")
        if spec.fixed_atom_ids:
            refuse("fixed_atom_ids", f"{len(spec.fixed_atom_ids)} atom(s) are fixed, but a "
                                     "change of cell moves every atom with it, so they would "
                                     "not stay where they are. Clear the fixed atoms or relax "
                                     "at fixed cell.")
        if abs(gs.charge_e) > 1e-9:
            refuse("charge_e", "The stress of a charged cell with a uniform compensating "
                               "background includes an unphysical background term that "
                               "depends on the volume; a variable-cell relaxation of it is "
                               "not supported.")
        if spec.stress_tol_eV_A3 is None or not (0.0 < spec.stress_tol_eV_A3 <= 0.1):
            refuse("stress_tol_eV_A3", "stress_tol_eV_A3 must be positive and at most "
                                       "0.1 eV/A^3 (16 GPa).")
        mask = spec.cell_mask
        if mask is None or len(mask) != 6:
            refuse("cell_mask", "cell_mask must list six booleans (xx, yy, zz, yz, xz, xy).")
        elif not any(mask):
            refuse("cell_mask", "No strain component is free; that is a fixed-cell "
                                "relaxation. Choose mode fixed-cell.")
        elif spec.hydrostatic_strain and not all(mask):
            refuse("hydrostatic_strain", "Hydrostatic strain scales the whole cell, so a "
                                         "partial cell_mask would be ignored. Free every "
                                         "component or turn hydrostatic strain off.")
        if spec.hydrostatic_strain is None:
            refuse("hydrostatic_strain", "hydrostatic_strain must be true or false.")
        pressure = spec.target_pressure_GPa
        if pressure is None or not math.isfinite(pressure) or abs(pressure) > 100.0:
            refuse("target_pressure_GPa", "target_pressure_GPa must be a number between "
                                          "-100 and 100 GPa.")
        if gs.cutoff_eV is not None and gs.cutoff_eV < PULAY_CUTOFF_EV:
            warnings.append(f"A variable-cell relaxation at {gs.cutoff_eV:g} eV: at a fixed "
                            "cutoff the number of plane waves changes with the volume, "
                            "which adds a Pulay stress. Use at least "
                            f"{PULAY_CUTOFF_EV:g} eV, or check the equilibrium volume "
                            "against the cutoff.")
        if gs.spin_polarized:
            warnings.append("Each change of cell rebuilds the GPAW calculation and restarts "
                            "the SCF from the initial magnetic moments, so the magnetic "
                            "state is re-found at every cell step.")
    if spec.symmetry == "preserve" and spec.fixed_atom_ids:
        refuse("symmetry", f"{len(spec.fixed_atom_ids)} atom(s) are fixed, but GPAW "
                           "detects symmetry from the atoms alone and symmetrises the forces "
                           "without knowing about the constraint. A symmetry that maps a "
                           "fixed atom onto a mobile one is broken by the first step and "
                           "GPAW stops. Use symmetry off with fixed atoms.")
    elif spec.symmetry == "preserve":
        warnings.append("Symmetry preserved: GPAW symmetrises the forces, so the relaxed "
                        "structure keeps every symmetry of the starting geometry and cannot "
                        "find a lower-symmetry minimum.")
    blocking = list(general)
    for texts in by_field.values():
        for text in texts:
            if text not in blocking:
                blocking.append(text)
    options = dict(report.options)
    options.update({
        "modes": ["fixed-cell", "variable-cell"] if (
            gs.boundary == "bulk" and gs.representation == "pw") else ["fixed-cell"],
        "optimizers": list(OPTIMIZERS), "symmetry": list(SYMMETRY),
        "voigt": list(VOIGT),
    })
    return specs.SpecReport(blocking, warnings, by_field, report.boundary, report.electrons,
                            options)


def require(spec: RelaxationSpec, environment=None) -> specs.SpecReport:
    report = check(spec, environment)
    if not report.ok:
        raise specs.SpecRefused(report)
    return report


def gpaw_parameters(spec: RelaxationSpec) -> Dict[str, Any]:
    """The keyword arguments handed to ``gpaw.GPAW`` for every ionic step."""
    parameters = specs.gpaw_parameters(spec.ground_state)
    parameters["symmetry"] = dict(SYMMETRY_PARAMETERS[spec.symmetry])
    return parameters


def relaxation_settings(spec: RelaxationSpec) -> Dict[str, Any]:
    """The ionic-search block of the worker job, in the worker's units (eV, A)."""
    cell_filter = None
    if spec.variable_cell:
        cell_filter = {"name": "FrechetCellFilter",
                       "mask": [bool(v) for v in (spec.cell_mask or (True,) * 6)],
                       "hydrostatic_strain": bool(spec.hydrostatic_strain),
                       "scalar_pressure": float(spec.target_pressure_GPa or 0.0)
                       / EV_PER_A3_TO_GPA}
    return {"mode": spec.mode, "optimizer": spec.optimizer,
            "maxstep": float(spec.maxstep_A), "fmax": float(spec.fmax_eV_A),
            "stress_tol": (None if spec.stress_tol_eV_A3 is None
                           else float(spec.stress_tol_eV_A3)),
            "max_steps": int(spec.max_steps), "fixed_indices": spec.fixed_indices(),
            "filter": cell_filter}


def worker_job(spec: RelaxationSpec) -> Dict[str, Any]:
    gs = spec.ground_state
    return {
        "structure": specs.worker_structure(gs),
        "parameters": gpaw_parameters(spec),
        "observables": list(gs.observables),
        "expected_datasets": {s: {"path": p, "sha256": h} for s, p, h in gs.paw_datasets},
        "relaxation": relaxation_settings(spec),
    }


def describe(spec: RelaxationSpec, report: Optional[specs.SpecReport] = None) -> dict:
    """The relaxation variables and the ground-state description, for display."""
    report = report if report is not None else check(spec)
    items = []
    for info in FIELDS:
        item = info.as_dict()
        item["value"] = specs._plain(getattr(spec, info.name))
        item["problems"] = list(report.by_field.get(info.name, []))
        items.append(item)
    return {"schema": spec.schema, "version": spec.version, "digest": spec.digest,
            "short_digest": spec.short_digest, "relaxation": items,
            "settings": spec.settings(), "ground_state": specs.describe(spec.ground_state,
                                                                        report),
            "gpaw_parameters": gpaw_parameters(spec),
            "relaxation_settings": relaxation_settings(spec)}


def structure_of(spec: RelaxationSpec, positions=None, cell=None):
    """The frozen structure, or the same atoms at another geometry."""
    from ...core_model.cell import Cell
    from ...core_model.structure import Structure

    gs = spec.ground_state
    s = Structure(list(gs.numbers),
                  np.asarray(gs.positions_A if positions is None else positions, dtype=float),
                  Cell(np.asarray(gs.cell_A if cell is None else cell, dtype=float),
                       tuple(gs.pbc)),
                  ids=list(gs.atom_ids), mass_numbers=list(gs.mass_numbers))
    s.formal_charges[:] = np.asarray(gs.formal_charges_e, dtype=float)
    s.magnetic_moments[:] = np.asarray(spec.structure_magnetic_moments, dtype=float)
    return s


@dataclass
class RelaxationOutcome:
    """Everything one relaxation produced, not yet stored anywhere."""

    run_id: str
    spec: RelaxationSpec
    status: str
    reason: str = ""
    results: Dict[str, Result] = field(default_factory=dict)
    arrays: Dict[str, StoredArray] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    final_spec: Optional[specs.GroundStateSpec] = None
    output_geometry_digest: str = ""
    wall_time_s: float = 0.0
    audit: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in SUCCESS

    @property
    def converged(self) -> bool:
        return self.status == STATUS_CONVERGED


class _Invalid(Exception):
    pass


def _approximations(spec: RelaxationSpec) -> List[str]:
    out = [
        f"Geometry relaxed with the ASE {spec.optimizer} optimiser on GPAW forces"
        + (" and the plane-wave stress through ASE's FrechetCellFilter"
           if spec.variable_cell else " at fixed cell") + ". A relaxation finds a local "
        "minimum near the starting geometry, not the global one.",
        f"Converged when every mobile atom's force is at most {spec.fmax_eV_A:g} eV/A"
        + (f" and every free stress component is within {spec.stress_tol_eV_A3:g} eV/A^3 "
           f"of -({spec.target_pressure_GPa:g} GPa)" if spec.variable_cell else "") + ".",
    ]
    if spec.symmetry == "preserve":
        out.append("GPAW point-group and time-reversal symmetry kept: forces are "
                   "symmetrised, so no symmetry of the starting geometry can be broken.")
    else:
        out.append("Symmetry off: no symmetrisation of forces or k-points.")
    if spec.fixed_atom_ids:
        out.append(f"{len(spec.fixed_atom_ids)} atom(s) held fixed and excluded from the "
                   "force criterion.")
    if spec.variable_cell:
        out.append("The k-point divisions and the plane-wave cutoff are held fixed while the "
                   "cell changes, so the k-point density and the number of plane waves "
                   "change with it.")
    return out


def _cell_report(initial: np.ndarray, final: np.ndarray) -> Dict[str, Any]:
    from ...core_model.cell import Cell

    before = Cell(initial, (True, True, True))
    after = Cell(final, (True, True, True))
    v0 = abs(float(np.linalg.det(initial)))
    v1 = abs(float(np.linalg.det(final)))
    strain = None
    if v0 > 1e-12:
        gradient = np.linalg.solve(initial, final)
        strain = (0.5 * (gradient + gradient.T) - np.eye(3)).tolist()
    return {"initial_cell_A": initial.tolist(), "final_cell_A": final.tolist(),
            "initial_volume_A3": v0, "final_volume_A3": v1,
            "volume_change_percent": (100.0 * (v1 - v0) / v0) if v0 > 1e-12 else 0.0,
            "initial_lengths_A": before.lengths.tolist(),
            "final_lengths_A": after.lengths.tolist(),
            "initial_angles_deg": before.angles_deg.tolist(),
            "final_angles_deg": after.angles_deg.tolist(),
            "strain": strain,
            "strain_note": "Symmetric small-strain tensor 0.5 (D + D^T) - I, with the "
                           "final cell = initial cell D in the row-vector convention."}


def _validate(spec: RelaxationSpec, run: runner.GPAWRun, job: Dict[str, Any]) -> None:
    result = run.result
    mismatch = ground.parameter_mismatch(job["parameters"], result.get("parameters_used"))
    if mismatch:
        raise _Invalid("GPAW did not run with the parameters Materia sent, so the result "
                       f"would not describe this specification: {mismatch}")
    sent = dict(job["relaxation"])
    mismatch = ground.parameter_mismatch(sent, result.get("relaxation_used"), "relaxation.")
    if mismatch:
        raise _Invalid("The optimiser the worker ran differs from the one specified: "
                       f"{mismatch}")
    relax = result.get("relaxation")
    if not isinstance(relax, dict) or not relax.get("steps"):
        raise _Invalid("The worker reported no relaxation steps.")
    n = spec.ground_state.n_atoms
    positions = np.asarray(relax.get("final_positions"), dtype=float)
    cell = np.asarray(relax.get("final_cell"), dtype=float)
    if positions.shape != (n, 3):
        raise _Invalid(f"The final geometry has shape {positions.shape}, not ({n}, 3); atoms "
                       "were lost or added.")
    if cell.shape != (3, 3):
        raise _Invalid("The final cell is not a 3x3 matrix.")
    if not (np.isfinite(positions).all() and np.isfinite(cell).all()):
        raise _Invalid("The final geometry is not finite.")
    for row in relax["steps"]:
        for key in ("energy_free_eV", "energy_eV", "max_force_eV_A"):
            value = row.get(key)
            if value is None or not math.isfinite(float(value)):
                raise _Invalid(f"Ionic step {row.get('step')} reported a non-finite {key}.")
    steps = [int(row.get("step", -1)) for row in relax["steps"]]
    if steps != list(range(len(steps))):
        raise _Invalid("The ionic steps are not numbered 0, 1, 2 and so on.")
    if len(steps) > spec.max_steps + 1:
        raise _Invalid("The worker took more ionic steps than the specification allows.")
    initial = np.asarray(spec.ground_state.positions_A, dtype=float)
    fixed = spec.fixed_indices()
    if fixed and float(np.abs(positions[fixed] - initial[fixed]).max()) > POSITION_TOLERANCE_A:
        raise _Invalid("A fixed atom moved during the relaxation.")
    if not spec.variable_cell and float(np.abs(cell - np.asarray(
            spec.ground_state.cell_A)).max()) > POSITION_TOLERANCE_A:
        raise _Invalid("The cell changed during a fixed-cell relaxation.")
    last = relax["steps"][-1]
    converged = bool(relax.get("converged"))
    meets = float(last["max_force_eV_A"]) <= spec.fmax_eV_A and (
        not spec.variable_cell or (last.get("stress_residual_eV_A3") is not None and
                                   float(last["stress_residual_eV_A3"])
                                   <= float(spec.stress_tol_eV_A3)))
    if converged != meets:
        raise _Invalid("The worker's convergence verdict does not match its own last step.")
    if converged != (run.status == runner.STATUS_CONVERGED):
        raise _Invalid("The worker's status does not match its relaxation verdict.")
    forces = result.get("forces_eV_per_A")
    if forces is None or np.asarray(forces, dtype=float).shape != (n, 3):
        raise _Invalid("The final forces are missing or have the wrong shape.")
    if spec.variable_cell and result.get("stress_eV_per_A3") is None:
        raise _Invalid("A variable-cell relaxation reported no final stress.")
    for key in ("energy_free_eV", "energy_extrapolated_eV"):
        value = result.get(key)
        if value is None or not math.isfinite(float(value)):
            raise _Invalid(f"The final {key} is missing or not finite.")


def execute(spec: RelaxationSpec, environment=None, *, run_id: Optional[str] = None,
            progress: Optional[Callable[[dict], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None) -> RelaxationOutcome:
    """Run one relaxation.  Raises :class:`~.spec.SpecRefused` if it may not run."""
    from ...python_api.api import structure_state_digest

    environment = specs._environment(environment)
    report = require(spec, environment)
    run_id = run_id or ground.new_run_id()
    started = time.perf_counter()
    warnings = list(report.warnings)
    job = worker_job(spec)
    outcome = RelaxationOutcome(run_id=run_id, spec=spec, status=STATUS_FAILED,
                                warnings=warnings)
    if cancelled is not None and cancelled():
        outcome.status = STATUS_CANCELLED
        outcome.reason = "Cancelled before GPAW was started. Nothing was kept."
        return outcome
    run = runner.run_job(job, environment, worker=ground.WORKER, progress=progress,
                         cancelled=cancelled, timeout_s=timeout_s)
    outcome.wall_time_s = time.perf_counter() - started
    outcome.audit = {"worker_status": run.status, "returncode": run.returncode,
                     "command": list(run.command), "stderr_tail": run.stderr[-2000:],
                     "log_tail": str(run.result.get("log_tail") or "")[-3000:],
                     "steps_reported": len((run.result.get("relaxation") or {})
                                           .get("steps") or [])}
    if run.status == runner.STATUS_CANCELLED or (cancelled is not None and cancelled()):
        outcome.status = STATUS_CANCELLED
        outcome.reason = (f"The relaxation was cancelled after {outcome.audit['steps_reported']} "
                          "ionic step(s). Nothing was kept and the project was not changed.")
        return outcome
    if run.status == runner.STATUS_FAILED:
        detail = run.error or "no error message was produced"
        if timeout_s is not None and "stopped the calculation after" in run.stderr:
            detail = f"GPAW was stopped after the {timeout_s:g} s time limit"
        outcome.reason = (f"The relaxation failed: {detail.rstrip('.')}. Nothing was kept "
                          "and the project was not changed.")
        return outcome
    try:
        _validate(spec, run, job)
    except _Invalid as exc:
        outcome.reason = f"{exc} Nothing was kept and the project was not changed."
        return outcome

    result = run.result
    relax = result["relaxation"]
    gs = spec.ground_state
    initial_positions = np.asarray(gs.positions_A, dtype=float)
    initial_cell = np.asarray(gs.cell_A, dtype=float)
    final_positions = np.asarray(relax["final_positions"], dtype=float)
    final_cell = np.asarray(relax["final_cell"], dtype=float)
    produced = structure_of(spec, final_positions, final_cell)
    output_digest = specs.geometry_digest(produced)
    final_spec = replace(gs, positions_A=tuple(tuple(float(x) for x in row)
                                               for row in final_positions),
                         cell_A=tuple(tuple(float(x) for x in row) for row in final_cell),
                         geometry_digest=output_digest)
    geometry = bc.inspect(final_positions, final_cell, gs.pbc)
    for text in list(geometry.problems) + list(geometry.warnings):
        warnings.append(f"Final geometry: {text}")
    status = STATUS_CONVERGED if relax.get("converged") else STATUS_NOT_CONVERGED
    outcome.status = status
    outcome.final_spec = final_spec
    outcome.output_geometry_digest = output_digest

    restart = {"key": None, "reused_from": None, "loaded": False, "kept": False,
               "reason": "restart data does not apply to a relaxation, whose geometry "
                         "changes at every step"}
    final_run = runner.GPAWRun(status=runner.STATUS_CONVERGED, result=result,
                               stderr=run.stderr, returncode=run.returncode,
                               wall_time_s=run.wall_time_s,
                               iterations=len(result.get("scf_history") or []),
                               command=run.command, arrays=run.arrays,
                               events=run.events)
    history = result.get("scf_history") or []
    scf = ground._convergence(final_spec, final_run)
    base_extra = {
        "run_id": run_id, "structure_key": gs.structure_key,
        "geometry_digest": output_digest, "input_geometry_digest": gs.geometry_digest,
        "spec_digest": final_spec.digest, "relaxation_spec_digest": spec.digest,
        "mass_numbers": list(gs.mass_numbers), "status": status,
        "classification": ground.CLASSIFICATION, "warnings": warnings,
        "software_warnings": [str(w) for w in (result.get("warnings") or [])],
        "study_id": None, "n_atoms": gs.n_atoms, "formula": gs.formula,
        "boundary": gs.boundary, "kind": "relaxation", "mode": spec.mode,
    }
    staged = ground.GroundStateOutcome(run_id=run_id, spec=final_spec,
                                       status=runner.STATUS_CONVERGED, warnings=warnings)
    ground.collect(staged, final_spec, final_run, report, restart, base_extra, scf, history)
    outcome.results = staged.results
    outcome.arrays = staged.arrays

    rows = relax["steps"]
    first, last = rows[0], rows[-1]
    displacement = final_positions - initial_positions
    distances = np.linalg.norm(displacement, axis=1)
    ids = [int(i) for i in gs.atom_ids]
    worst = int(np.argmax(distances)) if len(distances) else 0
    convergence = Convergence(
        converged=status == STATUS_CONVERGED, iterations=len(rows) - 1,
        residual=float(last["max_force_eV_A"]),
        residual_metric="largest force on a mobile atom, eV/A",
        tolerance=spec.fmax_eV_A,
        message=(f"{relax.get('stop_reason', '')}: after {len(rows) - 1} optimiser step(s) "
                 f"the largest force on a mobile atom is {float(last['max_force_eV_A']):.3e} "
                 f"eV/A (criterion {spec.fmax_eV_A:g})"
                 + (f" and the largest stress deviation is "
                    f"{float(last['stress_residual_eV_A3']):.3e} eV/A^3 (criterion "
                    f"{spec.stress_tol_eV_A3:g})" if spec.variable_cell else "")
                 + ("." if status == STATUS_CONVERGED else
                    ". The criteria were not met, so this geometry is not a minimum and "
                    "cannot be applied.")))
    history_rows = [{k: row.get(k) for k in ("step", "energy_free_eV", "energy_eV",
                                             "max_force_eV_A", "stress_residual_eV_A3",
                                             "volume_A3", "scf_iterations")}
                    for row in rows]
    summary = {
        "status": status, "mode": spec.mode, "optimizer": spec.optimizer,
        "converged": status == STATUS_CONVERGED, "stop_reason": relax.get("stop_reason"),
        "optimizer_steps": len(rows) - 1,
        "scf_iterations_total": int(result.get("relaxation_scf_iterations") or 0),
        "initial_energy_eV": float(first["energy_free_eV"]),
        "final_energy_eV": float(last["energy_free_eV"]),
        "energy_change_eV": float(last["energy_free_eV"]) - float(first["energy_free_eV"]),
        "initial_max_force_eV_A": float(first["max_force_eV_A"]),
        "final_max_force_eV_A": float(last["max_force_eV_A"]),
        "final_stress_residual_eV_A3": last.get("stress_residual_eV_A3"),
        "displacement": {
            "max_A": float(distances.max()) if len(distances) else 0.0,
            "rms_A": float(np.sqrt(np.mean(distances ** 2))) if len(distances) else 0.0,
            "max_atom_id": ids[worst] if ids else None,
            "by_atom_id": {i: [float(v) for v in d] for i, d in zip(ids, displacement)},
        },
        "cell": _cell_report(initial_cell, final_cell),
        "fixed_atom_ids": list(spec.fixed_atom_ids),
        "history": history_rows,
        "echo": result.get("relaxation_used"),
        "input_geometry_digest": gs.geometry_digest,
        "output_geometry_digest": output_digest,
        "final_ground_state_digest": final_spec.digest,
        "input_structure_state_digest": structure_state_digest(structure_of(spec)),
    }
    prov_origin = Origin.CALCULATED if status == STATUS_CONVERGED else Origin.ESTIMATED
    extra = dict(base_extra)
    extra.update({"output_geometry_digest": output_digest, "wall_time_s": outcome.wall_time_s,
                  "worker_wall_time_s": run.wall_time_s,
                  "gpaw_log_tail": outcome.audit["log_tail"]})
    template = outcome.results["energy"].provenance
    record = Result("dft_relaxation", summary, "", ground._copy_provenance(template),
                    convergence=convergence)
    record.extra.update(extra)
    outcome.results["relaxation"] = record
    run_record = outcome.results["run"]
    run_record.value = dict(run_record.value or {})
    run_record.value.update({"status": status, "relaxation_spec_digest": spec.digest,
                             "kind": "relaxation"})
    run_record.convergence = convergence
    run_record.extra.update(extra)
    trajectory_positions = np.asarray(run.arrays.get("relax_positions"), dtype=float)
    trajectory_cells = np.asarray(run.arrays.get("relax_cells"), dtype=float)
    meta = {"atom_ids": ids, "numbers": [int(z) for z in gs.numbers],
            "pbc": list(gs.pbc), "steps": [int(r["step"]) for r in rows]}
    outcome.arrays.update({
        "initial_positions": StoredArray(initial_positions, "A",
                                         "Positions before the relaxation", "per-atom",
                                         dict(meta)),
        "final_positions": StoredArray(final_positions, "A",
                                       "Positions after the relaxation", "per-atom",
                                       dict(meta)),
        "initial_cell": StoredArray(initial_cell, "A", "Cell before the relaxation",
                                    "table", {}),
        "final_cell": StoredArray(final_cell, "A", "Cell after the relaxation", "table", {}),
        "final_forces": StoredArray(np.asarray(result["forces_eV_per_A"], dtype=float),
                                    "eV/A", "Forces at the final geometry", "per-atom",
                                    dict(meta)),
    })
    if trajectory_positions.ndim == 3 and trajectory_positions.shape[1:] == (gs.n_atoms, 3):
        outcome.arrays["step_positions"] = StoredArray(
            trajectory_positions, "A", "Positions at every ionic step", "trajectory",
            dict(meta))
    if trajectory_cells.ndim == 3 and trajectory_cells.shape[1:] == (3, 3):
        outcome.arrays["step_cells"] = StoredArray(
            trajectory_cells, "A", "Cell at every ionic step", "trajectory", dict(meta))
    relax_parameters = {"relaxation_spec": spec.as_dict(),
                        "relaxation_spec_digest": spec.digest,
                        "relaxation_settings": job["relaxation"],
                        "relaxation_used": result.get("relaxation_used"),
                        "gpaw_parameters": job["parameters"],
                        "initial_geometry_digest": gs.geometry_digest,
                        "output_geometry_digest": output_digest}
    approximations = _approximations(spec)
    for item in outcome.results.values():
        item.provenance.model = MODEL
        item.provenance.origin = prov_origin
        item.provenance.fidelity = Fidelity.TIER3_EXTERNAL
        item.provenance.inputs_digest = spec.digest
        item.provenance.parameters.update(relax_parameters)
        item.provenance.approximations = list(item.provenance.approximations) + [
            a for a in approximations if a not in item.provenance.approximations]
        item.extra["warnings"] = list(warnings)
        item.extra.setdefault("output_geometry_digest", output_digest)
    outcome.warnings = warnings
    return outcome
