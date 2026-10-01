"""Equation of state of a bulk crystal: energy against volume, through GPAW.

An :class:`EOSSpec` is a frozen ground-state specification of a reference
crystal together with the volumes to sample, as scale factors of the
reference volume.  Every point scales the cell and the atoms isotropically,
keeping fractional coordinates, and is a full ground-state experiment with
exactly the reference's electronic settings: the same functional, the same
plane-wave cutoff and, above all, the same k-point grid.  An automatic grid
chosen per cell would change between volumes and make the energies
incomparable; here the grid is pinned once and each point is checked to use
it.  It is immutable and versioned (:data:`SCHEMA`, :data:`VERSION`).

A real-space grid changes its number of points with the cell and steps the
energy curve, so only plane waves are accepted.  A slab, wire or cluster has
vacuum and no crystal volume, and a charged cell carries a volume-dependent
background term, so those are refused.

The equation of state is the static-lattice one at 0 K, so the fitted
energy is GPAW's energy extrapolated to zero smearing width, not the free
energy E - TS of the smeared occupations.  With fixed occupations the two are
identical.  With smeared occupations the free energy carries an electronic
entropy term that changes with volume and moves the minimum, so it is kept
as a separate curve and fitted separately.  GPAW's stress, like its forces,
is the derivative of that free energy at the finite width, so it is compared
with -dF/dV of the free-energy fit, never with the zero-width curve.
Specification version 1.0 fitted the free energy and is not read.

The energies per cell are fitted with the third-order Birch-Murnaghan
equation of state.  Nothing is kept unless every point converged with the
settings that were sent, the fitted minimum lies inside the sampled volumes,
B0 is positive, B' lies between 1 and 12 and the fit reproduces every point
to :data:`MAX_RESIDUAL_EV_PER_ATOM`.  Where GPAW reports stress, its pressure
is compared with -dE/dV of the fit; a large difference means the plane-wave
basis is not converged (Pulay stress).  Fractional coordinates are held
fixed, so a crystal with free internal parameters is not relaxed at each
volume; the largest force at every point is reported.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ...core_model.cell import Cell
from ...project_format.arrays import StoredArray
from ...provenance import Convergence, Fidelity, Origin, Result
from . import run as ground
from . import spec as specs
from .dos import electronic_digest

SCHEMA = "materia.dft.equation-of-state"
VERSION = "2.0"
ENERGY_DEFINITION = "zero-width extrapolated energy"
MODEL = "external:gpaw/equation-of-state"
FIT = "birch-murnaghan-3"
REFERENCES: Tuple[str, ...] = (
    "F. Birch, Phys. Rev. 71 (1947) 809",
    "F. D. Murnaghan, Proc. Natl. Acad. Sci. USA 30 (1944) 244",
)

EV_A3_TO_GPA = 160.21766208
MIN_SCALE = 0.7
MAX_SCALE = 1.3
MIN_POINTS = 5
MAX_POINTS = 15
MAX_RESIDUAL_EV_PER_ATOM = 2.0e-3
FORCE_WARNING_EV_A = 0.05
PRESSURE_WARNING_GPA = 1.0
SOURCES: Tuple[str, ...] = ("structure", "ground-state", "relaxation")
GEOMETRY_FIELDS = ("structure_key", "formula", "geometry_digest", "positions_A", "cell_A",
                   "observables")

STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

FIELDS: Tuple[specs.FieldInfo, ...] = (
    specs.FieldInfo("source", "eos", "reference crystal", "",
                    "Where the reference geometry and electronic settings come from: a "
                    "structure, a stored converged ground state, or a stored converged "
                    "relaxation.", settable=False),
    specs.FieldInfo("volume_min_scale", "eos", "smallest volume", "V / V_ref",
                    "Smallest volume sampled, as a fraction of the reference volume. 0.94 "
                    "is a linear strain of 2 percent."),
    specs.FieldInfo("volume_max_scale", "eos", "largest volume", "V / V_ref",
                    "Largest volume sampled, as a fraction of the reference volume."),
    specs.FieldInfo("n_points", "eos", "volumes", "points",
                    "Number of volumes, evenly spaced; 5 to 15. More points make the fit "
                    "and its residual more informative."),
    specs.FieldInfo("volume_scales", "eos", "sampled volumes", "V / V_ref",
                    "Every volume sampled, as stored.", settable=False),
    specs.FieldInfo("fit", "eos", "equation of state", "",
                    "Third-order Birch-Murnaghan, fitted to the energy per cell.",
                    settable=False),
)
FIELD_INDEX = {f.name: f for f in FIELDS}
SETTABLE: Tuple[str, ...] = tuple(f.name for f in FIELDS if f.settable)


def birch_murnaghan(volume, e0: float, v0: float, b0: float, b1: float):
    """Third-order Birch-Murnaghan energy, eV, for volume in A^3 and B0 in eV/A^3."""
    x = (v0 / np.asarray(volume, dtype=float)) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((x - 1.0) ** 3 * b1 + (x - 1.0) ** 2 * (6.0 - 4.0 * x))


def birch_murnaghan_pressure(volume, v0: float, b0: float, b1: float):
    """-dE/dV of :func:`birch_murnaghan`, eV/A^3."""
    eta = (v0 / np.asarray(volume, dtype=float)) ** (1.0 / 3.0)
    return 1.5 * b0 * (eta ** 7 - eta ** 5) * (1.0 + 0.75 * (b1 - 4.0) * (eta ** 2 - 1.0))


def fit_birch_murnaghan(volumes, energies) -> Dict[str, float]:
    """Least-squares third-order Birch-Murnaghan fit, started from a parabola."""
    from scipy.optimize import curve_fit

    v = np.asarray(volumes, dtype=float)
    e = np.asarray(energies, dtype=float)
    a, b, c = np.polyfit(v, e, 2)
    v_guess = -b / (2 * a) if a > 0 else float(v[np.argmin(e)])
    if not (v.min() * 0.8 < v_guess < v.max() * 1.2):
        v_guess = float(v[np.argmin(e)])
    b_guess = max(2 * a * v_guess, 1e-3)
    params, _ = curve_fit(birch_murnaghan, v, e,
                          p0=(float(e.min()), float(v_guess), float(b_guess), 4.0),
                          maxfev=20000)
    residual = e - birch_murnaghan(v, *params)
    return {"E0_eV": float(params[0]), "V0_A3": float(params[1]), "B0_eV_A3": float(params[2]),
            "B1": float(params[3]), "rms_eV": float(np.sqrt(np.mean(residual ** 2))),
            "max_abs_eV": float(np.abs(residual).max())}


@dataclass(frozen=True)
class EOSSpec:
    """One equation of state, frozen.  See :data:`FIELDS`."""

    ground_state: specs.GroundStateSpec
    source: Dict[str, Any]
    volume_min_scale: float
    volume_max_scale: float
    n_points: int
    volume_scales: Tuple[float, ...]
    fit: str = FIT
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
    def from_dict(data: Mapping[str, Any]) -> "EOSSpec":
        if data.get("schema") != SCHEMA:
            raise specs.SpecError(f"Not an equation-of-state specification: "
                                  f"{data.get('schema')!r}.")
        if str(data.get("version")) != VERSION:
            raise specs.SpecError(f"Equation-of-state specification version "
                                  f"{data.get('version')} is not understood by this Materia "
                                  f"(it reads {VERSION}).")
        names = {f.name for f in fields(EOSSpec)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise specs.SpecError(f"Unknown equation-of-state field(s): {', '.join(unknown)}.")
        missing = sorted(n for n in names - set(data) if n not in ("schema", "version"))
        if missing:
            raise specs.SpecError(f"Missing equation-of-state field(s): {', '.join(missing)}.")
        values = dict(data)
        values["ground_state"] = specs.GroundStateSpec.from_dict(data["ground_state"])
        values["source"] = dict(data["source"])
        values["volume_scales"] = tuple(float(v) for v in data["volume_scales"])
        values["volume_min_scale"] = float(data["volume_min_scale"])
        values["volume_max_scale"] = float(data["volume_max_scale"])
        values["n_points"] = int(data["n_points"])
        values["schema"] = SCHEMA
        values["version"] = VERSION
        return EOSSpec(**values)

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
    def reference_volume_A3(self) -> float:
        return abs(float(np.linalg.det(np.asarray(self.ground_state.cell_A, dtype=float))))

    def volumes(self) -> np.ndarray:
        return self.reference_volume_A3 * np.asarray(self.volume_scales, dtype=float)

    def settings(self) -> Dict[str, Any]:
        return {"volume_min_scale": self.volume_min_scale,
                "volume_max_scale": self.volume_max_scale, "n_points": self.n_points,
                "volume_scales": list(self.volume_scales), "fit": self.fit,
                "kpoints": list(self.ground_state.kpoints),
                "cutoff_eV": self.ground_state.cutoff_eV}


def scales(low: float, high: float, n: int) -> Tuple[float, ...]:
    return tuple(float(v) for v in np.linspace(low, high, int(n)))


def _observables(gs: specs.GroundStateSpec) -> Tuple[str, ...]:
    chosen = ["energy", "forces"]
    if gs.representation == "pw":
        chosen.append("stress")
    return tuple(sorted(chosen))


def _coerce(name: str, value: Any) -> Any:
    if name in ("volume_min_scale", "volume_max_scale"):
        return specs._number(name, value, "V / V_ref")
    if name == "n_points":
        return specs._integer(name, value)
    raise specs.SpecError(f"{name} is not an equation-of-state variable.")


def _split(variables: Mapping[str, Any]):
    eos_vars = {k: v for k, v in variables.items() if k in SETTABLE}
    rest = {k: v for k, v in variables.items() if k not in SETTABLE}
    for name in ("source", "volume_scales", "fit"):
        if name in rest:
            raise specs.SpecError(f"{name} is not settable; it follows from the reference and "
                                  "the volume range.")
    return eos_vars, rest


def _finish(gs: specs.GroundStateSpec, source: Dict[str, Any],
            eos_vars: Dict[str, Any]) -> EOSSpec:
    values = {name: _coerce(name, value) for name, value in eos_vars.items()}
    low = values.get("volume_min_scale", 0.94)
    high = values.get("volume_max_scale", 1.06)
    n = values.get("n_points", 7)
    gs = replace(gs, observables=_observables(gs))
    return EOSSpec(ground_state=gs, source=dict(source), volume_min_scale=float(low),
                   volume_max_scale=float(high), n_points=int(n),
                   volume_scales=scales(low, high, n) if n >= 1 else ())


def build(structure, environment=None, structure_key: Optional[str] = None,
          **variables: Any) -> EOSSpec:
    """An equation-of-state specification for ``structure``.

    Ground-state variables are those of
    :func:`materia.experiments.dft.spec.build`; the k-point grid they
    produce, automatic or explicit, is the one every volume uses.
    """
    eos_vars, rest = _split(variables)
    environment = specs._environment(environment)
    rest.pop("observables", None)
    gs = specs.build(structure, environment, structure_key=structure_key, **rest)
    gs = replace(gs, observables=_observables(gs))
    source = {"kind": "structure", "run_id": None, "spec_digest": None,
              "geometry_digest": gs.geometry_digest,
              "electronic_digest": electronic_digest(gs), "energy_eV": None}
    return _finish(gs, source, eos_vars)


def build_from_run(project, kind: str, run_id: str, environment=None,
                   **variables: Any) -> EOSSpec:
    """An equation of state around the geometry and settings of a stored run."""
    from . import dos

    eos_vars, rest = _split(variables)
    if rest:
        raise specs.SpecError(
            f"{', '.join(sorted(rest))}: an equation of state built from a stored {kind} run "
            "keeps that run's geometry and electronic settings exactly. Build it from the "
            "structure to change them.")
    original, energy = dos.source_spec(project, kind, run_id)
    gs = replace(original, observables=_observables(original))
    source = {"kind": kind, "run_id": run_id, "spec_digest": original.digest,
              "geometry_digest": original.geometry_digest,
              "electronic_digest": electronic_digest(original), "energy_eV": energy}
    return _finish(gs, source, eos_vars)


def changed(spec: EOSSpec, environment=None, **variables: Any) -> EOSSpec:
    eos_vars, rest = _split(variables)
    current = {"volume_min_scale": spec.volume_min_scale,
               "volume_max_scale": spec.volume_max_scale, "n_points": spec.n_points}
    current.update(eos_vars)
    gs, source = spec.ground_state, dict(spec.source)
    if rest:
        if spec.source.get("kind") != "structure":
            raise specs.SpecError(f"{', '.join(sorted(rest))}: this equation of state keeps "
                                  f"the electronic settings of {spec.source.get('kind')} run "
                                  f"{spec.source.get('run_id')}.")
        rest.pop("observables", None)
        gs = specs.changed(spec.ground_state, specs._environment(environment), **rest)
        gs = replace(gs, observables=_observables(gs))
        source["electronic_digest"] = electronic_digest(gs)
    return _finish(gs, source, current)


def structure_of(gs: specs.GroundStateSpec, scale: float = 1.0):
    """The reference crystal, its volume scaled by ``scale`` with fractional coordinates fixed."""
    from ...core_model.structure import Structure

    linear = float(scale) ** (1.0 / 3.0)
    crystal = Structure(np.asarray(gs.numbers, dtype=int),
                        np.asarray(gs.positions_A, dtype=float) * linear,
                        Cell(np.asarray(gs.cell_A, dtype=float) * linear, gs.pbc),
                        ids=list(gs.atom_ids), mass_numbers=list(gs.mass_numbers))
    crystal.formal_charges[:] = np.asarray(gs.formal_charges_e, dtype=float)
    crystal.magnetic_moments[:] = np.asarray(gs.initial_magnetic_moments_muB, dtype=float)
    return crystal


def _electronic_part(gs: specs.GroundStateSpec) -> Dict[str, Any]:
    data = gs.as_dict()
    for name in GEOMETRY_FIELDS:
        data.pop(name, None)
    return data


def point_specs(spec: EOSSpec, environment=None) -> List[specs.GroundStateSpec]:
    """One ground-state specification per volume, each with the reference's settings.

    Raises :class:`~.spec.SpecError` if any point would differ from the
    reference in anything but its geometry.
    """
    environment = specs._environment(environment)
    base = spec.ground_state
    settings = {k: v for k, v in base.settings().items() if k in specs.SETTABLE}
    settings["kpoints"] = list(base.kpoints)
    settings["observables"] = list(base.observables)
    reference = _electronic_part(base)
    out = []
    for scale in spec.volume_scales:
        point = specs.build(structure_of(base, scale), environment, structure_key=None,
                            **settings)
        if _electronic_part(point) != reference:
            differing = sorted(k for k in reference if _electronic_part(point).get(k)
                               != reference[k])
            raise specs.SpecError(f"The point at V/V_ref = {scale:g} would change "
                                  f"{', '.join(differing)}; every volume must use the "
                                  "reference's electronic settings.")
        out.append(point)
    return out


def check(spec: EOSSpec, environment=None) -> specs.SpecReport:
    environment = specs._environment(environment)
    gs = spec.ground_state
    report = specs.check(gs, environment)
    by_field = {k: list(v) for k, v in report.by_field.items()}
    warnings = [w for w in report.warnings if "automatic choice" not in w]
    general = [t for t in report.blocking
               if not any(t in texts for texts in report.by_field.values())]

    def refuse(name: str, text: str) -> None:
        by_field.setdefault(name, []).append(text)

    if spec.schema != SCHEMA or spec.version != VERSION:
        general.append(f"Equation-of-state specification {spec.schema} {spec.version} is not "
                       f"{SCHEMA} {VERSION}.")
    source = spec.source
    if source.get("kind") not in SOURCES:
        refuse("source", f"The reference must be one of {', '.join(SOURCES)}.")
    if source.get("geometry_digest") != gs.geometry_digest:
        refuse("source", "The geometry differs from the source's geometry.")
    if source.get("electronic_digest") != electronic_digest(gs):
        refuse("source", f"The electronic settings differ from those of the "
                         f"{source.get('kind')} source.")
    if gs.boundary != "bulk":
        refuse("source", f"A {gs.boundary} has vacuum, so its cell volume is not the volume of "
                         "a crystal and an equation of state would describe the box. Use a "
                         "bulk crystal.")
    if gs.representation != "pw":
        refuse("representation", "The equation of state needs plane waves. With a real-space "
                                 "grid or LCAO the number of grid points changes with the cell, "
                                 "which steps the energy curve.")
    if abs(gs.charge_e) > 1e-9:
        refuse("charge_e", "A charged cell carries a compensating background whose energy "
                           "depends on the volume, so its equation of state is not the "
                           "crystal's.")
    if spec.fit != FIT:
        refuse("fit", f"fit must be {FIT}.")
    low, high, n = spec.volume_min_scale, spec.volume_max_scale, spec.n_points
    if not (math.isfinite(low) and math.isfinite(high)) or not (
            MIN_SCALE <= low < high <= MAX_SCALE):
        refuse("volume_min_scale", f"The volumes must satisfy {MIN_SCALE:g} <= smallest < "
                                   f"largest <= {MAX_SCALE:g} of the reference volume.")
    elif not (low < 1.0 < high):
        warnings.append("The range does not include the reference volume; the minimum must "
                        "still fall inside it or nothing is kept.")
    if not (MIN_POINTS <= n <= MAX_POINTS):
        refuse("n_points", f"Use {MIN_POINTS} to {MAX_POINTS} volumes; a four-parameter fit "
                           "needs at least five.")
    expected = scales(low, high, n) if n >= 1 else ()
    if tuple(spec.volume_scales) != expected:
        refuse("volume_scales", "The stored volumes are not the evenly spaced range the "
                                "smallest, largest and count define.")
    if high - low < 0.04 and not by_field.get("volume_min_scale"):
        warnings.append("A range narrower than 4 percent in volume constrains B' poorly.")
    if gs.representation == "pw" and gs.cutoff_eV is not None:
        k = "x".join(str(v) for v in gs.kpoints)
        warnings.append(f"Every volume uses the same {k} k-point grid and "
                        f"{gs.cutoff_eV:g} eV cutoff, pinned from the reference; check their "
                        "convergence with the convergence laboratory.")
    warnings.append("Fractional coordinates are held fixed while the volume changes. For a "
                    "crystal with free internal parameters the energies are not relaxed at "
                    "each volume; the largest force at every point is reported.")
    if not by_field and not general:
        try:
            point_specs(spec, environment)
        except specs.SpecError as exc:
            general.append(str(exc))
    blocking = list(general)
    for texts in by_field.values():
        for text in texts:
            if text not in blocking:
                blocking.append(text)
    options = dict(report.options)
    options.update({"reference_volume_A3": spec.reference_volume_A3,
                    "volumes_A3": spec.volumes().tolist(),
                    "kpoints": list(gs.kpoints), "cutoff_eV": gs.cutoff_eV})
    return specs.SpecReport(blocking, warnings, by_field, report.boundary, report.electrons,
                            options)


def require(spec: EOSSpec, environment=None) -> specs.SpecReport:
    report = check(spec, environment)
    if not report.ok:
        raise specs.SpecRefused(report)
    return report


def describe(spec: EOSSpec, report: Optional[specs.SpecReport] = None) -> dict:
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
            "short_digest": spec.short_digest, "eos": items, "settings": spec.settings(),
            "source": dict(spec.source), "reference_volume_A3": spec.reference_volume_A3,
            "volumes_A3": spec.volumes().tolist(),
            "ground_state": specs.describe(spec.ground_state, report),
            "gpaw_parameters": specs.gpaw_parameters(spec.ground_state)}


@dataclass
class EOSOutcome:
    run_id: str
    spec: EOSSpec
    status: str
    reason: str = ""
    results: Dict[str, Result] = field(default_factory=dict)
    arrays: Dict[str, StoredArray] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    wall_time_s: float = 0.0
    points: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_COMPLETE


def _approximations(spec: EOSSpec) -> List[str]:
    gs = spec.ground_state
    return [
        f"Energies of {spec.n_points} isotropically scaled cells from V/V_ref = "
        f"{spec.volume_min_scale:g} to {spec.volume_max_scale:g}, each a converged GPAW "
        f"ground state with the same {'x'.join(map(str, gs.kpoints))} k-point grid and "
        f"{gs.cutoff_eV:g} eV plane-wave cutoff.",
        "Third-order Birch-Murnaghan fit to GPAW's energy per cell extrapolated to zero "
        "smearing width; the free energy E - TS is fitted separately and GPAW's stress, its "
        "derivative, is compared with that fit.",
        "Fractional coordinates fixed at every volume: internal parameters are not relaxed.",
        "Fixed cutoff at every volume: the basis changes slightly with the cell, which can "
        "bias the volume (Pulay stress); the stress comparison measures it.",
        "Static lattice at 0 K: no zero-point or thermal expansion.",
    ]


def execute(spec: EOSSpec, environment=None, *, run_id: Optional[str] = None,
            progress: Optional[Callable[[float, str], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None) -> EOSOutcome:
    """Run every volume, fit, verify.  Raises :class:`~.spec.SpecRefused` if it may not run."""
    environment = specs._environment(environment)
    report = require(spec, environment)
    run_id = run_id or ground.new_run_id()
    started = time.perf_counter()
    started_unix = time.time()
    warnings = list(report.warnings)
    outcome = EOSOutcome(run_id=run_id, spec=spec, status=STATUS_FAILED, warnings=warnings)
    points = point_specs(spec, environment)
    rows: List[Dict[str, Any]] = []
    template = None
    n_atoms = spec.ground_state.n_atoms
    for index, (scale, point) in enumerate(zip(spec.volume_scales, points)):
        if cancelled is not None and cancelled():
            outcome.status = STATUS_CANCELLED
            outcome.reason = (f"Cancelled after {index} of {spec.n_points} volumes. Nothing "
                              "was kept.")
            return outcome
        if progress is not None:
            progress(index / len(points), f"volume {index + 1} of {len(points)}, "
                                          f"V/V_ref = {scale:.4f}")
        run = ground.execute(point, environment, run_id=f"{run_id}-v{index}",
                             cancelled=cancelled, timeout_s=timeout_s)
        if cancelled is not None and cancelled() or run.status == ground.STATUS_CANCELLED:
            outcome.status = STATUS_CANCELLED
            outcome.reason = (f"Cancelled at volume {index + 1} of {spec.n_points}. Nothing "
                              "was kept.")
            return outcome
        if not run.converged:
            outcome.reason = (f"The ground state at V/V_ref = {scale:g} did not produce a "
                              f"result: {run.reason.rstrip('.')}. Nothing was kept.")
            return outcome
        free_energy = float(run.results["energy"].value)
        energy = float(run.results["energy"].extra["extrapolated_energy_eV"])
        if not (math.isfinite(energy) and math.isfinite(free_energy)):
            outcome.reason = f"The energy at V/V_ref = {scale:g} is not finite. Nothing was kept."
            return outcome
        forces = run.results.get("forces")
        stress = run.results.get("stress")
        row = {"index": index, "scale": float(scale),
               "volume_A3": float(abs(np.linalg.det(np.asarray(point.cell_A)))),
               "energy_eV": energy, "free_energy_eV": free_energy,
               "run_id": run.run_id, "spec_digest": point.digest,
               "scf_iterations": run.record.convergence.iterations
               if run.record.convergence else None,
               "max_force_eV_A": float(forces.extra["max_force_eV_A"]) if forces else None,
               "pressure_GPa": float(stress.extra["pressure_GPa"]) if stress else None}
        if forces is not None and not math.isfinite(row["max_force_eV_A"]):
            outcome.reason = f"The forces at V/V_ref = {scale:g} are not finite. Nothing was kept."
            return outcome
        rows.append(row)
        if template is None:
            template = run.results["energy"].provenance
    outcome.points = rows
    volumes = np.array([r["volume_A3"] for r in rows])
    energies = np.array([r["energy_eV"] for r in rows])
    free_energies = np.array([r["free_energy_eV"] for r in rows])
    if not np.allclose(volumes, spec.volumes(), rtol=1e-9, atol=0):
        outcome.reason = "The computed cells do not have the specified volumes. Nothing was kept."
        return outcome
    try:
        fit = fit_birch_murnaghan(volumes, energies)
    except (RuntimeError, ValueError) as exc:
        outcome.reason = f"The Birch-Murnaghan fit did not converge: {exc}. Nothing was kept."
        return outcome
    v0, b0, b1 = fit["V0_A3"], fit["B0_eV_A3"], fit["B1"]
    per_atom = fit["max_abs_eV"] / n_atoms
    if per_atom > MAX_RESIDUAL_EV_PER_ATOM:
        outcome.reason = (f"The fit misses a point by {per_atom * 1000:.2f} meV/atom, more than "
                          f"{MAX_RESIDUAL_EV_PER_ATOM * 1000:g}: the energies are not a smooth "
                          "curve. Check convergence. Nothing was kept.")
        return outcome
    if not (volumes.min() < v0 < volumes.max()):
        side = "smaller" if v0 <= volumes.min() else "larger"
        outcome.reason = (f"The fitted minimum, {v0:.4f} A^3, lies outside the sampled volumes "
                          f"({volumes.min():.4f} to {volumes.max():.4f} A^3). Sample {side} "
                          "volumes. Nothing was kept.")
        return outcome
    if not (b0 > 0 and 1.0 < b1 < 12.0):
        outcome.reason = (f"The fit is unphysical (B0 = {b0 * EV_A3_TO_GPA:.3g} GPa, "
                          f"B' = {b1:.3g}); the energies do not describe a stable crystal in "
                          "this range. Nothing was kept.")
        return outcome
    fit_pressure = birch_murnaghan_pressure(volumes, v0, b0, b1) * EV_A3_TO_GPA
    try:
        free_fit = fit_birch_murnaghan(volumes, free_energies)
    except (RuntimeError, ValueError) as exc:
        outcome.reason = (f"The Birch-Murnaghan fit of the free energy did not converge: {exc}. "
                          "Nothing was kept.")
        return outcome
    free_pressure = birch_murnaghan_pressure(volumes, free_fit["V0_A3"], free_fit["B0_eV_A3"],
                                             free_fit["B1"]) * EV_A3_TO_GPA
    checks: Dict[str, Any] = {"fit_rms_meV_per_atom": fit["rms_eV"] / n_atoms * 1000,
                              "fit_max_meV_per_atom": per_atom * 1000,
                              "minimum_inside_range": True,
                              "free_energy_fit_rms_meV_per_atom":
                                  free_fit["rms_eV"] / n_atoms * 1000,
                              "free_minus_zero_width_V0_relative":
                                  free_fit["V0_A3"] / v0 - 1.0,
                              "largest_entropy_term_meV_per_atom":
                                  float(np.abs(free_energies - energies).max() / n_atoms * 1000)}
    stress_pressures = [r["pressure_GPa"] for r in rows]
    if all(p is not None for p in stress_pressures):
        difference = np.abs(np.array(stress_pressures) - free_pressure)
        checks["stress_pressure_max_difference_GPa"] = float(difference.max())
        checks["stress_compared_with"] = "-dF/dV of the free-energy fit"
        if difference.max() > PRESSURE_WARNING_GPA:
            warnings.append(f"The pressure from GPAW's stress differs from -dF/dV of the free-"
                            f"energy fit by up to {difference.max():.2f} GPa: the plane-wave "
                            "basis is not converged for this volume range (Pulay stress). "
                            "Raise the cutoff.")
    forces_max = [r["max_force_eV_A"] for r in rows if r["max_force_eV_A"] is not None]
    if forces_max:
        checks["largest_force_eV_A"] = float(max(forces_max))
        if max(forces_max) > FORCE_WARNING_EV_A:
            warnings.append(f"Forces up to {max(forces_max):.3f} eV/A at fixed fractional "
                            "coordinates: this crystal has free internal parameters, and the "
                            "energies are not relaxed at each volume.")
    source_energy = spec.source.get("energy_eV")
    reference_row = min(rows, key=lambda r: abs(r["scale"] - 1.0))
    if source_energy is not None and abs(reference_row["scale"] - 1.0) < 1e-12:
        checks["source_energy_difference_eV"] = reference_row["energy_eV"] - float(source_energy)
    linear = (v0 / spec.reference_volume_A3) ** (1.0 / 3.0)
    lengths = np.linalg.norm(np.asarray(spec.ground_state.cell_A, dtype=float), axis=1)
    gs = spec.ground_state
    summary = {
        "status": STATUS_COMPLETE, "fit": FIT, "n_points": spec.n_points, "n_atoms": n_atoms,
        "energy_definition": ENERGY_DEFINITION,
        "occupations": {"name": gs.occupations, "width_eV": gs.smearing_eV},
        "free_energy_fit": {"E0_eV": free_fit["E0_eV"], "V0_A3": free_fit["V0_A3"],
                            "V0_A3_per_atom": free_fit["V0_A3"] / n_atoms,
                            "B0_GPa": free_fit["B0_eV_A3"] * EV_A3_TO_GPA, "B1": free_fit["B1"],
                            "meaning": "Birch-Murnaghan fit of the free energy E - TS at the "
                                       "finite smearing width; GPAW's stress is its "
                                       "derivative"},
        "E0_eV": fit["E0_eV"], "E0_eV_per_atom": fit["E0_eV"] / n_atoms,
        "V0_A3": v0, "V0_A3_per_atom": v0 / n_atoms,
        "B0_GPa": b0 * EV_A3_TO_GPA, "B1": b1, "linear_scale": linear,
        "equilibrium_cell_lengths_A": (lengths * linear).tolist(),
        "reference_volume_A3": spec.reference_volume_A3,
        "points": rows, "fit_pressure_GPa": fit_pressure.tolist(),
        "free_energy_fit_pressure_GPa": free_pressure.tolist(), "checks": checks,
        "kpoints": list(spec.ground_state.kpoints), "cutoff_eV": spec.ground_state.cutoff_eV,
        "source": dict(spec.source), "started_unix": started_unix,
        "finished_unix": time.time(), "wall_time_s": time.perf_counter() - started,
        "units": {"energy": "eV per cell, zero-width extrapolated",
                  "free_energy": "eV per cell, E - TS at the smearing width",
                  "volume": "A^3 per cell", "pressure": "GPa", "B0": "GPa"},
        "experimental_comparison": None,
    }
    meta = {"eos_spec_digest": spec.digest, "fit": FIT,
            "energy_definition": ENERGY_DEFINITION}
    outcome.arrays = {
        "scales": StoredArray(np.asarray(spec.volume_scales, dtype=float), "",
                              "Volume over reference volume", "table", dict(meta)),
        "volumes": StoredArray(volumes, "A^3", "Cell volume at each point", "table", dict(meta)),
        "energies": StoredArray(energies, "eV",
                                "Zero-width extrapolated energy per cell at each point, the "
                                "fitted quantity", "table", dict(meta)),
        "free_energies": StoredArray(free_energies, "eV",
                                     "Free energy E - TS per cell at the smearing width",
                                     "table", dict(meta)),
        "fit_pressures": StoredArray(np.asarray(fit_pressure), "GPa",
                                     "-dE/dV of the zero-width fit: the 0 K pressure", "table",
                                     dict(meta)),
        "free_energy_fit_pressures": StoredArray(
            np.asarray(free_pressure), "GPa",
            "-dF/dV of the free-energy fit, the quantity GPAW's stress estimates", "table",
            dict(meta)),
    }
    if all(p is not None for p in stress_pressures):
        outcome.arrays["stress_pressures"] = StoredArray(
            np.asarray(stress_pressures, dtype=float), "GPa",
            "Pressure from GPAW's stress at each point", "table", dict(meta))
    if len(forces_max) == len(rows):
        outcome.arrays["max_forces"] = StoredArray(np.asarray(forces_max, dtype=float), "eV/A",
                                                   "Largest force at each point", "table",
                                                   dict(meta))
    summary["array_sha256"] = {name: stored.sha256() for name, stored in outcome.arrays.items()}
    provenance = ground._copy_provenance(template)
    provenance.model = MODEL
    provenance.fidelity = Fidelity.TIER3_EXTERNAL
    provenance.origin = Origin.CALCULATED
    provenance.inputs_digest = spec.digest
    provenance.parameters.update({
        "eos_spec": spec.as_dict(), "eos_spec_digest": spec.digest,
        "gpaw_parameters": specs.gpaw_parameters(spec.ground_state),
        "point_spec_digests": [r["spec_digest"] for r in rows],
        "point_run_ids": [r["run_id"] for r in rows], "source": dict(spec.source),
        "array_sha256": summary["array_sha256"]})
    provenance.references = list(provenance.references) + [
        r for r in REFERENCES if r not in provenance.references]
    provenance.approximations = list(provenance.approximations) + _approximations(spec)
    convergence = Convergence(
        converged=True, iterations=sum(int(r["scf_iterations"] or 0) for r in rows),
        residual=fit["rms_eV"] / n_atoms,
        residual_metric="root-mean-square residual of the Birch-Murnaghan fit, eV/atom",
        tolerance=MAX_RESIDUAL_EV_PER_ATOM,
        message=(f"{spec.n_points} ground states converged with identical settings; the fit "
                 f"reproduces them to {fit['rms_eV'] / n_atoms * 1000:.3f} meV/atom rms and "
                 "its minimum lies inside the sampled range."))
    record = Result("dft_equation_of_state", summary, "eV", provenance, convergence=convergence)
    record.extra.update({"run_id": run_id, "structure_key": spec.structure_key,
                         "geometry_digest": spec.geometry_digest, "eos_spec_digest": spec.digest,
                         "status": STATUS_COMPLETE, "classification": ground.CLASSIFICATION,
                         "warnings": list(warnings), "kind": "eos",
                         "equilibrium_geometry_digest": specs.geometry_digest(
                             structure_of(spec.ground_state, v0 / spec.reference_volume_A3))})
    outcome.results = {"eos": record}
    outcome.status = STATUS_COMPLETE
    outcome.wall_time_s = time.perf_counter() - started
    outcome.warnings = warnings
    return outcome
