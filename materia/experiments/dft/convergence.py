"""Convergence laboratory: how much a result still depends on its numerics.

A study takes one ground-state specification, varies one numerical parameter
over a short list of values, runs every point as an ordinary ground-state
calculation, and reports how the chosen observable changes as the parameter
is made more accurate.  Every run is kept.  The study says:

* the observable at each value, ordered from least to most accurate;
* the change between successive values and the deviation of each from the
  most accurate one;
* the **residual**: the change between the two most accurate values, which
  is the only honest measure of what is left without extrapolating;
* whether that residual is within the tolerance the user chose.

It says nothing about the error of the exchange-correlation functional or
any other physical approximation.  A study that meets its tolerance shows
that a number is converged with respect to one discretisation parameter, not
that it is right.

Restart data is reused only where it is safe.  Varying the smearing width
leaves the atoms, grid, k-points, electron count and spin setup unchanged, so
each point starts from the previous converged state.  Every other parameter
changes the discretisation or the atoms, so those points start fresh, and the
study records that it did.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core_model.cell import Cell
from ...core_model.structure import Structure
from ...provenance import Convergence, Fidelity, Origin, Provenance, Result
from ...provenance.classification import Classification
from . import boundary as bc
from . import spec as specs
from .restart import RestartStore
from .run import GroundStateOutcome, execute, new_run_id

MAX_VALUES = 8

PARAMETERS: Dict[str, Dict[str, Any]] = {
    "grid_spacing_A": {
        "unit": "A", "more_accurate": "smaller", "label": "grid spacing",
        "explanation": "Real-space grid spacing, for fd and lcao."},
    "cutoff_eV": {
        "unit": "eV", "more_accurate": "larger", "label": "plane-wave cutoff",
        "explanation": "Kinetic-energy cutoff, for pw."},
    "kpoint_density_A": {
        "unit": "A", "more_accurate": "larger", "label": "k-point density",
        "explanation": "Divisions along each periodic axis are the smallest giving at "
                       "least this value of divisions times lattice length."},
    "vacuum_A": {
        "unit": "A", "more_accurate": "larger", "label": "vacuum per side",
        "explanation": "Vacuum between the outermost atom and each open cell face. The "
                       "atoms are kept, the open cell vectors are resized."},
    "supercell": {
        "unit": "repetitions", "more_accurate": "larger", "label": "supercell size",
        "explanation": "Repetitions along every periodic axis, with the k-point "
                       "divisions divided by the same factor so the sampling density "
                       "stays the same."},
    "n_bands": {
        "unit": "bands", "more_accurate": "larger", "label": "number of bands",
        "explanation": "Kohn-Sham states per spin and k-point."},
    "smearing_eV": {
        "unit": "eV", "more_accurate": "smaller", "label": "smearing width",
        "explanation": "Occupation smearing; the physical limit is zero width."},
}

OBSERVABLES: Dict[str, Dict[str, str]] = {
    "energy_per_atom_eV": {"unit": "eV/atom", "label": "energy per atom"},
    "energy_eV": {"unit": "eV", "label": "total energy"},
    "max_force_eV_A": {"unit": "eV/A", "label": "largest force"},
    "fermi_level_eV": {"unit": "eV", "label": "Fermi level"},
    "magnetic_moment_muB": {"unit": "mu_B", "label": "total magnetic moment"},
    "pressure_GPa": {"unit": "GPa", "label": "pressure"},
}

REUSES_RESTART = ("smearing_eV",)

PHYSICAL_NOTE = ("The residual measures convergence with respect to one numerical "
                 "parameter. It is not an uncertainty of the physical result: the "
                 "exchange-correlation functional, the frozen-core datasets and every "
                 "other approximation in the record keep their own errors, which no "
                 "numerical setting reduces.")


class StudyError(ValueError):
    """A convergence study that cannot be run as asked."""


@dataclass(frozen=True)
class StudySpec:
    """One bounded study of one parameter."""

    base: specs.GroundStateSpec
    parameter: str
    values: Tuple[Any, ...]
    observable: str = "energy_per_atom_eV"
    tolerance: float = 1.0e-3

    def as_dict(self) -> dict:
        return {"base": self.base.as_dict(), "base_digest": self.base.digest,
                "parameter": self.parameter, "values": list(self.values),
                "observable": self.observable, "tolerance": self.tolerance,
                "tolerance_unit": OBSERVABLES[self.observable]["unit"]}


def structure_of(spec: specs.GroundStateSpec) -> Structure:
    """The frozen geometry of a specification, as a structure."""
    structure = Structure(np.asarray(spec.numbers, dtype=int),
                          np.asarray(spec.positions_A, dtype=float),
                          Cell(np.asarray(spec.cell_A, dtype=float), spec.pbc),
                          ids=list(spec.atom_ids), mass_numbers=list(spec.mass_numbers))
    structure.formal_charges[:] = np.asarray(spec.formal_charges_e, dtype=float)
    structure.magnetic_moments[:] = np.asarray(spec.initial_magnetic_moments_muB,
                                               dtype=float)
    return structure


def _order(parameter: str, values: Sequence[Any]) -> List[Any]:
    direction = PARAMETERS[parameter]["more_accurate"]
    return sorted(values, reverse=(direction == "smaller"))


def plan(study: StudySpec, environment=None) -> List[Tuple[Any, specs.GroundStateSpec, dict]]:
    """The specifications a study will run, least accurate first.

    Raises :class:`StudyError` with the reason when the parameter does not
    apply to the base specification or the values are not usable.
    """
    base = study.base
    parameter = study.parameter
    if parameter not in PARAMETERS:
        raise StudyError(f"Unknown convergence parameter {parameter!r}. Available: "
                         f"{', '.join(PARAMETERS)}.")
    if study.observable not in OBSERVABLES:
        raise StudyError(f"Unknown observable {study.observable!r}. Available: "
                         f"{', '.join(OBSERVABLES)}.")
    values = list(study.values)
    if len(values) < 2:
        raise StudyError("A convergence study needs at least two values.")
    if len(values) > MAX_VALUES:
        raise StudyError(f"At most {MAX_VALUES} values per study; each is a full "
                         "self-consistent calculation.")
    if len(set(map(_hashable, values))) != len(values):
        raise StudyError("The values repeat; each must be different.")
    if not (isinstance(study.tolerance, (int, float)) and study.tolerance > 0
            and math.isfinite(study.tolerance)):
        raise StudyError("The tolerance must be a positive number in the observable's "
                         "unit.")
    observable = study.observable
    if observable == "pressure_GPa" and base.representation != "pw":
        raise StudyError("Pressure needs the stress tensor, which GPAW computes only in "
                         "plane-wave mode.")
    if observable == "magnetic_moment_muB" and not base.spin_polarized:
        raise StudyError("The magnetic moment is zero by construction in a spin-paired "
                         "calculation.")
    if observable == "max_force_eV_A" and "forces" not in base.observables:
        raise StudyError("Request forces in the base specification to study them.")
    if parameter == "supercell" and observable in ("energy_eV", "max_force_eV_A"):
        raise StudyError("A supercell study changes the number of atoms, so it needs an "
                         "intensive observable such as the energy per atom.")

    ordered = _order(parameter, values)
    out: List[Tuple[Any, specs.GroundStateSpec, dict]] = []
    if parameter == "grid_spacing_A":
        if base.representation == "pw":
            raise StudyError("The grid spacing does not apply to plane waves; study "
                             "cutoff_eV instead.")
        for value in ordered:
            out.append((float(value), specs.changed(base, environment,
                                                    grid_spacing_A=float(value)), {}))
    elif parameter == "cutoff_eV":
        if base.representation != "pw":
            raise StudyError("The cutoff applies only to plane waves; study "
                             "grid_spacing_A instead.")
        for value in ordered:
            out.append((float(value), specs.changed(base, environment,
                                                    cutoff_eV=float(value)), {}))
    elif parameter == "kpoint_density_A":
        if base.boundary == "cluster":
            raise StudyError("A cluster has no Brillouin zone to sample.")
        seen = {}
        cell = np.asarray(base.cell_A)
        for value in ordered:
            grid = specs.default_kpoints(cell, base.pbc, float(value))
            if grid in seen:
                raise StudyError(f"k-point densities {seen[grid]:g} and {float(value):g} A "
                                 f"give the same {grid} grid; choose values that differ.")
            seen[grid] = float(value)
            out.append((float(value), specs.changed(base, environment,
                                                    kpoints=list(grid)),
                        {"kpoints": list(grid)}))
    elif parameter == "n_bands":
        for value in ordered:
            out.append((int(value), specs.changed(base, environment, n_bands=int(value)), {}))
    elif parameter == "smearing_eV":
        if base.occupations == "fixed":
            raise StudyError("Fixed occupations have no smearing width to vary.")
        for value in ordered:
            out.append((float(value), specs.changed(base, environment,
                                                    smearing_eV=float(value)), {}))
    elif parameter == "vacuum_A":
        if base.boundary == "bulk":
            raise StudyError("A bulk crystal has no vacuum.")
        cell = np.asarray(base.cell_A)
        for axis in range(3):
            if not base.pbc[axis] and not bc._orthogonal(cell, axis):
                raise StudyError("The vacuum can only be resized along open axes that "
                                 "are perpendicular to the other cell vectors.")
        parent = structure_of(base)
        for value in ordered:
            positions, new_cell = bc.recentre_open_axes(parent.positions, cell, base.pbc,
                                                        float(value))
            derived = parent.copy()
            derived.positions = positions
            derived.cell = Cell(new_cell, base.pbc)
            settings = base.settings()
            settings.pop("kpoints", None)
            candidate = specs.build(derived, environment, structure_key=None,
                                    kpoints=list(base.kpoints), **_settable(settings))
            out.append((float(value), candidate,
                        {"description": f"{float(value):g} A of vacuum on each open side"}))
    elif parameter == "supercell":
        if base.boundary == "cluster":
            raise StudyError("A cluster has no periodic direction to repeat.")
        if abs(base.charge_e) > 1e-9:
            raise StudyError("Repeating a charged cell multiplies its charge; a charged "
                             "defect needs a defect-centred supercell construction, which "
                             "this study does not build.")
        parent = structure_of(base)
        for value in ordered:
            n = int(value)
            if n < 1 or n != value:
                raise StudyError("Supercell sizes are whole repetitions of at least 1.")
            reps = [n if base.pbc[i] else 1 for i in range(3)]
            derived = parent.repeat(*reps)
            kpts = [max(1, int(math.ceil(base.kpoints[i] / reps[i]))) for i in range(3)]
            settings = base.settings()
            for name in ("kpoints", "initial_magnetic_moments_muB", "n_bands", "charge_e"):
                settings.pop(name, None)
            derived.magnetic_moments[:] = np.tile(
                np.asarray(base.initial_magnetic_moments_muB, dtype=float), int(np.prod(reps)))
            candidate = specs.build(derived, environment, structure_key=None,
                                    kpoints=kpts, **_settable(settings))
            equivalent = all(base.kpoints[i] % reps[i] == 0 for i in range(3))
            description = f"{'x'.join(map(str, reps))} supercell"
            if not equivalent:
                description += (f"; its {'x'.join(map(str, kpts))} grid is not equivalent to "
                                f"the base {'x'.join(map(str, base.kpoints))} grid because the "
                                "repetition does not divide it, so part of the change is "
                                "k-point sampling, not cell size")
            out.append((n, candidate,
                        {"description": description, "kpoints": kpts,
                         "sampling_equivalent": equivalent}))
    for _, candidate, _ in out:
        report = specs.check(candidate, environment)
        if not report.ok:
            raise StudyError("A study point would be refused: " + "; ".join(report.blocking))
    return out


def _settable(settings: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in settings.items() if k in specs.SETTABLE}


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, (list, tuple)) else value


def observe(outcome: GroundStateOutcome, observable: str) -> Optional[float]:
    """The studied quantity from one converged run."""
    if not outcome.converged:
        return None
    results = outcome.results
    if observable == "energy_per_atom_eV":
        return float(results["energy"].extra["energy_per_atom_eV"])
    if observable == "energy_eV":
        return float(results["energy"].value)
    if observable == "max_force_eV_A" and "forces" in results:
        return float(results["forces"].extra["max_force_eV_A"])
    if observable == "fermi_level_eV" and "fermi_level" in results:
        value = results["fermi_level"].value
        return None if value is None else float(value)
    if observable == "magnetic_moment_muB" and "magnetic_moment" in results:
        return float(results["magnetic_moment"].value)
    if observable == "pressure_GPa" and "stress" in results:
        return float(results["stress"].extra["pressure_GPa"])
    return None


def analyse(parameter: str, values: Sequence[Any], observed: Sequence[Optional[float]],
            tolerance: float) -> Dict[str, Any]:
    """Successive changes, deviations from the most accurate point, and the verdict."""
    complete = all(v is not None for v in observed)
    changes: List[Optional[float]] = [None]
    for previous, current in zip(observed[:-1], observed[1:]):
        changes.append(None if previous is None or current is None
                       else abs(float(current) - float(previous)))
    last = observed[-1]
    deviations = [None if (v is None or last is None) else abs(float(v) - float(last))
                  for v in observed]
    residual = changes[-1] if len(changes) > 1 else None
    met = bool(complete and residual is not None and residual <= tolerance)
    converged_at = None
    if met:
        for index in range(len(observed) - 1):
            if all(c is not None and c <= tolerance for c in changes[index + 1:]):
                converged_at = values[index]
                break
    if not complete:
        verdict = ("Incomplete: at least one point did not converge, so the study cannot "
                   "say whether the tolerance is met.")
    elif met:
        verdict = (f"Met: the two most accurate points differ by {residual:.3g}, within "
                   f"the tolerance of {tolerance:g}.")
    else:
        verdict = (f"Not met: the two most accurate points still differ by "
                   f"{residual:.3g}, more than the tolerance of {tolerance:g}. Extend the "
                   "range.")
    return {"complete": complete, "changes": changes, "deviations_from_last": deviations,
            "residual": residual, "tolerance": tolerance, "tolerance_met": met,
            "converged_from": converged_at, "verdict": verdict}


@dataclass
class StudyOutcome:
    study_id: str
    study: StudySpec
    status: str
    points: List[dict] = field(default_factory=list)
    outcomes: List[GroundStateOutcome] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0

    def record(self) -> Result:
        """The study as a stored result, pointing at every run it made."""
        base = self.study.base
        info = OBSERVABLES[self.study.observable]
        origin = Origin.CALCULATED if self.analysis.get("complete") else Origin.UNSUPPORTED
        provenance = Provenance(
            model="materia/convergence-study",
            fidelity=Fidelity.TIER3_EXTERNAL,
            origin=origin,
            approximations=[PHYSICAL_NOTE],
            tolerances={"observable": self.study.observable,
                        "tolerance": self.study.tolerance,
                        "unit": info["unit"]},
            boundary_conditions=base.boundary,
            parameters={"study": self.study.as_dict(),
                        "parameter_info": PARAMETERS[self.study.parameter],
                        "restart_policy": ("each point starts from the previous converged "
                                           "state" if self.study.parameter in REUSES_RESTART
                                           else "each point starts fresh: the varied "
                                                "parameter changes the discretisation or "
                                                "the atoms, so stored states are not "
                                                "compatible")},
            inputs_digest=base.digest,
            notes=PHYSICAL_NOTE,
        )
        value = {"parameter": self.study.parameter, "observable": self.study.observable,
                 "unit": info["unit"], "points": self.points, **self.analysis}
        result = Result("convergence_study", value, info["unit"], provenance,
                        convergence=Convergence(
                            converged=bool(self.analysis.get("tolerance_met")),
                            iterations=len(self.points),
                            residual=self.analysis.get("residual"),
                            residual_metric=f"change of {info['label']} between the two "
                                            "most accurate points",
                            tolerance=self.study.tolerance,
                            message=self.analysis.get("verdict", "")))
        result.extra.update({"study_id": self.study_id, "status": self.status,
                             "run_ids": [p["run_id"] for p in self.points],
                             "classification": Classification.OBSERVATION.value,
                             "base_structure_key": base.structure_key,
                             "base_geometry_digest": base.geometry_digest,
                             "wall_time_s": self.wall_time_s})
        if origin is Origin.UNSUPPORTED:
            result.unsupported_reason = self.analysis.get("verdict", "")
        return result


def run_study(study: StudySpec, environment=None, *,
              progress: Optional[Callable[[float, str], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None,
              on_run: Optional[Callable[[GroundStateOutcome, dict], None]] = None,
              restart_store: Optional[RestartStore] = None,
              study_id: Optional[str] = None) -> StudyOutcome:
    """Run every point of a study in turn; each finished run is handed to ``on_run``.

    Cancelling stops at the current point: the runs already finished are kept,
    the interrupted one yields no value, and the study is recorded as
    cancelled and incomplete.
    """
    environment = specs._environment(environment)
    points_plan = plan(study, environment)
    study_id = study_id or f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:6]}"
    started = time.perf_counter()
    outcome = StudyOutcome(study_id=study_id, study=study, status="running")
    reuse = study.parameter in REUSES_RESTART
    store = restart_store or (RestartStore() if reuse else None)
    observed: List[Optional[float]] = []
    values: List[Any] = []
    base_parent = {"structure_key": study.base.structure_key,
                   "geometry_digest": study.base.geometry_digest}
    for index, (value, point_spec, detail) in enumerate(points_plan):
        if cancelled is not None and cancelled():
            outcome.status = "cancelled"
            break
        label = f"{PARAMETERS[study.parameter]['label']} = {value}"
        if progress is not None:
            progress(index / len(points_plan), f"point {index + 1} of {len(points_plan)}: "
                                               f"{label}")

        def relay(event: dict, index=index) -> None:
            if progress is not None and event.get("event") == "scf":
                fraction = (index + min(0.95, event.get("iteration", 0) /
                                        max(1, point_spec.max_scf_iterations) * 4)) \
                    / len(points_plan)
                progress(min(0.99, fraction),
                         f"point {index + 1} of {len(points_plan)}: {label}, SCF "
                         f"iteration {event.get('iteration')}")

        run = execute(point_spec, environment, run_id=new_run_id(), progress=relay,
                      cancelled=cancelled, restart_store=store,
                      reuse_restart=reuse and index > 0, keep_restart=reuse,
                      study_id=study_id)
        derived = None
        if point_spec.structure_key is None and study.base.structure_key is not None:
            derived = {**base_parent, "description": detail.get("description", "")}
            for item in run.results.values():
                item.extra["derived_from"] = derived
        point = {"index": index, "value": value, "run_id": run.run_id,
                 "status": run.status, "observable": observe(run, study.observable),
                 "spec_digest": point_spec.digest, "iterations":
                 run.record.convergence.iterations if run.record.convergence else None,
                 "restart": run.record.provenance.parameters.get("restart"),
                 "wall_time_s": run.wall_time_s, **{k: v for k, v in detail.items()}}
        outcome.points.append(point)
        outcome.outcomes.append(run)
        values.append(value)
        observed.append(point["observable"])
        if on_run is not None:
            on_run(run, point)
        if run.status == "cancelled":
            outcome.status = "cancelled"
            break
    if outcome.status == "running":
        outcome.status = "complete" if all(o is not None for o in observed) else "incomplete"
    if len(observed) < len(points_plan):
        missing = len(points_plan) - len(observed)
        observed = observed + [None] * missing
        values = values + [v for v, _, _ in points_plan[len(values):]]
    outcome.analysis = analyse(study.parameter, values, observed, study.tolerance)
    if outcome.status == "cancelled":
        outcome.analysis["verdict"] = ("Cancelled: " + str(sum(o is not None for o in observed))
                                       + " of " + str(len(points_plan)) + " points finished. "
                                       + outcome.analysis["verdict"])
    outcome.analysis["note"] = PHYSICAL_NOTE
    outcome.wall_time_s = time.perf_counter() - started
    if progress is not None:
        progress(1.0, outcome.analysis["verdict"])
    return outcome


def evidence_key(spec: specs.GroundStateSpec, parameter: str) -> str:
    """Fingerprint of a specification with one convergence parameter left out.

    Two specifications with the same key differ at most in that parameter, so
    a study of it on one is convergence evidence for the other.
    """
    from ...provenance import digest

    data = spec.as_dict()
    for name in {"grid_spacing_A": ("grid_spacing_A",), "cutoff_eV": ("cutoff_eV",),
                 "kpoint_density_A": ("kpoints",), "n_bands": ("n_bands",),
                 "smearing_eV": ("smearing_eV",), "vacuum_A": (),
                 "supercell": ()}.get(parameter, ()):
        data.pop(name, None)
    data.pop("structure_key", None)
    return digest(data)


def parameter_value(spec: specs.GroundStateSpec, parameter: str) -> Optional[float]:
    """Where a specification sits on a convergence parameter's axis."""
    if parameter == "grid_spacing_A":
        return spec.grid_spacing_A
    if parameter == "cutoff_eV":
        return spec.cutoff_eV
    if parameter == "n_bands":
        return None if spec.n_bands is None else float(spec.n_bands)
    if parameter == "smearing_eV":
        return spec.smearing_eV if spec.occupations != "fixed" else 0.0
    if parameter == "kpoint_density_A":
        lengths = np.linalg.norm(np.asarray(spec.cell_A, dtype=float), axis=1)
        periodic = [spec.kpoints[i] * lengths[i] for i in range(3) if spec.pbc[i]]
        return float(min(periodic)) if periodic else None
    if parameter == "vacuum_A":
        report = bc.inspect(np.asarray(spec.positions_A), np.asarray(spec.cell_A), spec.pbc)
        return report.min_vacuum_A()
    if parameter == "supercell":
        return 1.0
    return None


def _at_least_as_accurate(parameter: str, value: float, reference: float) -> bool:
    if PARAMETERS[parameter]["more_accurate"] == "smaller":
        return value <= reference + 1e-12
    return value >= reference - 1e-12


def evidence(spec: specs.GroundStateSpec, studies: Sequence[Result]) -> List[dict]:
    """Which stored studies show this specification converged, parameter by parameter.

    A study counts for a parameter when its base differs from ``spec`` at
    most in that parameter, its tolerance was met, and ``spec`` sits at a
    value at least as accurate as the one from which the study found the
    observable settled.
    """
    out: List[dict] = []
    for record in studies:
        value = record.value or {}
        if not isinstance(value, dict) or not value.get("tolerance_met"):
            continue
        data = record.provenance.parameters.get("study") or {}
        parameter = data.get("parameter")
        if parameter not in PARAMETERS:
            continue
        try:
            base = specs.GroundStateSpec.from_dict(data["base"])
        except (KeyError, specs.SpecError):
            continue
        if parameter in ("vacuum_A", "supercell"):
            same = base.digest == spec.digest or (
                evidence_key(base, parameter) == evidence_key(spec, parameter)
                and base.geometry_digest == spec.geometry_digest)
        else:
            same = evidence_key(base, parameter) == evidence_key(spec, parameter)
        if not same:
            continue
        mine = parameter_value(spec, parameter)
        settled = value.get("converged_from")
        if mine is None or settled is None:
            continue
        if not _at_least_as_accurate(parameter, float(mine), float(settled)):
            continue
        out.append({"parameter": parameter, "study_id": record.extra.get("study_id"),
                    "observable": value.get("observable"),
                    "tolerance": value.get("tolerance"), "residual": value.get("residual"),
                    "settled_from": settled, "this_run_at": mine})
    return out
