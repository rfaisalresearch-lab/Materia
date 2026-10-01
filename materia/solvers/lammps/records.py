"""LAMMPS runs held by a project: storing them, their state, and applying them.

A run is stored under ``lammps::<run_id>::<quantity>`` and its per-atom and
trajectory arrays under the same prefix in the project's checksummed array
store.  Only a finished run is ever stored, and it is stored in one step
after every check has passed: a refused, failed or cancelled run leaves no
result, no array and no history line behind.

State
-----
**current**
    The structure the run was computed for still has the geometry the run
    ended with.  After an energy run that is the geometry it started with.
**stale**
    The structure has changed since the run, or has not yet taken the run's
    final geometry.  In the second case the run can still be applied.
**detached**
    The run was computed for a structure the project does not hold.

Applying
--------
:func:`apply` writes the run's outcome onto its structure as exactly one
undoable project change: forces for an energy run; positions and forces for a
converged relaxation; positions, velocities and forces for dynamics.  It
refuses when the structure is no longer the one the run started from, when a
relaxation did not converge, and when the run is already applied.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ...project_format.history import LogEntry
from .run import PREFIX, STATUS_NOT_CONVERGED, LAMMPSOutcome, key
from .spec import LAMMPSRunSpec, SpecError

LABELS = {"energy": "LAMMPS forces", "relax": "LAMMPS relaxation",
          "md": "LAMMPS dynamics"}


class ApplyRefused(ValueError):
    """A run that cannot be written onto its structure, with the reason."""


def run_ids(project) -> List[str]:
    ids = {k.split("::")[1] for k in project.results
           if k.startswith(PREFIX) and k.count("::") == 2 and k.endswith("::run")}
    return sorted(ids, reverse=True)


def record(project, run_id: str):
    return project.results.get(key(run_id, "run"))


def spec_of(project, run_id: str) -> Optional[LAMMPSRunSpec]:
    rec = record(project, run_id)
    if rec is None:
        return None
    data = rec.provenance.parameters.get("run_spec")
    if data is None:
        return None
    try:
        return LAMMPSRunSpec.from_dict(data)
    except (SpecError, KeyError, TypeError, ValueError):
        return None


def store(project, outcome: LAMMPSOutcome, apply: bool = True) -> Dict[str, Any]:
    """Commit a finished outcome to ``project`` and optionally apply it."""
    if not outcome.ok:
        raise ValueError(f"A {outcome.status} LAMMPS run is not stored: {outcome.reason}")
    results = {key(outcome.run_id, name): result
               for name, result in outcome.results.items()}
    arrays = {key(outcome.run_id, name): stored
              for name, stored in outcome.arrays.items()}
    project.arrays.update(arrays)
    project.results.update(results)
    project.touch()
    applied, reason = False, "Not applied: apply was switched off."
    if apply:
        try:
            apply_run(project, outcome.run_id)
            applied, reason = True, "Applied to the structure as one undoable change."
        except ApplyRefused as exc:
            reason = f"Not applied: {exc}"
    for result in outcome.results.values():
        result.extra["applied_on_completion"] = applied
        result.extra["apply_reason"] = reason
    if not applied:
        spec = outcome.spec
        energy = outcome.results["energy"]
        project.history.log.append(LogEntry(
            operation=f"lammps.{spec.task}",
            label=f"LAMMPS {spec.task} with {spec.potential.get('id')}",
            parameters={"run_id": outcome.run_id, "structure_key": spec.structure_key,
                        "spec_digest": spec.digest, "status": outcome.status,
                        "lammps_version": spec.lammps.get("version")},
            result_summary=f"E = {energy.value:.6f} eV; {reason}"[:300], undoable=False))
        project.touch()
    return {"keys": sorted(results) + sorted(arrays), "applied": applied,
            "apply_reason": reason}


def _stored_arrays(project, run_id: str):
    names = ("final_positions", "final_velocities", "forces")
    found = {name: project.arrays.get(key(run_id, name)) for name in names}
    missing = [name for name, value in found.items() if value is None]
    if missing:
        raise ApplyRefused(f"run {run_id} has no stored {', '.join(missing)}; the record is "
                           "incomplete.")
    return found


def _live(project, spec: LAMMPSRunSpec, structure=None):
    if structure is not None:
        return structure
    if spec.structure_key is None:
        return None
    return project.structures.get(spec.structure_key)


def state(project, run_id: str, structure=None) -> Dict[str, Any]:
    """Whether a stored run is current, stale or detached, and why."""
    from ...python_api.api import structure_state_digest

    rec = record(project, run_id)
    if rec is None:
        return {"run_id": run_id, "state": "missing", "current": False, "applicable": False,
                "applied": False, "reason": f"No LAMMPS run {run_id}."}
    spec = spec_of(project, run_id)
    if spec is None:
        return {"run_id": run_id, "state": "missing", "current": False, "applicable": False,
                "applied": False, "reason": f"LAMMPS run {run_id} has no readable "
                                            "specification."}
    base = {"run_id": run_id, "structure_key": spec.structure_key, "task": spec.task,
            "status": rec.extra.get("status")}
    target = _live(project, spec, structure)
    if target is None:
        if spec.structure_key is None:
            return {**base, "state": "detached", "current": False, "applicable": False,
                    "applied": False,
                    "reason": "Computed for a structure this project does not hold."}
        return {**base, "state": "stale", "current": False, "applicable": False,
                "applied": False,
                "reason": f"Structure {spec.structure_key} is no longer in the project."}
    live = structure_state_digest(target)
    output = rec.extra.get("output_state_digest")
    converged = rec.extra.get("status") != STATUS_NOT_CONVERGED
    forces = project.arrays.get(key(run_id, "forces"))
    forces_attached = forces is not None and target.forces.shape == forces.data.shape \
        and np.array_equal(target.forces, forces.data)
    if live == output:
        applied = forces_attached
        return {**base, "state": "current", "current": True,
                "applicable": bool(converged and not applied and live == spec.input_state_digest
                                   and spec.structure_key is not None and structure is None),
                "applied": applied,
                "reason": "The structure has the geometry this run ended with."
                          + ("" if applied else " Its forces are not attached; apply the "
                                                "run to attach them.")}
    if live == spec.input_state_digest:
        why = ("The structure still has the geometry the run started from; apply the run "
               "to adopt its final geometry." if converged else
               "The relaxation did not converge, so its geometry is not an energy minimum "
               "and cannot be applied.")
        return {**base, "state": "stale", "current": False,
                "applicable": bool(converged and spec.structure_key is not None
                                   and structure is None),
                "applied": False, "reason": why}
    return {**base, "state": "stale", "current": False, "applicable": False,
            "applied": False,
            "reason": "The structure has changed since the run. The stored values describe "
                      "the earlier geometry and are kept as a record of it."}


def apply_run(project, run_id: str) -> str:
    """Write a stored run onto its structure as one undoable change."""
    from ...core_model.selection import Selection
    from ...python_api.api import structure_state_digest

    rec = record(project, run_id)
    if rec is None:
        raise ApplyRefused(f"there is no LAMMPS run {run_id}.")
    spec = spec_of(project, run_id)
    if spec is None:
        raise ApplyRefused(f"run {run_id} has no readable specification.")
    if rec.extra.get("status") == STATUS_NOT_CONVERGED:
        raise ApplyRefused("the relaxation did not reach its force tolerance, so its "
                           "geometry is not an energy minimum.")
    if spec.structure_key is None or spec.structure_key not in project.structures:
        raise ApplyRefused("the structure this run was computed for is not held by this "
                           "project.")
    current = project.structures[spec.structure_key]
    stored = _stored_arrays(project, run_id)
    live = structure_state_digest(current)
    if live != spec.input_state_digest:
        if live == rec.extra.get("output_state_digest") and \
                np.array_equal(current.forces, stored["forces"].data):
            raise ApplyRefused("the run is already applied.")
        raise ApplyRefused("the structure changed after the run started, so the result "
                           "describes a different geometry.")
    if spec.task == "energy" and np.array_equal(current.forces, stored["forces"].data):
        raise ApplyRefused("the run is already applied.")
    if list(current.ids) != list(spec.atom_ids):
        raise ApplyRefused("the atom identifiers of the structure no longer match the run.")
    before = current.copy()
    before_selection = Selection(list(project.selection.ids), project.selection.query)
    energy = project.results.get(key(run_id, "energy"))
    try:
        if spec.task in ("relax", "md"):
            current.positions = stored["final_positions"].data
        if spec.task == "md":
            current.velocities = stored["final_velocities"].data
        current.forces = stored["forces"].data
        recorded = project.record_change(
            current, before, before_selection, LABELS[spec.task],
            f"solver.lammps.{spec.task}",
            {"run_id": run_id, "potential": spec.potential.get("id"),
             "sha256": spec.potential.get("sha256"), "spec_digest": spec.digest,
             "lammps_version": spec.lammps.get("version")},
            result_summary=(f"E = {energy.value:.6f} eV" if energy is not None else ""))
        if not recorded:
            raise ApplyRefused("the structure is not held by this project.")
    except BaseException:
        current.restore_from(before)
        raise
    current.invalidate_bonds()
    return "Applied to the structure as one undoable change."


def summary(project, run_id: str) -> Dict[str, Any]:
    rec = record(project, run_id)
    energy = project.results.get(key(run_id, "energy"))
    spec = spec_of(project, run_id)
    info = state(project, run_id)
    return {
        "run_id": run_id, "task": spec.task if spec else None,
        "status": rec.extra.get("status") if rec else "missing",
        "energy_eV": energy.value if energy is not None else None,
        "formula": spec.formula if spec else None,
        "potential_id": spec.potential.get("id") if spec else None,
        "lammps_version": spec.lammps.get("version") if spec else None,
        "structure_key": spec.structure_key if spec else None,
        "spec_digest": spec.digest[:16] if spec else None,
        "state": info.get("state"), "current": info.get("current"),
        "applicable": info.get("applicable"), "applied": info.get("applied"),
    }
