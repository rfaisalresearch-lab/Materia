"""Storing ground-state runs in a project, and saying whether they still apply.

A run is stored under ``dft::<run_id>::<quantity>``; its large arrays under
the same prefix in the project's chunked array store.  Nothing here edits a
structure or records an undo point: a ground-state calculation observes a
structure, it does not change it.

A run is **current** while the structure it was computed for still has the
geometry fingerprint frozen into its specification, and **stale** as soon as
anything that fingerprint covers changes: atoms, elements, positions, cell,
periodicity, formal charges or magnetic moments.  Undoing the change makes it
current again.  An isotope change leaves it current, because the
Born-Oppenheimer ground state does not depend on nuclear mass, and the status
says so.  A run on a structure the project does not hold is **detached**.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...project_format.history import LogEntry
from ...provenance import Result
from . import spec as specs
from .run import GroundStateOutcome

PREFIX = "dft::"
STUDY_PREFIX = "dftstudy::"


def key(run_id: str, quantity: str) -> str:
    return f"{PREFIX}{run_id}::{quantity}"


def store(project, outcome: GroundStateOutcome, log: bool = True) -> List[str]:
    """Put a finished outcome into ``project`` and return the keys written."""
    written = []
    for name, stored in outcome.arrays.items():
        array_key = key(outcome.run_id, name)
        project.add_array(array_key, stored)
        written.append(array_key)
        result = outcome.results.get(name)
        if result is not None and isinstance(result.value, dict):
            result.value["array"] = array_key
    for name, result in outcome.results.items():
        project.add_result(key(outcome.run_id, name), result)
        written.append(key(outcome.run_id, name))
    if log:
        record = outcome.record
        summary = (f"E = {outcome.results['energy'].value:.6f} eV"
                   if "energy" in outcome.results else
                   f"{outcome.status}: {record.unsupported_reason}")
        project.history.log.append(LogEntry(
            operation="dft.ground_state",
            label=f"DFT ground state, {outcome.spec.formula}, {outcome.spec.xc}",
            parameters={"run_id": outcome.run_id,
                        "structure_key": outcome.spec.structure_key,
                        "spec_digest": outcome.spec.digest, "status": outcome.status},
            result_summary=summary[:300], undoable=False))
        project.touch()
    return written


def run_ids(project) -> List[str]:
    """Every stored run, newest first."""
    ids = {k.split("::")[1] for k in project.results
           if k.startswith(PREFIX) and k.count("::") == 2}
    return sorted(ids, reverse=True)


def record(project, run_id: str) -> Optional[Result]:
    return project.results.get(key(run_id, "run"))


def spec_of(project, run_id: str) -> Optional[specs.GroundStateSpec]:
    rec = record(project, run_id)
    if rec is None:
        return None
    data = rec.provenance.parameters.get("experiment_spec")
    return None if data is None else specs.GroundStateSpec.from_dict(data)


def status(project, run_id: str, structure=None) -> Dict[str, Any]:
    """Whether a run still describes its structure, and why not if it does not."""
    rec = record(project, run_id)
    if rec is None:
        return {"current": False, "state": "missing", "reason": f"No DFT run {run_id}."}
    spec = spec_of(project, run_id)
    base = {"run_id": run_id, "structure_key": spec.structure_key if spec else None,
            "converged": bool(rec.supported)}
    if not rec.supported:
        return {**base, "current": False, "state": rec.extra.get("status", "refused"),
                "reason": "The run produced no result, so there is nothing to be current."}
    derived = rec.extra.get("derived_from")
    target = structure
    if target is None and spec is not None and spec.structure_key is not None:
        target = project.structures.get(spec.structure_key)
    if target is None and derived:
        parent = project.structures.get(derived.get("structure_key") or "")
        if parent is None:
            return {**base, "current": False, "state": "stale",
                    "reason": "The structure this derived geometry was built from is no "
                              "longer in the project."}
        same = specs.geometry_digest(parent) == derived.get("geometry_digest")
        return {**base, "current": same, "state": "current" if same else "stale",
                "reason": (f"Derived from {derived.get('structure_key')} by "
                           f"{derived.get('description')}; the parent is unchanged."
                           if same else
                           f"Derived from {derived.get('structure_key')}, which has "
                           "changed since the run.")}
    if target is None:
        if spec is not None and spec.structure_key is None:
            return {**base, "current": False, "state": "detached",
                    "reason": "Computed for a structure this project does not hold. "
                              "Ask with that structure to check it."}
        return {**base, "current": False, "state": "stale",
                "reason": f"Structure {spec.structure_key if spec else '?'} is no longer "
                          "in the project."}
    now = specs.geometry_digest(target)
    if spec is not None and now == spec.geometry_digest:
        isotope_note = ""
        if [int(a) for a in target.mass_numbers] != list(spec.mass_numbers):
            isotope_note = (" The isotopes have changed since the run; the "
                            "Born-Oppenheimer electronic ground state does not depend "
                            "on nuclear mass, so the result still applies. Vibrational "
                            "and dynamical quantities would change.")
        return {**base, "current": True, "state": "current",
                "reason": "The structure still has the geometry this run was computed "
                          "for." + isotope_note,
                "isotopes_changed": bool(isotope_note)}
    change = specs.describe_geometry_change(spec, target) if spec else "it changed"
    return {**base, "current": False, "state": "stale",
            "reason": f"Stale: {change} since the run. The stored values describe the "
                      "earlier geometry and are kept as a record of it."}


def summary(project, run_id: str) -> Dict[str, Any]:
    """One line of the run list."""
    rec = record(project, run_id)
    energy = project.results.get(key(run_id, "energy"))
    spec = spec_of(project, run_id)
    info = status(project, run_id)
    return {
        "run_id": run_id,
        "status": rec.extra.get("status") if rec else "missing",
        "supported": bool(rec.supported) if rec else False,
        "energy_eV": energy.value if energy is not None else None,
        "formula": spec.formula if spec else None,
        "xc": spec.xc if spec else None,
        "representation": spec.representation if spec else None,
        "boundary": spec.boundary if spec else None,
        "structure_key": spec.structure_key if spec else None,
        "spec_digest": spec.digest[:16] if spec else None,
        "state": info.get("state"), "current": info.get("current"),
        "study_id": rec.extra.get("study_id") if rec else None,
    }


RELAX_PREFIX = "dftrelax::"
RELAX_LABELS = {"fixed-cell": "DFT relaxation", "variable-cell": "DFT variable-cell relaxation"}


class ApplyRefused(ValueError):
    """A relaxation that cannot be written onto its structure, with the reason."""


def relax_key(run_id: str, quantity: str) -> str:
    return f"{RELAX_PREFIX}{run_id}::{quantity}"


def relax_run_ids(project) -> List[str]:
    ids = {k.split("::")[1] for k in project.results
           if k.startswith(RELAX_PREFIX) and k.count("::") == 2 and k.endswith("::relaxation")}
    return sorted(ids, reverse=True)


def relax_record(project, run_id: str) -> Optional[Result]:
    return project.results.get(relax_key(run_id, "relaxation"))


def relax_spec_of(project, run_id: str):
    from .relaxation import RelaxationSpec

    rec = relax_record(project, run_id)
    if rec is None:
        return None
    data = rec.provenance.parameters.get("relaxation_spec")
    if data is None:
        return None
    try:
        return RelaxationSpec.from_dict(data)
    except (specs.SpecError, KeyError, TypeError, ValueError):
        return None


def store_relaxation(project, outcome, apply: bool = True) -> Dict[str, Any]:
    """Commit a finished relaxation in one step and optionally apply it.

    Only a converged or not-converged relaxation reaches here; a cancelled,
    failed or refused one has nothing to store and is rejected.  Results and
    arrays are written together, then the apply is attempted.  If the apply is
    refused the run is still kept, and one line that is not an undo point
    records it; if it is applied, the single undoable change is the only
    history line.
    """
    if not outcome.ok:
        raise ValueError(f"A {outcome.status} relaxation is not stored: {outcome.reason}")
    results = {relax_key(outcome.run_id, name): item
               for name, item in outcome.results.items()}
    arrays = {relax_key(outcome.run_id, name): stored
              for name, stored in outcome.arrays.items()}
    for item in outcome.results.values():
        if isinstance(item.value, dict) and isinstance(item.value.get("array"), str) \
                and not item.value["array"].startswith(RELAX_PREFIX):
            item.value["array"] = relax_key(outcome.run_id, item.value["array"])
    project.arrays.update(arrays)
    project.results.update(results)
    project.touch()
    applied, reason = False, "Not applied: apply was switched off."
    if apply:
        try:
            apply_relaxation(project, outcome.run_id)
            applied, reason = True, "Applied to the structure as one undoable change."
        except ApplyRefused as exc:
            reason = f"Not applied: {exc}"
    for item in outcome.results.values():
        item.extra["applied_on_completion"] = applied
        item.extra["apply_reason"] = reason
    if not applied:
        spec = outcome.spec
        summary = outcome.results["relaxation"].value
        project.history.log.append(LogEntry(
            operation=f"dft.relax.{spec.mode}",
            label=f"{RELAX_LABELS[spec.mode]}, {spec.ground_state.formula}, "
                  f"{spec.ground_state.xc}",
            parameters={"run_id": outcome.run_id, "structure_key": spec.structure_key,
                        "spec_digest": spec.digest, "status": outcome.status},
            result_summary=(f"E = {summary['final_energy_eV']:.6f} eV after "
                            f"{summary['optimizer_steps']} step(s); {reason}")[:300],
            undoable=False))
        project.touch()
    return {"keys": sorted(results) + sorted(arrays), "applied": applied,
            "apply_reason": reason}


def _relax_arrays(project, run_id: str) -> Dict[str, Any]:
    names = ("final_positions", "final_cell", "final_forces")
    found = {name: project.arrays.get(relax_key(run_id, name)) for name in names}
    missing = [name for name, value in found.items() if value is None]
    if missing:
        raise ApplyRefused(f"run {run_id} has no stored {', '.join(missing)}; the record is "
                           "incomplete.")
    return found


def relax_status(project, run_id: str, structure=None) -> Dict[str, Any]:
    """Whether a relaxation is current, stale or detached, and whether it can be applied."""
    import numpy as np

    rec = relax_record(project, run_id)
    spec = relax_spec_of(project, run_id)
    if rec is None or spec is None:
        return {"run_id": run_id, "state": "missing", "current": False, "applicable": False,
                "applied": False, "reason": f"No readable DFT relaxation {run_id}."}
    base = {"run_id": run_id, "structure_key": spec.structure_key, "mode": spec.mode,
            "status": rec.extra.get("status")}
    target = structure
    if target is None and spec.structure_key is not None:
        target = project.structures.get(spec.structure_key)
    if target is None:
        if spec.structure_key is None:
            return {**base, "state": "detached", "current": False, "applicable": False,
                    "applied": False,
                    "reason": "Computed for a structure this project does not hold."}
        return {**base, "state": "stale", "current": False, "applicable": False,
                "applied": False,
                "reason": f"Structure {spec.structure_key} is no longer in the project."}
    live = specs.geometry_digest(target)
    output = rec.extra.get("output_geometry_digest")
    converged = rec.extra.get("status") == "converged"
    held = structure is None and spec.structure_key is not None
    forces = project.arrays.get(relax_key(run_id, "final_forces"))
    attached = forces is not None and target.forces.shape == forces.data.shape and \
        np.array_equal(target.forces, forces.data)
    if live == output:
        return {**base, "state": "current", "current": True, "applied": attached,
                "applicable": bool(converged and held and not attached
                                   and live == spec.geometry_digest),
                "reason": "The structure has the geometry this relaxation ended with."
                          + ("" if attached else " Its forces are not attached.")}
    if live == spec.geometry_digest:
        return {**base, "state": "stale", "current": False, "applied": False,
                "applicable": bool(converged and held),
                "reason": ("The structure still has the geometry the relaxation started "
                           "from; apply it to adopt the relaxed geometry." if converged else
                           "The relaxation did not converge, so its geometry is not a "
                           "minimum and cannot be applied.")}
    change = specs.describe_geometry_change(spec.ground_state, target)
    return {**base, "state": "stale", "current": False, "applicable": False,
            "applied": False,
            "reason": f"Stale: {change} since the relaxation started. The stored values "
                      "describe the relaxed geometry and are kept as a record of it."}


def apply_relaxation(project, run_id: str) -> str:
    """Write a stored relaxation onto its structure as one undoable change."""
    import numpy as np

    from ...core_model.cell import Cell
    from ...core_model.selection import Selection

    rec = relax_record(project, run_id)
    spec = relax_spec_of(project, run_id)
    if rec is None or spec is None:
        raise ApplyRefused(f"there is no readable DFT relaxation {run_id}.")
    if rec.extra.get("status") != "converged":
        raise ApplyRefused("the relaxation did not meet its criteria, so its geometry is not "
                           "a minimum.")
    if spec.structure_key is None or spec.structure_key not in project.structures:
        raise ApplyRefused("the structure this relaxation was computed for is not held by "
                           "this project.")
    current = project.structures[spec.structure_key]
    stored = _relax_arrays(project, run_id)
    live = specs.geometry_digest(current)
    if live != spec.geometry_digest:
        if live == rec.extra.get("output_geometry_digest") and \
                np.array_equal(current.forces, stored["final_forces"].data):
            raise ApplyRefused("the relaxation is already applied.")
        raise ApplyRefused("the structure changed after the relaxation started ("
                           + specs.describe_geometry_change(spec.ground_state, current)
                           + "), so the relaxed geometry belongs to a different structure.")
    if [int(i) for i in current.ids] != list(spec.ground_state.atom_ids):
        raise ApplyRefused("the atom identifiers of the structure no longer match.")
    before = current.copy()
    before_selection = Selection(list(project.selection.ids), project.selection.query)
    energy = project.results.get(relax_key(run_id, "energy"))
    try:
        current.positions = stored["final_positions"].data
        if spec.variable_cell:
            current.cell = Cell(np.asarray(stored["final_cell"].data, dtype=float),
                                tuple(spec.ground_state.pbc))
        current.forces = stored["final_forces"].data
        recorded = project.record_change(
            current, before, before_selection, RELAX_LABELS[spec.mode],
            f"solver.dft.relax.{spec.mode}",
            {"run_id": run_id, "spec_digest": spec.digest, "xc": spec.ground_state.xc,
             "mode": spec.mode},
            result_summary=(f"E = {energy.value:.6f} eV" if energy is not None else ""))
        if not recorded:
            raise ApplyRefused("the structure is not held by this project.")
    except BaseException:
        current.restore_from(before)
        raise
    current.invalidate_bonds()
    return "Applied to the structure as one undoable change."


def relax_summary(project, run_id: str) -> Dict[str, Any]:
    rec = relax_record(project, run_id)
    spec = relax_spec_of(project, run_id)
    info = relax_status(project, run_id)
    value = (rec.value if rec is not None else None) or {}
    return {
        "run_id": run_id, "status": rec.extra.get("status") if rec else "missing",
        "mode": spec.mode if spec else None,
        "formula": spec.ground_state.formula if spec else None,
        "xc": spec.ground_state.xc if spec else None,
        "final_energy_eV": value.get("final_energy_eV"),
        "optimizer_steps": value.get("optimizer_steps"),
        "structure_key": spec.structure_key if spec else None,
        "spec_digest": spec.digest[:16] if spec else None,
        "state": info.get("state"), "current": info.get("current"),
        "applicable": info.get("applicable"), "applied": info.get("applied"),
    }


DOS_PREFIX = "dftdos::"


def dos_key(run_id: str, quantity: str) -> str:
    return f"{DOS_PREFIX}{run_id}::{quantity}"


def dos_run_ids(project) -> List[str]:
    ids = {k.split("::")[1] for k in project.results
           if k.startswith(DOS_PREFIX) and k.count("::") == 2 and k.endswith("::dos")}
    return sorted(ids, reverse=True)


def dos_record(project, run_id: str) -> Optional[Result]:
    return project.results.get(dos_key(run_id, "dos"))


def dos_spec_of(project, run_id: str):
    from .dos import DOSSpec

    rec = dos_record(project, run_id)
    if rec is None:
        return None
    data = rec.provenance.parameters.get("dos_spec")
    if data is None:
        return None
    try:
        return DOSSpec.from_dict(data)
    except (specs.SpecError, KeyError, TypeError, ValueError):
        return None


def store_dos(project, outcome) -> List[str]:
    """Commit a complete DOS in one step, with one history line that is not an undo point.

    A refused, failed or cancelled DOS has nothing to store and is rejected.
    The DOS observes a structure and never changes it.
    """
    if not outcome.ok:
        raise ValueError(f"A {outcome.status} DOS is not stored: {outcome.reason}")
    results = {dos_key(outcome.run_id, name): item for name, item in outcome.results.items()}
    arrays = {dos_key(outcome.run_id, name): stored
              for name, stored in outcome.arrays.items()}
    project.arrays.update(arrays)
    project.results.update(results)
    spec = outcome.spec
    gs = spec.ground_state
    source = spec.source
    project.history.log.append(LogEntry(
        operation="dft.dos",
        label=f"DFT DOS, {gs.formula}, {gs.xc}"
              + (f", from {source.get('kind')} {source.get('run_id')}"
                 if source.get("run_id") else ""),
        parameters={"run_id": outcome.run_id, "structure_key": spec.structure_key,
                    "spec_digest": spec.digest, "source": dict(source)},
        result_summary=(f"E_F = {outcome.results['dos'].value['fermi_level_eV']:.4f} eV, "
                        f"{len(spec.projections)} projection(s)")[:300],
        undoable=False))
    project.touch()
    return sorted(results) + sorted(arrays)


def dos_status(project, run_id: str, structure=None) -> Dict[str, Any]:
    """Whether a DOS still describes its structure: current, stale or detached."""
    rec = dos_record(project, run_id)
    spec = dos_spec_of(project, run_id)
    if rec is None or spec is None:
        return {"run_id": run_id, "state": "missing", "current": False,
                "reason": f"No readable DOS {run_id}."}
    base = {"run_id": run_id, "structure_key": spec.structure_key,
            "source": dict(spec.source)}
    target = structure
    if target is None and spec.structure_key is not None:
        target = project.structures.get(spec.structure_key)
    if target is None:
        if spec.structure_key is None:
            return {**base, "state": "detached", "current": False,
                    "reason": "Computed for a structure this project does not hold."}
        return {**base, "state": "stale", "current": False,
                "reason": f"Structure {spec.structure_key} is no longer in the project."}
    live = specs.geometry_digest(target)
    if live == spec.geometry_digest:
        return {**base, "state": "current", "current": True,
                "reason": "The structure has the geometry this DOS was computed for."}
    if spec.source.get("kind") == "relaxation":
        relaxed = relax_record(project, spec.source.get("run_id") or "")
        if relaxed is not None and live == relaxed.extra.get("input_geometry_digest"):
            return {**base, "state": "stale", "current": False,
                    "reason": "This DOS is of the relaxed geometry, which has not been "
                              "applied to the structure. Apply the relaxation to make it "
                              "current."}
    change = specs.describe_geometry_change(spec.ground_state, target)
    return {**base, "state": "stale", "current": False,
            "reason": f"Stale: {change} since the DOS was computed. The stored DOS "
                      "describes the earlier geometry."}


def dos_summary(project, run_id: str) -> Dict[str, Any]:
    rec = dos_record(project, run_id)
    spec = dos_spec_of(project, run_id)
    info = dos_status(project, run_id)
    value = (rec.value if rec is not None else None) or {}
    return {
        "run_id": run_id, "status": value.get("status", "missing"),
        "formula": spec.ground_state.formula if spec else None,
        "xc": spec.ground_state.xc if spec else None,
        "fermi_level_eV": value.get("fermi_level_eV"),
        "broadening": spec.broadening if spec else None,
        "projections": len(spec.projections) if spec else 0,
        "source": dict(spec.source) if spec else None,
        "structure_key": spec.structure_key if spec else None,
        "spec_digest": spec.digest[:16] if spec else None,
        "state": info.get("state"), "current": info.get("current"),
    }


BANDS_PREFIX = "dftbands::"
BANDS_ARRAYS = ("eigenvalues", "kpoints_frac", "kpoints_cartesian", "distance")


def bands_key(run_id: str, quantity: str) -> str:
    return f"{BANDS_PREFIX}{run_id}::{quantity}"


def bands_run_ids(project) -> List[str]:
    ids = {k.split("::")[1] for k in project.results
           if k.startswith(BANDS_PREFIX) and k.count("::") == 2 and k.endswith("::bands")}
    return sorted(ids, reverse=True)


def bands_record(project, run_id: str) -> Optional[Result]:
    return project.results.get(bands_key(run_id, "bands"))


def bands_spec_of(project, run_id: str):
    from .bands import BandStructureSpec

    rec = bands_record(project, run_id)
    if rec is None:
        return None
    data = rec.provenance.parameters.get("bands_spec")
    if data is None:
        return None
    try:
        return BandStructureSpec.from_dict(data)
    except (specs.SpecError, KeyError, TypeError, ValueError):
        return None


def bands_integrity(project, run_id: str) -> Optional[str]:
    """Why the stored arrays of a band structure cannot be trusted, or None if they can."""
    rec = bands_record(project, run_id)
    spec = bands_spec_of(project, run_id)
    if rec is None or spec is None:
        return f"No readable band structure {run_id}."
    expected = (rec.value or {}).get("array_sha256") or {}
    if set(expected) != set(BANDS_ARRAYS):
        return "The record does not list a checksum for every stored array."
    for name in BANDS_ARRAYS:
        stored = project.arrays.get(bands_key(run_id, name))
        if stored is None:
            return f"The stored array {name} is missing."
        if stored.sha256() != expected[name]:
            return (f"The stored array {name} does not match the SHA-256 recorded when the "
                    "band structure was computed.")
    shape = project.arrays[bands_key(run_id, "eigenvalues")].data.shape
    if tuple(shape) != (spec.n_spins, spec.n_kpoints, spec.n_bands):
        return "The stored eigenvalues do not have the shape of the specification."
    import numpy as np

    kpts = np.asarray(project.arrays[bands_key(run_id, "kpoints_frac")].data, dtype=float)
    if kpts.shape != (spec.n_kpoints, 3) or not np.allclose(
            kpts, np.asarray(spec.kpoints_frac, dtype=float), rtol=0, atol=1e-12):
        return "The stored k-points are not the specification's path."
    return None


def store_bands(project, outcome) -> List[str]:
    """Commit a complete band structure in one step, with one history line.

    A refused, failed or cancelled band structure has nothing to store and is
    rejected.  Every array is checked against the checksum the outcome
    recorded before anything is written; results and arrays are then written
    together, and removed again if either write fails.  The history line is
    not an undo point: a band structure observes a structure and never
    changes it.
    """
    if not outcome.ok:
        raise ValueError(f"A {outcome.status} band structure is not stored: {outcome.reason}")
    expected = outcome.results["bands"].value.get("array_sha256") or {}
    if set(expected) != set(outcome.arrays):
        raise ValueError("The band structure's arrays and its checksums do not match; "
                         "nothing was stored.")
    for name, stored in outcome.arrays.items():
        if stored.sha256() != expected[name]:
            raise ValueError(f"Array {name} changed after it was verified; nothing was "
                             "stored.")
    results = {bands_key(outcome.run_id, name): item for name, item in outcome.results.items()}
    arrays = {bands_key(outcome.run_id, name): stored
              for name, stored in outcome.arrays.items()}
    clash = sorted(k for k in list(results) + list(arrays)
                   if k in project.results or k in project.arrays)
    if clash:
        raise ValueError(f"Run {outcome.run_id} is already stored; nothing was overwritten.")
    try:
        project.arrays.update(arrays)
        project.results.update(results)
    except BaseException:
        for k in arrays:
            project.arrays.pop(k, None)
        for k in results:
            project.results.pop(k, None)
        raise
    spec = outcome.spec
    gs = spec.ground_state
    source = spec.source
    edges = outcome.results["bands"].value.get("band_edges") or {}
    gap = edges.get("gap_eV")
    project.history.log.append(LogEntry(
        operation="dft.bands",
        label=f"DFT band structure, {gs.formula}, {gs.xc}"
              + (f", from {source.get('kind')} {source.get('run_id')}"
                 if source.get("run_id") else ""),
        parameters={"run_id": outcome.run_id, "structure_key": spec.structure_key,
                    "spec_digest": spec.digest, "source": dict(source)},
        result_summary=(f"E_F = {outcome.results['bands'].value['fermi_level_eV']:.4f} eV, "
                        f"{spec.n_kpoints} k-points, "
                        + (f"gap on path {gap:.3f} eV ({edges.get('gap_kind')})"
                           if gap is not None else "no gap on path"))[:300],
        undoable=False))
    project.touch()
    return sorted(results) + sorted(arrays)


def bands_status(project, run_id: str, structure=None) -> Dict[str, Any]:
    """Whether a band structure still describes its structure: current, stale or detached.

    A record whose arrays fail their checksums is ``corrupt`` whatever the
    geometry says.
    """
    rec = bands_record(project, run_id)
    spec = bands_spec_of(project, run_id)
    if rec is None or spec is None:
        return {"run_id": run_id, "state": "missing", "current": False,
                "reason": f"No readable band structure {run_id}."}
    base = {"run_id": run_id, "structure_key": spec.structure_key,
            "source": dict(spec.source)}
    damage = bands_integrity(project, run_id)
    if damage:
        return {**base, "state": "corrupt", "current": False,
                "reason": f"Refused: {damage} The stored bands are not shown."}
    target = structure
    if target is None and spec.structure_key is not None:
        target = project.structures.get(spec.structure_key)
    if target is None:
        if spec.structure_key is None:
            return {**base, "state": "detached", "current": False,
                    "reason": "Computed for a structure this project does not hold."}
        return {**base, "state": "stale", "current": False,
                "reason": f"Structure {spec.structure_key} is no longer in the project."}
    live = specs.geometry_digest(target)
    if live == spec.geometry_digest:
        return {**base, "state": "current", "current": True,
                "reason": "The structure has the geometry these bands were computed for."}
    if spec.source.get("kind") == "relaxation":
        relaxed = relax_record(project, spec.source.get("run_id") or "")
        if relaxed is not None and live == relaxed.extra.get("input_geometry_digest"):
            return {**base, "state": "stale", "current": False,
                    "reason": "These bands are of the relaxed geometry, which has not been "
                              "applied to the structure. Apply the relaxation to make them "
                              "current."}
    change = specs.describe_geometry_change(spec.ground_state, target)
    return {**base, "state": "stale", "current": False,
            "reason": f"Stale: {change} since the bands were computed. The stored bands "
                      "describe the earlier geometry."}


def bands_summary(project, run_id: str) -> Dict[str, Any]:
    rec = bands_record(project, run_id)
    spec = bands_spec_of(project, run_id)
    info = bands_status(project, run_id)
    value = (rec.value if rec is not None else None) or {}
    edges = value.get("band_edges") or {}
    return {
        "run_id": run_id, "status": value.get("status", "missing"),
        "formula": spec.ground_state.formula if spec else None,
        "xc": spec.ground_state.xc if spec else None,
        "fermi_level_eV": value.get("fermi_level_eV"),
        "path": value.get("path"), "n_kpoints": value.get("n_kpoints"),
        "gap_eV": edges.get("gap_eV"), "gap_kind": edges.get("gap_kind"),
        "source": dict(spec.source) if spec else None,
        "structure_key": spec.structure_key if spec else None,
        "spec_digest": spec.digest[:16] if spec else None,
        "state": info.get("state"), "current": info.get("current"),
    }


EOS_PREFIX = "dfteos::"


def eos_key(run_id: str, quantity: str) -> str:
    return f"{EOS_PREFIX}{run_id}::{quantity}"


def eos_run_ids(project) -> List[str]:
    ids = {k.split("::")[1] for k in project.results
           if k.startswith(EOS_PREFIX) and k.count("::") == 2 and k.endswith("::eos")}
    return sorted(ids, reverse=True)


def eos_record(project, run_id: str) -> Optional[Result]:
    return project.results.get(eos_key(run_id, "eos"))


def eos_spec_of(project, run_id: str):
    from .eos import EOSSpec

    rec = eos_record(project, run_id)
    if rec is None:
        return None
    data = rec.provenance.parameters.get("eos_spec")
    if data is None:
        return None
    try:
        return EOSSpec.from_dict(data)
    except (specs.SpecError, KeyError, TypeError, ValueError):
        return None


def eos_integrity(project, run_id: str) -> Optional[str]:
    """Why the stored arrays of an equation of state cannot be trusted, or None."""
    rec = eos_record(project, run_id)
    spec = eos_spec_of(project, run_id)
    if rec is None or spec is None:
        return f"No readable equation of state {run_id}."
    expected = (rec.value or {}).get("array_sha256") or {}
    if not {"scales", "volumes", "energies", "free_energies", "fit_pressures",
            "free_energy_fit_pressures"} <= set(expected):
        return "The record does not list a checksum for every stored array."
    for name, digest in expected.items():
        stored = project.arrays.get(eos_key(run_id, name))
        if stored is None:
            return f"The stored array {name} is missing."
        if stored.sha256() != digest:
            return (f"The stored array {name} does not match the SHA-256 recorded when the "
                    "equation of state was computed.")
    if project.arrays[eos_key(run_id, "energies")].data.shape != (spec.n_points,):
        return "The stored energies do not have one value per specified volume."
    return None


def store_eos(project, outcome) -> List[str]:
    """Commit a complete equation of state in one step, with one history line.

    Nothing is stored for a refused, failed or cancelled one.  Arrays are
    checked against the checksums recorded with them before anything is
    written, and results and arrays are written together.
    """
    if not outcome.ok:
        raise ValueError(f"A {outcome.status} equation of state is not stored: "
                         f"{outcome.reason}")
    expected = outcome.results["eos"].value.get("array_sha256") or {}
    if set(expected) != set(outcome.arrays):
        raise ValueError("The equation of state's arrays and checksums do not match; nothing "
                         "was stored.")
    for name, stored in outcome.arrays.items():
        if stored.sha256() != expected[name]:
            raise ValueError(f"Array {name} changed after it was verified; nothing was stored.")
    results = {eos_key(outcome.run_id, name): item for name, item in outcome.results.items()}
    arrays = {eos_key(outcome.run_id, name): stored for name, stored in outcome.arrays.items()}
    if any(k in project.results or k in project.arrays for k in list(results) + list(arrays)):
        raise ValueError(f"Run {outcome.run_id} is already stored; nothing was overwritten.")
    project.arrays.update(arrays)
    project.results.update(results)
    spec = outcome.spec
    value = outcome.results["eos"].value
    project.history.log.append(LogEntry(
        operation="dft.eos",
        label=f"DFT equation of state, {spec.ground_state.formula}, {spec.ground_state.xc}",
        parameters={"run_id": outcome.run_id, "structure_key": spec.structure_key,
                    "spec_digest": spec.digest, "source": dict(spec.source)},
        result_summary=(f"V0 = {value['V0_A3_per_atom']:.4f} A^3/atom, "
                        f"B0 = {value['B0_GPa']:.1f} GPa, B' = {value['B1']:.2f}")[:300],
        undoable=False))
    project.touch()
    return sorted(results) + sorted(arrays)


def eos_status(project, run_id: str, structure=None) -> Dict[str, Any]:
    """Current, applied, stale, detached or corrupt, and whether it can be applied."""
    rec = eos_record(project, run_id)
    spec = eos_spec_of(project, run_id)
    if rec is None or spec is None:
        return {"run_id": run_id, "state": "missing", "current": False, "applicable": False,
                "reason": f"No readable equation of state {run_id}."}
    base = {"run_id": run_id, "structure_key": spec.structure_key, "source": dict(spec.source)}
    damage = eos_integrity(project, run_id)
    if damage:
        return {**base, "state": "corrupt", "current": False, "applicable": False,
                "reason": f"Refused: {damage} The stored values are not shown."}
    target = structure
    if target is None and spec.structure_key is not None:
        target = project.structures.get(spec.structure_key)
    if target is None:
        if spec.structure_key is None:
            return {**base, "state": "detached", "current": False, "applicable": False,
                    "reason": "Computed for a structure this project does not hold."}
        return {**base, "state": "stale", "current": False, "applicable": False,
                "reason": f"Structure {spec.structure_key} is no longer in the project."}
    live = specs.geometry_digest(target)
    held = structure is None and spec.structure_key is not None
    if live == spec.geometry_digest:
        return {**base, "state": "current", "current": True, "applicable": held,
                "reason": "The structure has the reference geometry; the fitted equilibrium "
                          "volume can be applied."}
    if live == rec.extra.get("equilibrium_geometry_digest"):
        return {**base, "state": "applied", "current": True, "applicable": False,
                "reason": "The structure has been scaled to the fitted equilibrium volume."}
    change = specs.describe_geometry_change(spec.ground_state, target)
    return {**base, "state": "stale", "current": False, "applicable": False,
            "reason": f"Stale: {change} since the equation of state was computed."}


def apply_eos(project, run_id: str) -> str:
    """Scale the structure to the fitted equilibrium volume as one undoable change."""
    import numpy as np

    from ...core_model.cell import Cell
    from ...core_model.selection import Selection

    rec = eos_record(project, run_id)
    spec = eos_spec_of(project, run_id)
    if rec is None or spec is None:
        raise ApplyRefused(f"there is no readable equation of state {run_id}.")
    damage = eos_integrity(project, run_id)
    if damage:
        raise ApplyRefused(damage)
    if spec.structure_key is None or spec.structure_key not in project.structures:
        raise ApplyRefused("the structure this equation of state was computed for is not held "
                           "by this project.")
    current = project.structures[spec.structure_key]
    live = specs.geometry_digest(current)
    if live == rec.extra.get("equilibrium_geometry_digest"):
        raise ApplyRefused("the equilibrium volume is already applied.")
    if live != spec.geometry_digest:
        raise ApplyRefused("the structure changed after the equation of state was computed ("
                           + specs.describe_geometry_change(spec.ground_state, current)
                           + "), so its equilibrium belongs to a different structure.")
    value = rec.value
    linear = float(value["linear_scale"])
    before = current.copy()
    before_selection = Selection(list(project.selection.ids), project.selection.query)
    try:
        current.positions = np.asarray(current.positions, dtype=float) * linear
        current.cell = Cell(np.asarray(current.cell.matrix, dtype=float) * linear,
                            tuple(current.cell.pbc))
        recorded = project.record_change(
            current, before, before_selection, "DFT equilibrium volume", "solver.dft.eos",
            {"run_id": run_id, "spec_digest": spec.digest, "linear_scale": linear},
            result_summary=(f"V0 = {value['V0_A3_per_atom']:.4f} A^3/atom, "
                            f"B0 = {value['B0_GPa']:.1f} GPa"))
        if not recorded:
            raise ApplyRefused("the structure is not held by this project.")
    except BaseException:
        current.restore_from(before)
        raise
    current.invalidate_bonds()
    return (f"Scaled every lattice vector by {linear:.6f} to the fitted equilibrium volume, as "
            "one undoable change.")


def eos_summary(project, run_id: str) -> Dict[str, Any]:
    rec = eos_record(project, run_id)
    spec = eos_spec_of(project, run_id)
    info = eos_status(project, run_id)
    value = (rec.value if rec is not None else None) or {}
    return {
        "run_id": run_id, "status": value.get("status", "missing"),
        "formula": spec.ground_state.formula if spec else None,
        "xc": spec.ground_state.xc if spec else None,
        "V0_A3_per_atom": value.get("V0_A3_per_atom"), "B0_GPa": value.get("B0_GPa"),
        "B1": value.get("B1"), "source": dict(spec.source) if spec else None,
        "structure_key": spec.structure_key if spec else None,
        "spec_digest": spec.digest[:16] if spec else None,
        "state": info.get("state"), "current": info.get("current"),
        "applicable": info.get("applicable"),
    }


LDOS_PREFIX = "dftldos::"


def ldos_key(run_id: str, quantity: str) -> str:
    return f"{LDOS_PREFIX}{run_id}::{quantity}"


def ldos_run_ids(project) -> List[str]:
    ids = {k.split("::")[1] for k in project.results
           if k.startswith(LDOS_PREFIX) and k.count("::") == 2 and k.endswith("::ldos")}
    return sorted(ids, reverse=True)


def ldos_record(project, run_id: str) -> Optional[Result]:
    return project.results.get(ldos_key(run_id, "ldos"))


def ldos_spec_of(project, run_id: str):
    from .ldos import LDOSSpec

    rec = ldos_record(project, run_id)
    if rec is None:
        return None
    data = rec.provenance.parameters.get("ldos_spec")
    if data is None:
        return None
    try:
        return LDOSSpec.from_dict(data)
    except (specs.SpecError, KeyError, TypeError, ValueError):
        return None


def ldos_integrity(project, run_id: str) -> Optional[str]:
    """Why the stored LDOS arrays cannot be trusted, or None if they can."""
    rec = ldos_record(project, run_id)
    spec = ldos_spec_of(project, run_id)
    if rec is None or spec is None:
        return f"No readable LDOS {run_id}."
    expected = (rec.value or {}).get("array_sha256") or {}
    required = {"ldos", "eigenvalues", "kpoint_weights"}
    if spec.spin_channels == "resolved":
        required.add("ldos_spin")
    if set(expected) != required:
        return "The record does not list a checksum for every stored array."
    for name, digest in expected.items():
        stored = project.arrays.get(ldos_key(run_id, name))
        if stored is None:
            return f"The stored array {name} is missing."
        if stored.sha256() != digest:
            return (f"The stored array {name} does not match the SHA-256 recorded when the "
                    "LDOS was computed.")
    shape = list(project.arrays[ldos_key(run_id, "ldos")].data.shape)
    if shape != list((rec.value or {}).get("grid_shape") or []):
        return "The stored map does not have the grid the record describes."
    return None


def store_ldos(project, outcome) -> List[str]:
    """Commit a complete LDOS in one step, with one history line that is not an undo point."""
    if not outcome.ok:
        raise ValueError(f"A {outcome.status} LDOS is not stored: {outcome.reason}")
    expected = outcome.results["ldos"].value.get("array_sha256") or {}
    if set(expected) != set(outcome.arrays):
        raise ValueError("The LDOS arrays and checksums do not match; nothing was stored.")
    for name, stored in outcome.arrays.items():
        if stored.sha256() != expected[name]:
            raise ValueError(f"Array {name} changed after it was verified; nothing was stored.")
    results = {ldos_key(outcome.run_id, name): item for name, item in outcome.results.items()}
    arrays = {ldos_key(outcome.run_id, name): stored for name, stored in outcome.arrays.items()}
    if any(k in project.results or k in project.arrays for k in list(results) + list(arrays)):
        raise ValueError(f"Run {outcome.run_id} is already stored; nothing was overwritten.")
    project.arrays.update(arrays)
    project.results.update(results)
    spec = outcome.spec
    value = outcome.results["ldos"].value
    project.history.log.append(LogEntry(
        operation="dft.ldos",
        label=f"DFT LDOS, {spec.ground_state.formula}, {spec.ground_state.xc}",
        parameters={"run_id": outcome.run_id, "structure_key": spec.structure_key,
                    "spec_digest": spec.digest, "source": dict(spec.source)},
        result_summary=(f"{spec.energy_min_eV:g} to {spec.energy_max_eV:g} eV about E_F, "
                        f"{value['states_in_window']:.4g} states per cell")[:300],
        undoable=False))
    project.touch()
    return sorted(results) + sorted(arrays)


def ldos_status(project, run_id: str, structure=None) -> Dict[str, Any]:
    """Whether an LDOS still describes its structure: current, stale, detached or corrupt."""
    rec = ldos_record(project, run_id)
    spec = ldos_spec_of(project, run_id)
    if rec is None or spec is None:
        return {"run_id": run_id, "state": "missing", "current": False,
                "reason": f"No readable LDOS {run_id}."}
    base = {"run_id": run_id, "structure_key": spec.structure_key, "source": dict(spec.source)}
    damage = ldos_integrity(project, run_id)
    if damage:
        return {**base, "state": "corrupt", "current": False,
                "reason": f"Refused: {damage} The stored map is not shown."}
    target = structure
    if target is None and spec.structure_key is not None:
        target = project.structures.get(spec.structure_key)
    if target is None:
        if spec.structure_key is None:
            return {**base, "state": "detached", "current": False,
                    "reason": "Computed for a structure this project does not hold."}
        return {**base, "state": "stale", "current": False,
                "reason": f"Structure {spec.structure_key} is no longer in the project."}
    live = specs.geometry_digest(target)
    if live == spec.geometry_digest:
        return {**base, "state": "current", "current": True,
                "reason": "The structure has the geometry this LDOS was computed for."}
    if spec.source.get("kind") == "relaxation":
        relaxed = relax_record(project, spec.source.get("run_id") or "")
        if relaxed is not None and live == relaxed.extra.get("input_geometry_digest"):
            return {**base, "state": "stale", "current": False,
                    "reason": "This LDOS is of the relaxed geometry, which has not been "
                              "applied to the structure. Apply the relaxation to make it "
                              "current."}
    change = specs.describe_geometry_change(spec.ground_state, target)
    return {**base, "state": "stale", "current": False,
            "reason": f"Stale: {change} since the LDOS was computed."}


def ldos_summary(project, run_id: str) -> Dict[str, Any]:
    rec = ldos_record(project, run_id)
    spec = ldos_spec_of(project, run_id)
    info = ldos_status(project, run_id)
    value = (rec.value if rec is not None else None) or {}
    return {
        "run_id": run_id, "status": value.get("status", "missing"),
        "formula": spec.ground_state.formula if spec else None,
        "xc": spec.ground_state.xc if spec else None,
        "window_eV": value.get("window_eV"), "states_in_window": value.get("states_in_window"),
        "source": dict(spec.source) if spec else None,
        "structure_key": spec.structure_key if spec else None,
        "spec_digest": spec.digest[:16] if spec else None,
        "state": info.get("state"), "current": info.get("current"),
    }
