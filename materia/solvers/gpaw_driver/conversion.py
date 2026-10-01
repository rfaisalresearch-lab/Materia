"""Moving a structure between Materia and the GPAW worker.

The worker runs in another interpreter, so a structure crosses as plain JSON.
Everything GPAW can act on is carried: cell vectors, periodicity, atomic
numbers, positions, which atoms are held fixed, and initial magnetic moments.
Materia's stable atom ids travel with it so that forces come back attached to
the atoms they belong to rather than to positions in a list, and the
round trip is checked in both directions by the tests.

The conversion never mutates the structure it is given.  Applying a result to a
structure is a separate, explicit step, so a run that fails or is cancelled
cannot leave a project half-updated.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ...core_model.structure import Structure
from ...core_model.cell import Cell


class ConversionError(ValueError):
    """A structure GPAW cannot be given as it stands."""


def to_worker_spec(structure: Structure) -> Dict[str, Any]:
    """The JSON form of a structure, for the worker to rebuild."""
    if len(structure) == 0:
        raise ConversionError("The structure has no atoms.")
    positions = np.asarray(structure.positions, dtype=float)
    if not np.isfinite(positions).all():
        raise ConversionError("The structure has non-finite coordinates.")
    cell = np.asarray(structure.cell.matrix, dtype=float)
    if not np.isfinite(cell).all():
        raise ConversionError("The cell has non-finite vectors.")
    return {
        "numbers": [int(z) for z in structure.numbers],
        "positions": positions.tolist(),
        "cell": cell.tolist(),
        "pbc": [bool(p) for p in structure.cell.pbc],
        "magnetic_moments": [float(m) for m in structure.magnetic_moments],
        "fixed": [bool(f) for f in structure.fixed],
        "ids": [int(i) for i in structure.ids],
        "total_charge": float(structure.total_charge()),
    }


def from_worker_spec(spec: Dict[str, Any]) -> Structure:
    """Rebuild a structure from the JSON form, for round-trip checking."""
    numbers = np.asarray(spec["numbers"], dtype=int)
    positions = np.asarray(spec["positions"], dtype=float)
    cell = Cell(np.asarray(spec["cell"], dtype=float),
                tuple(bool(p) for p in spec["pbc"]))
    structure = Structure(numbers, positions, cell)
    moments = spec.get("magnetic_moments")
    if moments is not None:
        structure.magnetic_moments[:] = np.asarray(moments, dtype=float)
    fixed = spec.get("fixed")
    if fixed is not None:
        structure.fixed[:] = np.asarray(fixed, dtype=bool)
    ids = spec.get("ids")
    if ids is not None:
        structure.ids[:] = np.asarray(ids, dtype=int)
        structure._rebuild_index()
    return structure


INPUT_FIELDS = ("numbers", "positions", "cell", "pbc", "magnetic_moments",
                "fixed", "ids", "total_charge")

_FIELD_NAMES = {
    "numbers": "atomic numbers",
    "positions": "atom positions",
    "cell": "cell vectors",
    "pbc": "periodicity",
    "magnetic_moments": "initial magnetic moments",
    "fixed": "fixed flags",
    "ids": "atom ids or ordering",
    "total_charge": "total formal charge",
}


def input_digest(spec: Dict[str, Any], settings: Dict[str, Any]) -> str:
    """A deterministic fingerprint of exactly what GPAW was asked to compute.

    Covers the whole worker specification and the validated configuration, so
    a result can be checked against the structure it was computed for rather
    than trusted because the atom ids still match.
    """
    from ...provenance import digest

    payload = {
        "structure": {field: spec.get(field) for field in INPUT_FIELDS},
        "settings": {key: settings[key] for key in sorted(settings)},
    }
    return digest(payload)


def describe_difference(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """What changed between two worker specifications, in plain words.

    Returns an empty string when they describe the same input.
    """
    if len(before.get("numbers", [])) != len(after.get("numbers", [])):
        return (f"the atom count changed from {len(before.get('numbers', []))} "
                f"to {len(after.get('numbers', []))}")
    changed = []
    for field in INPUT_FIELDS:
        left = np.asarray(before.get(field), dtype=float)
        right = np.asarray(after.get(field), dtype=float)
        if left.shape != right.shape:
            changed.append(_FIELD_NAMES[field])
        elif not np.allclose(left, right, rtol=0.0, atol=1e-10):
            changed.append(_FIELD_NAMES[field])
    if not changed:
        return ""
    if len(changed) == 1:
        return f"the {changed[0]} changed"
    return "the " + ", ".join(changed[:-1]) + f" and {changed[-1]} changed"


def forces_by_id(spec: Dict[str, Any], forces: List[List[float]]) -> Dict[int, List[float]]:
    """Attach returned forces to the atom ids they were computed for."""
    ids = [int(i) for i in spec["ids"]]
    if len(forces) != len(ids):
        raise ConversionError(
            f"GPAW returned {len(forces)} force vectors for {len(ids)} atoms.")
    return {atom_id: [float(v) for v in row] for atom_id, row in zip(ids, forces)}


def apply_forces(structure: Structure, by_id: Dict[int, List[float]]) -> int:
    """Write forces onto the atoms they belong to, matching by id.

    Returns the number of atoms updated.  Raises before touching anything if
    any id is missing, so a partial write is impossible.
    """
    missing = [atom_id for atom_id in by_id if not structure.has_id(int(atom_id))]
    if missing:
        raise ConversionError(
            f"The structure no longer holds atom id(s) {missing[:8]}, so these forces "
            "cannot be attached. Re-run the calculation on the current structure.")
    updated = np.array(structure.forces, dtype=float, copy=True)
    for atom_id, vector in by_id.items():
        updated[structure.index_of(int(atom_id))] = vector
    structure.forces = updated
    return len(by_id)
