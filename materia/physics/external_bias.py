"""External mechanical forces and harmonic positional restraints.

Biases are stored on a structure by stable atom id. Constant forces contribute
``-F dot (r - r0)`` to the potential energy and ``F`` to the atomic force.
Harmonic restraints contribute ``k |r - r0|^2 / 2`` and ``-k (r - r0)``.
Coordinates are unwrapped Cartesian coordinates in angstrom. The energy zero of
a constant force is its recorded reference position and is therefore arbitrary.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure, StructureError
from .potentials import Potential

BIAS_KEY = "external_biases"
BIAS_VERSION = 1


class ExternalBiasError(ValueError):
    """An invalid or stale external-bias definition."""


def _vector(value: Sequence[float], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ExternalBiasError(f"{name} must contain three finite numbers.") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ExternalBiasError(f"{name} must contain three finite numbers.")
    return array


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ExternalBiasError(f"{name} must be a positive finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExternalBiasError(f"{name} must be a positive finite number.") from exc
    if not np.isfinite(number) or number <= 0:
        raise ExternalBiasError(f"{name} must be a positive finite number.")
    return number


def _state(structure: Structure, create: bool = False) -> Dict[str, Any]:
    value = structure.info.get(BIAS_KEY)
    if value is None:
        if not create:
            return {"version": BIAS_VERSION, "constant_forces": {}, "restraints": {}}
        value = {"version": BIAS_VERSION, "constant_forces": {}, "restraints": {}}
        structure.info[BIAS_KEY] = value
    if not isinstance(value, dict) or value.get("version") != BIAS_VERSION:
        raise ExternalBiasError("The structure contains an unsupported external-bias record.")
    for field in ("constant_forces", "restraints"):
        if not isinstance(value.get(field), dict):
            raise ExternalBiasError(f"The external-bias field {field!r} is invalid.")
    return value


def _atom_index(structure: Structure, atom_id: int) -> int:
    if isinstance(atom_id, bool):
        raise ExternalBiasError("atom_id must identify an atom in the structure.")
    try:
        return structure.index_of(int(atom_id))
    except (StructureError, TypeError, ValueError) as exc:
        raise ExternalBiasError(f"No atom with id {atom_id!r} exists in the structure.") from exc


def set_constant_force(
    structure: Structure,
    atom_id: int,
    force_eV_A: Sequence[float],
    reference_position_A: Optional[Sequence[float]] = None,
) -> dict:
    """Set or replace a constant Cartesian force on one atom."""
    index = _atom_index(structure, atom_id)
    force = _vector(force_eV_A, "force_eV_A")
    if float(np.linalg.norm(force)) == 0.0:
        raise ExternalBiasError("force_eV_A must not be the zero vector.")
    reference = (structure.positions[index].copy() if reference_position_A is None
                 else _vector(reference_position_A, "reference_position_A"))
    record = {
        "atom_id": int(atom_id),
        "force_eV_A": force.tolist(),
        "reference_position_A": reference.tolist(),
    }
    _state(structure, True)["constant_forces"][str(int(atom_id))] = record
    return copy.deepcopy(record)


def set_harmonic_restraint(
    structure: Structure,
    atom_id: int,
    spring_eV_A2: float,
    target_position_A: Optional[Sequence[float]] = None,
) -> dict:
    """Set or replace an isotropic Cartesian harmonic restraint on one atom."""
    index = _atom_index(structure, atom_id)
    spring = _positive(spring_eV_A2, "spring_eV_A2")
    target = (structure.positions[index].copy() if target_position_A is None
              else _vector(target_position_A, "target_position_A"))
    record = {
        "atom_id": int(atom_id),
        "spring_eV_A2": spring,
        "target_position_A": target.tolist(),
    }
    _state(structure, True)["restraints"][str(int(atom_id))] = record
    return copy.deepcopy(record)


def clear_external_biases(
    structure: Structure,
    atom_ids: Optional[Iterable[int]] = None,
    kind: Optional[str] = None,
) -> int:
    """Remove selected biases and return the number removed."""
    if kind not in (None, "constant_force", "restraint"):
        raise ExternalBiasError("kind must be 'constant_force', 'restraint', or None.")
    state = _state(structure)
    selected = None if atom_ids is None else {str(int(atom_id)) for atom_id in atom_ids}
    fields = []
    if kind in (None, "constant_force"):
        fields.append("constant_forces")
    if kind in (None, "restraint"):
        fields.append("restraints")
    removed = 0
    for field in fields:
        keys = list(state[field])
        for key in keys:
            if selected is None or key in selected:
                del state[field][key]
                removed += 1
    if removed and not state["constant_forces"] and not state["restraints"]:
        structure.info.pop(BIAS_KEY, None)
    return removed


def external_biases(structure: Structure) -> dict:
    """Return a detached, validated description of every active bias."""
    state = _state(structure)
    energy_and_forces(structure)
    return copy.deepcopy(state)


def has_external_bias(structure: Structure) -> bool:
    state = _state(structure)
    return bool(state["constant_forces"] or state["restraints"])


def energy_and_forces(structure: Structure) -> Tuple[float, np.ndarray]:
    """Evaluate only the external-bias energy and forces."""
    state = _state(structure)
    energy = 0.0
    forces = np.zeros_like(structure.positions, dtype=float)
    for key, record in state["constant_forces"].items():
        if not isinstance(record, dict) or str(record.get("atom_id")) != key:
            raise ExternalBiasError(f"Constant-force record {key!r} is invalid.")
        index = _atom_index(structure, record["atom_id"])
        force = _vector(record.get("force_eV_A"), "force_eV_A")
        reference = _vector(record.get("reference_position_A"), "reference_position_A")
        displacement = structure.positions[index] - reference
        energy -= float(np.dot(force, displacement))
        forces[index] += force
    for key, record in state["restraints"].items():
        if not isinstance(record, dict) or str(record.get("atom_id")) != key:
            raise ExternalBiasError(f"Harmonic-restraint record {key!r} is invalid.")
        index = _atom_index(structure, record["atom_id"])
        spring = _positive(record.get("spring_eV_A2"), "spring_eV_A2")
        target = _vector(record.get("target_position_A"), "target_position_A")
        displacement = structure.positions[index] - target
        energy += 0.5 * spring * float(np.dot(displacement, displacement))
        forces[index] -= spring * displacement
    return energy, forces


class ExternallyBiasedPotential(Potential):
    """Compose any Materia potential with biases stored on its input structure."""

    def __init__(self, base: Potential) -> None:
        self.base = base
        self.name = f"{base.name}+external-bias"
        self.fidelity = base.fidelity
        self.cutoff_A = base.cutoff_A

    def model_label(self) -> str:
        return f"{self.base.model_label()}+external-bias"

    def composition(self, structure: Structure):
        return self.base.composition(structure)

    def supports(self, structure: Structure):
        ok, reason = self.base.supports(structure)
        if not ok:
            return ok, reason
        try:
            energy_and_forces(structure)
        except ExternalBiasError as exc:
            return False, str(exc)
        return True, ""

    def energy_and_forces(self, structure: Structure):
        energy, forces = self.base.energy_and_forces(structure)
        bias_energy, bias_forces = energy_and_forces(structure)
        return float(energy) + bias_energy, np.asarray(forces, dtype=float) + bias_forces

    def diagnostics(self, structure: Structure) -> dict:
        diagnostics = dict(self.base.diagnostics(structure))
        state = _state(structure)
        diagnostics["external_bias"] = {
            "constant_force_count": len(state["constant_forces"]),
            "restraint_count": len(state["restraints"]),
            "coordinate_convention": "unwrapped Cartesian coordinates",
        }
        return diagnostics

    def describe(self) -> dict:
        description = dict(self.base.describe())
        parameters = dict(description.get("parameters", {}))
        parameters["external_bias_source"] = f"Structure.info[{BIAS_KEY!r}]"
        approximations = list(description.get("approximations", []))
        approximations.append(
            "External mechanical biases use unwrapped Cartesian positions; constant-force "
            "energy has an arbitrary zero at the recorded reference position."
        )
        description.update({
            "model": self.name,
            "parameters": parameters,
            "approximations": approximations,
        })
        return description


def potential_with_external_bias(base: Potential, structure: Structure) -> Potential:
    """Return the base potential or its bias composition for this structure."""
    return ExternallyBiasedPotential(base) if BIAS_KEY in structure.info else base
