"""Boundary-aware comparison of two atomistic structures by stable atom id."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..project_format.arrays import StoredArray
from ..provenance import Fidelity, Origin, Provenance, Result, digest
from .bonds import perceive_bonds


class StructureComparisonError(ValueError):
    """Structures cannot be compared under the requested convention."""


@dataclass
class StructureComparison:
    atom_ids: np.ndarray
    displacements_A: np.ndarray
    displacement_magnitudes_A: np.ndarray
    added_atom_ids: List[int]
    removed_atom_ids: List[int]
    element_changes: List[dict]
    bonds_formed: List[dict]
    bonds_broken: List[dict]
    deformation_gradient: Optional[np.ndarray]
    green_lagrange_strain: Optional[np.ndarray]
    metrics: Dict[str, Any]
    provenance: Provenance

    def results(self) -> Dict[str, Result]:
        extra = {
            "atom_ids": self.atom_ids.tolist(),
            "added_atom_ids": list(self.added_atom_ids),
            "removed_atom_ids": list(self.removed_atom_ids),
            "element_changes": list(self.element_changes),
            "bonds_formed": list(self.bonds_formed),
            "bonds_broken": list(self.bonds_broken),
            **self.metrics,
        }
        results = {
            "displacements": Result(
                "structure_displacements", self.displacements_A.copy(), "A",
                self.provenance, extra=extra),
            "rms_displacement": Result(
                "rms_displacement", self.metrics["rms_displacement_A"], "A",
                self.provenance, extra=extra),
            "max_displacement": Result(
                "max_displacement", self.metrics["max_displacement_A"], "A",
                self.provenance, extra=extra),
        }
        if self.deformation_gradient is not None:
            results["deformation_gradient"] = Result(
                "deformation_gradient", self.deformation_gradient.copy(), "dimensionless",
                self.provenance, extra=extra)
            results["green_lagrange_strain"] = Result(
                "green_lagrange_strain", self.green_lagrange_strain.copy(),
                "dimensionless", self.provenance, extra=extra)
        return results

    def stored_arrays(self) -> Dict[str, StoredArray]:
        meta = {
            "model": self.provenance.model,
            "inputs_digest": self.provenance.inputs_digest,
            "atom_ids": self.atom_ids.tolist(),
            "comparison_mode": self.metrics["mode"],
        }
        arrays = {
            "atom_ids": StoredArray(
                self.atom_ids, "stable atom id", "Matched atom ids", "table", meta),
            "displacements": StoredArray(
                self.displacements_A, "A", "Boundary-aware displacement vectors",
                "vectors", meta),
            "displacement_magnitudes": StoredArray(
                self.displacement_magnitudes_A, "A",
                "Boundary-aware displacement magnitudes", "table", meta),
        }
        if self.deformation_gradient is not None:
            arrays["deformation_gradient"] = StoredArray(
                self.deformation_gradient, "dimensionless",
                "Cell deformation gradient from before to after", "tensor", meta)
            arrays["green_lagrange_strain"] = StoredArray(
                self.green_lagrange_strain, "dimensionless",
                "Finite Green-Lagrange cell strain", "tensor", meta)
        return arrays


def _matched(before: Structure, after: Structure) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = {int(atom_id): index for index, atom_id in enumerate(before.ids)}
    second = {int(atom_id): index for index, atom_id in enumerate(after.ids)}
    common = np.asarray(sorted(first.keys() & second.keys()), dtype=np.int64)
    if not len(common):
        raise StructureComparisonError("The structures have no stable atom ids in common.")
    return common, np.asarray([first[int(value)] for value in common]), np.asarray(
        [second[int(value)] for value in common])


def _rigid_aligned_displacements(first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, dict]:
    first_center = np.mean(first, axis=0)
    second_center = np.mean(second, axis=0)
    x = first - first_center
    y = second - second_center
    covariance = x.T @ y
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    aligned = y @ rotation.T
    return aligned - x, {
        "rotation_after_to_before": rotation.T.tolist(),
        "before_centroid_A": first_center.tolist(),
        "after_centroid_A": second_center.tolist(),
    }


def _fractional_displacements(before: Structure, after: Structure,
                              first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if before.cell.pbc != after.cell.pbc:
        raise StructureComparisonError(
            "Fractional comparison requires identical periodic-boundary flags.")
    if abs(np.linalg.det(before.cell.matrix)) < 1e-12 or abs(
            np.linalg.det(after.cell.matrix)) < 1e-12:
        raise StructureComparisonError("Fractional comparison requires invertible cells.")
    first_fractional = before.cell.to_fractional(first)
    second_fractional = after.cell.to_fractional(second)
    difference = second_fractional - first_fractional
    for axis, periodic in enumerate(before.cell.pbc):
        if periodic:
            difference[:, axis] -= np.round(difference[:, axis])
    return difference @ before.cell.matrix


def _bond_changes(before: Structure, after: Structure) -> Tuple[List[dict], List[dict]]:
    first = {bond.key(): bond for bond in perceive_bonds(before)}
    second = {bond.key(): bond for bond in perceive_bonds(after)}
    formed = [second[key].as_dict() for key in sorted(second.keys() - first.keys())]
    broken = [first[key].as_dict() for key in sorted(first.keys() - second.keys())]
    return formed, broken


def compare_structures(before: Structure, after: Structure, *, mode: str = "auto",
                       compare_bonds: bool = True,
                       remove_translation: bool = False) -> StructureComparison:
    """Compare two structures without changing either input."""
    if mode not in ("auto", "cartesian", "minimum-image", "fractional", "rigid-align"):
        raise StructureComparisonError(
            "mode must be auto, cartesian, minimum-image, fractional, or rigid-align.")
    common, first_indices, second_indices = _matched(before, after)
    first = np.asarray(before.positions[first_indices], dtype=float)
    second = np.asarray(after.positions[second_indices], dtype=float)
    chosen = mode
    if chosen == "auto":
        if any(before.cell.pbc) and before.cell.pbc == after.cell.pbc:
            chosen = "fractional"
        elif not any(before.cell.pbc) and not any(after.cell.pbc):
            chosen = "rigid-align"
        else:
            chosen = "cartesian"
    alignment = {}
    if chosen == "cartesian":
        displacements = second - first
    elif chosen == "minimum-image":
        if before.cell.pbc != after.cell.pbc or not np.allclose(
                before.cell.matrix, after.cell.matrix, rtol=1e-10, atol=1e-12):
            raise StructureComparisonError(
                "Minimum-image comparison requires identical cells and boundary flags.")
        displacements = before.cell.minimum_image(second - first)
    elif chosen == "fractional":
        displacements = _fractional_displacements(before, after, first, second)
    else:
        if any(before.cell.pbc) or any(after.cell.pbc):
            raise StructureComparisonError(
                "Rigid alignment is defined here only for non-periodic structures.")
        displacements, alignment = _rigid_aligned_displacements(first, second)
    if remove_translation and len(displacements):
        translation = np.mean(displacements, axis=0)
        displacements = displacements - translation
        alignment["removed_translation_A"] = translation.tolist()
    magnitudes = np.linalg.norm(displacements, axis=1)

    first_ids = {int(value) for value in before.ids}
    second_ids = {int(value) for value in after.ids}
    added = sorted(second_ids - first_ids)
    removed = sorted(first_ids - second_ids)
    element_changes = []
    for atom_id, first_index, second_index in zip(common, first_indices, second_indices):
        before_z = int(before.numbers[first_index])
        after_z = int(after.numbers[second_index])
        if before_z != after_z:
            element_changes.append({
                "atom_id": int(atom_id),
                "before_atomic_number": before_z,
                "after_atomic_number": after_z,
            })
    formed, broken = _bond_changes(before, after) if compare_bonds else ([], [])

    deformation = None
    strain = None
    before_volume = before.cell.volume
    after_volume = after.cell.volume
    if before_volume > 1e-12 and after_volume > 1e-12:
        deformation = after.cell.matrix.T @ np.linalg.inv(before.cell.matrix.T)
        strain = 0.5 * (deformation.T @ deformation - np.eye(3))
    volume_change = after_volume - before_volume
    metrics = {
        "mode": chosen,
        "matched_atoms": int(len(common)),
        "before_atoms": len(before),
        "after_atoms": len(after),
        "rms_displacement_A": float(np.sqrt(np.mean(magnitudes * magnitudes))),
        "mean_displacement_A": float(np.mean(magnitudes)),
        "max_displacement_A": float(np.max(magnitudes)),
        "before_formula": before.formula(),
        "after_formula": after.formula(),
        "before_volume_A3": before_volume,
        "after_volume_A3": after_volume,
        "volume_change_A3": volume_change,
        "relative_volume_change": (volume_change / before_volume
                                   if before_volume > 1e-12 else None),
        "formed_bond_count": len(formed),
        "broken_bond_count": len(broken),
        **alignment,
    }
    parameters = {
        "mode": chosen,
        "compare_bonds": bool(compare_bonds),
        "remove_translation": bool(remove_translation),
    }
    approximations = []
    if compare_bonds:
        approximations.append(
            "Bond changes use Materia's covalent-radius and Voronoi perception heuristic, "
            "not an electronic bond-order calculation.")
    if chosen == "fractional":
        approximations.append(
            "Fractional displacements separate cell deformation from atomic motion by "
            "mapping both structures into the reference cell.")
    provenance = Provenance(
        model="structural-comparison/stable-id",
        fidelity=Fidelity.TIER0_STRUCTURAL,
        origin=Origin.CALCULATED,
        parameters=parameters,
        approximations=approximations,
        boundary_conditions=(f"before pbc={before.cell.pbc}; after pbc={after.cell.pbc}"),
        inputs_digest=digest({
            "before": before.as_dict(),
            "after": after.as_dict(),
            "parameters": parameters,
        }),
    )
    if not math.isfinite(metrics["rms_displacement_A"]):
        raise StructureComparisonError("The comparison produced a non-finite displacement.")
    return StructureComparison(
        common, displacements, magnitudes, added, removed, element_changes,
        formed, broken, deformation, strain, metrics, provenance)
