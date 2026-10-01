"""Preparation and analysis of classical atomistic collision experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core_model import Cell, Structure
from ..core_model.units import U_A2_FS2_TO_EV
from ..elements import periodic_table as pt
from ..physics.neighbors import neighbor_list


class CollisionError(ValueError):
    pass


def _direction(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or length <= 0:
        raise CollisionError("The collision direction must be a finite non-zero vector.")
    return vector / length


def _merge(a: Structure, b: Structure) -> Structure:
    numbers = np.concatenate([a.numbers, b.numbers])
    positions = np.vstack([a.positions, b.positions])
    roles = ["projectile"] * len(a) + ["target"] * len(b)
    result = Structure(numbers, positions, roles=roles)
    result.mass_numbers[:] = np.concatenate([a.mass_numbers, b.mass_numbers])
    result.formal_charges[:] = np.concatenate([a.formal_charges, b.formal_charges])
    result.partial_charges[:] = np.concatenate([a.partial_charges, b.partial_charges])
    result.magnetic_moments[:] = np.concatenate(
        [a.magnetic_moments, b.magnetic_moments])
    result.fixed[:] = np.concatenate([a.fixed, b.fixed])
    return result


def prepare(
    projectile: Structure,
    target: Structure,
    relative_speed_A_fs: float,
    direction: Sequence[float] = (1.0, 0.0, 0.0),
    impact_parameter_A: float = 0.0,
    gap_A: float = 4.0,
    padding_A: float = 8.0,
) -> Structure:
    """Place two structures on a collision course with zero total momentum."""
    if len(projectile) == 0 or len(target) == 0:
        raise CollisionError("A collision needs at least one projectile and one target atom.")
    speed = float(relative_speed_A_fs)
    impact = float(impact_parameter_A)
    gap = float(gap_A)
    padding = float(padding_A)
    if not np.isfinite([speed, impact, gap, padding]).all():
        raise CollisionError("Collision variables must be finite numbers.")
    if speed <= 0 or gap < 0 or padding < 2.5:
        raise CollisionError(
            "relative_speed_A_fs must be positive, gap_A non-negative and padding_A "
            "at least 2.5 A.")
    axis = _direction(direction)
    trial = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    transverse = np.cross(axis, trial)
    transverse /= np.linalg.norm(transverse)

    a = projectile.copy()
    b = target.copy()
    a.positions -= a.positions.mean(axis=0)
    b.positions -= b.positions.mean(axis=0)
    a_projection = a.positions @ axis
    b_projection = b.positions @ axis
    a_shift = -0.5 * gap - float(a_projection.max())
    b_shift = 0.5 * gap - float(b_projection.min())
    a.positions += a_shift * axis + 0.5 * impact * transverse
    b.positions += b_shift * axis - 0.5 * impact * transverse

    mass_a = float(a.masses().sum())
    mass_b = float(b.masses().sum())
    total = mass_a + mass_b
    a.velocities[:] = speed * mass_b / total * axis
    b.velocities[:] = -speed * mass_a / total * axis
    result = _merge(a, b)
    result.velocities[:] = np.vstack([a.velocities, b.velocities])

    lo, hi = result.bounding_box()
    lengths = np.maximum(hi - lo + 2.0 * padding, 2.0 * padding)
    result.positions += padding - lo
    result.cell = Cell(np.diag(lengths), (False, False, False))
    reduced_mass = mass_a * mass_b / total
    kinetic_eV = 0.5 * reduced_mass * speed * speed * U_A2_FS2_TO_EV
    result.info["collision"] = {
        "model_scope": "classical nuclei under the selected interatomic potential",
        "projectile_atoms": len(a),
        "target_atoms": len(b),
        "relative_speed_A_fs": speed,
        "impact_parameter_A": impact,
        "initial_gap_A": gap,
        "direction": axis.tolist(),
        "centre_of_mass_energy_eV": kinetic_eV,
        "warning": ("This is an atomistic collision, not a Standard Model particle "
                    "collision. It cannot produce or predict quarks, hadrons or Higgs "
                    "bosons."),
    }
    return result


def fragments(structure: Structure, positions: np.ndarray,
              scale: float = 1.35) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Connected components using covalent-radius contact as a structural diagnostic."""
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise CollisionError("Fragment contact scale must be a finite positive number.")
    points = np.asarray(positions, dtype=float).reshape(len(structure), 3)
    if not np.isfinite(points).all():
        raise CollisionError("Fragment positions must all be finite.")
    parent = np.arange(len(structure), dtype=np.int32)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def join(i: int, j: int) -> None:
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[rj] = ri

    radii = np.array([pt.covalent_radius(int(z)) or 1.2 for z in structure.numbers])
    if len(structure):
        contacts = neighbor_list(
            points, structure.cell, 2.0 * scale * float(radii.max()),
            half=True, coincident="exclude")
        limits = scale * (radii[contacts.i] + radii[contacts.j])
        for i, j in zip(contacts.i[contacts.d <= limits], contacts.j[contacts.d <= limits]):
            join(int(i), int(j))

    roots = [root(i) for i in range(len(structure))]
    ordered = sorted(set(roots), key=lambda r: (-roots.count(r), r))
    mapping = {value: index for index, value in enumerate(ordered)}
    labels = np.array([mapping[value] for value in roots], dtype=np.int32)
    summary: List[Dict[str, object]] = []
    for label in range(len(ordered)):
        idx = np.flatnonzero(labels == label)
        counts: Dict[str, int] = {}
        for z in structure.numbers[idx]:
            symbol = pt.symbol(int(z))
            counts[symbol] = counts.get(symbol, 0) + 1
        summary.append({
            "id": label,
            "atoms": int(len(idx)),
            "formula": "".join(symbol + (str(count) if count > 1 else "")
                               for symbol, count in sorted(counts.items())),
            "centre_A": points[idx].mean(axis=0).tolist(),
        })
    return labels, summary
