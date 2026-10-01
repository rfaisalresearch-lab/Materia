"""Physical boundary classification for a ground-state calculation.

The boundary class follows from which lattice directions are periodic:

========  ======================  ============================================
class     periodic directions     what the cell means
========  ======================  ============================================
cluster   none                    an isolated molecule or cluster in a box
wire      one                     a one-dimensional system along that axis
slab      two                     a surface slab with vacuum along the third
bulk      three                   an infinite crystal
========  ======================  ============================================

Along an open direction GPAW's real-space grid imposes a zero boundary
condition on the wavefunctions and the Hartree potential at the cell faces, so
the atoms must sit inside the box with vacuum on both sides.  The vacuum is
measured as the distance from the outermost nucleus to each face, along the
face normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

BOUNDARIES: Tuple[str, ...] = ("cluster", "wire", "slab", "bulk")

BOUNDARY_NOTES = {
    "cluster": "Isolated system: open boundaries in every direction, real-space grid, "
               "Gamma point only.",
    "wire": "Periodic along one axis, open in the other two.",
    "slab": "Periodic in the surface plane, open along the surface normal.",
    "bulk": "Periodic in all three directions.",
}

MIN_VACUUM_A = 2.5

ADVISED_VACUUM_A = 4.0

AXES = "abc"


@dataclass(frozen=True)
class BoundaryReport:
    """What the cell and the atom positions say about the boundary."""

    boundary: str
    pbc: Tuple[bool, bool, bool]
    periodic_axes: Tuple[int, ...]
    open_axes: Tuple[int, ...]
    vacuum_axis: Optional[int]
    vacuum_below_A: Tuple[Optional[float], ...]
    vacuum_above_A: Tuple[Optional[float], ...]
    open_axis_orthogonal: bool
    problems: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def min_vacuum_A(self) -> Optional[float]:
        values = [v for pair in zip(self.vacuum_below_A, self.vacuum_above_A)
                  for v in pair if v is not None]
        return min(values) if values else None

    def as_dict(self) -> dict:
        return {
            "boundary": self.boundary, "pbc": list(self.pbc),
            "periodic_axes": [AXES[i] for i in self.periodic_axes],
            "open_axes": [AXES[i] for i in self.open_axes],
            "vacuum_axis": None if self.vacuum_axis is None else AXES[self.vacuum_axis],
            "vacuum_below_A": list(self.vacuum_below_A),
            "vacuum_above_A": list(self.vacuum_above_A),
            "min_vacuum_A": self.min_vacuum_A(),
            "open_axis_orthogonal": self.open_axis_orthogonal,
            "problems": list(self.problems), "warnings": list(self.warnings),
            "note": BOUNDARY_NOTES[self.boundary],
        }


def classify(pbc: Sequence[bool]) -> str:
    """Boundary class from the periodic flags alone."""
    count = sum(bool(p) for p in pbc)
    return {0: "cluster", 1: "wire", 2: "slab", 3: "bulk"}[count]


def _face_heights(cell: np.ndarray) -> np.ndarray:
    """Perpendicular height of the cell along each lattice direction."""
    volume = abs(float(np.linalg.det(cell)))
    heights = np.zeros(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        area = float(np.linalg.norm(np.cross(cell[j], cell[k])))
        heights[i] = volume / area if area > 0 else 0.0
    return heights


def _orthogonal(cell: np.ndarray, axis: int) -> bool:
    """Whether lattice vector ``axis`` is perpendicular to the other two."""
    vector = cell[axis] / max(np.linalg.norm(cell[axis]), 1e-300)
    for other in range(3):
        if other == axis:
            continue
        direction = cell[other] / max(np.linalg.norm(cell[other]), 1e-300)
        if abs(float(vector @ direction)) > 1e-8:
            return False
    return True


def inspect(positions: np.ndarray, cell: np.ndarray, pbc: Sequence[bool]) -> BoundaryReport:
    """Classify the boundary and measure the vacuum along every open axis."""
    pbc = tuple(bool(p) for p in pbc)
    cell = np.asarray(cell, dtype=float)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    boundary = classify(pbc)
    periodic = tuple(i for i in range(3) if pbc[i])
    open_axes = tuple(i for i in range(3) if not pbc[i])
    problems: List[str] = []
    warnings: List[str] = []
    below: List[Optional[float]] = [None, None, None]
    above: List[Optional[float]] = [None, None, None]

    volume = abs(float(np.linalg.det(cell)))
    if not np.isfinite(volume) or volume < 1e-6:
        problems.append("The cell has zero volume, so it defines no grid for GPAW.")
        return BoundaryReport(boundary, pbc, periodic, open_axes,
                              open_axes[0] if boundary == "slab" else None,
                              tuple(below), tuple(above), False, tuple(problems), ())

    heights = _face_heights(cell)
    fractional = positions @ np.linalg.inv(cell)
    for axis in open_axes:
        if len(positions) == 0:
            continue
        low = float(fractional[:, axis].min())
        high = float(fractional[:, axis].max())
        below[axis] = low * heights[axis]
        above[axis] = (1.0 - high) * heights[axis]
        if low < 0.0 or high > 1.0:
            problems.append(
                f"Atoms lie outside the cell along the open axis {AXES[axis]} "
                f"(fractional coordinates {low:.3f} to {high:.3f}). Along an open "
                "direction the cell is the whole universe: GPAW's grid ends at the "
                "faces and cannot place charge beyond them. Move the atoms inside, "
                "for example by centring them along that axis.")
            continue
        gap = min(below[axis], above[axis])
        if gap < MIN_VACUUM_A - 1e-9:
            problems.append(
                f"Only {gap:.2f} A of vacuum separates the outermost atom from a cell "
                f"face along the open axis {AXES[axis]}. The zero boundary condition "
                f"there cuts into the electron density; at least {MIN_VACUUM_A:g} A "
                f"is required and {ADVISED_VACUUM_A:g} A or more is advised on each side.")
        elif gap < ADVISED_VACUUM_A - 1e-9:
            warnings.append(
                f"{gap:.2f} A of vacuum along {AXES[axis]} is below the advised "
                f"{ADVISED_VACUUM_A:g} A per side; the result may still depend on the "
                "box size. A vacuum convergence study measures by how much.")

    vacuum_axis = open_axes[0] if boundary == "slab" else None
    orthogonal = True
    if boundary == "slab":
        orthogonal = _orthogonal(cell, vacuum_axis)
        if not orthogonal:
            problems.append(
                f"The open axis {AXES[vacuum_axis]} is not perpendicular to the surface "
                "plane. A slab needs its vacuum direction along the surface normal so "
                "that in-plane periodicity and the vacuum are separable and a dipole "
                "correction can be applied.")
    return BoundaryReport(boundary, pbc, periodic, open_axes, vacuum_axis,
                          tuple(below), tuple(above), orthogonal,
                          tuple(problems), tuple(warnings))


def field_is_physical(field_V_per_A: Sequence[float], cell: np.ndarray,
                      pbc: Sequence[bool]) -> Tuple[bool, str]:
    """Whether a uniform field can be applied to this cell.

    A uniform field is the gradient of a linear potential.  Along a periodic
    direction a linear potential cannot be periodic, so GPAW would build a
    sawtooth with a discontinuity inside the material; along an open direction
    it is exact.  The field must therefore be perpendicular to every periodic
    lattice vector.
    """
    vector = np.asarray(field_V_per_A, dtype=float)
    if not np.any(vector):
        return True, ""
    cell = np.asarray(cell, dtype=float)
    offending = []
    for axis in range(3):
        if not pbc[axis]:
            continue
        direction = cell[axis] / max(np.linalg.norm(cell[axis]), 1e-300)
        if abs(float(vector @ direction)) > 1e-9 * max(1.0, float(np.linalg.norm(vector))):
            offending.append(AXES[axis])
    if offending:
        return False, (
            f"The field has a component along the periodic axis "
            f"{', '.join(offending)}. A uniform field along a periodic direction is "
            "the gradient of a potential that cannot be periodic; GPAW would build a "
            "sawtooth with an unphysical discontinuity inside the material. Apply the "
            "field only along open directions: any direction for a cluster, the "
            "surface normal for a slab, perpendicular to the axis for a wire. A bulk "
            "crystal cannot take a uniform field in this formulation.")
    return True, ""


def recentre_open_axes(positions: np.ndarray, cell: np.ndarray, pbc: Sequence[bool],
                       vacuum_A: float) -> Tuple[np.ndarray, np.ndarray]:
    """Positions and cell with ``vacuum_A`` on each side of every open axis.

    Used by the vacuum convergence study.  Only open axes change: an open
    lattice vector must be perpendicular to the others (checked by the
    caller), so rescaling it does not move the atoms in the periodic plane.
    """
    positions = np.asarray(positions, dtype=float).copy()
    cell = np.asarray(cell, dtype=float).copy()
    for axis in range(3):
        if pbc[axis]:
            continue
        unit = cell[axis] / np.linalg.norm(cell[axis])
        heights = positions @ unit
        low, high = float(heights.min()), float(heights.max())
        length = (high - low) + 2.0 * vacuum_A
        positions += np.outer(np.full(len(positions), vacuum_A - low), unit)
        cell[axis] = unit * length
    return positions, cell
