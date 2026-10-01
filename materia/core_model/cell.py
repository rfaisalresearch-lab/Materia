"""Periodic cells, lattice metric and minimum-image conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


_NEIGHBOUR_OFFSETS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)


@dataclass(frozen=True)
class Cell:
    """A (possibly partially) periodic simulation cell.

    ``matrix`` rows are the lattice vectors a1, a2, a3 in angstrom, so that
    ``cartesian = fractional @ matrix``.  This is the row-vector convention
    used by ASE, VASP and CIF tooling.

    ``pbc`` marks which lattice directions are periodic.  A slab is typically
    ``(True, True, False)``.
    """

    matrix: np.ndarray
    pbc: Tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float).reshape(3, 3)
        object.__setattr__(self, "matrix", m)
        object.__setattr__(self, "pbc", tuple(bool(p) for p in self.pbc))

    @staticmethod
    def orthorhombic(a: float, b: float, c: float, pbc=(True, True, True)) -> "Cell":
        return Cell(np.diag([a, b, c]).astype(float), pbc)

    @staticmethod
    def cubic(a: float, pbc=(True, True, True)) -> "Cell":
        return Cell.orthorhombic(a, a, a, pbc)

    @staticmethod
    def from_parameters(
        a: float, b: float, c: float, alpha: float, beta: float, gamma: float, pbc=(True, True, True)
    ) -> "Cell":
        """Build from lengths (A) and angles (degrees), standard crystallographic setting."""
        al, be, ga = np.radians([alpha, beta, gamma])
        va = np.array([a, 0.0, 0.0])
        vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
        cx = c * np.cos(be)
        cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
        cz2 = c * c - cx * cx - cy * cy
        if cz2 <= 0:
            raise ValueError("Cell parameters do not describe a valid right-handed cell")
        vc = np.array([cx, cy, np.sqrt(cz2)])
        return Cell(np.array([va, vb, vc]), pbc)

    @staticmethod
    def none() -> "Cell":
        """A non-periodic (cluster/molecule) cell."""
        return Cell(np.zeros((3, 3)), (False, False, False))

    @property
    def is_periodic(self) -> bool:
        return any(self.pbc)

    @property
    def lengths(self) -> np.ndarray:
        return np.linalg.norm(self.matrix, axis=1)

    @property
    def angles_deg(self) -> np.ndarray:
        m = self.matrix
        L = self.lengths
        with np.errstate(invalid="ignore", divide="ignore"):
            alpha = np.degrees(np.arccos(np.clip(m[1] @ m[2] / (L[1] * L[2]), -1, 1)))
            beta = np.degrees(np.arccos(np.clip(m[0] @ m[2] / (L[0] * L[2]), -1, 1)))
            gamma = np.degrees(np.arccos(np.clip(m[0] @ m[1] / (L[0] * L[1]), -1, 1)))
        return np.array([alpha, beta, gamma])

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.matrix)))

    @property
    def reciprocal(self) -> np.ndarray:
        """Reciprocal lattice vectors (rows), including the 2*pi factor, in 1/A."""
        return 2.0 * np.pi * np.linalg.inv(self.matrix).T

    def to_fractional(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(positions, dtype=float) @ np.linalg.inv(self.matrix)

    def to_cartesian(self, fractional: np.ndarray) -> np.ndarray:
        return np.asarray(fractional, dtype=float) @ self.matrix

    def wrap(self, positions: np.ndarray) -> np.ndarray:
        """Wrap positions into the cell along periodic directions only."""
        if not self.is_periodic:
            return np.array(positions, dtype=float)
        frac = self.to_fractional(positions)
        for i, periodic in enumerate(self.pbc):
            if periodic:
                frac[:, i] = frac[:, i] - np.floor(frac[:, i])
        return self.to_cartesian(frac)

    def minimum_image(self, dr: np.ndarray) -> np.ndarray:
        """Shortest periodic image of each displacement vector.

        Rounding the fractional coordinates alone is not sufficient for
        non-orthogonal cells: for a 60-degree hexagonal surface cell it can
        return an image one lattice vector away from the true minimum. The
        rounded result is therefore used as a starting point and the
        neighbouring images are searched, which is exact for any cell whose
        basis is Minkowski-reduced and correct in practice for the cells this
        program builds.

        :func:`materia.physics.neighbors.neighbor_list` does not rely on this
        function; it expands images explicitly and is exact for any cell.
        """
        if not self.is_periodic:
            return np.asarray(dr, dtype=float)
        dr = np.atleast_2d(np.asarray(dr, dtype=float))
        inverse = np.linalg.inv(self.matrix)
        frac = dr @ inverse
        for axis, periodic in enumerate(self.pbc):
            if periodic:
                frac[:, axis] -= np.round(frac[:, axis])

        offsets = [o for o in _NEIGHBOUR_OFFSETS
                   if all(self.pbc[a] or o[a] == 0 for a in range(3))]
        best = None
        best_length = None
        for offset in offsets:
            candidate = (frac + offset) @ self.matrix
            length = np.einsum("ij,ij->i", candidate, candidate)
            if best is None:
                best, best_length = candidate, length
            else:
                closer = length < best_length
                best = np.where(closer[:, None], candidate, best)
                best_length = np.where(closer, length, best_length)
        return best

    def repeat(self, nx: int, ny: int, nz: int) -> "Cell":
        m = self.matrix.copy()
        m[0] *= nx
        m[1] *= ny
        m[2] *= nz
        return Cell(m, self.pbc)

    def as_dict(self) -> dict:
        return {"matrix": self.matrix.tolist(), "pbc": list(self.pbc), "unit": "A"}

    @staticmethod
    def from_dict(d: dict) -> "Cell":
        return Cell(np.array(d["matrix"], dtype=float), tuple(bool(p) for p in d["pbc"]))

    def __repr__(self) -> str:
        L = self.lengths
        A = self.angles_deg
        return (
            f"Cell(a={L[0]:.4f} b={L[1]:.4f} c={L[2]:.4f} A, "
            f"alpha={A[0]:.2f} beta={A[1]:.2f} gamma={A[2]:.2f} deg, pbc={self.pbc})"
        )
