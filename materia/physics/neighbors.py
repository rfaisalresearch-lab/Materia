"""Neighbour lists with full periodic-image handling.

Implementation
--------------
Positions are replicated into the periodic images that can reach within
``cutoff`` of the primary cell, then a k-d tree (``scipy.spatial.cKDTree``)
answers the range query.  Image replication (rather than the minimum-image
convention) keeps the list correct for cells thinner than ``2 * cutoff``,
which happens routinely for thin slabs and small unit cells.

The search runs on a copy of the positions wrapped into the cell along its
periodic directions, and each pair's integer lattice shift is corrected by the
wrap, so an atom stored any number of lattice vectors outside the cell has
exactly the neighbours, distances and displacements it has inside it. The
stored coordinates are never changed.

The returned pair list is *directed*: every neighbour relation appears once as
``(i -> j)`` and once as ``(j -> i)`` unless ``half=True``.  Displacements
``D`` point from ``i`` to ``j`` and already include the image shift, so
``positions[j] + shifts @ cell.matrix - positions[i] == D``.

A true self-pair, the same atom in the zero lattice image, is excluded by
index and image. Distinct atoms closer than :data:`COLLISION_TOLERANCE_A` are
refused, or left out when a caller asks for that explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..core_model.cell import Cell

COLLISION_TOLERANCE_A = 1e-5
"""Distance at or below which two distinct atoms are treated as coincident.

1e-5 A is 1000 fm, about ten times the diameter of the largest nuclei and
four orders of magnitude below the shortest interatomic distance that occurs
in any material, including the most strongly compressed. Two atoms closer
than this are not a configuration any interatomic model describes, and no
coordinate that arises from a real structure, a relaxation or a dynamics step
lands there by accident. The value is also far above floating-point rounding
of coordinates in any cell Materia builds, so it never mistakes rounding for
a collision.
"""


class CoincidentAtoms(ValueError):
    """Two distinct atoms, or an atom and a periodic image of another, coincide.

    ``i`` and ``j`` are array indices; ``lattice_shift`` is the integer lattice
    translation applied to ``j`` relative to the stored coordinates, all zero
    for two atoms in the same image.
    """

    def __init__(self, i: int, j: int, lattice_shift: Tuple[int, int, int],
                 distance: float, tolerance: float = COLLISION_TOLERANCE_A,
                 labels: Optional[Tuple[str, str]] = None) -> None:
        self.i, self.j = int(i), int(j)
        self.lattice_shift = tuple(int(v) for v in lattice_shift)
        self.distance = float(distance)
        self.tolerance = float(tolerance)
        a, b = labels or (f"index {self.i}", f"index {self.j}")
        where = ("" if not any(self.lattice_shift) else
                 f" in the periodic image shifted by lattice vector(s) {self.lattice_shift}")
        if self.i == self.j:
            what = (f"Atom {a} coincides with its own periodic image{where}: the cell "
                    "is too small to hold it")
        else:
            what = f"Atoms {a} and {b}{where} are {self.distance:.3g} A apart"
        super().__init__(
            f"{what}, at or below the {self.tolerance:g} A collision tolerance. Two "
            "distinct nuclei cannot occupy the same point, and no interatomic model "
            "defines an energy for them. Move or remove one of the atoms.")

    def relabel(self, labels: Tuple[str, str]) -> "CoincidentAtoms":
        return CoincidentAtoms(self.i, self.j, self.lattice_shift, self.distance,
                               self.tolerance, labels)


def wrap_for_search(positions: np.ndarray, cell: Cell) -> Tuple[np.ndarray, np.ndarray]:
    """Positions wrapped into the cell along periodic axes, and the integer wrap.

    Returns ``(wrapped, k)`` with ``positions == wrapped + k @ cell.matrix``.
    The stored coordinates are never changed; searching on the wrapped copy is
    what makes image expansion exact for atoms stored anywhere.
    """
    pos = np.asarray(positions, dtype=float)
    k = np.zeros((len(pos), 3))
    if not cell.is_periodic or len(pos) == 0:
        return pos.copy(), k
    frac = cell.to_fractional(pos)
    for axis, periodic in enumerate(cell.pbc):
        if periodic:
            k[:, axis] = np.floor(frac[:, axis])
    return pos - k @ cell.matrix, k


def image_grid(cell: Cell, cutoff: float) -> np.ndarray:
    """Integer lattice translations whose images can reach within ``cutoff``."""
    nx, ny, nz = _image_range(cell, cutoff) if cell.is_periodic else (0, 0, 0)
    return np.array([(a, b, c) for a in range(-nx, nx + 1)
                     for b in range(-ny, ny + 1) for c in range(-nz, nz + 1)], dtype=float)


def find_coincidence(positions: np.ndarray, cell: Cell,
                     tolerance: float = COLLISION_TOLERANCE_A) -> Optional[CoincidentAtoms]:
    """The first pair of distinct atoms within ``tolerance``, or ``None``.

    A true self-pair, the same atom in the zero lattice image, is excluded by
    index and image, never by distance, so two different atoms at the same
    point are always found.
    """
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if n == 0:
        return None
    wrapped, k = wrap_for_search(pos, cell)
    grid = image_grid(cell, tolerance)
    zero = int(np.nonzero(~grid.any(axis=1))[0][0])
    ghosts = (wrapped[None, :, :] + (grid @ cell.matrix if cell.is_periodic
                                     else np.zeros((1, 3)))[:, None, :]).reshape(-1, 3)
    found = cKDTree(wrapped).sparse_distance_matrix(cKDTree(ghosts), tolerance,
                                                    output_type="ndarray")
    if found.size == 0:
        return None
    i = found["i"].astype(np.int64)
    flat = found["j"].astype(np.int64)
    j = flat % n
    image = flat // n
    distinct = ~((i == j) & (image == zero))
    if not distinct.any():
        return None
    first = int(np.nonzero(distinct)[0][0])
    a, b = int(i[first]), int(j[first])
    shift = grid[image[first]] - k[b] + k[a]
    return CoincidentAtoms(a, b, tuple(shift), float(found["v"][first]), tolerance)


@dataclass
class NeighborList:
    """Directed neighbour pairs within a cutoff."""

    i: np.ndarray
    j: np.ndarray
    D: np.ndarray
    d: np.ndarray
    shifts: np.ndarray
    cutoff: float
    n_atoms: int

    def __len__(self) -> int:
        return int(self.i.shape[0])

    def counts(self) -> np.ndarray:
        """Number of neighbours per atom."""
        return np.bincount(self.i, minlength=self.n_atoms)

    def neighbors_of(self, index: int) -> np.ndarray:
        return self.j[self.i == index]

    def half(self) -> "NeighborList":
        """Keep only one direction of each pair (i < j, or tie-broken by shift)."""
        keep = (self.i < self.j) | ((self.i == self.j) & (
            (self.shifts[:, 0] > 0)
            | ((self.shifts[:, 0] == 0) & (self.shifts[:, 1] > 0))
            | ((self.shifts[:, 0] == 0) & (self.shifts[:, 1] == 0) & (self.shifts[:, 2] > 0))
        ))
        return NeighborList(self.i[keep], self.j[keep], self.D[keep], self.d[keep],
                            self.shifts[keep], self.cutoff, self.n_atoms)


def _image_range(cell: Cell, cutoff: float) -> Tuple[int, int, int]:
    """Number of lattice repetitions needed in each direction to cover cutoff."""
    reps = [0, 0, 0]
    m = cell.matrix
    for axis in range(3):
        if not cell.pbc[axis]:
            continue
        other = [m[k] for k in range(3) if k != axis]
        normal = np.cross(other[0], other[1])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            raise ValueError("Degenerate cell: lattice vectors are coplanar")
        width = abs(m[axis] @ normal) / norm
        if width < 1e-9:
            raise ValueError("Degenerate cell: zero width along a periodic axis")
        reps[axis] = int(np.ceil(cutoff / width))
    return tuple(reps)


def neighbor_list(
    positions: np.ndarray,
    cell: Cell,
    cutoff: float,
    *,
    half: bool = False,
    max_pairs: int = 60_000_000,
    coincident: str = "refuse",
) -> NeighborList:
    """Build a neighbour list.

    Parameters
    ----------
    positions
        ``(N, 3)`` cartesian coordinates in angstrom, stored anywhere: an atom
        outside the cell, by any number of lattice vectors, is found exactly
        as if it had been wrapped. The coordinates are never modified.
    cell
        Periodic cell; non-periodic directions are not replicated.
    cutoff
        Interaction cutoff in angstrom.
    half
        Return only one direction per pair.
    max_pairs
        Guard against accidental memory blow-ups.
    coincident
        What to do with two distinct atoms, or an atom and an image of
        another, within :data:`COLLISION_TOLERANCE_A`. ``"refuse"`` raises
        :class:`CoincidentAtoms`; energy models use it. ``"exclude"`` leaves
        the pair out; only geometric perception such as bond detection uses
        it, because a coincident pair is not a bond. A true self-pair, the same
        atom in the zero image, is always excluded, by index and image.

    Returned ``shifts`` are integer lattice translations relative to the
    stored coordinates, so ``positions[j] + shifts @ cell.matrix - positions[i]``
    equals ``D`` for every pair.
    """
    if coincident not in ("refuse", "exclude"):
        raise ValueError("coincident must be 'refuse' or 'exclude'")
    pos = np.ascontiguousarray(np.asarray(positions, dtype=float))
    n = pos.shape[0]
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return NeighborList(z, z, np.zeros((0, 3)), np.zeros(0),
                            np.zeros((0, 3), dtype=np.int32), cutoff, 0)
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    wrapped, wrap = wrap_for_search(pos, cell)
    shifts = image_grid(cell, cutoff).astype(np.int32)
    zero = int(np.nonzero(~shifts.any(axis=1))[0][0])
    offsets = shifts.astype(float) @ cell.matrix if cell.is_periodic else np.zeros((1, 3))
    ghosts = (wrapped[None, :, :] + offsets[:, None, :]).reshape(-1, 3)

    tree_primary = cKDTree(wrapped)
    tree_ghosts = cKDTree(ghosts)
    pairs = tree_primary.query_ball_tree(tree_ghosts, r=cutoff)

    total = sum(len(p) for p in pairs)
    if total > max_pairs:
        raise MemoryError(
            f"Neighbour list would contain {total} pairs (limit {max_pairs}). "
            "Reduce the cutoff or the region size."
        )

    i_idx = np.repeat(np.arange(n, dtype=np.int64), [len(p) for p in pairs])
    flat = np.fromiter((q for p in pairs for q in p), dtype=np.int64, count=total)
    img_idx = flat // n
    j_idx = flat % n
    keep = ~((i_idx == j_idx) & (img_idx == zero))
    i_idx, j_idx, img_idx = i_idx[keep], j_idx[keep], img_idx[keep]

    lattice = shifts[img_idx].astype(np.int64)
    if cell.is_periodic:
        lattice = lattice - wrap[j_idx].astype(np.int64) + wrap[i_idx].astype(np.int64)
        D = pos[j_idx] + lattice.astype(float) @ cell.matrix - pos[i_idx]
    else:
        D = pos[j_idx] - pos[i_idx]
    d = np.linalg.norm(D, axis=1)

    close = d <= COLLISION_TOLERANCE_A
    if close.any():
        if coincident == "refuse":
            k = int(np.nonzero(close)[0][0])
            raise CoincidentAtoms(int(i_idx[k]), int(j_idx[k]), tuple(lattice[k]), float(d[k]))
        far = ~close
        i_idx, j_idx, lattice, D, d = i_idx[far], j_idx[far], lattice[far], D[far], d[far]

    nl = NeighborList(i_idx, j_idx, D, d, lattice.astype(np.int32), float(cutoff), n)
    return nl.half() if half else nl


def coordination_numbers(nl: NeighborList) -> np.ndarray:
    return nl.counts()


def radial_distribution(
    nl: NeighborList, n_atoms: int, volume: float, r_max: float, bins: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair distribution function g(r).

    Normalised against an ideal gas of the same number density.  Only valid
    for ``r_max <= cutoff`` of the neighbour list used.
    """
    if r_max > nl.cutoff + 1e-9:
        raise ValueError(f"r_max {r_max} exceeds neighbour-list cutoff {nl.cutoff}")
    hist, edges = np.histogram(nl.d, bins=bins, range=(0.0, r_max))
    r = 0.5 * (edges[1:] + edges[:-1])
    dr = edges[1] - edges[0]
    shell = 4.0 * np.pi * r**2 * dr
    rho = n_atoms / volume if volume > 0 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = hist / (shell * rho * n_atoms)
    return r, np.nan_to_num(g)
