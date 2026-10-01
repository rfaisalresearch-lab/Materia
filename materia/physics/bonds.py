"""Bond perception heuristics.

Materia never claims a perceived bond is a computed chemical bond.  A pair is
a candidate when it satisfies the standard covalent-radius criterion

    d_ij <= scale * (r_cov(i) + r_cov(j)) + tolerance

with ``scale = 1.15`` and ``tolerance = 0.0 A``.  That criterion alone also
joins atoms that are close but screened by others: the Al-Al pairs across the
shared faces of corundum octahedra, the metal-metal pairs of MoS2 and WSe2, and
the second shell of a bcc metal.  A candidate is therefore kept only if the
two atoms share a Voronoi face that subtends at least
:data:`MIN_FACE_FRACTION` of the full solid angle seen from either atom.
Covalent, ionic and metallic first-shell bonds in the shipped crystals subtend
8 to 30 percent; the screened pairs above subtend 0.4 to 3.5 percent.  A face
open to vacuum counts as large.  Above :data:`VORONOI_MAX_ATOMS` atoms, and where a
Voronoi decomposition is undefined
(fewer than five atoms, or all atoms in one plane or line) only the distance
criterion is applied, and the bonds say so in their ``origin``.

Bond *orders* from this rule are always 1.0; a real order requires a Tier-2
or Tier-3 electronic-structure calculation (see
:mod:`materia.solvers.tight_binding`, which can supply Mayer-like overlap
populations for supported systems).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core_model.structure import Bond, Structure
from ..elements import periodic_table as pt
from .neighbors import neighbor_list

DEFAULT_SCALE = 1.15
FALLBACK_RADIUS_A = 1.5
MIN_FACE_FRACTION = 0.05
VORONOI_MAX_ATOMS = 20000
METHODS = ("voronoi", "distance")


def covalent_radii(structure: Structure) -> np.ndarray:
    out = np.empty(len(structure), dtype=float)
    for k, z in enumerate(structure.numbers):
        r = pt.covalent_radius(int(z))
        out[k] = FALLBACK_RADIUS_A if r is None else r
    return out


def _padded_points(positions: np.ndarray, cell: np.ndarray, pbc, padding: float):
    """Atoms plus every periodic image within ``padding`` of the cell."""
    n = len(positions)
    periodic = [bool(p) for p in pbc]
    if not any(periodic):
        return positions, np.arange(n), np.zeros((n, 3), dtype=int)
    inverse = np.linalg.inv(cell)
    frac = positions @ inverse
    widths = np.array([abs(np.linalg.det(cell)) / np.linalg.norm(
        np.cross(cell[(a + 1) % 3], cell[(a + 2) % 3])) for a in range(3)])
    ranges = [range(-int(np.ceil(padding / widths[a])) - 1,
                    int(np.ceil(padding / widths[a])) + 2) if periodic[a] else range(0, 1)
              for a in range(3)]
    points, atoms, shifts = [positions], [np.arange(n)], [np.zeros((n, 3), dtype=int)]
    for s0 in ranges[0]:
        for s1 in ranges[1]:
            for s2 in ranges[2]:
                shift = np.array([s0, s1, s2])
                if not shift.any():
                    continue
                image = frac + shift
                outside = np.maximum(0.0, np.maximum(-image, image - 1.0)) * widths
                keep = np.all(outside < padding, axis=1)
                if keep.any():
                    points.append((image[keep]) @ cell)
                    atoms.append(np.nonzero(keep)[0])
                    shifts.append(np.tile(shift, (int(keep.sum()), 1)))
    return np.vstack(points), np.concatenate(atoms), np.vstack(shifts)


def _polygon_fractions(origins: np.ndarray, polygons: np.ndarray,
                       normals: np.ndarray) -> np.ndarray:
    """Solid angles over 4 pi of planar convex polygons [m, k, 3] seen from origins [m, 3]."""
    centre = polygons.mean(axis=1, keepdims=True)
    helper = np.where(np.abs(normals[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]),
                      np.array([[0.0, 1.0, 0.0]]))
    u = np.cross(normals, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(normals, u)
    offsets = polygons - centre
    angle = np.arctan2(np.einsum("mkc,mc->mk", offsets, w), np.einsum("mkc,mc->mk", offsets, u))
    order = np.argsort(angle, axis=1)
    ordered = np.take_along_axis(polygons, order[..., None], axis=1) - origins[:, None, :]
    a = ordered[:, :1, :]
    b = ordered[:, 1:-1, :]
    c = ordered[:, 2:, :]
    la = np.linalg.norm(a, axis=2)
    lb = np.linalg.norm(b, axis=2)
    lc = np.linalg.norm(c, axis=2)
    numerator = np.abs(np.einsum("mtc,mtc->mt", np.broadcast_to(a, b.shape), np.cross(b, c)))
    dot = lambda x, y: np.einsum("mtc,mtc->mt", np.broadcast_to(x, y.shape), y)
    denominator = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + np.einsum(
        "mtc,mtc->mt", b, c) * la
    return 2.0 * np.arctan2(numerator, denominator).sum(axis=1) / (4.0 * np.pi)


def _face_fractions(structure: Structure, ci: np.ndarray, cj: np.ndarray,
                    cshift: np.ndarray, padding: float) -> Optional[np.ndarray]:
    """Solid-angle fraction of the Voronoi face each candidate pair shares, or None.

    A pair with no shared face gets 0; a face open to vacuum gets infinity.
    """
    from scipy.spatial import QhullError, Voronoi

    positions = np.asarray(structure.positions, dtype=float)
    cell = np.asarray(structure.cell.matrix, dtype=float)
    points, atoms, shifts = _padded_points(positions, cell, structure.cell.pbc, padding)
    centred = points - points.mean(axis=0)
    if len(points) < 5 or np.linalg.matrix_rank(centred, tol=1e-6) < 3:
        return None
    try:
        voronoi = Voronoi(points)
    except (QhullError, ValueError):
        return None
    n = len(positions)
    span = int(max(np.abs(shifts).max(), np.abs(cshift).max() if len(cshift) else 0)) + 1
    base = 2 * span + 1

    def encode(i, j, shift):
        code = ((shift[:, 0] + span) * base + (shift[:, 1] + span)) * base + (shift[:, 2] + span)
        return (i.astype(np.int64) * n + j.astype(np.int64)) * base ** 3 + code

    candidates = encode(ci, cj, cshift)
    ridge = np.asarray(voronoi.ridge_points)
    faces = np.zeros(len(ci))
    order = np.argsort(candidates)
    sorted_candidates = candidates[order]
    found = np.zeros(len(ci), dtype=bool)
    matched_ridges = np.full(len(ci), -1)
    matched_origin = np.zeros(len(ci), dtype=int)
    matched_other = np.zeros(len(ci), dtype=int)
    for first, second in ((0, 1), (1, 0)):
        a, b = ridge[:, first], ridge[:, second]
        usable = a < n
        idx = np.nonzero(usable)[0]
        keys = encode(a[idx], atoms[b[idx]], shifts[b[idx]])
        reverse = encode(atoms[b[idx]], a[idx], -shifts[b[idx]])
        for probe in (keys, reverse):
            position = np.searchsorted(sorted_candidates, probe)
            position = np.clip(position, 0, len(sorted_candidates) - 1)
            hit = sorted_candidates[position] == probe
            targets = order[position[hit]]
            fresh = ~found[targets]
            targets, rows = targets[fresh], idx[hit][fresh]
            targets, unique = np.unique(targets, return_index=True)
            rows = rows[unique]
            found[targets] = True
            matched_ridges[targets] = rows
            matched_origin[targets] = a[rows]
            matched_other[targets] = b[rows]
    vertices = voronoi.vertices
    groups: Dict[int, List[int]] = {}
    for target in np.nonzero(found)[0]:
        loop = voronoi.ridge_vertices[matched_ridges[target]]
        if -1 in loop:
            faces[target] = np.inf
        else:
            groups.setdefault(len(loop), []).append(int(target))
    for size, members in groups.items():
        members = np.asarray(members)
        polygons = np.stack([vertices[voronoi.ridge_vertices[matched_ridges[m]]]
                             for m in members])
        origins = points[matched_origin[members]]
        normals = points[matched_other[members]] - origins
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        faces[members] = _polygon_fractions(origins, polygons, normals)
    return faces


def perceive_bonds(
    structure: Structure,
    scale: float = DEFAULT_SCALE,
    tolerance_A: float = 0.0,
    max_cutoff_A: Optional[float] = None,
    method: str = "voronoi",
    min_face_fraction: float = MIN_FACE_FRACTION,
    voronoi_max_atoms: int = VORONOI_MAX_ATOMS,
) -> List[Bond]:
    """Return heuristic bonds.  Does not mutate ``structure``.

    ``method="voronoi"`` keeps a distance candidate only if the pair shares a
    Voronoi face of at least ``min_face_fraction`` of the full solid angle;
    ``method="distance"`` applies the distance criterion alone.  Above
    ``voronoi_max_atoms`` the Voronoi filter is skipped to keep interactive
    editing fast, and the bonds' ``origin`` says so.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}, got {method!r}.")
    n = len(structure)
    if n < 2:
        return []
    radii = covalent_radii(structure)
    cutoff = float(scale * 2.0 * radii.max() + tolerance_A)
    if max_cutoff_A is not None:
        cutoff = min(cutoff, max_cutoff_A)
    nl = neighbor_list(structure.positions, structure.cell, cutoff, half=True,
                       coincident="exclude")
    limit = scale * (radii[nl.i] + radii[nl.j]) + tolerance_A
    keep = nl.d <= limit
    ci, cj, cd = nl.i[keep], nl.j[keep], nl.d[keep]
    cshift = np.asarray(nl.shifts)[keep].astype(int) if len(nl) else np.zeros((0, 3), int)
    origin = f"distance-heuristic(scale={scale:g})"
    if method == "voronoi" and len(ci) and n > voronoi_max_atoms:
        origin = (f"distance-heuristic(scale={scale:g}); Voronoi filter skipped above "
                  f"{voronoi_max_atoms} atoms")
    elif method == "voronoi" and len(ci):
        face = _face_fractions(structure, ci, cj, cshift, padding=2.0 * float(cd.max()) + 1.0)
        if face is not None:
            selected = face >= min_face_fraction
            ci, cj, cd = ci[selected], cj[selected], cd[selected]
            origin = (f"distance-heuristic(scale={scale:g}) with a shared Voronoi face of at "
                      f"least {min_face_fraction:g} of the solid angle")
    ids = structure.ids
    return [
        Bond(a=int(ids[i]), b=int(ids[j]), order=1.0, length_A=float(d), origin=origin)
        for i, j, d in zip(ci, cj, cd)
    ]


def attach_bonds(structure: Structure, **kwargs) -> Structure:
    """Perceive and store bonds on the structure (in place). Returns it."""
    structure.set_bonds(perceive_bonds(structure, **kwargs))
    return structure
