"""Primitive-lattice detection.

Material definitions are written in the *conventional* crystallographic cell,
which for centred lattices (F, I, C, R) contains several primitive cells.  A
correct surface construction has to enumerate vectors of the real translation
lattice, otherwise the smallest surface cell it can find is the conventional
one -- e.g. a 7.68 A Si(111) cell instead of the true 3.84 A one.

This module recovers the translation lattice directly from the structure: a
fractional vector ``t`` is a lattice translation if shifting every atom by
``t`` maps the decorated lattice onto itself (same element at every site).
No space-group tables are consulted, so it works for user-supplied materials
and for supercells containing defects (where it correctly finds no extra
translations).
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure

TOL = 1e-4


def _wrap(frac: np.ndarray) -> np.ndarray:
    return frac - np.floor(frac + TOL)


def translation_lattice(structure: Structure, tol: float = TOL) -> List[np.ndarray]:
    """Fractional translations in [0,1)^3 that map the structure onto itself."""
    if len(structure) == 0 or not structure.cell.is_periodic:
        return [np.zeros(3)]
    frac = _wrap(structure.cell.to_fractional(structure.positions))
    numbers = np.asarray(structure.numbers)
    z0 = numbers[0]
    candidates = [_wrap(frac[j] - frac[0]) for j in np.nonzero(numbers == z0)[0]]

    valid: List[np.ndarray] = []
    for t in candidates:
        shifted = _wrap(frac + t)
        if _maps_onto(shifted, frac, numbers, tol):
            valid.append(t)
    if not valid:
        valid = [np.zeros(3)]
    return valid


def _maps_onto(shifted: np.ndarray, frac: np.ndarray, numbers: np.ndarray, tol: float) -> bool:
    used = np.zeros(len(frac), dtype=bool)
    for i in range(len(shifted)):
        d = shifted[i] - frac
        d -= np.round(d)
        hit = (np.abs(d) < tol).all(axis=1) & (numbers == numbers[i]) & ~used
        idx = np.nonzero(hit)[0]
        if idx.size == 0:
            return False
        used[idx[0]] = True
    return True


def _reduce_basis(P: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """Greedy Minkowski-style reduction of a 3x3 fractional basis."""
    P = P.astype(float).copy()

    def length(v):
        return float(np.sqrt(v @ metric @ v))

    for _ in range(100):
        changed = False
        order = sorted(range(3), key=lambda i: length(P[i]))
        P = P[order]
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                denom = P[j] @ metric @ P[j]
                if denom <= 0:
                    continue
                mu = round((P[i] @ metric @ P[j]) / denom)
                if mu != 0:
                    cand = P[i] - mu * P[j]
                    if length(cand) < length(P[i]) - 1e-9 and abs(np.linalg.det(
                            np.array([cand if k == i else P[k] for k in range(3)]))) > 1e-9:
                        P[i] = cand
                        changed = True
        if not changed:
            break
    if np.linalg.det(P) < 0:
        P[[0, 1]] = P[[1, 0]]
    return P


def primitive_basis(structure: Structure) -> Tuple[np.ndarray, int]:
    """Return ``(P_frac, multiplicity)``.

    ``P_frac`` rows are the primitive lattice vectors expressed in fractional
    coordinates of the input cell; ``multiplicity`` is how many primitive cells
    the input cell contains.
    """
    translations = translation_lattice(structure)
    n = len(translations)
    if n == 1:
        return np.eye(3), 1

    pool: List[np.ndarray] = []
    for t in translations:
        for shift in itertools.product((-1, 0, 1), repeat=3):
            v = t + np.array(shift, dtype=float)
            if np.linalg.norm(v) < TOL:
                continue
            pool.append(v)
    m = structure.cell.matrix
    metric = m @ m.T
    pool.sort(key=lambda v: float(np.linalg.norm(v @ m)))

    chosen: List[np.ndarray] = []
    for v in pool:
        if not chosen:
            chosen.append(v)
            continue
        if len(chosen) == 1:
            if np.linalg.norm(np.cross(chosen[0] @ m, v @ m)) > 1e-6:
                chosen.append(v)
            continue
        det = np.linalg.det(np.array([chosen[0], chosen[1], v]))
        if abs(det) > 1e-9:
            chosen.append(v)
            break
    if len(chosen) != 3:
        return np.eye(3), 1

    P = _reduce_basis(np.array(chosen), metric)
    det = abs(np.linalg.det(P))
    expected = 1.0 / n
    if abs(det - expected) > 1e-6:
        return np.eye(3), 1
    return P, n


def primitive_structure(structure: Structure) -> Structure:
    """Reduce a periodic structure to its primitive cell.

    Returns the input unchanged (a copy) when no smaller cell exists.
    """
    P, mult = primitive_basis(structure)
    if mult == 1:
        return structure.copy()

    prim_matrix = P @ structure.cell.matrix
    inv = np.linalg.inv(prim_matrix)
    frac = _wrap(structure.positions @ inv)

    keep: List[int] = []
    seen: List[np.ndarray] = []
    for i in range(len(structure)):
        dup = False
        for j, f in enumerate(seen):
            d = frac[i] - f
            d -= np.round(d)
            if (np.abs(d) < 1e-3).all() and structure.numbers[i] == structure.numbers[keep[j]]:
                dup = True
                break
        if not dup:
            seen.append(frac[i])
            keep.append(i)

    expected = len(structure) // mult
    if len(keep) != expected:
        raise ValueError(
            f"Primitive reduction produced {len(keep)} atoms, expected {expected}. "
            "The cell may contain a defect or a partially occupied site; "
            "use the conventional cell instead."
        )

    out = Structure(
        structure.numbers[keep],
        frac[keep] @ prim_matrix,
        Cell(prim_matrix, structure.cell.pbc),
        labels=[str(structure.labels[i]) for i in keep],
        roles=[str(structure.roles[i]) for i in keep],
    )
    out.info = dict(structure.info)
    out.info["primitive_reduction"] = {
        "multiplicity": mult,
        "transformation_fractional": P.tolist(),
        "method": "translation-lattice detection from the decorated structure",
    }
    return out
