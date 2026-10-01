"""Selections over a structure.

A :class:`Selection` is an ordered, de-duplicated set of stable atom ids plus
the query that produced it.  Keeping the query makes selections reproducible
after edits and lets the UI show "17 atoms: element=P within 8 A of (12,12,4)".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np

from .structure import AtomView, Structure, StructureError


@dataclass
class Selection:
    """Ids of selected atoms, with the query text that produced them."""

    ids: List[int] = field(default_factory=list)
    query: str = ""

    def __post_init__(self) -> None:
        seen: Set[int] = set()
        ordered = []
        for i in self.ids:
            i = int(i)
            if i not in seen:
                seen.add(i)
                ordered.append(i)
        self.ids = ordered

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def __contains__(self, atom_id: int) -> bool:
        return int(atom_id) in set(self.ids)

    def is_empty(self) -> bool:
        return not self.ids

    def union(self, other: "Selection") -> "Selection":
        return Selection(self.ids + other.ids, f"({self.query}) OR ({other.query})")

    def intersection(self, other: "Selection") -> "Selection":
        o = set(other.ids)
        return Selection([i for i in self.ids if i in o], f"({self.query}) AND ({other.query})")

    def difference(self, other: "Selection") -> "Selection":
        o = set(other.ids)
        return Selection([i for i in self.ids if i not in o], f"({self.query}) NOT ({other.query})")

    def invert(self, structure: Structure) -> "Selection":
        mine = set(self.ids)
        return Selection([int(i) for i in structure.ids if int(i) not in mine], f"NOT ({self.query})")

    def atoms(self, structure: Structure) -> List[AtomView]:
        return [structure.atom(i) for i in self.ids if structure.has_id(i)]

    def one(self, structure: Structure) -> AtomView:
        """The single selected atom; raises if the selection is not a singleton."""
        live = [i for i in self.ids if structure.has_id(i)]
        if len(live) != 1:
            raise StructureError(
                f"selected_one() requires exactly one selected atom, got {len(live)}"
            )
        return structure.atom(live[0])

    def prune(self, structure: Structure) -> "Selection":
        """Drop ids that no longer exist (e.g. after a deletion)."""
        return Selection([i for i in self.ids if structure.has_id(i)], self.query)

    def as_dict(self) -> dict:
        return {"ids": list(self.ids), "query": self.query}

    @staticmethod
    def from_dict(d: dict) -> "Selection":
        return Selection(list(d.get("ids", [])), d.get("query", ""))


def by_element(structure: Structure, *symbols: str) -> Selection:
    from ..elements import periodic_table as pt

    wanted = {pt.atomic_number(s) for s in symbols}
    ids = [int(i) for i, z in zip(structure.ids, structure.numbers) if int(z) in wanted]
    return Selection(ids, f"element in {sorted(pt.symbol(z) for z in wanted)}")


def by_role(structure: Structure, *roles: str) -> Selection:
    wanted = set(roles)
    ids = [int(i) for i, r in zip(structure.ids, structure.roles) if str(r) in wanted]
    return Selection(ids, f"role in {sorted(wanted)}")


def within_radius(structure: Structure, center: Sequence[float], radius_A: float,
                  mic: bool = True) -> Selection:
    c = np.asarray(center, dtype=float)
    d = structure.positions - c
    if mic and structure.cell.is_periodic:
        d = structure.cell.minimum_image(d)
    r = np.linalg.norm(d, axis=1)
    ids = [int(i) for i, dist in zip(structure.ids, r) if dist <= radius_A]
    return Selection(ids, f"within {radius_A:g} A of ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")


def neighbors_within(structure: Structure, atom_id: int, radius_A: float) -> Selection:
    center = structure.positions[structure.index_of(atom_id)]
    sel = within_radius(structure, center, radius_A)
    sel.ids = [i for i in sel.ids if i != int(atom_id)]
    sel.query = f"neighbours of #{atom_id} within {radius_A:g} A"
    return sel


def in_box(structure: Structure, lo: Sequence[float], hi: Sequence[float]) -> Selection:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    p = structure.positions
    mask = np.all((p >= lo) & (p <= hi), axis=1)
    ids = [int(i) for i, m in zip(structure.ids, mask) if m]
    return Selection(ids, f"box {lo.tolist()} -> {hi.tolist()}")


def near_plane(structure: Structure, miller: Sequence[int], offset_A: float,
               tolerance_A: float = 0.5) -> Selection:
    """Atoms within ``tolerance_A`` of the lattice plane (hkl) at ``offset_A``.

    The plane normal is computed in the reciprocal lattice so that (hkl) has
    its crystallographic meaning for non-cubic cells.
    """
    h, k, l = (int(v) for v in miller)
    if (h, k, l) == (0, 0, 0):
        raise ValueError("(000) is not a lattice plane")
    if not structure.cell.is_periodic:
        raise ValueError("near_plane() requires a periodic cell")
    g = np.array([h, k, l], dtype=float) @ structure.cell.reciprocal
    n = g / np.linalg.norm(g)
    d = structure.positions @ n - offset_A
    ids = [int(i) for i, dd in zip(structure.ids, d) if abs(dd) <= tolerance_A]
    return Selection(ids, f"plane ({h}{k}{l}) at {offset_A:g} A +/- {tolerance_A:g} A")
