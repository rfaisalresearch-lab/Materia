"""The central atomistic data model.

Design
------
``Structure`` is a structure-of-arrays container.  Bulk numeric data lives in
contiguous NumPy arrays so that solvers and the renderer can operate without
per-object overhead, while every atom also carries a *stable integer id* that
survives insertion, deletion and reordering.  UI selections, undo records,
bond records and provenance all reference ids, never array indices.

Per-atom scalar fields are dense arrays.  Rarely-populated fields (e.g.
per-atom solver metadata) live in ``Structure.atom_meta``, a sparse dict keyed
by id.

Units are the internal set from :mod:`materia.core_model.units`:
positions in angstrom, velocities in A/fs, forces in eV/A, charges in e,
magnetic moments in Bohr magnetons.
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..elements import periodic_table as pt
from .cell import Cell

NATURAL_ABUNDANCE = 0

SITE_ROLES = (
    "bulk",
    "surface",
    "subsurface",
    "dimer",
    "adatom",
    "dopant",
    "interstitial",
    "vacancy-neighbour",
    "adsorbate",
    "edge",
    "grain-boundary",
    "unknown",
)


class StructureError(Exception):
    """Invalid structural operation."""


@dataclass
class Bond:
    """A bond record between two atom ids.

    Bonds in Materia are *derived* quantities unless imported.  ``order`` is
    a heuristic unless a solver supplied it; ``origin`` records which.
    """

    a: int
    b: int
    order: float = 1.0
    length_A: float = 0.0
    origin: str = "distance-heuristic"

    def key(self) -> Tuple[int, int]:
        return (self.a, self.b) if self.a < self.b else (self.b, self.a)

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "order": self.order,
                "length_A": self.length_A, "origin": self.origin}


class Structure:
    """A collection of atoms in an optional periodic cell."""

    __slots__ = (
        "_ids", "_numbers", "_positions", "_mass_numbers", "_formal_charges",
        "_partial_charges", "_magnetic_moments", "_velocities", "_forces",
        "_fixed", "_roles", "_labels", "cell", "atom_meta", "info", "_next_id",
        "_bonds", "_index_cache",
    )

    def __init__(
        self,
        numbers: Optional[Sequence[int]] = None,
        positions: Optional[np.ndarray] = None,
        cell: Optional[Cell] = None,
        *,
        ids: Optional[Sequence[int]] = None,
        mass_numbers: Optional[Sequence[int]] = None,
        roles: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        info: Optional[dict] = None,
    ) -> None:
        n = 0 if numbers is None else len(numbers)
        self._numbers = np.zeros(n, dtype=np.int32) if numbers is None else np.asarray(numbers, dtype=np.int32)
        if positions is None:
            self._positions = np.zeros((n, 3), dtype=float)
        else:
            self._positions = np.asarray(positions, dtype=float).reshape(n, 3).copy()
        self._ids = (
            np.arange(1, n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        )
        self._mass_numbers = (
            np.full(n, NATURAL_ABUNDANCE, dtype=np.int32)
            if mass_numbers is None else np.asarray(mass_numbers, dtype=np.int32)
        )
        self._formal_charges = np.zeros(n, dtype=float)
        self._partial_charges = np.zeros(n, dtype=float)
        self._magnetic_moments = np.zeros(n, dtype=float)
        self._velocities = np.zeros((n, 3), dtype=float)
        self._forces = np.full((n, 3), np.nan, dtype=float)
        self._fixed = np.zeros(n, dtype=bool)
        self._roles = np.array(list(roles) if roles is not None else ["bulk"] * n, dtype=object)
        self._labels = np.array(list(labels) if labels is not None else [""] * n, dtype=object)
        self.cell: Cell = cell if cell is not None else Cell.none()
        self.atom_meta: Dict[int, dict] = {}
        self.info: dict = dict(info or {})
        self._bonds: Optional[List[Bond]] = None
        self._next_id = int(self._ids.max()) + 1 if n else 1
        self._index_cache: Optional[Dict[int, int]] = None

    def __len__(self) -> int:
        return int(self._numbers.shape[0])

    def __repr__(self) -> str:
        return f"<Structure {len(self)} atoms, {self.formula()}, {self.cell!r}>"

    def copy(self) -> "Structure":
        new = Structure.__new__(Structure)
        for slot in ("_ids", "_numbers", "_positions", "_mass_numbers", "_formal_charges",
                     "_partial_charges", "_magnetic_moments", "_velocities", "_forces",
                     "_fixed", "_roles", "_labels"):
            setattr(new, slot, getattr(self, slot).copy())
        new.cell = self.cell
        new.atom_meta = _copy.deepcopy(self.atom_meta)
        new.info = _copy.deepcopy(self.info)
        new._bonds = None if self._bonds is None else [Bond(**b.as_dict()) for b in self._bonds]
        new._next_id = self._next_id
        new._index_cache = None
        return new

    def restore_from(self, other: "Structure") -> None:
        """Adopt the full contents of ``other`` without changing identity.

        The inverse of :meth:`copy`: every caller still holding a reference to
        this object sees the restored state.  It is what lets an operation that
        mutates a structure in place roll itself back when it fails part-way,
        instead of leaving a half-applied geometry behind.
        """
        for slot in ("_ids", "_numbers", "_positions", "_mass_numbers", "_formal_charges",
                     "_partial_charges", "_magnetic_moments", "_velocities", "_forces",
                     "_fixed", "_roles", "_labels"):
            setattr(self, slot, getattr(other, slot).copy())
        self.cell = other.cell
        self.atom_meta = _copy.deepcopy(other.atom_meta)
        self.info = _copy.deepcopy(other.info)
        self._bonds = (None if other._bonds is None
                       else [Bond(**b.as_dict()) for b in other._bonds])
        self._next_id = other._next_id
        self._index_cache = None

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def numbers(self) -> np.ndarray:
        return self._numbers

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @positions.setter
    def positions(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=float).reshape(len(self), 3)
        self._positions = arr.copy()
        self.invalidate_bonds()

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    @velocities.setter
    def velocities(self, value: np.ndarray) -> None:
        self._velocities = np.asarray(value, dtype=float).reshape(len(self), 3).copy()

    @property
    def forces(self) -> np.ndarray:
        return self._forces

    @forces.setter
    def forces(self, value: np.ndarray) -> None:
        self._forces = np.asarray(value, dtype=float).reshape(len(self), 3).copy()

    @property
    def formal_charges(self) -> np.ndarray:
        return self._formal_charges

    @property
    def partial_charges(self) -> np.ndarray:
        return self._partial_charges

    @property
    def magnetic_moments(self) -> np.ndarray:
        return self._magnetic_moments

    @property
    def mass_numbers(self) -> np.ndarray:
        return self._mass_numbers

    @property
    def fixed(self) -> np.ndarray:
        return self._fixed

    @property
    def roles(self) -> np.ndarray:
        return self._roles

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def _rebuild_index(self) -> None:
        self._index_cache = {int(i): k for k, i in enumerate(self._ids)}

    def index_of(self, atom_id: int) -> int:
        """Array index for a stable atom id."""
        if self._index_cache is None:
            self._rebuild_index()
        try:
            return self._index_cache[int(atom_id)]
        except KeyError as exc:
            raise StructureError(f"No atom with id {atom_id}") from exc

    def indices_of(self, atom_ids: Iterable[int]) -> np.ndarray:
        return np.array([self.index_of(i) for i in atom_ids], dtype=np.int64)

    def has_id(self, atom_id: int) -> bool:
        if self._index_cache is None:
            self._rebuild_index()
        return int(atom_id) in self._index_cache

    def symbols(self) -> List[str]:
        return [pt.symbol(int(z)) for z in self._numbers]

    def masses(self) -> np.ndarray:
        """Per-atom mass in u, honouring explicit isotope assignments."""
        out = np.empty(len(self), dtype=float)
        for k, (z, a) in enumerate(zip(self._numbers, self._mass_numbers)):
            out[k] = pt.mass(int(z), int(a) or None)
        return out

    def formula(self, reduce: bool = False) -> str:
        if len(self) == 0:
            return "(empty)"
        uniq, counts = np.unique(self._numbers, return_counts=True)
        if reduce:
            from math import gcd
            g = 0
            for c in counts:
                g = gcd(g, int(c))
            counts = counts // max(g, 1)
        order = np.argsort([pt.symbol(int(z)) for z in uniq])
        parts = []
        for i in order:
            s = pt.symbol(int(uniq[i]))
            c = int(counts[i])
            parts.append(s if c == 1 else f"{s}{c}")
        return "".join(parts)

    def total_charge(self) -> float:
        return float(self._formal_charges.sum())

    def total_electrons(self) -> float:
        """Total electron count implied by Z and formal charges."""
        return float(self._numbers.sum() - self._formal_charges.sum())

    def total_magnetic_moment(self) -> float:
        return float(self._magnetic_moments.sum())

    def add_atom(
        self,
        symbol_or_z: Union[str, int],
        position: Sequence[float],
        *,
        mass_number: int = NATURAL_ABUNDANCE,
        role: str = "unknown",
        label: str = "",
        formal_charge: float = 0.0,
        magnetic_moment: float = 0.0,
        meta: Optional[dict] = None,
    ) -> int:
        """Append an atom and return its new stable id."""
        z = pt.atomic_number(symbol_or_z)
        pos = np.asarray(position, dtype=float).reshape(1, 3)
        new_id = self._next_id
        self._next_id += 1
        self._ids = np.append(self._ids, np.int64(new_id))
        self._numbers = np.append(self._numbers, np.int32(z))
        self._positions = np.vstack([self._positions, pos])
        self._mass_numbers = np.append(self._mass_numbers, np.int32(mass_number))
        self._formal_charges = np.append(self._formal_charges, float(formal_charge))
        self._partial_charges = np.append(self._partial_charges, 0.0)
        self._magnetic_moments = np.append(self._magnetic_moments, float(magnetic_moment))
        self._velocities = np.vstack([self._velocities, np.zeros((1, 3))])
        self._forces = np.vstack([self._forces, np.full((1, 3), np.nan)])
        self._fixed = np.append(self._fixed, False)
        self._roles = np.append(self._roles, np.array([role], dtype=object))
        self._labels = np.append(self._labels, np.array([label], dtype=object))
        if meta:
            self.atom_meta[new_id] = dict(meta)
        self._index_cache = None
        self.invalidate_bonds()
        return new_id

    def remove_atoms(self, atom_ids: Iterable[int]) -> None:
        wanted = {int(i) for i in atom_ids}
        missing = wanted - set(int(i) for i in self._ids)
        if missing:
            raise StructureError(f"Cannot remove unknown atom ids: {sorted(missing)}")
        keep = np.array([int(i) not in wanted for i in self._ids], dtype=bool)
        self._apply_mask(keep)
        for i in wanted:
            self.atom_meta.pop(i, None)

    def _apply_mask(self, keep: np.ndarray) -> None:
        for slot in ("_ids", "_numbers", "_mass_numbers", "_formal_charges", "_partial_charges",
                     "_magnetic_moments", "_fixed", "_roles", "_labels"):
            setattr(self, slot, getattr(self, slot)[keep])
        for slot in ("_positions", "_velocities", "_forces"):
            setattr(self, slot, getattr(self, slot)[keep])
        self._index_cache = None
        self.invalidate_bonds()

    def substitute(self, atom_id: int, symbol_or_z: Union[str, int], *,
                   role: str = "dopant", mass_number: int = NATURAL_ABUNDANCE) -> None:
        """Replace the element of an existing atom, keeping its id and position."""
        k = self.index_of(atom_id)
        self._numbers[k] = pt.atomic_number(symbol_or_z)
        self._mass_numbers[k] = mass_number
        self._roles[k] = role
        self.invalidate_bonds()

    def set_position(self, atom_id: int, position: Sequence[float]) -> None:
        self._positions[self.index_of(atom_id)] = np.asarray(position, dtype=float)
        self.invalidate_bonds()

    def translate(self, delta: Sequence[float]) -> None:
        self._positions += np.asarray(delta, dtype=float)
        self.invalidate_bonds()

    def wrap(self) -> None:
        self._positions = self.cell.wrap(self._positions)
        self.invalidate_bonds()

    def extend(self, other: "Structure") -> List[int]:
        """Append all atoms of ``other``, returning the new ids."""
        new_ids = []
        for k in range(len(other)):
            new_ids.append(
                self.add_atom(
                    int(other._numbers[k]),
                    other._positions[k],
                    mass_number=int(other._mass_numbers[k]),
                    role=str(other._roles[k]),
                    label=str(other._labels[k]),
                    formal_charge=float(other._formal_charges[k]),
                    magnetic_moment=float(other._magnetic_moments[k]),
                    meta=other.atom_meta.get(int(other._ids[k])),
                )
            )
        return new_ids

    def subset(self, atom_ids: Sequence[int], keep_cell: bool = True) -> "Structure":
        """A new Structure containing only the requested atoms (ids preserved)."""
        idx = self.indices_of(atom_ids)
        sub = Structure(
            numbers=self._numbers[idx],
            positions=self._positions[idx],
            cell=self.cell if keep_cell else Cell.none(),
            ids=self._ids[idx],
            mass_numbers=self._mass_numbers[idx],
            roles=list(self._roles[idx]),
            labels=list(self._labels[idx]),
            info=dict(self.info),
        )
        sub._formal_charges = self._formal_charges[idx].copy()
        sub._partial_charges = self._partial_charges[idx].copy()
        sub._magnetic_moments = self._magnetic_moments[idx].copy()
        sub._velocities = self._velocities[idx].copy()
        sub._forces = self._forces[idx].copy()
        sub._fixed = self._fixed[idx].copy()
        sub._next_id = self._next_id
        sub.atom_meta = {int(i): _copy.deepcopy(self.atom_meta[int(i)])
                         for i in sub._ids if int(i) in self.atom_meta}
        return sub

    def repeat(self, nx: int, ny: int, nz: int) -> "Structure":
        """Tile a periodic structure.  Ids are re-issued for the copies."""
        if not self.cell.is_periodic:
            raise StructureError("repeat() requires a periodic cell")
        base = self.copy()
        out = Structure(cell=self.cell.repeat(nx, ny, nz))
        m = self.cell.matrix
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    shift = i * m[0] + j * m[1] + k * m[2]
                    img = base.copy()
                    img._positions = img._positions + shift
                    out.extend(img)
        out.info = dict(self.info)
        return out

    @property
    def bonds(self) -> Optional[List[Bond]]:
        return self._bonds

    def set_bonds(self, bonds: List[Bond]) -> None:
        self._bonds = list(bonds)

    def invalidate_bonds(self) -> None:
        self._bonds = None

    def ensure_bonds(self, **kwargs) -> List[Bond]:
        """Perceive bonds if they have not been computed yet."""
        if self._bonds is None:
            from ..physics.bonds import perceive_bonds

            self._bonds = perceive_bonds(self, **kwargs)
        return self._bonds

    def bonds_of(self, atom_id: int) -> List[Bond]:
        if self._bonds is None:
            return []
        aid = int(atom_id)
        return [b for b in self._bonds if b.a == aid or b.b == aid]

    def neighbors_of(self, atom_id: int) -> List[int]:
        """Ids bonded to ``atom_id``, perceiving bonds first if needed."""
        self.ensure_bonds()
        aid = int(atom_id)
        return [b.b if b.a == aid else b.a for b in self.bonds_of(aid)]

    def distance(self, id_a: int, id_b: int, mic: bool = True) -> float:
        """Distance in angstrom, minimum-image by default for periodic cells."""
        pa = self._positions[self.index_of(id_a)]
        pb = self._positions[self.index_of(id_b)]
        d = pb - pa
        if mic and self.cell.is_periodic:
            d = self.cell.minimum_image(d[None, :])[0]
        return float(np.linalg.norm(d))

    def angle(self, id_a: int, id_b: int, id_c: int, mic: bool = True) -> float:
        """Angle a-b-c in degrees (b is the vertex)."""
        pa = self._positions[self.index_of(id_a)]
        pb = self._positions[self.index_of(id_b)]
        pc = self._positions[self.index_of(id_c)]
        v1, v2 = pa - pb, pc - pb
        if mic and self.cell.is_periodic:
            v1 = self.cell.minimum_image(v1[None, :])[0]
            v2 = self.cell.minimum_image(v2[None, :])[0]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            raise StructureError("Degenerate angle: coincident atoms")
        return float(np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1.0, 1.0))))

    def dihedral(self, i: int, j: int, k: int, l: int) -> float:
        """Dihedral i-j-k-l in degrees."""
        p = [self._positions[self.index_of(x)] for x in (i, j, k, l)]
        b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
        b1n = b1 / np.linalg.norm(b1)
        v = b0 - (b0 @ b1n) * b1n
        w = b2 - (b2 @ b1n) * b1n
        x = v @ w
        y = np.cross(b1n, v) @ w
        return float(np.degrees(np.arctan2(y, x)))

    def center_of_mass(self) -> np.ndarray:
        m = self.masses()
        return (self._positions * m[:, None]).sum(axis=0) / m.sum()

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self) == 0:
            z = np.zeros(3)
            return z, z
        return self._positions.min(axis=0), self._positions.max(axis=0)

    def as_dict(self) -> dict:
        return {
            "ids": self._ids.tolist(),
            "numbers": self._numbers.tolist(),
            "positions": self._positions.tolist(),
            "mass_numbers": self._mass_numbers.tolist(),
            "formal_charges": self._formal_charges.tolist(),
            "partial_charges": self._partial_charges.tolist(),
            "magnetic_moments": self._magnetic_moments.tolist(),
            "velocities": self._velocities.tolist(),
            "forces": np.where(np.isnan(self._forces), None, self._forces).tolist(),
            "fixed": self._fixed.tolist(),
            "roles": [str(r) for r in self._roles],
            "labels": [str(s) for s in self._labels],
            "cell": self.cell.as_dict(),
            "atom_meta": {str(k): v for k, v in self.atom_meta.items()},
            "info": self.info,
            "next_id": int(self._next_id),
            "bonds": [b.as_dict() for b in self._bonds] if self._bonds is not None else None,
            "units": {"positions": "A", "velocities": "A/fs", "forces": "eV/A",
                      "charges": "e", "magnetic_moments": "mu_B"},
        }

    @staticmethod
    def from_dict(d: dict) -> "Structure":
        s = Structure(
            numbers=d["numbers"],
            positions=np.array(d["positions"], dtype=float),
            cell=Cell.from_dict(d["cell"]),
            ids=d["ids"],
            mass_numbers=d["mass_numbers"],
            roles=d["roles"],
            labels=d["labels"],
            info=d.get("info", {}),
        )
        s._formal_charges = np.array(d["formal_charges"], dtype=float)
        s._partial_charges = np.array(d["partial_charges"], dtype=float)
        s._magnetic_moments = np.array(d["magnetic_moments"], dtype=float)
        s._velocities = np.array(d["velocities"], dtype=float)
        f = np.array([[np.nan if v is None else v for v in row] for row in d["forces"]], dtype=float)
        s._forces = f.reshape(len(s), 3) if len(s) else np.zeros((0, 3))
        s._fixed = np.array(d["fixed"], dtype=bool)
        s.atom_meta = {int(k): v for k, v in d.get("atom_meta", {}).items()}
        s._next_id = int(d.get("next_id", (max(d["ids"]) + 1) if d["ids"] else 1))
        if d.get("bonds") is not None:
            s._bonds = [Bond(**b) for b in d["bonds"]]
        return s

    def atom(self, atom_id: int) -> "AtomView":
        return AtomView(self, int(atom_id))

    def __iter__(self) -> Iterator["AtomView"]:
        for i in self._ids:
            yield AtomView(self, int(i))


class AtomView:
    """Object-style handle onto one atom of a :class:`Structure`.

    The view holds no data of its own; it reads and writes through the parent
    structure so that Python-API edits are visible to solvers immediately.
    """

    __slots__ = ("_s", "_id")

    def __init__(self, structure: Structure, atom_id: int) -> None:
        self._s = structure
        self._id = int(atom_id)
        structure.index_of(self._id)

    @property
    def id(self) -> int:
        return self._id

    @property
    def _k(self) -> int:
        return self._s.index_of(self._id)

    @property
    def atomic_number(self) -> int:
        return int(self._s.numbers[self._k])

    @property
    def symbol(self) -> str:
        return pt.symbol(self.atomic_number)

    @property
    def element(self):
        return pt.element(self.atomic_number)

    @property
    def mass_number(self) -> int:
        return int(self._s.mass_numbers[self._k])

    @mass_number.setter
    def mass_number(self, value: int) -> None:
        self._s.mass_numbers[self._k] = int(value)

    @property
    def mass(self) -> float:
        return pt.mass(self.atomic_number, self.mass_number or None)

    @property
    def position(self) -> np.ndarray:
        return self._s.positions[self._k]

    @position.setter
    def position(self, value) -> None:
        self._s.set_position(self._id, value)

    @property
    def velocity(self) -> np.ndarray:
        return self._s.velocities[self._k]

    @property
    def force(self) -> np.ndarray:
        return self._s.forces[self._k]

    @property
    def charge(self) -> float:
        return float(self._s.formal_charges[self._k])

    @charge.setter
    def charge(self, value: float) -> None:
        self._s.formal_charges[self._k] = float(value)

    @property
    def partial_charge(self) -> float:
        return float(self._s.partial_charges[self._k])

    @property
    def magnetic_moment(self) -> float:
        return float(self._s.magnetic_moments[self._k])

    @magnetic_moment.setter
    def magnetic_moment(self, value: float) -> None:
        self._s.magnetic_moments[self._k] = float(value)

    @property
    def spin(self) -> float:
        """Total spin S implied by the magnetic moment (S = mu / 2 mu_B)."""
        return abs(self.magnetic_moment) / 2.0

    @property
    def role(self) -> str:
        return str(self._s.roles[self._k])

    @role.setter
    def role(self, value: str) -> None:
        self._s.roles[self._k] = str(value)

    @property
    def label(self) -> str:
        return str(self._s.labels[self._k])

    @label.setter
    def label(self, value: str) -> None:
        self._s.labels[self._k] = str(value)

    @property
    def fixed(self) -> bool:
        return bool(self._s.fixed[self._k])

    @fixed.setter
    def fixed(self, value: bool) -> None:
        self._s.fixed[self._k] = bool(value)

    @property
    def meta(self) -> dict:
        return self._s.atom_meta.setdefault(self._id, {})

    @property
    def electron_configuration(self) -> str:
        from ..elements.configuration import configuration_string
        return configuration_string(self.atomic_number, int(round(self.charge)))

    @property
    def coordination(self) -> int:
        """Number of perceived bonds.

        Bonds are a derived quantity. If none have been perceived yet they are
        computed on demand with the default covalent-radius criterion, so this
        never silently reports zero for a bonded atom.
        """
        self._s.ensure_bonds()
        return len(self._s.neighbors_of(self._id))

    @property
    def neighbors(self) -> List["AtomView"]:
        self._s.ensure_bonds()
        return [self._s.atom(i) for i in self._s.neighbors_of(self._id)]

    def set_charge(self, charge: float) -> None:
        self.charge = charge

    def set_spin(self, spin: float) -> None:
        """Set spin S; stores magnetic moment 2S mu_B (spin-only approximation)."""
        self.magnetic_moment = 2.0 * float(spin)

    def as_dict(self) -> dict:
        return {
            "id": self._id,
            "element": self.symbol,
            "atomic_number": self.atomic_number,
            "mass_number": self.mass_number or None,
            "mass_u": self.mass,
            "position_A": self.position.tolist(),
            "velocity_A_fs": self.velocity.tolist(),
            "force_eV_A": [None if np.isnan(v) else float(v) for v in self.force],
            "formal_charge_e": self.charge,
            "partial_charge_e": self.partial_charge,
            "magnetic_moment_muB": self.magnetic_moment,
            "role": self.role,
            "label": self.label,
            "fixed": self.fixed,
            "coordination": self.coordination,
            "meta": dict(self._s.atom_meta.get(self._id, {})),
        }

    def __repr__(self) -> str:
        p = self.position
        return f"<Atom #{self._id} {self.symbol} at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) A>"
