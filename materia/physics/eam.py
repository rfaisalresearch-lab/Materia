"""Embedded-atom method potentials read from setfl (eam/alloy) files.

Model
-----
The embedded-atom method (Daw and Baskes 1984) writes the energy of a metal as

``E = sum_i F_a(rho_i) + 1/2 sum_{i != j} phi_ab(r_ij)``,
``rho_i = sum_{j != i} f_b(r_ij)``

where ``a`` and ``b`` are the elements of atoms ``i`` and ``j``, ``F_a`` is the
embedding energy of an atom of element ``a`` in a host electron density
``rho``, ``f_b`` the density contributed by a neighbour of element ``b`` and
``phi_ab`` a pair repulsion.  All three functions come from a parameter file;
nothing is fitted, inferred or mixed here.

Units: distances in angstrom, energies in eV, forces in eV/A.  The density is
in the arbitrary units of the file and is never shown as a physical quantity.

File format
-----------
Only the setfl format read by LAMMPS ``pair_style eam/alloy`` is supported:
three comment lines; a line with the number of elements and their symbols; a
line with ``Nrho drho Nr dr cutoff``; for each element a line
``Z mass lattice_constant lattice_type`` followed by ``Nrho`` values of
``F(rho)`` and ``Nr`` values of ``f(r)``; then ``Nr`` values of ``r phi(r)``
for every element pair ``i >= j`` in file order.  The grids are
``rho_k = k drho`` and ``r_k = k dr``.  A file with too few or too many
values, a non-finite value, an unknown element, an atomic number that does
not match its symbol, or a cutoff beyond its own radial table is refused.

Interpolation
-------------
Each tabulated function is interpolated with the piecewise cubic Hermite
scheme of LAMMPS ``pair_style eam`` and the OpenKIM ``EAM_Dynamo`` driver:
slopes from fourth-order central differences (second order at the ends),
one cubic per interval, and derivatives from the same cubic, so the forces
are the exact gradient of the interpolated energy.  A density above the last
tabulated value is extrapolated linearly from the end of the table, as LAMMPS
does, and every result reports whether that happened.

References
----------
M. S. Daw and M. I. Baskes, Phys. Rev. B 29 (1984) 6443.
S. M. Foiles, M. I. Baskes and M. S. Daw, Phys. Rev. B 33 (1986) 7983.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..elements import periodic_table as pt
from ..provenance import Fidelity
from .neighbors import (
    COLLISION_TOLERANCE_A, CoincidentAtoms, find_coincidence, image_grid, wrap_for_search,
)
from .potentials import (
    Potential, PotentialError, UnsupportedSystem, refuse_overlap,
)

POTENTIAL_DIR = Path(__file__).resolve().parents[1] / "potentials" / "eam"
MANIFEST_PATH = POTENTIAL_DIR / "manifest.json"
USER_DIR_ENV = "MATERIA_POTENTIALS_DIR"
USER_SUFFIXES = (".eam.alloy", ".setfl")
DEFAULT_SKIN_A = 0.5
PAIR_CHUNK = 2_000_000
MIN_TABLE_POINTS = 5

REFERENCES = [
    "M. S. Daw and M. I. Baskes, Phys. Rev. B 29 (1984) 6443",
    "S. M. Foiles, M. I. Baskes and M. S. Daw, Phys. Rev. B 33 (1986) 7983",
]


class EAMFileError(PotentialError):
    """A parameter file is malformed, truncated, inconsistent or altered."""


class HermiteTable:
    """A function tabulated on ``x_k = k delta``, interpolated as in LAMMPS."""

    __slots__ = ("delta", "n", "c0", "c1", "c2", "c3", "c4", "c5", "c6")

    def __init__(self, values: np.ndarray, delta: float) -> None:
        f = np.asarray(values, dtype=float)
        n = f.size
        if n < MIN_TABLE_POINTS:
            raise EAMFileError(f"A tabulated function needs at least {MIN_TABLE_POINTS} "
                               f"points; this one has {n}.")
        slope = np.empty(n)
        slope[0] = f[1] - f[0]
        slope[1] = 0.5 * (f[2] - f[0])
        slope[n - 2] = 0.5 * (f[n - 1] - f[n - 3])
        slope[n - 1] = f[n - 1] - f[n - 2]
        slope[2:n - 2] = ((f[0:n - 4] - f[4:n]) + 8.0 * (f[3:n - 1] - f[1:n - 3])) / 12.0
        c4 = np.zeros(n)
        c3 = np.zeros(n)
        c4[:-1] = 3.0 * (f[1:] - f[:-1]) - 2.0 * slope[:-1] - slope[1:]
        c3[:-1] = slope[:-1] + slope[1:] - 2.0 * (f[1:] - f[:-1])
        self.delta = float(delta)
        self.n = n
        self.c6 = f
        self.c5 = slope
        self.c4 = c4
        self.c3 = c3
        self.c2 = slope / delta
        self.c1 = 2.0 * c4 / delta
        self.c0 = 3.0 * c3 / delta

    @property
    def x_max(self) -> float:
        return (self.n - 1) * self.delta

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Value and derivative at ``x`` (clamped to the table, as LAMMPS does)."""
        x = np.asarray(x, dtype=float)
        p = x / self.delta
        m = np.clip(np.floor(p).astype(np.int64), 0, self.n - 2)
        p = np.minimum(p - m, 1.0)
        value = ((self.c3[m] * p + self.c4[m]) * p + self.c5[m]) * p + self.c6[m]
        slope = (self.c0[m] * p + self.c1[m]) * p + self.c2[m]
        return value, slope


@dataclass
class SetflData:
    """The content of one setfl file, validated."""

    comments: Tuple[str, str, str]
    elements: Tuple[str, ...]
    atomic_numbers: Tuple[int, ...]
    masses: Tuple[float, ...]
    lattice_constants: Tuple[float, ...]
    lattice_types: Tuple[str, ...]
    nrho: int
    drho: float
    nr: int
    dr: float
    cutoff: float
    embedding: List[np.ndarray]
    density: List[np.ndarray]
    pair: Dict[Tuple[int, int], np.ndarray]
    header_corrections: Tuple[str, ...] = ()

    def header(self) -> dict:
        return {"elements": list(self.elements), "atomic_numbers": list(self.atomic_numbers),
                "masses_u": list(self.masses),
                "header_lattice_constants_A": list(self.lattice_constants),
                "header_lattice_types": list(self.lattice_types),
                "nrho": self.nrho, "drho": self.drho, "nr": self.nr, "dr_A": self.dr,
                "cutoff_A": self.cutoff, "comments": list(self.comments)}


class _Lines:
    def __init__(self, text: str) -> None:
        self.lines = text.splitlines()
        self.index = 0

    def next_line(self, what: str) -> str:
        while self.index < len(self.lines):
            line = self.lines[self.index]
            self.index += 1
            return line
        raise EAMFileError(f"The file ends before {what}.")

    def numbers(self, count: int, what: str) -> np.ndarray:
        out: List[str] = []
        while len(out) < count:
            if self.index >= len(self.lines):
                raise EAMFileError(
                    f"The file is truncated: {what} needs {count} values and only "
                    f"{len(out)} were found.")
            tokens = self.lines[self.index].split()
            self.index += 1
            out.extend(tokens)
        if len(out) != count:
            raise EAMFileError(
                f"{what} is followed by {len(out) - count} unexpected value(s) on the "
                "same line; the table sizes in the header do not match the data.")
        try:
            values = np.array([float(t.replace("D", "E").replace("d", "e")) for t in out])
        except ValueError:
            bad = next(t for t in out if not _is_float(t))
            raise EAMFileError(f"{what} contains the non-numeric entry {bad!r}.") from None
        if not np.all(np.isfinite(values)):
            raise EAMFileError(f"{what} contains a value that is not finite.")
        return values

    def remainder(self) -> List[str]:
        return [t for line in self.lines[self.index:] for t in line.split()]


def _is_float(token: str) -> bool:
    try:
        float(token.replace("D", "E").replace("d", "e"))
        return True
    except ValueError:
        return False


def parse_setfl(text: str, trust_element_symbols: bool = False) -> SetflData:
    """Parse and validate setfl text.  Raises :class:`EAMFileError`.

    The atomic number in each element header must match the element symbol.
    LAMMPS reads only the symbol, and some distributed files carry a wrong
    number; ``trust_element_symbols`` accepts such a file, uses the symbol and
    records every correction in ``header_corrections``.
    """
    lines = _Lines(text)
    comments = tuple(lines.next_line("the three comment lines") for _ in range(3))
    head = lines.next_line("the element line").split()
    if not head:
        raise EAMFileError("The element line (line 4) is empty.")
    try:
        n_el = int(head[0])
    except ValueError:
        raise EAMFileError(f"Line 4 must start with the number of elements; found {head[0]!r}.") from None
    symbols = head[1:]
    if n_el < 1 or len(symbols) != n_el:
        raise EAMFileError(f"Line 4 declares {n_el} element(s) but names {len(symbols)}.")
    try:
        elements = tuple(pt.symbol(s) for s in symbols)
    except Exception:
        raise EAMFileError(f"Line 4 names an unknown element among {symbols}.") from None
    if len(set(elements)) != len(elements):
        raise EAMFileError(f"Line 4 lists an element twice: {symbols}.")
    grid = lines.next_line("the grid line").split()
    if len(grid) < 5:
        raise EAMFileError("Line 5 must give Nrho, drho, Nr, dr and the cutoff.")
    try:
        nrho, drho, nr, dr, cutoff = int(grid[0]), float(grid[1]), int(grid[2]), float(grid[3]), float(grid[4])
    except ValueError:
        raise EAMFileError(f"Line 5 is not Nrho drho Nr dr cutoff: {' '.join(grid)!r}.") from None
    if nrho < MIN_TABLE_POINTS or nr < MIN_TABLE_POINTS:
        raise EAMFileError(f"The tables need at least {MIN_TABLE_POINTS} points; "
                           f"Nrho = {nrho}, Nr = {nr}.")
    for name, value in (("drho", drho), ("dr", dr), ("cutoff", cutoff)):
        if not (math.isfinite(value) and value > 0):
            raise EAMFileError(f"{name} must be a positive number; found {value}.")
    if cutoff > (nr - 1) * dr * (1.0 + 1e-6):
        raise EAMFileError(
            f"The cutoff {cutoff} A lies beyond the radial table, which ends at "
            f"{(nr - 1) * dr:.6f} A; the functions are undefined between the two.")
    numbers, masses, lattice_a, lattice_type = [], [], [], []
    embedding, density = [], []
    corrections = []
    for k, element in enumerate(elements):
        parts = lines.next_line(f"the header of element {element}").split()
        if len(parts) < 3:
            raise EAMFileError(f"The header of element {element} must give Z, mass and "
                               "lattice constant.")
        try:
            z = int(float(parts[0]))
            mass = float(parts[1])
            a = float(parts[2])
        except ValueError:
            raise EAMFileError(f"The header of element {element} is not numeric: "
                               f"{' '.join(parts)!r}.") from None
        if z != pt.element(element).number:
            if not trust_element_symbols:
                raise EAMFileError(f"The file gives atomic number {z} for {element}, which "
                                   f"is element {pt.element(element).number}.")
            corrections.append(f"The header gives atomic number {z} for {element}; the "
                               f"symbol was used (Z = {pt.element(element).number}), as "
                               "LAMMPS does.")
            z = pt.element(element).number
        if not (mass > 0 and math.isfinite(mass)):
            raise EAMFileError(f"The mass of {element} must be positive; found {mass}.")
        numbers.append(z)
        masses.append(mass)
        lattice_a.append(a)
        lattice_type.append(parts[3] if len(parts) > 3 else "")
        embedding.append(lines.numbers(nrho, f"F(rho) for {element}"))
        density.append(lines.numbers(nr, f"f(r) for {element}"))
    pair: Dict[Tuple[int, int], np.ndarray] = {}
    for i in range(n_el):
        for j in range(i + 1):
            pair[(i, j)] = lines.numbers(nr, f"r*phi(r) for {elements[i]}-{elements[j]}")
    extra = lines.remainder()
    if extra:
        raise EAMFileError(f"The file has {len(extra)} value(s) after the last pair table; "
                           "it does not match its own header.")
    return SetflData(comments=comments, elements=elements, atomic_numbers=tuple(numbers),
                     masses=tuple(masses), lattice_constants=tuple(lattice_a),
                     lattice_types=tuple(lattice_type), nrho=nrho, drho=drho, nr=nr,
                     dr=dr, cutoff=cutoff, embedding=embedding, density=density, pair=pair,
                     header_corrections=tuple(corrections))


@dataclass(frozen=True)
class PotentialIdentity:
    """Who made a parameter file, where it came from, and exactly which bytes."""

    id: str
    file_name: str
    sha256: str
    elements: Tuple[str, ...]
    family: str = "embedded-atom method, eam/alloy (setfl) tabulation"
    title: str = ""
    shipped: bool = False
    path: str = ""
    source_url: str = ""
    openkim_id: str = ""
    openkim_doi: str = ""
    content_origin: str = ""
    retrieved: str = ""
    license: str = "unknown"
    license_text: str = ""
    license_source: str = ""
    citations: Tuple[str, ...] = ()
    intended_structures: Tuple[Tuple[str, str], ...] = ()
    notes: Tuple[str, ...] = ()
    interpolation: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "file": self.file_name, "sha256": self.sha256,
                "elements": list(self.elements), "family": self.family, "title": self.title,
                "shipped": self.shipped, "source_url": self.source_url,
                "openkim_id": self.openkim_id, "openkim_doi": self.openkim_doi,
                "content_origin": self.content_origin, "retrieved": self.retrieved,
                "license": self.license, "license_text": self.license_text,
                "license_source": self.license_source, "citations": list(self.citations),
                "intended_structures": dict(self.intended_structures),
                "notes": list(self.notes), "interpolation": self.interpolation}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_PARSED: Dict[str, SetflData] = {}


def _load_bytes(data: bytes, trust_element_symbols: bool = False) -> SetflData:
    key = _sha256(data) + (":symbols" if trust_element_symbols else "")
    if key not in _PARSED:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        _PARSED[key] = parse_setfl(text, trust_element_symbols)
    return _PARSED[key]


def catalog() -> List[dict]:
    """The shipped potentials as recorded in the manifest."""
    return list(json.loads(MANIFEST_PATH.read_text())["potentials"])


def shipped_ids() -> List[str]:
    return [entry["id"] for entry in catalog()]


def _identity_from_manifest(entry: dict) -> PotentialIdentity:
    return PotentialIdentity(
        id=entry["id"], file_name=entry["file"], sha256=entry["sha256"],
        elements=tuple(entry["elements"]), family=entry.get("family", ""),
        title=entry.get("title", ""), shipped=True, path=str(POTENTIAL_DIR / entry["file"]),
        source_url=entry.get("source_url", ""), openkim_id=entry.get("openkim_id", ""),
        openkim_doi=entry.get("openkim_doi", ""),
        content_origin=entry.get("content_origin", ""), retrieved=entry.get("retrieved", ""),
        license=entry.get("license", "unknown"), license_text=entry.get("license_text", ""),
        license_source=entry.get("license_source", ""),
        citations=tuple(entry.get("citations", [])),
        intended_structures=tuple(sorted(entry.get("intended_structures", {}).items())),
        notes=tuple(entry.get("notes", [])), interpolation=entry.get("interpolation", ""))


def load_shipped(potential_id: str) -> "EAMPotential":
    """A shipped potential, after checking its bytes against the manifest."""
    entries = {entry["id"]: entry for entry in catalog()}
    if potential_id not in entries:
        raise UnsupportedSystem(f"No shipped EAM potential {potential_id!r}. "
                                f"Shipped: {', '.join(sorted(entries))}.")
    identity = _identity_from_manifest(entries[potential_id])
    data = Path(identity.path).read_bytes()
    digest = _sha256(data)
    if digest != identity.sha256:
        raise EAMFileError(
            f"{identity.file_name} does not match its recorded checksum (expected "
            f"{identity.sha256[:16]}..., found {digest[:16]}...). The file has been altered "
            "and will not be used.")
    return EAMPotential(_load_bytes(data), identity)


def load_file(path: str, potential_id: Optional[str] = None) -> "EAMPotential":
    """A user-supplied setfl file.  Its licence is unknown to Materia."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise EAMFileError(f"No such parameter file: {p}.")
    data = p.read_bytes()
    table = _load_bytes(data)
    digest = _sha256(data)
    identity = PotentialIdentity(
        id=potential_id or f"file:{p.name}:{digest[:8]}", file_name=p.name, sha256=digest,
        elements=table.elements, shipped=False, path=str(p),
        title=f"User-supplied setfl file {p.name}",
        license="unknown (user-supplied)",
        notes=("User-supplied file. Materia has not validated it and does not know its "
               "licence or provenance beyond its checksum.",))
    return EAMPotential(table, identity)


def load_lammps(name: str, citation: str = "") -> "EAMPotential":
    """An eam/alloy file from the LAMMPS potentials directory, downloaded and cached.

    The file is distributed with LAMMPS under GPL-2.0 and is never shipped by
    Materia.  Element symbols are trusted over header atomic numbers, as in
    LAMMPS; any correction is recorded in the potential's notes.
    """
    from .tersoff import LAMMPS_URL, fetch_lammps_file

    if not name.endswith(".eam.alloy"):
        raise EAMFileError("Only eam/alloy (setfl) files are read here.")
    path = fetch_lammps_file(name)
    data = path.read_bytes()
    table = _load_bytes(data, trust_element_symbols=True)
    digest = _sha256(data)
    identity = PotentialIdentity(
        id=f"lammps:{name}:{digest[:8]}", file_name=name, sha256=digest,
        elements=table.elements, shipped=False, path=str(path), source_url=LAMMPS_URL + name,
        title=f"LAMMPS distribution file {name}", license="GPL-2.0 (LAMMPS distribution)",
        citations=(citation,) if citation else (),
        notes=("Downloaded from the LAMMPS repository at run time; not shipped.",)
        + tuple(table.header_corrections))
    return EAMPotential(table, identity)


def user_directory() -> Path:
    return Path(os.environ.get(USER_DIR_ENV) or Path.home() / ".materia" / "potentials")


def user_files() -> List[Path]:
    root = user_directory()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_file() and any(p.name.endswith(s) for s in USER_SUFFIXES))


def select_shipped(elements: Sequence[str]) -> Tuple[Optional[str], str]:
    """The shipped potential that covers exactly these elements, or why none does.

    The choice is deterministic: a potential whose element set equals the
    structure's is preferred, then the one with the fewest extra elements,
    then the lowest identifier.  A structure whose elements are split across
    several single-element files gets no potential, because combining files
    would mean constructing cross interactions the files do not contain.
    """
    wanted = {pt.symbol(e) for e in elements}
    if not wanted:
        return None, "The structure has no atoms."
    candidates = []
    for entry in catalog():
        have = set(entry["elements"])
        if wanted <= have:
            candidates.append((len(have - wanted), entry["id"]))
    if candidates:
        return sorted(candidates)[0][1], ""
    covered = sorted({e for entry in catalog() for e in entry["elements"]})
    missing = sorted(wanted - set(covered))
    if missing:
        return None, (f"No shipped EAM potential covers {', '.join(missing)}. Shipped "
                      f"potentials cover {', '.join(covered)}, one element per file.")
    return None, (f"No shipped EAM potential contains {', '.join(sorted(wanted))} "
                  "together. Each shipped file describes one element, and Materia does "
                  "not combine files: the cross interactions of an alloy would have to be "
                  "constructed rather than read. Supply a setfl file that contains all of "
                  "these elements.")


def pair_search(positions: np.ndarray, cell: Cell, cutoff: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed pairs within ``cutoff``: indices ``i``, ``j`` and image shifts.

    Periodic images are expanded explicitly, so the list is exact for cells
    thinner than twice the cutoff.  Positions are wrapped into the cell along
    periodic directions for the search, and the wrap is folded back into each
    pair's shift, so ``positions[j] + shift - positions[i]`` is the true
    separation for positions stored anywhere, including outside the cell.
    The order is fixed by the k-d tree and is identical for identical input.

    Only a true self-pair, the same atom in the zero lattice image, is left
    out, and it is identified by index and image, never by distance. Two
    distinct atoms, or an atom and an image of another, closer than
    :data:`~materia.physics.neighbors.COLLISION_TOLERANCE_A` raise
    :class:`~materia.physics.neighbors.CoincidentAtoms`; nothing is returned
    for such a geometry.
    """
    raw = np.asarray(positions, dtype=float)
    n = len(raw)
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros((0, 3))
    pos, wrap = wrap_for_search(raw, cell)
    wrap_shift = wrap @ cell.matrix if cell.is_periodic else np.zeros_like(raw)
    grid = image_grid(cell, cutoff)
    offsets = grid @ cell.matrix if cell.is_periodic else np.zeros((1, 3))
    zero = int(np.nonzero(~grid.any(axis=1))[0][0])
    ghosts = (pos[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    tree = cKDTree(ghosts)
    density = 4.0 / 3.0 * math.pi * cutoff ** 3 * n / max(
        cell.volume if cell.is_periodic and cell.volume > 0 else 1.0, 1.0)
    step = max(1, int(PAIR_CHUNK // max(1.0, min(density, float(len(ghosts))))))
    i_parts, j_parts, s_parts = [], [], []
    for start in range(0, n, step):
        stop = min(n, start + step)
        found = cKDTree(pos[start:stop]).sparse_distance_matrix(tree, cutoff,
                                                                output_type="ndarray")
        if found.size == 0:
            continue
        flat = found["j"].astype(np.int64)
        i_idx = found["i"].astype(np.int64) + start
        j_idx = flat % n
        keep = ~((i_idx == j_idx) & (flat // n == zero))
        flat, i_idx, j_idx, d = flat[keep], i_idx[keep], j_idx[keep], found["v"][keep]
        close = d <= COLLISION_TOLERANCE_A
        if close.any():
            k = int(np.nonzero(close)[0][0])
            a, b = int(i_idx[k]), int(j_idx[k])
            raise CoincidentAtoms(a, b, tuple(grid[flat[k] // n] - wrap[b] + wrap[a]),
                                  float(d[k]))
        i_parts.append(i_idx)
        j_parts.append(j_idx)
        s_parts.append(offsets[flat // n] - wrap_shift[j_idx] + wrap_shift[i_idx])
    if not i_parts:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros((0, 3))
    return np.concatenate(i_parts), np.concatenate(j_parts), np.concatenate(s_parts)


class _PairCache:
    """A Verlet list built at ``cutoff + skin`` and reused while atoms move little."""

    def __init__(self, skin: float) -> None:
        self.skin = float(skin)
        self.key = None
        self.reference: Optional[np.ndarray] = None
        self.i = self.j = self.shift = None
        self.builds = 0

    def pairs(self, structure: Structure, cutoff: float):
        """Pairs within ``cutoff`` for the structure as it is now.

        A new list is stored only after a search that succeeded, so a geometry
        with coincident atoms never leaves a list behind. Coincidence is
        checked on every call, including when the stored list is reused.
        """
        pos = structure.positions
        key = (len(structure), structure.numbers.tobytes(), structure.cell.matrix.tobytes(),
               structure.cell.pbc, cutoff)
        rebuild = self.key != key or self.reference is None
        if not rebuild:
            moved = np.sqrt(np.einsum("ij,ij->i", pos - self.reference, pos - self.reference))
            rebuild = bool(moved.size) and float(moved.max()) > 0.5 * self.skin
        if rebuild:
            found = find_coincidence(pos, structure.cell)
            if found is not None:
                raise found
            self.i, self.j, self.shift = pair_search(pos, structure.cell, cutoff + self.skin)
            self.reference = pos.copy()
            self.key = key
            self.builds += 1
        D = pos[self.j] + self.shift - pos[self.i]
        d = np.sqrt(np.einsum("ij,ij->i", D, D))
        close = d <= COLLISION_TOLERANCE_A
        if close.any():
            k = int(np.nonzero(close)[0][0])
            shift = (np.rint(structure.cell.to_fractional(self.shift[k][None, :])[0])
                     if structure.cell.is_periodic else np.zeros(3))
            raise CoincidentAtoms(int(self.i[k]), int(self.j[k]), tuple(shift), float(d[k]))
        keep = d < cutoff
        return self.i[keep], self.j[keep], D[keep], d[keep]


class EAMPotential(Potential):
    """An embedded-atom potential defined entirely by one setfl file."""

    fidelity = Fidelity.TIER1_CLASSICAL

    def __init__(self, table: SetflData, identity: PotentialIdentity,
                 skin_A: float = DEFAULT_SKIN_A) -> None:
        self.table = table
        self.identity = identity
        self.name = f"eam/{identity.id}"
        self.cutoff_A = table.cutoff
        self.elements = table.elements
        self._index = {e: k for k, e in enumerate(table.elements)}
        self._F = [HermiteTable(v, table.drho) for v in table.embedding]
        self._f = [HermiteTable(v, table.dr) for v in table.density]
        self._z2r = {key: HermiteTable(v, table.dr) for key, v in table.pair.items()}
        self._cache = _PairCache(skin_A)
        self.last_evaluation: dict = {}

    def pair_table(self, a: int, b: int) -> HermiteTable:
        return self._z2r[(a, b) if a >= b else (b, a)]

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        present = sorted({pt.symbol(int(z)) for z in structure.numbers})
        missing = [e for e in present if e not in self._index]
        if missing:
            return False, (
                f"{self.name} describes {', '.join(self.elements)} only; this structure "
                f"also contains {', '.join(missing)}. An EAM file has no interactions for "
                "elements it does not list, and none are invented.")
        if all(structure.cell.pbc) and structure.cell.volume < 1e-9:
            return False, "The periodic cell has zero volume."
        return True, ""

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        out = self.evaluate(structure)
        return out["energy_eV"], out["forces_eV_A"]

    def evaluate(self, structure: Structure) -> dict:
        """Energy, forces, per-atom densities and embedding energies.

        For a cell periodic in all three directions the virial stress
        ``sigma_ab = (1 / V) dE / d(epsilon_ab)`` is returned in eV/A^3, positive
        when the crystal would lower its energy by expanding.  It is ``None``
        otherwise, because a stress needs a volume.
        """
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        n = len(structure)
        forces = np.zeros((n, 3))
        if n == 0:
            self.last_evaluation = {"n_pairs": 0, "rho_extrapolated": 0}
            return {"energy_eV": 0.0, "forces_eV_A": forces, "density": np.zeros(0),
                    "embedding_eV": np.zeros(0), "pair_energy_eV": 0.0, "stress_eV_A3": None,
                    "n_pairs": 0, "rho_extrapolated": 0}
        types = np.array([self._index[pt.symbol(int(z))] for z in structure.numbers])
        try:
            i, j, D, d = self._cache.pairs(structure, self.cutoff_A)
        except CoincidentAtoms as exc:
            raise refuse_overlap(structure, exc) from None
        n_types = len(self.elements)
        rho_j = np.zeros(d.size)
        drho_j = np.zeros(d.size)
        drho_i = np.zeros(d.size)
        for t in range(n_types):
            sel = types[j] == t
            if sel.any():
                rho_j[sel], drho_j[sel] = self._f[t](d[sel])
            sel = types[i] == t
            if sel.any():
                drho_i[sel] = self._f[t](d[sel])[1]
        rho = np.bincount(i, weights=rho_j, minlength=n)
        embed = np.zeros(n)
        dF = np.zeros(n)
        extrapolated = 0
        for t in range(n_types):
            sel = types == t
            if not sel.any():
                continue
            table = self._F[t]
            r = rho[sel]
            value, slope = table(np.minimum(r, table.x_max))
            over = r > table.x_max
            if over.any():
                extrapolated += int(over.sum())
                value = value + np.where(over, slope * (r - table.x_max), 0.0)
            embed[sel] = value
            dF[sel] = slope
        phi = np.zeros(d.size)
        dphi = np.zeros(d.size)
        for a in range(n_types):
            for b in range(n_types):
                sel = (types[i] == a) & (types[j] == b)
                if not sel.any():
                    continue
                z2r, dz2r = self.pair_table(a, b)(d[sel])
                r = d[sel]
                phi[sel] = z2r / r
                dphi[sel] = (dz2r - z2r / r) / r
        dEdr = dF[i] * drho_j + dF[j] * drho_i + dphi
        vec = (dEdr / d)[:, None] * D
        for axis in range(3):
            forces[:, axis] = np.bincount(i, weights=vec[:, axis], minlength=n)
        pair_energy = 0.5 * float(phi.sum())
        energy = float(embed.sum()) + pair_energy
        stress = None
        if all(structure.cell.pbc):
            stress = (0.5 / structure.cell.volume) * np.einsum("p,pa,pb->ab", dEdr / d, D, D)
        self.last_evaluation = {"n_pairs": int(d.size), "rho_extrapolated": extrapolated,
                                "neighbour_list_builds": self._cache.builds}
        return {"energy_eV": energy, "forces_eV_A": forces, "density": rho,
                "embedding_eV": embed, "pair_energy_eV": pair_energy,
                "stress_eV_A3": stress,
                "n_pairs": int(d.size), "rho_extrapolated": extrapolated}

    def cutoff_residuals(self) -> dict:
        """Size of each tabulated function at the cutoff: the energy step there."""
        out = {}
        for k, e in enumerate(self.elements):
            out[f"f_{e}"] = float(self._f[k](np.array([self.cutoff_A]))[0][0])
        for (a, b) in self._z2r:
            value = float(self._z2r[(a, b)](np.array([self.cutoff_A]))[0][0]) / self.cutoff_A
            out[f"phi_{self.elements[a]}-{self.elements[b]}_eV"] = value
        return out

    def describe(self) -> dict:
        ident = self.identity
        approximations = [
            "Embedded-atom method: each atom's energy depends on a pair repulsion and on "
            "the host electron density it is embedded in. The density is a sum of "
            "spherical atomic contributions; there is no angular dependence, no "
            "explicit electrons and no magnetism.",
            "Tabulated functions interpolated with the cubic Hermite scheme of LAMMPS "
            "pair_style eam/alloy; forces are the exact gradient of the interpolated "
            "energy.",
            f"Interactions end at {self.cutoff_A:.6f} A, the cutoff given in the file.",
        ]
        if ident.shipped:
            approximations.append(
                "Parameters fitted by their authors to properties of the pure elements. "
                "Agreement with those fitted properties is not evidence of accuracy for "
                "surfaces, defects, alloys or conditions outside the fit.")
        else:
            approximations.append(
                "User-supplied parameter file: its fitting data, validation and licence "
                "are unknown to Materia.")
        approximations.extend(ident.notes)
        return {
            "model": self.name,
            "fidelity": self.fidelity.value,
            "cutoff_A": self.cutoff_A,
            "parameters": {"potential": ident.as_dict(), "file_header": self.table.header(),
                           "cutoff_residuals": self.cutoff_residuals()},
            "approximations": approximations,
            "references": list(ident.citations) + REFERENCES,
        }
