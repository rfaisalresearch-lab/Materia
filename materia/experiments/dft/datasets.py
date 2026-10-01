"""Which PAW datasets GPAW will use, identified before a calculation starts.

GPAW finds a dataset by file name, ``<symbol>.<functional>`` with or without a
``.gz`` suffix, in the first directory of its setup path that holds one.
Materia repeats that search in its own process, reads the element, nuclear
charge, frozen-core and valence electron counts from the dataset header, and
fingerprints the file with SHA-256.  The experiment specification pins those
fingerprints, and the worker reports the file GPAW actually opened, so a
dataset that changed between specification and run is detected rather than
silently used.

A functional is only offered when every element of the structure has a
dataset for it.  GPAW selects datasets by the functional's own name, so a
functional without datasets of its own (PBEsol and BLYP in the standard
distribution) cannot run at all; it is refused with the missing file named,
not mapped onto another functional's datasets.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FUNCTIONALS: Tuple[str, ...] = ("LDA", "PBE", "RPBE", "revPBE", "PBEsol", "BLYP")

FUNCTIONAL_NOTES: Dict[str, str] = {
    "LDA": "Local density approximation (Perdew-Zunger parametrisation of the "
           "Ceperley-Alder electron gas). Overbinds; lattice constants typically "
           "1 to 2 percent short.",
    "PBE": "Perdew-Burke-Ernzerhof generalised-gradient functional. Underbinds "
           "slightly; lattice constants typically 1 percent long, band gaps too small.",
    "RPBE": "Hammer-Hansen-Norskov revision of PBE, fitted for adsorption energies.",
    "revPBE": "Zhang-Yang revision of PBE.",
    "PBEsol": "PBE revised for solids.",
    "BLYP": "Becke exchange with Lee-Yang-Parr correlation.",
}

_ATOM = re.compile(
    r'<atom\s+symbol="(?P<symbol>[A-Za-z]+)"\s+Z="(?P<Z>[0-9.]+)"\s+'
    r'core="(?P<core>[0-9.]+)"\s+valence="(?P<valence>[0-9.]+)"')
_GENERATOR = re.compile(r'<generator\s+type="(?P<type>[^"]*)"\s+name="(?P<name>[^"]*)"')


class DatasetError(ValueError):
    """A PAW dataset that is missing, unreadable or inconsistent."""


@dataclass(frozen=True)
class PAWDataset:
    """One PAW dataset file, as GPAW would find it."""

    symbol: str
    functional: str
    path: str
    sha256: str
    nuclear_charge: float
    core_electrons: float
    valence_electrons: float
    generator: str = ""
    relativistic: str = ""

    @property
    def file(self) -> str:
        return os.path.basename(self.path)

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "functional": self.functional,
                "path": self.path, "file": self.file, "sha256": self.sha256,
                "nuclear_charge": self.nuclear_charge,
                "core_electrons": self.core_electrons,
                "valence_electrons": self.valence_electrons,
                "generator": self.generator, "relativistic": self.relativistic}


@dataclass(frozen=True)
class BasisFile:
    """An LCAO basis file for one element."""

    symbol: str
    name: str
    path: str
    sha256: str

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "basis": self.name, "path": self.path,
                "file": os.path.basename(self.path), "sha256": self.sha256}


def _search(paths: Sequence[str], name: str) -> Optional[str]:
    for directory in paths:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
        if os.path.isfile(candidate + ".gz"):
            return candidate + ".gz"
    return None


@lru_cache(maxsize=512)
def _fingerprint(path: str, size: int, mtime: float) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_of(path: str) -> str:
    """SHA-256 of a file, cached on its size and modification time."""
    stat = os.stat(path)
    return _fingerprint(path, stat.st_size, stat.st_mtime)


def _header(path: str) -> str:
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rb") as handle:
            return handle.read(4096).decode("utf-8", errors="replace")
    except OSError as exc:
        raise DatasetError(f"The PAW dataset {path} could not be read: {exc}") from None


def find_dataset(symbol: str, functional: str, paths: Sequence[str]) -> PAWDataset:
    """The dataset GPAW would load for ``symbol`` with ``functional``.

    Raises :class:`DatasetError` naming the file that is missing, or the
    header field that does not match, so the refusal can say exactly why.
    """
    name = f"{symbol}.{functional}"
    path = _search(paths, name)
    if path is None:
        searched = ", ".join(paths) or "no directory"
        raise DatasetError(
            f"No PAW dataset {name} is installed (searched {searched}). GPAW "
            f"selects datasets by the functional's name, so {functional} cannot "
            f"run for {symbol} here; Materia will not substitute another "
            "functional's dataset.")
    text = _header(path)
    atom = _ATOM.search(text)
    if atom is None:
        raise DatasetError(f"{path} has no readable <atom> header.")
    if atom.group("symbol") != symbol:
        raise DatasetError(
            f"{path} describes {atom.group('symbol')}, not {symbol}.")
    generator = _GENERATOR.search(text)
    return PAWDataset(
        symbol=symbol, functional=functional, path=path, sha256=sha256_of(path),
        nuclear_charge=float(atom.group("Z")),
        core_electrons=float(atom.group("core")),
        valence_electrons=float(atom.group("valence")),
        generator=generator.group("name") if generator else "",
        relativistic=generator.group("type") if generator else "",
    )


def find_basis(symbol: str, basis: str, paths: Sequence[str]) -> BasisFile:
    """The LCAO basis file GPAW would load for ``symbol``."""
    name = f"{symbol}.{basis}.basis"
    path = _search(paths, name)
    if path is None:
        raise DatasetError(
            f"No LCAO basis {name} is installed, so an LCAO calculation with basis "
            f"{basis!r} cannot describe {symbol}.")
    return BasisFile(symbol=symbol, name=basis, path=path, sha256=sha256_of(path))


def available_functionals(symbols: Iterable[str], paths: Sequence[str]) -> List[str]:
    """Functionals for which every element in ``symbols`` has a dataset."""
    wanted = sorted(set(symbols))
    out = []
    for functional in FUNCTIONALS:
        if all(_search(paths, f"{symbol}.{functional}") for symbol in wanted):
            out.append(functional)
    return out


def available_bases(symbols: Iterable[str], paths: Sequence[str]) -> List[str]:
    """LCAO basis names present for every element in ``symbols``."""
    wanted = sorted(set(symbols))
    if not wanted:
        return []
    found: Dict[str, set] = {}
    for directory in paths:
        for full in glob.glob(os.path.join(directory, "*.basis*")):
            parts = os.path.basename(full).split(".")
            if parts and parts[-1] == "gz":
                parts = parts[:-1]
            if len(parts) == 3 and parts[2] == "basis":
                found.setdefault(parts[0], set()).add(parts[1])
    common = None
    for symbol in wanted:
        names = found.get(symbol, set())
        common = names if common is None else common & names
    return sorted(common or [])


def dataset_package(paths: Sequence[str]) -> str:
    """The distribution the datasets came from, when it can be identified."""
    for directory in paths:
        parent = os.path.dirname(os.path.dirname(os.path.abspath(directory)))
        for info in sorted(glob.glob(os.path.join(parent, "gpaw_data-*.dist-info"))):
            version = os.path.basename(info)[len("gpaw_data-"):-len(".dist-info")]
            return f"gpaw-data {version}"
    return "unversioned dataset directory"


@dataclass
class DatasetSelection:
    """The datasets and basis files one calculation will read."""

    functional: str
    datasets: List[PAWDataset] = field(default_factory=list)
    basis: List[BasisFile] = field(default_factory=list)
    package: str = ""

    def by_symbol(self) -> Dict[str, PAWDataset]:
        return {d.symbol: d for d in self.datasets}

    def valence_electrons(self, symbols: Sequence[str]) -> float:
        table = self.by_symbol()
        return float(sum(table[s].valence_electrons for s in symbols))

    def core_electrons(self, symbols: Sequence[str]) -> float:
        table = self.by_symbol()
        return float(sum(table[s].core_electrons for s in symbols))


def select(symbols: Sequence[str], functional: str, paths: Sequence[str],
           basis: Optional[str] = None) -> DatasetSelection:
    """Every dataset (and basis file) a calculation on ``symbols`` needs."""
    unique = sorted(set(symbols))
    chosen = DatasetSelection(functional=functional, package=dataset_package(paths))
    for symbol in unique:
        chosen.datasets.append(find_dataset(symbol, functional, paths))
        if basis:
            chosen.basis.append(find_basis(symbol, basis, paths))
    return chosen


ANGULAR = ("s", "p", "d", "f")
_STATE = re.compile(r"<state\s+([^>]*?)/?>")
_ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')


@lru_cache(maxsize=512)
def _bound_channels(path: str, size: int, mtime: float) -> Tuple[str, ...]:
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rb") as handle:
            text = handle.read(1 << 18).decode("utf-8", errors="replace")
    except OSError as exc:
        raise DatasetError(f"The PAW dataset {path} could not be read: {exc}") from None
    start = text.find("<valence_states")
    end = text.find("</valence_states>")
    if start < 0 or end < 0:
        return ()
    channels = set()
    for match in _STATE.finditer(text[start:end]):
        attributes = dict(_ATTRIBUTE.findall(match.group(1)))
        if "n" in attributes and "l" in attributes:
            channels.add(int(attributes["l"]))
    return tuple(ANGULAR[l] for l in sorted(channels) if l < len(ANGULAR))


def bound_channels(path: str) -> Tuple[str, ...]:
    """Angular channels with a bound partial wave in a dataset, e.g. ``("s", "p")``.

    GPAW projects the density of states onto the PAW projectors of bound
    partial waves only (``gpaw.dos.get_projector_numbers`` keeps ``n >= 0``),
    so these are exactly the channels a projected DOS can resolve.  Read from
    the dataset's ``<valence_states>`` block, where a bound state carries a
    principal quantum number ``n`` and an unbound one does not.
    """
    stat = os.stat(path)
    return _bound_channels(path, stat.st_size, stat.st_mtime)
