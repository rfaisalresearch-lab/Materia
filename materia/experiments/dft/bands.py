"""Kohn-Sham electronic band structure along a reciprocal-space path, through GPAW.

A :class:`BandStructureSpec` is a frozen ground-state specification, the
geometry and electronic state the bands belong to, together with the exact
path they are sampled on: the labelled special points in fractional
coordinates of the reciprocal lattice, the branches of the path, the number of
intervals in every segment, and the full list of k-points those define.  It is
immutable and versioned (:data:`SCHEMA`, :data:`VERSION`).

A standard path is generated once, with ASE's Bravais-lattice tables for the
pinned cell and periodicity, when the specification is built; the generated
coordinates, labels and the ASE version are stored in the specification and
never regenerated behind the user's back.  Asking for ``path="standard"``
again regenerates it, visibly, as a new specification.  A path generated for
one cell is refused on another.

The worker converges the ground state with the frozen settings and the pinned
symmetry choice, then runs GPAW's ``fixed_density`` on exactly the stored
k-points with symmetry switched off, so no point is folded or reordered.  It
reports the parameters it used, the k-points GPAW actually computed, the
reciprocal cell of GPAW's grid and the cumulative path distance it derived
from that cell.  Every one is compared: the k-points in order, the band
count, the labels and breaks, and the distance, which Materia recomputes from
the pinned cell.  Eigenvalues must be finite and ordered at every k-point.

Along a path the k-points carry no Brillouin-zone integration weight, so no
occupation is derived from the path itself: the energy zero is the Fermi level
of the self-consistent ground state on its own k-point grid.  Band edges and
gaps reported here are the extremes of the sampled path only.

Orbital character (fat bands) is not computed.  PAW projector weights are
projections onto bound partial waves inside the augmentation spheres; they
are not normalised orbital populations and their sum is not one, so they are
deferred rather than approximated.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...project_format.arrays import StoredArray
from ...provenance import Convergence, Fidelity, Origin, Result
from ...solvers.gpaw_driver import runner
from . import run as ground
from . import spec as specs
from .dos import _minimal, _occupied_bands, electronic_digest

SCHEMA = "materia.dft.band-structure"
VERSION = "1.0"
MODEL = "external:gpaw/band-structure"

REFERENCES: Tuple[str, ...] = ("fermi-level", "absolute")
SYMMETRY: Tuple[str, ...] = ("preserve", "off")
SYMMETRY_PARAMETERS = {
    "preserve": {"point_group": True, "time_reversal": True},
    "off": {"point_group": False, "time_reversal": False},
}
SOURCES: Tuple[str, ...] = ("structure", "ground-state", "relaxation")

PATH_CITATIONS: Tuple[str, ...] = (
    "W. Setyawan and S. Curtarolo, Comput. Mater. Sci. 49 (2010) 299",
    "A. H. Larsen et al., J. Phys.: Condens. Matter 29 (2017) 273002",
)

MAX_KPOINTS = 1000
MAX_INTERVALS = 400
MAX_BANDS = 2000
MAX_EXTRA_BANDS = 200
MAX_SPECIAL_POINTS = 64
MIN_DENSITY = 0.5
MAX_DENSITY = 200.0
MAX_FRACTION = 1.0
DEFAULT_DENSITY = 10.0
PATH_TOLERANCE = 1e-10
DISTANCE_TOLERANCE = 1e-8
LABEL_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9']{0,7}$")
TOKEN_PATTERN = re.compile(r"[A-Z][a-z0-9']*")

STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

FIELDS: Tuple[specs.FieldInfo, ...] = (
    specs.FieldInfo("source", "bands", "geometry source", "",
                    "Where the geometry and electronic settings come from: a structure, a "
                    "stored converged ground-state run, or a stored converged relaxation.",
                    settable=False),
    specs.FieldInfo("energy_reference", "bands", "energy zero", "",
                    "fermi-level puts zero at the Fermi level of the self-consistent ground "
                    "state. Absolute Kohn-Sham energies of a periodic cell have an arbitrary "
                    "zero, so they are refused."),
    specs.FieldInfo("scf_symmetry", "bands", "ground-state symmetry", "",
                    "preserve lets GPAW reduce the ground-state k-point grid by point-group "
                    "and time-reversal symmetry; off uses the full grid. The path itself is "
                    "always computed with symmetry off, point by point."),
    specs.FieldInfo("path", "bands", "path", "",
                    "Branches of labelled special points, such as GXWKGLUWLK,UX. A comma "
                    "starts a new branch: the plot jumps there without distance. Type "
                    "standard to regenerate the standard path of this lattice."),
    specs.FieldInfo("special_points", "bands", "special points", "fractional",
                    "Label and fractional coordinates of each special point, in the basis of "
                    "the reciprocal lattice of this cell. Generated once from ASE's "
                    "Bravais-lattice tables and stored exactly.", advanced=True),
    specs.FieldInfo("path_origin", "bands", "path origin", "",
                    "How the special points were obtained: generator, lattice, ASE version "
                    "and the cell they were generated for.", settable=False),
    specs.FieldInfo("sampling_density_per_invA", "bands", "sampling density", "points/(1/A)",
                    "Intervals per 1/A of path length (2 pi included) used to choose the "
                    "intervals of each segment. Empty when the intervals were set one by "
                    "one."),
    specs.FieldInfo("segment_intervals", "bands", "segment intervals", "intervals",
                    "Number of equal steps along each segment, in path order. Each segment "
                    "includes both of its special points.", advanced=True),
    specs.FieldInfo("kpoints_frac", "bands", "path k-points", "fractional",
                    "Every k-point of the path, in order, as stored. GPAW computes exactly "
                    "these.", settable=False),
    specs.FieldInfo("n_bands", "bands", "bands", "bands",
                    "Kohn-Sham bands per spin returned at every k-point, all converged. At "
                    "least the occupied bands."),
    specs.FieldInfo("extra_bands", "bands", "buffer bands", "bands",
                    "Additional bands computed but not converged or kept. They help the "
                    "eigensolver converge the top requested band.", advanced=True),
)
FIELD_INDEX = {f.name: f for f in FIELDS}
SETTABLE: Tuple[str, ...] = tuple(f.name for f in FIELDS if f.settable)


def reciprocal_cell(cell: Sequence[Sequence[float]]) -> np.ndarray:
    """Rows are the reciprocal lattice vectors in 1/A, with the factor 2 pi."""
    return 2.0 * math.pi * np.linalg.inv(np.asarray(cell, dtype=float)).T


def path_kpoints(branches: Sequence[Sequence[str]], points: Mapping[str, Sequence[float]],
                 intervals: Sequence[int]) -> Tuple[np.ndarray, List[str], List[int]]:
    """The k-points, the label of each (or ''), and the indices that start a new branch."""
    kpts: List[np.ndarray] = []
    labels: List[str] = []
    breaks: List[int] = []
    segment = 0
    for b, branch in enumerate(branches):
        if b:
            breaks.append(len(kpts))
        start = np.asarray(points[branch[0]], dtype=float)
        kpts.append(start)
        labels.append(branch[0])
        for end_label in branch[1:]:
            end = np.asarray(points[end_label], dtype=float)
            n = int(intervals[segment])
            for t in range(1, n + 1):
                kpts.append(start + (end - start) * (t / n))
                labels.append(end_label if t == n else "")
            start = end
            segment += 1
    return np.asarray(kpts, dtype=float).reshape(-1, 3), labels, breaks


def path_distance(kpoints: np.ndarray, cell: Sequence[Sequence[float]],
                  breaks: Sequence[int]) -> np.ndarray:
    """Cumulative distance along the path in 1/A; no distance is added across a break."""
    cart = np.asarray(kpoints, dtype=float) @ reciprocal_cell(cell)
    steps = np.linalg.norm(np.diff(cart, axis=0), axis=1) if len(cart) > 1 else np.zeros(0)
    for index in breaks:
        if 0 < index <= len(steps):
            steps[index - 1] = 0.0
    return np.concatenate([[0.0], np.cumsum(steps)])


def parse_path(value: Any) -> Tuple[Tuple[str, ...], ...]:
    """``'GXW,UX'`` or ``[['G', 'X', 'W'], ['U', 'X']]`` as branches of labels."""
    if isinstance(value, str):
        text = value.replace(" ", "")
        if not text:
            raise specs.SpecError("The path is empty.")
        out = []
        for chunk in text.split(","):
            tokens = TOKEN_PATTERN.findall(chunk)
            if "".join(tokens) != chunk or not tokens:
                raise specs.SpecError(
                    f"Cannot read path branch {chunk!r}: labels start with a capital letter "
                    "and may continue with lowercase letters or digits, such as G, X, K1.")
            out.append(tuple(tokens))
        return tuple(out)
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(b, (list, tuple)) for b in value):
        out = []
        for branch in value:
            if not all(isinstance(label, str) for label in branch):
                raise specs.SpecError("Every path label must be text.")
            out.append(tuple(str(label) for label in branch))
        return tuple(out)
    raise specs.SpecError(f"path must be text such as 'GXWKGL,UX' or a list of branches, "
                          f"got {value!r}.")


def path_text(branches: Sequence[Sequence[str]]) -> str:
    return ",".join("".join(branch) for branch in branches)


def _coerce_points(value: Any) -> Tuple[Tuple[str, Tuple[float, float, float]], ...]:
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise specs.SpecError("special_points must map labels to three fractional "
                                      "coordinates.")
            items.append((item[0], item[1]))
    else:
        raise specs.SpecError("special_points must map labels to three fractional "
                              f"coordinates, got {value!r}.")
    out = []
    for label, coords in items:
        if not isinstance(label, str):
            raise specs.SpecError(f"A special-point label must be text, got {label!r}.")
        if isinstance(coords, (str, bytes)) or not isinstance(coords, (list, tuple, np.ndarray)) \
                or len(coords) != 3:
            raise specs.SpecError(f"Special point {label} needs three fractional coordinates.")
        out.append((label, tuple(specs._number(f"special point {label}", c, "fractional")
                                 for c in coords)))
    return tuple(sorted(out))


def standard_path(cell: Sequence[Sequence[float]], pbc: Sequence[bool]):
    """ASE's standard special points and path for this cell, or the reason there are none."""
    pbc = tuple(bool(p) for p in pbc)
    origin: Dict[str, Any] = {"generator": "ase", "cell_A": np.asarray(cell, float).tolist(),
                              "pbc": list(pbc)}
    if not any(pbc):
        return (), (), {**origin, "generator": "none",
                        "reason": "A finite cluster has no Brillouin zone."}
    try:
        import ase
        from ase.cell import Cell
    except ImportError:
        return (), (), {**origin, "generator": "none",
                        "reason": "ASE is not installed in Materia's interpreter, so no "
                                  "standard path can be generated. Give special_points and "
                                  "path explicitly, or install ase."}
    try:
        ase_cell = Cell(np.asarray(cell, dtype=float))
        lattice = ase_cell.get_bravais_lattice(pbc=pbc)
        bandpath = ase_cell.bandpath(pbc=pbc, npoints=0)
        points = {str(k): tuple(float(x) for x in v)
                  for k, v in bandpath.special_points.items()}
        branches = parse_path(bandpath.path)
    except Exception as exc:
        return (), (), {**origin, "generator": "none",
                        "reason": f"ASE could not identify a Bravais lattice for this cell "
                                  f"and periodicity ({type(exc).__name__}: {exc}). Give "
                                  "special_points and path explicitly."}
    used = {label for branch in branches for label in branch}
    special = tuple(sorted((k, v) for k, v in points.items() if k in used))
    origin.update({"ase_version": str(getattr(ase, "__version__", "")),
                   "lattice": str(lattice.name), "lattice_description": str(lattice),
                   "path": path_text(branches),
                   "convention": "Setyawan and Curtarolo special points, as tabulated by "
                                 "ASE, transformed to this cell's reciprocal basis"})
    return special, branches, origin


@dataclass(frozen=True)
class BandStructureSpec:
    """One band-structure calculation, frozen.  See :data:`FIELDS`."""

    ground_state: specs.GroundStateSpec
    source: Dict[str, Any]
    energy_reference: str
    scf_symmetry: str
    path: Tuple[Tuple[str, ...], ...]
    special_points: Tuple[Tuple[str, Tuple[float, float, float]], ...]
    path_origin: Dict[str, Any]
    sampling_density_per_invA: Optional[float]
    segment_intervals: Tuple[int, ...]
    kpoints_frac: Tuple[Tuple[float, float, float], ...]
    n_bands: int
    extra_bands: int
    schema: str = SCHEMA
    version: str = VERSION

    def as_dict(self) -> dict:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "ground_state":
                out[f.name] = value.as_dict()
            elif f.name in ("source", "path_origin"):
                out[f.name] = json.loads(json.dumps(value))
            else:
                out[f.name] = specs._plain(value)
        return out

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "BandStructureSpec":
        if data.get("schema") != SCHEMA:
            raise specs.SpecError(f"Not a band-structure specification: {data.get('schema')!r}.")
        if str(data.get("version")) != VERSION:
            raise specs.SpecError(f"Band-structure specification version {data.get('version')} "
                                  f"is not understood by this Materia (it reads {VERSION}).")
        names = {f.name for f in fields(BandStructureSpec)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise specs.SpecError(f"Unknown band-structure field(s): {', '.join(unknown)}.")
        missing = sorted(n for n in names - set(data) if n not in ("schema", "version"))
        if missing:
            raise specs.SpecError(f"Missing band-structure field(s): {', '.join(missing)}.")
        values = dict(data)
        values["ground_state"] = specs.GroundStateSpec.from_dict(data["ground_state"])
        values["source"] = dict(data["source"])
        values["path_origin"] = dict(data["path_origin"])
        values["path"] = tuple(tuple(str(x) for x in branch) for branch in data["path"])
        values["special_points"] = tuple((str(label), tuple(float(x) for x in coords))
                                         for label, coords in data["special_points"])
        values["segment_intervals"] = tuple(int(n) for n in data["segment_intervals"])
        values["kpoints_frac"] = tuple(tuple(float(x) for x in k) for k in data["kpoints_frac"])
        density = data["sampling_density_per_invA"]
        values["sampling_density_per_invA"] = None if density is None else float(density)
        values["n_bands"] = int(data["n_bands"])
        values["extra_bands"] = int(data["extra_bands"])
        values["schema"] = SCHEMA
        values["version"] = VERSION
        return BandStructureSpec(**values)

    @property
    def digest(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def short_digest(self) -> str:
        return self.digest[:16]

    @property
    def structure_key(self) -> Optional[str]:
        return self.ground_state.structure_key

    @property
    def geometry_digest(self) -> str:
        return self.ground_state.geometry_digest

    @property
    def n_kpoints(self) -> int:
        return len(self.kpoints_frac)

    @property
    def n_spins(self) -> int:
        return 2 if self.ground_state.spin_polarized else 1

    def points(self) -> Dict[str, Tuple[float, float, float]]:
        return {label: coords for label, coords in self.special_points}

    def segments(self) -> List[Tuple[int, str, str]]:
        return [(b, branch[i], branch[i + 1]) for b, branch in enumerate(self.path)
                for i in range(len(branch) - 1)]

    def layout(self) -> Tuple[np.ndarray, List[str], List[int]]:
        """The path its special points and intervals define, recomputed."""
        return path_kpoints(self.path, self.points(), self.segment_intervals)

    def labels(self) -> List[str]:
        return self.layout()[1]

    def breaks(self) -> List[int]:
        return self.layout()[2]

    def distance(self) -> np.ndarray:
        return path_distance(np.asarray(self.kpoints_frac, dtype=float),
                             self.ground_state.cell_A, self.breaks())

    def ticks(self) -> List[Dict[str, Any]]:
        """Labelled positions on the distance axis; a break joins two labels as 'K|U'."""
        labels = self.labels()
        breaks = set(self.breaks())
        x = self.distance()
        out: List[Dict[str, Any]] = []
        for i, label in enumerate(labels):
            if not label:
                continue
            if i in breaks and out:
                out[-1] = {**out[-1], "label": f"{out[-1]['label']}|{label}", "break": True,
                           "indices": out[-1]["indices"] + [i]}
                continue
            out.append({"label": label, "distance_invA": float(x[i]), "indices": [i],
                        "break": False})
        return out

    def settings(self) -> Dict[str, Any]:
        return {"energy_reference": self.energy_reference, "scf_symmetry": self.scf_symmetry,
                "path": path_text(self.path),
                "special_points": {label: list(coords) for label, coords in self.special_points},
                "sampling_density_per_invA": self.sampling_density_per_invA,
                "segment_intervals": list(self.segment_intervals),
                "n_bands": self.n_bands, "extra_bands": self.extra_bands}


def _coerce(name: str, value: Any) -> Any:
    if name == "energy_reference":
        return specs._choice(name, value, REFERENCES)
    if name == "scf_symmetry":
        return specs._choice(name, value, SYMMETRY)
    if name == "path":
        if value == "standard":
            return "standard"
        return parse_path(value)
    if name == "special_points":
        return _coerce_points(value)
    if name == "sampling_density_per_invA":
        return specs._number(name, value, "points per 1/A")
    if name == "segment_intervals":
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise specs.SpecError("segment_intervals must be a list of whole numbers, one per "
                                  "segment.")
        return tuple(specs._integer(name, v) for v in value)
    if name in ("n_bands", "extra_bands"):
        return specs._integer(name, value)
    raise specs.SpecError(f"{name} is not a band-structure variable.")


def _split(variables: Mapping[str, Any]):
    band_vars = {k: v for k, v in variables.items() if k in SETTABLE}
    rest = {k: v for k, v in variables.items() if k not in SETTABLE}
    for name in ("source", "path_origin", "kpoints_frac"):
        if name in rest:
            raise specs.SpecError(f"{name} is not settable; it follows from the geometry "
                                  "source and the path.")
    return band_vars, rest


def _intervals(branches, points, cell, density: float) -> Tuple[int, ...]:
    recip = reciprocal_cell(cell)
    out = []
    for branch in branches:
        for a, b in zip(branch[:-1], branch[1:]):
            if a not in points or b not in points:
                out.append(1)
                continue
            length = float(np.linalg.norm((np.asarray(points[b]) - np.asarray(points[a])) @ recip))
            out.append(max(1, int(round(length * density))))
    return tuple(out)


def _assemble(gs: specs.GroundStateSpec, source: Dict[str, Any], values: Dict[str, Any],
              base: Optional[BandStructureSpec], environment) -> BandStructureSpec:
    """Resolve the path, then the k-points, from the values given and the previous spec."""
    regenerate = values.get("path") == "standard" or base is None
    if regenerate and "special_points" not in values:
        special, branches, origin = standard_path(gs.cell_A, gs.pbc)
    elif "special_points" in values:
        special = values["special_points"]
        branches = base.path if base is not None else ()
        origin = {"generator": "user", "cell_A": [list(r) for r in gs.cell_A],
                  "pbc": list(gs.pbc)}
    else:
        special, branches, origin = base.special_points, base.path, dict(base.path_origin)
    if isinstance(values.get("path"), tuple):
        branches = values["path"]
        if origin.get("generator") == "ase" and path_text(branches) != origin.get("path"):
            origin = {**origin, "path_chosen": path_text(branches)}
    elif values.get("path") == "standard" and "special_points" in values:
        _, branches, _ = standard_path(gs.cell_A, gs.pbc)
    points = dict(special)
    n_segments = sum(max(0, len(b) - 1) for b in branches)
    if "segment_intervals" in values:
        intervals = tuple(values["segment_intervals"])
        density = None
    else:
        density = values.get("sampling_density_per_invA")
        if density is None:
            density = base.sampling_density_per_invA if (
                base is not None and base.sampling_density_per_invA is not None) \
                else DEFAULT_DENSITY
        keep = (base is not None and base.sampling_density_per_invA is None
                and "sampling_density_per_invA" not in values
                and base.path == branches and base.special_points == special
                and len(base.segment_intervals) == n_segments)
        if keep:
            intervals, density = base.segment_intervals, None
        else:
            intervals = _intervals(branches, points, gs.cell_A, float(density))
    usable = (all(label in points for branch in branches for label in branch)
              and len(intervals) == n_segments and all(n >= 1 for n in intervals)
              and sum(intervals) + len(branches) <= 10 * MAX_KPOINTS)
    if usable and branches:
        kpts, _, _ = path_kpoints(branches, points, intervals)
        kpoints = tuple(tuple(float(x) for x in k) for k in kpts)
    else:
        kpoints = ()
    occupied = _occupied_bands(gs, environment)
    if occupied is None:
        occupied = int(math.ceil(sum(gs.numbers) / 2.0))
    if "n_bands" in values:
        n_bands = values["n_bands"]
    elif base is not None:
        n_bands = base.n_bands
    else:
        n_bands = occupied + max(4, int(math.ceil(0.5 * occupied)))
    if "extra_bands" in values:
        extra = values["extra_bands"]
    elif base is not None and "n_bands" not in values:
        extra = base.extra_bands
    else:
        extra = max(4, int(math.ceil(0.25 * n_bands)))
    if "scf_symmetry" in values:
        symmetry = values["scf_symmetry"]
    elif base is not None:
        symmetry = base.scf_symmetry
    else:
        symmetry = source.pop("default_symmetry", "preserve")
    source.pop("default_symmetry", None)
    reference = values.get("energy_reference",
                           base.energy_reference if base is not None else "fermi-level")
    return BandStructureSpec(
        ground_state=gs, source=dict(source), energy_reference=reference,
        scf_symmetry=symmetry, path=tuple(tuple(b) for b in branches),
        special_points=tuple(special), path_origin=origin,
        sampling_density_per_invA=None if density is None else float(density),
        segment_intervals=tuple(int(n) for n in intervals), kpoints_frac=kpoints,
        n_bands=int(n_bands), extra_bands=int(extra))


def build(structure, environment=None, structure_key: Optional[str] = None,
          **variables: Any) -> BandStructureSpec:
    """A band-structure specification for ``structure``: its own ground state, then the path.

    Ground-state variables are those of :func:`materia.experiments.dft.spec.build`;
    band-structure variables are :data:`SETTABLE`.  The standard path of the
    lattice is generated here, once, unless ``special_points`` and ``path``
    are given.  Not validated here; :func:`check` does that.
    """
    band_vars, rest = _split(variables)
    environment = specs._environment(environment)
    rest.pop("observables", None)
    gs = _minimal(specs.build(structure, environment, structure_key=structure_key, **rest))
    source = {"kind": "structure", "run_id": None, "spec_digest": None,
              "geometry_digest": gs.geometry_digest,
              "electronic_digest": electronic_digest(gs), "energy_eV": None}
    values = {name: _coerce(name, value) for name, value in band_vars.items()}
    return _assemble(gs, source, values, None, environment)


def build_from_run(project, kind: str, run_id: str, environment=None,
                   **variables: Any) -> BandStructureSpec:
    """A band-structure specification on the exact geometry and settings of a stored run."""
    from . import dos, records

    band_vars, rest = _split(variables)
    if rest:
        raise specs.SpecError(
            f"{', '.join(sorted(rest))}: a band structure built from a stored {kind} run keeps "
            "that run's geometry and electronic settings exactly. Build it from the structure "
            "to change them.")
    environment = specs._environment(environment)
    original, energy = dos.source_spec(project, kind, run_id)
    gs = _minimal(original)
    source: Dict[str, Any] = {"kind": kind, "run_id": run_id, "spec_digest": original.digest,
                              "geometry_digest": original.geometry_digest,
                              "electronic_digest": electronic_digest(original),
                              "energy_eV": energy}
    if kind == "relaxation":
        relaxed = records.relax_spec_of(project, run_id)
        if relaxed is not None:
            source["default_symmetry"] = relaxed.symmetry
    values = {name: _coerce(name, value) for name, value in band_vars.items()}
    return _assemble(gs, source, values, None, environment)


def changed(spec: BandStructureSpec, environment=None, **variables: Any) -> BandStructureSpec:
    """A new specification.  The stored path is kept unless the path variables change."""
    band_vars, rest = _split(variables)
    environment = specs._environment(environment)
    values = {name: _coerce(name, value) for name, value in band_vars.items()}
    gs, source = spec.ground_state, dict(spec.source)
    if rest:
        if spec.source.get("kind") != "structure":
            raise specs.SpecError(f"{', '.join(sorted(rest))}: this band structure keeps the "
                                  f"electronic settings of {spec.source.get('kind')} run "
                                  f"{spec.source.get('run_id')}.")
        rest.pop("observables", None)
        gs = _minimal(specs.changed(spec.ground_state, environment, **rest))
        source = dict(spec.source, electronic_digest=electronic_digest(gs),
                      geometry_digest=gs.geometry_digest)
    return _assemble(gs, source, values, spec, environment)


def check(spec: BandStructureSpec, environment=None) -> specs.SpecReport:
    """Every ground-state rule plus every band-structure rule, keyed by variable."""
    environment = specs._environment(environment)
    gs = spec.ground_state
    report = specs.check(gs, environment)
    by_field = {k: list(v) for k, v in report.by_field.items()}
    warnings = list(report.warnings)
    general = [t for t in report.blocking
               if not any(t in texts for texts in report.by_field.values())]

    def refuse(name: str, text: str) -> None:
        by_field.setdefault(name, []).append(text)

    if spec.schema != SCHEMA or spec.version != VERSION:
        general.append(f"Band-structure specification {spec.schema} {spec.version} is not "
                       f"{SCHEMA} {VERSION}.")
    source = spec.source
    if source.get("kind") not in SOURCES:
        refuse("source", f"The geometry source must be one of {', '.join(SOURCES)}.")
    if source.get("geometry_digest") != gs.geometry_digest:
        refuse("source", "The geometry differs from the source's geometry.")
    if source.get("electronic_digest") != electronic_digest(gs):
        refuse("source", f"The electronic settings differ from those of the "
                         f"{source.get('kind')} source, so the bands would not describe it.")
    if spec.energy_reference not in REFERENCES:
        refuse("energy_reference", f"energy_reference must be one of {', '.join(REFERENCES)}.")
    elif spec.energy_reference == "absolute":
        refuse("energy_reference", "The Kohn-Sham energies of a periodic cell are measured "
                                   "from the average electrostatic potential, an arbitrary "
                                   "zero, not the vacuum. Use fermi-level.")
    if spec.scf_symmetry not in SYMMETRY:
        refuse("scf_symmetry", f"scf_symmetry must be one of {', '.join(SYMMETRY)}.")
    periodic = [i for i in range(3) if gs.pbc[i]]
    if gs.boundary == "cluster" or not periodic:
        refuse("path", "A finite cluster has no Brillouin zone and no band dispersion: its "
                       "Kohn-Sham levels are discrete. A path through k-space would only "
                       "repeat the Gamma-point levels. Use the ground-state eigenvalues or "
                       "the DOS of the cluster instead.")
    else:
        _check_path(spec, gs, periodic, refuse, warnings)
    occupied = _occupied_bands(gs, environment)
    if not (1 <= spec.n_bands <= MAX_BANDS):
        refuse("n_bands", f"n_bands must be 1 to {MAX_BANDS}.")
    elif occupied is not None and spec.n_bands < occupied:
        refuse("n_bands", f"{spec.n_bands} bands cannot hold the occupied states; at least "
                          f"{occupied} are occupied.")
    elif occupied is not None and spec.n_bands == occupied:
        warnings.append("No empty band is requested, so the plot will show no conduction "
                        "band and no gap.")
    if not (0 <= spec.extra_bands <= MAX_EXTRA_BANDS):
        refuse("extra_bands", f"extra_bands must be 0 to {MAX_EXTRA_BANDS}.")
    elif spec.extra_bands == 0:
        warnings.append("With no buffer bands the highest requested band converges slowly "
                        "and may not converge at all.")
    if abs(gs.charge_e) > 1e-9 and gs.boundary == "bulk":
        warnings.append("A charged cell with a uniform background shifts every level by a "
                        "finite-size term; only energies relative to the Fermi level are "
                        "comparable.")
    if gs.boundary in ("slab", "wire"):
        warnings.append(f"A {gs.boundary}: the path runs only along the periodic directions. "
                        "Energies are relative to the Fermi level, not to the vacuum level.")
    if gs.spin_polarized:
        warnings.append("Spin-polarised: both spin channels are computed on the same path and "
                        "share one Fermi level.")
    warnings.append("Band edges and gaps are read along the sampled path only; the true "
                    "extremes can lie off the path or between sampled points.")
    blocking = list(general)
    for texts in by_field.values():
        for text in texts:
            if text not in blocking:
                blocking.append(text)
    options = dict(report.options)
    options.update({
        "references": ["fermi-level"], "scf_symmetry": list(SYMMETRY),
        "occupied_bands": occupied, "boundary": gs.boundary,
        "standard_path": spec.path_origin.get("path"),
        "lattice": spec.path_origin.get("lattice"),
        "n_kpoints": spec.n_kpoints, "n_segments": len(spec.segments()),
        "path_length_invA": float(spec.distance()[-1]) if spec.n_kpoints and not by_field.get(
            "path") and not by_field.get("kpoints_frac") else None,
        "orbital_character": "deferred: PAW projector weights are not normalised orbital "
                             "populations, so fat bands are not offered",
    })
    return specs.SpecReport(blocking, warnings, by_field, report.boundary, report.electrons,
                            options)


def _check_path(spec: BandStructureSpec, gs: specs.GroundStateSpec, periodic: List[int],
                refuse: Callable[[str, str], None], warnings: List[str]) -> None:
    origin = spec.path_origin
    if not spec.path:
        refuse("path", "No path is defined. " + str(origin.get("reason") or
                                                   "Give special_points and path."))
        return
    if origin.get("generator") not in ("ase", "user"):
        refuse("path_origin", "The special points have no recorded origin.")
    generated = origin.get("cell_A")
    if generated is None or np.asarray(generated, dtype=float).shape != (3, 3) or \
            not np.allclose(np.asarray(generated, dtype=float),
                            np.asarray(gs.cell_A, dtype=float), rtol=0, atol=1e-9) or \
            [bool(p) for p in origin.get("pbc", [])] != list(gs.pbc):
        refuse("path_origin", "The path was generated for a different cell or periodicity. "
                              "Its labels would not name the symmetry points of this lattice. "
                              "Set path to standard to regenerate it, or give special_points.")
    points = spec.points()
    if len(spec.special_points) > MAX_SPECIAL_POINTS:
        refuse("special_points", f"More than {MAX_SPECIAL_POINTS} special points.")
    if len({label for label, _ in spec.special_points}) != len(spec.special_points):
        refuse("special_points", "Two special points share a label.")
    for label, coords in spec.special_points:
        if not LABEL_PATTERN.match(label):
            refuse("special_points", f"Label {label!r} is not a capital letter followed by at "
                                     "most seven letters, digits or primes.")
        values = np.asarray(coords, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            refuse("special_points", f"Special point {label} is not three finite numbers.")
            continue
        if np.any(np.abs(values) > MAX_FRACTION + 1e-12):
            refuse("special_points", f"Special point {label} has a fractional coordinate "
                                     f"beyond {MAX_FRACTION:g}; use its equivalent in the first "
                                     "zone.")
        for axis in range(3):
            if axis not in periodic and abs(values[axis]) > 1e-12:
                refuse("special_points", f"Special point {label} has a component along "
                                         f"{'abc'[axis]}, which is open. There is no Bloch "
                                         "wavevector along an open direction.")
    for branch in spec.path:
        if len(branch) < 2:
            refuse("path", f"Branch {''.join(branch)} has one point and no segment.")
        for label in branch:
            if label not in points:
                refuse("path", f"Path label {label} is not a special point. Known: "
                               f"{', '.join(sorted(points)) or 'none'}.")
        for a, b in zip(branch[:-1], branch[1:]):
            if a in points and b in points and np.allclose(points[a], points[b], atol=1e-12):
                refuse("path", f"Segment {a} to {b} has zero length.")
    n_segments = len(spec.segments())
    if len(spec.segment_intervals) != n_segments:
        refuse("segment_intervals", f"{len(spec.segment_intervals)} interval counts for "
                                    f"{n_segments} segments.")
    for n in spec.segment_intervals:
        if not (1 <= n <= MAX_INTERVALS):
            refuse("segment_intervals", f"Each segment needs 1 to {MAX_INTERVALS} intervals, "
                                        f"got {n}.")
            break
    density = spec.sampling_density_per_invA
    if density is not None and not (MIN_DENSITY <= density <= MAX_DENSITY):
        refuse("sampling_density_per_invA", f"The sampling density must be {MIN_DENSITY:g} to "
                                            f"{MAX_DENSITY:g} points per 1/A.")
    if spec.n_kpoints > MAX_KPOINTS:
        refuse("segment_intervals", f"{spec.n_kpoints} k-points exceed the {MAX_KPOINTS} "
                                    "allowed; lower the sampling density.")
    try:
        expected, _, _ = spec.layout()
    except (KeyError, IndexError, ValueError):
        return
    stored = np.asarray(spec.kpoints_frac, dtype=float).reshape(-1, 3)
    if stored.shape != expected.shape or not np.allclose(stored, expected, rtol=0,
                                                          atol=PATH_TOLERANCE):
        refuse("kpoints_frac", "The stored k-points are not the path that the special points, "
                               "branches and segment intervals define, in that order.")
    elif not np.all(np.isfinite(stored)):
        refuse("kpoints_frac", "A stored k-point is not finite.")
    if spec.n_kpoints and spec.n_kpoints < 2 * n_segments:
        warnings.append("Some segments have a single interval: the band dispersion between "
                        "their end points is drawn as a straight line.")


def require(spec: BandStructureSpec, environment=None) -> specs.SpecReport:
    report = check(spec, environment)
    if not report.ok:
        raise specs.SpecRefused(report)
    return report


def gpaw_parameters(spec: BandStructureSpec) -> Dict[str, Any]:
    """The keyword arguments handed to ``gpaw.GPAW`` for the ground state."""
    parameters = specs.gpaw_parameters(spec.ground_state)
    parameters["symmetry"] = dict(SYMMETRY_PARAMETERS[spec.scf_symmetry])
    return parameters


def nscf_parameters(spec: BandStructureSpec) -> Dict[str, Any]:
    """The keyword arguments handed to ``GPAW.fixed_density``."""
    return {"kpts": [list(k) for k in spec.kpoints_frac],
            "nbands": int(spec.n_bands + spec.extra_bands), "symmetry": "off",
            "convergence": {"bands": int(spec.n_bands)}}


def band_settings(spec: BandStructureSpec) -> Dict[str, Any]:
    """The band-structure block of the worker job."""
    return {"nscf": nscf_parameters(spec), "reference": spec.energy_reference,
            "n_bands": int(spec.n_bands), "labels": spec.labels(), "breaks": spec.breaks()}


def worker_job(spec: BandStructureSpec) -> Dict[str, Any]:
    gs = spec.ground_state
    return {"structure": specs.worker_structure(gs), "parameters": gpaw_parameters(spec),
            "observables": list(gs.observables),
            "expected_datasets": {s: {"path": p, "sha256": h} for s, p, h in gs.paw_datasets},
            "bands": band_settings(spec)}


def describe(spec: BandStructureSpec, report: Optional[specs.SpecReport] = None) -> dict:
    report = report if report is not None else check(spec)
    items = []
    for info in FIELDS:
        item = info.as_dict()
        value = getattr(spec, info.name)
        if info.name in ("source", "path_origin"):
            item["value"] = json.loads(json.dumps(value))
        elif info.name == "path":
            item["value"] = path_text(value)
        else:
            item["value"] = specs._plain(value)
        item["problems"] = list(report.by_field.get(info.name, []))
        items.append(item)
    try:
        ticks = spec.ticks() if spec.n_kpoints and "kpoints_frac" not in report.by_field else []
    except (KeyError, IndexError, ValueError):
        ticks = []
    return {"schema": spec.schema, "version": spec.version, "digest": spec.digest,
            "short_digest": spec.short_digest, "bands": items, "settings": spec.settings(),
            "source": dict(spec.source), "n_kpoints": spec.n_kpoints,
            "segments": [{"branch": b, "from": a, "to": c, "intervals": n}
                         for (b, a, c), n in zip(spec.segments(), spec.segment_intervals)],
            "ticks": ticks, "path_origin": dict(spec.path_origin),
            "ground_state": specs.describe(spec.ground_state, report),
            "gpaw_parameters": gpaw_parameters(spec), "nscf_parameters": nscf_parameters(spec)}


def band_edges(eigen: np.ndarray, fermi: float, labels: Sequence[str],
               kpoints: np.ndarray) -> Dict[str, Any]:
    """Band edges and gaps along the path, per spin and together, relative to the Fermi level.

    A band counts as occupied when it lies below the Fermi level at every
    sampled k-point and empty when it lies above it at every one.  If a band
    crosses the Fermi level on the path, the system is metallic along the
    path and no gap is reported.
    """
    def point(k: int) -> Dict[str, Any]:
        return {"index": int(k), "label": labels[k] or None,
                "kpoint_frac": [float(x) for x in kpoints[k]]}

    per_spin = []
    vbm_all, cbm_all = [], []
    for s in range(eigen.shape[0]):
        e = eigen[s] - fermi
        below = np.all(e < 0.0, axis=0)
        above = np.all(e > 0.0, axis=0)
        crossing = [int(n) for n in np.where(~below & ~above)[0]]
        entry: Dict[str, Any] = {"spin": s, "crossing_bands": crossing,
                                 "occupied_bands": int(below.sum())}
        if crossing:
            entry["metallic_on_path"] = True
        else:
            entry["metallic_on_path"] = False
            n_occ = int(below.sum())
            if n_occ >= 1:
                v = e[:, n_occ - 1]
                k = int(np.argmax(v))
                entry["vbm"] = {"energy_eV": float(v[k]), **point(k)}
                vbm_all.append((float(v[k]), s, k))
            if n_occ < e.shape[1]:
                c = e[:, n_occ]
                k = int(np.argmin(c))
                entry["cbm"] = {"energy_eV": float(c[k]), **point(k)}
                cbm_all.append((float(c[k]), s, k))
            if 1 <= n_occ < e.shape[1]:
                direct = e[:, n_occ] - e[:, n_occ - 1]
                kd = int(np.argmin(direct))
                entry["gap_eV"] = entry["cbm"]["energy_eV"] - entry["vbm"]["energy_eV"]
                entry["direct_gap_eV"] = float(direct[kd])
                entry["direct_gap_at"] = point(kd)
                entry["gap_kind"] = "direct" if entry["cbm"]["index"] == entry["vbm"]["index"] \
                    else "indirect"
        per_spin.append(entry)
    out: Dict[str, Any] = {"per_spin": per_spin, "reference": "self-consistent Fermi level",
                           "scope": "extremes over the sampled path points only"}
    metallic = any(p["metallic_on_path"] for p in per_spin)
    out["metallic_on_path"] = metallic
    if not metallic and vbm_all and cbm_all:
        v, vs, vk = max(vbm_all)
        c, cs, ck = min(cbm_all)
        out.update({"gap_eV": c - v, "vbm_eV": v, "cbm_eV": c, "vbm_spin": vs, "cbm_spin": cs,
                    "vbm": point(vk), "cbm": point(ck),
                    "gap_kind": "direct" if (vk == ck) else "indirect"})
    return out


@dataclass
class BandStructureOutcome:
    run_id: str
    spec: BandStructureSpec
    status: str
    reason: str = ""
    results: Dict[str, Result] = field(default_factory=dict)
    arrays: Dict[str, StoredArray] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    wall_time_s: float = 0.0
    audit: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_COMPLETE


class _Invalid(Exception):
    pass


def _approximations(spec: BandStructureSpec) -> List[str]:
    return [
        f"Kohn-Sham eigenvalues from a non-self-consistent GPAW calculation at the "
        f"self-consistent density, on {spec.n_kpoints} explicit k-points of the path "
        f"{path_text(spec.path)}, with symmetry off and {spec.n_bands} converged bands "
        f"({spec.extra_bands} unconverged buffer bands computed and discarded).",
        "Kohn-Sham bands are not quasiparticle bands: semilocal functionals underestimate "
        "band gaps, often by 30 to 50 percent in semiconductors.",
        "Energies relative to the Fermi level of the self-consistent ground state on its own "
        "k-point grid; the path carries no integration weight and sets no occupation.",
        "Band edges and gaps are extremes of the sampled path points; the true extremes can "
        "lie between points or off the path.",
        "Bands are sorted by energy at each k-point; crossings are not disentangled by "
        "symmetry, so a line joining equal band indices can switch character at a crossing.",
        "No orbital character is computed.",
    ]


def _validate(spec: BandStructureSpec, run: runner.GPAWRun, job: Dict[str, Any],
              arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    result = run.result
    bands = result.get("bands")
    if not isinstance(bands, dict):
        raise _Invalid("The worker reported no band structure.")
    mismatch = ground.parameter_mismatch(job["parameters"], result.get("parameters_used"))
    if mismatch:
        raise _Invalid("GPAW did not run the ground state with the parameters Materia sent: "
                       f"{mismatch}")
    mismatch = ground.parameter_mismatch(job["bands"]["nscf"],
                                         bands.get("nscf_parameters_used"), "nscf.")
    if mismatch:
        raise _Invalid("GPAW did not run the non-self-consistent path step with the "
                       f"parameters Materia sent: {mismatch}")
    sent = {k: v for k, v in job["bands"].items() if k != "nscf"}
    mismatch = ground.parameter_mismatch(sent, bands.get("used"), "bands.")
    if mismatch:
        raise _Invalid(f"The worker evaluated a different path from the one specified: "
                       f"{mismatch}")
    if not bands.get("nscf_converged"):
        raise _Invalid("The non-self-consistent step did not converge every requested band.")
    operations = bands.get("nscf_symmetry_operations")
    if operations != 1:
        raise _Invalid(f"The path step used {operations} symmetry operations, not the identity "
                       "alone, so GPAW may have folded or reordered path points.")
    fermi_scf = bands.get("fermi_level_scf_eV")
    fermi_nscf = bands.get("fermi_level_nscf_eV")
    if fermi_scf is None or not math.isfinite(float(fermi_scf)):
        raise _Invalid("The worker reported no finite Fermi level.")
    if fermi_nscf is None or abs(float(fermi_nscf) - float(fermi_scf)) > 1e-9:
        raise _Invalid("The non-self-consistent step moved the Fermi level; the energy zero "
                       "would not be the ground state's.")
    expected = np.asarray(spec.kpoints_frac, dtype=float)
    nk = len(expected)
    for name in ("band_kpoints", "band_bz_kpoints"):
        got = np.asarray(arrays.get(name), dtype=float)
        if got.shape != expected.shape:
            raise _Invalid(f"GPAW computed {got.shape[0] if got.ndim else 0} k-points, not the "
                           f"{nk} of the path.")
        if not np.all(np.isfinite(got)) or not np.allclose(got, expected, rtol=0,
                                                           atol=PATH_TOLERANCE):
            raise _Invalid("The k-points GPAW computed are not the path's k-points in the "
                           "path's order.")
    total = spec.n_bands + spec.extra_bands
    eigen = np.asarray(arrays.get("band_eigenvalues"), dtype=float)
    if eigen.ndim != 3:
        raise _Invalid("The eigenvalues have the wrong number of dimensions.")
    if eigen.shape[0] != spec.n_spins:
        raise _Invalid(f"The worker returned {eigen.shape[0]} spin channel(s), not "
                       f"{spec.n_spins}.")
    if eigen.shape[1] != nk:
        raise _Invalid(f"The eigenvalues cover {eigen.shape[1]} k-points, not {nk}.")
    if eigen.shape[2] != total:
        raise _Invalid(f"The eigenvalues hold {eigen.shape[2]} bands, not the {total} "
                       "computed; a requested band is missing.")
    kept = eigen[:, :, :spec.n_bands]
    if not np.all(np.isfinite(eigen)):
        raise _Invalid("The worker returned a non-finite eigenvalue.")
    if np.any(np.diff(kept, axis=2) < -1e-9):
        raise _Invalid("The eigenvalues are not in ascending order at every k-point.")
    recip = reciprocal_cell(spec.ground_state.cell_A)
    reported = np.asarray(bands.get("reciprocal_cell_invA"), dtype=float)
    if reported.shape != (3, 3) or not np.allclose(reported, recip, rtol=DISTANCE_TOLERANCE,
                                                   atol=DISTANCE_TOLERANCE):
        raise _Invalid("GPAW's reciprocal cell is not the one of the pinned cell.")
    distance = spec.distance()
    worker_distance = np.asarray(arrays.get("band_distance"), dtype=float)
    if worker_distance.shape != distance.shape:
        raise _Invalid("The path distance the worker returned has the wrong length.")
    scale = max(1.0, float(distance[-1]))
    error = float(np.abs(worker_distance - distance).max()) / scale
    if not np.all(np.isfinite(worker_distance)) or error > DISTANCE_TOLERANCE:
        raise _Invalid(f"The cumulative path distance from GPAW's reciprocal cell differs from "
                       f"Materia's recomputation by {error:.2e} of the path length.")
    return {"fermi_level_eV": float(fermi_scf), "eigen": kept, "distance": distance,
            "distance_error": error, "recip": recip}


def execute(spec: BandStructureSpec, environment=None, *, run_id: Optional[str] = None,
            progress: Optional[Callable[[dict], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None) -> BandStructureOutcome:
    """Run one band structure.  Raises :class:`~.spec.SpecRefused` if it may not run."""
    environment = specs._environment(environment)
    report = require(spec, environment)
    run_id = run_id or ground.new_run_id()
    started = time.perf_counter()
    started_unix = time.time()
    warnings = list(report.warnings)
    job = worker_job(spec)
    outcome = BandStructureOutcome(run_id=run_id, spec=spec, status=STATUS_FAILED,
                                   warnings=warnings)
    if cancelled is not None and cancelled():
        outcome.status = STATUS_CANCELLED
        outcome.reason = "Cancelled before GPAW was started. Nothing was kept."
        return outcome
    run = runner.run_job(job, environment, worker=ground.WORKER, progress=progress,
                         cancelled=cancelled, timeout_s=timeout_s)
    outcome.wall_time_s = time.perf_counter() - started
    outcome.audit = {"worker_status": run.status, "returncode": run.returncode,
                     "command": list(run.command), "stderr_tail": run.stderr[-2000:],
                     "log_tail": str(run.result.get("log_tail") or "")[-3000:]}
    if run.status == runner.STATUS_CANCELLED or (cancelled is not None and cancelled()):
        outcome.status = STATUS_CANCELLED
        outcome.reason = "The band-structure calculation was cancelled. Nothing was kept."
        return outcome
    if run.status == runner.STATUS_NOT_CONVERGED:
        outcome.reason = ("The ground state did not converge, so there is no density to "
                          "compute bands from. Nothing was kept.")
        return outcome
    if run.status != runner.STATUS_CONVERGED:
        detail = run.error or "no error message was produced"
        if timeout_s is not None and "stopped the calculation after" in run.stderr:
            detail = f"GPAW was stopped after the {timeout_s:g} s time limit"
        outcome.reason = (f"The band-structure calculation failed: {detail.rstrip('.')}. "
                          "Nothing was kept.")
        return outcome
    arrays = {k: np.asarray(v) for k, v in run.arrays.items()}
    try:
        data = _validate(spec, run, job, arrays)
    except _Invalid as exc:
        outcome.reason = f"{exc} Nothing was kept."
        return outcome

    gs = spec.ground_state
    restart = {"key": None, "reused_from": None, "loaded": False, "kept": False,
               "reason": "the band structure converges its own ground state; no restart "
                         "data is used"}
    history = run.result.get("scf_history") or []
    scf = ground._convergence(gs, run)
    base_extra = {
        "run_id": run_id, "structure_key": gs.structure_key,
        "geometry_digest": gs.geometry_digest, "spec_digest": gs.digest,
        "bands_spec_digest": spec.digest, "mass_numbers": list(gs.mass_numbers),
        "status": STATUS_COMPLETE, "classification": ground.CLASSIFICATION,
        "warnings": warnings,
        "software_warnings": [str(w) for w in (run.result.get("warnings") or [])],
        "study_id": None, "n_atoms": gs.n_atoms, "formula": gs.formula,
        "boundary": gs.boundary, "kind": "bands", "source": dict(spec.source),
    }
    staged = ground.GroundStateOutcome(run_id=run_id, spec=gs,
                                       status=runner.STATUS_CONVERGED, warnings=warnings)
    scf_run = runner.GPAWRun(status=runner.STATUS_CONVERGED, result=run.result,
                             stderr=run.stderr, returncode=run.returncode,
                             wall_time_s=run.wall_time_s, iterations=run.iterations,
                             command=run.command, arrays={}, events=run.events)
    ground.collect(staged, gs, scf_run, report, restart, base_extra, scf, history)
    bands = run.result["bands"]
    fermi = data["fermi_level_eV"]
    labels = spec.labels()
    breaks = spec.breaks()
    kfrac = np.asarray(spec.kpoints_frac, dtype=float)
    edges = band_edges(data["eigen"], fermi, labels, kfrac)
    checks: Dict[str, Any] = {
        "kpoints_match": True, "kpoint_tolerance": PATH_TOLERANCE,
        "distance_max_relative_error": data["distance_error"],
        "distance_tolerance": DISTANCE_TOLERANCE,
        "all_eigenvalues_finite": True, "ascending_at_every_kpoint": True,
        "nscf_converged_bands": spec.n_bands, "buffer_bands_discarded": spec.extra_bands,
        "fermi_level_scf_eV": fermi, "fermi_level_nscf_eV": float(bands["fermi_level_nscf_eV"]),
        "nscf_symmetry_operations": int(bands.get("nscf_symmetry_operations") or 0),
        "nscf_iterations": int(bands.get("nscf_iterations") or 0),
    }
    source_energy = spec.source.get("energy_eV")
    energy = staged.results["energy"].value
    if source_energy is not None:
        checks["source_energy_difference_eV"] = float(energy) - float(source_energy)
        if abs(checks["source_energy_difference_eV"]) > 1e-3:
            warnings.append(f"The band-structure run's ground-state energy differs from its "
                            f"source's by {checks['source_energy_difference_eV']:.2e} eV.")
    if edges.get("metallic_on_path"):
        warnings.append("At least one band crosses the Fermi level along the path: metallic "
                        "along the path, so no gap is reported.")
    ticks = spec.ticks()
    meta = {"fermi_level_eV": fermi, "reference": spec.energy_reference,
            "bands_spec_digest": spec.digest, "labels": labels, "breaks": breaks,
            "path": path_text(spec.path)}
    outcome.arrays = {
        "eigenvalues": StoredArray(np.ascontiguousarray(data["eigen"]), "eV",
                                   "Kohn-Sham eigenvalues along the path [spin, k, band], "
                                   "absolute GPAW energies; subtract the Fermi level for the plot",
                                   "table", dict(meta, axes=["spin", "kpoint", "band"],
                                                 absolute=True)),
        "kpoints_frac": StoredArray(kfrac, "fractional",
                                    "Path k-points in the reciprocal basis of the cell",
                                    "table", dict(meta, axes=["kpoint", "component"])),
        "kpoints_cartesian": StoredArray(kfrac @ data["recip"], "1/A",
                                         "Path k-points in Cartesian coordinates, 2 pi included",
                                         "table", dict(meta, axes=["kpoint", "component"])),
        "distance": StoredArray(data["distance"], "1/A",
                                "Cumulative distance along the path, 2 pi included; no "
                                "distance across a break", "table", dict(meta)),
    }
    checksums = {name: stored.sha256() for name, stored in outcome.arrays.items()}
    results = dict(staged.results)
    template = results["energy"].provenance
    summary = {
        "status": STATUS_COMPLETE, "energy_reference": spec.energy_reference,
        "fermi_level_eV": fermi, "fermi_reference": "self-consistent ground state, "
                                                    "its own k-point grid",
        "n_spins": spec.n_spins, "n_bands": spec.n_bands, "extra_bands": spec.extra_bands,
        "n_kpoints": spec.n_kpoints, "path": path_text(spec.path),
        "special_points": {label: list(c) for label, c in spec.special_points},
        "segment_intervals": list(spec.segment_intervals), "labels": labels,
        "breaks": breaks, "ticks": ticks,
        "path_length_invA": float(data["distance"][-1]),
        "reciprocal_cell_invA": data["recip"].tolist(),
        "band_edges": edges, "checks": checks, "source": dict(spec.source),
        "arrays": {name: f"dftbands::{run_id}::{name}" for name in outcome.arrays},
        "array_sha256": checksums,
        "units": {"energy": "eV", "distance": "1/A (2 pi included)",
                  "kpoints_frac": "fractional coordinates of the reciprocal lattice",
                  "kpoints_cartesian": "1/A (2 pi included)"},
        "occupations": "not stored: path k-points carry no Brillouin-zone weight; a state is "
                       "occupied if it lies below the self-consistent Fermi level",
        "orbital_character": "not computed",
        "nscf_parameters_used": bands.get("nscf_parameters_used"), "used": bands.get("used"),
        "started_unix": started_unix, "finished_unix": time.time(),
        "wall_time_s": outcome.wall_time_s,
        "experimental_comparison": None,
    }
    convergence = Convergence(
        converged=True, iterations=int(run.iterations),
        residual=data["distance_error"],
        residual_metric="largest relative difference between the path distance from GPAW's "
                        "reciprocal cell and Materia's recomputation from the pinned cell",
        tolerance=DISTANCE_TOLERANCE,
        message=(f"Ground state converged in {run.iterations} SCF iterations; the path step "
                 f"converged all {spec.n_bands} requested bands at {spec.n_kpoints} k-points "
                 f"in {checks['nscf_iterations']} iterations; k-points, labels and distance "
                 "verified."))
    record = Result("dft_band_structure", summary, "eV", ground._copy_provenance(template),
                    convergence=convergence)
    record.extra.update(base_extra)
    results["bands"] = record
    results["fermi_level"].value = fermi
    parameters = {"bands_spec": spec.as_dict(), "bands_spec_digest": spec.digest,
                  "gpaw_parameters": job["parameters"],
                  "nscf_parameters": job["bands"]["nscf"],
                  "nscf_parameters_used": bands.get("nscf_parameters_used"),
                  "band_settings": job["bands"], "bands_used": bands.get("used"),
                  "path_origin": dict(spec.path_origin), "source": dict(spec.source),
                  "array_sha256": checksums}
    approximations = _approximations(spec)
    for item in results.values():
        item.provenance.model = MODEL
        item.provenance.fidelity = Fidelity.TIER3_EXTERNAL
        item.provenance.origin = Origin.CALCULATED
        item.provenance.inputs_digest = spec.digest
        item.provenance.parameters.update(parameters)
        item.provenance.references = list(item.provenance.references) + [
            r for r in PATH_CITATIONS if r not in item.provenance.references]
        item.provenance.approximations = list(item.provenance.approximations) + [
            a for a in approximations if a not in item.provenance.approximations]
        item.extra["warnings"] = list(warnings)
    outcome.results = results
    outcome.status = STATUS_COMPLETE
    outcome.warnings = warnings
    return outcome
