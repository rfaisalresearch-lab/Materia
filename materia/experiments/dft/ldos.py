"""Spatially resolved local density of states and Tersoff-Hamann STM images, through GPAW.

An :class:`LDOSSpec` is a frozen ground-state specification together with an
energy window relative to the self-consistent Fermi level, the bands of the
non-self-consistent step and the spin channels.  It is immutable and
versioned (:data:`SCHEMA`, :data:`VERSION`).

The worker converges the ground state, then runs GPAW's ``fixed_density`` on
the ground state's own k-point grid with point-group symmetry switched off
(time reversal kept, which leaves every ``|psi|^2`` unchanged) and every band
converged, and accumulates

    n(r) = (2 / n_spins) sum_s sum_k w_k sum_n [E_min < e_nks - E_F < E_max] |psi~_nks(r)|^2

on GPAW's coarse real-space grid, in states per A^3 for both spins.  This is
the quantity ASE's ``ase.dft.stm.STM`` evaluates (without its factor 2/n_spins)
and the one the Tersoff-Hamann approximation makes proportional to the
tunnelling current at a sample bias spanning the window.  Point-group symmetry
is off because irreducible k-points would give a map with the surface's
symmetry broken.

The wavefunctions are GPAW's pseudo-wavefunctions.  They equal the
all-electron wavefunctions outside the PAW augmentation spheres, whose radii
are recorded, and differ inside them; the map is therefore quantitative in
the vacuum where STM tips are, and not inside the atoms.  STM images are only
taken at heights above those spheres and away from the cell face, where the
real-space boundary condition forces the wavefunctions to zero.

A constant-height image is the map interpolated linearly along the surface
normal at one height; a constant-current image is the height at which the map
equals a chosen value, found from the vacuum downwards, as ASE does.  No
conversion to amperes is made: Tersoff-Hamann gives a proportionality, and an
empirical conversion would add a number Materia cannot verify.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ...project_format.arrays import StoredArray
from ...provenance import Convergence, Fidelity, Origin, Result
from ...solvers.gpaw_driver import runner
from . import run as ground
from . import spec as specs
from .dos import _minimal, _occupied_bands, electronic_digest

SCHEMA = "materia.dft.ldos"
VERSION = "1.0"
MODEL = "external:gpaw/local-density-of-states"
REFERENCES: Tuple[str, ...] = (
    "J. Tersoff and D. R. Hamann, Phys. Rev. B 31 (1985) 805",
)

SPIN_CHANNELS: Tuple[str, ...] = ("total", "resolved")
SOURCES: Tuple[str, ...] = ("structure", "ground-state", "relaxation")
MAX_WINDOW_EV = 20.0
MIN_WIDTH_EV = 0.01
MAX_BANDS = 2000
MAX_GRID_POINTS = 8_000_000
BOHR_A = 0.529177210903
MIN_TIP_HEIGHT_A = 2.0
FACE_MARGIN_A = 2.0
COUNT_TOLERANCE = 1e-9

STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

FIELDS: Tuple[specs.FieldInfo, ...] = (
    specs.FieldInfo("source", "ldos", "geometry source", "",
                    "Where the geometry and electronic settings come from: a structure, a "
                    "stored converged ground state or a stored converged relaxation.",
                    settable=False),
    specs.FieldInfo("energy_min_eV", "ldos", "window start", "eV",
                    "Lower edge of the energy window, relative to the Fermi level. For an "
                    "STM image at negative sample bias V this is eV, and the upper edge 0."),
    specs.FieldInfo("energy_max_eV", "ldos", "window end", "eV",
                    "Upper edge of the window, relative to the Fermi level. For positive "
                    "sample bias V the window is 0 to eV: empty states."),
    specs.FieldInfo("spin_channels", "ldos", "spin channels", "",
                    "total sums both spins; resolved also stores each spin, which needs a "
                    "spin-polarised calculation."),
    specs.FieldInfo("n_bands", "ldos", "bands", "bands",
                    "Kohn-Sham states per spin and k-point in the non-self-consistent step, "
                    "all converged; a few unconverged buffer bands above them are computed and "
                    "never used. The converged bands must reach above the window at every "
                    "k-point or nothing is kept."),
)
FIELD_INDEX = {f.name: f for f in FIELDS}
SETTABLE: Tuple[str, ...] = tuple(f.name for f in FIELDS if f.settable)
LDOS_SYMMETRY = {"point_group": False, "time_reversal": True}


@dataclass(frozen=True)
class LDOSSpec:
    """One LDOS calculation, frozen.  See :data:`FIELDS`."""

    ground_state: specs.GroundStateSpec
    source: Dict[str, Any]
    energy_min_eV: float
    energy_max_eV: float
    spin_channels: str
    n_bands: int
    schema: str = SCHEMA
    version: str = VERSION

    def as_dict(self) -> dict:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "ground_state":
                out[f.name] = value.as_dict()
            elif f.name == "source":
                out[f.name] = json.loads(json.dumps(value))
            else:
                out[f.name] = specs._plain(value)
        return out

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "LDOSSpec":
        if data.get("schema") != SCHEMA:
            raise specs.SpecError(f"Not an LDOS specification: {data.get('schema')!r}.")
        if str(data.get("version")) != VERSION:
            raise specs.SpecError(f"LDOS specification version {data.get('version')} is not "
                                  f"understood by this Materia (it reads {VERSION}).")
        names = {f.name for f in fields(LDOSSpec)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise specs.SpecError(f"Unknown LDOS field(s): {', '.join(unknown)}.")
        missing = sorted(n for n in names - set(data) if n not in ("schema", "version"))
        if missing:
            raise specs.SpecError(f"Missing LDOS field(s): {', '.join(missing)}.")
        values = dict(data)
        values["ground_state"] = specs.GroundStateSpec.from_dict(data["ground_state"])
        values["source"] = dict(data["source"])
        values["energy_min_eV"] = float(data["energy_min_eV"])
        values["energy_max_eV"] = float(data["energy_max_eV"])
        values["n_bands"] = int(data["n_bands"])
        values["schema"] = SCHEMA
        values["version"] = VERSION
        return LDOSSpec(**values)

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
    def n_spins(self) -> int:
        return 2 if self.ground_state.spin_polarized else 1

    def settings(self) -> Dict[str, Any]:
        return {name: specs._plain(getattr(self, name)) for name in SETTABLE}


def _coerce(name: str, value: Any) -> Any:
    if name in ("energy_min_eV", "energy_max_eV"):
        return specs._number(name, value, "eV")
    if name == "spin_channels":
        return specs._choice(name, value, SPIN_CHANNELS)
    if name == "n_bands":
        return specs._integer(name, value)
    raise specs.SpecError(f"{name} is not an LDOS variable.")


def _split(variables: Mapping[str, Any]):
    own = {k: v for k, v in variables.items() if k in SETTABLE}
    rest = {k: v for k, v in variables.items() if k not in SETTABLE}
    if "source" in rest:
        raise specs.SpecError("source is not settable; it follows from where the LDOS is "
                              "built from.")
    return own, rest


def _finish(gs: specs.GroundStateSpec, source: Dict[str, Any], own: Dict[str, Any],
            environment, base: Optional["LDOSSpec"] = None) -> LDOSSpec:
    values = {name: _coerce(name, value) for name, value in own.items()}
    occupied = _occupied_bands(gs, environment)
    if occupied is None:
        occupied = int(math.ceil(sum(gs.numbers) / 2.0))
    defaults = {
        "energy_min_eV": base.energy_min_eV if base else -1.0,
        "energy_max_eV": base.energy_max_eV if base else 0.0,
        "spin_channels": base.spin_channels if base else (
            "resolved" if gs.spin_polarized else "total"),
        "n_bands": base.n_bands if base else occupied + max(4, int(math.ceil(0.3 * occupied))),
    }
    for name, value in defaults.items():
        values.setdefault(name, value)
    return LDOSSpec(ground_state=gs, source=dict(source), **values)


def build(structure, environment=None, structure_key: Optional[str] = None,
          **variables: Any) -> LDOSSpec:
    """An LDOS specification for ``structure``: its own ground state, then the window."""
    own, rest = _split(variables)
    environment = specs._environment(environment)
    rest.pop("observables", None)
    gs = _minimal(specs.build(structure, environment, structure_key=structure_key, **rest))
    source = {"kind": "structure", "run_id": None, "spec_digest": None,
              "geometry_digest": gs.geometry_digest,
              "electronic_digest": electronic_digest(gs), "energy_eV": None}
    return _finish(gs, source, own, environment)


def build_from_run(project, kind: str, run_id: str, environment=None,
                   **variables: Any) -> LDOSSpec:
    """An LDOS specification on the exact geometry and settings of a stored run."""
    from . import dos

    own, rest = _split(variables)
    if rest:
        raise specs.SpecError(
            f"{', '.join(sorted(rest))}: an LDOS built from a stored {kind} run keeps that "
            "run's geometry and electronic settings exactly. Build it from the structure to "
            "change them.")
    environment = specs._environment(environment)
    original, energy = dos.source_spec(project, kind, run_id)
    gs = _minimal(original)
    source = {"kind": kind, "run_id": run_id, "spec_digest": original.digest,
              "geometry_digest": original.geometry_digest,
              "electronic_digest": electronic_digest(original), "energy_eV": energy}
    return _finish(gs, source, own, environment)


def changed(spec: LDOSSpec, environment=None, **variables: Any) -> LDOSSpec:
    own, rest = _split(variables)
    environment = specs._environment(environment)
    gs, source = spec.ground_state, dict(spec.source)
    if rest:
        if spec.source.get("kind") != "structure":
            raise specs.SpecError(f"{', '.join(sorted(rest))}: this LDOS keeps the electronic "
                                  f"settings of {spec.source.get('kind')} run "
                                  f"{spec.source.get('run_id')}.")
        rest.pop("observables", None)
        gs = _minimal(specs.changed(spec.ground_state, environment, **rest))
        source["electronic_digest"] = electronic_digest(gs)
    return _finish(gs, source, own, environment, base=spec)


def estimated_grid_points(gs: specs.GroundStateSpec) -> int:
    """Approximate number of points of GPAW's coarse grid, for the size check."""
    return int(specs.estimate_fine_grid(gs) / 8)


def check(spec: LDOSSpec, environment=None) -> specs.SpecReport:
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
        general.append(f"LDOS specification {spec.schema} {spec.version} is not {SCHEMA} "
                       f"{VERSION}.")
    source = spec.source
    if source.get("kind") not in SOURCES:
        refuse("source", f"The geometry source must be one of {', '.join(SOURCES)}.")
    if source.get("geometry_digest") != gs.geometry_digest:
        refuse("source", "The geometry differs from the source's geometry.")
    if source.get("electronic_digest") != electronic_digest(gs):
        refuse("source", f"The electronic settings differ from those of the "
                         f"{source.get('kind')} source.")
    low, high = spec.energy_min_eV, spec.energy_max_eV
    if not (math.isfinite(low) and math.isfinite(high)) or high - low < MIN_WIDTH_EV:
        refuse("energy_max_eV", f"The window must end at least {MIN_WIDTH_EV:g} eV above where "
                                "it starts.")
    elif abs(low) > MAX_WINDOW_EV or abs(high) > MAX_WINDOW_EV:
        refuse("energy_min_eV", f"The window must lie within {MAX_WINDOW_EV:g} eV of the Fermi "
                                "level.")
    elif low < 0.0 < high:
        warnings.append("The window spans the Fermi level: it mixes filled and empty states, "
                        "which no single sample bias images.")
    if spec.spin_channels not in SPIN_CHANNELS:
        refuse("spin_channels", f"spin_channels must be one of {', '.join(SPIN_CHANNELS)}.")
    elif spec.spin_channels == "resolved" and not gs.spin_polarized:
        refuse("spin_channels", "The calculation is spin-paired, so both spins are identical by "
                                "construction. Use total, or turn spin polarisation on.")
    occupied = _occupied_bands(gs, environment)
    if not (1 <= spec.n_bands <= MAX_BANDS):
        refuse("n_bands", f"n_bands must be 1 to {MAX_BANDS}.")
    elif occupied is not None and spec.n_bands < occupied:
        refuse("n_bands", f"{spec.n_bands} bands cannot hold the {occupied} occupied states.")
    points = estimated_grid_points(gs)
    arrays = 1 + (2 if spec.spin_channels == "resolved" else 0)
    if points * arrays > MAX_GRID_POINTS:
        refuse("grid_spacing_A" if gs.representation != "pw" else "cutoff_eV",
               f"About {points:,} grid points per map is more than this interface stores; "
               "use a coarser grid or a smaller cell.")
    if gs.boundary != "cluster" and all(k == 1 for k in gs.kpoints):
        warnings.append("A periodic LDOS from the Gamma point alone shows the states of one k, "
                        "not the surface's LDOS; use more k-points.")
    warnings.append("The map is built from pseudo-wavefunctions: it equals the all-electron "
                    "LDOS outside the PAW augmentation spheres and not inside them.")
    blocking = list(general)
    for texts in by_field.values():
        for text in texts:
            if text not in blocking:
                blocking.append(text)
    options = dict(report.options)
    options.update({"spin_channels": ["total", "resolved"] if gs.spin_polarized else ["total"],
                    "occupied_bands": occupied, "estimated_grid_points": points,
                    "stm_images": stm_supported(gs)[0]})
    return specs.SpecReport(blocking, warnings, by_field, report.boundary, report.electrons,
                            options)


def require(spec: LDOSSpec, environment=None) -> specs.SpecReport:
    report = check(spec, environment)
    if not report.ok:
        raise specs.SpecRefused(report)
    return report


def buffer_bands(n_bands: int) -> int:
    """Unconverged bands computed above the requested ones, to help the eigensolver."""
    return max(4, int(math.ceil(0.25 * n_bands)))


def nscf_parameters(spec: LDOSSpec) -> Dict[str, Any]:
    gs = spec.ground_state
    return {"kpts": {"size": list(gs.kpoints), "gamma": bool(gs.kpoints_gamma_centered)},
            "nbands": int(spec.n_bands + buffer_bands(spec.n_bands)),
            "symmetry": dict(LDOS_SYMMETRY), "convergence": {"bands": int(spec.n_bands)}}


def ldos_settings(spec: LDOSSpec) -> Dict[str, Any]:
    return {"nscf": nscf_parameters(spec), "energy_min": float(spec.energy_min_eV),
            "energy_max": float(spec.energy_max_eV), "spin": spec.spin_channels,
            "n_bands": int(spec.n_bands)}


def worker_job(spec: LDOSSpec) -> Dict[str, Any]:
    gs = spec.ground_state
    return {"structure": specs.worker_structure(gs), "parameters": specs.gpaw_parameters(gs),
            "observables": list(gs.observables),
            "expected_datasets": {s: {"path": p, "sha256": h} for s, p, h in gs.paw_datasets},
            "ldos": ldos_settings(spec)}


def describe(spec: LDOSSpec, report: Optional[specs.SpecReport] = None) -> dict:
    report = report if report is not None else check(spec)
    items = []
    for info in FIELDS:
        item = info.as_dict()
        value = getattr(spec, info.name)
        item["value"] = (json.loads(json.dumps(value)) if info.name == "source"
                         else specs._plain(value))
        item["problems"] = list(report.by_field.get(info.name, []))
        items.append(item)
    return {"schema": spec.schema, "version": spec.version, "digest": spec.digest,
            "short_digest": spec.short_digest, "ldos": items, "settings": spec.settings(),
            "source": dict(spec.source), "ground_state": specs.describe(spec.ground_state, report),
            "gpaw_parameters": specs.gpaw_parameters(spec.ground_state),
            "nscf_parameters": nscf_parameters(spec)}


def window_count(eigenvalues: np.ndarray, weights: np.ndarray, fermi: float, low: float,
                 high: float) -> float:
    """States per cell in the open window, both spins, from eigenvalues [s, k, n] and weights."""
    relative = np.asarray(eigenvalues, dtype=float) - fermi
    inside = (relative > low) & (relative < high)
    degeneracy = 2.0 / relative.shape[0]
    return float(degeneracy * np.einsum("k,skn->", np.asarray(weights, dtype=float), inside))


def stm_supported(gs) -> Tuple[bool, str]:
    """Whether a Tersoff-Hamann image can be taken of this geometry, and why not."""
    pbc = tuple(bool(p) for p in gs.pbc)
    cell = np.asarray(gs.cell_A, dtype=float)
    if pbc != (True, True, False):
        return False, ("STM images need a slab periodic along a and b with vacuum along c; "
                       f"this cell is {gs.boundary}.")
    if np.any(np.abs(cell[2, :2]) > 1e-9) or np.any(np.abs(cell[:2, 2]) > 1e-9):
        return False, "STM images need c perpendicular to a and b."
    return True, ""


def _column_interpolate(values: np.ndarray, index: float) -> np.ndarray:
    lower = int(math.floor(index))
    fraction = index - lower
    return (1.0 - fraction) * values[:, :, lower] + fraction * values[:, :, lower + 1]


def stm_image(ldos: np.ndarray, cell: np.ndarray, positions: np.ndarray, mode: str,
              height_A: Optional[float] = None, isovalue: Optional[float] = None,
              augmentation_radius_A: float = 0.0) -> Dict[str, Any]:
    """A Tersoff-Hamann image from a stored LDOS map on a slab.

    ``constant-height``: the map interpolated linearly along c at ``height_A``
    above the topmost atom.  ``constant-current``: for every column, the height
    above the topmost atom where the map equals ``isovalue``, searched from the
    top of the valid vacuum band downwards.  Raises ``ValueError`` when the
    request leaves the region where the map is meaningful.
    """
    data = np.asarray(ldos, dtype=float)
    cell = np.asarray(cell, dtype=float)
    nz = data.shape[2]
    length = float(cell[2, 2])
    spacing = length / nz
    top = float(np.asarray(positions, dtype=float)[:, 2].max())
    floor_z = top + max(MIN_TIP_HEIGHT_A, augmentation_radius_A + 1.0)
    ceiling_z = length - FACE_MARGIN_A
    if ceiling_z <= floor_z + spacing:
        raise ValueError(f"The vacuum above the slab ({length - top:.2f} A) leaves no band "
                         f"between {floor_z - top:.2f} A above the surface and "
                         f"{FACE_MARGIN_A:g} A below the cell face; enlarge the vacuum.")
    grid = np.indices(data.shape[:2]).reshape(2, -1).T / np.array(data.shape[:2])
    xy = (grid @ cell[:2, :2]).reshape(data.shape[:2] + (2,))
    out: Dict[str, Any] = {"mode": mode, "shape": list(data.shape[:2]),
                           "x_A": xy[..., 0], "y_A": xy[..., 1],
                           "surface_z_A": top, "valid_heights_A": [floor_z - top,
                                                                   ceiling_z - top],
                           "grid_spacing_A": spacing}
    if mode == "constant-height":
        if height_A is None or not math.isfinite(height_A):
            raise ValueError("A constant-height image needs a height.")
        z = top + float(height_A)
        if not (floor_z <= z <= ceiling_z):
            raise ValueError(f"A height of {height_A:g} A is outside the valid band "
                             f"{floor_z - top:.2f} to {ceiling_z - top:.2f} A above the "
                             "surface: closer, the pseudo-wavefunctions are not the "
                             "all-electron ones or the tip is not in vacuum; farther, the "
                             "cell face forces the wavefunctions to zero.")
        out["values"] = _column_interpolate(data, z / spacing)
        out["height_A"] = float(height_A)
        out["unit"] = "states/A^3"
        return out
    if mode == "constant-current":
        if isovalue is None or not (math.isfinite(isovalue) and isovalue > 0):
            raise ValueError("A constant-current image needs a positive LDOS value.")
        start = int(math.floor(ceiling_z / spacing))
        stop = int(math.ceil(floor_z / spacing))
        heights = np.full(data.shape[:2], np.nan)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                column = data[i, j]
                for n in range(start - 1, stop - 1, -1):
                    if column[n] > isovalue:
                        c2, c1 = column[n], column[n + 1]
                        heights[i, j] = (n + (c2 - isovalue) / (c2 - c1)) * spacing
                        break
        if np.isnan(heights).any():
            missing = int(np.isnan(heights).sum())
            raise ValueError(f"The value {isovalue:g} states/A^3 is not reached inside the "
                             f"valid band in {missing} of {heights.size} columns; choose a "
                             "value between the map's values at the two ends of the band.")
        if heights.max() > ceiling_z - 1e-9 or heights.min() < floor_z - spacing:
            raise ValueError("The constant-value surface leaves the valid band.")
        out["values"] = heights - top
        out["isovalue"] = float(isovalue)
        out["unit"] = "A above the topmost atom"
        return out
    raise ValueError(f"mode must be constant-height or constant-current, got {mode!r}.")


@dataclass
class LDOSOutcome:
    run_id: str
    spec: LDOSSpec
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


def _validate(spec: LDOSSpec, run: runner.GPAWRun, job: Dict[str, Any],
              arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    result = run.result
    ldos = result.get("ldos")
    if not isinstance(ldos, dict):
        raise _Invalid("The worker reported no LDOS.")
    mismatch = ground.parameter_mismatch(job["parameters"], result.get("parameters_used"))
    if mismatch:
        raise _Invalid("GPAW did not run the ground state with the parameters Materia sent: "
                       f"{mismatch}")
    mismatch = ground.parameter_mismatch(job["ldos"]["nscf"], ldos.get("nscf_parameters_used"),
                                         "nscf.")
    if mismatch:
        raise _Invalid("GPAW did not run the non-self-consistent step with the parameters "
                       f"Materia sent: {mismatch}")
    sent = {k: v for k, v in job["ldos"].items() if k != "nscf"}
    mismatch = ground.parameter_mismatch(sent, ldos.get("used"), "ldos.")
    if mismatch:
        raise _Invalid(f"The worker evaluated a different window from the one specified: "
                       f"{mismatch}")
    if not ldos.get("nscf_converged"):
        raise _Invalid("The non-self-consistent step did not converge every band.")
    fermi = ldos.get("fermi_level_scf_eV")
    if fermi is None or not math.isfinite(float(fermi)):
        raise _Invalid("The worker reported no finite Fermi level.")
    if ldos.get("fermi_level_nscf_eV") is None or abs(
            float(ldos["fermi_level_nscf_eV"]) - float(fermi)) > 1e-9:
        raise _Invalid("The non-self-consistent step moved the Fermi level; the window would "
                       "not be the ground state's.")
    eigen = np.asarray(arrays.get("ldos_eigenvalues"), dtype=float)
    weights = np.asarray(arrays.get("ldos_kweights"), dtype=float)
    computed = spec.n_bands + buffer_bands(spec.n_bands)
    if eigen.ndim != 3 or eigen.shape[0] != spec.n_spins or eigen.shape[2] != computed \
            or weights.shape != (eigen.shape[1],):
        raise _Invalid("The eigenvalues or k-point weights have the wrong shape; a band may be "
                       "missing.")
    eigen = eigen[:, :, :spec.n_bands]
    if not (np.all(np.isfinite(eigen)) and np.all(np.isfinite(weights))):
        raise _Invalid("The worker returned a non-finite eigenvalue or weight.")
    if abs(float(weights.sum()) - 1.0) > 1e-9 or np.any(weights <= 0):
        raise _Invalid("The k-point weights do not form a sampling of the zone.")
    total = np.asarray(arrays.get("ldos_total"), dtype=float)
    shape = tuple(int(n) for n in ldos.get("grid") or [])
    if total.ndim != 3 or tuple(total.shape) != shape:
        raise _Invalid("The LDOS map does not have the grid the worker reported.")
    cell = np.asarray(ldos.get("cell_A"), dtype=float)
    if cell.shape != (3, 3) or not np.allclose(cell, np.asarray(spec.ground_state.cell_A),
                                               rtol=0, atol=1e-9):
        raise _Invalid("The LDOS map is on a different cell from the specification's.")
    if not np.all(np.isfinite(total)):
        raise _Invalid("The LDOS map contains a non-finite value.")
    scale = max(1e-300, float(np.abs(total).max()))
    if float(total.min()) < -1e-12 * scale:
        raise _Invalid("The LDOS map is negative somewhere; a sum of |psi|^2 cannot be.")
    if spec.spin_channels == "resolved":
        spin = np.asarray(arrays.get("ldos_spin"), dtype=float)
        if spin.shape != (2,) + shape or not np.all(np.isfinite(spin)):
            raise _Invalid("The spin-resolved LDOS has the wrong shape or a non-finite value.")
        if float(np.abs(spin.sum(axis=0) - total).max()) > 1e-9 * scale:
            raise _Invalid("Spin up and spin down do not add up to the total LDOS.")
    elif "ldos_spin" in arrays:
        raise _Invalid("A spin-resolved LDOS was returned but not requested.")
    relative = eigen - float(fermi)
    highest = float(relative.max(axis=2).min())
    if highest <= spec.energy_max_eV:
        raise _Invalid(f"The highest computed band reaches only {highest:.3f} eV above the "
                       f"Fermi level at some k-point, not above the window's top "
                       f"({spec.energy_max_eV:g} eV): states in the window would be missing. "
                       "Raise n_bands.")
    count = window_count(eigen, weights, float(fermi), spec.energy_min_eV, spec.energy_max_eV)
    reported = ldos.get("states_in_window")
    if reported is None or abs(float(reported) - count) > COUNT_TOLERANCE * max(1.0, count):
        raise _Invalid(f"The worker counted {reported} states in the window; the returned "
                       f"eigenvalues and weights give {count:.12g}.")
    volume = abs(float(np.linalg.det(cell)))
    integral = float(total.sum() * volume / total.size)
    if count > 0 and not (0.5 < integral / count < 1.5):
        raise _Invalid(f"The map integrates to {integral:.4g} states against {count:.4g} in the "
                       "window; pseudo-wavefunction norms lie close to one, so the map does not "
                       "describe these states.")
    if count == 0:
        raise _Invalid("No Kohn-Sham state of this k-point grid lies in the window, so the map "
                       "would be zero everywhere. Widen the window or use a k-point grid that "
                       "samples the states near this energy.")
    radii = {str(k): float(v) for k, v in (ldos.get("augmentation_radii_A") or {}).items()}
    if set(radii) != set(spec.ground_state.symbols):
        raise _Invalid("The worker did not report an augmentation radius for every element.")
    return {"fermi_level_eV": float(fermi), "eigen": eigen, "weights": weights,
            "total": total, "count": count, "integral": integral, "radii": radii,
            "cell": cell, "highest_above_EF": highest, "shape": shape}


def _approximations(spec: LDOSSpec) -> List[str]:
    gs = spec.ground_state
    return [
        f"Local density of states from GPAW pseudo-wavefunctions in the window "
        f"{spec.energy_min_eV:g} to {spec.energy_max_eV:g} eV about the self-consistent Fermi "
        f"level, on the {'x'.join(map(str, gs.kpoints))} k-point grid with point-group "
        f"symmetry off, {spec.n_bands} converged bands and "
        f"{buffer_bands(spec.n_bands)} unconverged buffer bands that never enter the map.",
        "Equal to the all-electron LDOS outside the PAW augmentation spheres only.",
        "A sharp window: each Kohn-Sham state counts fully or not at all; finite k-point "
        "sampling makes the map a sum over discrete states.",
        "Tersoff-Hamann: an s-wave tip, the current proportional to the LDOS at the tip "
        "centre; no tip electronic structure, no tip-sample interaction.",
        "Kohn-Sham states of a semilocal functional, not quasiparticles.",
    ]


def execute(spec: LDOSSpec, environment=None, *, run_id: Optional[str] = None,
            progress: Optional[Callable[[dict], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None) -> LDOSOutcome:
    """Run one LDOS.  Raises :class:`~.spec.SpecRefused` if it may not run."""
    environment = specs._environment(environment)
    report = require(spec, environment)
    run_id = run_id or ground.new_run_id()
    started = time.perf_counter()
    started_unix = time.time()
    warnings = list(report.warnings)
    job = worker_job(spec)
    outcome = LDOSOutcome(run_id=run_id, spec=spec, status=STATUS_FAILED, warnings=warnings)
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
        outcome.reason = "The LDOS calculation was cancelled. Nothing was kept."
        return outcome
    if run.status == runner.STATUS_NOT_CONVERGED:
        outcome.reason = ("The ground state did not converge, so there is no LDOS to take. "
                          "Nothing was kept.")
        return outcome
    if run.status != runner.STATUS_CONVERGED:
        detail = run.error or "no error message was produced"
        if timeout_s is not None and "stopped the calculation after" in run.stderr:
            detail = f"GPAW was stopped after the {timeout_s:g} s time limit"
        outcome.reason = f"The LDOS calculation failed: {detail.rstrip('.')}. Nothing was kept."
        return outcome
    arrays = {k: np.asarray(v) for k, v in run.arrays.items()}
    try:
        data = _validate(spec, run, job, arrays)
    except _Invalid as exc:
        outcome.reason = f"{exc} Nothing was kept."
        return outcome

    gs = spec.ground_state
    restart = {"key": None, "reused_from": None, "loaded": False, "kept": False,
               "reason": "the LDOS converges its own ground state; no restart data is used"}
    history = run.result.get("scf_history") or []
    scf = ground._convergence(gs, run)
    base_extra = {
        "run_id": run_id, "structure_key": gs.structure_key,
        "geometry_digest": gs.geometry_digest, "spec_digest": gs.digest,
        "ldos_spec_digest": spec.digest, "mass_numbers": list(gs.mass_numbers),
        "status": STATUS_COMPLETE, "classification": ground.CLASSIFICATION,
        "warnings": warnings,
        "software_warnings": [str(w) for w in (run.result.get("warnings") or [])],
        "study_id": None, "n_atoms": gs.n_atoms, "formula": gs.formula,
        "boundary": gs.boundary, "kind": "ldos", "source": dict(spec.source),
    }
    staged = ground.GroundStateOutcome(run_id=run_id, spec=gs,
                                       status=runner.STATUS_CONVERGED, warnings=warnings)
    scf_run = runner.GPAWRun(status=runner.STATUS_CONVERGED, result=run.result,
                             stderr=run.stderr, returncode=run.returncode,
                             wall_time_s=run.wall_time_s, iterations=run.iterations,
                             command=run.command, arrays={}, events=run.events)
    ground.collect(staged, gs, scf_run, report, restart, base_extra, scf, history)
    ldos = run.result["ldos"]
    meta = {"grid_shape": list(data["shape"]), "cell_A": data["cell"].tolist(),
            "pbc": list(gs.pbc), "origin_A": [0.0, 0.0, 0.0],
            "convention": "point (i, j, k) sits at fractional coordinates (i/N1, j/N2, k/N3); "
                          "the end point of each axis is not repeated",
            "grid": "coarse", "window_eV": [spec.energy_min_eV, spec.energy_max_eV],
            "reference": "self-consistent Fermi level", "ldos_spec_digest": spec.digest}
    outcome.arrays = {
        "ldos": StoredArray(np.ascontiguousarray(data["total"]), "states/A^3",
                            "Local density of states in the window, both spins", "volumetric",
                            dict(meta)),
        "eigenvalues": StoredArray(data["eigen"], "eV",
                                   "Kohn-Sham eigenvalues of the LDOS step [spin, k, band], "
                                   "absolute", "table", {"axes": ["spin", "kpoint", "band"]}),
        "kpoint_weights": StoredArray(data["weights"], "", "k-point weights, sum 1", "table",
                                      {}),
    }
    if spec.spin_channels == "resolved":
        outcome.arrays["ldos_spin"] = StoredArray(
            np.ascontiguousarray(np.asarray(arrays["ldos_spin"], dtype=float)), "states/A^3",
            "Local density of states per spin [up, down]", "volumetric",
            dict(meta, axes=["spin", "a", "b", "c"]))
    checksums = {name: stored.sha256() for name, stored in outcome.arrays.items()}
    stm_ok, stm_reason = stm_supported(gs)
    top = float(np.asarray(gs.positions_A, dtype=float)[:, 2].max())
    summary = {
        "status": STATUS_COMPLETE, "fermi_level_eV": data["fermi_level_eV"],
        "window_eV": [spec.energy_min_eV, spec.energy_max_eV],
        "window_absolute_eV": [data["fermi_level_eV"] + spec.energy_min_eV,
                               data["fermi_level_eV"] + spec.energy_max_eV],
        "spin_channels": spec.spin_channels, "n_spins": spec.n_spins,
        "n_bands": spec.n_bands, "n_kpoints": int(len(data["weights"])),
        "grid_shape": list(data["shape"]), "cell_A": data["cell"].tolist(),
        "states_in_window": data["count"], "map_integral_states": data["integral"],
        "pseudo_norm_ratio": (data["integral"] / data["count"]) if data["count"] else None,
        "augmentation_radii_A": data["radii"],
        "max_augmentation_radius_A": max(data["radii"].values()) if data["radii"] else 0.0,
        "highest_band_above_EF_eV": data["highest_above_EF"],
        "max_ldos": float(data["total"].max()),
        "stm": {"supported": stm_ok, "reason": stm_reason, "surface_z_A": top},
        "symmetry": dict(LDOS_SYMMETRY),
        "nscf_parameters_used": ldos.get("nscf_parameters_used"), "used": ldos.get("used"),
        "array_sha256": checksums, "started_unix": started_unix, "finished_unix": time.time(),
        "wall_time_s": outcome.wall_time_s,
        "units": {"ldos": "states per A^3 in the window, both spins", "energy": "eV",
                  "length": "A"},
        "experimental_comparison": None,
    }
    template = staged.results["energy"].provenance
    convergence = Convergence(
        converged=True, iterations=int(run.iterations),
        residual=abs(float(ldos["states_in_window"]) - data["count"]),
        residual_metric="difference between the worker's state count in the window and the "
                        "count recomputed from the returned eigenvalues and weights",
        tolerance=COUNT_TOLERANCE,
        message=(f"Ground state converged in {run.iterations} SCF iterations; every band of "
                 f"the LDOS step converged; {data['count']:.6g} states in the window, the map "
                 f"integrating to {data['integral']:.6g}."))
    record = Result("dft_ldos", summary, "states/A^3", ground._copy_provenance(template),
                    convergence=convergence)
    record.extra.update(base_extra)
    results = dict(staged.results)
    results["ldos"] = record
    parameters = {"ldos_spec": spec.as_dict(), "ldos_spec_digest": spec.digest,
                  "nscf_parameters": job["ldos"]["nscf"],
                  "nscf_parameters_used": ldos.get("nscf_parameters_used"),
                  "ldos_settings": job["ldos"], "ldos_used": ldos.get("used"),
                  "source": dict(spec.source), "array_sha256": checksums}
    approximations = _approximations(spec)
    for item in results.values():
        item.provenance.model = MODEL
        item.provenance.fidelity = Fidelity.TIER3_EXTERNAL
        item.provenance.origin = Origin.CALCULATED
        item.provenance.inputs_digest = spec.digest
        item.provenance.parameters.update(parameters)
        item.provenance.references = list(item.provenance.references) + [
            r for r in REFERENCES if r not in item.provenance.references]
        item.provenance.approximations = list(item.provenance.approximations) + [
            a for a in approximations if a not in item.provenance.approximations]
        item.extra["warnings"] = list(warnings)
    outcome.results = results
    outcome.status = STATUS_COMPLETE
    outcome.warnings = warnings
    return outcome
