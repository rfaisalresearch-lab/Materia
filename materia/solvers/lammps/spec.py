"""The LAMMPS run specification: every variable a run depends on, frozen.

A :class:`LAMMPSRunSpec` is one LAMMPS calculation described completely
enough to be repeated: the atoms with their stable identifiers, elements,
isotopic masses, positions and velocities; the cell and boundary conditions;
the mapping from elements and masses to LAMMPS atom types; the potential
style, file and SHA-256; the unit style; the task and every control it uses;
the outputs requested; and the exact LAMMPS installation it was frozen for.

It is immutable and versioned (:data:`SPEC_SCHEMA`, :data:`SPEC_VERSION`).
:func:`build` creates one from a structure and a dictionary of settings and
refuses unknown or mistyped settings; :func:`check` lists every reason the
specification must not run, keyed by variable.  A variable that does not
apply to the chosen task is stored as ``None`` rather than carried along
silently.

Units are explicit in every field name that has one.  The specification is
written in Materia units; :mod:`materia.solvers.lammps.native` converts to
LAMMPS ``metal`` units when it writes the input files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...core_model.cell import Cell
from ...core_model.structure import Structure
from ...elements import periodic_table as pt
from ...version import __version__
from .native import BoxError, box_geometry

SPEC_SCHEMA = "materia.lammps.run"
SPEC_VERSION = "1.0"

TASKS: Tuple[str, ...] = ("energy", "relax", "md")
ENSEMBLES: Tuple[str, ...] = ("nve", "langevin", "nvt")
MIN_STYLES: Tuple[str, ...] = ("cg", "fire", "sd")
INITIAL_VELOCITIES: Tuple[str, ...] = ("keep", "create")
OUTPUTS: Tuple[str, ...] = ("energy", "forces", "stress", "thermo", "trajectory")
POTENTIAL_STYLES: Tuple[str, ...] = ("eam/alloy",)
UNITS = "metal"
ATOM_STYLE = "atomic"

MAX_ATOM_ID = 2 ** 31 - 1
MAX_FRAMES = 20_000
MAX_THERMO_ROWS = 20_000
MAX_TRAJECTORY_BYTES = 1 << 30
MAX_TIMESTEP_FS = 10.0
MAX_SEED = 900_000_000

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "energy": {"neighbor_skin_A": 2.0, "outputs": None},
    "relax": {"neighbor_skin_A": 2.0, "outputs": None, "min_style": "cg",
              "ftol_eV_A": 0.01, "etol": 0.0, "max_iterations": 1000,
              "max_evaluations": 10000},
    "md": {"neighbor_skin_A": 2.0, "outputs": None, "steps": 200, "timestep_fs": 1.0,
           "ensemble": "nve", "temperature_K": 300.0, "damping_fs": 100.0,
           "seed": 12345, "initial_velocities": "keep", "pressure_bar": None,
           "sample_every": 10},
}

FIELD_NOTES: Dict[str, str] = {
    "neighbor_skin_A": "Neighbour-list skin added to the potential cutoff, angstrom.",
    "outputs": "Quantities to extract: energy, forces, stress, thermo, trajectory.",
    "min_style": "LAMMPS minimiser: cg (Polak-Ribiere conjugate gradient), fire or sd.",
    "ftol_eV_A": "Force tolerance: the largest force on any mobile atom, eV/A "
                 "(min_modify norm max).",
    "etol": "Relative energy tolerance, dimensionless; 0 disables it so that only the "
            "force criterion can stop the minimisation.",
    "max_iterations": "Maximum minimiser iterations.",
    "max_evaluations": "Maximum force evaluations.",
    "steps": "Number of MD time steps.",
    "timestep_fs": "MD time step, femtoseconds.",
    "ensemble": "nve (velocity Verlet), langevin (nve plus a Langevin thermostat) or "
                "nvt (Nose-Hoover chain).",
    "temperature_K": "Target temperature for langevin and nvt, and for created "
                     "velocities, kelvin.",
    "damping_fs": "Thermostat relaxation time, femtoseconds.",
    "seed": "Random seed for the Langevin force and for created velocities, 1 to "
            f"{MAX_SEED}.",
    "initial_velocities": "keep (the structure's velocities) or create (Gaussian at "
                          "temperature_K with zero total momentum).",
    "pressure_bar": "Target pressure. Must be None: this adapter keeps the cell fixed "
                    "and has no barostat.",
    "sample_every": "Store one trajectory frame and one thermodynamic row every this "
                    "many steps.",
}


class SpecError(ValueError):
    """A setting that is unknown, of the wrong type, or not settable."""


class SpecRefused(ValueError):
    """A specification that must not be run, carrying every reason."""

    def __init__(self, report: "SpecReport") -> None:
        super().__init__("; ".join(report.messages()))
        self.report = report


@dataclass
class SpecReport:
    blocking: Dict[str, List[str]] = field(default_factory=dict)
    warnings: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.blocking

    def block(self, variable: str, message: str) -> None:
        self.blocking.setdefault(variable, []).append(message)

    def warn(self, variable: str, message: str) -> None:
        self.warnings.setdefault(variable, []).append(message)

    def messages(self) -> List[str]:
        return [f"{k}: {m}" for k, items in self.blocking.items() for m in items]

    def as_dict(self) -> dict:
        return {"ok": self.ok, "blocking": {k: list(v) for k, v in self.blocking.items()},
                "warnings": {k: list(v) for k, v in self.warnings.items()},
                "messages": self.messages()}


@dataclass(frozen=True)
class LAMMPSRunSpec:
    """One LAMMPS calculation, frozen.  See the module docstring."""

    task: str
    structure_key: Optional[str]
    input_state_digest: str
    formula: str
    atom_ids: Tuple[int, ...]
    numbers: Tuple[int, ...]
    mass_numbers: Tuple[int, ...]
    roles: Tuple[str, ...]
    atom_types: Tuple[int, ...]
    type_table: Tuple[Tuple[int, str, int, float], ...]
    positions_A: Tuple[Tuple[float, float, float], ...]
    velocities_A_fs: Tuple[Tuple[float, float, float], ...]
    fixed_ids: Tuple[int, ...]
    cell_A: Tuple[Tuple[float, float, float], ...]
    pbc: Tuple[bool, bool, bool]
    boundary: Tuple[str, str, str]
    potential: Dict[str, Any]
    lammps: Dict[str, Any]
    outputs: Tuple[str, ...]
    neighbor_skin_A: float
    thermo_every: int
    min_style: Optional[str] = None
    ftol_eV_A: Optional[float] = None
    etol: Optional[float] = None
    max_iterations: Optional[int] = None
    max_evaluations: Optional[int] = None
    steps: Optional[int] = None
    timestep_fs: Optional[float] = None
    ensemble: Optional[str] = None
    temperature_K: Optional[float] = None
    damping_fs: Optional[float] = None
    seed: Optional[int] = None
    initial_velocities: Optional[str] = None
    pressure_bar: Optional[float] = None
    sample_every: Optional[int] = None
    collision: Dict[str, Any] = field(default_factory=dict)
    units: str = UNITS
    atom_style: str = ATOM_STYLE
    schema: str = SPEC_SCHEMA
    version: str = SPEC_VERSION
    created_with: str = __version__

    def as_dict(self) -> dict:
        data = asdict(self)
        data["type_table"] = [list(row) for row in self.type_table]
        return json.loads(json.dumps(data))

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "LAMMPSRunSpec":
        if data.get("schema") != SPEC_SCHEMA:
            raise SpecError(f"Not a LAMMPS run specification: schema {data.get('schema')!r}.")
        if str(data.get("version")) != SPEC_VERSION:
            raise SpecError(f"LAMMPS run specification version {data.get('version')} is "
                            f"not {SPEC_VERSION}, the version this Materia reads.")
        known = {f.name for f in fields(LAMMPSRunSpec)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise SpecError(f"Unknown field(s) in the specification: {', '.join(unknown)}.")
        kwargs = dict(data)
        for name in ("atom_ids", "numbers", "mass_numbers", "atom_types", "fixed_ids"):
            kwargs[name] = tuple(int(v) for v in data[name])
        kwargs["roles"] = tuple(str(v) for v in data["roles"])
        kwargs["outputs"] = tuple(str(v) for v in data["outputs"])
        kwargs["type_table"] = tuple((int(r[0]), str(r[1]), int(r[2]), float(r[3]))
                                     for r in data["type_table"])
        for name in ("positions_A", "velocities_A_fs", "cell_A"):
            kwargs[name] = tuple(tuple(float(x) for x in row) for row in data[name])
        kwargs["pbc"] = tuple(bool(v) for v in data["pbc"])
        kwargs["boundary"] = tuple(str(v) for v in data["boundary"])
        kwargs["potential"] = dict(data["potential"])
        kwargs["lammps"] = dict(data["lammps"])
        kwargs["collision"] = dict(data.get("collision") or {})
        return LAMMPSRunSpec(**kwargs)

    @property
    def digest(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    @property
    def n_atoms(self) -> int:
        return len(self.atom_ids)

    def positions(self) -> np.ndarray:
        return np.array(self.positions_A, dtype=float).reshape(self.n_atoms, 3)

    def velocities(self) -> np.ndarray:
        return np.array(self.velocities_A_fs, dtype=float).reshape(self.n_atoms, 3)

    def cell(self) -> Cell:
        return Cell(np.array(self.cell_A, dtype=float), tuple(self.pbc))

    def masses_amu(self) -> np.ndarray:
        by_type = {row[0]: row[3] for row in self.type_table}
        return np.array([by_type[t] for t in self.atom_types], dtype=float)

    def fixed_mask(self) -> np.ndarray:
        fixed = set(self.fixed_ids)
        return np.array([i in fixed for i in self.atom_ids], dtype=bool)

    def structure(self) -> Structure:
        """The frozen structure, rebuilt."""
        s = Structure(list(self.numbers), self.positions(), self.cell(),
                      ids=list(self.atom_ids), mass_numbers=list(self.mass_numbers),
                      roles=list(self.roles))
        s.velocities = self.velocities()
        s.fixed[:] = self.fixed_mask()
        if self.collision:
            s.info["collision"] = dict(self.collision)
        return s

    def settings(self) -> Dict[str, Any]:
        """The task controls, as they were frozen."""
        return {k: getattr(self, k) for k in DEFAULTS[self.task]}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def potential_record(potential) -> Dict[str, Any]:
    """What a specification pins about a Materia EAM potential object."""
    identity = potential.identity
    path = str(identity.path)
    return {
        "style": "eam/alloy", "id": identity.id, "file_name": identity.file_name,
        "path": path, "sha256": identity.sha256,
        "size_bytes": os.path.getsize(path) if os.path.isfile(path) else None,
        "elements": list(potential.elements), "cutoff_A": float(potential.cutoff_A),
        "shipped": bool(identity.shipped), "license": identity.license,
        "title": identity.title, "citations": list(identity.citations),
    }


def _coerce(task: str, name: str, value: Any) -> Any:
    if value is None:
        return None
    try:
        if name in ("max_iterations", "max_evaluations", "steps", "seed", "sample_every"):
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise TypeError
            return int(value)
        if name in ("neighbor_skin_A", "ftol_eV_A", "etol", "timestep_fs",
                    "temperature_K", "damping_fs", "pressure_bar"):
            if isinstance(value, bool):
                raise TypeError
            return float(value)
        if name in ("min_style", "ensemble", "initial_velocities"):
            if not isinstance(value, str):
                raise TypeError
            return value.strip().lower()
        if name == "outputs":
            if isinstance(value, str):
                value = [value]
            return tuple(str(v).strip().lower() for v in value)
    except (TypeError, ValueError):
        raise SpecError(f"{name} for a {task} run cannot be {value!r}.") from None
    return value


def default_outputs(task: str, pbc: Sequence[bool]) -> Tuple[str, ...]:
    out = ["energy", "forces"]
    if all(pbc):
        out.append("stress")
    out.append("thermo")
    if task == "md":
        out.append("trajectory")
    return tuple(out)


def _type_table(structure: Structure, file_elements: Sequence[str]
                ) -> Tuple[Tuple[Tuple[int, str, int, float], ...], Tuple[int, ...]]:
    masses = structure.masses()
    symbols = structure.symbols()
    order = {e: k for k, e in enumerate(file_elements)}
    keys = sorted({(symbols[k], float(masses[k])) for k in range(len(structure))},
                  key=lambda item: (order.get(item[0], len(order)), item[0], item[1]))
    table = tuple((index + 1, symbol, int(pt.element(symbol).number), mass)
                  for index, (symbol, mass) in enumerate(keys))
    lookup = {(row[1], row[3]): row[0] for row in table}
    types = tuple(lookup[(symbols[k], float(masses[k]))] for k in range(len(structure)))
    return table, types


def build(structure: Structure, task: str, potential, environment,
          structure_key: Optional[str] = None,
          settings: Optional[Mapping[str, Any]] = None, **overrides) -> LAMMPSRunSpec:
    """Freeze a structure, a potential and task settings into a specification.

    ``potential`` is a Materia :class:`~materia.physics.eam.EAMPotential`, whose
    file identity and checksum are pinned.  ``environment`` is the
    :class:`~materia.solvers.lammps.environment.LAMMPSEnvironment` the run is
    meant for; an unavailable one is pinned as unavailable and :func:`check`
    refuses it.  Raises :class:`SpecError` for an unknown task or setting, or
    a setting of the wrong type.  Nothing is refused on physical grounds here;
    :func:`check` does that.
    """
    from ...python_api.api import structure_state_digest

    if task not in TASKS:
        raise SpecError(f"Unknown LAMMPS task {task!r}. Available: {', '.join(TASKS)}.")
    given = dict(settings or {})
    given.update(overrides)
    allowed = DEFAULTS[task]
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise SpecError(f"Unknown setting(s) for a LAMMPS {task} run: {', '.join(unknown)}. "
                        f"Known: {', '.join(sorted(allowed))}.")
    merged = dict(allowed)
    for name, value in given.items():
        merged[name] = _coerce(task, name, value)
    pbc = tuple(bool(p) for p in structure.cell.pbc)
    if merged.get("outputs") is None:
        merged["outputs"] = default_outputs(task, pbc)
    else:
        merged["outputs"] = tuple(dict.fromkeys(merged["outputs"]))
    if task == "energy":
        thermo_every = 1
    elif task == "relax":
        thermo_every = max(1, int(math.ceil(max(1, merged["max_iterations"] or 1)
                                            / MAX_THERMO_ROWS)))
    else:
        thermo_every = max(1, int(merged.get("sample_every") or 1))
    table, types = _type_table(structure, list(potential.elements))
    ids = tuple(int(i) for i in structure.ids)
    fixed = tuple(int(i) for i, f in zip(structure.ids, structure.fixed) if f)
    kwargs = {k: merged.get(k) for k in DEFAULTS[task] if k not in ("outputs",)}
    return LAMMPSRunSpec(
        task=task, structure_key=structure_key,
        input_state_digest=structure_state_digest(structure),
        formula=structure.formula(), atom_ids=ids,
        numbers=tuple(int(z) for z in structure.numbers),
        mass_numbers=tuple(int(a) for a in structure.mass_numbers),
        roles=tuple(str(r) for r in structure.roles), atom_types=types, type_table=table,
        positions_A=tuple(tuple(float(x) for x in row) for row in structure.positions),
        velocities_A_fs=tuple(tuple(float(x) for x in row) for row in structure.velocities),
        fixed_ids=fixed,
        cell_A=tuple(tuple(float(x) for x in row) for row in structure.cell.matrix),
        pbc=pbc, boundary=tuple("p" if p else "s" for p in pbc),
        potential=potential_record(potential),
        lammps=environment.identity() if environment.available else
        {"kind": "", "path": "", "version": "", "available": False},
        outputs=merged["outputs"], thermo_every=thermo_every,
        collision=dict(structure.info.get("collision") or {}), **kwargs)


def changed(spec: LAMMPSRunSpec, **values) -> LAMMPSRunSpec:
    """A new specification with some task settings changed and re-coerced."""
    allowed = set(DEFAULTS[spec.task])
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise SpecError(f"Only task settings can be changed; not: {', '.join(unknown)}.")
    coerced = {k: _coerce(spec.task, k, v) for k, v in values.items()}
    if "outputs" in coerced and coerced["outputs"] is None:
        coerced["outputs"] = default_outputs(spec.task, spec.pbc)
    new = replace(spec, **coerced)
    if spec.task == "md" and "sample_every" in coerced:
        new = replace(new, thermo_every=max(1, int(new.sample_every or 1)))
    return new


def _finite(values) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values, dtype=float))))


def check(spec: LAMMPSRunSpec, environment=None, verify_potential: bool = True
          ) -> SpecReport:
    """Every reason ``spec`` must not run, and every warning, keyed by variable."""
    report = SpecReport()
    if spec.schema != SPEC_SCHEMA or spec.version != SPEC_VERSION:
        report.block("schema", f"Specification {spec.schema} {spec.version} is not "
                               f"{SPEC_SCHEMA} {SPEC_VERSION}.")
    if spec.task not in TASKS:
        report.block("task", f"Unknown task {spec.task!r}.")
    if spec.units != UNITS:
        report.block("units", f"Only LAMMPS units {UNITS} are supported; {spec.units!r} "
                              "is not, because the EAM files are tabulated in metal units.")
    if spec.atom_style != ATOM_STYLE:
        report.block("atom_style", f"Only atom_style {ATOM_STYLE} is supported; "
                                   f"{spec.atom_style!r} is refused.")
    n = spec.n_atoms
    if n == 0:
        report.block("structure", "The structure has no atoms.")
    for name in ("numbers", "mass_numbers", "roles", "atom_types", "positions_A",
                 "velocities_A_fs"):
        if len(getattr(spec, name)) != n:
            report.block("structure", f"{name} has {len(getattr(spec, name))} entries for "
                                      f"{n} atoms.")
    ids = list(spec.atom_ids)
    if len(set(ids)) != len(ids):
        report.block("atom_ids", "Atom identifiers are not unique.")
    if ids and (min(ids) < 1 or max(ids) > MAX_ATOM_ID):
        report.block("atom_ids", f"LAMMPS atom identifiers must lie between 1 and "
                                 f"{MAX_ATOM_ID}; this structure has {min(ids)} to "
                                 f"{max(ids)}.")
    if not set(spec.fixed_ids) <= set(ids):
        report.block("fixed_ids", "A fixed atom is not in the structure.")
    types_seen = {row[0] for row in spec.type_table}
    if sorted(types_seen) != list(range(1, len(spec.type_table) + 1)):
        report.block("type_table", "Atom types must be numbered 1 to N without gaps.")
    if not set(spec.atom_types) <= types_seen:
        report.block("atom_types", "An atom has a type that is not in the type table.")
    for type_id, element, z, mass in spec.type_table:
        if not (math.isfinite(mass) and mass > 0):
            report.block("masses", f"Type {type_id} ({element}) has no usable mass "
                                   f"({mass!r}). LAMMPS needs a positive mass for every "
                                   "type.")
        if pt.element(element).number != z:
            report.block("type_table", f"Type {type_id} names {element} with atomic "
                                       f"number {z}.")
    by_type = {row[0]: row for row in spec.type_table}
    for k, (z, t) in enumerate(zip(spec.numbers, spec.atom_types)):
        if t in by_type and by_type[t][2] != z:
            report.block("atom_types", f"Atom {spec.atom_ids[k]} is element {z} but "
                                       f"type {t} is {by_type[t][1]}.")
            break
    if n and not _finite(spec.positions_A):
        report.block("positions_A", "A position is not a finite number.")
    if n and not _finite(spec.velocities_A_fs):
        report.block("velocities_A_fs", "A velocity is not a finite number.")
    if not _finite(spec.cell_A):
        report.block("cell_A", "The cell is not finite.")
    expected = tuple("p" if p else "s" for p in spec.pbc)
    if tuple(spec.boundary) != expected:
        report.block("boundary", f"Boundary {' '.join(spec.boundary)} does not match the "
                                 f"structure's periodicity {list(spec.pbc)}; Materia maps "
                                 "periodic to p and non-periodic to s (shrink-wrapped).")
    for flag in spec.boundary:
        if flag not in ("p", "s"):
            report.block("boundary", f"Boundary style {flag!r} is refused: only p and s, "
                                     "which cannot lose atoms, are supported.")
    if n and _finite(spec.positions_A) and _finite(spec.cell_A):
        try:
            box_geometry(np.array(spec.cell_A), spec.pbc, spec.positions())
        except BoxError as exc:
            report.block("cell_A", str(exc))

    potential = spec.potential
    if potential.get("style") not in POTENTIAL_STYLES:
        report.block("potential", f"Potential style {potential.get('style')!r} is not "
                                  f"supported; supported: {', '.join(POTENTIAL_STYLES)}.")
    present = sorted({row[1] for row in spec.type_table})
    missing = [e for e in present if e not in potential.get("elements", [])]
    if missing:
        report.block("potential", f"{potential.get('id')} describes "
                                  f"{', '.join(potential.get('elements', []))} only; the "
                                  f"structure also contains {', '.join(missing)}.")
    if verify_potential:
        path = potential.get("path", "")
        if not path or not os.path.isfile(path):
            report.block("potential", f"The potential file {path or '(none)'} does not exist.")
        elif file_sha256(path) != potential.get("sha256"):
            report.block("potential", f"{potential.get('file_name')} no longer matches the "
                                      "SHA-256 frozen into this specification.")

    outputs = set(spec.outputs)
    unknown = sorted(outputs - set(OUTPUTS))
    if unknown:
        report.block("outputs", f"Unknown output(s): {', '.join(unknown)}.")
    if "energy" not in outputs or "forces" not in outputs:
        report.block("outputs", "energy and forces are always extracted and cannot be "
                                "removed from the outputs.")
    if "stress" in outputs and not all(spec.pbc):
        report.block("outputs", "A stress tensor needs a cell periodic in all three "
                                "directions; this structure is not, so its volume is "
                                "undefined.")
    if "trajectory" in outputs and spec.task != "md":
        report.block("outputs", "A trajectory is produced only by an md run.")
    if not (spec.neighbor_skin_A is not None and math.isfinite(spec.neighbor_skin_A)
            and spec.neighbor_skin_A > 0):
        report.block("neighbor_skin_A", "The neighbour skin must be a positive distance.")

    if spec.task == "relax":
        if spec.min_style not in MIN_STYLES:
            report.block("min_style", f"min_style must be one of {', '.join(MIN_STYLES)}.")
        if not (spec.ftol_eV_A is not None and math.isfinite(spec.ftol_eV_A)
                and spec.ftol_eV_A > 0):
            report.block("ftol_eV_A", "The force tolerance must be a positive number.")
        if not (spec.etol is not None and math.isfinite(spec.etol) and spec.etol >= 0):
            report.block("etol", "The energy tolerance must be zero or positive.")
        if not (spec.max_iterations or 0) >= 1:
            report.block("max_iterations", "At least one iteration is needed.")
        if (spec.max_evaluations or 0) < (spec.max_iterations or 0):
            report.block("max_evaluations", "max_evaluations must be at least "
                                            "max_iterations.")
        if spec.fixed_ids and len(spec.fixed_ids) == n:
            report.block("fixed_ids", "Every atom is fixed; there is nothing to relax.")
        if spec.etol:
            report.warn("etol", "A non-zero energy tolerance can stop the minimiser "
                                "before the force tolerance is met; such a run is reported "
                                "as not converged.")
    if spec.task == "md":
        _check_md(spec, report)
    if spec.fixed_ids and spec.task == "md":
        moving = [i for i, v in zip(spec.atom_ids, spec.velocities_A_fs)
                  if i in set(spec.fixed_ids) and any(x != 0.0 for x in v)]
        if moving and spec.initial_velocities == "keep":
            report.block("velocities_A_fs", f"{len(moving)} fixed atom(s) have a non-zero "
                                            "velocity; a fixed atom cannot move.")
    _check_environment(spec, environment, report)
    return report


def _check_md(spec: LAMMPSRunSpec, report: SpecReport) -> None:
    steps = spec.steps or 0
    if steps < 1:
        report.block("steps", "An md run needs at least one step.")
    if not (spec.timestep_fs is not None and math.isfinite(spec.timestep_fs)
            and 0 < spec.timestep_fs <= MAX_TIMESTEP_FS):
        report.block("timestep_fs", f"The time step must lie in (0, {MAX_TIMESTEP_FS:g}] fs.")
    if spec.ensemble not in ENSEMBLES:
        report.block("ensemble", f"ensemble must be one of {', '.join(ENSEMBLES)}.")
    if spec.pressure_bar is not None:
        report.block("pressure_bar", "Pressure control is refused: this adapter runs at "
                                     "fixed cell and does not drive a barostat. Leave "
                                     "pressure_bar as None.")
    if spec.initial_velocities not in INITIAL_VELOCITIES:
        report.block("initial_velocities", "initial_velocities must be keep or create.")
    temperature = spec.temperature_K
    thermostatted = spec.ensemble in ("langevin", "nvt")
    needs_temperature = thermostatted or spec.initial_velocities == "create"
    if temperature is None or not math.isfinite(temperature) or temperature < 0:
        report.block("temperature_K", "temperature_K must be a finite number, at least 0.")
    elif needs_temperature and temperature <= 0:
        report.block("temperature_K", "A thermostat or created velocities need a "
                                      "temperature above 0 K. Use ensemble nve with "
                                      "initial_velocities keep for constant energy from "
                                      "the given velocities.")
    if thermostatted and not (spec.damping_fs is not None and math.isfinite(spec.damping_fs)
                              and spec.damping_fs > 0):
        report.block("damping_fs", "The thermostat relaxation time must be positive.")
    if thermostatted and spec.timestep_fs and spec.damping_fs and \
            spec.damping_fs < 10 * spec.timestep_fs:
        report.warn("damping_fs", "A relaxation time under ten time steps couples the "
                                  "thermostat very strongly.")
    needs_seed = spec.ensemble == "langevin" or spec.initial_velocities == "create"
    if needs_seed and not (spec.seed is not None and 1 <= spec.seed <= MAX_SEED):
        report.block("seed", f"LAMMPS needs a seed between 1 and {MAX_SEED}.")
    every = spec.sample_every or 0
    if every < 1:
        report.block("sample_every", "sample_every must be at least 1.")
    elif steps >= 1:
        if steps % every:
            report.block("sample_every", f"steps ({steps}) must be a multiple of "
                                         f"sample_every ({every}) so that every stored "
                                         "frame has a matching thermodynamic row and the "
                                         "last step is stored.")
        frames = steps // every + 1
        if frames > MAX_FRAMES:
            report.block("sample_every", f"{frames} frames exceed the limit of "
                                         f"{MAX_FRAMES}; sample less often.")
        size = frames * spec.n_atoms * 3 * 8 * 2
        if "trajectory" in spec.outputs and size > MAX_TRAJECTORY_BYTES:
            report.block("sample_every", f"The trajectory would need {size / 2**20:.0f} "
                                         f"MiB, above the {MAX_TRAJECTORY_BYTES // 2**20} MiB "
                                         "limit; sample less often.")
    if spec.fixed_ids and len(spec.fixed_ids) == spec.n_atoms:
        report.block("fixed_ids", "Every atom is fixed; there is nothing to integrate.")
    if not all(spec.pbc) and spec.ensemble == "nvt":
        report.warn("ensemble", "Nose-Hoover dynamics of a finite system also couples to "
                                "overall rotation, which is not removed.")


def required_styles(spec: LAMMPSRunSpec) -> Dict[str, List[str]]:
    required: Dict[str, List[str]] = {"atom": [ATOM_STYLE], "pair": [spec.potential.get(
        "style", "eam/alloy")]}
    fixes: List[str] = []
    if spec.fixed_ids and spec.task in ("relax", "md"):
        fixes.append("setforce")
    if spec.task == "relax" and spec.min_style:
        required["minimize"] = [spec.min_style]
    if spec.task == "md":
        fixes.append("nvt" if spec.ensemble == "nvt" else "nve")
        if spec.ensemble == "langevin":
            fixes.append("langevin")
    if fixes:
        required["fix"] = fixes
    return required


def _check_environment(spec: LAMMPSRunSpec, environment, report: SpecReport) -> None:
    from .environment import INSTALL_HINT, discover

    environment = environment if environment is not None else discover()
    if not environment.available:
        report.block("lammps", f"{environment.blocking_reason()} {INSTALL_HINT}")
        return
    pinned = spec.lammps or {}
    if not pinned.get("version"):
        report.block("lammps", "This specification was frozen when no LAMMPS was "
                               "available. Freeze it again now that LAMMPS "
                               f"{environment.version} is present.")
        return
    if (pinned.get("path"), pinned.get("version"), pinned.get("kind")) != \
            (environment.path, environment.version, environment.kind):
        report.block("lammps", f"The specification is pinned to LAMMPS "
                               f"{pinned.get('version')} at {pinned.get('path')}, but the "
                               f"LAMMPS found now is {environment.version} at "
                               f"{environment.path}. Freeze the specification again.")
        return
    missing = environment.missing_styles(required_styles(spec))
    if missing:
        report.block("lammps", f"LAMMPS {environment.version} at {environment.path} was "
                               f"built without {', '.join(missing)}. Rebuild it with the "
                               "packages that provide them (pair eam/alloy is in MANYBODY) "
                               "or install the conda-forge build.")


def describe(spec: LAMMPSRunSpec, report: Optional[SpecReport] = None) -> dict:
    """The specification for display, without per-atom arrays."""
    report = report or check(spec)
    return {
        "schema": spec.schema, "version": spec.version, "digest": spec.digest,
        "task": spec.task, "units": spec.units, "atom_style": spec.atom_style,
        "formula": spec.formula, "n_atoms": spec.n_atoms,
        "structure_key": spec.structure_key, "input_state_digest": spec.input_state_digest,
        "pbc": list(spec.pbc), "boundary": list(spec.boundary),
        "cell_A": [list(r) for r in spec.cell_A],
        "types": [{"type": r[0], "element": r[1], "atomic_number": r[2], "mass_amu": r[3]}
                  for r in spec.type_table],
        "fixed_atoms": len(spec.fixed_ids),
        "potential": dict(spec.potential), "lammps": dict(spec.lammps),
        "outputs": list(spec.outputs), "thermo_every": spec.thermo_every,
        "settings": spec.settings(), "notes": {k: FIELD_NOTES.get(k, "")
                                               for k in spec.settings()},
        "check": report.as_dict(),
    }
