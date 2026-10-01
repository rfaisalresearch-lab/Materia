"""The ground-state DFT experiment specification.

A :class:`GroundStateSpec` is one self-consistent Kohn-Sham calculation,
described completely enough to be repeated: the frozen geometry, the
electronic state asked for, every numerical parameter, the observables to
extract, and the software and PAW datasets it is pinned to.  It is immutable
and versioned; :func:`changed` returns a new specification rather than editing
one.

Every variable carries a unit, an explanation, a support rule and a
validation rule (:data:`FIELDS` and :func:`check`).  :func:`gpaw_parameters`
turns a specification into the exact keyword arguments handed to
``gpaw.GPAW``; nothing is added or dropped on the way, so a variable that
changes also changes the calculation GPAW performs.  A variable the selected
formulation cannot honour is refused with the reason, never accepted and
ignored.

Nuclear masses are the one deliberate exception, and it is physics rather than
a limitation: the Born-Oppenheimer electronic ground state does not depend on
nuclear mass.  Isotopes are recorded, passed to the atoms GPAW receives, and
affect nuclear dynamics and vibrations; they do not change any number this
calculation produces, and the interface says so.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...elements import periodic_table as pt
from ...provenance import digest as short_digest
from ...version import __version__
from . import boundary as bc
from .datasets import (
    FUNCTIONAL_NOTES, FUNCTIONALS, DatasetError, available_bases,
    available_functionals, dataset_package, select,
)

SPEC_SCHEMA = "materia.dft.ground-state"
SPEC_VERSION = "1.0"

REPRESENTATIONS: Tuple[str, ...] = ("fd", "pw", "lcao")
REPRESENTATION_NOTES = {
    "fd": "Real-space finite-difference grid. Works for every boundary class; "
          "accuracy is set by the grid spacing.",
    "pw": "Plane waves. Periodic in all three directions by construction, so only "
          "for bulk crystals; accuracy is set by the kinetic-energy cutoff. The "
          "only representation in which GPAW computes the stress tensor.",
    "lcao": "Linear combination of atomic orbitals. Fast, but a finite atom-centred "
            "basis is not a variational limit.",
}
OCCUPATIONS: Tuple[str, ...] = ("fixed", "fermi-dirac", "marzari-vanderbilt")
POISSON: Tuple[str, ...] = ("gpaw-default", "moment-corrected", "dipole-layer")
CHARGED_PERIODIC: Tuple[str, ...] = ("none", "uniform-background")

OBSERVABLES: Tuple[str, ...] = (
    "energy", "forces", "stress", "density", "spin_density",
    "electrostatic_potential", "eigenvalues", "occupations", "fermi_level",
    "magnetic_moment",
)
ALWAYS_RECORDED: Tuple[str, ...] = ("charge_accounting", "scf_history", "convergence")

OBSERVABLE_NOTES = {
    "energy": "Total energy: the free energy E - TS that forces are derivatives of, "
              "and its zero-width extrapolation.",
    "forces": "Hellmann-Feynman forces on every atom, eV/A.",
    "stress": "Stress tensor from the plane-wave calculation, eV/A^3.",
    "density": "All-electron density including the frozen core, on GPAW's fine grid.",
    "spin_density": "All-electron spin density n_up - n_down.",
    "electrostatic_potential": "Electrostatic potential energy of an electron, "
                               "-e phi, on the fine grid (GPAW convention).",
    "eigenvalues": "Kohn-Sham eigenvalues per spin, k-point and band.",
    "occupations": "Occupation of each Kohn-Sham state.",
    "fermi_level": "Fermi level. With integer occupations GPAW places it between "
                   "the highest occupied and lowest unoccupied state.",
    "magnetic_moment": "Total magnetic moment, and local moments integrated in atomic "
                       "spheres.",
}

MAX_VOLUME_BYTES = 256 * 1024 * 1024

MAX_FIELD_V_PER_A = 2.0
ADVISED_FIELD_V_PER_A = 0.5

K_DENSITY_A = 30.0
STRESS_CUTOFF_EV = 600.0


class SpecError(ValueError):
    """A variable that is unknown, of the wrong type or not settable."""


class SpecRefused(ValueError):
    """A specification that must not be run, with every reason."""

    def __init__(self, report: "SpecReport") -> None:
        super().__init__("; ".join(report.blocking))
        self.report = report


@dataclass(frozen=True)
class FieldInfo:
    """What one variable means, and when it applies."""

    name: str
    section: str
    label: str
    unit: str
    explanation: str
    applies: str = "always"
    advanced: bool = False
    settable: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


SECTIONS = (
    ("identity", "Structure identity"),
    ("system", "Physical system"),
    ("electronic", "Electronic state"),
    ("numerical", "Numerical accuracy"),
    ("observables", "Requested observables"),
    ("software", "Software and datasets"),
)

FIELDS: Tuple[FieldInfo, ...] = (
    FieldInfo("structure_key", "identity", "structure", "",
              "Project slot the geometry was taken from; empty for a structure the "
              "project does not hold.", settable=False),
    FieldInfo("formula", "identity", "formula", "", "Chemical formula of the frozen "
              "geometry.", settable=False),
    FieldInfo("geometry_digest", "identity", "geometry digest", "SHA-256",
              "Fingerprint of ids, elements, positions, cell, periodicity, formal "
              "charges and magnetic moments. A result is current while the "
              "structure still has this fingerprint.", settable=False),
    FieldInfo("atom_ids", "identity", "atom ids", "", "Stable Materia atom ids, in "
              "the order GPAW receives the atoms.", settable=False),
    FieldInfo("numbers", "identity", "atomic numbers", "", "Nuclear charges Z.",
              settable=False),
    FieldInfo("positions_A", "identity", "positions", "A", "Cartesian positions.",
              settable=False),
    FieldInfo("mass_numbers", "identity", "isotopes", "mass number",
              "Isotope of each atom; 0 means the natural-abundance mixture. Sets the "
              "nuclear masses. The Born-Oppenheimer electronic ground state does not "
              "depend on nuclear mass, so no number in this calculation changes with "
              "the isotope; vibrations and dynamics do.", settable=False),
    FieldInfo("formal_charges_e", "identity", "formal charges", "e",
              "Per-atom formal charges labelled on the structure. Recorded for identity; "
              "only the net charge enters the calculation.", settable=False),
    FieldInfo("cell_A", "identity", "cell vectors", "A", "Lattice vectors as rows.",
              settable=False),
    FieldInfo("pbc", "identity", "periodic axes", "", "Which lattice directions are "
              "periodic.", settable=False),
    FieldInfo("boundary", "system", "boundary class", "",
              "cluster, wire, slab or bulk, from the periodic axes.", settable=False),
    FieldInfo("vacuum_axis", "system", "vacuum direction", "",
              "For a slab, the open axis along the surface normal.", settable=False),
    FieldInfo("charge_e", "system", "net charge", "e",
              "Net charge of the whole system in units of the elementary charge: +1 "
              "removes one electron. A whole number. Defaults to the sum of the formal "
              "charges labelled on the atoms; the calculation uses this value."),
    FieldInfo("charged_periodic_policy", "system", "charged cell treatment", "",
              "What neutralises a charged bulk cell. 'uniform-background' is the "
              "homogeneous compensating charge GPAW adds to every charged periodic "
              "cell; choosing it makes that treatment explicit, and a charged bulk cell "
              "without it is refused. It leaves a finite-size error that decays only "
              "slowly with cell size. Allowed only for a charged bulk cell.",
              applies="charged bulk only"),
    FieldInfo("poisson", "system", "electrostatic boundary", "",
              "'moment-corrected' removes the monopole and dipole of a cluster "
              "analytically so the zero boundary condition does not distort them; "
              "'dipole-layer' cancels the artificial field across the vacuum of an "
              "asymmetric slab; 'gpaw-default' uses GPAW's own solver unchanged.",
              applies="moment-corrected: cluster; dipole-layer: slab"),
    FieldInfo("external_field_V_per_A", "system", "external electric field", "V/A",
              "Uniform static field vector. Only along open directions: any direction "
              "for a cluster, the surface normal for a slab. Zero for none.",
              applies="fd and lcao, open directions only", advanced=True),
    FieldInfo("spin_polarized", "electronic", "spin polarisation", "",
              "Treat the two spin channels independently. Required for an odd "
              "electron count in a finite system and for any non-zero initial moment."),
    FieldInfo("initial_magnetic_moments_muB", "electronic", "initial magnetic moments",
              "mu_B", "Starting moment on each atom. A starting guess, not a "
              "constraint: the self-consistent solution decides the final moments.",
              applies="spin-polarised only"),
    FieldInfo("occupations", "electronic", "occupations", "",
              "'fixed' fills states with whole electrons; 'fermi-dirac' and "
              "'marzari-vanderbilt' smear them, which metals need to converge."),
    FieldInfo("smearing_eV", "electronic", "smearing width", "eV",
              "Width of the occupation smearing. Must be 0 for fixed occupations and "
              "positive otherwise.", applies="fermi-dirac, marzari-vanderbilt"),
    FieldInfo("n_bands", "electronic", "number of bands", "bands",
              "Kohn-Sham states per spin and k-point. Empty lets GPAW choose; a value "
              "below the number of occupied states is refused.", advanced=True),
    FieldInfo("xc", "numerical", "exchange-correlation functional", "",
              "Only functionals with a PAW dataset for every element are offered. The "
              "functional's systematic error is not reduced by any numerical setting."),
    FieldInfo("representation", "numerical", "representation", "",
              "fd (real-space grid), pw (plane waves, bulk only) or lcao (atomic "
              "orbitals)."),
    FieldInfo("grid_spacing_A", "numerical", "grid spacing", "A",
              "Real-space grid spacing. Smaller is more accurate and more expensive.",
              applies="fd and lcao"),
    FieldInfo("cutoff_eV", "numerical", "plane-wave cutoff", "eV",
              "Kinetic-energy cutoff of the plane-wave basis.", applies="pw"),
    FieldInfo("basis", "numerical", "LCAO basis", "", "Atomic-orbital basis set.",
              applies="lcao"),
    FieldInfo("kpoints", "numerical", "k-point grid", "divisions",
              "Brillouin-zone sampling along a, b, c. Must be 1 along every open "
              "direction."),
    FieldInfo("kpoints_gamma_centered", "numerical", "Gamma-centred grid", "",
              "Include the Gamma point. With odd divisions a Monkhorst-Pack grid is "
              "already Gamma-centred and both choices give the same points.",
              advanced=True),
    FieldInfo("energy_tol_eV_per_electron", "numerical", "energy tolerance",
              "eV/electron", "Largest change of the total energy per valence electron "
              "over the last three SCF iterations."),
    FieldInfo("density_tol_electrons_per_electron", "numerical", "density tolerance",
              "electrons/electron", "Integrated absolute density change per valence "
              "electron.", advanced=True),
    FieldInfo("eigenstates_tol_eV2_per_electron", "numerical", "eigenstate tolerance",
              "eV^2/electron", "Integrated squared Kohn-Sham residual per valence "
              "electron.", advanced=True),
    FieldInfo("forces_tol_eV_A", "numerical", "force tolerance", "eV/A",
              "Optional: also require the forces to stop changing by more than this. "
              "Empty for none.", advanced=True),
    FieldInfo("max_scf_iterations", "numerical", "SCF iteration limit", "iterations",
              "A run that has not converged by then is reported as not converged and "
              "yields no result."),
    FieldInfo("random_seed", "numerical", "random seed", "",
              "Not applicable: GPAW starts from atomic densities and LCAO "
              "wavefunctions, which are deterministic. Must be empty.", advanced=True),
    FieldInfo("observables", "observables", "requested observables", "",
              "Quantities to extract. Charge accounting, SCF history and the "
              "convergence record are always kept."),
    FieldInfo("software", "software", "software", "", "GPAW, ASE and Python versions "
              "and the dataset distribution this specification is pinned to.",
              settable=False),
    FieldInfo("paw_datasets", "software", "PAW datasets", "SHA-256",
              "Element, file and SHA-256 of every PAW dataset.", settable=False),
    FieldInfo("basis_files", "software", "LCAO basis files", "SHA-256",
              "Element, file and SHA-256 of every basis file.", settable=False),
)

FIELD_INDEX = {f.name: f for f in FIELDS}
SETTABLE = tuple(f.name for f in FIELDS if f.settable)


@dataclass(frozen=True)
class GroundStateSpec:
    """One ground-state calculation, frozen.  See :data:`FIELDS`."""

    structure_key: Optional[str]
    formula: str
    geometry_digest: str
    atom_ids: Tuple[int, ...]
    numbers: Tuple[int, ...]
    positions_A: Tuple[Tuple[float, float, float], ...]
    mass_numbers: Tuple[int, ...]
    formal_charges_e: Tuple[float, ...]
    cell_A: Tuple[Tuple[float, float, float], ...]
    pbc: Tuple[bool, bool, bool]
    boundary: str
    vacuum_axis: Optional[int]
    charge_e: float
    charged_periodic_policy: str
    poisson: str
    external_field_V_per_A: Tuple[float, float, float]
    spin_polarized: bool
    initial_magnetic_moments_muB: Tuple[float, ...]
    occupations: str
    smearing_eV: float
    n_bands: Optional[int]
    xc: str
    representation: str
    grid_spacing_A: Optional[float]
    cutoff_eV: Optional[float]
    basis: Optional[str]
    kpoints: Tuple[int, int, int]
    kpoints_gamma_centered: bool
    energy_tol_eV_per_electron: float
    density_tol_electrons_per_electron: float
    eigenstates_tol_eV2_per_electron: float
    forces_tol_eV_A: Optional[float]
    max_scf_iterations: int
    random_seed: Optional[int]
    observables: Tuple[str, ...]
    software: Tuple[Tuple[str, str], ...]
    paw_datasets: Tuple[Tuple[str, str, str], ...]
    basis_files: Tuple[Tuple[str, str, str], ...]
    schema: str = SPEC_SCHEMA
    version: str = SPEC_VERSION

    def as_dict(self) -> dict:
        out: Dict[str, Any] = {}
        for f in fields(self):
            out[f.name] = _plain(getattr(self, f.name))
        return out

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "GroundStateSpec":
        """Rebuild a specification saved with a project or sent over HTTP."""
        if data.get("schema", SPEC_SCHEMA) != SPEC_SCHEMA:
            raise SpecError(f"Not a ground-state specification: {data.get('schema')!r}.")
        version = str(data.get("version", SPEC_VERSION))
        if version != SPEC_VERSION:
            raise SpecError(
                f"Specification version {version} is not understood by this Materia "
                f"(it reads {SPEC_VERSION}).")
        names = {f.name for f in fields(GroundStateSpec)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise SpecError(f"Unknown specification field(s): {', '.join(unknown)}.")
        missing = sorted(n for n in names - set(data) if n not in ("schema", "version"))
        if missing:
            raise SpecError(f"Missing specification field(s): {', '.join(missing)}.")
        values = dict(data)
        values["atom_ids"] = tuple(int(v) for v in values["atom_ids"])
        values["numbers"] = tuple(int(v) for v in values["numbers"])
        values["positions_A"] = tuple(tuple(float(x) for x in row)
                                      for row in values["positions_A"])
        values["mass_numbers"] = tuple(int(v) for v in values["mass_numbers"])
        values["formal_charges_e"] = tuple(float(v) for v in values["formal_charges_e"])
        values["cell_A"] = tuple(tuple(float(x) for x in row) for row in values["cell_A"])
        values["pbc"] = tuple(bool(v) for v in values["pbc"])
        values["external_field_V_per_A"] = tuple(float(v) for v in
                                                 values["external_field_V_per_A"])
        values["initial_magnetic_moments_muB"] = tuple(
            float(v) for v in values["initial_magnetic_moments_muB"])
        values["kpoints"] = tuple(int(v) for v in values["kpoints"])
        values["observables"] = tuple(str(v) for v in values["observables"])
        values["software"] = tuple((str(k), str(v)) for k, v in values["software"])
        values["paw_datasets"] = tuple(tuple(str(x) for x in row)
                                       for row in values["paw_datasets"])
        values["basis_files"] = tuple(tuple(str(x) for x in row)
                                      for row in values["basis_files"])
        values["schema"] = SPEC_SCHEMA
        values["version"] = SPEC_VERSION
        return GroundStateSpec(**values)

    @property
    def digest(self) -> str:
        """SHA-256 of the whole specification: the experiment's identity."""
        import hashlib

        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def short_digest(self) -> str:
        return self.digest[:16]

    @property
    def symbols(self) -> List[str]:
        return [pt.symbol(int(z)) for z in self.numbers]

    @property
    def n_atoms(self) -> int:
        return len(self.numbers)

    def units(self) -> Dict[str, str]:
        return {f.name: f.unit for f in FIELDS if f.unit}

    def settings(self) -> Dict[str, Any]:
        """Only the settable variables, as plain data."""
        return {name: _plain(getattr(self, name)) for name in SETTABLE}


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def geometry_digest(structure) -> str:
    """Fingerprint of everything in a structure that the ground state depends on.

    Isotopes are deliberately left out: nuclear masses do not enter the
    Born-Oppenheimer electronic problem, so changing one leaves a result
    current.  :func:`isotope_digest` fingerprints them separately.
    """
    return short_digest({
        "ids": [int(i) for i in structure.ids],
        "numbers": [int(z) for z in structure.numbers],
        "positions": np.round(np.asarray(structure.positions, dtype=float), 10).tolist(),
        "cell": np.round(np.asarray(structure.cell.matrix, dtype=float), 10).tolist(),
        "pbc": [bool(p) for p in structure.cell.pbc],
        "formal_charges": np.round(np.asarray(structure.formal_charges, float), 10).tolist(),
        "magnetic_moments": np.round(np.asarray(structure.magnetic_moments, float),
                                     10).tolist(),
    })


def isotope_digest(structure) -> str:
    return short_digest({"ids": [int(i) for i in structure.ids],
                         "mass_numbers": [int(a) for a in structure.mass_numbers]})


def describe_geometry_change(spec: "GroundStateSpec", structure) -> str:
    """What differs between the frozen geometry and a structure, in words."""
    if structure is None:
        return "the structure is no longer in the project"
    if len(structure) != spec.n_atoms:
        return f"the atom count changed from {spec.n_atoms} to {len(structure)}"
    changed = []
    if [int(i) for i in structure.ids] != list(spec.atom_ids):
        changed.append("atom ids or ordering")
    if [int(z) for z in structure.numbers] != list(spec.numbers):
        changed.append("elements")
    if not np.allclose(np.asarray(structure.positions, float),
                       np.asarray(spec.positions_A), rtol=0.0, atol=1e-10):
        changed.append("atom positions")
    if (not np.allclose(np.asarray(structure.cell.matrix, float),
                        np.asarray(spec.cell_A), rtol=0.0, atol=1e-10)
            or tuple(bool(p) for p in structure.cell.pbc) != spec.pbc):
        changed.append("cell or periodicity")
    if not np.allclose(np.asarray(structure.formal_charges, float),
                       np.asarray(spec.formal_charges_e), rtol=0.0, atol=1e-10):
        changed.append("formal charges")
    if not changed:
        changed.append("initial magnetic moments")
    if len(changed) == 1:
        return f"the {changed[0]} changed"
    return "the " + ", ".join(changed[:-1]) + f" and {changed[-1]} changed"


def default_kpoints(cell: np.ndarray, pbc: Sequence[bool],
                    density_A: float = K_DENSITY_A) -> Tuple[int, int, int]:
    """Divisions giving at least ``density_A`` of k times lattice length."""
    lengths = np.linalg.norm(np.asarray(cell, dtype=float), axis=1)
    out = []
    for axis in range(3):
        if not pbc[axis] or lengths[axis] <= 0:
            out.append(1)
        else:
            out.append(int(min(16, max(1, math.ceil(density_A / lengths[axis])))))
    return tuple(out)


def _environment(environment=None):
    if environment is not None:
        return environment
    from ...solvers.gpaw_driver.environment import discover

    return discover()


def software_record(environment) -> Tuple[Tuple[str, str], ...]:
    paths = list(getattr(environment, "setup_paths", []) or [])
    return tuple(sorted({
        "gpaw": str(getattr(environment, "gpaw_version", "") or ""),
        "ase": str(getattr(environment, "ase_version", "") or ""),
        "gpaw_python": str(getattr(environment, "python_version", "") or ""),
        "interpreter": str(getattr(environment, "interpreter", "") or ""),
        "paw_dataset_package": dataset_package(paths) if paths else "",
        "setup_paths": ":".join(paths),
        "materia": __version__,
    }.items()))


def _electron_parity_counts(numbers: Sequence[int], charge: float) -> float:
    return float(sum(int(z) for z in numbers)) - float(charge)


def build(structure, environment=None, structure_key: Optional[str] = None,
          **variables: Any) -> GroundStateSpec:
    """A specification for ``structure`` with safe defaults, then ``variables``.

    Defaults depend on the boundary class: a cluster gets a real-space grid,
    open boundaries and a moment-corrected Poisson solver; a slab gets a
    real-space grid, in-plane k-points, metallic smearing and a dipole layer;
    a bulk crystal gets plane waves, k-points and metallic smearing.  An odd
    electron count, or any non-zero moment on the structure, turns spin
    polarisation on with the structure's moments or one Bohr magneton shared
    over the atoms.

    Spin polarisation defaults on when the electron count, after any
    requested net charge, is odd; its starting moments then default to one
    Bohr magneton shared over the atoms unless moments were given or are set
    on the structure.

    The result is not validated here; :func:`check` says whether it may run.
    Only settable variables may be given, with the right types.
    """
    environment = _environment(environment)
    unknown = sorted(set(variables) - set(SETTABLE))
    if unknown:
        fixed = sorted(set(unknown) & set(FIELD_INDEX))
        if fixed:
            raise SpecError(
                f"{', '.join(fixed)} come(s) from the structure or the environment and "
                "cannot be set; edit the structure instead.")
        raise SpecError(
            f"Unknown variable(s): {', '.join(unknown)}. Settable: "
            f"{', '.join(SETTABLE)}.")

    positions = np.asarray(structure.positions, dtype=float)
    cell = np.asarray(structure.cell.matrix, dtype=float)
    pbc = tuple(bool(p) for p in structure.cell.pbc)
    report = bc.inspect(positions, cell, pbc)
    boundary = report.boundary
    numbers = tuple(int(z) for z in structure.numbers)
    symbols = [pt.symbol(z) for z in numbers]
    charge = float(structure.total_charge())
    moments = tuple(float(m) for m in structure.magnetic_moments)
    electrons = _electron_parity_counts(numbers, charge)
    odd = abs(electrons - round(electrons)) < 1e-9 and int(round(electrons)) % 2 == 1

    paths = list(getattr(environment, "setup_paths", []) or [])
    functionals = available_functionals(symbols, paths) if paths else []
    xc = "PBE" if ("PBE" in functionals or not functionals) else functionals[0]

    periodic_default = boundary != "cluster"
    defaults: Dict[str, Any] = {
        "charge_e": charge,
        "charged_periodic_policy": "none",
        "poisson": {"cluster": "moment-corrected", "slab": "dipole-layer"}.get(
            boundary, "gpaw-default"),
        "external_field_V_per_A": (0.0, 0.0, 0.0),
        "spin_polarized": bool(odd or any(abs(m) > 1e-12 for m in moments)),
        "initial_magnetic_moments_muB": moments,
        "occupations": "fermi-dirac" if periodic_default else "fixed",
        "smearing_eV": 0.1 if periodic_default else 0.0,
        "n_bands": None,
        "xc": xc,
        "representation": "pw" if boundary == "bulk" else "fd",
        "grid_spacing_A": None if boundary == "bulk" else 0.20,
        "cutoff_eV": 400.0 if boundary == "bulk" else None,
        "basis": None,
        "kpoints": default_kpoints(cell, pbc),
        "kpoints_gamma_centered": True,
        "energy_tol_eV_per_electron": 5.0e-4,
        "density_tol_electrons_per_electron": 1.0e-4,
        "eigenstates_tol_eV2_per_electron": 4.0e-8,
        "forces_tol_eV_A": None,
        "max_scf_iterations": 333,
        "random_seed": None,
    }
    requested_representation = variables.get("representation", defaults["representation"])
    if requested_representation in ("fd", "lcao") and "representation" in variables:
        defaults["cutoff_eV"] = None
        defaults["grid_spacing_A"] = 0.20
    if requested_representation == "pw" and "representation" in variables:
        defaults["grid_spacing_A"] = None
        defaults["cutoff_eV"] = 400.0
    if requested_representation == "lcao":
        defaults["basis"] = "dzp"
    requested_occupations = variables.get("occupations", defaults["occupations"])
    if "occupations" in variables:
        defaults["smearing_eV"] = 0.0 if requested_occupations == "fixed" else 0.1

    values = dict(defaults)
    for name, value in variables.items():
        values[name] = _coerce(name, value, len(numbers))
    if "charge_e" in variables and "spin_polarized" not in variables:
        electrons = _electron_parity_counts(numbers, values["charge_e"])
        odd = abs(electrons - round(electrons)) < 1e-9 and int(round(electrons)) % 2 == 1
        values["spin_polarized"] = bool(odd or any(abs(m) > 1e-12 for m in moments))
    electrons = _electron_parity_counts(numbers, values["charge_e"])
    odd = abs(electrons - round(electrons)) < 1e-9 and int(round(electrons)) % 2 == 1
    if (values["spin_polarized"] and odd and "initial_magnetic_moments_muB" not in variables
            and not any(abs(m) > 1e-12 for m in values["initial_magnetic_moments_muB"])):
        share = 1.0 / max(1, len(numbers))
        values["initial_magnetic_moments_muB"] = tuple(share for _ in numbers)

    spin = values["spin_polarized"]
    observables = variables.get("observables")
    if observables is None:
        chosen = ["energy", "forces", "density", "electrostatic_potential",
                  "eigenvalues", "occupations", "fermi_level"]
        if spin:
            chosen += ["spin_density", "magnetic_moment"]
        if values["representation"] == "pw":
            chosen.append("stress")
        values["observables"] = tuple(sorted(chosen))
    else:
        values["observables"] = _coerce("observables", observables, len(numbers))

    selection_paw: Tuple[Tuple[str, str, str], ...] = ()
    selection_basis: Tuple[Tuple[str, str, str], ...] = ()
    if paths:
        try:
            chosen_sets = select(symbols, str(values["xc"]), paths,
                                 basis=values["basis"] if values["representation"] == "lcao"
                                 else None)
            selection_paw = tuple((d.symbol, d.path, d.sha256) for d in chosen_sets.datasets)
            selection_basis = tuple((b.symbol, b.path, b.sha256) for b in chosen_sets.basis)
        except DatasetError:
            pass

    return GroundStateSpec(
        structure_key=structure_key,
        formula=structure.formula(),
        geometry_digest=geometry_digest(structure),
        atom_ids=tuple(int(i) for i in structure.ids),
        numbers=numbers,
        positions_A=tuple(tuple(float(x) for x in row) for row in positions),
        mass_numbers=tuple(int(a) for a in structure.mass_numbers),
        formal_charges_e=tuple(float(q) for q in structure.formal_charges),
        cell_A=tuple(tuple(float(x) for x in row) for row in cell),
        pbc=pbc,
        boundary=boundary,
        vacuum_axis=report.vacuum_axis,
        charge_e=float(values["charge_e"]),
        charged_periodic_policy=str(values["charged_periodic_policy"]),
        poisson=str(values["poisson"]),
        external_field_V_per_A=tuple(values["external_field_V_per_A"]),
        spin_polarized=bool(values["spin_polarized"]),
        initial_magnetic_moments_muB=tuple(values["initial_magnetic_moments_muB"]),
        occupations=str(values["occupations"]),
        smearing_eV=float(values["smearing_eV"]),
        n_bands=values["n_bands"],
        xc=str(values["xc"]),
        representation=str(values["representation"]),
        grid_spacing_A=values["grid_spacing_A"],
        cutoff_eV=values["cutoff_eV"],
        basis=values["basis"],
        kpoints=tuple(values["kpoints"]),
        kpoints_gamma_centered=bool(values["kpoints_gamma_centered"]),
        energy_tol_eV_per_electron=float(values["energy_tol_eV_per_electron"]),
        density_tol_electrons_per_electron=float(
            values["density_tol_electrons_per_electron"]),
        eigenstates_tol_eV2_per_electron=float(values["eigenstates_tol_eV2_per_electron"]),
        forces_tol_eV_A=values["forces_tol_eV_A"],
        max_scf_iterations=int(values["max_scf_iterations"]),
        random_seed=values["random_seed"],
        observables=values["observables"],
        software=software_record(environment),
        paw_datasets=selection_paw,
        basis_files=selection_basis,
    )


def changed(spec: GroundStateSpec, environment=None, **variables: Any) -> GroundStateSpec:
    """A new specification with some settable variables changed.

    Dataset identity is re-resolved when the functional or basis changes, so a
    changed functional is pinned to its own datasets.
    """
    unknown = sorted(set(variables) - set(SETTABLE))
    if unknown:
        raise SpecError(f"Cannot change {', '.join(unknown)}: not a settable variable.")
    values = {name: _coerce(name, value, spec.n_atoms) for name, value in variables.items()}
    updated = replace(spec, **values)
    if ("xc" in values or "basis" in values or "representation" in values):
        environment = _environment(environment)
        paths = list(getattr(environment, "setup_paths", []) or [])
        paw: Tuple[Tuple[str, str, str], ...] = ()
        basis_files: Tuple[Tuple[str, str, str], ...] = ()
        if paths:
            try:
                sets = select(updated.symbols, updated.xc, paths,
                              basis=updated.basis if updated.representation == "lcao"
                              else None)
                paw = tuple((d.symbol, d.path, d.sha256) for d in sets.datasets)
                basis_files = tuple((b.symbol, b.path, b.sha256) for b in sets.basis)
            except DatasetError:
                pass
        updated = replace(updated, paw_datasets=paw, basis_files=basis_files)
    return updated


def _number(name: str, value: Any, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer,
                                                         np.floating)):
        raise SpecError(f"{name} must be a number in {unit}, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise SpecError(f"{name} must be finite, got {value!r}.")
    return number


def _optional_number(name: str, value: Any, unit: str) -> Optional[float]:
    return None if value is None else _number(name, value, unit)


def _integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise SpecError(f"{name} must be a whole number, got {value!r}.")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    if not isinstance(value, (int, np.integer)):
        raise SpecError(f"{name} must be a whole number, got {value!r}.")
    return int(value)


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise SpecError(
            f"{name} must be true or false, got {type(value).__name__} {value!r}. It "
            "is not coerced: bool('false') is True.")
    return bool(value)


def _choice(name: str, value: Any, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise SpecError(f"{name} must be one of {', '.join(choices)}, got {value!r}.")
    return value


def _coerce(name: str, value: Any, n_atoms: int) -> Any:
    """Type-check one settable variable without reinterpreting it."""
    if name == "charge_e":
        return _number(name, value, "e")
    if name == "charged_periodic_policy":
        return _choice(name, value, CHARGED_PERIODIC)
    if name == "poisson":
        return _choice(name, value, POISSON)
    if name == "external_field_V_per_A":
        if value is None:
            return (0.0, 0.0, 0.0)
        try:
            items = list(value)
        except TypeError:
            raise SpecError(f"{name} must be a vector of three components in V/A, "
                            f"got {value!r}.") from None
        if len(items) != 3:
            raise SpecError(f"{name} must have three components, got {len(items)}.")
        return tuple(_number(name, v, "V/A") + 0.0 for v in items)
    if name == "spin_polarized":
        return _boolean(name, value)
    if name == "initial_magnetic_moments_muB":
        try:
            items = list(value)
        except TypeError:
            raise SpecError(f"{name} must list one moment per atom in mu_B.") from None
        if len(items) != n_atoms:
            raise SpecError(f"{name} must list {n_atoms} moments, one per atom, got "
                            f"{len(items)}.")
        return tuple(_number(name, v, "mu_B") + 0.0 for v in items)
    if name == "occupations":
        return _choice(name, value, OCCUPATIONS)
    if name == "smearing_eV":
        return _number(name, value, "eV")
    if name == "n_bands":
        return None if value is None else _integer(name, value)
    if name == "xc":
        return _choice(name, value, FUNCTIONALS)
    if name == "representation":
        return _choice(name, value, REPRESENTATIONS)
    if name == "grid_spacing_A":
        return _optional_number(name, value, "A")
    if name == "cutoff_eV":
        return _optional_number(name, value, "eV")
    if name == "basis":
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise SpecError(f"basis must name an LCAO basis set, got {value!r}.")
        return value.strip()
    if name == "kpoints":
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            raise SpecError("kpoints must give three divisions, one per lattice "
                            "direction, not a single number.")
        try:
            items = list(value)
        except TypeError:
            raise SpecError(f"kpoints must be three divisions, got {value!r}.") from None
        if len(items) != 3:
            raise SpecError(f"kpoints must be three divisions, got {len(items)}.")
        return tuple(_integer("kpoints", v) for v in items)
    if name == "kpoints_gamma_centered":
        return _boolean(name, value)
    if name in ("energy_tol_eV_per_electron", "density_tol_electrons_per_electron",
                "eigenstates_tol_eV2_per_electron"):
        return _number(name, value, FIELD_INDEX[name].unit)
    if name == "forces_tol_eV_A":
        return _optional_number(name, value, "eV/A")
    if name == "max_scf_iterations":
        return _integer(name, value)
    if name == "random_seed":
        return None if value is None else _integer(name, value)
    if name == "observables":
        if isinstance(value, str):
            raise SpecError("observables must be a list of names, not one string.")
        try:
            items = [str(v) for v in value]
        except TypeError:
            raise SpecError(f"observables must be a list, got {value!r}.") from None
        unknown = sorted(set(items) - set(OBSERVABLES))
        if unknown:
            extra = ""
            if set(unknown) & set(ALWAYS_RECORDED):
                extra = (f" {', '.join(sorted(set(unknown) & set(ALWAYS_RECORDED)))} "
                         "is always recorded and need not be requested.")
            raise SpecError(f"Unknown observable(s): {', '.join(unknown)}. Available: "
                            f"{', '.join(OBSERVABLES)}.{extra}")
        return tuple(sorted(set(items)))
    raise SpecError(f"{name} is not a settable variable.")


@dataclass
class SpecReport:
    """Whether a specification may run, and why not."""

    blocking: List[str]
    warnings: List[str]
    by_field: Dict[str, List[str]]
    boundary: Dict[str, Any]
    electrons: Dict[str, Any]
    options: Dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def as_dict(self) -> dict:
        return {"ok": self.ok, "blocking": list(self.blocking),
                "warnings": list(self.warnings),
                "by_field": {k: list(v) for k, v in self.by_field.items()},
                "boundary": self.boundary, "electrons": self.electrons,
                "options": self.options}


def check(spec: GroundStateSpec, environment=None) -> SpecReport:
    """Every support and validation rule, applied to one specification.

    Returns every reason at once, keyed by the variable it concerns, so the
    interface can show each refusal next to its control.
    """
    environment = _environment(environment)
    by_field: Dict[str, List[str]] = {}
    warnings: List[str] = []

    def refuse(name: str, text: str) -> None:
        by_field.setdefault(name, []).append(text)

    general: List[str] = []
    if not getattr(environment, "operational", False):
        reason = (environment.blocking_reason() if hasattr(environment, "blocking_reason")
                  else "GPAW is not available.")
        general.append(reason)

    n = spec.n_atoms
    if n == 0:
        general.append("The structure has no atoms.")
    positions = np.asarray(spec.positions_A, dtype=float).reshape(-1, 3)
    cell = np.asarray(spec.cell_A, dtype=float)
    if not np.isfinite(positions).all() or not np.isfinite(cell).all():
        general.append("The geometry has non-finite coordinates.")
    report = bc.inspect(positions, cell, spec.pbc)
    for text in report.problems:
        refuse("boundary", text)
    warnings.extend(report.warnings)
    if report.boundary != spec.boundary:
        refuse("boundary", f"The specification says {spec.boundary!r} but the cell's "
                           f"periodic axes make it a {report.boundary}.")

    if n and len(positions) > 1:
        from ...core_model.cell import Cell
        from ...physics.neighbors import COLLISION_TOLERANCE_A, find_coincidence

        hit = find_coincidence(positions, Cell(cell, spec.pbc), COLLISION_TOLERANCE_A)
        if hit is not None:
            refuse("boundary", f"Atoms #{spec.atom_ids[hit.i]} and #{spec.atom_ids[hit.j]} "
                               f"are {hit.distance:.3g} A apart. Two nuclei cannot occupy "
                               "the same point.")

    paths = list(getattr(environment, "setup_paths", []) or [])
    symbols = spec.symbols
    datasets = None
    if paths and n:
        try:
            datasets = select(symbols, spec.xc, paths,
                              basis=spec.basis if spec.representation == "lcao" else None)
        except DatasetError as exc:
            refuse("xc" if "dataset" in str(exc) else "basis", str(exc))
        if datasets is not None:
            pinned = {(s, p, h) for s, p, h in spec.paw_datasets}
            found = {(d.symbol, d.path, d.sha256) for d in datasets.datasets}
            if pinned != found:
                refuse("paw_datasets",
                       "The PAW datasets on disk differ from the ones this "
                       "specification was pinned to (file or SHA-256). Rebuild the "
                       "specification so it names the datasets that will be used.")
            pinned_basis = {(s, p, h) for s, p, h in spec.basis_files}
            found_basis = {(b.symbol, b.path, b.sha256) for b in datasets.basis}
            if pinned_basis != found_basis:
                refuse("basis_files", "The LCAO basis files on disk differ from the "
                                      "ones this specification was pinned to.")
    if getattr(environment, "operational", False):
        current = dict(software_record(environment))
        pinned_software = dict(spec.software)
        for key in ("gpaw", "ase", "paw_dataset_package"):
            if pinned_software.get(key, "") != current.get(key, ""):
                refuse("software",
                       f"The specification was pinned to {key} "
                       f"{pinned_software.get(key) or 'unknown'}, but "
                       f"{current.get(key) or 'nothing'} is installed now. Rebuild "
                       "the specification for the software that will run it.")

    boundary = spec.boundary
    representation = spec.representation

    if representation == "pw":
        if boundary != "bulk":
            refuse("representation",
                   "Plane waves are periodic in all three directions by construction, "
                   f"and this {boundary} is open along "
                   f"{', '.join(bc.AXES[i] for i in report.open_axes)}. GPAW would "
                   "silently make it periodic. Use fd or lcao.")
        if spec.cutoff_eV is None:
            refuse("cutoff_eV", "A plane-wave calculation needs a cutoff in eV.")
        if spec.grid_spacing_A is not None:
            refuse("grid_spacing_A", "The grid spacing does not apply to plane waves, "
                                     "which set their grid from the cutoff. Leave it empty.")
        if spec.basis is not None:
            refuse("basis", "An LCAO basis does not apply to plane waves. Leave it empty.")
    else:
        if spec.grid_spacing_A is None:
            refuse("grid_spacing_A", f"The {representation} representation needs a grid "
                                     "spacing in A.")
        if spec.cutoff_eV is not None:
            refuse("cutoff_eV", "A plane-wave cutoff does not apply to the "
                                f"{representation} representation. Leave it empty.")
        if representation == "lcao" and spec.basis is None:
            refuse("basis", "The lcao representation needs a basis set, for example dzp.")
        if representation == "fd" and spec.basis is not None:
            refuse("basis", "A basis set does not apply to the fd representation. Leave "
                            "it empty.")

    if spec.cutoff_eV is not None:
        if spec.cutoff_eV < 100.0 or spec.cutoff_eV > 3000.0:
            refuse("cutoff_eV", f"cutoff_eV = {spec.cutoff_eV:g} eV is outside 100 to "
                                "3000 eV; below 100 eV a plane-wave basis cannot "
                                "represent a PAW pseudo-wavefunction at all.")
        elif spec.cutoff_eV < 300.0:
            warnings.append(f"A {spec.cutoff_eV:g} eV cutoff is below the 300 to 500 eV "
                            "usually needed with GPAW's PAW datasets; check convergence.")
    if spec.grid_spacing_A is not None:
        if not (0.05 <= spec.grid_spacing_A <= 0.35):
            refuse("grid_spacing_A", f"grid_spacing_A = {spec.grid_spacing_A:g} A is "
                                     "outside 0.05 to 0.35 A. Coarser than 0.35 A is not "
                                     "converged in any useful sense.")
        elif spec.grid_spacing_A > 0.25:
            warnings.append(f"A {spec.grid_spacing_A:g} A grid is coarse; energies can "
                            "be off by 0.1 eV per atom or more. Check convergence.")

    for axis, divisions in enumerate(spec.kpoints):
        if divisions < 1:
            refuse("kpoints", f"kpoints needs at least one division along "
                              f"{bc.AXES[axis]}, got {divisions}.")
        elif divisions > 1 and not spec.pbc[axis]:
            refuse("kpoints", f"{divisions} divisions along {bc.AXES[axis]}, which is "
                              "open. There is no Brillouin zone along an open direction; "
                              "use 1.")
        elif divisions > 32:
            refuse("kpoints", f"{divisions} divisions along {bc.AXES[axis]} is beyond "
                              "the 32 this interface allows.")
    if boundary != "cluster" and all(k == 1 for k in spec.kpoints):
        warnings.append("Only the Gamma point samples a periodic direction. Unless the "
                        "cell is large this is not converged; a k-point convergence "
                        "study measures by how much.")
    automatic = default_kpoints(np.asarray(spec.cell_A, dtype=float), spec.pbc)
    if boundary != "cluster" and tuple(spec.kpoints) == automatic and \
            any(k > 1 for k in spec.kpoints):
        warnings.append(
            f"The k-point grid {'x'.join(str(k) for k in spec.kpoints)} is the automatic "
            f"choice for this cell size (k times lattice length at least {K_DENSITY_A:g} A). "
            "A cell of a different size can get a different automatic grid, and total "
            "energies from different grids are not comparable. Set kpoints explicitly "
            "and keep it fixed when comparing strained, scaled or relaxed cells.")

    charge = spec.charge_e
    carried = float(sum(spec.formal_charges_e))
    if abs(charge - round(charge)) > 1e-9:
        refuse("charge_e", f"charge_e = {charge:+g} e is not a whole number. A "
                           "fractional net charge describes an ensemble or a doping "
                           "model, not the ground state of one system.")
    if abs(carried - charge) > 1e-9:
        warnings.append(
            f"This calculation uses a net charge of {charge:+g} e, while the formal charges "
            f"labelled on the atoms add up to {carried:+g} e. The Kohn-Sham calculation "
            "takes only the total electron count, sum(Z) minus the net charge; per-atom "
            "formal charges are labels for point-charge models and do not enter it.")
    charged = abs(charge) > 1e-9
    policy = spec.charged_periodic_policy
    if charged and boundary == "bulk":
        if policy != "uniform-background":
            refuse("charged_periodic_policy",
                   f"A bulk cell with net charge {charge:+g} e has infinite electrostatic "
                   "energy unless something neutralises it. Choose "
                   "'uniform-background' explicitly; the finite-size error that "
                   "leaves is reported with the result.")
        else:
            length = abs(float(np.linalg.det(cell))) ** (1.0 / 3.0)
            estimate = 14.399645 * charge * charge * 2.8373 / (2.0 * length)
            warnings.append(
                f"Charged bulk cell with a uniform compensating background: the "
                f"leading finite-size (Makov-Payne) error for a point charge in vacuum "
                f"is about {estimate:.2f} eV for this {length:.2f} A cell and decays "
                "only as 1/L. It is not corrected. Dielectric screening reduces it by "
                "the dielectric constant; defect energies need an explicit correction.")
    elif policy != "none":
        if not charged:
            refuse("charged_periodic_policy",
                   "A background policy was chosen for a neutral system, where it would "
                   "do nothing. Set it to 'none'.")
        else:
            refuse("charged_periodic_policy",
                   f"'{policy}' applies only to a charged bulk cell. For a {boundary} "
                   "it would be ignored.")
    if charged and boundary in ("slab", "wire"):
        refuse("charge_e",
               f"A charged {boundary} is not supported: with open boundaries along some "
               "directions and periodic ones along others, GPAW's grid pins the "
               "potential to zero at the vacuum faces, which models charged plates "
               "rather than an isolated charged surface. No correction for that "
               "geometry is validated here.")

    poisson = spec.poisson
    if poisson == "moment-corrected" and boundary != "cluster":
        refuse("poisson", "The moment-corrected solver applies to a cluster, which is "
                          f"open in every direction; this is a {boundary}.")
    if poisson == "dipole-layer":
        if boundary != "slab":
            refuse("poisson", f"A dipole layer applies only to a slab; this is a {boundary}.")
        elif not report.open_axis_orthogonal:
            refuse("poisson", "A dipole layer needs the vacuum axis perpendicular to the "
                              "surface plane.")
    if poisson != "gpaw-default" and representation == "pw":
        refuse("poisson", "Plane-wave mode solves the Poisson equation in reciprocal "
                          "space for a bulk cell; only 'gpaw-default' applies.")
    if boundary == "slab" and poisson == "gpaw-default":
        warnings.append("No dipole correction: an asymmetric slab then carries an "
                        "artificial field across the vacuum.")

    field = np.asarray(spec.external_field_V_per_A, dtype=float)
    magnitude = float(np.linalg.norm(field))
    if magnitude > 0.0:
        if representation == "pw":
            refuse("external_field_V_per_A",
                   "A uniform field cannot be applied in plane-wave mode, which is "
                   "periodic in every direction.")
        ok, reason = bc.field_is_physical(field, cell, spec.pbc)
        if not ok:
            refuse("external_field_V_per_A", reason)
        if magnitude > MAX_FIELD_V_PER_A:
            refuse("external_field_V_per_A",
                   f"A {magnitude:g} V/A field exceeds {MAX_FIELD_V_PER_A:g} V/A. At such "
                   "fields electrons tunnel out of the system into the vacuum, so no "
                   "bound ground state exists; the box would hide that.")
        elif magnitude > ADVISED_FIELD_V_PER_A:
            warnings.append(f"A {magnitude:g} V/A field is strong; field emission into "
                            "the vacuum region may occur and the box would truncate it.")

    electrons_total = float(sum(spec.numbers)) - charge
    valence = None
    core = None
    if datasets is not None:
        valence = datasets.valence_electrons(symbols) - charge
        core = datasets.core_electrons(symbols)
        if valence <= 0:
            refuse("charge_e", f"A charge of {charge:+g} e removes every valence electron.")
    odd = abs(electrons_total - round(electrons_total)) < 1e-9 and \
        int(round(electrons_total)) % 2 == 1
    moments = np.asarray(spec.initial_magnetic_moments_muB, dtype=float)
    if len(moments) != n:
        refuse("initial_magnetic_moments_muB", f"{len(moments)} moments were given for "
                                               f"{n} atoms.")
        moments = np.zeros(n)
    if not spec.spin_polarized:
        if np.any(np.abs(moments) > 1e-12):
            refuse("spin_polarized",
                   "Initial magnetic moments are set but the calculation is spin-paired, "
                   "which would ignore them. Turn spin polarisation on, or clear the "
                   "moments.")
        if odd:
            if boundary == "cluster" or spec.occupations == "fixed":
                refuse("spin_polarized",
                       f"The system has {int(round(electrons_total))} electrons, an odd "
                       "number, so at least one is unpaired. A spin-paired calculation "
                       "would put half an electron in each spin channel of that state. "
                       "Turn spin polarisation on.")
            else:
                warnings.append(
                    "Odd electron count per cell treated spin-paired with smeared "
                    "occupations: valid for a non-magnetic metal, but no magnetic state "
                    "was explored.")
    else:
        if not np.any(np.abs(moments) > 1e-12):
            warnings.append("Spin-polarised with every initial moment zero: the "
                            "spin-symmetric start may stay non-magnetic unless the "
                            "occupations force a moment.")
        if datasets is not None:
            table = datasets.by_symbol()
            for index, (symbol, m) in enumerate(zip(symbols, moments)):
                if abs(m) > table[symbol].valence_electrons + 1e-9:
                    refuse("initial_magnetic_moments_muB",
                           f"Atom #{spec.atom_ids[index]} ({symbol}) is given "
                           f"{m:+g} mu_B but has only "
                           f"{table[symbol].valence_electrons:g} valence electrons.")
        if valence is not None and abs(float(moments.sum())) > valence + 1e-9:
            refuse("initial_magnetic_moments_muB",
                   f"The initial moments add up to {moments.sum():+g} mu_B, more than "
                   f"the {valence:g} valence electrons can carry.")

    occupations = spec.occupations
    width = spec.smearing_eV
    if occupations == "fixed":
        if width != 0.0:
            refuse("smearing_eV", f"Fixed occupations use no smearing, so "
                                  f"smearing_eV = {width:g} would be ignored. Set it to 0.")
        if boundary != "cluster":
            warnings.append("Integer occupations in a periodic system fail to converge "
                            "if it is a metal.")
    else:
        if width <= 0.0:
            refuse("smearing_eV", f"{occupations} occupations need a positive width, got "
                                  f"{width:g} eV. For integer occupations choose 'fixed'.")
        elif width > 1.0:
            refuse("smearing_eV", f"A {width:g} eV smearing is an electronic temperature "
                                  "above 10000 K; the result would not be a ground state.")
        elif width > 0.3:
            warnings.append(f"A {width:g} eV smearing is wide; the free energy and the "
                            "zero-width extrapolation will differ noticeably.")

    if spec.n_bands is not None:
        if spec.n_bands < 1:
            refuse("n_bands", f"n_bands must be at least 1, got {spec.n_bands}.")
        elif valence is not None:
            per_channel = valence if spec.spin_polarized else valence / 2.0
            if spec.spin_polarized:
                majority = (valence + abs(float(moments.sum()))) / 2.0
                per_channel = max(per_channel / 2.0, majority)
            needed = int(math.ceil(per_channel - 1e-9))
            if spec.n_bands < needed:
                refuse("n_bands", f"{spec.n_bands} bands cannot hold {valence:g} valence "
                                  f"electrons; at least {needed} are occupied.")
            elif occupations != "fixed" and spec.n_bands < needed + 1:
                warnings.append("Smeared occupations with no empty band: GPAW may fail to "
                                "converge the highest states.")

    for name, low, high in (("energy_tol_eV_per_electron", 0.0, 5e-3),
                            ("density_tol_electrons_per_electron", 0.0, 1e-2),
                            ("eigenstates_tol_eV2_per_electron", 0.0, 1e-4)):
        value = getattr(spec, name)
        if not (low < value <= high):
            refuse(name, f"{name} = {value:g} {FIELD_INDEX[name].unit} must be positive "
                         f"and at most {high:g}; looser than that is not a converged "
                         "ground state.")
    if spec.forces_tol_eV_A is not None and not (0.0 < spec.forces_tol_eV_A <= 0.5):
        refuse("forces_tol_eV_A", "forces_tol_eV_A must be positive and at most 0.5 eV/A, "
                                  "or empty.")
    if not (1 <= spec.max_scf_iterations <= 2000):
        refuse("max_scf_iterations", f"max_scf_iterations must be 1 to 2000, got "
                                     f"{spec.max_scf_iterations}.")
    if spec.random_seed is not None:
        refuse("random_seed", "A random seed does not apply: GPAW's ground state starts "
                              "from atomic densities and LCAO wavefunctions, which are "
                              "deterministic. Leave it empty.")

    observables = set(spec.observables)
    if "energy" not in observables:
        refuse("observables", "The total energy is always computed and must be listed.")
    if "stress" in observables and representation != "pw":
        refuse("observables", "GPAW computes the stress tensor only in plane-wave mode, "
                              "which needs a bulk cell. Remove 'stress' or use pw.")
    elif "stress" in observables and spec.cutoff_eV is not None and \
            spec.cutoff_eV < STRESS_CUTOFF_EV:
        warnings.append(
            f"Stress converges more slowly with the plane-wave cutoff than energy does (Pulay "
            f"stress). At {spec.cutoff_eV:g} eV it can be several GPa off: copper at its "
            f"PBE equilibrium reads -7.3 GPa at 400 eV and -0.02 GPa at 600 eV. Use at "
            f"least {STRESS_CUTOFF_EV:g} eV for stress, variable-cell relaxation and "
            "pressures.")
    if "spin_density" in observables and not spec.spin_polarized:
        refuse("observables", "The spin density is zero by construction in a spin-paired "
                              "calculation. Remove it or turn spin polarisation on.")
    if "magnetic_moment" in observables and not spec.spin_polarized:
        refuse("observables", "The magnetic moment is zero by construction in a "
                              "spin-paired calculation. Remove it or turn spin "
                              "polarisation on.")
    volumes = [o for o in ("density", "spin_density", "electrostatic_potential")
               if o in observables]
    if volumes and n and np.isfinite(cell).all():
        points = estimate_fine_grid(spec)
        size = points * 8
        if size > MAX_VOLUME_BYTES:
            refuse("observables",
                   f"The fine grid would hold about {points:,} points "
                   f"({size / 2**20:.0f} MB per quantity), above the "
                   f"{MAX_VOLUME_BYTES // 2**20} MB Materia stores. Remove "
                   f"{', '.join(volumes)} or coarsen the grid.")

    blocking = list(general)
    for texts in by_field.values():
        for text in texts:
            if text not in blocking:
                blocking.append(text)

    options = {
        "functionals": available_functionals(symbols, paths) if paths else [],
        "bases": available_bases(symbols, paths) if paths else [],
        "representations": ["pw", "fd", "lcao"] if boundary == "bulk" else ["fd", "lcao"],
        "poisson": {"cluster": ["moment-corrected", "gpaw-default"],
                    "slab": ["dipole-layer", "gpaw-default"]}.get(boundary, ["gpaw-default"]),
        "charged_periodic_policy": (["uniform-background", "none"]
                                    if boundary == "bulk" else ["none"]),
        "occupations": list(OCCUPATIONS),
        "observables": list(OBSERVABLES),
        "field_allowed": boundary != "bulk" and representation != "pw",
        "functional_notes": {k: FUNCTIONAL_NOTES.get(k, "") for k in FUNCTIONALS},
        "representation_notes": dict(REPRESENTATION_NOTES),
        "observable_notes": dict(OBSERVABLE_NOTES),
    }
    electrons = {
        "total": electrons_total, "odd": odd,
        "valence": valence, "core": core,
        "nuclear_charge": float(sum(spec.numbers)),
    }
    return SpecReport(blocking, warnings, by_field, report.as_dict(), electrons, options)


def require(spec: GroundStateSpec, environment=None) -> SpecReport:
    """Raise :class:`SpecRefused` unless the specification may run."""
    report = check(spec, environment)
    if not report.ok:
        raise SpecRefused(report)
    return report


def estimate_fine_grid(spec: GroundStateSpec) -> int:
    """Approximate number of points on GPAW's fine grid, for size checks."""
    cell = np.asarray(spec.cell_A, dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    if spec.representation == "pw" and spec.cutoff_eV:
        spacing = math.pi / math.sqrt(2.0 * spec.cutoff_eV / 27.211386) * 0.529177 / 2.0
    else:
        spacing = float(spec.grid_spacing_A or 0.2)
    coarse = [max(4, int(4 * math.ceil(length / spacing / 4.0))) for length in lengths]
    return int(np.prod([2 * c for c in coarse]))


def worker_structure(spec: GroundStateSpec) -> Dict[str, Any]:
    """The atoms as the worker builds them for GPAW."""
    masses = [pt.mass(int(z), int(a) if a else None)
              for z, a in zip(spec.numbers, spec.mass_numbers)]
    return {
        "numbers": list(spec.numbers),
        "positions": [list(p) for p in spec.positions_A],
        "cell": [list(r) for r in spec.cell_A],
        "pbc": list(spec.pbc),
        "magnetic_moments": list(spec.initial_magnetic_moments_muB)
        if spec.spin_polarized else [0.0] * spec.n_atoms,
        "masses": masses,
        "ids": list(spec.atom_ids),
    }


def gpaw_parameters(spec: GroundStateSpec) -> Dict[str, Any]:
    """The keyword arguments handed verbatim to ``gpaw.GPAW``.

    Plain JSON, because the worker runs in another interpreter.  Every
    settable variable that affects the calculation appears here, which is what
    the test that changes each variable in turn checks.
    """
    if spec.representation == "pw":
        mode: Dict[str, Any] = {"name": "pw", "ecut": float(spec.cutoff_eV)}
    else:
        mode = {"name": spec.representation}
    if spec.occupations == "fixed":
        occupations = {"name": "fermi-dirac", "width": 0.0}
    else:
        occupations = {"name": spec.occupations, "width": float(spec.smearing_eV)}
    convergence: Dict[str, Any] = {
        "energy": float(spec.energy_tol_eV_per_electron),
        "density": float(spec.density_tol_electrons_per_electron),
        "eigenstates": float(spec.eigenstates_tol_eV2_per_electron),
        "bands": "occupied",
    }
    if spec.forces_tol_eV_A is not None:
        convergence["forces"] = float(spec.forces_tol_eV_A)
    parameters: Dict[str, Any] = {
        "mode": mode,
        "xc": spec.xc,
        "kpts": {"size": list(spec.kpoints), "gamma": bool(spec.kpoints_gamma_centered)},
        "occupations": occupations,
        "charge": float(spec.charge_e),
        "spinpol": bool(spec.spin_polarized),
        "convergence": convergence,
        "maxiter": int(spec.max_scf_iterations),
    }
    if spec.representation != "pw":
        parameters["h"] = float(spec.grid_spacing_A)
    if spec.representation == "lcao":
        parameters["basis"] = spec.basis
    if spec.n_bands is not None:
        parameters["nbands"] = int(spec.n_bands)
    if spec.poisson == "moment-corrected":
        parameters["poissonsolver"] = {"name": "MomentCorrectionPoissonSolver",
                                       "poissonsolver": "fast",
                                       "moment_corrections": 4}
    elif spec.poisson == "dipole-layer":
        parameters["poissonsolver"] = {"dipolelayer": int(spec.vacuum_axis)}
    field = np.asarray(spec.external_field_V_per_A, dtype=float)
    magnitude = float(np.linalg.norm(field))
    if magnitude > 0.0:
        parameters["external"] = {"name": "ConstantElectricField",
                                  "strength": magnitude,
                                  "direction": (field / magnitude).tolist()}
    return parameters


def restart_key(spec: GroundStateSpec) -> str:
    """Fingerprint of what a stored wavefunction depends on.

    Two specifications with the same key describe the same atoms, the same
    discretisation, the same electron count and the same spin setup, so the
    converged wavefunctions of one are a safe starting point for the other.
    The functional, occupations and smearing, convergence tolerances and the
    external field may differ: GPAW accepts those on a restarted calculator
    and the SCF cycle re-converges to the new ground state from that start.
    GPAW refuses to change anything else on a restarted calculator, so the
    iteration limit and the electrostatic boundary are part of the key too.
    """
    return short_digest({
        "numbers": list(spec.numbers),
        "positions": np.round(np.asarray(spec.positions_A), 10).tolist(),
        "cell": np.round(np.asarray(spec.cell_A), 10).tolist(),
        "pbc": list(spec.pbc),
        "charge": spec.charge_e,
        "spin": spec.spin_polarized,
        "moments": list(spec.initial_magnetic_moments_muB),
        "representation": spec.representation,
        "grid": spec.grid_spacing_A, "cutoff": spec.cutoff_eV, "basis": spec.basis,
        "kpoints": list(spec.kpoints), "gamma": spec.kpoints_gamma_centered,
        "bands": spec.n_bands,
        "maxiter": spec.max_scf_iterations, "poisson": spec.poisson,
        "software": [kv for kv in spec.software if kv[0] in ("gpaw", "paw_dataset_package")],
        "datasets": [list(d) for d in spec.paw_datasets],
        "basis_files": [list(b) for b in spec.basis_files],
    })


RESTART_COMPATIBLE_VARIABLES = (
    "xc", "occupations", "smearing_eV", "energy_tol_eV_per_electron",
    "density_tol_electrons_per_electron", "eigenstates_tol_eV2_per_electron",
    "forces_tol_eV_A", "external_field_V_per_A", "observables",
)


def describe(spec: GroundStateSpec, report: Optional[SpecReport] = None) -> dict:
    """Everything the interface shows, grouped by section, with explanations."""
    sections = []
    for key, title in SECTIONS:
        items = []
        for info in FIELDS:
            if info.section != key:
                continue
            item = info.as_dict()
            item["value"] = _plain(getattr(spec, info.name))
            if report is not None:
                item["problems"] = list(report.by_field.get(info.name, []))
            items.append(item)
        sections.append({"key": key, "title": title, "fields": items})
    return {"schema": spec.schema, "version": spec.version, "digest": spec.digest,
            "short_digest": spec.short_digest, "sections": sections,
            "settings": spec.settings(), "gpaw_parameters": gpaw_parameters(spec)}
