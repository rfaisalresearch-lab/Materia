"""Harmonic vibrations by finite displacements: force constants and normal modes.

Method
------
Every free Cartesian coordinate ``u_j`` is displaced by ``+-h`` (or by
``+-h`` and ``+-2h``) and the forces are recomputed.  The force constants are
the central-difference derivatives

``Phi_ij = d^2 E / du_i du_j = -dF_i / du_j``

in eV/A^2, from the two-point stencil ``-(F(+h) - F(-h)) / 2h`` or the
four-point stencil ``-(-F(+2h) + 8 F(+h) - 8 F(-h) + F(-2h)) / 12h``.  The
exact matrix is symmetric; the measured asymmetry is reported and the
symmetric part is kept.

Sum rules
---------
A model whose energy does not change when the whole system is translated has
``sum_j Phi_ij = 0`` along each Cartesian direction (the acoustic sum rule).
Finite differences break it slightly.  It is restored by the orthogonal
projection ``Phi' = P Phi P`` with ``P = 1 - V V^T`` and ``V`` an orthonormal
basis of rigid translations, which keeps ``Phi'`` symmetric, satisfies the
rule exactly and is the smallest change in the Frobenius norm that does.  For
an isolated system at a stationary point the energy is also invariant under
rigid rotation and ``V`` can include the rotations.  The projection is refused
when the measured violation is too large to be numerical error, because then
the model itself is not translation invariant (atoms tied to fixed points, or
fixed atoms); the caller must then choose ``sum_rule="none"``.

Normal modes
------------
The mass-weighted matrix ``D = M^-1/2 Phi M^-1/2`` has eigenvalues
``omega^2`` in eV / (A^2 u) and orthonormal eigenvectors ``e``.  Frequencies
are reported in THz, cm^-1 and meV.  A negative eigenvalue is an imaginary
frequency and is reported as a negative number, the usual convention; the
structure is then not at a minimum along that mode.  Cartesian displacement
patterns are ``M^-1/2 e`` normalised to unit length, and the reduced mass of a
mode is ``1 / sum_i (e_i^2 / m_i)``, the convention of Wilson, Decius and Cross
used by most quantum chemistry codes: with ``l`` the unit Cartesian
displacement, ``omega^2 = (l . Phi l) / mu``.  For a homonuclear diatomic it is
the atomic mass, not the two-body reduced mass.

The eigenvalue resolution is the largest of three measured quantities: the
spectral norm of the change the symmetrisation and sum rule made to ``D``
(by Weyl's inequality no eigenvalue of the symmetric part moved by more), the
spectral norm of the antisymmetric part of the raw matrix, which measures the
finite-difference error directly, and the truncation error of the stencil,
``(h / 1 A)^p`` of the largest eigenvalue for a ``p``-point stencil, which is
the relative error of a central difference for interactions that vary over
about one angstrom and which neither of the others detects.  Eigenvalues smaller than
the resolution are not distinguished from zero.  A mode that lies in the
space of rigid translations (and, for an isolated system, rigid rotations)
to better than 0.999 and whose eigenvalue is within the resolution is
labelled rigid-body, not a vibration, whether or not a sum rule removed it.

Periodic cells
--------------
For a periodic cell the result is the set of Gamma-point modes of that cell.
When the cell is a supercell of a primitive cell these are the phonons at
every wavevector commensurate with the supercell, folded to Gamma.  Materia
does not identify primitive cells or unfold modes, and it computes no
dispersion: any request for a wavevector other than Gamma is refused.  Force
constants between an atom and its own periodic images are summed, which is
correct for the Gamma point of the periodic crystal.  The zero-point energy of
a periodic cell is a Gamma-point sample, not a Brillouin-zone integral.  No
non-analytic term is added, so in a polar crystal the Gamma modes are the
transverse-optical limit and LO-TO splitting is absent.

References
----------
M. Born and K. Huang, Dynamical Theory of Crystal Lattices (Oxford, 1954).
E. B. Wilson, J. C. Decius and P. C. Cross, Molecular Vibrations (McGraw-Hill, 1955).
K. Parlinski, Z. Q. Li and Y. Kawazoe, Phys. Rev. Lett. 78 (1997) 4063.
H. Weyl, Math. Ann. 71 (1912) 441.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..core_model.units import (
    ATOMIC_MASS_UNIT_KG,
    BOLTZMANN_EV_K,
    ELEMENTARY_CHARGE_C,
    PLANCK_J_S,
)
from ..provenance import Convergence, Fidelity, Origin, Provenance, Result, digest

SPEED_OF_LIGHT_CM_S = 2.99792458e10
EIGENVALUE_TO_RAD_S2 = ELEMENTARY_CHARGE_C / (1e-20 * ATOMIC_MASS_UNIT_KG)
THZ_TO_CM1 = 1e12 / SPEED_OF_LIGHT_CM_S
THZ_TO_MEV = PLANCK_J_S * 1e12 / ELEMENTARY_CHARGE_C * 1e3
SUM_RULES = ("translational", "translational+rotational", "none")
STENCILS = (2, 4)
MAX_FREE_ATOMS = 1500
RIGID_OVERLAP = 0.999
TRUNCATION_LENGTH_A = 1.0

REFERENCES = [
    "M. Born and K. Huang, Dynamical Theory of Crystal Lattices (Oxford, 1954)",
    "E. B. Wilson, J. C. Decius and P. C. Cross, Molecular Vibrations (McGraw-Hill, 1955)",
    "K. Parlinski, Z. Q. Li and Y. Kawazoe, Phys. Rev. Lett. 78 (1997) 4063",
]

ForceCallback = Callable[[np.ndarray], np.ndarray]
Progress = Callable[[float, str], Optional[bool]]


class PhononError(ValueError):
    """The harmonic analysis could not be carried out."""


class PhononRefused(PhononError):
    """The request is outside what the method can answer honestly."""


class PhononCancelled(PhononError):
    """The caller cancelled the analysis."""


@dataclass(frozen=True)
class PhononSettings:
    """Every choice that changes the numbers.

    ``displacement_A`` is the finite-difference step.  ``max_residual_force_eV_A``
    is the largest force allowed on a free atom of the reference structure;
    above it the structure is not a stationary point and the analysis is
    refused unless ``allow_nonstationary`` is set, in which case every result
    is labelled estimated.  ``asr_tolerance`` is the largest sum-rule
    violation, relative to the largest diagonal force constant, that is
    treated as numerical error.
    """

    displacement_A: float = 0.01
    stencil: int = 2
    sum_rule: str = "translational"
    max_residual_force_eV_A: float = 5e-3
    allow_nonstationary: bool = False
    asr_tolerance: float = 0.02
    q_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> "PhononSettings":
        if not (1e-4 <= float(self.displacement_A) <= 0.1):
            raise PhononError("displacement_A must lie between 1e-4 and 0.1 A.")
        if self.stencil not in STENCILS:
            raise PhononError(f"stencil must be one of {STENCILS}.")
        if self.sum_rule not in SUM_RULES:
            raise PhononError(f"sum_rule must be one of {SUM_RULES}.")
        if not float(self.max_residual_force_eV_A) > 0:
            raise PhononError("max_residual_force_eV_A must be positive.")
        if not (0 < float(self.asr_tolerance) < 1):
            raise PhononError("asr_tolerance must lie between 0 and 1.")
        q = np.asarray(self.q_point, dtype=float)
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            raise PhononError("q_point must be three finite numbers.")
        if np.any(np.abs(q - np.round(q)) > 1e-12):
            raise PhononRefused(
                f"Only the Gamma point is computed; q = {q.tolist()} was requested. "
                "Phonon dispersion needs force constants resolved by lattice vector and "
                "Fourier interpolation, which Materia does not implement. A supercell "
                "commensurate with the wavevector gives its modes folded to Gamma.")
        return self

    def as_dict(self) -> dict:
        d = asdict(self)
        d["q_point"] = [float(v) for v in self.q_point]
        return d


@dataclass
class ForceModel:
    """A force callback and what it is."""

    callback: ForceCallback
    name: str
    fidelity: Fidelity
    description: Dict[str, Any] = field(default_factory=dict)
    cutoff_A: Optional[float] = None


def force_model(source: Any, structure: Structure, name: str = "",
                fidelity: Optional[Fidelity] = None) -> ForceModel:
    """Wrap a force source as a callback from positions (N, 3) in A to forces in eV/A.

    ``source`` may be a Materia :class:`~materia.physics.potentials.Potential`,
    an object with an ASE-style ``get_forces(atoms)`` method, or a plain
    callable taking positions and returning forces.  A plain callable needs a
    ``name`` and a ``fidelity`` from the caller.
    """
    if hasattr(source, "energy_and_forces") and hasattr(source, "describe"):
        work = structure.copy()

        def callback(positions: np.ndarray) -> np.ndarray:
            work.positions = positions
            return np.asarray(source.energy_and_forces(work)[1], dtype=float)

        described = source.describe()
        return ForceModel(callback, name or source.model_label(),
                          fidelity or source.fidelity, described,
                          float(getattr(source, "cutoff_A", 0.0)) or None)
    if hasattr(source, "get_forces"):
        from ase import Atoms

        atoms = Atoms(numbers=structure.numbers, positions=structure.positions,
                      cell=structure.cell.matrix, pbc=list(structure.cell.pbc),
                      masses=structure.masses())
        atoms.calc = source

        def callback(positions: np.ndarray) -> np.ndarray:
            atoms.positions = positions
            return np.asarray(atoms.get_forces(), dtype=float)

        label = name or f"ase/{type(source).__name__}"
        return ForceModel(callback, label, fidelity or Fidelity.NON_PHYSICAL,
                          {"calculator": type(source).__name__,
                           "parameters": dict(getattr(source, "parameters", {}) or {})})
    if callable(source):
        if not name or fidelity is None:
            raise PhononError("A plain force callback needs a model name and a fidelity.")
        return ForceModel(source, name, fidelity, {"callback": getattr(source, "__name__", "")})
    raise PhononError("The force source must be a Materia potential, an ASE calculator "
                      "or a callable returning forces.")


def rigid_body_basis(positions: np.ndarray, free: np.ndarray, rotations: bool) -> np.ndarray:
    """Orthonormal columns spanning rigid translations (and rotations) of the free atoms."""
    x = positions[free]
    n = len(x)
    vectors = []
    for axis in range(3):
        t = np.zeros((n, 3))
        t[:, axis] = 1.0
        vectors.append(t.ravel())
    if rotations:
        centred = x - x.mean(axis=0)
        for axis in range(3):
            unit = np.zeros(3)
            unit[axis] = 1.0
            vectors.append(np.cross(unit, centred).ravel())
    matrix = np.array(vectors).T
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int((s > 1e-8 * s.max()).sum())
    return u[:, :rank]


def _report(progress: Optional[Progress], cancelled, fraction: float, message: str) -> None:
    if cancelled is not None and cancelled():
        raise PhononCancelled("Phonon analysis cancelled.")
    if progress is not None and progress(fraction, message) is False:
        raise PhononCancelled("Phonon analysis cancelled.")


def force_constants(model: ForceModel, positions: np.ndarray, free: np.ndarray,
                    settings: PhononSettings, progress: Optional[Progress] = None,
                    cancelled=None) -> Tuple[np.ndarray, int]:
    """The raw (unsymmetrised) force-constant matrix over free coordinates, and the call count."""
    h = float(settings.displacement_A)
    atoms = np.flatnonzero(free)
    columns = [(a, alpha) for a in atoms for alpha in range(3)]
    rows = np.repeat(atoms, 3) * 3 + np.tile(np.arange(3), len(atoms))
    steps = (1, -1) if settings.stencil == 2 else (2, 1, -1, -2)
    weights = ({1: 1.0, -1: -1.0} if settings.stencil == 2
               else {2: -1.0, 1: 8.0, -1: -8.0, -2: 1.0})
    denominator = 2.0 * h if settings.stencil == 2 else 12.0 * h
    phi = np.zeros((len(columns), len(columns)))
    calls = 0
    for column, (atom, alpha) in enumerate(columns):
        derivative = np.zeros(len(rows))
        for step in steps:
            displaced = positions.copy()
            displaced[atom, alpha] += step * h
            forces = np.asarray(model.callback(displaced), dtype=float)
            calls += 1
            if forces.shape != positions.shape or not np.all(np.isfinite(forces)):
                raise PhononError(f"The force model returned an invalid force array when "
                                  f"atom index {atom} was displaced along {'xyz'[alpha]}.")
            derivative += weights[step] * forces.ravel()[rows]
        phi[:, column] = -derivative / denominator
        _report(progress, cancelled, 0.05 + 0.85 * (column + 1) / len(columns),
                f"displacement {column + 1} of {len(columns)}")
    return phi, calls


@dataclass
class PhononResult:
    """Everything one harmonic analysis produced, in stated units."""

    frequencies_THz: np.ndarray
    eigenvalues_eV_A2_u: np.ndarray
    eigenvectors: np.ndarray
    displacements: np.ndarray
    reduced_masses_u: np.ndarray
    participation: np.ndarray
    kinds: List[str]
    force_constants_eV_A2: np.ndarray
    free_atom_ids: List[int]
    masses_u: np.ndarray
    diagnostics: Dict[str, Any]
    zero_point_energy_eV: Optional[float]
    zero_point_note: str
    provenance: Provenance
    convergence: Convergence

    @property
    def frequencies_cm1(self) -> np.ndarray:
        return self.frequencies_THz * THZ_TO_CM1

    @property
    def energies_meV(self) -> np.ndarray:
        return self.frequencies_THz * THZ_TO_MEV

    def imaginary_modes(self) -> List[int]:
        return [k for k, kind in enumerate(self.kinds) if kind == "imaginary"]

    def vibrational_modes(self) -> List[int]:
        return [k for k, kind in enumerate(self.kinds) if kind in ("real", "imaginary")]

    def mode_table(self) -> List[dict]:
        return [{"index": k, "kind": self.kinds[k],
                 "frequency_THz": float(self.frequencies_THz[k]),
                 "frequency_cm1": float(self.frequencies_cm1[k]),
                 "energy_meV": float(self.energies_meV[k]),
                 "reduced_mass_u": float(self.reduced_masses_u[k]),
                 "participation_ratio": float(self.participation[k])}
                for k in range(len(self.kinds))]

    def thermodynamics(self, temperatures_K: Sequence[float]) -> Dict[str, np.ndarray]:
        """Harmonic vibrational thermodynamics at the requested temperatures.

        Values are for the modeled cell or isolated structure. Rigid motions
        and numerically unresolved zero modes are excluded. An imaginary mode
        means the reference is not a minimum, so equilibrium harmonic
        thermodynamics is refused.
        """
        temperatures = np.asarray(temperatures_K, dtype=float)
        if temperatures.ndim == 0:
            temperatures = temperatures.reshape(1)
        if temperatures.ndim != 1 or temperatures.size == 0:
            raise PhononError("temperatures_K must be a non-empty one-dimensional sequence.")
        if not np.all(np.isfinite(temperatures)) or np.any(temperatures < 0):
            raise PhononError("temperatures_K must contain finite non-negative values.")
        imaginary = self.imaginary_modes()
        if imaginary:
            raise PhononRefused(
                f"The reference has {len(imaginary)} imaginary mode(s), so it is not a "
                "local minimum and equilibrium harmonic thermodynamics is undefined.")
        if not self.convergence.converged or self.zero_point_energy_eV is None:
            raise PhononRefused(
                "Harmonic thermodynamics requires a converged stationary reference.")
        mode_indices = [k for k, kind in enumerate(self.kinds) if kind == "real"]
        energies = np.asarray(
            [self.frequencies_THz[k] * THZ_TO_MEV / 1000.0 for k in mode_indices],
            dtype=float)
        zpe = 0.5 * float(energies.sum())
        free = np.full(temperatures.shape, zpe, dtype=float)
        internal = np.full(temperatures.shape, zpe, dtype=float)
        entropy = np.zeros(temperatures.shape, dtype=float)
        heat_capacity = np.zeros(temperatures.shape, dtype=float)
        positive = temperatures > 0
        for index in np.flatnonzero(positive):
            temperature = float(temperatures[index])
            x = energies / (BOLTZMANN_EV_K * temperature)
            occupation = np.zeros_like(x)
            finite = x < 700.0
            occupation[finite] = 1.0 / np.expm1(x[finite])
            thermal_free = BOLTZMANN_EV_K * temperature * np.log(
                -np.expm1(-x)).sum()
            thermal_internal = float(np.sum(energies * occupation))
            free[index] = zpe + thermal_free
            internal[index] = zpe + thermal_internal
            entropy[index] = (internal[index] - free[index]) / temperature
            heat_capacity[index] = BOLTZMANN_EV_K * float(
                np.sum(x[finite] ** 2 * occupation[finite] * (occupation[finite] + 1.0)))
        return {
            "temperatures_K": temperatures,
            "helmholtz_free_energy_eV": free,
            "internal_energy_eV": internal,
            "entropy_eV_K": entropy,
            "heat_capacity_eV_K": heat_capacity,
        }

    def thermodynamics_results(self, temperatures_K: Sequence[float]) -> Dict[str, Result]:
        """Harmonic thermodynamic arrays as provenance-bearing result records."""
        names = {
            "helmholtz_free_energy_eV": ("vibrational_helmholtz_free_energy", "eV"),
            "internal_energy_eV": ("vibrational_internal_energy", "eV"),
            "entropy_eV_K": ("vibrational_entropy", "eV/K"),
            "heat_capacity_eV_K": ("vibrational_heat_capacity", "eV/K"),
        }
        try:
            values = self.thermodynamics(temperatures_K)
        except PhononRefused as exc:
            return {
                key: Result(
                    name, None, unit,
                    Provenance(self.provenance.model, self.provenance.fidelity,
                               Origin.UNSUPPORTED, notes=str(exc)),
                    unsupported_reason=str(exc))
                for key, (name, unit) in names.items()
            }
        temperatures = values["temperatures_K"]
        extra = {
            "temperatures_K": temperatures.tolist(),
            "mode_count": len(self.vibrational_modes()),
            "scope": ("Gamma-point sample of the modeled periodic cell"
                      if self.diagnostics.get("geometry") != "isolated"
                      else "isolated structure"),
        }
        return {
            key: Result(name, values[key], unit, self.provenance,
                        convergence=self.convergence, extra=dict(extra))
            for key, (name, unit) in names.items()
        }

    def results(self) -> Dict[str, Result]:
        prov, conv = self.provenance, self.convergence
        common = {"free_atom_ids": list(self.free_atom_ids), "kinds": list(self.kinds),
                  "diagnostics": dict(self.diagnostics)}
        uncertainty = self.diagnostics["frequency_resolution_THz"]
        out = {
            "frequencies": Result("phonon_frequencies", self.frequencies_THz.copy(), "THz",
                                  prov, uncertainty=uncertainty, uncertainty_kind="bound",
                                  convergence=conv,
                                  extra={**common, "modes": self.mode_table(),
                                         "negative_means": "imaginary frequency"}),
            "force_constants": Result("force_constants", self.force_constants_eV_A2.copy(),
                                      "eV/A^2", prov, convergence=conv,
                                      extra={**common, "ordering": "atom-major, x y z"}),
            "eigenvectors": Result("mass_weighted_eigenvectors", self.eigenvectors.copy(), "",
                                   prov, convergence=conv,
                                   extra={**common, "layout": "columns are modes"}),
            "displacements": Result("mode_displacements", self.displacements.copy(), "", prov,
                                    convergence=conv,
                                    extra={**common, "layout": "(mode, atom, xyz), unit norm"}),
        }
        if self.zero_point_energy_eV is None:
            out["zero_point_energy"] = Result(
                "zero_point_energy", None, "eV",
                Provenance(prov.model, prov.fidelity, Origin.UNSUPPORTED,
                           notes=self.zero_point_note),
                unsupported_reason=self.zero_point_note)
        else:
            out["zero_point_energy"] = Result(
                "zero_point_energy", float(self.zero_point_energy_eV), "eV", prov,
                convergence=conv, extra={**common, "note": self.zero_point_note})
        return out

    def stored_arrays(self) -> dict:
        """The arrays in Materia's checksummed project storage form."""
        from ..project_format.arrays import StoredArray

        meta = {"free_atom_ids": list(self.free_atom_ids), "model": self.provenance.model}
        return {
            "force_constants": StoredArray(self.force_constants_eV_A2, "eV/A^2",
                                           "harmonic force constants", "table", meta),
            "frequencies": StoredArray(self.frequencies_THz, "THz",
                                       "Gamma-point frequencies, negative = imaginary",
                                       "table", {**meta, "kinds": list(self.kinds)}),
            "eigenvectors": StoredArray(self.eigenvectors, "",
                                        "mass-weighted eigenvectors, columns are modes",
                                        "table", meta),
        }

    def as_dict(self) -> dict:
        return {"modes": self.mode_table(), "free_atom_ids": list(self.free_atom_ids),
                "diagnostics": dict(self.diagnostics),
                "zero_point_energy_eV": self.zero_point_energy_eV,
                "zero_point_note": self.zero_point_note,
                "provenance": self.provenance.as_dict(),
                "convergence": self.convergence.as_dict()}


def _geometry(structure: Structure) -> str:
    periodic = int(sum(bool(p) for p in structure.cell.pbc))
    return {0: "isolated", 1: "wire", 2: "slab", 3: "bulk"}[periodic]


def harmonic_analysis(structure: Structure, source: Any,
                      settings: Optional[PhononSettings] = None, *,
                      name: str = "", fidelity: Optional[Fidelity] = None,
                      progress: Optional[Progress] = None,
                      cancelled: Optional[Callable[[], bool]] = None) -> PhononResult:
    """Gamma-point harmonic normal modes of ``structure`` under ``source``.

    Nothing is returned unless the analysis completed: refusals raise
    :class:`PhononRefused`, cancellation raises :class:`PhononCancelled` and
    invalid inputs or model output raise :class:`PhononError`.
    """
    t0 = time.perf_counter()
    settings = (settings or PhononSettings()).validate()
    n = len(structure)
    if n == 0:
        raise PhononError("The structure has no atoms.")
    free = ~np.asarray(structure.fixed, dtype=bool)
    n_free = int(free.sum())
    if n_free == 0:
        raise PhononRefused("Every atom is fixed, so there is nothing to vibrate.")
    if n_free > MAX_FREE_ATOMS:
        raise PhononRefused(f"{n_free} free atoms would need a dense {3 * n_free} x "
                            f"{3 * n_free} matrix; the limit is {MAX_FREE_ATOMS} atoms.")
    geometry = _geometry(structure)
    if settings.sum_rule != "none" and not free.all():
        raise PhononRefused(
            f"{n - n_free} atoms are fixed. Holding atoms fixed ties the free atoms to "
            "fixed points, so the translational sum rule does not hold for this partial "
            "force-constant matrix. Use sum_rule='none'.")
    if settings.sum_rule == "translational+rotational" and geometry != "isolated":
        raise PhononRefused(
            f"The rotational sum rule holds only for an isolated system; this structure "
            f"is {geometry}. Use sum_rule='translational'.")
    model = force_model(source, structure, name, fidelity)
    positions = np.asarray(structure.positions, dtype=float).copy()
    masses = structure.masses()[free]
    _report(progress, cancelled, 0.0, "reference forces")
    reference = np.asarray(model.callback(positions), dtype=float)
    if reference.shape != positions.shape or not np.all(np.isfinite(reference)):
        raise PhononError("The force model returned an invalid force array.")
    residual = float(np.linalg.norm(reference[free], axis=1).max())
    stationary = residual <= settings.max_residual_force_eV_A
    if not stationary and not settings.allow_nonstationary:
        raise PhononRefused(
            f"The largest force on a free atom is {residual:.3g} eV/A, above "
            f"{settings.max_residual_force_eV_A:g} eV/A. Harmonic normal modes are defined "
            "at a stationary point: relax the structure first, or set "
            "allow_nonstationary to obtain an estimated curvature analysis.")
    if settings.sum_rule == "translational+rotational" and not stationary:
        raise PhononRefused("The rotational sum rule holds only at a stationary point.")
    raw, calls = force_constants(model, positions, free, settings, progress, cancelled)
    calls += 1
    _report(progress, cancelled, 0.92, "sum rules and diagonalisation")
    scale = float(np.abs(np.diag(raw)).max()) or 1.0
    asymmetry = float(np.abs(raw - raw.T).max())
    phi = 0.5 * (raw + raw.T)
    basis = rigid_body_basis(positions, free, rotations=False)
    violation = float(np.abs(phi @ basis).max()) * math.sqrt(n_free)
    rigid = np.zeros((3 * n_free, 0))
    if settings.sum_rule != "none":
        if violation > settings.asr_tolerance * scale:
            raise PhononRefused(
                f"The acoustic sum rule is violated by {violation:.3g} eV/A^2, "
                f"{violation / scale:.2%} of the largest diagonal force constant, above the "
                f"{settings.asr_tolerance:.0%} expected of finite-difference error. The "
                "force model is not invariant under translation, for example because "
                "atoms are tied to fixed points. Use sum_rule='none'.")
        rigid = rigid_body_basis(positions, free,
                                 rotations=settings.sum_rule == "translational+rotational")
        projector = np.eye(3 * n_free) - rigid @ rigid.T
        phi = projector @ phi @ projector
        phi = 0.5 * (phi + phi.T)
    inv_sqrt = 1.0 / np.sqrt(np.repeat(masses, 3))
    dynamical = phi * inv_sqrt[:, None] * inv_sqrt[None, :]
    raw_dynamical = raw * inv_sqrt[:, None] * inv_sqrt[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(dynamical)
    corrected = float(np.linalg.norm(raw_dynamical - dynamical, 2))
    antisymmetric = float(np.linalg.norm(0.5 * (raw_dynamical - raw_dynamical.T), 2))
    floor = float(np.abs(eigenvalues).max()) * (
        settings.displacement_A / TRUNCATION_LENGTH_A) ** settings.stencil
    resolution = max(corrected, antisymmetric, floor)
    omega = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues) * EIGENVALUE_TO_RAD_S2)
    frequencies = omega / (2.0 * math.pi * 1e12)
    resolution_THz = math.sqrt(resolution * EIGENVALUE_TO_RAD_S2) / (2.0 * math.pi * 1e12)
    motions = rigid_body_basis(positions, free, rotations=geometry == "isolated")
    weighted_rigid, _ = np.linalg.qr(motions / inv_sqrt[:, None])
    kinds = []
    for k in range(len(eigenvalues)):
        overlap = float(np.sum((weighted_rigid.T @ eigenvectors[:, k]) ** 2))
        if overlap > RIGID_OVERLAP and abs(eigenvalues[k]) <= resolution:
            kinds.append("rigid-body")
        elif abs(eigenvalues[k]) <= resolution:
            kinds.append("zero-within-resolution")
        elif eigenvalues[k] < 0:
            kinds.append("imaginary")
        else:
            kinds.append("real")
    cartesian = eigenvectors * inv_sqrt[:, None]
    displacements = (cartesian / np.linalg.norm(cartesian, axis=0)).T.reshape(-1, n_free, 3)
    reduced = 1.0 / np.sum(eigenvectors ** 2 * (inv_sqrt ** 2)[:, None], axis=0)
    per_atom = np.sum(eigenvectors.reshape(n_free, 3, -1) ** 2, axis=1)
    participation = 1.0 / (n_free * np.sum(per_atom ** 2, axis=0))
    imaginary = [k for k, kind in enumerate(kinds) if kind == "imaginary"]
    undetermined = [k for k, kind in enumerate(kinds) if kind == "zero-within-resolution"]
    if imaginary:
        zpe, zpe_note = None, (
            f"{len(imaginary)} imaginary modes, the largest {abs(frequencies[imaginary[0]]):.4g}i "
            "THz: the structure is not at a minimum, so no zero-point energy is defined.")
    elif not stationary:
        zpe, zpe_note = None, "The structure is not a stationary point."
    else:
        real = [k for k, kind in enumerate(kinds) if kind == "real"]
        zpe = 0.5 * sum(float(omega[k]) for k in real) * PLANCK_J_S / (2 * math.pi) \
            / ELEMENTARY_CHARGE_C
        zpe_note = ("Half the sum of hbar omega over the real vibrational modes; rigid-body "
                    "modes and modes indistinguishable from zero are excluded.")
        if geometry != "isolated":
            zpe_note += (" For a periodic cell this samples only the Gamma point of the "
                         "cell and is not a converged Brillouin-zone integral.")
    free_ids = [int(i) for i in np.asarray(structure.ids)[free]]
    cutoff = model.cutoff_A
    diagnostics = {
        "geometry": geometry,
        "force_calls": calls,
        "residual_force_eV_A": residual,
        "stationary": stationary,
        "asymmetry_eV_A2": asymmetry,
        "asymmetry_relative": asymmetry / scale,
        "sum_rule_violation_eV_A2": violation,
        "sum_rule_violation_relative": violation / scale,
        "sum_rule_applied": settings.sum_rule,
        "rigid_body_modes_removed": int(rigid.shape[1]),
        "eigenvalue_resolution_eV_A2_u": resolution,
        "frequency_resolution_THz": resolution_THz,
        "imaginary_modes": imaginary,
        "zero_within_resolution_modes": undetermined,
        "wall_time_s": time.perf_counter() - t0,
    }
    if geometry != "isolated" and cutoff:
        widths = _periodic_widths(structure)
        if widths and min(widths) < 2.0 * cutoff:
            diagnostics["self_image_note"] = (
                f"The narrowest periodic width, {min(widths):.3f} A, is less than twice the "
                f"{cutoff:g} A cutoff, so atoms interact with their own images. The force "
                "constants are the image sums that the Gamma point of this periodic crystal "
                "needs, not those of an isolated defect.")
    approximations = [
        "Harmonic approximation: energy expanded to second order about the reference "
        "geometry; no anharmonic shifts, lifetimes or thermal expansion.",
        f"Central finite differences, {settings.stencil}-point stencil, step "
        f"{settings.displacement_A:g} A.",
        "Force constants symmetrised" + (
            f", {settings.sum_rule} sum rule imposed by orthogonal projection."
            if settings.sum_rule != "none" else "; no sum rule imposed."),
        "Classical nuclear masses from the structure's isotopes.",
    ]
    if geometry != "isolated":
        approximations.append(
            "Gamma point of the given periodic cell only: no dispersion, no interpolation "
            "and no unfolding to a primitive cell.")
        approximations.append(
            "No non-analytic correction: in a polar crystal the Gamma modes are those of "
            "the analytic part, the transverse-optical limit. Longitudinal-optical "
            "frequencies and LO-TO splitting need Born effective charges and the "
            "dielectric tensor, which are not computed.")
    if not free.all():
        approximations.append(
            f"{n - n_free} fixed atoms are treated as infinitely heavy (partial Hessian).")
    if not stationary:
        approximations.append(
            f"Reference geometry is not stationary (largest force {residual:.3g} eV/A); the "
            "modes describe the local curvature, not vibrations about a minimum.")
    boundary = {"isolated": "isolated system, open boundaries",
                "wire": "one periodic direction, Gamma point",
                "slab": "two periodic directions, Gamma point",
                "bulk": "three periodic directions, Gamma point"}[geometry]
    inputs = digest({"numbers": structure.numbers.tolist(),
                     "positions": np.round(positions, 10).tolist(),
                     "cell": structure.cell.matrix.tolist(),
                     "pbc": [bool(p) for p in structure.cell.pbc],
                     "masses": structure.masses().tolist(), "free": free.tolist(),
                     "settings": settings.as_dict(), "model": model.name,
                     "model_parameters": model.description.get("parameters", {})})
    provenance = Provenance(
        model=f"phonons/finite-displacement[{model.name}]",
        fidelity=model.fidelity,
        origin=Origin.CALCULATED if stationary else Origin.ESTIMATED,
        approximations=approximations,
        tolerances={"displacement_A": settings.displacement_A,
                    "max_residual_force_eV_A": settings.max_residual_force_eV_A,
                    "asr_tolerance": settings.asr_tolerance},
        boundary_conditions=boundary,
        parameters={"settings": settings.as_dict(), "force_model": model.name,
                    "force_model_parameters": model.description.get("parameters", {}),
                    "masses_u": masses.tolist(), "free_atom_ids": free_ids},
        references=list(REFERENCES) + list(model.description.get("references", [])),
        inputs_digest=inputs,
    )
    converged = stationary and diagnostics["asymmetry_relative"] < settings.asr_tolerance
    convergence = Convergence(
        converged=converged, iterations=calls, residual=residual,
        residual_metric="largest force on a free atom of the reference geometry",
        tolerance=settings.max_residual_force_eV_A,
        message=("Stationary reference; force constants symmetric to "
                 f"{diagnostics['asymmetry_relative']:.2e} of the largest diagonal term."
                 if converged else
                 "Reference not stationary or force constants strongly asymmetric; "
                 "treat the modes as estimates."))
    _report(progress, cancelled, 1.0, "done")
    return PhononResult(
        frequencies_THz=frequencies, eigenvalues_eV_A2_u=eigenvalues,
        eigenvectors=eigenvectors, displacements=displacements, reduced_masses_u=reduced,
        participation=participation, kinds=kinds, force_constants_eV_A2=phi,
        free_atom_ids=free_ids, masses_u=masses, diagnostics=diagnostics,
        zero_point_energy_eV=zpe, zero_point_note=zpe_note, provenance=provenance,
        convergence=convergence)


def _periodic_widths(structure: Structure) -> List[float]:
    matrix = structure.cell.matrix
    volume = abs(float(np.linalg.det(matrix)))
    widths = []
    for k in range(3):
        if structure.cell.pbc[k]:
            normal = np.cross(matrix[(k + 1) % 3], matrix[(k + 2) % 3])
            widths.append(volume / float(np.linalg.norm(normal)))
    return widths
