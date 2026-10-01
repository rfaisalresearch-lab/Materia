"""Periodic phonon dispersion and density of states from real-space force constants.

The primitive periodic structure is displaced in an explicitly chosen supercell.
ASE's finite-displacement machinery evaluates and symmetrises the real-space force
constants, applies the requested acoustic sum rule, and Fourier-interpolates them to
the supplied reciprocal-space path and a Monkhorst-Pack mesh.  Materia potentials are
exposed to ASE through a calculator that reconstructs a Materia Structure for every
force evaluation without changing the input structure.

Frequencies are signed: a negative value is the magnitude of an imaginary mode with
a minus sign.  The density of states uses the same convention and can therefore have
weight below zero for an unstable structure.  No non-analytic dipole correction is
implemented, so polar materials have no LO-TO splitting.  Branches are sorted by
frequency independently at each q point and are not connected by eigenvector overlap.
"""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..provenance import Convergence, Fidelity, Origin, Provenance, Result, digest
from .phonons import EIGENVALUE_TO_RAD_S2, PhononCancelled, PhononError, PhononRefused

MAX_PRIMITIVE_ATOMS = 256
MAX_SUPERCELL_ATOMS = 8192
MAX_Q_PATH_POINTS = 20000
MAX_DOS_Q_POINTS = 65536
MAX_SPECTRAL_VALUES = 5_000_000
MAX_DOS_GRID_POINTS = 20000
TRUNCATION_LENGTH_A = 1.0
RIGID_OVERLAP = 0.999

REFERENCES = [
    "K. Parlinski, Z. Q. Li and Y. Kawazoe, Phys. Rev. Lett. 78 (1997) 4063",
    "A. H. Larsen et al., J. Phys.: Condens. Matter 29 (2017) 273002",
]

Progress = Callable[[float, str], Optional[bool]]


@dataclass(frozen=True)
class PeriodicPhononSettings:
    """Immutable choices defining a periodic dispersion and DOS calculation."""

    supercell: Tuple[int, int, int]
    q_path: Tuple[Tuple[float, float, float], ...]
    dos_mesh: Tuple[int, int, int]
    q_labels: Tuple[str, ...] = ()
    displacement_A: float = 0.01
    max_residual_force_eV_A: float = 5e-3
    allow_nonstationary: bool = False
    acoustic_sum_rule: bool = True
    asr_tolerance: float = 0.02
    symmetrize_iterations: int = 3
    dos_points: int = 801
    dos_width_THz: float = 0.10

    def __post_init__(self) -> None:
        object.__setattr__(self, "supercell", tuple(self.supercell))
        object.__setattr__(self, "q_path", tuple(tuple(q) for q in self.q_path))
        object.__setattr__(self, "dos_mesh", tuple(self.dos_mesh))
        object.__setattr__(self, "q_labels", tuple(str(v) for v in self.q_labels))
        self.validate()

    def validate(self) -> "PeriodicPhononSettings":
        _positive_integer_triplet(self.supercell, "supercell")
        _positive_integer_triplet(self.dos_mesh, "dos_mesh")
        if not (1e-4 <= float(self.displacement_A) <= 0.1):
            raise PhononError("displacement_A must lie between 1e-4 and 0.1 A.")
        if not float(self.max_residual_force_eV_A) > 0:
            raise PhononError("max_residual_force_eV_A must be positive.")
        if not (0 < float(self.asr_tolerance) < 1):
            raise PhononError("asr_tolerance must lie between 0 and 1.")
        if not isinstance(self.symmetrize_iterations, int) or isinstance(
                self.symmetrize_iterations, bool) or not 1 <= self.symmetrize_iterations <= 20:
            raise PhononError("symmetrize_iterations must be an integer from 1 through 20.")
        if not self.q_path:
            raise PhononError("q_path must contain at least one explicit reciprocal point.")
        if len(self.q_path) > MAX_Q_PATH_POINTS:
            raise PhononRefused(
                f"q_path has {len(self.q_path)} points; the limit is {MAX_Q_PATH_POINTS}.")
        q = np.asarray(self.q_path, dtype=float)
        if q.shape != (len(self.q_path), 3) or not np.all(np.isfinite(q)):
            raise PhononError("q_path must contain triples of finite reciprocal coordinates.")
        if self.q_labels and len(self.q_labels) != len(self.q_path):
            raise PhononError("q_labels must be empty or have one entry for every q_path point.")
        mesh_points = math.prod(self.dos_mesh)
        if mesh_points > MAX_DOS_Q_POINTS:
            raise PhononRefused(
                f"dos_mesh contains {mesh_points} q points; the limit is {MAX_DOS_Q_POINTS}.")
        if not isinstance(self.dos_points, int) or isinstance(self.dos_points, bool) or not (
                51 <= self.dos_points <= MAX_DOS_GRID_POINTS):
            raise PhononError(
                f"dos_points must be an integer from 51 through {MAX_DOS_GRID_POINTS}.")
        if not (1e-4 <= float(self.dos_width_THz) <= 100.0):
            raise PhononError("dos_width_THz must lie between 1e-4 and 100 THz.")
        return self

    def as_dict(self) -> dict:
        value = asdict(self)
        value["supercell"] = list(self.supercell)
        value["q_path"] = [list(q) for q in self.q_path]
        value["dos_mesh"] = list(self.dos_mesh)
        value["q_labels"] = list(self.q_labels)
        return value


def _positive_integer_triplet(value: tuple, name: str) -> None:
    if len(value) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                              for v in value):
        raise PhononError(f"{name} must be three positive integers.")


@dataclass
class PeriodicPhononResult:
    """Array-ready dispersion, DOS and real-space force constants."""

    q_points_scaled: np.ndarray
    q_distances_A_inv: np.ndarray
    q_labels: Tuple[str, ...]
    frequencies_THz: np.ndarray
    mode_kinds: List[List[str]]
    frequency_resolution_THz: np.ndarray
    dos_q_points_scaled: np.ndarray
    dos_raw_frequencies_THz: np.ndarray
    dos_raw_weights: np.ndarray
    dos_frequency_grid_THz: np.ndarray
    dos_density_states_per_THz: np.ndarray
    force_constants_eV_A2: np.ndarray
    lattice_vectors: np.ndarray
    diagnostics: Dict[str, Any]
    provenance: Provenance
    convergence: Convergence

    def results(self) -> Dict[str, Result]:
        common = {
            "q_points_scaled": self.q_points_scaled.tolist(),
            "q_distances_A_inv": self.q_distances_A_inv.tolist(),
            "q_labels": list(self.q_labels),
            "mode_kinds": [list(row) for row in self.mode_kinds],
            "frequency_resolution_THz": self.frequency_resolution_THz.tolist(),
            "negative_means": "imaginary frequency",
            "diagnostics": dict(self.diagnostics),
        }
        dos_extra = {
            "frequency_grid_THz": self.dos_frequency_grid_THz.tolist(),
            "mesh": list(self.provenance.parameters["settings"]["dos_mesh"]),
            "gaussian_width_THz": self.provenance.parameters["settings"]["dos_width_THz"],
            "integral_states": self.diagnostics["dos_integral_states"],
            "negative_frequency_weight_states": self.diagnostics[
                "dos_negative_frequency_weight_states"],
        }
        return {
            "phonon_dispersion": Result(
                "phonon_dispersion", self.frequencies_THz.copy(), "THz", self.provenance,
                uncertainty=float(np.max(self.frequency_resolution_THz)),
                uncertainty_kind="", convergence=self.convergence,
                extra={
                    **common,
                    "resolution_note": (
                        "Classification threshold from measured matrix correction, raw "
                        "anti-Hermitian part and an h^2 finite-difference scale; it is not "
                        "a bound on force-model or supercell error."),
                }),
            "phonon_dos": Result(
                "phonon_density_of_states", self.dos_density_states_per_THz.copy(),
                "states/THz", self.provenance, convergence=self.convergence,
                extra=dos_extra),
            "real_space_force_constants": Result(
                "real_space_force_constants", self.force_constants_eV_A2.copy(),
                "eV/A^2", self.provenance, convergence=self.convergence,
                extra={
                    "lattice_vectors": self.lattice_vectors.tolist(),
                    "layout": "(supercell lattice vector, 3N primitive, 3N primitive)",
                }),
        }

    def stored_arrays(self) -> Dict[str, Any]:
        """Return every numerical product in Materia's checksummed array form."""
        from ..project_format.arrays import StoredArray

        meta = {
            "model": self.provenance.model,
            "inputs_digest": self.provenance.inputs_digest,
            "negative_means": "imaginary frequency",
        }
        return {
            "phonon_q_points": StoredArray(
                self.q_points_scaled, "reciprocal-lattice coordinates",
                "Explicit phonon path points", "table", meta),
            "phonon_q_distances": StoredArray(
                self.q_distances_A_inv, "1/A", "Cumulative reciprocal-path distance",
                "table", meta),
            "phonon_frequencies": StoredArray(
                self.frequencies_THz, "THz",
                "Signed phonon dispersion; negative frequencies are imaginary modes",
                "table", {**meta, "q_labels": list(self.q_labels)}),
            "phonon_frequency_resolution": StoredArray(
                self.frequency_resolution_THz, "THz",
                "Per-q upper resolution estimate", "table", meta),
            "phonon_real_space_force_constants": StoredArray(
                self.force_constants_eV_A2, "eV/A^2",
                "Real-space force constants indexed by supercell lattice vector",
                "table", meta),
            "phonon_lattice_vectors": StoredArray(
                self.lattice_vectors, "primitive lattice vectors",
                "Integer cell vectors associated with the real-space force constants",
                "table", meta),
            "phonon_dos_q_points": StoredArray(
                self.dos_q_points_scaled, "reciprocal-lattice coordinates",
                "Monkhorst-Pack q points used for the phonon DOS", "table", meta),
            "phonon_dos_raw_frequencies": StoredArray(
                self.dos_raw_frequencies_THz, "THz",
                "Signed mesh frequencies before Gaussian broadening", "table", meta),
            "phonon_dos_raw_weights": StoredArray(
                self.dos_raw_weights, "states",
                "Uniform q-point weights for every raw phonon mode", "table", meta),
            "phonon_dos_frequency_grid": StoredArray(
                self.dos_frequency_grid_THz, "THz",
                "Frequency grid for the broadened phonon DOS", "table", meta),
            "phonon_dos_density": StoredArray(
                self.dos_density_states_per_THz, "states/THz",
                "Gaussian-broadened phonon density of states", "spectrum", meta),
        }

    def as_dict(self) -> dict:
        arrays = self.stored_arrays()
        return {
            "q_points_scaled": self.q_points_scaled.tolist(),
            "q_distances_A_inv": self.q_distances_A_inv.tolist(),
            "q_labels": list(self.q_labels),
            "mode_kinds": [list(row) for row in self.mode_kinds],
            "diagnostics": dict(self.diagnostics),
            "array_checksums_sha256": {key: value.sha256() for key, value in arrays.items()},
            "array_shapes": {key: value.shape for key, value in arrays.items()},
            "provenance": self.provenance.as_dict(),
            "convergence": self.convergence.as_dict(),
        }


def materia_ase_calculator(potential: Any) -> Any:
    """Adapt a Materia potential to an ASE calculator without changing either input."""
    try:
        from ase.calculators.calculator import Calculator, all_changes
    except ImportError as exc:
        raise PhononRefused(
            "Periodic phonons require the optional ASE adapter dependency.") from exc

    class MateriaPotentialCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def calculate(self, atoms=None, properties=("energy", "forces"),
                      system_changes=all_changes) -> None:
            super().calculate(atoms, properties, system_changes)
            structure = Structure(
                numbers=np.asarray(atoms.numbers, dtype=int),
                positions=np.asarray(atoms.positions, dtype=float),
                cell=Cell(np.asarray(atoms.cell, dtype=float), tuple(bool(v) for v in atoms.pbc)),
            )
            energy, forces = self.wrapped.energy_and_forces(structure)
            forces = np.asarray(forces, dtype=float)
            if forces.shape != (len(structure), 3) or not np.all(np.isfinite(forces)):
                raise PhononError("The Materia potential returned an invalid force array.")
            if not math.isfinite(float(energy)):
                raise PhononError("The Materia potential returned a non-finite energy.")
            self.results = {"energy": float(energy), "forces": forces}

    return MateriaPotentialCalculator(potential)


def periodic_phonon_analysis(
    structure: Structure,
    source: Any,
    settings: PeriodicPhononSettings,
    *,
    name: str = "",
    fidelity: Optional[Fidelity] = None,
    progress: Optional[Progress] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> PeriodicPhononResult:
    """Compute periodic dispersion and DOS, returning nothing after refusal or cancellation."""
    t0 = time.perf_counter()
    settings.validate()
    from .external_bias import has_external_bias

    if has_external_bias(structure):
        raise PhononRefused(
            "Periodic phonon dispersion refuses structures with external forces or "
            "restraints because their supercell repetition and translational symmetry "
            "are ambiguous. Clear the biases before this calculation.")
    _validate_structure_and_size(structure, settings)
    calculator, model_name, model_fidelity, description, cutoff = _source(
        source, structure, name, fidelity)
    _validate_cutoff(structure, settings.supercell, cutoff)
    try:
        import ase
        from ase import Atoms
        from ase.dft.kpoints import monkhorst_pack
        from ase.phonons import Phonons
    except ImportError as exc:
        raise PhononRefused(
            "Periodic phonons require the optional ASE adapter dependency.") from exc

    q_path = np.asarray(settings.q_path, dtype=float)
    atoms = Atoms(
        numbers=structure.numbers,
        positions=np.asarray(structure.positions, dtype=float).copy(),
        cell=np.asarray(structure.cell.matrix, dtype=float),
        pbc=True,
        masses=structure.masses(),
    )
    _report(progress, cancelled, 0.0, "reference supercell forces")
    super_atoms = atoms * settings.supercell
    super_atoms.calc = calculator
    reference_forces = np.asarray(super_atoms.get_forces(), dtype=float)
    if reference_forces.shape != (len(super_atoms), 3) or not np.all(
            np.isfinite(reference_forces)):
        raise PhononError("The force source returned an invalid reference force array.")
    residual = float(np.linalg.norm(reference_forces, axis=1).max())
    stationary = residual <= settings.max_residual_force_eV_A
    if not stationary and not settings.allow_nonstationary:
        raise PhononRefused(
            f"The largest force in the reference supercell is {residual:.3g} eV/A, above "
            f"{settings.max_residual_force_eV_A:g} eV/A. Phonons are defined at a "
            "stationary point: relax the structure first, or set allow_nonstationary for "
            "an explicitly estimated curvature calculation.")

    total_displacements = 1 + 6 * len(structure)
    completed = 0

    class CancelablePhonons(Phonons):
        def calculate(self, atoms_N, disp):
            nonlocal completed
            _report(
                progress, cancelled, 0.05 + 0.62 * completed / total_displacements,
                f"finite displacement {completed + 1} of {total_displacements}")
            output = super().calculate(atoms_N, disp)
            completed += 1
            _report(
                progress, cancelled, 0.05 + 0.62 * completed / total_displacements,
                f"finite displacement {completed} of {total_displacements}")
            return output

    temporary_parent = None
    raw_force_constants = None
    final_force_constants = None
    lattice_vectors = None
    path_frequencies = None
    dos_q_points = monkhorst_pack(settings.dos_mesh)
    dos_frequencies = None
    ph = None
    with tempfile.TemporaryDirectory(prefix="materia-phonons-") as temporary_parent:
        ph = CancelablePhonons(
            atoms, calculator, supercell=settings.supercell,
            name=str(Path(temporary_parent) / "displacements"),
            delta=settings.displacement_A, center_refcell=True)
        try:
            ph.run()
            _report(progress, cancelled, 0.69, "reading real-space force constants")
            ph.read(method="standard", symmetrize=0, acoustic=False)
            raw_force_constants = np.asarray(ph.C_N, dtype=float).copy()
            raw_violation, raw_violation_relative = _sum_rule_violation(
                raw_force_constants, structure.masses())
            if raw_violation_relative > settings.asr_tolerance:
                raise PhononRefused(
                    f"The acoustic sum rule is violated by {raw_violation:.3g} eV/A^2, "
                    f"{raw_violation_relative:.2%} of the largest Gamma diagonal force "
                    "constant (on-site for one atom), above the "
                    f"{settings.asr_tolerance:.0%} tolerance. The force model or finite-difference calculation is not sufficiently "
                    "translation invariant.")
            ph.read(
                method="standard", symmetrize=settings.symmetrize_iterations,
                acoustic=settings.acoustic_sum_rule)
            final_force_constants = np.asarray(ph.C_N, dtype=float).copy()
            lattice_vectors = np.asarray(ph._lattice_vectors_array, dtype=int).T.copy()
            path_frequencies = _band_frequencies(
                ph, q_path, 0.72, 0.84, progress, cancelled)
            dos_frequencies = _band_frequencies(
                ph, dos_q_points, 0.84, 0.95, progress, cancelled)
        finally:
            ph.clean()

    if raw_force_constants is None or final_force_constants is None \
            or lattice_vectors is None or path_frequencies is None or dos_frequencies is None:
        raise PhononError("The periodic phonon calculation did not produce complete arrays.")
    _report(progress, cancelled, 0.96, "classifying modes and broadening DOS")
    resolutions, kinds, acoustic_zeroed, diagnostic_terms = _classify_path(
        ph, q_path, path_frequencies, raw_force_constants, final_force_constants,
        structure.masses(), settings)
    grid, density, raw_weights = _dos(
        dos_frequencies, settings.dos_points, settings.dos_width_THz)
    distances = _q_distances(q_path, structure.cell.reciprocal)
    final_violation, final_violation_relative = _sum_rule_violation(
        final_force_constants, structure.masses())
    negative_path = int(np.count_nonzero(path_frequencies < -resolutions[:, None]))
    negative_dos_weight = float(raw_weights[dos_frequencies < 0].sum())
    integral = _trapezoid(density, grid)
    diagnostics = {
        "ase_version": ase.__version__,
        "primitive_atoms": len(structure),
        "supercell": list(settings.supercell),
        "supercell_atoms": len(structure) * math.prod(settings.supercell),
        "force_evaluations": 1 + completed,
        "residual_force_eV_A": residual,
        "stationary": stationary,
        "raw_sum_rule_violation_eV_A2": raw_violation,
        "raw_sum_rule_violation_relative": raw_violation_relative,
        "final_sum_rule_violation_eV_A2": final_violation,
        "final_sum_rule_violation_relative": final_violation_relative,
        "acoustic_sum_rule_applied": settings.acoustic_sum_rule,
        "gamma_acoustic_modes_zeroed": acoustic_zeroed,
        "imaginary_path_modes": negative_path,
        "dos_mesh_q_points": len(dos_q_points),
        "dos_integral_states": integral,
        "dos_negative_frequency_weight_states": negative_dos_weight,
        "temporary_displacements_cleaned": not Path(temporary_parent).exists(),
        "wall_time_s": time.perf_counter() - t0,
        **diagnostic_terms,
    }
    approximations = list(description.get("approximations", [])) + [
        "Harmonic approximation: the potential energy is expanded to second order about "
        "the supplied reference structure; no anharmonic shifts, lifetimes or thermal "
        "expansion are included.",
        f"Two-point central finite differences with a {settings.displacement_A:g} A "
        "displacement, evaluated in the explicit "
        f"{settings.supercell[0]} x {settings.supercell[1]} x {settings.supercell[2]} "
        "supercell.",
        "Real-space force constants are truncated by the chosen supercell and Fourier "
        "interpolated by ASE; supercell convergence must be checked for quantitative use.",
        ("The acoustic sum rule is imposed during ASE force-constant symmetrisation."
         if settings.acoustic_sum_rule else
         "The acoustic sum rule is checked but not imposed."),
        f"The DOS uses a uniform {settings.dos_mesh[0]} x {settings.dos_mesh[1]} x "
        f"{settings.dos_mesh[2]} Monkhorst-Pack mesh and Gaussian broadening of "
        f"{settings.dos_width_THz:g} THz; its integral is normalised to 3N states.",
        "No non-analytic dipole correction is applied: Born effective charges and the "
        "dielectric tensor are not computed, so LO-TO splitting is absent.",
        "Branches are sorted by frequency independently at each q point; crossings are not "
        "tracked by eigenvector overlap.",
        "The reported frequency resolution is a numerical classification threshold, not "
        "a bound on force-model error or finite-supercell truncation.",
        "Classical nuclear masses are taken from the structure, including assigned isotopes.",
    ]
    if not stationary:
        approximations.append(
            f"The reference is not stationary (largest force {residual:.3g} eV/A); the "
            "reported eigenvalues are local curvatures, not vibrations about equilibrium.")
    settings_dict = settings.as_dict()
    inputs_digest = digest({
        "numbers": structure.numbers.tolist(),
        "positions": np.asarray(structure.positions, dtype=float).tolist(),
        "cell": structure.cell.matrix.tolist(),
        "pbc": list(structure.cell.pbc),
        "masses": structure.masses().tolist(),
        "settings": settings_dict,
        "model": model_name,
        "model_parameters": description.get("parameters", {}),
        "ase_version": ase.__version__,
    })
    provenance = Provenance(
        model=f"phonons/ase-real-space-finite-displacement[{model_name}]",
        fidelity=model_fidelity,
        origin=Origin.CALCULATED if stationary else Origin.ESTIMATED,
        approximations=approximations,
        tolerances={
            "displacement_A": settings.displacement_A,
            "max_residual_force_eV_A": settings.max_residual_force_eV_A,
            "asr_tolerance": settings.asr_tolerance,
            "dos_width_THz": settings.dos_width_THz,
        },
        boundary_conditions="3D periodic primitive cell with explicit finite supercell",
        parameters={
            "settings": settings_dict,
            "force_model": model_name,
            "force_model_parameters": description.get("parameters", {}),
            "ase_version": ase.__version__,
            "ase_force_constant_method": "standard",
            "lo_to_correction": False,
        },
        references=list(REFERENCES) + list(description.get("references", [])),
        inputs_digest=inputs_digest,
    )
    converged = stationary and raw_violation_relative <= settings.asr_tolerance
    convergence = Convergence(
        converged=converged,
        iterations=1 + completed,
        residual=residual,
        residual_metric="largest force in the undisplaced supercell",
        tolerance=settings.max_residual_force_eV_A,
        message=(
            "Stationary reference and acoustic sum-rule violation within tolerance; "
            "supercell, displacement and q-mesh convergence remain user checks."
            if converged else
            "Reference is nonstationary; force constants are an estimated local curvature."),
    )
    _report(progress, cancelled, 1.0, "done")
    return PeriodicPhononResult(
        q_points_scaled=q_path,
        q_distances_A_inv=distances,
        q_labels=settings.q_labels,
        frequencies_THz=path_frequencies,
        mode_kinds=kinds,
        frequency_resolution_THz=resolutions,
        dos_q_points_scaled=np.ascontiguousarray(dos_q_points, dtype=float),
        dos_raw_frequencies_THz=dos_frequencies,
        dos_raw_weights=raw_weights,
        dos_frequency_grid_THz=grid,
        dos_density_states_per_THz=density,
        force_constants_eV_A2=final_force_constants,
        lattice_vectors=lattice_vectors,
        diagnostics=diagnostics,
        provenance=provenance,
        convergence=convergence,
    )


def _validate_structure_and_size(
        structure: Structure, settings: PeriodicPhononSettings) -> None:
    if len(structure) == 0:
        raise PhononError("The primitive structure has no atoms.")
    if not all(structure.cell.pbc):
        raise PhononRefused(
            "Periodic dispersion requires a fully three-dimensional periodic primitive "
            "cell; wire and slab dispersions are not implemented.")
    if not math.isfinite(structure.cell.volume) or structure.cell.volume <= 1e-9:
        raise PhononError("The periodic primitive cell must have a finite non-zero volume.")
    if np.asarray(structure.fixed, dtype=bool).any():
        raise PhononRefused(
            "Periodic dispersion does not support fixed atoms because a partial force-"
            "constant matrix does not obey crystal translational symmetry.")
    if len(structure) > MAX_PRIMITIVE_ATOMS:
        raise PhononRefused(
            f"The primitive cell has {len(structure)} atoms; the limit is "
            f"{MAX_PRIMITIVE_ATOMS}.")
    supercell_atoms = len(structure) * math.prod(settings.supercell)
    if supercell_atoms > MAX_SUPERCELL_ATOMS:
        raise PhononRefused(
            f"The requested supercell has {supercell_atoms} atoms; the limit is "
            f"{MAX_SUPERCELL_ATOMS}.")
    bands = 3 * len(structure)
    path_values = len(settings.q_path) * bands
    dos_values = math.prod(settings.dos_mesh) * bands
    if path_values > MAX_SPECTRAL_VALUES or dos_values > MAX_SPECTRAL_VALUES:
        raise PhononRefused(
            f"The request would create {path_values} path and {dos_values} DOS spectral "
            f"values; each is limited to {MAX_SPECTRAL_VALUES}.")


def _source(source: Any, structure: Structure, name: str,
            fidelity: Optional[Fidelity]) -> Tuple[Any, str, Fidelity, dict, Optional[float]]:
    if hasattr(source, "energy_and_forces") and hasattr(source, "describe"):
        if hasattr(source, "supports"):
            ok, reason = source.supports(structure)
            if not ok:
                raise PhononRefused(reason)
        description = dict(source.describe())
        model_name = name or source.model_label()
        model_fidelity = fidelity or source.fidelity
        cutoff = float(getattr(source, "cutoff_A", 0.0)) or None
        return materia_ase_calculator(source), model_name, model_fidelity, description, cutoff
    if hasattr(source, "get_forces"):
        parameters = dict(getattr(source, "parameters", {}) or {})
        model_name = name or f"ase/{type(source).__name__}"
        description = {
            "parameters": parameters,
            "approximations": [
                "External ASE calculator supplied by the caller; Materia has not validated "
                "its force model, cutoff or domain of applicability."],
            "references": [],
        }
        return source, model_name, fidelity or Fidelity.NON_PHYSICAL, description, None
    raise PhononError(
        "Periodic phonons require a Materia potential or an ASE calculator capable of "
        "evaluating arbitrary repeated supercells.")


def _validate_cutoff(structure: Structure, supercell: tuple,
                     cutoff_A: Optional[float]) -> None:
    if cutoff_A is None:
        return
    matrix = np.asarray(structure.cell.matrix, dtype=float).copy()
    for axis, repeat in enumerate(supercell):
        matrix[axis] *= repeat
    volume = abs(float(np.linalg.det(matrix)))
    widths = []
    for axis in range(3):
        normal = np.cross(matrix[(axis + 1) % 3], matrix[(axis + 2) % 3])
        widths.append(volume / float(np.linalg.norm(normal)))
    narrowest = min(widths)
    if narrowest <= 2.0 * cutoff_A:
        raise PhononRefused(
            f"The narrowest requested supercell width is {narrowest:.3f} A, not larger "
            f"than twice the {cutoff_A:g} A force cutoff. Real-space interactions would "
            "alias across periodic images; increase the supercell.")


def _report(progress: Optional[Progress], cancelled: Optional[Callable[[], bool]],
            fraction: float, message: str) -> None:
    if cancelled is not None and cancelled():
        raise PhononCancelled("Periodic phonon analysis cancelled.")
    if progress is not None and progress(float(fraction), message) is False:
        raise PhononCancelled("Periodic phonon analysis cancelled.")


def _band_frequencies(ph: Any, q_points: np.ndarray, start: float, stop: float,
                      progress: Optional[Progress], cancelled: Optional[Callable[[], bool]]) \
        -> np.ndarray:
    from .phonons import THZ_TO_MEV

    chunks = []
    batch = 256
    for first in range(0, len(q_points), batch):
        last = min(len(q_points), first + batch)
        energy_eV = ph.band_structure(q_points[first:last], verbose=False)
        chunks.append(np.asarray(energy_eV, dtype=float) * 1000.0 / THZ_TO_MEV)
        fraction = start + (stop - start) * last / len(q_points)
        _report(progress, cancelled, fraction, f"Fourier interpolation {last} of {len(q_points)}")
    return np.concatenate(chunks, axis=0)


def _sum_rule_violation(force_constants: np.ndarray,
                        masses: np.ndarray) -> Tuple[float, float]:
    gamma = np.asarray(force_constants, dtype=float).sum(axis=0)
    n = len(masses)
    translations = np.zeros((3 * n, 3))
    for axis in range(3):
        translations[axis::3, axis] = 1.0 / math.sqrt(n)
    violation = float(np.abs(gamma @ translations).max()) * math.sqrt(n)
    if n == 1:
        scale = float(np.abs(np.diagonal(force_constants, axis1=1, axis2=2)).max()) or 1.0
    else:
        scale = float(np.abs(np.diag(gamma)).max()) or 1.0
    return violation, violation / scale


def _classify_path(
    ph: Any,
    q_points: np.ndarray,
    frequencies: np.ndarray,
    raw_force_constants: np.ndarray,
    final_force_constants: np.ndarray,
    masses: np.ndarray,
    settings: PeriodicPhononSettings,
) -> Tuple[np.ndarray, List[List[str]], int, Dict[str, float]]:
    repeated_inverse_mass = np.repeat(np.asarray(masses, dtype=float) ** -0.5, 3)
    mass_factor = np.outer(repeated_inverse_mass, repeated_inverse_mass)
    raw_d = raw_force_constants * mass_factor[None, :, :]
    final_d = final_force_constants * mass_factor[None, :, :]
    translation = np.zeros((3 * len(masses), 3))
    for axis in range(3):
        translation[axis::3, axis] = np.sqrt(masses)
    translation, _ = np.linalg.qr(translation)
    resolutions = []
    kinds: List[List[str]] = []
    max_correction = 0.0
    max_antisymmetry = 0.0
    acoustic_zeroed = 0
    for iq, q in enumerate(q_points):
        raw_q = ph.compute_dynamical_matrix(q, raw_d)
        final_q = ph.compute_dynamical_matrix(q, final_d)
        hermitian_raw = 0.5 * (raw_q + raw_q.conj().T)
        hermitian_final = 0.5 * (final_q + final_q.conj().T)
        correction = float(np.linalg.norm(hermitian_final - hermitian_raw, 2))
        antisymmetry = float(np.linalg.norm(0.5 * (raw_q - raw_q.conj().T), 2))
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian_final)
        floor = float(np.abs(eigenvalues).max()) * (
            settings.displacement_A / TRUNCATION_LENGTH_A) ** 2
        resolution = max(correction, antisymmetry, floor)
        resolution_thz = math.sqrt(resolution * EIGENVALUE_TO_RAD_S2) / (
            2.0 * math.pi * 1e12)
        resolutions.append(resolution_thz)
        row = []
        is_gamma = bool(np.all(np.abs(q - np.round(q)) < 1e-12))
        for mode, eigenvalue in enumerate(eigenvalues):
            overlap = float(np.sum(np.abs(translation.conj().T @ eigenvectors[:, mode]) ** 2))
            if is_gamma and settings.acoustic_sum_rule and overlap > RIGID_OVERLAP \
                    and abs(eigenvalue) <= resolution:
                row.append("acoustic")
                frequencies[iq, mode] = 0.0
                acoustic_zeroed += 1
            elif abs(eigenvalue) <= resolution:
                row.append("zero-within-resolution")
            elif eigenvalue < 0:
                row.append("imaginary")
            else:
                row.append("real")
        kinds.append(row)
        max_correction = max(max_correction, correction)
        max_antisymmetry = max(max_antisymmetry, antisymmetry)
    return np.asarray(resolutions), kinds, acoustic_zeroed, {
        "max_force_constant_correction_eigenvalue_eV_A2_u": max_correction,
        "max_raw_antihermitian_eigenvalue_eV_A2_u": max_antisymmetry,
    }


def _dos(frequencies: np.ndarray, points: int,
         width: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(frequencies, dtype=float).ravel()
    weight = np.full(flat.shape, 1.0 / frequencies.shape[0], dtype=float)
    lower = float(flat.min()) - 5.0 * width
    upper = float(flat.max()) + 5.0 * width
    if upper <= lower:
        lower, upper = -5.0 * width, 5.0 * width
    grid = np.linspace(lower, upper, points)
    density = np.zeros(points, dtype=float)
    batch = max(1, min(2048, 1_000_000 // points))
    factor = 1.0 / (math.sqrt(2.0 * math.pi) * width)
    for first in range(0, len(flat), batch):
        values = flat[first:first + batch]
        weights = weight[first:first + batch]
        z = (grid[:, None] - values[None, :]) / width
        density += factor * np.sum(np.exp(-0.5 * z * z) * weights[None, :], axis=1)
    target = float(weight.sum())
    integral = _trapezoid(density, grid)
    if integral > 0:
        density *= target / integral
    return grid, density, weight.reshape(frequencies.shape)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _q_distances(q_points: np.ndarray, reciprocal: np.ndarray) -> np.ndarray:
    cartesian = np.asarray(q_points, dtype=float) @ np.asarray(reciprocal, dtype=float)
    distances = np.zeros(len(q_points), dtype=float)
    if len(q_points) > 1:
        distances[1:] = np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))
    return distances


__all__ = [
    "MAX_PRIMITIVE_ATOMS", "MAX_SUPERCELL_ATOMS", "MAX_Q_PATH_POINTS",
    "MAX_DOS_Q_POINTS", "MAX_SPECTRAL_VALUES", "MAX_DOS_GRID_POINTS",
    "PeriodicPhononSettings", "PeriodicPhononResult", "materia_ase_calculator",
    "periodic_phonon_analysis",
]
