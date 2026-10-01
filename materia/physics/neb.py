"""Nudged elastic band paths and climbing-image reaction barriers.

The implementation uses ASE's NEB geometry and FIRE optimiser while every
energy and force evaluation comes from the selected Materia potential. Input
structures are copied, never changed. Endpoints must describe the same atoms
in the same order and, by default, must already be stationary minima.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..provenance import Convergence, Origin, Provenance, Result, digest
from ..project_format.arrays import StoredArray

NEB_REFERENCE = (
    "G. Henkelman and H. Jonsson, J. Chem. Phys. 113 (2000) 9978"
)
CLIMB_REFERENCE = (
    "G. Henkelman, B. P. Uberuaga and H. Jonsson, J. Chem. Phys. 113 (2000) 9901"
)
MAX_IMAGES = 64
MAX_ATOMS = 20000


class NEBError(ValueError):
    """A reaction-path calculation has invalid input or failed."""


class NEBRefused(NEBError):
    """The requested path cannot be represented honestly by this method."""


class NEBCancelled(NEBError):
    """The caller cancelled the path optimisation."""


@dataclass(frozen=True)
class NEBSettings:
    images: int = 7
    spring_eV_A2: float = 0.1
    fmax_eV_A: float = 0.05
    endpoint_fmax_eV_A: float = 0.05
    max_steps: int = 1000
    interpolation: str = "idpp"
    climb: bool = True
    allow_unrelaxed_endpoints: bool = False
    remove_rotation_translation: bool = False
    fire_dt: float = 0.1
    fire_maxstep_A: float = 0.2

    def validate(self) -> "NEBSettings":
        if type(self.images) is not int or not 3 <= self.images <= MAX_IMAGES:
            raise NEBError(f"images must be an integer from 3 through {MAX_IMAGES}.")
        if not np.isfinite(self.spring_eV_A2) or self.spring_eV_A2 <= 0:
            raise NEBError("spring_eV_A2 must be finite and positive.")
        if not np.isfinite(self.fmax_eV_A) or not 0 < self.fmax_eV_A <= 10:
            raise NEBError("fmax_eV_A must be finite and lie between 0 and 10.")
        if not np.isfinite(self.endpoint_fmax_eV_A) or not 0 < self.endpoint_fmax_eV_A <= 10:
            raise NEBError("endpoint_fmax_eV_A must be finite and lie between 0 and 10.")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise NEBError("max_steps must be a positive integer.")
        if self.interpolation not in ("linear", "idpp"):
            raise NEBError("interpolation must be 'linear' or 'idpp'.")
        if not np.isfinite(self.fire_dt) or self.fire_dt <= 0:
            raise NEBError("fire_dt must be finite and positive.")
        if not np.isfinite(self.fire_maxstep_A) or not 0 < self.fire_maxstep_A <= 1:
            raise NEBError("fire_maxstep_A must be finite and lie between 0 and 1 A.")
        return self

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class NEBPath:
    images: List[Structure]
    energies_eV: np.ndarray
    forces_eV_A: np.ndarray
    reaction_coordinate_A: np.ndarray
    history_step: np.ndarray
    history_barrier_eV: np.ndarray
    history_max_force_eV_A: np.ndarray
    barrier_eV: float
    forward_barrier_eV: float
    reverse_barrier_eV: float
    reaction_energy_eV: float
    saddle_image: int
    endpoint_forces_eV_A: Sequence[float]
    settings: NEBSettings
    provenance: Provenance
    convergence: Convergence

    def results(self) -> Dict[str, Result]:
        common = {
            "saddle_image": self.saddle_image,
            "endpoint_forces_eV_A": list(self.endpoint_forces_eV_A),
            "images": len(self.images),
            "settings": self.settings.as_dict(),
        }
        return {
            "barrier": Result(
                "reaction_barrier", self.barrier_eV, "eV", self.provenance,
                convergence=self.convergence,
                extra={**common, "forward_eV": self.forward_barrier_eV,
                       "reverse_eV": self.reverse_barrier_eV,
                       "reaction_energy_eV": self.reaction_energy_eV}),
            "energies": Result(
                "neb_image_energies", self.energies_eV.copy(), "eV", self.provenance,
                convergence=self.convergence,
                extra={**common, "relative_to_initial_eV":
                       (self.energies_eV - self.energies_eV[0]).tolist()}),
            "reaction_coordinate": Result(
                "neb_reaction_coordinate", self.reaction_coordinate_A.copy(), "A",
                self.provenance, convergence=self.convergence, extra=common),
            "path_forces": Result(
                "neb_path_forces", self.forces_eV_A.copy(), "eV/A", self.provenance,
                convergence=self.convergence, extra=common),
        }

    def stored_arrays(self) -> Dict[str, StoredArray]:
        meta = {"model": self.provenance.model,
                "input_digest": self.provenance.inputs_digest,
                "saddle_image": self.saddle_image}
        return {
            "positions": StoredArray(
                np.asarray([image.positions for image in self.images]), "A",
                "NEB path image positions", "trajectory", meta),
            "energies": StoredArray(
                self.energies_eV, "eV", "NEB image energies", "table", meta),
            "forces": StoredArray(
                self.forces_eV_A, "eV/A", "physical forces on every path image",
                "trajectory", meta),
            "reaction_coordinate": StoredArray(
                self.reaction_coordinate_A, "A", "cumulative path length", "table", meta),
            "history": StoredArray(
                np.column_stack([self.history_step, self.history_barrier_eV,
                                 self.history_max_force_eV_A]),
                "mixed", "step, barrier_eV, maximum NEB force_eV_A", "table", meta),
        }


def _check_endpoints(initial: Structure, final: Structure, settings: NEBSettings) -> None:
    if not 1 <= len(initial) <= MAX_ATOMS:
        raise NEBRefused(
            f"A path must contain 1 through {MAX_ATOMS} atoms; found {len(initial)}.")
    if len(initial) != len(final):
        raise NEBRefused("Initial and final structures have different atom counts.")
    if not np.array_equal(initial.ids, final.ids):
        raise NEBRefused("Initial and final structures must have the same atom ids in order.")
    if not np.array_equal(initial.numbers, final.numbers):
        raise NEBRefused("Initial and final structures must have the same elements in order.")
    if initial.cell.pbc != final.cell.pbc or not np.allclose(
            initial.cell.matrix, final.cell.matrix, rtol=0.0, atol=1e-10):
        raise NEBRefused("Initial and final structures must have the same cell and periodicity.")
    if not np.array_equal(initial.fixed, final.fixed):
        raise NEBRefused("Initial and final structures must have the same fixed atoms.")
    if np.all(initial.fixed):
        raise NEBRefused("Every atom is fixed, so no reaction path can move.")
    if settings.remove_rotation_translation and any(initial.cell.pbc):
        raise NEBRefused(
            "Rotation and translation removal is only defined for an isolated system.")
    if np.any(initial.fixed):
        moved = np.linalg.norm(
            initial.cell.minimum_image(final.positions - initial.positions), axis=1)
        if np.any(moved[initial.fixed] > 1e-10):
            raise NEBRefused("A fixed atom has different endpoint positions.")
    settings.validate()


def _ase_atoms(structure: Structure):
    from ase import Atoms
    from ase.constraints import FixAtoms

    atoms = Atoms(
        numbers=structure.numbers, positions=structure.positions,
        cell=structure.cell.matrix, pbc=structure.cell.pbc,
        masses=structure.masses())
    if np.any(structure.fixed):
        atoms.set_constraint(FixAtoms(mask=structure.fixed))
    return atoms


def _calculator(potential, template: Structure):
    from ase.calculators.calculator import Calculator, all_changes

    class MateriaCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"),
                      system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            structure = template.copy()
            structure.positions = np.asarray(atoms.positions, dtype=float)
            structure.cell = Cell(np.asarray(atoms.cell), tuple(bool(v) for v in atoms.pbc))
            energy, forces = potential.energy_and_forces(structure)
            energy = float(energy)
            forces = np.asarray(forces, dtype=float)
            if not np.isfinite(energy) or forces.shape != (len(structure), 3) or not np.all(
                    np.isfinite(forces)):
                raise NEBError("The force model returned non-finite or malformed values.")
            self.results = {"energy": energy, "forces": forces}

    return MateriaCalculator()


def _coordinate(images: Sequence[Structure]) -> np.ndarray:
    coordinate = [0.0]
    for before, after in zip(images, images[1:]):
        delta = before.cell.minimum_image(after.positions - before.positions)
        coordinate.append(coordinate[-1] + float(np.linalg.norm(delta)))
    return np.asarray(coordinate)


def run(initial: Structure, final: Structure, potential,
        settings: Optional[NEBSettings] = None,
        progress: Optional[Callable[[float, str], Optional[bool]]] = None,
        cancelled: Optional[Callable[[], bool]] = None) -> NEBPath:
    """Optimise a minimum-energy path between two endpoint structures."""
    settings = (settings or NEBSettings()).validate()
    from .external_bias import has_external_bias

    if has_external_bias(initial) or has_external_bias(final):
        raise NEBRefused(
            "NEB endpoints carry external forces or restraints. Clear the biases or "
            "construct an explicit unbiased path; implicit path-wide bias semantics are "
            "not defined.")
    _check_endpoints(initial, final, settings)
    for endpoint, label in ((initial, "initial"), (final, "final")):
        supported, reason = potential.supports(endpoint)
        if not supported:
            raise NEBRefused(f"The force model does not support the {label} endpoint: {reason}")
    endpoint_values = []
    for endpoint in (initial, final):
        energy, forces = potential.energy_and_forces(endpoint.copy())
        forces = np.asarray(forces, dtype=float)
        free = ~np.asarray(endpoint.fixed, dtype=bool)
        fmax = float(np.linalg.norm(forces[free], axis=1).max()) if np.any(free) else 0.0
        endpoint_values.append((float(energy), forces, fmax))
    endpoint_fmax = [value[2] for value in endpoint_values]
    if max(endpoint_fmax) > settings.endpoint_fmax_eV_A and not settings.allow_unrelaxed_endpoints:
        raise NEBRefused(
            f"Endpoint maximum forces are {endpoint_fmax[0]:.4g} and "
            f"{endpoint_fmax[1]:.4g} eV/A, above the endpoint tolerance "
            f"{settings.endpoint_fmax_eV_A:g} eV/A. Relax both endpoints first or "
            "explicitly allow an estimated path.")
    if cancelled is not None and cancelled():
        raise NEBCancelled("NEB calculation cancelled before interpolation.")
    from ase.mep import NEB
    from ase.optimize import FIRE

    ase_images = [_ase_atoms(initial)]
    ase_images.extend(_ase_atoms(initial) for _ in range(settings.images - 2))
    ase_images.append(_ase_atoms(final))
    neb = NEB(
        ase_images, k=settings.spring_eV_A2, climb=settings.climb,
        remove_rotation_and_translation=settings.remove_rotation_translation,
        allow_shared_calculator=False)
    neb.interpolate(method=settings.interpolation, mic=any(initial.cell.pbc))
    for atoms in ase_images:
        atoms.calc = _calculator(potential, initial)
    history_step: List[int] = []
    history_barrier: List[float] = []
    history_force: List[float] = []
    optimizer = FIRE(
        neb, logfile=None, dt=settings.fire_dt, maxstep=settings.fire_maxstep_A)

    def observe() -> None:
        if cancelled is not None and cancelled():
            raise NEBCancelled("NEB calculation cancelled during optimisation.")
        energies = np.asarray([atoms.get_potential_energy() for atoms in ase_images])
        forces = np.asarray(neb.get_forces())
        maximum = float(np.linalg.norm(forces.reshape(-1, 3), axis=1).max())
        step = int(optimizer.nsteps)
        history_step.append(step)
        history_barrier.append(float(energies.max() - energies[0]))
        history_force.append(maximum)
        if progress is not None:
            fraction = min(1.0, step / settings.max_steps)
            if progress(fraction, f"NEB step {step}, max force {maximum:.4g} eV/A") is False:
                raise NEBCancelled("NEB calculation cancelled by the progress callback.")

    optimizer.attach(observe, interval=1)
    converged = bool(optimizer.run(fmax=settings.fmax_eV_A, steps=settings.max_steps))
    structures = []
    energies = []
    forces = []
    for atoms in ase_images:
        structure = initial.copy()
        structure.positions = np.asarray(atoms.positions, dtype=float)
        structure.info["neb_image"] = len(structures)
        structures.append(structure)
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), dtype=float))
    energies_array = np.asarray(energies)
    forces_array = np.asarray(forces)
    saddle = int(np.argmax(energies_array))
    forward = float(energies_array[saddle] - energies_array[0])
    reverse = float(energies_array[saddle] - energies_array[-1])
    reaction = float(energies_array[-1] - energies_array[0])
    final_neb_forces = np.asarray(neb.get_forces())
    residual = float(np.linalg.norm(final_neb_forces.reshape(-1, 3), axis=1).max())
    message = (
        f"NEB converged in {optimizer.nsteps} steps."
        if converged else
        f"NEB stopped after {optimizer.nsteps} steps with a maximum projected force of "
        f"{residual:.4g} eV/A.")
    convergence = Convergence(
        converged=converged, iterations=int(optimizer.nsteps), residual=residual,
        residual_metric="maximum projected NEB force", tolerance=settings.fmax_eV_A,
        message=message)
    described = potential.describe()
    approximations = list(described.get("approximations", [])) + [
        f"Nudged elastic band with {settings.images} discrete images and a "
        f"{settings.spring_eV_A2:g} eV/A^2 spring.",
        f"{settings.interpolation.upper()} endpoint interpolation.",
        "The path is a local result that depends on the endpoints and initial interpolation.",
    ]
    if settings.climb:
        approximations.append("The highest-energy image uses the climbing-image force.")
    if max(endpoint_fmax) > settings.endpoint_fmax_eV_A:
        approximations.append("One or both endpoints are not stationary; the barrier is estimated.")
    if not converged:
        approximations.append("The path did not meet the requested force tolerance.")
    inputs = digest({
        "initial_numbers": initial.numbers.tolist(),
        "initial_positions": np.round(initial.positions, 10).tolist(),
        "final_positions": np.round(final.positions, 10).tolist(),
        "cell": initial.cell.matrix.tolist(), "pbc": list(initial.cell.pbc),
        "fixed": initial.fixed.tolist(), "settings": settings.as_dict(),
        "model": potential.name, "parameters": described.get("parameters", {}),
    })
    origin = (Origin.CALCULATED if converged and
              max(endpoint_fmax) <= settings.endpoint_fmax_eV_A else Origin.ESTIMATED)
    provenance = Provenance(
        model=f"neb/ase-fire[{potential.model_label()}]",
        fidelity=potential.fidelity, origin=origin, approximations=approximations,
        tolerances={"path_fmax_eV_A": settings.fmax_eV_A,
                    "endpoint_fmax_eV_A": settings.endpoint_fmax_eV_A},
        boundary_conditions=f"pbc={initial.cell.pbc}, fixed cell",
        parameters={"settings": settings.as_dict(),
                    "force_model": described.get("parameters", {})},
        references=[NEB_REFERENCE, CLIMB_REFERENCE] + list(described.get("references", [])),
        inputs_digest=inputs)
    if not history_step or history_step[-1] != optimizer.nsteps:
        history_step.append(int(optimizer.nsteps))
        history_barrier.append(forward)
        history_force.append(residual)
    return NEBPath(
        images=structures, energies_eV=energies_array, forces_eV_A=forces_array,
        reaction_coordinate_A=_coordinate(structures),
        history_step=np.asarray(history_step, dtype=int),
        history_barrier_eV=np.asarray(history_barrier),
        history_max_force_eV_A=np.asarray(history_force),
        barrier_eV=max(forward, 0.0), forward_barrier_eV=forward,
        reverse_barrier_eV=reverse, reaction_energy_eV=reaction,
        saddle_image=saddle, endpoint_forces_eV_A=endpoint_fmax,
        settings=settings, provenance=provenance, convergence=convergence)
