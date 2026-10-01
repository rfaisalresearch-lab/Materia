"""Tier-1 classical solver: energy, forces, relaxation, dynamics and harmonic modes."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..core_model.units import BOLTZMANN_EV_K, U_A2_FS2_TO_EV
from ..physics.potentials import LennardJones, Potential, StillingerWeber, UnsupportedSystem
from ..provenance import Convergence, Fidelity, Origin, Provenance, Result
from .base import Capability, Solver, SolverResult, SupportReport


class ClassicalSolver(Solver):
    """Wraps a :class:`~materia.physics.potentials.Potential`.

    Capabilities are energy, forces, relaxation and dynamics.  Everything
    electronic is explicitly unsupported: a classical potential contains no
    electrons, so asking it for a density of states is a category error and is
    reported as one.
    """

    fidelity = Fidelity.TIER1_CLASSICAL
    capabilities = {Capability.ENERGY, Capability.FORCES,
                    Capability.RELAX, Capability.DYNAMICS, Capability.PHONONS}

    def __init__(self, potential: Potential) -> None:
        self.potential = potential
        self.name = f"classical/{potential.name}"
        self.description = (
            f"Classical potential {potential.name}. No electronic degrees of freedom."
        )

    def supports(self, structure: Structure) -> SupportReport:
        ok, why = self._potential(structure).supports(structure)
        warnings: List[str] = []
        if abs(structure.total_charge()) > 1e-9:
            warnings.append(
                "The structure carries a net charge, which a neutral classical "
                "potential ignores entirely. Energies and forces here are those of "
                "the neutral system."
            )
        return SupportReport(ok, [] if ok else [why], warnings)

    def _potential(self, structure: Structure) -> Potential:
        from ..physics.external_bias import potential_with_external_bias

        return potential_with_external_bias(self.potential, structure)

    def _provenance(self, structure: Optional[Structure] = None,
                    origin: Origin = Origin.CALCULATED, **extra) -> Provenance:
        potential = self.potential if structure is None else self._potential(structure)
        d = potential.describe()
        parameters = {**d.get("parameters", {}), **extra.pop("parameters", {})}
        if structure is not None:
            from ..physics.external_bias import external_biases, has_external_bias

            if has_external_bias(structure):
                parameters["external_biases"] = external_biases(structure)
        return Provenance(
            model=f"classical/{potential.name}",
            fidelity=self.fidelity,
            origin=origin,
            approximations=list(d.get("approximations", [])) + [
                "Classical point-nucleus dynamics; no electronic structure, no "
                "zero-point motion and no quantum statistics.",
            ],
            tolerances=extra.pop("tolerances", {}),
            boundary_conditions=extra.pop("boundary_conditions", ""),
            parameters=parameters,
            references=list(d.get("references", [])),
            **extra,
        )

    def _checks(self, structure: Structure, out: SolverResult) -> dict:
        checks = self._potential(structure).diagnostics(structure)
        if not checks:
            return {}
        if checks.get("ewald_converged") is False:
            out.log.append(f"Coulomb sum not confirmed: {checks.get('ewald_message', '')}")
        return {"potential_checks": checks}

    def run(self, structure: Structure, task: str = "energy", **kwargs) -> SolverResult:
        if task == "energy":
            return self.single_point(structure)
        if task == "relax":
            return self.relax(structure, **kwargs)
        if task in ("md", "dynamics"):
            return self.dynamics(structure, **kwargs)
        if task == "phonons":
            return self.phonons(structure, **kwargs)
        if task in ("phonon_dispersion", "phonon_dos"):
            return self.phonon_dispersion(structure, **kwargs)
        raise ValueError(
            f"Unknown task {task!r} for {self.name}. Supported: energy, relax, md, phonons, "
            "phonon_dispersion, phonon_dos."
        )

    def phonons(self, structure: Structure, settings=None, temperatures_K=None,
                progress: Optional[Callable[[float, str], Optional[bool]]] = None,
                cancelled: Optional[Callable[[], bool]] = None) -> SolverResult:
        """Gamma-point harmonic modes by finite displacements of this potential.

        See :mod:`materia.physics.phonons`.  A refusal or a cancellation returns
        an unsupported ``frequencies`` result that says why, and no numbers.
        """
        from ..physics import phonons as ph
        from ..provenance import unsupported

        t0 = time.perf_counter()
        out = SolverResult(solver=self.name, structure=structure)
        report = self.supports(structure)
        if not report.ok:
            out.results["frequencies"] = unsupported(
                "phonon_frequencies", self.name, "; ".join(report.blocking), unit="THz")
            out.log.extend(report.blocking)
            return out
        try:
            analysis = ph.harmonic_analysis(structure, self._potential(structure), settings,
                                            name=self.name, progress=progress,
                                            cancelled=cancelled)
        except (ph.PhononRefused, ph.PhononCancelled) as exc:
            out.results["frequencies"] = unsupported(
                "phonon_frequencies", self.name, str(exc), unit="THz")
            out.results["frequencies"].extra["status"] = (
                "cancelled" if isinstance(exc, ph.PhononCancelled) else "refused")
            out.log.append(str(exc))
            out.wall_time_s = time.perf_counter() - t0
            return out
        out.results.update(analysis.results())
        if temperatures_K is not None:
            out.results.update(analysis.thermodynamics_results(temperatures_K))
        out.convergence = analysis.convergence
        out.log.extend(report.warnings)
        note = analysis.diagnostics.get("self_image_note")
        if note:
            out.log.append(note)
        if analysis.imaginary_modes():
            out.log.append(analysis.zero_point_note)
        out.wall_time_s = time.perf_counter() - t0
        return out

    def phonon_dispersion(self, structure: Structure, settings,
                          progress: Optional[Callable[[float, str], Optional[bool]]] = None,
                          cancelled: Optional[Callable[[], bool]] = None) -> SolverResult:
        """Periodic real-space force constants, dispersion and vibrational DOS."""
        from ..physics import phonon_dispersion as pd
        from ..provenance import unsupported

        t0 = time.perf_counter()
        out = SolverResult(solver=self.name, structure=structure)
        report = self.supports(structure)
        if not report.ok:
            reason = "; ".join(report.blocking)
            out.results["phonon_dispersion"] = unsupported(
                "phonon_dispersion", self.name, reason, unit="THz")
            out.log.extend(report.blocking)
            return out
        try:
            analysis = pd.periodic_phonon_analysis(
                structure, self.potential, settings, name=self.name,
                progress=progress, cancelled=cancelled)
        except (pd.PhononRefused, pd.PhononCancelled) as exc:
            out.results["phonon_dispersion"] = unsupported(
                "phonon_dispersion", self.name, str(exc), unit="THz")
            out.results["phonon_dispersion"].extra["status"] = (
                "cancelled" if isinstance(exc, pd.PhononCancelled) else "refused")
            out.log.append(str(exc))
            out.wall_time_s = time.perf_counter() - t0
            return out
        out.results.update(analysis.results())
        out.convergence = analysis.convergence
        out.log.extend(report.warnings)
        if analysis.diagnostics["imaginary_path_modes"]:
            out.log.append(
                f"{analysis.diagnostics['imaginary_path_modes']} path modes are imaginary; "
                "negative frequencies carry their magnitudes.")
        out.log.append(
            "No LO-TO correction is applied; Born effective charges and dielectric tensors "
            "are not computed.")
        out.wall_time_s = time.perf_counter() - t0
        return out

    def single_point(self, structure: Structure) -> SolverResult:
        t0 = time.perf_counter()
        report = self.supports(structure)
        out = SolverResult(solver=self.name, structure=structure)
        if not report.ok:
            from ..provenance import unsupported
            out.results["energy"] = unsupported(
                "energy", self.name, "; ".join(report.blocking),
                suggested_models=["external:lammps", "external:gpaw"], unit="eV")
            out.log.extend(report.blocking)
            return out
        energy, forces = self._potential(structure).energy_and_forces(structure)
        structure.forces = forces
        bc = "3D periodic" if all(structure.cell.pbc) else f"pbc={structure.cell.pbc}"
        prov = self._provenance(structure, boundary_conditions=bc)
        out.results["energy"] = Result(
            "energy", float(energy), "eV", prov,
            extra={"energy_per_atom_eV": float(energy) / max(1, len(structure)),
                   **self._checks(structure, out)},
        )
        out.results["forces"] = Result(
            "forces", forces, "eV/A", prov,
            extra={"max_force_eV_A": float(np.abs(forces).max()) if len(forces) else 0.0},
        )
        out.log.extend(report.warnings)
        out.wall_time_s = time.perf_counter() - t0
        return out

    def relax(
        self,
        structure: Structure,
        fmax_eV_A: float = 0.02,
        max_steps: int = 500,
        dt_fs: float = 1.0,
        dt_max_fs: float = 5.0,
        in_place: bool = False,
        callback: Optional[Callable[[int, float, float], bool]] = None,
    ) -> SolverResult:
        """FIRE geometry optimisation.

        Implements the Fast Inertial Relaxation Engine of Bitzek et al.,
        Phys. Rev. Lett. 97 (2006) 170201, with the standard parameter set.
        Atoms flagged ``fixed`` are held.
        """
        t0 = time.perf_counter()
        report = self.supports(structure)
        work = structure if in_place else structure.copy()
        initial_positions = structure.positions.copy()
        out = SolverResult(solver=self.name, structure=work)
        if not report.ok:
            from ..provenance import unsupported
            out.results["relaxed_structure"] = unsupported(
                "relaxed_structure", self.name, "; ".join(report.blocking),
                suggested_models=["external:lammps", "external:gpaw"])
            out.log.extend(report.blocking)
            return out

        n_min, f_inc, f_dec = 5, 1.1, 0.5
        alpha_start, f_alpha = 0.1, 0.99
        alpha = alpha_start
        dt = dt_fs
        steps_since_negative = 0

        masses = work.masses()[:, None]
        free = ~work.fixed
        v = np.zeros_like(work.positions)
        potential = self._potential(work)
        energy, forces = potential.energy_and_forces(work)
        e_initial = float(energy)
        history: List[Tuple[int, float, float]] = []
        converged = False
        step = 0
        fmax = float(np.linalg.norm(forces[free], axis=1).max()) if free.any() else 0.0

        for step in range(1, max_steps + 1):
            f = np.where(free[:, None], forces, 0.0)
            power = float((f * v).sum())
            fnorm = np.linalg.norm(f)
            vnorm = np.linalg.norm(v)
            if fnorm > 0:
                v = (1.0 - alpha) * v + alpha * (f / fnorm) * vnorm
            if power > 0:
                steps_since_negative += 1
                if steps_since_negative > n_min:
                    dt = min(dt * f_inc, dt_max_fs)
                    alpha *= f_alpha
            else:
                v[:] = 0.0
                dt *= f_dec
                alpha = alpha_start
                steps_since_negative = 0

            accel = f / (masses * U_A2_FS2_TO_EV)
            v = v + accel * dt
            v = np.where(free[:, None], v, 0.0)
            work.positions = work.positions + v * dt
            energy, forces = potential.energy_and_forces(work)
            fmax = float(np.linalg.norm(forces[free], axis=1).max()) if free.any() else 0.0
            history.append((step, float(energy), fmax))
            if callback is not None and callback(step, float(energy), fmax) is False:
                out.log.append(f"Cancelled by caller at step {step}.")
                break
            if fmax < fmax_eV_A:
                converged = True
                break

        work.forces = forces
        conv = Convergence(
            converged=converged,
            iterations=step,
            residual=fmax,
            residual_metric="max |F| on free atoms",
            tolerance=fmax_eV_A,
            message=("Converged." if converged else
                     f"Not converged after {step} steps; max force {fmax:.4f} eV/A "
                     f"exceeds the tolerance {fmax_eV_A} eV/A."),
        )
        prov = self._provenance(
            work,
            tolerances={"fmax_eV_A": fmax_eV_A, "max_steps": max_steps},
            boundary_conditions=("3D periodic" if all(work.cell.pbc)
                                 else f"pbc={work.cell.pbc}"),
            parameters={"minimiser": "FIRE", "dt_fs": dt_fs, "dt_max_fs": dt_max_fs,
                        "n_fixed_atoms": int((~free).sum())},
        )
        prov.references.append("E. Bitzek et al., Phys. Rev. Lett. 97 (2006) 170201")
        prov.approximations.append(
            "Local minimisation only: the result is the nearest local minimum, "
            "not a global one, and the cell is held fixed."
        )
        out.results["energy"] = Result("energy", float(energy), "eV", prov, convergence=conv,
                                       extra={"initial_energy_eV": e_initial,
                                              "energy_change_eV": float(energy) - e_initial,
                                              **self._checks(work, out)})
        out.results["forces"] = Result("forces", forces, "eV/A", prov, convergence=conv)
        out.results["relaxed_structure"] = Result(
            "relaxed_structure", work, "", prov, convergence=conv,
            extra={"max_displacement_A": float(
                np.linalg.norm(work.positions - initial_positions, axis=1).max())
                if len(work) == len(initial_positions) else None,
                "rms_displacement_A": float(np.sqrt(
                    (np.linalg.norm(work.positions - initial_positions, axis=1) ** 2).mean()))
                if len(work) == len(initial_positions) else None},
        )
        out.results["relaxation_history"] = Result(
            "relaxation_history",
            {"step": [h[0] for h in history], "energy_eV": [h[1] for h in history],
             "fmax_eV_A": [h[2] for h in history]},
            "mixed", prov, convergence=conv,
        )
        out.convergence = conv
        out.log.extend(report.warnings)
        out.wall_time_s = time.perf_counter() - t0
        return out

    def dynamics(
        self,
        structure: Structure,
        steps: int = 100,
        dt_fs: float = 1.0,
        temperature_K: Optional[float] = None,
        thermostat: str = "none",
        friction_per_fs: float = 0.01,
        seed: int = 0,
        initialise_velocities: bool = True,
        in_place: bool = False,
        sample_every: int = 1,
        callback: Optional[Callable[[int, float, float], bool]] = None,
    ) -> SolverResult:
        """Velocity-Verlet MD, optionally with a BAOAB Langevin thermostat.

        The RNG is seeded explicitly so that a run is bit-reproducible from the
        project file.  ``thermostat='none'`` integrates the microcanonical (NVE)
        equations; ``'langevin'`` samples the canonical ensemble.
        """
        t0 = time.perf_counter()
        report = self.supports(structure)
        work = structure if in_place else structure.copy()
        out = SolverResult(solver=self.name, structure=work)
        if not report.ok:
            from ..provenance import unsupported
            out.results["trajectory"] = unsupported(
                "trajectory", self.name, "; ".join(report.blocking))
            out.log.extend(report.blocking)
            return out
        if thermostat not in ("none", "langevin"):
            raise ValueError(
                f"Unknown thermostat {thermostat!r}. Implemented: 'none' (NVE), "
                "'langevin' (BAOAB). Nose-Hoover and Berendsen are not implemented."
            )
        if thermostat == "langevin" and temperature_K is None:
            raise ValueError("thermostat='langevin' requires temperature_K")

        rng = np.random.default_rng(seed)
        masses = work.masses()[:, None]
        free = ~work.fixed
        n_free = int(free.sum())
        dof = max(1, 3 * n_free - 3)

        if initialise_velocities and temperature_K:
            sigma = np.sqrt(BOLTZMANN_EV_K * temperature_K / (masses * U_A2_FS2_TO_EV))
            v = rng.normal(0.0, 1.0, work.positions.shape) * sigma
            v = np.where(free[:, None], v, 0.0)
            v -= (v * masses).sum(axis=0) / masses.sum()
            v = np.where(free[:, None], v, 0.0)
        else:
            v = work.velocities.copy()

        potential = self._potential(work)
        energy, forces = potential.energy_and_forces(work)
        kt = BOLTZMANN_EV_K * (temperature_K or 0.0)
        c1 = np.exp(-friction_per_fs * dt_fs)
        c2 = np.sqrt(1.0 - c1 * c1)

        traj = {"step": [], "time_fs": [], "potential_eV": [], "kinetic_eV": [],
                "total_eV": [], "temperature_K": [], "positions_A": [],
                "velocities_A_fs": []}

        def kinetic(vel):
            return 0.5 * float((masses * vel * vel).sum()) * U_A2_FS2_TO_EV

        for step in range(1, steps + 1):
            a = np.where(free[:, None], forces / (masses * U_A2_FS2_TO_EV), 0.0)
            v = v + 0.5 * dt_fs * a
            work.positions = work.positions + 0.5 * dt_fs * v
            if thermostat == "langevin":
                noise = rng.normal(0.0, 1.0, v.shape)
                v = c1 * v + c2 * np.sqrt(kt / (masses * U_A2_FS2_TO_EV)) * noise
                v = np.where(free[:, None], v, 0.0)
            work.positions = work.positions + 0.5 * dt_fs * v
            energy, forces = potential.energy_and_forces(work)
            a = np.where(free[:, None], forces / (masses * U_A2_FS2_TO_EV), 0.0)
            v = v + 0.5 * dt_fs * a
            v = np.where(free[:, None], v, 0.0)

            if step % sample_every == 0 or step == steps:
                ke = kinetic(v)
                temp = 2.0 * ke / (dof * BOLTZMANN_EV_K)
                traj["step"].append(step)
                traj["time_fs"].append(step * dt_fs)
                traj["potential_eV"].append(float(energy))
                traj["kinetic_eV"].append(ke)
                traj["total_eV"].append(float(energy) + ke)
                traj["temperature_K"].append(temp)
                traj["positions_A"].append(work.positions.copy())
                traj["velocities_A_fs"].append(v.copy())
                if callback is not None and callback(step, float(energy), temp) is False:
                    out.log.append(f"Cancelled by caller at step {step}.")
                    break

        work.velocities = v
        work.forces = forces
        prov = self._provenance(
            work,
            boundary_conditions=("3D periodic" if all(work.cell.pbc)
                                 else f"pbc={work.cell.pbc}"),
            parameters={"integrator": "velocity-Verlet (BAOAB splitting with Langevin)",
                        "steps": steps, "dt_fs": dt_fs, "thermostat": thermostat,
                        "temperature_K": temperature_K,
                        "friction_per_fs": friction_per_fs},
            seed=seed,
        )
        prov.approximations.append(
            "Classical nuclei: no zero-point energy and no quantum heat capacity. "
            "Below the Debye temperature the classical equipartition temperature is "
            "not the thermodynamic temperature of the real solid."
        )
        if thermostat == "langevin":
            prov.references.append(
                "B. Leimkuhler and C. Matthews, Appl. Math. Res. Express 2013 (2013) 34 (BAOAB)"
            )
        drift = (abs(traj["total_eV"][-1] - traj["total_eV"][0])
                 if len(traj["total_eV"]) > 1 else 0.0)
        conv = Convergence(
            converged=True, iterations=len(traj["step"]),
            residual=drift, residual_metric="|E_total(end) - E_total(start)|",
            message=("Energy drift is a diagnostic, not a convergence criterion."
                     if thermostat == "none" else
                     "Langevin dynamics does not conserve energy by construction."),
        )
        out.results["trajectory"] = Result("trajectory", traj, "mixed", prov, convergence=conv)
        out.results["energy"] = Result("energy", float(energy), "eV", prov,
                                       extra=self._checks(work, out))
        out.results["final_structure"] = Result("final_structure", work, "", prov)
        if traj["temperature_K"]:
            mean_t = float(np.mean(traj["temperature_K"][len(traj["temperature_K"]) // 2:]))
            out.results["mean_temperature"] = Result(
                "mean_temperature", mean_t, "K", prov,
                uncertainty=float(np.std(traj["temperature_K"][len(traj["temperature_K"]) // 2:])),
                uncertainty_kind="stddev",
                extra={"note": "Mean over the second half of the run."},
            )
        out.convergence = conv
        out.log.extend(report.warnings)
        out.wall_time_s = time.perf_counter() - t0
        return out


def stillinger_weber(element: str = "Si") -> ClassicalSolver:
    return ClassicalSolver(StillingerWeber(element))


def lennard_jones(material=None, params: Optional[Dict[str, Tuple[float, float]]] = None,
                  cutoff_A: float = 10.0) -> ClassicalSolver:
    if params is not None:
        return ClassicalSolver(LennardJones(params, cutoff_A=cutoff_A))
    if material is None:
        raise ValueError("lennard_jones() needs either a material or explicit params")
    return ClassicalSolver(LennardJones.from_material(material, cutoff_A=cutoff_A))
