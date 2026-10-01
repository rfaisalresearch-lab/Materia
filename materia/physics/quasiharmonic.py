"""Quasi-harmonic thermodynamics of a crystal: thermal expansion and free energy.

The Helmholtz free energy of a crystal at volume ``V`` and temperature ``T`` is
taken as the static energy plus the free energy of independent harmonic
phonons at that volume:

``F(V, T) = E(V) + sum_qj [h nu_qj / 2 + kT ln(1 - exp(-h nu_qj / kT))] / N_q``

over a uniform q mesh.  At each temperature ``F(V, T) + P V`` is fitted with
the third-order Birch-Murnaghan form; its minimum gives the equilibrium volume
``V(T)``, the isothermal bulk modulus ``B_T(T)`` and the Gibbs energy ``G(T)``.
The volumetric thermal expansion is ``alpha = d ln V / dT`` by central
differences over the temperature grid (one-sided at its ends, and exactly zero
at 0 K by the third law), ``C_P = C_V + alpha^2 B_T V T`` and the
thermodynamic Gruneisen parameter is ``alpha B_T V / C_V``.

Phonons come from :func:`~materia.physics.phonon_dispersion.periodic_phonon_analysis`
at every volume, so any Materia potential (native, LAMMPS, ML, QE) can be used.
The cell is scaled isotropically; internal coordinates are relaxed at fixed
cell when requested.  Nothing is returned if a volume has imaginary modes away
from the acoustic modes at Gamma, or if the free-energy minimum at any
requested temperature leaves the sampled volume range.

The approximation neglects intrinsic anharmonicity (phonon-phonon
interaction at fixed volume), which grows with temperature; above roughly half
the melting point quasi-harmonic expansion is usually too small or too large
depending on the material, and molecular dynamics is the better tool.

References
----------
A. A. Maradudin, E. W. Montroll and G. H. Weiss, Theory of Lattice Dynamics in
the Harmonic Approximation (Academic Press, 1963).
S. Baroni, P. Giannozzi and E. Isaev, Rev. Mineral. Geochem. 71 (2010) 39.
A. Togo and I. Tanaka, Scr. Mater. 108 (2015) 1.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..core_model.units import BOLTZMANN_EV_K, ELEMENTARY_CHARGE_C, PLANCK_J_S
from ..provenance import Origin, Provenance, Result

PLANCK_EV_S = PLANCK_J_S / ELEMENTARY_CHARGE_C
EV_A3_GPA = 160.21766208
GAMMA_ACOUSTIC_THz = 0.05
UNSTABLE_THz = 0.05


class QuasiHarmonicError(ValueError):
    pass


def scaled(structure: Structure, volume_scale: float) -> Structure:
    factor = float(volume_scale) ** (1.0 / 3.0)
    out = structure.copy()
    out.cell = Cell(np.asarray(structure.cell.matrix, dtype=float) * factor, structure.cell.pbc)
    out.positions = np.asarray(structure.positions, dtype=float) * factor
    return out


def harmonic_free_energy(frequencies_THz: np.ndarray, weights: np.ndarray,
                         temperature_K: float) -> Tuple[float, float, float]:
    """Free energy (eV), entropy (eV/K) and heat capacity (eV/K) per cell."""
    nu = np.asarray(frequencies_THz, dtype=float) * 1e12
    w = np.asarray(weights, dtype=float)
    zero_point = 0.5 * PLANCK_EV_S * float(np.sum(w * nu))
    if temperature_K <= 0:
        return zero_point, 0.0, 0.0
    kt = BOLTZMANN_EV_K * temperature_K
    x = PLANCK_EV_S * nu / kt
    with np.errstate(over="ignore"):
        expm1 = np.expm1(x)
    log_term = np.log1p(-np.exp(-x))
    free = zero_point + kt * float(np.sum(w * log_term))
    entropy = BOLTZMANN_EV_K * float(np.sum(w * (x / expm1 - log_term)))
    with np.errstate(over="ignore", invalid="ignore"):
        cv_mode = np.where(x < 700, x * x * np.exp(-x) / (-np.expm1(-x)) ** 2, 0.0)
    heat_capacity = BOLTZMANN_EV_K * float(np.sum(w * cv_mode))
    return free, entropy, heat_capacity


def _modes(phonons) -> Tuple[np.ndarray, np.ndarray]:
    frequencies = np.asarray(phonons.dos_raw_frequencies_THz, dtype=float)
    weights = np.asarray(phonons.dos_raw_weights, dtype=float)
    q = np.asarray(phonons.dos_q_points_scaled, dtype=float)
    at_gamma = np.all(np.abs(q - np.round(q)) < 1e-9, axis=1)
    keep = np.ones_like(frequencies, dtype=bool)
    for i in np.flatnonzero(at_gamma):
        order = np.argsort(np.abs(frequencies[i]))[:3]
        if np.any(np.abs(frequencies[i, order]) > GAMMA_ACOUSTIC_THz):
            raise QuasiHarmonicError("The three acoustic modes at Gamma are not near zero; the "
                                     "acoustic sum rule is violated.")
        keep[i, order] = False
    if np.any(frequencies[keep] < -UNSTABLE_THz):
        raise QuasiHarmonicError(f"Imaginary modes down to {frequencies[keep].min():.3f} THz: the "
                                 "crystal is dynamically unstable at this volume, and harmonic "
                                 "free energy is undefined.")
    values = np.abs(frequencies[keep])
    if np.any(values < 1e-6):
        raise QuasiHarmonicError("A zero-frequency mode away from Gamma makes the free energy "
                                 "undefined; use a mesh without accidental zeros or a larger "
                                 "supercell.")
    return values, weights[keep]


def quasiharmonic(structure: Structure, potential: Any,
                  volume_scales: Sequence[float], temperatures_K: Sequence[float],
                  supercell: Tuple[int, int, int], dos_mesh: Tuple[int, int, int],
                  pressure_GPa: float = 0.0, relax_internal: bool = False,
                  fmax_eV_A: float = 1e-3, displacement_A: float = 0.01) -> Result:
    """Equilibrium volume, expansion, bulk modulus, heat capacity and Gibbs energy against T."""
    from ..experiments.dft.eos import fit_birch_murnaghan
    from .phonon_dispersion import PeriodicPhononSettings, periodic_phonon_analysis

    started = time.perf_counter()
    scales = np.array(sorted(float(s) for s in volume_scales))
    temperatures = np.array([float(t) for t in temperatures_K])
    if not all(structure.cell.pbc):
        raise QuasiHarmonicError("Quasi-harmonic thermodynamics needs a crystal periodic in "
                                 "all three directions.")
    if len(scales) < 5:
        raise QuasiHarmonicError("At least five volumes are needed for an equation-of-state fit.")
    if scales.min() <= 0.7 or scales.max() >= 1.3:
        raise QuasiHarmonicError("Volume scales must lie between 0.7 and 1.3.")
    if len(temperatures) < 3 or np.any(np.diff(temperatures) <= 0) or temperatures[0] < 0:
        raise QuasiHarmonicError("Give at least three increasing, non-negative temperatures so "
                                 "that the expansion coefficient can be differenced.")
    pressure = float(pressure_GPa) / EV_A3_GPA
    volumes, energies, mode_sets, residuals = [], [], [], []
    for s in scales:
        cell = scaled(structure, s)
        if relax_internal:
            from ..solvers.classical import ClassicalSolver

            out = ClassicalSolver(potential).relax(cell, fmax_eV_A=fmax_eV_A, max_steps=5000)
            if not out.convergence.converged:
                raise QuasiHarmonicError(f"Internal relaxation at volume scale {s:g} did not "
                                         "converge.")
            cell = out.structure
        energy, forces = potential.energy_and_forces(cell)
        residual = float(np.abs(np.asarray(forces)).max()) if len(cell) else 0.0
        if residual > 5 * fmax_eV_A:
            raise QuasiHarmonicError(f"Largest force {residual:.3g} eV/A at volume scale {s:g}: "
                                     "the atoms are not at equilibrium. Set relax_internal.")
        settings = PeriodicPhononSettings(supercell=tuple(supercell), q_path=((0, 0, 0),),
                                          dos_mesh=tuple(dos_mesh),
                                          displacement_A=displacement_A,
                                          max_residual_force_eV_A=5 * fmax_eV_A)
        phonons = periodic_phonon_analysis(cell, potential, settings)
        try:
            mode_sets.append(_modes(phonons))
        except QuasiHarmonicError as exc:
            raise QuasiHarmonicError(f"Volume scale {s:g}: {exc}") from exc
        volumes.append(cell.cell.volume)
        energies.append(float(energy))
        residuals.append(residual)
    volumes = np.array(volumes)
    energies = np.array(energies)
    static = fit_birch_murnaghan(volumes, energies + pressure * volumes)
    rows = []
    for t in temperatures:
        thermal = np.array([harmonic_free_energy(f, w, t) for f, w in mode_sets])
        total = energies + thermal[:, 0] + pressure * volumes
        try:
            fit = fit_birch_murnaghan(volumes, total)
        except (RuntimeError, ValueError) as exc:
            raise QuasiHarmonicError(f"The free-energy fit at {t:g} K did not converge: "
                                     f"{exc}.") from exc
        v0 = fit["V0_A3"]
        if not volumes.min() < v0 < volumes.max():
            raise QuasiHarmonicError(f"At {t:g} K the free-energy minimum ({v0:.4f} A^3) lies "
                                     f"outside the sampled volumes ({volumes.min():.4f} to "
                                     f"{volumes.max():.4f} A^3); sample larger volumes or lower "
                                     "the temperature.")
        if not (fit["B0_eV_A3"] > 0 and 1 < fit["B1"] < 12):
            raise QuasiHarmonicError(f"The free-energy fit at {t:g} K is unphysical.")
        coefficients = np.polyfit(volumes, thermal[:, 2], 2)
        cv = float(np.polyval(coefficients, v0))
        s_coefficients = np.polyfit(volumes, thermal[:, 1], 2)
        rows.append({"temperature_K": float(t), "volume_A3": v0,
                     "bulk_modulus_GPa": fit["B0_eV_A3"] * EV_A3_GPA, "B1": fit["B1"],
                     "gibbs_energy_eV": fit["E0_eV"],
                     "entropy_eV_K": float(np.polyval(s_coefficients, v0)),
                     "heat_capacity_v_eV_K": cv, "fit_rms_eV": fit["rms_eV"]})
    v = np.array([r["volume_A3"] for r in rows])
    alpha = np.gradient(np.log(v), temperatures, edge_order=1)
    alpha[temperatures == 0] = 0.0
    for r, a in zip(rows, alpha):
        b = r["bulk_modulus_GPa"] / EV_A3_GPA
        r["volumetric_expansion_per_K"] = float(a)
        r["linear_expansion_per_K"] = float(a) / 3.0
        r["heat_capacity_p_eV_K"] = r["heat_capacity_v_eV_K"] + float(a) ** 2 * b * \
            r["volume_A3"] * r["temperature_K"]
        r["gruneisen"] = (float(a) * b * r["volume_A3"] / r["heat_capacity_v_eV_K"]
                          if r["heat_capacity_v_eV_K"] > 0 else None)
    label = potential.model_label() if hasattr(potential, "model_label") else potential.name
    describe = potential.describe() if hasattr(potential, "describe") else {}
    prov = Provenance(
        model=f"quasiharmonic[{label}]", fidelity=getattr(potential, "fidelity", None),
        origin=Origin.CALCULATED,
        approximations=[
            "Quasi-harmonic approximation: harmonic phonons at each volume; intrinsic "
            "anharmonicity at fixed volume is neglected and grows with temperature.",
            "Isotropic volume scaling" + (" with internal coordinates relaxed at each volume."
                                          if relax_internal else "; internal coordinates "
                                          "fixed in fractional form."),
            f"Third-order Birch-Murnaghan fits of F + PV at each temperature; phonon free "
            f"energy summed over a {dos_mesh[0]}x{dos_mesh[1]}x{dos_mesh[2]} q mesh from a "
            f"{supercell[0]}x{supercell[1]}x{supercell[2]} finite-displacement supercell.",
            "Thermal expansion by central differences over the given temperatures; C_V and S "
            "interpolated quadratically in volume to V(T).",
            "No electronic free energy: valid for insulators, or metals where the electronic "
            "entropy is negligible at these temperatures.",
        ],
        parameters={"volume_scales": scales.tolist(), "temperatures_K": temperatures.tolist(),
                    "supercell": list(supercell), "dos_mesh": list(dos_mesh),
                    "pressure_GPa": float(pressure_GPa), "relax_internal": relax_internal,
                    "displacement_A": displacement_A,
                    "potential": describe.get("parameters", {})},
        references=["S. Baroni, P. Giannozzi and E. Isaev, Rev. Mineral. Geochem. 71 (2010) 39",
                    "A. Togo and I. Tanaka, Scr. Mater. 108 (2015) 1"]
        + list(describe.get("references", [])))
    return Result("quasiharmonic_thermodynamics", rows, "mixed", prov,
                  extra={"volumes_A3": volumes.tolist(), "static_energies_eV": energies.tolist(),
                         "static_fit": {"V0_A3": static["V0_A3"],
                                        "B0_GPa": static["B0_eV_A3"] * EV_A3_GPA,
                                        "B1": static["B1"]},
                         "largest_residual_force_eV_A": max(residuals),
                         "atoms_per_cell": len(structure),
                         "wall_time_s": time.perf_counter() - started})
