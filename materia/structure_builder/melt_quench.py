"""Amorphous structures by melt and quench molecular dynamics.

A crystal is melted with Langevin dynamics, held, cooled in steps at a stated
rate, held at the final temperature and relaxed to the nearest local minimum,
all at fixed volume with the chosen potential.  The result depends on the
potential, the cooling rate, the volume and the seed, and the record carries
all of them.  Cooling rates reachable in molecular dynamics (about 1e12 to
1e14 K/s) are many orders of magnitude faster than any laboratory quench, so
the glass is less relaxed than a real one.

The structure is characterised by its radial distribution function,
coordination numbers within a cutoff, the bond-angle distribution, and a
crystalline order parameter: ``|S(G)|^2 / N^2`` at the strongest reflection of
the starting crystal, relative to its value in that crystal (1 for the
crystal, of order ``1/N`` without long-range order).

References
----------
F. Wooten, K. Winer and D. Weaire, Phys. Rev. Lett. 54 (1985) 1392 (context).
W. D. Luedtke and U. Landman, Phys. Rev. B 40 (1989) 1164 (Stillinger-Weber a-Si).
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core_model.structure import Structure
from ..provenance import Fidelity, Origin, Provenance, Result


class MeltQuenchError(ValueError):
    pass


def structure_factor_order(structure: Structure, reference: Structure,
                           reciprocal_vectors: Sequence[Sequence[float]]) -> float:
    """``|S(G)|^2 / N^2`` averaged over ``G``, relative to the same quantity in ``reference``."""
    def order(s: Structure) -> float:
        n = len(s)
        values = [abs(np.exp(1j * np.asarray(s.positions) @ np.asarray(g)).sum()) ** 2 / n ** 2
                  for g in reciprocal_vectors]
        return float(np.mean(values))
    base = order(reference)
    if base <= 0:
        raise MeltQuenchError("The reference crystal has no intensity at the given vectors.")
    return order(structure) / base


def analyse(structure: Structure, bond_cutoff_A: float, r_max_A: float = 8.0,
            bins: int = 160) -> Dict[str, object]:
    """Radial distribution, coordination and bond angles of a periodic structure."""
    from ..physics.neighbors import neighbor_list

    nl = neighbor_list(structure.positions, structure.cell, r_max_A)
    n = len(structure)
    volume = structure.cell.volume
    edges = np.linspace(0.0, r_max_A, bins + 1)
    counts, _ = np.histogram(nl.d, bins=edges)
    shell = 4.0 / 3.0 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = counts / (n * shell * n / volume)
    centres = 0.5 * (edges[1:] + edges[:-1])
    bonded = nl.d < bond_cutoff_A
    coordination = np.bincount(nl.i[bonded], minlength=n)
    angles: List[float] = []
    for atom in range(n):
        vectors = nl.D[(nl.i == atom) & bonded]
        for p in range(len(vectors)):
            for q in range(p + 1, len(vectors)):
                c = vectors[p] @ vectors[q] / (np.linalg.norm(vectors[p]) *
                                               np.linalg.norm(vectors[q]))
                angles.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    angles_arr = np.array(angles)
    inside = centres < bond_cutoff_A
    first_peak = float(centres[inside][np.argmax(g[inside])]) if inside.any() else None
    values, frequency = np.unique(coordination, return_counts=True)
    return {"r_A": centres.tolist(), "g_r": g.tolist(), "first_peak_A": first_peak,
            "bond_cutoff_A": bond_cutoff_A,
            "mean_coordination": float(coordination.mean()),
            "coordination_histogram": {str(int(v)): int(f) for v, f in zip(values, frequency)},
            "bond_angle_mean_deg": float(angles_arr.mean()) if angles else None,
            "bond_angle_rms_deg": float(angles_arr.std()) if angles else None,
            "density_g_cm3": float(structure.masses().sum() / volume * 1.66053906660)}


def melt_quench(potential, crystal: Structure, melt_K: float, final_K: float = 300.0,
                melt_ps: float = 20.0, cooling_K_per_ps: float = 30.0, hold_ps: float = 10.0,
                dt_fs: float = 1.0, friction_per_fs: float = 0.01, seed: int = 0,
                step_K: float = 100.0, bond_cutoff_A: Optional[float] = None,
                reciprocal_vectors: Optional[Sequence[Sequence[float]]] = None,
                max_melt_order: float = 0.1,
                progress: Optional[Callable[[float, str], Optional[bool]]] = None) -> Result:
    """Melt, quench and relax a crystal at fixed volume; return the glass and its analysis."""
    from ..solvers.classical import ClassicalSolver

    if not all(crystal.cell.pbc):
        raise MeltQuenchError("Melt and quench needs a three-dimensional periodic cell.")
    if not (melt_K > final_K > 0 and cooling_K_per_ps > 0 and melt_ps > 0 and hold_ps >= 0):
        raise MeltQuenchError("Need melt_K > final_K > 0 and positive times and rates.")
    ok, why = potential.supports(crystal)
    if not ok:
        raise MeltQuenchError(why)
    t0 = time.perf_counter()
    solver = ClassicalSolver(potential)
    work = crystal.copy()
    temperatures = [melt_K]
    t = melt_K
    while t - step_K > final_K:
        t -= step_K
        temperatures.append(t)
    temperatures.append(final_K)
    schedule = [(melt_K, melt_ps)] + [(temp, step_K / cooling_K_per_ps)
                                      for temp in temperatures[1:-1]] + [(final_K, hold_ps)]
    total_steps = sum(int(round(ps * 1000 / dt_fs)) for _, ps in schedule)
    done = 0
    trace = {"temperature_K": [], "mean_temperature_K": [], "potential_eV_per_atom": []}
    for index, (temperature, duration) in enumerate(schedule):
        steps = max(1, int(round(duration * 1000 / dt_fs)))
        out = solver.dynamics(work, steps=steps, dt_fs=dt_fs, temperature_K=temperature,
                              thermostat="langevin", friction_per_fs=friction_per_fs,
                              seed=seed + index, initialise_velocities=(index == 0),
                              in_place=True, sample_every=max(1, steps // 10))
        trajectory = out.results["trajectory"].value
        trace["temperature_K"].append(temperature)
        trace["mean_temperature_K"].append(float(np.mean(trajectory["temperature_K"])))
        trace["potential_eV_per_atom"].append(float(trajectory["potential_eV"][-1]) / len(work))
        done += steps
        if index == 0 and reciprocal_vectors is not None:
            melted_order = structure_factor_order(work, crystal, reciprocal_vectors)
            trace["order_after_melt"] = melted_order
            if melted_order > max_melt_order:
                raise MeltQuenchError(
                    f"The crystal did not melt: after {melt_ps:g} ps at {melt_K:g} K its "
                    f"crystalline order is {melted_order:.3f} (limit {max_melt_order:g}). "
                    "Raise melt_K or melt_ps; quenching a solid would not give a glass.")
        if progress is not None and progress(done / total_steps,
                                             f"{temperature:.0f} K, {done} of {total_steps}"
                                             ) is False:
            raise MeltQuenchError("Melt and quench cancelled.")
    relaxed = solver.relax(work, fmax_eV_A=1e-3, max_steps=20000)
    glass = relaxed.structure
    crystal_energy = float(potential.energy_and_forces(crystal)[0]) / len(crystal)
    glass_energy = float(relaxed.results["energy"].value) / len(glass)
    cutoff = bond_cutoff_A
    if cutoff is None:
        from ..physics.neighbors import neighbor_list
        nl = neighbor_list(crystal.positions, crystal.cell, 6.0)
        cutoff = float(np.min(nl.d)) * 1.2
    analysis = analyse(glass, cutoff)
    if reciprocal_vectors is not None:
        analysis["crystalline_order"] = structure_factor_order(glass, crystal, reciprocal_vectors)
    label = getattr(potential, "model_label", lambda: potential.name)()
    prov = Provenance(
        model=f"melt-quench[{label}]",
        fidelity=getattr(potential, "fidelity", Fidelity.TIER1_CLASSICAL),
        origin=Origin.CALCULATED if relaxed.convergence.converged else Origin.ESTIMATED,
        approximations=[
            f"Langevin dynamics at fixed volume (density {analysis['density_g_cm3']:.3f} "
            "g/cm^3), melt at "
            f"{melt_K:g} K for {melt_ps:g} ps, cooled in {step_K:g} K steps at "
            f"{cooling_K_per_ps:g} K/ps ({cooling_K_per_ps * 1e12:.1e} K/s), held "
            f"{hold_ps:g} ps at {final_K:g} K, then relaxed to a local minimum.",
            "Molecular-dynamics cooling rates are far faster than laboratory quenches; the "
            "glass is less relaxed and its structure depends on the potential.",
        ],
        parameters={"melt_K": melt_K, "final_K": final_K, "melt_ps": melt_ps,
                    "cooling_K_per_ps": cooling_K_per_ps, "hold_ps": hold_ps, "dt_fs": dt_fs,
                    "friction_per_fs": friction_per_fs, "seed": seed, "atoms": len(glass),
                    "md_steps": total_steps},
        seed=seed,
        references=["W. D. Luedtke and U. Landman, Phys. Rev. B 40 (1989) 1164"])
    return Result("amorphous_structure", glass, "", prov,
                  extra={"analysis": analysis, "trace": trace,
                         "energy_above_crystal_eV_per_atom": glass_energy - crystal_energy,
                         "wall_time_s": time.perf_counter() - t0})
