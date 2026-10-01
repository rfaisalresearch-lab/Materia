"""Benchmark scenes.

These measure what this machine actually does. Nothing here is a claimed or
extrapolated figure: every number is produced by running the code. Results
vary with hardware, BLAS build and thread count, which are reported alongside.
"""

from __future__ import annotations

import platform
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .materials import load as load_material
from .physics.bonds import perceive_bonds
from .physics.neighbors import neighbor_list
from .physics.potentials import StillingerWeber
from .structure_builder.surface import make_surface
from .version import __version__


def environment() -> dict:
    info = {
        "materia_version": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "numpy": np.__version__,
    }
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = "not installed"
    try:
        config = np.show_config(mode="dicts")
        build = config.get("Build Dependencies", {}).get("blas", {})
        info["blas"] = f"{build.get('name', '?')} {build.get('version', '')}".strip()
    except Exception:
        info["blas"] = "unknown"
    return info


def _slab_for(target_atoms: int):
    """Build a Si(111) slab with approximately ``target_atoms`` atoms."""
    silicon = load_material("silicon")
    layers = 6
    per_cell = 2 * layers
    side = max(1, int(round((target_atoms / per_cell) ** 0.5)))
    return make_surface(silicon, (1, 1, 1), size=(side, side, layers), vacuum_A=14.0)


def _timed(fn, repeats: int = 1):
    best = float("inf")
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        best = min(best, time.perf_counter() - start)
    return best, value


def run_benchmarks(sizes: Sequence[int] = (1000, 10000, 100000),
                   include_scan: bool = True) -> List[dict]:
    """Run every benchmark scene and return measured timings."""
    rows: List[dict] = []
    for target in sizes:
        slab = _slab_for(target)
        n = len(slab)

        seconds, nl = _timed(
            lambda: neighbor_list(slab.positions, slab.cell, 3.0), repeats=2)
        rows.append({"scene": "neighbour list, 3.0 A cutoff", "n_atoms": n,
                     "seconds": seconds,
                     "note": f"{len(nl)} directed pairs"})

        seconds, bonds = _timed(lambda: perceive_bonds(slab), repeats=1)
        rows.append({"scene": "bond perception", "n_atoms": n, "seconds": seconds,
                     "note": f"{len(bonds)} bonds"})

        potential = StillingerWeber("Si")
        seconds, result = _timed(
            lambda: potential.energy_and_forces(slab), repeats=2)
        rows.append({"scene": "Stillinger-Weber energy and forces", "n_atoms": n,
                     "seconds": seconds,
                     "note": f"E/atom = {result[0] / n:.6f} eV"})

        if n <= 20000:
            from .solvers.classical import ClassicalSolver
            solver = ClassicalSolver(potential)
            seconds, out = _timed(
                lambda: solver.relax(slab, fmax_eV_A=0.05, max_steps=20), repeats=1)
            rows.append({"scene": "FIRE relaxation, 20 steps", "n_atoms": n,
                         "seconds": seconds,
                         "note": f"max |F| = {out.convergence.residual:.4f} eV/A"})

    if include_scan:
        rows.extend(scan_benchmarks())
    return rows


def rock_salt_supercell(target_atoms: int, slab: bool = False):
    """A rock-salt (NaCl) supercell with about ``target_atoms`` atoms.

    Used to time point-charge electrostatics on a structure whose exact
    Madelung energy is known, so every timing row also reports how far the
    computed energy is from it.
    """
    from .core_model.cell import Cell
    from .core_model.structure import Structure

    a = 5.64
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    basis = np.vstack([fcc, fcc + [0.5, 0, 0]])
    n = max(1, int(round((target_atoms / 8) ** (1.0 / 3.0))))
    grid = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)])
    positions = ((basis[None, :, :] + grid[:, None, :]).reshape(-1, 3)) * a
    numbers = np.tile([11] * 4 + [17] * 4, len(grid))
    pbc = (True, True, False) if slab else (True, True, True)
    return Structure(numbers, positions, Cell.cubic(n * a, pbc))


def electrostatics_benchmarks(sizes: Sequence[int] = (1000, 10000, 100000),
                              checked_limit: int = 20000,
                              slab_limit: int = 20000) -> List[dict]:
    """Ewald timings for rock-salt bulk and slabs with unit point charges."""
    from .core_model.units import COULOMB_K_EV_A
    from .physics.electrostatics import ChargeModel, EwaldSettings, compute

    model = ChargeModel.per_element({"Na": 1.0, "Cl": -1.0}, "benchmark")
    rows: List[dict] = []
    for target in sizes:
        for slab in (False, True):
            structure = rock_salt_supercell(target, slab=slab)
            n = len(structure)
            if slab and n > slab_limit:
                continue
            for checked in (False, True):
                if checked and n > checked_limit:
                    continue
                settings = EwaldSettings(check_convergence=checked)
                seconds, out = _timed(lambda: compute(structure, model, settings))
                note = (f"{out.parameters['n_kvectors_half_space']} k vectors, "
                        f"r_c {out.parameters['real_cutoff_A']:.1f} A")
                if not slab:
                    m = -out.energy_eV / (n / 2) * (5.64 / 2) / COULOMB_K_EV_A
                    note += f", Madelung {m:.9f}"
                rows.append({"scene": ("Ewald slab" if slab else "Ewald bulk")
                                      + (", with convergence check" if checked else ""),
                             "n_atoms": n, "seconds": seconds, "note": note})
    return rows


def eam_benchmarks(sizes: Sequence[int] = (1000, 10000, 100000)) -> List[dict]:
    """EAM timings for fcc Cu with the shipped Zhou 2004 potential.

    Each size is timed three ways: a first energy and force evaluation, which
    includes building the neighbour list; a repeat evaluation after a small
    displacement, which reuses the Verlet list; and ten molecular-dynamics
    steps, which is what an interactive relaxation or dynamics run costs per
    step.
    """
    from .core_model.cell import Cell
    from .core_model.structure import Structure
    from .physics import eam
    from .solvers.classical import ClassicalSolver

    rows: List[dict] = []
    a = 3.615
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    for target in sizes:
        n_side = max(1, int(round((target / 4) ** (1.0 / 3.0))))
        base = Structure([29] * 4, fcc * a, Cell.cubic(a)).repeat(n_side, n_side, n_side)
        n = len(base)
        potential = eam.load_shipped("Cu-Zhou04")
        seconds, out = _timed(lambda: potential.evaluate(base))
        rows.append({"scene": "EAM energy and forces, first call", "n_atoms": n,
                     "seconds": seconds,
                     "note": f"{out['n_pairs']} directed pairs, "
                             f"E/atom {out['energy_eV'] / n:.6f} eV"})
        nudged = base.copy()
        nudged.positions = base.positions + 0.01
        seconds, _ = _timed(lambda: potential.evaluate(nudged), repeats=2)
        rows.append({"scene": "EAM energy and forces, Verlet list reused", "n_atoms": n,
                     "seconds": seconds,
                     "note": f"{potential.last_evaluation['neighbour_list_builds']} list build(s)"})
        solver = ClassicalSolver(eam.load_shipped("Cu-Zhou04"))
        seconds, md = _timed(lambda: solver.dynamics(base, steps=10, dt_fs=2.0,
                                                     temperature_K=300.0,
                                                     thermostat="langevin", seed=1))
        rows.append({"scene": "EAM molecular dynamics, 10 steps", "n_atoms": n,
                     "seconds": seconds,
                     "note": f"{seconds / 10 * 1000:.1f} ms per step"})
    return rows


def scan_benchmarks() -> List[dict]:
    """Scanning-probe timings on a small Si(111) surface."""
    from .microscopy.noise import NoiseModel
    from .microscopy.stm import STMSettings, STMSimulator
    from .microscopy.afm import AFMSettings, AFMSimulator
    from .solvers.tight_binding import TightBinding

    rows: List[dict] = []
    silicon = load_material("silicon")
    slab = make_surface(silicon, (1, 1, 1), size=(4, 4, 4), vacuum_A=14.0)
    n = len(slab)

    STMSimulator.clear_cache()
    simulator = STMSimulator(TightBinding("sp3s*-Si"))
    for resolution, label in (((128, 128), "128 x 128"), ((256, 256), "256 x 256")):
        settings = STMSettings(bias_V=1.0, resolution=resolution, kgrid=(2, 2),
                               noise=NoiseModel.realistic(0))
        seconds, scan = _timed(lambda: simulator.scan(slab, settings), repeats=1)
        cached = scan.provenance.parameters.get("electronic_from_cache")
        rows.append({
            "scene": f"STM constant current {label}"
                     + (" (cached electronic structure)" if cached else ""),
            "n_atoms": n, "seconds": seconds,
            "note": f"corrugation {float(np.ptp(scan.channel('topography'))):.3f} A",
        })

    afm = AFMSimulator()
    settings = AFMSettings(mode="fm-afm", resolution=(160, 160),
                           noise=NoiseModel.realistic(0))
    seconds, scan = _timed(lambda: afm.scan(slab, settings), repeats=1)
    rows.append({"scene": "AFM frequency shift 160 x 160", "n_atoms": n,
                 "seconds": seconds,
                 "note": f"range {float(np.ptp(scan.channel('frequency_shift'))):.3f} Hz"})
    return rows


def render_benchmark(counts: Sequence[int] = (1000, 10000, 100000, 1000000)) -> List[dict]:
    """Measure how long it takes to prepare renderer payloads."""
    from .desktop_ui.service import Service

    rows: List[dict] = []
    service = Service()
    for target in counts:
        try:
            slab = _slab_for(target)
        except MemoryError:
            rows.append({"scene": f"render payload target {target}", "n_atoms": 0,
                         "seconds": float("nan"), "note": "out of memory"})
            continue
        service.project.add_structure(slab)
        seconds, payload = _timed(
            lambda: service.render_payload(max_atoms=250_000, include_bonds=target <= 60000))
        rows.append({
            "scene": "renderer payload" + (" (level of detail)" if payload["truncated"] else ""),
            "n_atoms": len(slab), "seconds": seconds,
            "note": f"{len(payload['ids'])} instances, stride {payload['stride']}",
        })
    return rows
