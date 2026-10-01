"""DFT relaxation through GPAW: a molecule at fixed cell, a crystal with a variable cell.

Run with:
    materia run examples/scripts/14_dft_relaxation.py
or paste into the embedded Python console.

Each relaxation is a frozen, versioned specification: every ground-state
variable plus the optimiser, its criteria and, for a variable cell, the free
strain components and the target pressure. It is checked before GPAW starts.
A converged relaxation is applied to its structure as one undoable change; one
that is cancelled, times out or fails keeps nothing.
"""

import numpy as np

from materia.core_model import Cell, Structure

if not dft.available():
    print("GPAW is not usable here:", dft.status()["blocking_reason"])
    print(dft.status()["install_hint"])
else:
    box = 6.0
    h2 = Structure(np.array([1, 1]),
                   np.array([[box / 2, box / 2, box / 2 - 0.40],
                             [box / 2, box / 2, box / 2 + 0.40]]),
                   Cell(np.eye(3) * box, (False, False, False)))
    project.add_structure(h2)
    spec = dft.relax_spec(h2, xc="PBE", grid_spacing_A=0.22, fmax_eV_A=0.02,
                          forces_tol_eV_A=0.01, observables=["energy", "forces"])
    print(f"H2 relaxation {spec.short_digest}: {spec.mode}, {spec.optimizer}, "
          f"refusals: {dft.relax_check(spec)['blocking'] or 'none'}")
    run = dft.relax(spec=spec)
    summary = run["relaxation"].value
    bond = float(np.linalg.norm(h2.positions[1] - h2.positions[0]))
    print(f"{summary['optimizer_steps']} steps, E {summary['initial_energy_eV']:.6f} -> "
          f"{summary['final_energy_eV']:.6f} eV, bond {bond:.4f} A")
    print(run["relaxation"].convergence.message)
    print("state:", dft.relax_state()["state"], "| applied:", run["relaxation"].extra["apply_reason"])
    lab.undo()
    print("after undo:", dft.relax_state()["state"], "| can apply again:",
          dft.relax_state()["applicable"])

    a = 5.60
    si = Structure(np.array([14, 14]), np.array([[0.0, 0.0, 0.0], [a / 4] * 3]),
                   Cell(0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                        (True, True, True)))
    project.add_structure(si)
    crystal = dft.relax(si, mode="variable-cell", xc="PBE", cutoff_eV=500.0,
                        kpoints=[4, 4, 4], occupations="fixed", smearing_eV=0.0,
                        fmax_eV_A=0.02, stress_tol_eV_A3=0.001, forces_tol_eV_A=0.005,
                        observables=["energy", "forces", "stress"])
    cell = crystal["relaxation"].value["cell"]
    lattice = (4.0 * cell["final_volume_A3"]) ** (1.0 / 3.0)
    print(f"Si volume {cell['initial_volume_A3']:.4f} -> {cell['final_volume_A3']:.4f} A^3 "
          f"({cell['volume_change_percent']:+.3f} %), lattice constant {lattice:.4f} A")
    print(f"final stress deviation {crystal['relaxation'].value['final_stress_residual_eV_A3']:.2e} eV/A^3")
