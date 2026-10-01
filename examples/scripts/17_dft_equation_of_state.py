"""Equation of state of diamond silicon through GPAW, and its equilibrium volume.

Run with:
    materia run examples/scripts/17_dft_equation_of_state.py
or paste into the embedded Python console.

The equation of state is a frozen, versioned specification: a reference
crystal, its electronic settings and the volumes to sample. Every volume is a
full ground state with exactly the reference's k-point grid and plane-wave
cutoff, because energies from different grids cannot be compared. The
Birch-Murnaghan fit is kept only if every point converged, the minimum lies
inside the sampled range and the fit reproduces every point. Applying the
equilibrium volume scales the structure as one undoable change.
"""

import numpy as np

from materia.core_model import Cell, Structure

if not dft.available():
    print("GPAW is not usable here:", dft.status()["blocking_reason"])
    print(dft.status()["install_hint"])
else:
    a = 5.43
    si = Structure(np.array([14, 14]), np.array([[0.0, 0.0, 0.0], [a / 4] * 3]),
                   Cell(0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                        (True, True, True)))
    project.add_structure(si)
    spec = dft.eos_spec(si, xc="PBE", cutoff_eV=400.0, kpoints=[8, 8, 8],
                        occupations="fermi-dirac", smearing_eV=0.01,
                        volume_min_scale=0.97, volume_max_scale=1.09, n_points=7)
    print(f"Si equation of state {spec.short_digest}: {spec.n_points} volumes, "
          f"refusals: {dft.eos_check(spec)['blocking'] or 'none'}")
    result = dft.eos(spec=spec)
    value = result.value
    print(f"Si equation of state: V0 {value['V0_A3_per_atom']:.4f} A^3/atom, "
          f"B0 {value['B0_GPa']:.1f} GPa, B' {value['B1']:.2f}, fit "
          f"{value['checks']['fit_rms_meV_per_atom']:.4f} meV/atom rms")
    print(f"  equilibrium cubic lattice constant "
          f"{(4 * value['V0_A3_per_atom'] * 2) ** (1 / 3):.4f} A")
    print("state:", dft.eos_state()["state"])
    print(dft.eos_apply())
    print("state:", dft.eos_state()["state"])
