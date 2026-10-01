"""Density of states and projected DOS through GPAW: a molecule and a crystal.

Run with:
    materia run examples/scripts/15_dft_dos.py
or paste into the embedded Python console.

Each DOS is a frozen, versioned specification checked before GPAW starts: the
ground state it belongs to, the energy window and grid, the broadening, the
spin channels, the k-points and bands of the non-self-consistent step, and the
projections. GPAW's own DOS and projected DOS are recomputed from the returned
eigenvalues before anything is stored. A DOS observes a structure and never
changes it.
"""

import numpy as np

from materia.core_model import Cell, Structure

if not dft.available():
    print("GPAW is not usable here:", dft.status()["blocking_reason"])
    print(dft.status()["install_hint"])
else:
    box = 6.0
    h2 = Structure(np.array([1, 1]),
                   np.array([[box / 2, box / 2, box / 2 - 0.37],
                             [box / 2, box / 2, box / 2 + 0.37]]),
                   Cell(np.eye(3) * box, (False, False, False)))
    project.add_structure(h2)
    spec = dft.dos_spec(h2, xc="PBE", grid_spacing_A=0.22, energy_reference="absolute",
                        energy_min_eV=-15.0, energy_max_eV=0.0, energy_step_eV=0.01,
                        width_eV=0.05, n_bands=4)
    print(f"H2 DOS {spec.short_digest}: {spec.npoints} points, projections "
          f"{[p[0] for p in spec.projections]}, refusals: {dft.dos_check(spec)['blocking'] or 'none'}")
    molecule = dft.dos(spec=spec)
    summary = molecule["dos"].value
    eigenvalues = dft.dos_array(name="eigenvalues").data
    print(f"H2 DOS: highest occupied Kohn-Sham level {eigenvalues[0, 0, 0]:.4f} eV, "
          f"electrons below E_F {summary['checks']['integral_to_fermi_level_e']:.6f}")
    print("state:", dft.dos_state()["state"])

    a = 5.47
    si = Structure(np.array([14, 14]), np.array([[0.0, 0.0, 0.0], [a / 4] * 3]),
                   Cell(0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                        (True, True, True)))
    project.add_structure(si)
    crystal = dft.dos(si, xc="PBE", cutoff_eV=300.0, kpoints=[4, 4, 4],
                      occupations="fixed", smearing_eV=0.0, dos_kpoints=[6, 6, 6],
                      n_bands=12, energy_min_eV=-14.0, energy_max_eV=6.0,
                      energy_step_eV=0.02, width_eV=0.1)
    checks = crystal["dos"].value["checks"]
    print(f"Si DOS: Fermi level {crystal['dos'].value['fermi_level_eV']:.4f} eV, electrons "
          f"below E_F {checks['integral_to_fermi_level_e']:.6f} of "
          f"{checks['expected_valence_electrons']:g}, recomputation agrees to "
          f"{checks['recompute_max_relative_error']:.1e}")
    for label, value in checks["projection_integrals_states"].items():
        print(f"  {label}: {value:.4f} states in the window")
    bands = dft.dos_array(name="eigenvalues").data[0] - crystal["dos"].value["fermi_level_eV"]
    occupied = int(round(checks["expected_valence_electrons"] / 2))
    top, bottom = bands[:, :occupied].max(), bands[:, occupied:].min()
    print(f"  Kohn-Sham gap on the DOS grid {bottom - top:.4f} eV "
          f"(valence top {top:+.4f} eV, conduction bottom {bottom:+.4f} eV from E_F)")
