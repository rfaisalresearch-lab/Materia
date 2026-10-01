"""Ground-state DFT through GPAW: a molecule, a cation, a crystal, a convergence check.

Run with:
    materia run examples/scripts/11_dft_ground_state.py
or paste into the embedded Python console.

Every run is a frozen, versioned specification checked before GPAW starts,
executed in GPAW's own interpreter, and stored with its provenance, SCF
history, electron accounting and grids. A run that does not converge keeps no
value. Numbers are relative to GPAW's reference atoms, and the functional's
own error is not reduced by any numerical setting.
"""

import numpy as np

from materia.core_model import Cell, Structure

if not dft.available():
    print("GPAW is not usable here:", dft.status()["blocking_reason"])
    print(dft.status()["install_hint"])
else:
    box = 8.0
    h2 = Structure(np.array([1, 1]),
                   np.array([[box / 2, box / 2, box / 2 - 0.37],
                             [box / 2, box / 2, box / 2 + 0.37]]),
                   Cell(np.eye(3) * box, (False, False, False)))
    spec = dft.experiment(h2, xc="PBE", grid_spacing_A=0.20)
    print(f"H2 specification {spec.short_digest}: {spec.boundary}, {spec.representation}, "
          f"poisson {spec.poisson}, refusals: {dft.check(spec)['blocking'] or 'none'}")
    molecule = dft.ground_state(spec=spec)
    energy = molecule["energy"]
    print(f"H2 free energy {energy.value:.6f} eV. {molecule['run'].convergence.message}")
    print(molecule["charge_accounting"].value["message"])
    print("state:", dft.state()["state"], "(the molecule is not held by the project)")

    cation = h2.copy()
    cation.formal_charges[:] = 0.5
    ion = dft.ground_state(cation, xc="PBE", grid_spacing_A=0.20)
    print(f"H2+ is spin-polarised by default: moment {ion['magnetic_moment'].value:.3f} mu_B")
    print(f"vertical ionisation energy {ion['energy'].value - energy.value:.2f} eV "
          "(experiment about 16.4 eV; PBE's error is part of the result)")

    refused = dft.check(dft.experiment(h2, representation="pw", cutoff_eV=300.0,
                                       grid_spacing_A=None))
    print("plane waves on an isolated molecule are refused:", refused["blocking"][0][:90], "...")

    silicon = materials.load("silicon").bulk(repeat=(1, 1, 1))
    crystal = dft.ground_state(silicon, xc="LDA", cutoff_eV=300.0, kpoints=[2, 2, 2])
    print(f"Si, 8 atoms: {crystal['energy'].extra['energy_per_atom_eV']:.5f} eV/atom, "
          f"pressure {crystal['stress'].extra['pressure_GPa']:.3f} GPa, "
          f"state {dft.state()['state']}")
    density = dft.array(name="density")
    print(f"density grid {density.shape}, stored in chunks in the project file")

    study = dft.convergence_study(silicon, parameter="cutoff_eV", values=[250, 300, 350],
                                  tolerance=0.01, xc="LDA", kpoints=[2, 2, 2])
    for point in study.value["points"]:
        print(f"  cutoff {point['value']:>5} eV: {point['observable']:.5f} eV/atom")
    print(study.value["verdict"])
    print(study.value["note"])
