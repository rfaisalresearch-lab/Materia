"""Local density of states of graphene through GPAW, and a Tersoff-Hamann STM image.

Run with:
    materia run examples/scripts/18_dft_ldos_stm.py
or paste into the embedded Python console.

The LDOS is a frozen, versioned specification: the ground state, an energy
window about the Fermi level and the bands of a non-self-consistent step run
with point-group symmetry off. The map is the sum of |psi|^2 of the Kohn-Sham
states in the window, from GPAW's pseudo-wavefunctions, exact in the vacuum
above the PAW spheres. STM images are taken from it at heights in that vacuum;
the value is proportional to the tunnelling current and is not converted to
amperes.
"""

import math

import numpy as np

from materia.core_model import Cell, Structure

if not dft.available():
    print("GPAW is not usable here:", dft.status()["blocking_reason"])
    print(dft.status()["install_hint"])
else:
    a, vacuum = 2.46, 16.0
    graphene = Structure(np.array([6, 6]),
                         np.array([[0.0, 0.0, 5.0], [0.0, a / math.sqrt(3), 5.0]]),
                         Cell(np.array([[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0],
                                        [0, 0, vacuum]]), (True, True, False)))
    project.add_structure(graphene)
    spec = dft.ldos_spec(graphene, xc="PBE", grid_spacing_A=0.2, kpoints=[6, 6, 1],
                         smearing_eV=0.05, energy_min_eV=-1.0, energy_max_eV=0.0, n_bands=8)
    print(f"graphene LDOS {spec.short_digest}: refusals "
          f"{dft.ldos_check(spec)['blocking'] or 'none'}")
    results = dft.ldos(spec=spec)
    summary = results["ldos"].value
    print(f"graphene LDOS: {summary['states_in_window']:.4f} states per cell in "
          f"{summary['window_eV']} eV, map integral {summary['map_integral_states']:.4f}, "
          f"PAW sphere radii {summary['augmentation_radii_A']}")
    image = dft.stm_image(mode="constant-height", height_A=3.0)
    values = image["values"]
    print(f"constant-height image at 3 A: {values.shape}, LDOS {values.min():.3e} to "
          f"{values.max():.3e} states/A^3, valid heights {image['valid_heights_A']}")
    print("state:", dft.ldos_state()["state"])
