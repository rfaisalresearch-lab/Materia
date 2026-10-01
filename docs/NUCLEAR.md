# Nuclear masses, decay and particle data

Code: `materia/physics/nuclear.py`. Tests: `tests/unit/test_nuclear.py`.

| Capability | Source | Native or engine |
| --- | --- | --- |
| Atomic masses, mass excesses, binding per nucleon for 3,558 nuclides | AME2020 table shipped unchanged, checksum verified | published data |
| Binding energies, separation energies (n, p, 2n, alpha), decay Q values (alpha, beta-, beta+, EC, n, p, double beta-) | computed by Materia from AME2020 masses | Materia-native |
| Liquid-drop (semi-empirical) mass formula | coefficients fitted by Materia to the 2,484 measured AME2020 nuclides with A >= 16; rms 3.3 MeV | Materia-native model |
| Half-lives, decay modes, branching, progeny, inventories after time t | ICRP-107 via `radioactivedecay` (MIT) | external engine |
| Particle masses, widths, charges, spins, lifetimes | PDG via scikit-hep `particle` (BSD-3-Clause) | external engine |

Masses are atomic, so electron binding energies are neglected in differences;
`beta+` subtracts two electron masses and `ec` does not. Masses marked `#` in
AME2020 are estimates from systematics and every result built on one says so.
radioactivedecay's year is 365.2422 days.

## Checked

| Quantity | Materia | Reference |
| --- | --- | --- |
| Deuteron binding | 2.224566 MeV | 2.224566 MeV (AME2020) |
| Binding from masses against AME2020's tabulated B/A (H-2, He-4, Fe-56, Pb-208, U-238) | agree within the table's rounding | |
| U-238 alpha Q value | 4.2699 MeV | 4.270 MeV |
| Free neutron beta Q value | 0.78235 MeV | 782.3 keV |
| EC minus beta+ Q value | 1.021998 MeV | 2 m_e c^2 |
| Sr-90 after 10 years | Bateman closed form to 1e-9; Y-90 in secular equilibrium | |
| Proton and electron masses | 938.27208943 and 0.51099895 MeV | PDG |

## Reactions and stopping in matter

* `reaction_q_value(["d", "t"], ["a", "n"])`: Q values of balanced reactions
  from AME2020 masses (D-T 17.5893 MeV, D-D 3.2689 and 4.0327 MeV).
* `bethe_stopping_power(T, material)`: electronic mass stopping power of a
  charged particle from the Bethe formula with ICRU mean excitation energies
  (water 75 eV, air 85.7, Si 173, Al 166, Cu 322, Pb 823 eV), without shell,
  Barkas, Bloch or density corrections; refused below 1 MeV per nucleon and
  above about 1 GeV per nucleon.
* `csda_range(T, material)`: continuous-slowing-down range above that floor,
  with the omitted part bounded.

| Protons in water | Materia | NIST PSTAR |
| --- | --- | --- |
| Stopping at 10 MeV | 45.95 MeV cm^2/g | 45.67 |
| Stopping at 100 MeV | 7.290 MeV cm^2/g | 7.289 |
| CSDA range at 10 MeV | 0.1191 g/cm^2 plus at most 0.0038 below 1 MeV | 0.1230 |
| CSDA range at 100 MeV | 7.709 g/cm^2 plus at most 0.0038 below 1 MeV | 7.718 |

## Refused

Reaction cross sections, fission yields, shell-model or ab initio
nuclear structure, and collision, event generation or detector simulation
are not provided. Unknown nuclides, unknown decay modes and a modified data
file are refused with the reason.
