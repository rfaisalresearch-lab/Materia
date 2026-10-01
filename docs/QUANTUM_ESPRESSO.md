# Quantum ESPRESSO (pw.x)

Materia drives Quantum ESPRESSO's plane-wave DFT code `pw.x` as a separate
program. It does not implement plane-wave DFT itself; results are labelled
`qe/pw.x` and record the pw.x version and every pseudopotential's file name
and SHA-256.

Code: `materia/physics/espresso.py`; solver `qe/pw.x` through
`materia.solvers.molecular.create("qe/pw.x", ...)`. Tests:
`tests/validation/test_espresso_live.py` (skipped as BLOCKED without pw.x).

## Setup

```bash
CONDA_SUBDIR=osx-64 conda create -n materia-qe -c conda-forge qe
```

conda-forge has no Apple-silicon build, so the x86_64 build runs under
Rosetta. Materia finds `pw.x` through `MATERIA_QE_PW`, then `PATH`, then
conda environments (`materia-qe` first). Pseudopotentials are read from
`MATERIA_PSEUDO_DIR` or `~/.cache/materia/pseudopotentials`;
`fetch_pseudopotential` downloads PSlibrary 1.0.0 PBE files from the
Quantum ESPRESSO site. Materia ships no pseudopotential.

## What it computes

SCF energy, forces and stress for three-dimensional periodic cells, with
cutoffs, Monkhorst-Pack k points, smearing, spin polarisation and starting
magnetisation, and with or without symmetry. Because it is a Materia
potential, relaxation, molecular dynamics, finite-displacement phonons and
NEB work with it, at one pw.x run per force evaluation.

## Native pw.x workflows

* `relax(structure, variable_cell=False)` runs pw.x's own BFGS (`relax`) or
  cell and ion relaxation (`vc-relax`, optional target pressure) and returns
  the relaxed structure.
* `band_structure(structure, path="LGXUKG")` runs the SCF on the k grid and a
  `bands` calculation along an ASE high-symmetry path, and reports
  eigenvalues, the valence-band maximum, conduction-band minimum and gap.

Every pw.x call runs in a scratch directory with one thread and a timeout.

## Phonons by density-functional perturbation theory

`EspressoPhonons(potential).gamma(structure)` runs ph.x at Gamma and dynmat.x
with the crystal acoustic sum rule; `.dispersion(structure, q_grid, path)`
runs ph.x on a q grid, q2r.x for real-space force constants and matdyn.x along
an ASE high-symmetry path (Baroni et al., Rev. Mod. Phys. 73 (2001) 515).

| Si, PBE, 45/360 Ry, 6x6x6 k, a = 5.4704 A | Materia | Reference |
| --- | --- | --- |
| Gamma optical, ph.x | 15.1338 THz | Materia finite displacements with pw.x forces: 15.130 THz |
| X (4x4x4 q) | TA 4.385, LA/LO 12.023, TO 13.382 THz | measured 4.49, 12.32, 13.90 THz |
| L (4x4x4 q) | TA 3.314, LA 11.161, LO 11.956, TO 14.321 THz | measured 3.43, 11.35, 12.60, 14.68 THz |

Measured values are inelastic neutron scattering (G. Nilsson and G. Nelin,
Phys. Rev. B 6 (1972) 3777); PBE underestimates them by 2 to 4 percent, as is
usual. The 4x4x4 dispersion took about ten minutes.

## Projected DOS and magnetism

* `projected_dos(potential, structure, nscf_kpts)` runs an nscf step on a
  denser grid and projwfc.x: total DOS, Loewdin-projected DOS per atom and
  orbital, and the spilling parameter that measures what the atomic
  projectors miss.
* Spin-polarised runs report the total and absolute magnetisation.

| Check | Result |
| --- | --- |
| Si occupied DOS integrated | 8.0003 electrons for 2 atoms x 4 valence electrons |
| Si projection completeness | 0.991 = 1 - spilling (0.0089) |
| Si orbital character | valence-band top p-dominated (p/s = 46), band bottom s-dominated |
| bcc Fe, PBE, a = 2.83 A, 64/782 Ry, 8x8x8 k | 2.11 muB per atom against 2.22 muB measured |

## Dielectric response, Born charges and LO-TO splitting

`EspressoPhonons(potential).dielectric(crystal, q_direction)` runs ph.x at
Gamma with the electric-field perturbation and returns epsilon_infinity, the
Born effective charge tensor of every atom, the charge-neutrality violation
`max |sum Z*|` (a measure of k-point convergence), and the Gamma frequencies
without and with the non-analytic term for a phonon along `q_direction`. For
a cubic crystal with two atoms, Materia also computes the LO frequency itself
from `w_LO^2 = w_TO^2 + Z*^2 e^2 / (eps0 eps_inf Omega mu)`, and the static
constant from Lyddane-Sachs-Teller. It needs an insulator: use
`smearing="fixed"` (fixed occupations), which is now supported; smearing is
refused. For an insulator, ph.x's q-grid run in `dispersion()` computes the
dielectric tensor and charges at Gamma by itself, so matdyn.x adds the LO-TO
term along paths (`lo_to_splitting` in the result); with smearing it cannot.
For AlAs on a 4x4x4 q grid (k 8x8x8) the LO branch next to Gamma is 11.442
THz, equal to `dielectric()` at the same k grid.

AlAs, PBE, 40/320 Ry, a = 5.73 A:

| k grid | eps_inf | Z*(Al) | Z*(As) | sum rule |
| --- | ---: | ---: | ---: | ---: |
| 6x6x6 | 11.47 | 2.097 | -2.423 | 0.33 |
| 8x8x8 | 10.13 | 2.144 | -2.223 | 0.08 |
| 12x12x12 | 9.61 | 2.159 | -2.165 | 0.006 |
| 16x16x16 | 9.56 | 2.160 | -2.161 | 0.0005 |

At 12x12x12: TO 10.42 THz and LO 11.47 THz against 10.82 and 12.05 THz
measured (PBE softens both by about 4%); Z* 2.16 against 2.18 measured;
eps_inf 9.6 against 8.16 measured, the usual PBE overestimate from a too
small gap. Materia's LO frequency equals dynmat.x's non-analytic result to
2e-6. The table shows why the sum-rule violation is reported: coarse k grids
give badly unconverged charges.

## Orbital-resolved bands

`fat_bands(potential, structure, path)` runs SCF, a bands run along the path
and projwfc.x with `lsym = .false.`, then reads `atomic_proj.xml`. Every state
carries its squared Loewdin projections onto the pseudopotential's atomic
wavefunctions, summed into element and angular-momentum channels
(`character["Si:p"]`) and into atoms (`atom_character`), with the spilling
`1 - sum` of each state. The projected eigenvalues are checked against the
pw.x band energies, and a state whose weights sum above one is refused.
Spin-polarized fat bands are refused for now.

Silicon, 45/360 Ry: the lowest state at Gamma is 99.6% s, the threefold
valence-band top 96% p, the two atoms carry equal total valence character as
symmetry requires (individual degenerate bands may split it arbitrarily), and the mean spilling of the valence states is 1.4%.

## Choices made explicit

* A cutoff below a pseudopotential's suggested minimum is refused unless
  `allow_low_cutoff=True`.
* Pseudopotentials generated with different functionals are refused.
* Smearing defaults to Gaussian. Marzari-Vanderbilt occupations are not
  monotonic and, for gapped silicon at 0.01 Ry, gave pw.x energies that jump
  by 4 meV between geometries 0.005 A apart while the forces stayed smooth;
  use them for metals only.
* Energies and forces are read from pw.x output with CODATA 2018 constants,
  as pw.x uses. ASE's own reader uses CODATA 2006 and differs by about
  1e-7 relative.

## Checked

| Check | Result |
| --- | --- |
| Adapter against the same input written by hand and run with pw.x directly | equal to 2e-6 eV |
| Force on a displaced Si atom against central differences of the energy | within 2e-3 eV/A |
| Si PBE lattice constant, 45/360 Ry, 6x6x6 k | 5.4707 A against 5.4686 A all-electron (WIEN2k, Lejaeghere et al. 2016) |
| Si lattice constant from pw.x vc-relax | 5.4704 A, consistent with the energy-volume fit |
| Si PBE band gap along L-G-X-U-K-G | 0.611 eV, indirect, against about 0.61 eV in PBE literature |
| Stress of compressed Si (a = 5.431 A) | isotropic and compressive, as expected below a0 |

## Limitations

* DFPT runs assume an insulator or small smearing. LO-TO splitting needs
  fixed occupations (`smearing="fixed"`).
* Slabs and molecules must be padded with vacuum in a fully periodic cell.
* Serial runs; no MPI.
