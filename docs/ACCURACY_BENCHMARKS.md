# Accuracy benchmarks

This is the record of a systematic audit of the numbers Materia ships and
computes, each compared with a reference that shares no code with it. It
lists what agreed, what was wrong and has been fixed, and what is a limit of
the underlying model rather than of the implementation. Measured on
2026-09-29 on this machine: GPAW 25.7.0, ASE 3.29.0, NumPy 2.5.3, SciPy 1.18.1.

Three kinds of evidence are kept apart:

* **Implementation**: Materia against an independent implementation of the
  same model or an independently maintained copy of the same data. Agreement
  should be to rounding.
* **Model**: Materia against the published properties of the model it
  implements. Agreement shows the model is the published one.
* **Physics**: the model against all-electron calculations or experiment.
  Disagreement here is the model's error, reported and not tuned away.

Nothing here makes Materia a replacement for GPAW, LAMMPS, Quantum ESPRESSO,
VASP or ORCA. Its first-principles numbers are GPAW's, and it reproduces a
direct GPAW script exactly.

The permanent cases are `tests/validation/test_reference_benchmarks.py` (no
external solver, runs every time) and `tests/validation/test_dft_eos_live.py`
(GPAW, skipped as BLOCKED without it).

## Reference data

| Data | Reference | Result |
| --- | --- | --- |
| Physical constants in `core_model/units.py` | SciPy CODATA 2022 | agree to 1.4e-9 relative; Materia uses CODATA 2018 |
| Tunnelling prefactor `DECAY_PREFACTOR_INV_A_SQRT_EV` | sqrt(2 m e)/hbar | **was half the correct value; fixed.** Unused by the STM simulator, which calls the correct function |
| Standard atomic weights, 118 elements | ASE's IUPAC table | stable elements agree to 0.011 u |
| Masses of radioactive elements | AME atomic masses | **carried the bare mass number (up to 0.2 percent off); now the reference isotope's atomic mass**, except Lr, Sg, Rg, Mc, Ts |
| Covalent and van der Waals radii | ASE's Cordero and Alvarez tables | exact |
| Electronegativity, ionisation energy, electron affinity | CRC Handbook | agree to the stated precision; a few lanthanide affinities differ between published measurements |
| Ground-state electron configurations | NIST, including every Aufbau exception | exact |
| Isotope abundances and masses | IUPAC representative compositions, AME | abundances sum to one; weighted masses reproduce standard weights except lead (variable in nature) and selenium (IUPAC's own 0.012 u gap) |
| `Isotope.neutrons` | N = A - Z | **raised NotImplementedError; fixed** |

## Material library

| Check | Result |
| --- | --- |
| Lattice parameters of all 21 materials against diffraction references | agree |
| Densities, gaps, melting points, conductivities, elastic constants, cohesive energies, work functions | agree with handbook values |
| Density of each generated cell against the tabulated density | within 1.3 percent; GaN, MoS2 and WSe2 differences are handbook bulk densities against X-ray densities, now annotated |
| **4H-SiC basis** | **wrong stacking: a Si-C contact at 0.63 A and coordination 2 to 5. Fixed to the ABAC stacking of P6_3mc; every atom tetrahedral, Si-C 1.886 and 1.890 A** |
| **alpha-quartz dielectric constant** | **was the amorphous thermal-oxide value 3.9; now 4.52 (4.64 along c)** |
| Contacts and first-shell coordination of every shipped crystal | now checked for all 21 materials |
| Built slabs: Si(111), Si(100), Si(110), Cu(111), Cu(100), W(110), GaAs(110) | analytic layer spacings and bond lengths exactly |
| **Displayed bond network** | **the covalent-radius rule alone added Al-Al bonds in corundum (Al coordination 7), Mo-Mo and W-W bonds in MoS2 and WSe2 (metal coordination 12) and the second shell of bcc tungsten (14). A shared Voronoi face of at least 5 percent of the solid angle is now also required: first-shell bonds subtend 8 to 30 percent, the screened contacts 0.4 to 3.5 percent. All 21 crystals now show textbook coordination; molecules keep every bond** |
| **Adatom default height** | **was 2.0 A above the surface for every element; now the covalent contact with the nearest surface atom** |

## Implementations

| Component | Reference | Result |
| --- | --- | --- |
| Neighbour list | ASE `neighbor_list`, 60 random triclinic cells, mixed periodicity, cutoffs longer than the cell | identical pair and image sets |
| Tight binding sp3s* Si, Ge, GaAs | independent transcription of Vogl's k-space Hamiltonian, 16 k-points each | **2 to 3e-4 eV before; 2e-14 eV after tying the scaling bond lengths to the lattices** |
| Tight binding pz graphene | 6\|t\| = 16.2 eV | **16.178 eV before; 16.200 eV after** |
| EAM Cu, Au, W | LAMMPS results of the NIST repository and ASE's EAM calculator on the same files | agree (existing cases in `test_eam_validation.py`) |
| Ewald electrostatics | Madelung constants | agree to 1e-8 (existing cases) |
| Materia's GPAW driver | the same calculation as a direct ASE and GPAW script | identical energies and fitted volumes |

## Models against their published properties

**Stillinger-Weber silicon** against Balamane, Halicioglu and Tiller (1992) and
Cowley (1988):

| Property | Materia | Published |
| --- | ---: | ---: |
| a0 | 5.43095 A | 5.431 A |
| E_coh | -4.3366 eV | -4.3366 eV |
| C11 | 151.2 GPa | 151.4 GPa |
| C12 | 76.0 GPa | 76.4 GPa |
| C44, internally relaxed | 56.45 GPa | 56.4 GPa |
| B | 101.1 GPa | 101.4 GPa |
| Vacancy formation | 4.3366 eV ideal (a local minimum); 2.694 eV reconstructed, 216 atoms | about 2.6 to 2.8 eV reconstructed |

The ideal vacancy is a genuine local minimum because its neighbours sit just
beyond the potential's cutoff; the reconstructed one needs an inward start.

**Lennard-Jones metals** derived from a material's cohesive energy and lattice
constant: **the derivation used infinite lattice sums while the potential is
cut and shifted at 10 A, so gold gave -3.660 instead of -3.81 eV/atom with the
minimum 0.17 percent too wide.** The parameters now come from the shells
actually inside the cutoff and reproduce both inputs exactly for Cu, Ag, Au,
Ni and Pt.

**Tight binding against Vogl's fitted values**: Si gap 1.1713 eV (fitted to
1.17) with its minimum at 0.731 of Gamma to X, not near 0.85 as the validation
record previously said; 0.85 is the experimental position. Ge 0.765 eV at L,
GaAs 1.550 eV at Gamma.

**STM decay**: above an atom the apparent decay of the current is 0.901, 0.932
and 0.965 of 2 kappa over 6 to 9, 10 to 16 and 20 to 28 A, approaching 2 kappa
from below as the neighbours' slower-decaying tails fade. A 12 percent
tolerance used to hide this trend; the case now checks it.

**Langevin thermostat**: 501.2 +- 1.6 K over 12 ps at a 500 K target.

## First principles: PBE equations of state

Seven-point Birch-Murnaghan fits through Materia's ground-state experiment
with a fixed k-point grid, against WIEN2k all-electron PBE from the Delta
project (Lejaeghere et al., Science 351 (2016) aad3000).

| Element | Settings | V0 / A^3 per atom | B0 / GPa |
| --- | --- | --- | --- |
| Si | 500 eV, 12x12x12 | 20.520 (+0.33 %) | 88.4 (-0.1 %) |
| Al | 500 eV, 18x18x18 | 16.526 (+0.28 %) | 77.6 (-0.5 %) |
| Cu | 600 eV, 16x16x16 | 12.096 (+1.22 %) | 137.1 (-3.0 %) |

Copper's offset is GPAW's PAW dataset: GPAW run directly, without Materia,
gives the same volume to 1e-4 A^3 at 600 and 900 eV.

### What the default settings give

| Element | Old defaults (20 A k-density) | New defaults (30 A) |
| --- | --- | --- |
| Si | +0.39 % V0, +0.3 % B0 | +0.33 % V0, +0.1 % B0, identical to converged |
| Al | **+5.3 % V0, +135 % B0, negative B'** | +0.38 % V0, +0.9 % B0 |
| Cu | **+2.9 % V0, B' -13.9** | +1.17 % V0, -5.6 % B0 |

The old aluminium result was wrong because the automatic grid changed from
8x8x8 to 7x7x7 part way through the scan: energies from different grids are
not comparable. On a fixed 8x8x8 grid the same scan gives +0.39 percent and
+3.7 percent. Copper on a fixed grid needs 14x14x14 for its bulk modulus to be
within 2 percent of the converged value. The default density is now 30 A,
and every check warns while the automatic grid is in use, because copper's
automatic grid still changes across a 6 percent volume scan.

The convergence laboratory's supercell study now records whether each point
samples the Brillouin zone equivalently to the base cell.

### Stress needs a higher cutoff than energy

The equation-of-state tool compares GPAW's stress with -dE/dV of the fitted
energies. For copper at the default 400 eV they disagreed by 8 GPa, and a direct
GPAW calculation at the fitted volume confirmed it is the basis, not the fit:
the pressure is -7.3 GPa at 400 eV, -0.02 GPa at 600 eV and -0.64 GPa at 800
eV. Energy-derived volumes are much less sensitive. The check now warns
whenever stress is requested below 600 eV, which covers variable-cell
relaxation, whose lattice would otherwise be pulled by that spurious stress.

### The equation-of-state experiment

**The first version fitted the wrong energy.** It fitted GPAW's free energy
E - TS of the smeared occupations while calling the result a 0 K equation of
state. For copper at 0.1 eV Fermi-Dirac smearing, TS runs from -5.69 to -6.45
meV across the scan and the free-energy fit gives V0 = 12.0893 A^3 against
12.0818 A^3 from the zero-width energy. The experiment now fits the zero-width
energy, keeps the free energy as a separate curve and fit, and compares GPAW's
stress, the derivative of the free energy, with -dF/dV. A regression test with
a volume-dependent entropy term fails on the first version (it recovers V0 =
20.145 against the true 20.0) and passes now.

`docs/DFT_EQUATION_OF_STATE.md` pins the reference's grid and cutoff at every
volume, so the failure above cannot happen through it.

| Case | Result |
| --- | --- |
| Si, 500 eV, 12x12x12, 0.97 to 1.09 V_ref | V0 20.520 A^3/atom (+0.33 % against all-electron), B0 88.55 GPa, B' 4.35, fit 0.002 meV/atom rms; stress against fit 0.012 GPa; after applying the equilibrium a fresh ground state reads -0.007 GPa |
| Cu, automatic 12x12x12 pinned, 0.90 to 1.10 V_ref, where per-cell grids would be 13x13x13 and 12x12x12 | smooth: 0.36 meV/atom rms, B' 5.21, V0 12.073 A^3/atom, B0 137.5 GPa; the stress check flagged the 400 eV Pulay stress described above |

## What remains a limit of the models

* Stillinger-Weber gives a symmetric Si(100) dimer 7 percent longer than LEED
  and no buckling (`VALIDATION.md`).
* PBE overestimates lattice constants and underestimates gaps; the silicon
  Kohn-Sham gap is about 0.55 eV against 1.17 eV measured.
* The tight-binding silicon conduction minimum sits at 0.73 of Gamma to X
  against about 0.85 measured.
* Bonds are drawn from geometry, not computed; above 20,000 atoms the Voronoi
  filter is skipped for speed and the bonds say so (`LIMITATIONS.md`).
