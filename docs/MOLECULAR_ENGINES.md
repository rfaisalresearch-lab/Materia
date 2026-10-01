# Molecular and quantum-chemistry engines

Materia drives three established open-source engines for molecules. It does
not implement their physics: each result names the engine and version that
produced it.

| Model | Engine and licence | Physics | Systems |
| --- | --- | --- | --- |
| `rdkit/mmff94`, `rdkit/uff` | RDKit, BSD-3-Clause | MMFF94 (Halgren 1996) and UFF (Rappe et al. 1992) force fields on a fixed bonded topology | isolated molecules |
| `xtb/gfn2`, `xtb/gfn1` | tblite, LGPL-3.0-or-later | GFN2-xTB and GFN1-xTB tight binding; bonds can form and break | molecules and 3D crystals |
| `pyscf/hf`, `pyscf/b3lyp`, `pyscf/pbe`, `pyscf/mp2` | PySCF, Apache-2.0 | Hartree-Fock, Kohn-Sham DFT and MP2 with analytic gradients; CCSD and CCSD(T) energies through `pyscf_single_point` | isolated molecules |
| `openmm/amber14-gbn2`, `openmm/amber14`, `openmm/charmm36`, `openmm/amber99sbildn` | OpenMM, MIT and LGPL-3.0 | biomolecular force fields: Amber ff14SB with GBn2 implicit solvent, Amber14 or CHARMM36 with explicit water and PME, Amber99SB-ILDN | proteins and nucleic acids from PDB files |

Code: `materia/physics/molecular.py` (engines, SMILES builder),
`materia/solvers/molecular.py` (solvers). Engines are called through their
Python APIs; no engine code is copied into Materia.

## Use

```python
from materia.physics.molecular import molecule_from_smiles, pyscf_single_point
from materia.solvers.molecular import create

ethanol = molecule_from_smiles("CCO", seed=1)        # ETKDGv3 embedding, MMFF94 pre-optimised
solver = create("pyscf/b3lyp", basis="def2-tzvp")    # any basis, charge or spin
relaxed = solver.relax(ethanol, fmax_eV_A=1e-3).structure
modes = solver.phonons(relaxed)                      # finite-displacement vibrations
pyscf_single_point(relaxed, "ccsd(t)", "cc-pvtz")    # energy, dipole, charges, gap
```

Every engine is a Materia potential, so relaxation, molecular dynamics,
harmonic vibrations and thermodynamics, NEB pathways, structure comparison and
model ensembles work with it. `create(name, **options)` passes engine options
such as `basis`, `xc`, `charge`, `spin`, `unpaired` or `accuracy`.

## Biomolecules

```python
from materia.physics.biomolecular import load_pdb, simulate
protein = load_pdb("1l2y.pdb", fix=False)          # fix=True runs PDBFixer at pH 7
energy = create("openmm/amber14-gbn2").single_point(protein)
run = simulate(protein, steps=50_000, temperature_K=300)   # OpenMM's own Langevin MD
```

`load_pdb` keeps the residue topology the force field needs, records the
sequence, chains and whether PDBFixer repaired the file, and ignores the
placeholder `CRYST1` box of NMR entries. `OpenMMPotential` evaluates the force
field without constraints so that Materia's minimisers, dynamics, vibrations
and NEB see exact gradients. `simulate` hands production dynamics to OpenMM
(LangevinMiddle integrator, bonds to hydrogen constrained, local minimisation
first) and says so in its provenance.

## Solvation, excited states and conformers

| Capability | Engine | Call |
| --- | --- | --- |
| Implicit solvent ALPB or GBSA for GFN-xTB | tblite | `XTBPotential(solvent="water", solvation_model="alpb")` |
| Implicit solvent PCM or ddCOSMO for HF and DFT, energies and gradients | PySCF | `PySCFPotential("dft", dielectric=78.3553)` |
| Explicit water and ions in a periodic box, PME | OpenMM Modeller | `solvate(protein, padding_A=10, ionic_strength_M=0.15)` |
| NPT dynamics with a Monte Carlo barostat | OpenMM | `simulate(boxed, pressure_bar=1.0)` |
| Vertical singlet excitations (TDDFT, TDA, CIS) with oscillator strengths | PySCF | `excited_states(molecule, nstates=5, method="tddft")` |
| Distinct low-energy conformers (ETKDGv3, RMSD pruning, MMFF94 or UFF ranking) | RDKit | `conformers("CCCCO", count=20)` |

Implicit models are continua: electrostatics plus, for ALPB and GBSA, fitted
non-polar terms; PCM and ddCOSMO here include electrostatics only. Measured
for water: ALPB -9.2, GBSA -6.1 and PCM (HF/6-31G*) -7.3 kcal/mol against the
experimental hydration free energy of about -6.3 kcal/mol. A solvated Trp-cage
box equilibrates in NPT to 1.02 to 1.03 g/cm^3.

## Electric fields

`field_response(molecule, method, basis)` applies uniform fields of +-F and
+-2F along each axis in the one-electron Hamiltonian and returns the dipole
(from the density and from -dE/dF) and the static polarizability tensor
(from dipole derivatives and, on the diagonal, from energy second
derivatives). Water, HF/aug-cc-pVDZ: the three dipole routes agree with
PySCF's analytic dipole to 1e-5 a.u. (0.793 a.u.); the two polarizability
routes agree to 1e-4 a.u.; isotropic polarizability 8.28 a.u. against 9.8 a.u.
measured, the usual Hartree-Fock underestimate.

## Psi4, a second quantum-chemistry engine

`Psi4Potential(method, basis)` (models `psi4/hf`, `psi4/b3lyp`, `psi4/mp2`)
runs Psi4 1.9.1 (LGPL-3.0) in its own conda environment, one worker process
per energy or gradient, with one thread and a timeout. `single_point(molecule,
"ccsd(t)")` gives energies for methods without gradients here. Exact
integrals are the default. Install with
`conda create -n materia-psi4 -c conda-forge psi4=1.9.1 'libint<2.10' 'libxc-c<7'`:
newer conda-forge builds either cannot load Libint or fail on a Libxc 7
functional.

Parity with PySCF on water, cc-pVDZ:

| Quantity | Difference |
| --- | --- |
| HF energy | 1e-9 eV |
| HF forces | 5e-6 eV/A |
| MP2 energy | 4e-8 eV |
| MP2 forces | 1e-4 eV/A |
| CCSD(T) energy | 4e-7 eV |
| B3LYP energy, (99, 590) grid against PySCF grid level 9 | 2e-7 eV |
| O-H bonds after relaxation with each engine | within 2e-4 A |

Both codes' `b3lyp` uses the VWN-RPA correlation of the original B3LYP.
PySCF's `b3lyp5` (VWN5, the convention of some other programs) differs by
1.01 eV for water, so the functional variant must be recorded when comparing
with literature.

## IR and Raman spectra

`vibrational_spectra(molecule, method, basis, raman=False)` returns harmonic
wavenumbers, IR intensities (km/mol) and, with `raman`, Raman activities
(A^4/amu) and depolarization ratios. Normal modes come from PySCF's analytic
Hessian, projected onto the space orthogonal to rigid translations and
rotations. Dipole derivatives are central differences of the SCF dipole over
Cartesian displacements; polarizability derivatives use finite fields at every
displaced geometry. The geometry must be a stationary point; otherwise the
calculation is refused. The atomic-polar-tensor sum rule (the sum over atoms
equals the charge times the identity) is reported as a check on the
differences.

Validation:

- Water, HF/def2-SVP: wavenumbers agree with ASE's independent
  finite-difference `Infrared` module to 0.01 cm^-1 and intensities to 0.004%.
- CO2 obeys the mutual exclusion rule: the symmetric stretch is Raman active
  and IR silent, and the bend and antisymmetric stretch are IR active and
  Raman silent.
- Results are invariant under rotation of the molecule. The B2 stretch of
  water has a depolarization ratio of exactly 0.75.
- Water, B3LYP/aug-cc-pVDZ: 1619, 3794 and 3904 cm^-1 against experimental
  harmonic values of about 1649, 3832 and 3943; intensities 71, 4 and 61
  km/mol against about 54, 2 and 45 measured (DFT overestimates them, as
  usual); Raman activity of the symmetric stretch 100.5 A^4/amu with
  depolarization ratio 0.06.

Anharmonicity, overtones and line shapes are not computed.

## UV-Vis absorption

`absorption_spectrum(excitations, energies_eV, fwhm_eV)` turns the vertical
excitations of `excited_states` (PySCF) or `Psi4Potential.excited_states`
into a molar absorption coefficient in L/(mol cm). Each band is a Gaussian in
wavenumber whose integral is 2.3148e8 f L mol^-1 cm^-2; the unit test checks
that integral to 1e-6. The width is a model parameter, not a computed line
shape.

TDDFT and TDA from the two engines agree for formaldehyde, B3LYP/def2-SVP on
fine grids: excitation energies to 1e-5 eV and oscillator strengths to 1e-5
(3.878, 8.241, 8.855 eV with f = 0, 0.127, 0.002).

## NMR shieldings and chemical shifts

`nmr_shielding(molecule, method="hf", basis="pcseg-2")` returns GIAO shielding
tensors, isotropic values and anisotropies for every nucleus of a
closed-shell molecule, through the PySCF properties extension
(`pip install pyscf-properties`, Apache-2.0). `chemical_shifts(molecule,
reference)` gives `sigma_ref - sigma` against a reference molecule computed at
the same level. Version 0.1.0 of the extension assumes the coupled-perturbed
solver always passes three trial vectors, which PySCF 2.14 does not.
Materia supplies the same induced-potential function for any number of
vectors through the hook PySCF provides, and the test checks that the result
equals the original code wherever the original runs (to 1e-6 ppm).

HF at B3LYP/def2-TZVP geometries against gas-phase absolute shieldings:

| Nucleus | pcseg-2 | pcseg-3 | Experiment |
| --- | ---: | ---: | ---: |
| CH4 13C | 193.9 | 193.3 | 195.1 |
| CH4 1H | 31.5 | 31.4 | 30.6 |
| NH3 15N | 263.0 | 259.5 | 264.5 |
| H2O 17O | 325.8 | 323.1 | 323.6 |

Shieldings do not change when the molecule is moved or rotated (to the
coupled-perturbed convergence, 1e-4 ppm), equivalent nuclei agree, and the
tetrahedral carbon has zero anisotropy. No relativistic, vibrational or
solvent corrections are applied.

## Core-level (XPS) binding energies

`core_binding_energy(molecule, atom, basis="cc-pcvtz", xc="tpss")` computes the
1s binding energy of one atom as `E(cation with a 1s hole) - E(neutral)`. The
core orbitals are rotated so that one has the largest Mulliken population on
the chosen atom, so a hole can be placed on one of several equivalent atoms;
the cation is solved with the maximum overlap method and checked afterwards to
still hold the hole. Hydrogen receives cc-pVXZ because it has no core-valence
set. Hydrogen and helium are refused, and so are open-shell ground states.

TPSS/cc-pCVTZ at TPSS/def2-TZVP geometries, nonrelativistic, against
gas-phase experiment (Jolly, Bomben and Eyermann 1984):

| Molecule, atom | Materia (eV) | Experiment (eV) | Difference |
| --- | ---: | ---: | ---: |
| CH4, C | 290.80 | 290.84 | -0.04 |
| NH3, N | 405.33 | 405.52 | -0.19 |
| H2O, O | 539.34 | 539.90 | -0.56 |
| CO, C | 296.11 | 296.23 | -0.12 |
| CO, O | 542.14 | 542.55 | -0.41 |
| C2H2, C (either atom) | 291.17 | 291.14 | +0.03 |

Scalar-relativistic corrections, about +0.1 eV for C and +0.35 eV for O, are not
added and account for most of the oxygen difference. The chemical shift
between CO and CH4 is 5.32 eV against 5.39 eV measured. The frozen-orbital
estimate (the Kohn-Sham orbital energy) is about 20 eV too low, which is why
the relaxed Delta-SCF calculation is needed.

## Occupation changes (Delta-SCF)

`delta_scf(molecule, hole=0, particle=0)` moves one electron from HOMO-hole to
LUMO+particle and holds that occupation through the SCF with the maximum
overlap method (Gilbert, Besley and Gill 2008), then forms the singlet by spin
purification from the mixed determinant and the high-spin triplet (Ziegler,
Rauk and Baerends 1977). Formaldehyde n to pi*, B3LYP/def2-SVP: triplet
3.21 eV, purified singlet 3.53 eV, against 3.88 eV from TDDFT; triplet
<S^2> = 2.005.

## QM/MM

`materia.physics.qmmm.QMMMPotential(qm_atoms, mm_molecules, qm=PySCFPotential(...))`
embeds whole QM molecules in rigid TIP3P water with electrostatic embedding
through `pyscf.qmmm`; QM-MM Lennard-Jones contacts and MM-MM Coulomb plus
Lennard-Jones complete the energy. Forces on QM atoms and MM charges are
analytic. No covalent bond may cross the boundary (no link atoms); MM atoms
should be fixed in relaxation and dynamics because no intramolecular MM terms
are included; periodic MM is not implemented.

| Water dimer, QM donor, TIP3P acceptor, B3LYP/def2-SVP | Result |
| --- | --- |
| QM/MM interaction | -6.02 kcal/mol |
| Full QM interaction (no counterpoise) | -7.13 kcal/mol |
| CCSD(T) complete-basis benchmark | about -5.0 kcal/mol |
| Forces against finite differences | MM atoms 4e-8, QM atoms 2e-4 eV/A |

## Refusals

* An engine whose package is missing refuses with the install command.
* PySCF and RDKit refuse periodic structures; tblite refuses slabs and wires.
* An electron count inconsistent with the spin (PySCF `spin`, tblite
  `unpaired`) is refused rather than guessed.
* MMFF94 refuses molecules it has no parameters for and names UFF instead.
* An SCF that does not converge raises an engine failure; no energy is kept.

## Checked so far

| Check | Result |
| --- | --- |
| MMFF94 and UFF energies against RDKit called directly | equal to 1e-10 eV |
| GFN2-xTB energy and forces against tblite called directly | equal to 1e-9 |
| B3LYP/def2-SVP water against PySCF called directly | equal to 1e-7 eV |
| HF/STO-3G H2 at 1.4 bohr | -1.11671 Eh (Szabo and Ostlund: -1.1167) |
| Forces against finite differences (MMFF94, UFF, xTB, MP2) | within 1e-6, 2e-4 and 1e-4 eV/A |
| HF/STO-3G water vibrations from Materia's finite displacements against PySCF's analytic Hessian | within 0.05 cm^-1 |
| GFN2-xTB ammonia inversion through Materia's NEB | symmetric, 0.265 eV against about 0.25 eV measured |
| Amber14/GBn2 energy of Trp-cage (PDB 1L2Y, 304 atoms) against OpenMM called directly | equal to 4e-13 eV |
| OpenMM forces against finite differences (Amber14/GBn2, CHARMM36, Amber99SB-ILDN) | within 1.3e-7 eV/A |
| OpenMM Langevin dynamics of Trp-cage at 300 K | mean temperature 264 to 295 K over 4 ps |
| ALPB and GBSA energies against tblite called directly | equal to 1e-9 eV |
| PCM forces against finite differences | within 1e-6 eV/A |
| TDA excitations against PySCF called directly | equal to 1e-5 eV |
| Solvated Trp-cage PME energy against OpenMM called directly | within 1e-4 eV on the CPU platform |

## Limitations

* Biomolecules need a complete PDB topology the chosen force field covers;
  ligands and non-standard residues need their own parameters (not provided).
* Molecular engines only: no periodic Gaussian-basis or plane-wave work
  through PySCF here, no explicit-solvent QM, no excited-state gradients, no
  relativistic Hamiltonians.
* CCSD and CCSD(T) give energies only; gradients are HF, DFT and MP2.
* Force-field topology is fixed at construction: MMFF94 and UFF cannot
  describe bond breaking. Use xTB or PySCF for reactions.
* Results are as accurate as the method and basis chosen; nothing is
  extrapolated to the basis-set limit.
