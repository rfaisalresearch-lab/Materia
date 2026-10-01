# Third-party engines, libraries and data

Materia drives mature open-source codes rather than re-implementing them. This
file lists every one, its licence, how Materia uses it and what has to be
cited. Licences were read from each installed package's own metadata on
1 October 2026. Nothing listed here is copied into Materia's source tree
except the AME2020 mass table, shipped unchanged with its checksum.

## Python libraries imported in-process (optional)

| Package | Version tested | Licence | Used for | Cite |
| --- | --- | --- | --- | --- |
| ASE | 3.29.0 | LGPL-2.1-or-later | structure I/O, phonon force constants, band paths | A. H. Larsen et al., J. Phys.: Condens. Matter 29 (2017) 273002 |
| PySCF | 2.14.0 | Apache-2.0 | HF, DFT, MP2, CCSD(T), TDDFT, Hessians, solvation, QM/MM, fields, core holes | Q. Sun et al., J. Chem. Phys. 153 (2020) 024109 |
| pyscf-properties | 0.1.0 | Apache-2.0 | GIAO NMR shieldings | PySCF developers |
| RDKit | 2026.03 | BSD-3-Clause | SMILES, conformers, MMFF94 and UFF | RDKit, https://www.rdkit.org |
| tblite | 0.7.0 | LGPL-3.0-or-later | GFN1/GFN2-xTB, ALPB and GBSA | C. Bannwarth et al., J. Chem. Theory Comput. 15 (2019) 1652 |
| OpenMM | 8.6.1 | MIT for the API and CPU platforms, LGPL for GPU platforms (OpenMM's licence) | biomolecular force fields and dynamics | P. Eastman et al., PLoS Comput. Biol. 13 (2017) e1005659 |
| PDBFixer | 1.12.0 | MIT | completing PDB structures | OpenMM project |
| radioactivedecay | 0.6.1 | MIT; bundles ICRP-107 decay data under that data's own terms | decay chains and inventories | A. Malins and T. Lemoine, J. Open Source Softw. 7 (2022) 3318 |
| particle | 1.0.1 | BSD-3-Clause | PDG particle properties | Scikit-HEP particle |
| DEVSIM | 2.11.0 | Apache-2.0 | drift-diffusion device simulation | J. E. Sanchez, DEVSIM, https://devsim.org |
| matscipy | 1.3.0 | LGPL-2.1 | dislocation construction | matscipy, https://github.com/libAtoms/matscipy |
| XrayDB | 4.5.8 | MIT | form factors, anomalous terms, attenuation, edges | M. Newville, https://github.com/xraypy/XrayDB, and the tables it holds (Waasmaier-Kirfel, Chantler, Elam) |
| phonopy | 4.7.2 | BSD-3-Clause | validation reference for quasi-harmonic results only | A. Togo, J. Phys. Soc. Jpn. 92 (2023) 012001 |
| spglib | 2.7.0 | BSD-3-Clause | dependency of phonopy | A. Togo et al., Sci. Technol. Adv. Mater. Methods 4 (2024) 2384822 |

## Programs run in separate processes

GPL programs are run as separate executables or worker processes and are not
linked into Materia.

| Program | Licence | Used for |
| --- | --- | --- |
| LAMMPS (22 Jul 2025) | GPL-2.0 | classical MD and parity with the native EAM and Tersoff |
| Quantum ESPRESSO 7.5 (pw.x, ph.x, q2r.x, matdyn.x, dynmat.x, projwfc.x) | GPL-2.0 | plane-wave DFT, DFPT phonons, projected DOS and fat bands |
| GPAW | GPL-3.0 | DFT ground state, relaxation, bands, LDOS and STM |
| Psi4 1.9.1, in the `materia-psi4` environment | LGPL-3.0 | second quantum-chemistry engine for parity with PySCF; cite D. G. A. Smith et al., J. Chem. Phys. 152 (2020) 184108 |
| CP2K 7.1 | GPL-2.0 | adapter written, not validated; parked |
| MACE (mace-torch 0.3.16) with PyTorch, in the `materia-ml` environment | MIT (MACE and MACE-MP-0 weights); PyTorch BSD-style | machine-learned interatomic potential |
| pymatgen 2026.9.24, in the `materia-ml` environment | MIT | validation reference for powder diffraction only |

## Data downloaded at run time, never shipped

| Data | Source | Terms |
| --- | --- | --- |
| PSlibrary 1.0.0 pseudopotentials | pseudopotentials.quantum-espresso.org | PSlibrary's licence; cite A. Dal Corso, Comput. Mater. Sci. 95 (2014) 337 |
| LAMMPS potential files (Tersoff, Mishin Cu and others) | LAMMPS GitHub repository | distributed with LAMMPS under GPL-2.0; cite each file's original paper |
| MACE-MP-0 small and medium weights | ACEsuit releases | MIT; cite I. Batatia et al., arXiv:2401.00096 |

## Data shipped

| Data | Source | Notes |
| --- | --- | --- |
| AME2020 atomic mass table (`mass_1.mas20.txt`) | IAEA Atomic Mass Data Center | unchanged, SHA-256 checked; cite M. Wang et al., Chin. Phys. C 45 (2021) 030003 |
| Trp-cage NMR model (`tests/data/1l2y_model1.pdb`) | RCSB PDB 1L2Y | CC0; test data only |
