# Cross-engine consistency

Materia drives several independent codes. Where they model the same physics
they should agree, and where they disagree the difference should be
explainable. `tests/validation/test_cross_engine.py` and the parity tests in
each engine's documentation check this.

## Silicon, PBE equation of state

| Engine (driven by Materia) | V0 (A^3/atom) | B0 (GPa) |
| --- | --- | --- |
| GPAW, PAW, 500 eV, 12^3 k | 20.520 | 88.45 |
| Quantum ESPRESSO, PSlibrary USPP, 45/360 Ry, 8^3 k | 20.455 | 88.74 |
| WIEN2k all-electron (Delta project reference) | 20.453 | 88.5 |
| MACE-MP-0 small (machine-learned PBE surrogate) | 20.398 | 72.8 |

GPAW and Quantum ESPRESSO agree to 0.3 percent in volume and bulk modulus;
Quantum ESPRESSO matches the all-electron volume to 0.01 percent. MACE-MP-0
gets the volume within 0.3 percent but its bulk modulus is 18 percent soft,
consistent with its 25 percent soft optical phonon.

## Other cross-checks in this branch

| Pair | Quantity | Agreement |
| --- | --- | --- |
| QE DFPT (ph.x) vs Materia finite displacements on QE forces | Si Gamma optical phonon | 0.03 percent |
| PySCF analytic Hessian vs Materia finite displacements on PySCF forces | H2O HF/STO-3G frequencies | 0.05 cm^-1 |
| ASE Vibrations vs Materia phonons | EMT Cu19 cluster, 57 modes | 3.5e-8 THz |
| LAMMPS vs Materia | Tersoff Si and SiC energies and forces | 1e-11 eV, 2e-10 eV/A |
| LAMMPS vs Materia | Cu EAM elastic tensor | 0.05 GPa |
| tblite, PySCF, RDKit, OpenMM called directly vs through Materia | energies | 1e-7 eV or better |
