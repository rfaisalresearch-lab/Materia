# Elastic constants

`materia.physics.elasticity.elastic_tensor(potential, structure)` returns the
6x6 elastic tensor in GPa from second derivatives of the energy with respect
to homogeneous strain, for any Materia potential: classical, EAM, rigid-ion,
GFN-xTB, Quantum ESPRESSO or an ASE calculator wrapped as one.

* Engineering shear strains, Voigt notation, central differences.
* Refused when the cell carries more than 0.1 GPa of stress: energy
  derivatives away from zero stress are not the stress-strain elastic
  constants.
* `relax_internal=True` relaxes internal coordinates at every strain
  (needed for diamond, wurtzite and other multi-atom bases).
* Voigt, Reuss and Hill bulk and shear moduli, Born stability.
* Every tensor is recomputed at twice the strain. If any significant
  component changes by more than 2 percent the result is labelled estimated.
  Tabulated potentials need this: for the Zhou copper EAM file, a strain of
  1e-3 gives C11 = 279 GPa from spline kinks, while 1e-2 gives 170.9 GPa,
  stable to 0.6 percent against 2e-2.

## Checked

| Case | Result |
| --- | --- |
| fcc Lennard-Jones at zero stress against the Born lattice sum (Born and Huang 1954) | C11 and C12 = C44 equal to 1e-4 relative; the Cauchy relation holds |
| Cu, Zhou 2004 EAM, against LAMMPS energies of the same strained cells with the same file | equal to 0.05 GPa |
| Cu, Zhou 2004 EAM | C11 170.9, C12 122.0, C44 76.5 GPa (measured near 0 K: about 176, 125 and 82 GPa) |
