# STM spectroscopy and tip models (GPAW)

`materia/experiments/dft/stm_spectroscopy.py` extends the GPAW LDOS experiment
(`docs/DFT_LDOS.md`) without changing its frozen specification. One
non-self-consistent GPAW step can now also return:

* **A p-wave tip map** by Chen's derivative rule (C. J. Chen, Phys. Rev. B 42
  (1990) 8841): the sum over states in the window of |dpsi/dx|^2 +
  |dpsi/dy|^2, the standard first model of a CO-terminated tip (Gross et al.
  2011). Derivatives are central differences of the pseudo-wavefunction on the
  coarse grid, converted to Cartesian axes for any cell. `ldos.stm_image`
  takes constant-height or constant-current images of it.
* **An energy-resolved LDOS stack** with Gaussian broadening at chosen
  energies about the Fermi level. Slices are dI/dV maps and
  `point_spectrum` gives dI/dV-proportional spectra at any tip position, in
  the Tersoff-Hamann low-bias picture.

Each stack slice is checked against the broadened state count computed
independently from the returned eigenvalues, and the run is refused if the
computed bands do not reach the highest requested energy plus five widths.

## Checked (hydrogen atom, PBE, live GPAW)

| Check | Result |
| --- | --- |
| p-tip image of the 1s state 3 A above the atom | a ring with zero on the axis (5e-26 of its maximum), radius 1.35 A against 1.27 A predicted from the measured decay constant kappa = 1.67 1/A |
| Point spectrum above the atom against the eigenvalue DOS | peaks at the same energy; slice integrals match the state count |

## Limitations

* Tip models: s and p (p_x + p_y) only; no d tips, no explicit tip
  electronic structure, no tip relaxation or tip-sample forces (the
  probe-particle model is not implemented).
* Pseudo-wavefunctions: exact outside the PAW spheres only.
* Gamma-point or k-sampled Kohn-Sham states; no quasiparticle corrections; no
  current in amperes.
