# Reaction rates

`materia.physics.kinetics.harmonic_tst_rate(minimum, saddle, barrier, T)`
turns two phonon analyses (from `materia.physics.phonons`) and a barrier
(from NEB or single points) into a rate constant by harmonic
transition-state theory (Vineyard 1957), with optional Kramers friction and
Wigner tunnelling corrections. It refuses a "minimum" with an imaginary mode,
a saddle without exactly one, mode counts that do not differ by one, and
modes indistinguishable from zero.

Any Materia potential can supply the frequencies: classical, EAM, Tersoff,
GFN-xTB, PySCF, Quantum ESPRESSO or MACE.

## Checked

| Case | Result |
| --- | --- |
| Prefactor for a quartic double well | equal to the analytic attempt frequency to 1e-6 |
| Langevin dynamics on the same well, Eb/kT = 5, friction 1e13/s, 3 ns | 381 well-to-well transitions, measured rate 1.27e11/s against 1.41e11/s from Kramers-corrected harmonic TST (ratio 0.90) |

## Limitations

* Harmonic TST: no recrossing beyond Kramers' factor, no anharmonic
  partition functions, classical vibrations.
* Kramers' factor assumes the spatial-diffusion regime; at very weak friction
  the energy-diffusion regime is slower and is not modelled.
* Wigner's correction is refused once it exceeds 1.5; deep tunnelling needs
  instanton or path-integral methods, which are not provided.
