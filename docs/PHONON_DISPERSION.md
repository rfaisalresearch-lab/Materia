# Periodic phonon dispersion and density of states

Materia can build real-space harmonic force constants for a fully periodic primitive
cell, Fourier-interpolate a phonon dispersion on an explicit reciprocal-space path,
and sample a phonon density of states on an explicit Monkhorst-Pack mesh. The backend
uses ASE 3.29 phonon machinery. Materia classical potentials are connected through an
ASE calculator adapter that evaluates the original potential on each repeated
supercell without modifying the input structure.

The public entry points are `surface.phonon_dispersion(...)`,
`lab.phonon_dispersion(...)`, the top-level script function
`phonon_dispersion(...)`, and `materia.physics.periodic_phonon_analysis`. Its
`PeriodicPhononSettings` value is frozen and requires `supercell`, `q_path`, and
`dos_mesh`; no crystal path or convergence parameter is guessed. The classical solver
also accepts `task="phonon_dispersion"` or `task="phonon_dos"` with those settings.

## Numerical contract

The reference supercell must be stationary within the requested force tolerance. A
nonstationary calculation is refused by default. If explicitly allowed, it is marked
estimated and does not claim to be a vibrational spectrum about equilibrium.

The implementation uses two-point central finite differences. It reads the uncorrected
real-space force constants first and refuses a force model whose acoustic sum-rule
violation exceeds the requested relative tolerance. Acoustic enforcement is explicit
in the settings. When enabled, translational Gamma modes that lie within the measured
resolution are reported as exactly zero and classified as acoustic.

Negative frequencies are retained. They represent imaginary modes and indicate a
negative eigenvalue of the dynamical matrix. The DOS uses the same signed-frequency
convention, so an unstable structure can have spectral weight below zero.

For a Materia potential with a known finite cutoff, a supercell whose narrowest
periodic width is not greater than twice that cutoff is refused. Such a cell cannot
resolve the real-space interaction range without image aliasing. For external ASE
calculators, Materia cannot infer the force range, so the caller remains responsible
for supercell convergence.

The result provides checksummed `StoredArray` objects for the path, frequencies,
per-q resolution, real-space force constants and lattice vectors, raw DOS samples,
weights, the broadened DOS grid, and density. Its provenance includes the complete
settings, force-model parameters, ASE version, input digest, tolerances, boundary
conditions, and method references. Temporary displacement files are deleted whether
the calculation succeeds, fails, or is cancelled.

## Hard limits

- Primitive cells: 256 atoms.
- Repeated supercells: 8,192 atoms.
- Explicit path: 20,000 q points.
- DOS mesh: 65,536 q points.
- Each path or DOS frequency array: 5,000,000 values.
- Broadened DOS grid: 20,000 points.

## Limitations

- Only fully three-dimensional periodic cells are supported. Slab and wire dispersion
  are not implemented.
- Fixed atoms are refused because a partial crystal Hessian does not preserve lattice
  translation symmetry.
- There is no non-analytic dipole correction. Born effective charges and dielectric
  tensors are not computed, so polar crystals have no LO-TO splitting.
- Real-space force constants are truncated by the chosen supercell. The caller must
  converge the supercell, finite-displacement step, q path, and DOS mesh.
- Gaussian broadening is a presentation of the sampled DOS, not additional physical
  lifetime information.
- Branches are sorted by frequency independently at each q point. Crossings are not
  tracked by eigenvector overlap.
- The harmonic approximation includes no anharmonic shifts, linewidths, thermal
  expansion, isotope disorder, or quantum nuclear dynamics.

## Validation gates

The independent validation suite contains two physics checks. Silicon at Gamma under
the Stillinger-Weber potential is compared with Materia's separate full-cell harmonic
Hessian implementation. An fcc Lennard-Jones path is compared against a direct
Born-von Karman lattice sum built from the analytic central-force Hessian. These tests
must pass before the feature is described as scientifically validated.

The backend and validation gates are built, but the tests have not yet been
run because the current feature push defers the combined test phase.
