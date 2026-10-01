# Harmonic vibrations and Gamma-point phonons

Materia computes harmonic force constants by finite displacements and
diagonalises the mass-weighted matrix to give normal modes: frequencies,
eigenvectors, Cartesian displacement patterns, reduced masses, imaginary
modes and a zero-point energy where one is defined. For a periodic cell it
gives the Gamma-point modes of that cell and nothing else: there is no
dispersion.

Code: `materia/physics/phonons.py` (the method),
`materia/solvers/classical.py` (`ClassicalSolver.phonons`, task `"phonons"`).
Tests: `tests/unit/test_phonons.py`, `tests/integration/test_phonons_classical.py`,
`tests/validation/test_phonons_validation.py`.

## Force sources

`harmonic_analysis(structure, source, settings)` accepts:

| Source | How it is called | Fidelity recorded |
| --- | --- | --- |
| A Materia potential (Stillinger-Weber, EAM, Lennard-Jones, rigid-ion, Harmonic) | `energy_and_forces` on a working copy | the potential's |
| An ASE-style calculator (any object with `get_forces(atoms)`) | on an ASE `Atoms` built from the structure, with its masses | non-physical unless the caller states one |
| A plain callable `positions (N, 3) in A -> forces in eV/A` | directly | must be given with a model name |

The structure passed in is never changed. Every classical solver now declares
the `phonons` capability and runs it as `solver.phonons(structure, settings)`
or `solver.run(structure, task="phonons")`.

From Materia's public Python workspace, use `surface.phonons(stencil=4)`,
`lab.phonons(surface, stencil=4)` or `phonons(surface, stencil=4)`. The run
returns all result records and saves the numerical arrays in checksummed
project storage under `phonons::frequencies`, `phonons::force_constants`,
`phonons::eigenvectors` and `phonons::displacements`.

Pass `temperatures_K=[0, 100, 200, 300]` to also calculate the harmonic
Helmholtz free energy, internal energy, entropy and constant-volume heat
capacity. The zero-temperature values use the zero-point energy exactly.
Imaginary modes and non-stationary reference structures return explicit
unsupported thermodynamic records because they do not define an equilibrium
harmonic state. Periodic values are for the modeled cell at Gamma only and
are labelled as such rather than presented as Brillouin-zone-converged bulk
thermodynamics.

## Method

Every free Cartesian coordinate is displaced by `+-h` (two-point stencil) or
`+-h, +-2h` (four-point stencil). Force constants are
`Phi_ij = -dF_i / du_j` in eV/A^2. The measured asymmetry `|Phi - Phi^T|` is
reported and the symmetric part is kept.

**Acoustic sum rule.** A translation-invariant model has
`sum_j Phi_ij = 0` along each direction. The rule is restored by the
orthogonal projection `Phi' = P Phi P`, `P = 1 - V V^T`, with `V` an
orthonormal basis of rigid translations. The result stays symmetric and meets
the rule exactly. `sum_rule="translational+rotational"` adds rigid rotations
to `V`, and is allowed only for isolated systems at a stationary point.
The projection is refused, and `sum_rule="none"` must be chosen, when:

* the measured violation exceeds `asr_tolerance` (default 2 percent) of the
  largest diagonal force constant, which means the model itself is not
  translation invariant (atoms tied to fixed points, as in an Einstein solid);
* any atom is fixed, since a partial Hessian has no sum rule.

**Normal modes.** `D = M^-1/2 Phi M^-1/2`, eigenvalues `omega^2` in
eV/(A^2 u). Frequencies are reported in THz, cm^-1 (x 33.35641) and meV
(x 4.135668). Negative eigenvalues are imaginary frequencies and are reported
as negative numbers. Displacement patterns are `M^-1/2 e` normalised to unit
length. The reduced mass is `1 / sum_i (e_i^2 / m_i)`, the Wilson, Decius and
Cross convention used by quantum chemistry codes, for which
`omega^2 = (l . Phi l) / mu` with `l` the unit displacement; for a homonuclear
diatomic it is the atomic mass, not the two-body reduced mass. The
participation ratio is `1 / (N sum_a |e_a|^4)`.

**Resolution and classification.** The eigenvalue resolution is the largest
of: the spectral norm of the change the symmetrisation and sum rule made to
`D` (by Weyl's inequality no eigenvalue of the symmetric part moved by more),
the spectral norm of the antisymmetric part of the raw matrix, and the
stencil's truncation error `(h / 1 A)^p` of the largest eigenvalue. Each mode
is classified as:

| Kind | Meaning |
| --- | --- |
| `rigid-body` | lies in the space of rigid translations (and, for an isolated system, rotations) to better than 0.999, with an eigenvalue within the resolution |
| `zero-within-resolution` | eigenvalue within the resolution, not a rigid-body motion |
| `imaginary` | negative eigenvalue beyond the resolution: the structure is not at a minimum along this mode |
| `real` | positive eigenvalue beyond the resolution |

**Stationarity.** Normal modes are defined about a stationary point. If the
largest force on a free atom exceeds `max_residual_force_eV_A` (default
5e-3 eV/A) the analysis is refused. With `allow_nonstationary=True` it runs,
every result is labelled `estimated` and no zero-point energy is given.

**Zero-point energy.** Half the sum of `hbar omega` over real modes. It is
refused when any mode is imaginary or the structure is not stationary.

**Fixed atoms** are not displaced and are treated as infinitely heavy (a
partial Hessian); the record says so.

## Periodic cells and what is refused

For a periodic cell the modes are those at the Gamma point of the given cell.
When the cell is a supercell of a primitive cell they are the phonons at
every wavevector commensurate with the supercell, folded to Gamma. Materia
does not identify the primitive cell, unfold modes or label them by
wavevector. Force constants between an atom and its own periodic images are
summed, which is exactly what the Gamma point of the periodic crystal needs;
when the cell is narrower than twice the model's cutoff the log says so.

* **Dispersion is refused.** Any `q_point` other than Gamma (or an equivalent
  reciprocal lattice vector) raises `PhononRefused`: dispersion needs force
  constants resolved by lattice vector and Fourier interpolation, which are
  not implemented.
* **No non-analytic correction.** For a polar crystal the Gamma modes are
  those of the analytic part, the transverse-optical limit. Longitudinal
  optical frequencies and LO-TO splitting need Born effective charges and the
  dielectric tensor, which are not computed.
* **The zero-point energy of a periodic cell is a Gamma-point sample**, not a
  Brillouin-zone integral, and no thermodynamic free energy is computed.

## Results and provenance

`PhononResult.results()` gives Materia `Result` records:

| Key | Unit | Contents |
| --- | --- | --- |
| `frequencies` | THz | all modes, negative = imaginary; `extra` holds the mode table, kinds and diagnostics; the uncertainty is the frequency resolution, kind `bound` |
| `force_constants` | eV/A^2 | the final symmetric matrix over free atoms, atom-major x y z |
| `eigenvectors` | none | mass-weighted, columns are modes |
| `displacements` | none | (mode, atom, xyz), unit norm |
| `zero_point_energy` | eV | or an unsupported record saying why |
| `helmholtz_free_energy_eV` | eV | harmonic free energy at each requested temperature |
| `internal_energy_eV` | eV | harmonic internal energy at each requested temperature |
| `entropy_eV_K` | eV/K | harmonic vibrational entropy |
| `heat_capacity_eV_K` | eV/K | harmonic constant-volume heat capacity |

The provenance model is `phonons/finite-displacement[<force model>]`, with the
force model's fidelity, `calculated` at a stationary point and `estimated`
otherwise. It records every setting, the masses, the free atom ids, the
force model's parameters, an input digest, the approximations above and the
references. `stored_arrays()` returns the force constants, frequencies and
eigenvectors as checksummable `StoredArray` objects for project storage.
Cancellation through `progress` or `cancelled` raises `PhononCancelled` and
returns nothing; the solver hook turns refusals and cancellations into
unsupported results.

## Validation

| Case | Reference | Measured | Tolerance |
| --- | --- | --- | --- |
| Lennard-Jones dimer, 4-point stencil | `omega = sqrt(k / mu)`, `k = 36 2^(2/3) epsilon / sigma^2` | agree | 1e-7 relative |
| Heteronuclear Lennard-Jones dimer (Ar-Kr, Lorentz-Berthelot) | same, with two-body reduced mass; amplitude ratio `m2 / m1` | agree | 1e-7 relative |
| fcc Lennard-Jones crystal, 32-atom supercell, 96 modes | Born-von Karman dynamical matrices at the 32 commensurate wavevectors, lattice sum written independently | agree to 2.4e-8 THz | 1e-6 THz |
| Einstein solid (Harmonic potential), 12 modes | `sqrt(k / m)`; translational sum rule refused | agree | 1e-9 relative |
| Analytic saddle, one atom | one imaginary mode `sqrt(k_y / m)`, zero-point energy refused | agree | 1e-9 relative |
| Relaxed EMT Cu19 cluster, 57 modes | ASE `Vibrations`, an independent implementation of the same differences | agree to 3.5e-8 THz | 1e-6 THz |

Model properties measured here, reported and not validated:

* Stillinger-Weber silicon: Gamma optical mode 17.832 THz, triply degenerate,
  the same in the 2-atom primitive and 8-atom conventional cells; the measured
  Raman frequency of silicon is about 15.5 THz, so this potential overestimates
  it by about 15 percent. The conventional cell also gives the X-point modes
  folded to Gamma (6.652, 12.993 and 15.628 THz).
* BKS alpha-quartz relaxed at the measured cell: 27 Gamma modes, 3 rigid-body,
  no imaginary mode, the highest at 1228 cm^-1, all transverse-optical limit.

## Limitations

* Harmonic only: no anharmonic shifts, lifetimes, thermal expansion or
  temperature-dependent frequencies.
* Gamma point of the given cell only; no dispersion, density of states,
  unfolding, group velocities or thermal conductivity.
* No non-analytic (LO-TO) correction for polar crystals.
* Classical force models and ASE calculators only. GPAW is not yet a force
  source for this analysis: each force call would be a full self-consistent
  calculation run as a separate job, which the DFT experiment system does not
  yet schedule.
* Dense matrices: at most 1500 free atoms.
* Symmetry is not used to reduce the number of displacements, so a cell of `N`
  free atoms costs `6N + 1` force calls with the two-point stencil and
  `12N + 1` with the four-point stencil.
