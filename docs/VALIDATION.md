# Validation report

**A passing software test is not scientific validation.** This project keeps
the two apart:

* `tests/unit`, `tests/integration`, `tests/performance`, *implementation*
  tests. They show the code does what the model says.
* `tests/validation`, *physical validation*. Each case compares a computed
  result against an analytic value, a published measurement, or a conservation
  law, and states its tolerance and why that tolerance is right.

Run them separately:

```bash
pytest tests/unit tests/integration        # implementation
pytest -m validation                       # physical validation
```

The desktop interface also contains **Help ▸ Run interface self-check**. It
verifies the live local service, material library, required laboratory panels,
all four view switches, visible canvas dimensions, and uncaught browser errors.
This is an engineering smoke check, not scientific validation.

## Summary

| Area | Cases | Status |
| --- | ---: | --- |
| Crystallography | 9 | pass |
| Classical potential | 6 | pass |
| Tight binding | 8 | pass |
| Scanning tunnelling microscopy | 4 | pass |
| Atomic force microscopy | 3 | pass |
| Surface reconstruction | 17 | pass |
| GPAW first principles | 6 | pass (skipped without GPAW) |
| Ground-state DFT experiments | 8 | pass (skipped without GPAW) |
| DFT relaxation (fixed and variable cell) | 4 | pass with GPAW 25.7.0 here (skipped as BLOCKED without GPAW) |
| DFT DOS and PDOS | 5 | pass with GPAW 25.7.0 here (skipped as BLOCKED without GPAW) |
| DFT band structure | 5 | pass with GPAW 25.7.0 here (skipped as BLOCKED without GPAW) |
| DFT LDOS and Tersoff-Hamann images | 4 | pass with GPAW 25.7.0 here (skipped as BLOCKED without GPAW) |
| Reference benchmarks (data, structures, independent implementations, model properties) | 39 | pass; see ACCURACY_BENCHMARKS.md |
| PBE equations of state against all-electron references, and the equation-of-state tool (Si, Al, Cu) | 5 | pass with GPAW 25.7.0 here (skipped as BLOCKED without GPAW) |
| Point-charge electrostatics | 11 | pass |
| Rigid-ion potentials (Born-Mayer closed form, BKS quartz) | 6 | pass |
| Harmonic vibrations (dimer, fcc lattice sum, Einstein solid, ASE Vibrations) | 4 | pass |
| Embedded-atom metals (Cu, Au, W) | 29 | pass |
| LAMMPS live parity (Cu) | 3 | **blocked**: no LAMMPS installed; skipped, not passed |

Every case that does not require an external solver runs on every test
invocation. GPAW cases run when both GPAW and its PAW datasets are available
and otherwise state the exact reason for the skip. No tolerance is widened to
make a case pass. Where a tolerance is loose, the reason is stated.

The complete repository suite on 2026-09-30 reported 1,448 passed, 3 skipped
and 48 warnings in 1110.32 seconds. The three skips were the live LAMMPS parity
cases below. GPAW ground-state, relaxation, DOS and band-structure validation
ran and passed.

---

## Crystallography

Exact arithmetic, so the tolerances are numerical only.

| Case | Expected | Tolerance |
| --- | --- | --- |
| Nearest-neighbour distance and coordination, diamond cubic (Si, Ge, C) | a√3/4, coordination 4 | 1 part in 10⁹ |
| Same, face-centred cubic (Au, Cu) | a/√2, coordination 12 | 1 part in 10⁹ |
| Same, body-centred cubic (W) | a√3/2, coordination 8 | 1 part in 10⁹ |
| Interplanar spacings for (100), (110), (111), (200), (311) | a/√(h²+k²+l²) | 1 part in 10⁹ |
| Density from the generated cell vs the tabulated value, eight materials | agreement | 1 % |
| Graphene C-C bond length | 1.42 Å | 0.01 Å |
| α-Al₂O₃ Al-O distances | 1.856 and 1.971 Å (Ishizawa et al., Acta Cryst. B 36, 228) | 0.02 Å |
| α-SiO₂ Si-O distances | 1.605 and 1.614 Å (Levien et al., Am. Mineral. 65, 920) | 0.01 Å |
| 2H-MoS₂ Mo-S distance | 2.41 Å (Schoenfeld et al., Acta Cryst. B 39, 404) | 0.02 Å |
| Primitive-cell volume vs conventional | V/multiplicity | 1 part in 10⁹ |

**The α-quartz case earned its place.** The first implementation used a
plausible but wrong space-group setting and produced Si-O distances of 1.29 and
1.87 Å. The validation case caught it, the operator set and origin were
corrected, and the shipped structure now reproduces both published bond lengths
and the 2.61-2.64 Å O-O distances. A material whose generated cell does not
reproduce its own cited geometry is a bug, and this is how it is found.

## Classical potential

| Case | Expected | Tolerance | Why |
| --- | --- | --- | --- |
| Si cohesive energy in the ideal diamond lattice | exactly −2ε = −4.3366 eV/atom | 1 part in 10⁷ | SW is constructed so this is exact; only floating point separates them |
| Ge cohesive energy | −2ε = −3.86 eV/atom | 1 part in 10⁴ | The Ding-Andersen parameters were fitted with a slightly different lattice constant, so the shipped Ge cell sits a few parts in 10⁶ off the potential's own minimum. The test records that rather than hiding it |
| Fitted lattice constant is the energy minimum | E(1.00a) < E(0.98a), E(1.02a) |, | A potential whose minimum is not where it was fitted is broken |
| Analytic forces vs central differences | agreement | 10⁻⁶ eV/Å | Step 10⁻⁵ Å on a randomly displaced slab |
| Newton's third law | Σ F = 0 | 10⁻⁹ eV/Å | Conservation law |
| Translational invariance | E unchanged by a rigid shift | 1 part in 10¹² | Conservation law |
| Microcanonical energy drift | small | 5×10⁻⁴ eV/atom over 500 fs | The only real correctness test of an integrator |
| Langevin equipartition | mean T = target | 3 % | Three standard errors of a 1.5 ps mean for 216 atoms; a 12 ps run gives 501.2 ± 1.6 K at 500 K |

## Tight binding

Against the values the Vogl parameterisation was fitted to
(J. Phys. Chem. Solids 44 (1983) 365, Table II). **Agreement here is a
statement about the fit, not a prediction.**

| Case | Expected | Measured | Tolerance |
| --- | --- | --- | --- |
| Si valence bandwidth | 12.5 eV | 12.500 eV | 0.05 eV |
| Si valence-band maximum at Γ | 0 eV at (0,0,0) | 0 eV at Γ | 10⁻³ eV |
| Si indirect gap, bounded search along Γ to X, confirmed by a 12³ zone scan | 1.17 eV, not at Γ | 1.1713 eV at 0.731 of Γ to X (experiment: near 0.85; a property of the model) | 5×10⁻⁴ eV |
| GaAs direct gap at Γ | 1.55 eV | 1.550 eV | 0.02 eV |
| Graphene gap at K | 0 | 0 | 10⁻⁵ eV (K is not exactly representable in binary) |
| Graphene bandwidth | 6\|t\| = 16.2 eV | 16.200 eV | 10⁻⁹ eV |
| Graphene electron-hole symmetry | E₊ = −E₋ at every k | holds | 10⁻⁹ eV |
| Hamiltonian hermiticity at arbitrary k | H = H† | holds | 10⁻¹² |
| Valence-electron count | 4 per Si | holds | exact |
| Donor attracts charge | q(P) < mean q(Si) | holds |, |

## Scanning tunnelling microscopy

| Case | Expected | Tolerance |
| --- | --- | --- |
| Current decays towards exp(−2κz) | apparent decay 0.901, 0.932 and 0.965 of 2κ over 6 to 9, 10 to 16 and 20 to 28 Å, rising towards 1 | 0.005 |
| Corrugation falls with tip height | relative corrugation at 4 Å > at 8 Å |, |
| Si(111) corrugation magnitude | 0.05-1.5 Å, the experimental range | order of magnitude |
| Maxima follow the lattice | nearest-maximum spacing = site spacing | 12 % |

The decay-constant case is the one that would catch a wrong barrier or a wrong
vacuum tail, and it ties the simulated instrument to the physics it claims. The
apparent decay is slower than 2κ near the surface because the tails of an
atom's neighbours decay more slowly with height than its own orbital; the
apparent barrier of a superposition is lower than the true one, as in
experiment. The earlier 12 % tolerance hid that trend; the case now checks it.

## Atomic force microscopy

| Case | Expected | Tolerance |
| --- | --- | --- |
| Small-amplitude limit | Δf → −(f₀/2k)·∂F/∂z as A → 0 | 0.1 % |
| Quadrature exact for a linear force | Δf = −f₀c/2k at any A and any node count | 10⁻¹⁰ |
| Hamaker background | F ∝ 1/z² | 1 part in 10⁹ |

**The quadrature case found a real bug.** The frequency-shift integral was
being evaluated with second-kind Chebyshev nodes and equal weights, which
under-estimates it by a factor (n−1)/(n+1), about 6 % at 33 nodes. The
Giessibl integrand carries a 1/√(A²−q²) weight, which is the *first*-kind
weight. The fix is in `_chebyshev_nodes`, and the exactness case now guards it.

## Surface reconstruction

Si(100)-(2×1) built by the dimer generator and relaxed with Stillinger-Weber.
The generator imposes the pairing and a starting separation only, so every
distance here is an output of the relaxation. These cases test whether the
implementation reproduces the reconstruction the *model* predicts, and report
how far that model sits from experiment.

| Case | Expected | Tolerance |
| --- | --- | --- |
| Paired sites sat at the 1×1 spacing | a/√2 = 3.8403 Å | 10⁻⁴ Å |
| Relaxation converged | residual ≤ 0.005 eV/Å | criterion itself |
| Dimer is a bond, not a lattice spacing | 0.9-1.15 × the bulk 2.3517 Å bond, and > 1 Å below the 1×1 spacing | window |
| All dimers in the supercell equivalent | spread over 4 dimers = 0 | 10⁻⁶ Å |
| 1×1 translation broken | not invariant | 0.05 Å site matching |
| 2×1 translation held | invariant, in a 4-repeat cell where it is not the lattice vector | 0.05 Å |
| Independent of the construction guess | same bond from starts of 2.2, 2.6, 3.0, 3.4 Å | 10⁻³ Å |
| Independent of cell size | same bond at (2,2,8), (4,3,8), (4,2,10), (6,2,8) | 10⁻³ Å |
| Reconstruction lowers the energy | > 0.5 eV per dimer against the relaxed ideal truncation | measured 1.68 eV |
| Subsurface strain decays with depth | displacement 6 Å down < 25 % of the surface value | ratio |

Measured result: **2.4035 Å symmetric dimer**, **1.68 eV per dimer** below the
relaxed ideal truncation, zero buckling.

### Against experiment, reported not absorbed

The LEED structure determination of H. Over, J. Wasserfall, W. Ranke,
C. Ambiatello, R. Sawitzki, D. Wolf and W. Moritz, Phys. Rev. B **55** (1997)
4731 gives a buckled dimer: bond 2.24 ± 0.08 Å, vertical separation 0.72 ±
0.05 Å, tilt 19 ± 2°.

| Quantity | Computed | Published | Deviation | Inside quoted uncertainty |
| --- | ---: | ---: | ---: | --- |
| dimer bond | 2.4035 Å | 2.24 ± 0.08 Å | +0.164 Å (+7.3 %) | **no** |
| buckling | 0.0000 Å | 0.72 ± 0.05 Å | −0.72 Å | **no** |

Two cases assert exactly this: that the bond overestimate is bounded (< 15 %)
and positive and is *not* reported as agreeing, and that the buckling is zero
and is *not* reported as agreeing. The buckling is a Jahn-Teller distortion
driven by charge transfer between the dimer atoms; Stillinger-Weber is a
three-body potential of nuclear coordinates and has no mechanism for it. A
model that cannot produce a feature should say so, and the validation suite is
where that is enforced.

## GPAW first principles

Real self-consistent calculations, skipped when GPAW is not installed. Run on
GPAW 25.7.0 with ASE 3.29.0, serial, PBE, a 0.16 Å real-space grid and a
1e-5 eV/electron energy tolerance.

| Case | Expected | Tolerance |
| --- | --- | --- |
| Force equals minus the energy gradient | Hellmann-Feynman identity on GPAW's own energy surface | 0.02 eV/Å |
| Forces on a symmetric dimer | equal and opposite, no transverse component | 10⁻³ eV/Å |
| Energy curve has an interior minimum | between 0.72 and 0.80 Å | bracketing |
| H₂ bond length | within 3 % of the measured 0.7414 Å, and longer | 3 % |
| Curvature at the minimum | positive | sign |
| Same calculation twice | identical | 10⁻⁶ eV |

Measured: analytic force **0.387225 eV/Å** against a central-difference
derivative of **0.391169 eV/Å**, agreeing to **0.0039 eV/Å**. Equilibrium bond
length **0.7521 Å** against the experimental **0.7414 Å** of Huber and Herzberg
(1979), an overestimate of **+1.44 %**, the known behaviour of PBE for this
bond, reported rather than asserted away.

The forces case is the strongest check available without reference to
experiment: the expected relationship is exact, and the energy and the forces
reach it through different parts of the driver.

These results are for H₂. They say the driver reproduces what GPAW computes;
they say nothing about an arbitrary material, and none of these settings is
converged with respect to grid, box or k-points.

## Ground-state DFT experiments

`tests/validation/test_dft_ground_state_finite.py` and
`tests/validation/test_dft_ground_state_periodic.py` exercise the full
versioned experiment path, not the earlier energy-and-force adapter. They run
through the Python API, out-of-process GPAW worker, result conversion and
chunked project array store. They are skipped only when GPAW or its PAW
datasets are unavailable.

Finite calculations use PBE, a 0.22 A real-space grid and an 8 A box.

| Case | Expected | Tolerance | Why |
| --- | --- | --- | --- |
| Neutral H2 converges and carries the complete solver, dataset and input identity | calculated Tier 3 result with a pinned PAW SHA-256 | exact fields | A numerical value without this identity is not reproducible |
| Neutral H2 all-electron density and occupations | 2 electrons | 2e-6 electrons | Charge conservation, including the frozen core convention |
| H2 force on the upper atom against a central energy difference with a 0.01 A bond step | agreement | 0.03 eV/A | The force and total energy reach the identity through different GPAW paths; the finite grid leaves an egg-box error |
| H2+ density, occupied states and charge accounting | 1 electron | 2e-6 electrons | Charge conservation for a finite charged system |
| H2+ total magnetic moment | 1 mu_B | 0.002 mu_B | One unpaired electron in a collinear spin calculation |

Periodic calculations use the eight-atom conventional silicon cell, LDA,
a 200 eV plane-wave cutoff and the Gamma point. These deliberately economical
settings make the validation fast; their results are not presented as
production-converged silicon values.

| Case | Expected | Tolerance | Why |
| --- | --- | --- | --- |
| Silicon energy per atom | physically bound state between -5.5 and -3.5 eV/atom in GPAW's reference convention | stated window | Detects sign, normalization and reference failures without claiming experimental accuracy |
| Symmetry and translational force balance | maximum force below 0.05 eV/A and vector sum below 1e-6 eV/A | stated bounds | The ideal diamond cell has no preferred displacement and an isolated calculation cannot create net force |
| All-electron periodic density | 112 electrons for eight Si atoms | 2e-4 electrons | Charge conservation including 80 frozen-core electrons |
| Hydrostatic pressure from stress against the derivative of energies at plus and minus 0.4 percent isotropic strain | agreement | 1 GPa | Independent stress and energy routes, with a loose bound for the low plane-wave cutoff and Pulay stress |
| Two-point 150 to 200 eV cutoff study | both points stored, residual calculated, tolerance verdict recorded | exact record and 0.2 eV/atom study tolerance | Validates the convergence-laboratory contract while keeping the physical limitation explicit |

The implementation tests in `tests/unit/test_dft_experiment.py` and
`tests/integration/test_dft_experiment_service.py` separately check immutable
specifications, every solver parameter, physical refusal paths, restart
identity, failed and cancelled outcomes, parameter echoes, service and HTTP
access, project ownership, staleness, save and reopen, classified claims,
checksums and deliberate array corruption. Those are software guarantees, not
additional physical validation.

## DFT relaxation

`tests/validation/test_dft_relaxation_live.py` runs real GPAW relaxations
through the relaxation experiment and checks each relaxed geometry with an
independent ground-state run (the existing experiment, a fresh SCF at the
stored final geometry). Without GPAW each case skips with the reason
`BLOCKED`; a skip is not a pass. On this machine (GPAW 25.7.0, ASE 3.29.0,
PAW datasets 0.9.20000, serial) all four cases ran and passed.

| Case | Expected | Tolerance | Result here |
| --- | --- | --- | --- |
| H2, PBE, 0.22 A grid, 6 A box, from 0.80 A, fixed cell, BFGS, fmax 0.02 eV/A | PBE bond about 0.750 A | 0.01 A | 0.74895 A after 3 steps, largest force 0.00135 eV/A |
| Same geometry recomputed independently | energy equal, forces within fmax | 1e-4 eV; 0.02 eV/A | -6.7048205 against -6.7048205 eV (difference 5.7e-8 eV); largest force 0.00131 eV/A |
| H2 with one atom fixed (symmetry off, as required) | fixed atom unchanged, bond as above | exact; 0.01 A | fixed atom bit-identical; 0.74948 A after 4 steps |
| Diamond Si, PBE, 500 eV, 4x4x4, fixed occupations, from a = 5.60 A, variable cell, stress tolerance 0.001 eV/A^3 | PBE lattice constant about 5.47 A | 0.04 A | 5.4872 A after 3 steps, volume -5.922 %, stress deviation 1.2e-4 eV/A^3 |
| Same cell recomputed independently | energy equal, largest stress within tolerance | 1e-4 eV; 0.001 eV/A^3 | -10.6152546 eV in both; largest stress component 1.2e-4 eV/A^3 |
| Service run, auto-apply and undo with real GPAW | one undoable change, undo restores the start | exact | passed |

The Si lattice constant sits 0.017 A above the usual PBE value at these
settings; the 4x4x4 grid on a two-atom cell is not k-point converged, which
the tolerance allows for. These are checks of the relaxation machinery against
known physics, not converged production values.

The implementation tests in `tests/unit/test_dft_relaxation.py` (the
specification, every refusal, the job every variable reaches, the worker's
ionic loop run in-process with ASE's EMT calculator, and output validation)
and `tests/integration/test_dft_relaxation_service.py` (storage, apply, undo
and redo, stale and detached ownership, cancellation, time limit, failures,
persistence, corruption and HTTP, with the deterministic worker double in
`tests/support/fake_gpaw_relax.py`) are software guarantees, not physical
validation.

## DFT density of states

`tests/validation/test_dft_dos_live.py` runs real GPAW DOS and PDOS jobs and
checks spectral state counts and levels through routes independent of the
curve returned by GPAW. Without GPAW each case skips with the reason `BLOCKED`.
On this machine all five cases ran and passed.

| Case | Independent check | Result here |
| --- | --- | --- |
| H2, Gaussian DOS and H s projection | level against a fresh ground state; two electrons below the Fermi level | level -10.3454 eV; 2.000000 electrons |
| Diamond Si, Gaussian DOS and Si s and p projections | eight valence electrons; finite Kohn-Sham gap | 8.0000 electrons; 0.7336 eV gap |
| Diamond Si, tetrahedron DOS | eight valence electrons; independent reconstruction from eigenvalues | passed |
| Spin-polarised H atom | one occupied spin-up state and no occupied spin-down state | passed |
| DOS from a converged H2 relaxation | source energy agrees with a fresh calculation | passed |

The silicon gap is a PBE Kohn-Sham gap at the stated numerical settings. It is
not an experimental or quasiparticle gap. The implementation tests separately
cover frozen specifications, every variable, refusal paths, Gaussian and
tetrahedron recomputation, band-window completeness, spin and projection
consistency, storage, staleness, persistence, corrupted arrays, HTTP, CSV and
the desktop contract.

## DFT LDOS and STM images

`tests/validation/test_dft_ldos_live.py` checks the LDOS against an independent
implementation and against GPAW's own density. Without GPAW each case skips
with the reason `BLOCKED`. On this machine all four ran and passed on
2026-09-30.

| Case | Check and tolerance | Result here |
| --- | --- | --- |
| Graphene, PBE, 0.20 A grid, 6x6x1, window -1 to 0 eV | map against ASE's `ase.dft.stm.STM` run on the same job in a separate single-threaded GPAW process, 1e-4 of the maximum (the SCF criterion); constant-current heights against ASE's `scan`, 1e-3 A; constant-height values against linear interpolation of ASE's map | map bit-identical; heights within 1.8e-15 A; 0.1111 states per cell in the window, map integral 0.951 of it |
| H2, fixed occupations, window -20 to 0 eV | map equals GPAW's pseudo valence density of an independent run, 1e-4 of its maximum; exactly 2 states | passed |
| Spin-polarised H atom, window -10 to 0 eV | one state, all of it in spin up; channels add to the total | passed |
| LDOS of a converged H2 relaxation | current once the relaxation is applied; 2 states | passed |

A reference run with multithreaded BLAS differed from the single-threaded
worker by up to 8.6e-4 of the maximum, which is why both run single-threaded
and why the tolerance is the SCF criterion rather than zero.

## DFT band structure

`tests/validation/test_dft_bands_live.py` runs real GPAW band structures and
checks them against quantities obtained independently of the path step.
Without GPAW each case skips with the reason `BLOCKED`. On this machine all
five cases ran and passed on 2026-09-29 with GPAW 25.7.0 and ASE 3.29.0.

The setup is deliberately light: PBE, plane waves at 300 eV, a 4x4x4
Gamma-centred ground-state grid, fixed occupations, a = 5.43 A for silicon.
The tolerances are coarse because they are set for that setup and for the
path sampling (12 intervals from Gamma to X), not for converged PBE; they are
not widened to pass.

| Case | Check and tolerance | Result here |
| --- | --- | --- |
| Si, standard FCC path G-X-W-K-G-L (45 k-points) | gap indirect; valence top at Gamma; conduction bottom on Gamma to X at 0.7 to 0.95 of the way (PBE reference near 0.85); indirect gap 0.45 to 0.75 eV (PBE reference about 0.6 eV); direct gap at Gamma 2.35 to 2.75 eV (PBE reference about 2.55 eV); path distance against the pinned cell to 1e-10 | indirect; top at Gamma; bottom at 0.833 of Gamma to X, (0.4167, 0, 0.4167); gap 0.5528 eV; direct gap at Gamma 2.5409 eV; distance error 3.0e-16 |
| Si endpoints against an independent ground state | a separate GPAW ground-state run with the same settings, in its own process; at Gamma and at X (found in its irreducible grid as (0.5, 0.5, 0)) occupied eigenvalues to 2 meV, lowest conduction band to 20 meV, Fermi level to 1 meV | largest occupied difference 0.83 microeV at Gamma and 0.41 microeV at X; conduction 0.16 and 0.97 microeV; Fermi level identical |
| bcc Fe, spin-polarised, 8x8x8, 350 eV, path G-H-N-G | two spin channels returned; a band crosses the Fermi level in each; mean of spin-down d bands at Gamma 1 to 3 eV above spin-up (PBE exchange splitting about 2 eV) | both channels metallic; splitting 2.35 eV |
| Graphene sheet as a 2D slab, 0.18 A grid, 9x9x1, 12 A vacuum | path generated as G-M-K-G in the plane with no out-of-plane component; the two pi bands at K within 30 meV of each other and of the Fermi level | HEX2D path; pi bands at K at +0.17 and -0.17 meV, 0.35 meV apart |
| Si from a real converged relaxation (atom displaced, fixed cell, symmetry off) | relaxation converged and applied; bands current; ground-state energy against the relaxation's final energy to 1 meV; symmetry setting inherited; gap within 50 meV of the unrelaxed crystal | converged in 3 steps; current; energy difference 1.1e-8 eV; symmetry off; gap 0.5505 eV |

The silicon gap is a PBE Kohn-Sham gap at the stated settings. It is not an
experimental or quasiparticle gap (the measured indirect gap is 1.17 eV), and
no agreement with experiment is claimed. The endpoint agreement shows that the
path step reproduces GPAW's own self-consistent eigenvalues at the same density;
it does not validate the functional. The implementation tests separately cover
the frozen specification, stored paths that are never regenerated invisibly,
bulk, slab, wire and cluster rules, every refusal, echo and shape fault, path
order, non-finite and unordered eigenvalues, distance and reciprocal-cell
mismatches, cancellation, storage, staleness, detachment, checksum corruption,
persistence, HTTP, CSV and the desktop contract, all with the deterministic
worker double in `tests/support/fake_gpaw_bands.py`.

## Point-charge electrostatics

`tests/validation/test_electrostatics_validation.py`. Each case sums the
Coulomb energy of a point-charge lattice whose exact value is known and
compares it at an Ewald accuracy of 10^-10. These establish that the three
summation methods reproduce the point-charge model. They say nothing about
whether point charges describe a real material.

| Case | Reference | Measured | Tolerance |
| --- | --- | --- | --- |
| Rock-salt Madelung constant | 1.747564594633 (Borwein et al., *Lattice Sums Then and Now*, 2013) | 1.747564594657 | 10^-8 relative |
| Rock salt, primitive cell against a 2 x 2 x 2 conventional supercell | equal energy per ion pair | agree | 10^-9 relative |
| Caesium chloride Madelung constant | 1.762675 (Kittel; Tosi 1964) | 1.762674773111 | 10^-6 |
| Zinc-blende Madelung constant | 1.638055 (Kittel; Tosi 1964) | 1.638055053409 | 10^-6 |
| Madelung energy scales as q^2 (MgO-like, formal-point-ion model with +2/-2) | rock-salt constant | agree | 10^-8 relative |
| Square lattice of alternating charges (one NaCl(100) plane, slab method) | 1.615542626712 (Borwein et al. 2013) | 1.615542626706 | 10^-8 relative |
| Adding one NaCl(100) plane to a 5-plane slab adds one bulk plane's energy | -17.847028804 eV | -17.847028802 eV | 10^-7 relative |
| Point charge in a uniform background, simple cubic | alpha = 2.8373 (Makov and Payne 1995) | 2.8372974795 | 5 x 10^-5 |
| Isolated ion pair | Coulomb's law, -k_e / r | agree | 10^-14 relative |
| Neutral cluster in a growing box (vacuum surrounding) approaches the isolated sum | error falls with box size | 60 A box within 10^-3 eV | bracketing |
| Coulomb constant | 14.3996454784 eV A | agree | 10^-9 relative |

The slab-layer case ties the slab geometry, with its enlarged internal cell
and dipole correction, to the published bulk constant: the surface
contribution of a point-charge rock-salt slab decays exponentially with depth,
so the energy of one more plane must be the bulk energy of one plane.

Implementation checks that are not physical validation live in
`tests/unit/test_electrostatics.py`: forces agree with central differences of
the energy to 10^-6 eV/A in all three geometries, forces sum to zero, rigid
translation and the choice of periodic image leave tin-foil results unchanged,
the vacuum surrounding does change with the image as it must, the result is
independent of the splitting parameter to 10^-7 eV, repeated runs are bitwise
identical, and every refusal path refuses.

## Rigid-ion potentials

`tests/validation/test_rigid_ion_validation.py`. The model is Ewald
electrostatics plus Buckingham pair terms ([RIGID_ION.md](RIGID_ION.md)).

| Case | Reference | Measured | Tolerance |
| --- | --- | --- | --- |
| Born-Mayer rock salt (charges +1 and -1, illustrative A = 1200 eV, b = 3.125 1/A) at r = 2.6, 2.8, 3.0 A | closed form `-M k_e / r + 6 (phi(r) - phi(r_c))` with M = 1.747564594633 (Borwein et al. 2013) | agree | 10^-9 relative |
| Same crystal, minimum-energy distance | root of `M k_e / r^2 = 6 A b exp(-b r)` | agree | 10^-5 A |
| BKS quartz relaxed at the measured cell | three Si and six O sites stay equivalent | equivalent | 10^-4 A |
| BKS quartz Si-O bonds, plausibility only | 1.605 and 1.614 A measured (Levien et al. 1980) | 1.595 and 1.604 A | 0.02 A |
| BKS quartz NVE dynamics, 72 atoms, 200 fs | energy conserved to second order in the step | spread falls by about 4 when the step halves | spread below 2 x 10^-4 eV per atom at 1 fs; ratio 2.5 to 6 |

The Born-Mayer parameters are illustrative, chosen to test the sum, and are not
a fit to NaCl. The quartz bond lengths test a fixed-cell relaxation at the
experimental cell; they do not validate the BKS fit, which Materia takes as
published.

Implementation checks in `tests/unit/test_rigid_ion.py`: the shipped BKS values
match the publication, the Si-O dimer matches the closed form to 10^-12, the
Buckingham sum equals an explicit lattice-image sum written without the
neighbour list to 10^-12, the Coulomb part is exactly the electrostatics
module, forces equal central differences to 10^-6 eV/A for a crystal, slab
and cluster, the energy is continuous at the cutoff, the collapse barrier is
the pair interaction's inner maximum, a pair inside it is refused and a
dynamics run that crosses it stops, and a Si-O dimer relaxes to the pair
interaction's own minimum to 10^-5 A.

## Harmonic vibrations

`tests/validation/test_phonons_validation.py`, method in [PHONONS.md](PHONONS.md).

| Case | Reference | Measured | Tolerance |
| --- | --- | --- | --- |
| Lennard-Jones dimer, 4-point stencil | `sqrt(k / mu)`, `k = 36 2^(2/3) epsilon / sigma^2` | agree | 10^-7 relative |
| fcc Lennard-Jones, 32-atom supercell, 96 modes | Born-von Karman lattice sum at the 32 commensurate wavevectors (Born and Huang 1954) | 2.4 x 10^-8 THz | 10^-6 THz |
| Einstein solid, 12 modes | `sqrt(k / m)` | agree | 10^-10 relative |
| Relaxed EMT Cu19 cluster, 57 modes | ASE `Vibrations`, independent implementation | 3.5 x 10^-8 THz | 10^-6 THz |

The lattice-sum case checks mass weighting, the tension terms of a pair
potential away from its minimum and the folding of atoms onto their own
periodic images, since the supercell is narrower than twice the cutoff.

## Embedded-atom metals

`tests/validation/test_eam_validation.py`, shipped Zhou 2004 potentials
(`docs/EAM.md`). Three kinds of evidence are kept apart.

**1. Independent implementations of the same files.** ASE 3.29's EAM
calculator reads each file with its own spline; NIST's iprPy calculations used
LAMMPS on the same NIST retabulation (static, 0 K). Reference pages:
`https://www.ctcms.nist.gov/potentials/entry/2004--Zhou-X-W-Johnson-R-A-Wadley-H-N-G--<El>/2004--Zhou-X-W--<El>--LAMMPS--ipr2/calc.html`,
retrieved 2026-09-25.

| Quantity | Cu Materia / NIST | Au Materia / NIST | W Materia / NIST | Tolerance |
| --- | --- | --- | --- | --- |
| Ground state | fcc / fcc | fcc / fcc | bcc / bcc | at least 0.02 eV/atom lower than the other cubic structure |
| Lattice constant (A) | see below | 4.0800534225 / 4.0800534236 | 3.1648494565 / 3.1648494550 | 5e-8 A |
| Cohesive energy (eV) | 3.53998 / 3.54 | 3.93001 / 3.93 | 8.75999 / 8.76 | 0.005 eV (NIST gives two decimals) |
| C11, C12, C44 (GPa) | 169.989, 122.225, 75.927 / identical | 186.378, 157.343, 42.071 / identical | 522.536, 204.219, 160.752 / 522.536, 204.219, 160.753 | 0.05 GPa |
| Bulk modulus from hydrostatic strain (GPa) | 138.146 | 167.021 | 310.325 | 2e-4 relative to (C11 + 2 C12)/3 |
| Relaxed vacancy formation energy (eV) | 1.2778 / 1.279 | 0.9999 / 1.001 | 3.5761 / 3.575 | 0.005 eV |
| Relaxed surface energy (mJ/m^2) | (111) 1501.32 / 1501.45, (100) 1563.11 / 1563.12 | (111) 907.47 / 907.51, (100) 1016.05 / 1016.05 | (110) 2567.68 / 2567.68, (100) 2983.46 / 2983.46 | 5e-4 relative |
| Energy and forces against ASE, perturbed 3x3x3 cells | 7e-12 eV/atom, 5e-8 eV/A | 8e-12 eV/atom, 2e-8 eV/A | 2e-11 eV/atom, 3e-8 eV/A | 1e-8 eV/atom, 5e-7 eV/A |

Vacancies were relaxed in 256-atom (fcc) and 250-atom (bcc) cells at fixed
volume; surfaces in 14-layer slabs. The remaining differences are at the level
expected from cell size and relaxation tolerance. An energy-only route to the
W bulk modulus (310.3 GPa from the curvature of E(a)) agrees with the stress
route to 2e-3.

**Cu's lattice constant.** The fifth-neighbour shell reaches the file's cutoff
at 3.614959 A, where the energy steps up by about 1.2e-5 eV per atom because
the tabulated functions are not exactly zero at the cutoff. The test asserts
the step, the monotonic fall below it, and the stationary point above it at
3.6149789 A, which matches NIST's 3.6149788819 A to 5e-8 A. NIST's other two
values, 3.61470 and 3.61479 A, lie on the lower branch. Cu elastic constants,
vacancy and surface energies are compared at NIST's 3.6146974 A.

OpenKIM's own lattice-constant tests agree for W (3.164849452674 A) and Au
(4.080053508282 A) to 2e-9 and 9e-8 A. OpenKIM's elastic-constant query for Cu
returned 196.3, 122.1 and 92.9 GPa, which disagrees with both NIST and
Materia; the cause was not established and those numbers are not used.

**2. The parameter publication.** Not checked. The property tables of Zhou,
Johnson and Wadley, Phys. Rev. B 69 (2004) 144113 were not accessible when
this suite was written, so no case claims agreement with them.

**3. Experiment.** Materia's 0 K values against the room-temperature values in
the material library (CRC lattice constants, Kittel cohesive energies,
Simmons and Wang or Neighbours and Alers elastic constants). Bounds: 0.5 % on
a, 5 % on E_coh and on each elastic constant, because thermal expansion and
softening alone are of that order. The potential was fitted to data of this
kind, so agreement here is expected and says nothing about its accuracy for
surfaces, defects or alloys.

| | Cu | Au | W |
| --- | --- | --- | --- |
| a, potential / measured (A) | 3.61498 / 3.6149 | 4.08005 / 4.0782 | 3.16485 / 3.1652 |
| E_coh (eV) | 3.540 / 3.49 | 3.930 / 3.81 | 8.760 / 8.90 |
| C11, C12, C44 (GPa) | 170.0, 122.2, 75.9 / 168.3, 122.1, 75.7 | 186.4, 157.3, 42.1 / 192.9, 163.8, 41.5 | 522.5, 204.2, 160.8 / 522.4, 204.4, 160.6 |

**Dynamics.** NVE velocity Verlet at 1 fs on 108 Cu atoms started at 600 K:
the total energy varies by less than 2 % of the mean kinetic energy over 400
steps.

## LAMMPS adapter

`tests/validation/test_lammps_live.py` compares a real LAMMPS with Materia's
EAM on the shipped Cu-Zhou04 file, which both read byte for byte and
interpolate with the same cubic Hermite scheme:

| Case | Quantity | Tolerance |
| --- | --- | --- |
| 108-atom Cu, perturbed by 0.05 A | energy per atom | 1e-6 eV |
| | forces | 1e-5 eV/A |
| | stress | 1e-6 eV/A^3 |
| Relaxation from 0.03 A perturbations | Materia's force on LAMMPS's minimum | 5e-4 eV/A |
| | energy at LAMMPS's minimum | 1e-5 eV |
| NVE, 200 steps of 1 fs from 300 K | total-energy drift per atom | 1e-4 eV |

**Status: blocked.** No LAMMPS executable or Python module was installed on
the machine where the adapter was written, so these cases skip with the
reason `BLOCKED` and have not been performed. They are not counted as passes.
Run `pytest tests/validation/test_lammps_live.py -rs` after installing
LAMMPS (see [EXTERNAL_SOLVER_BACKENDS.md](EXTERNAL_SOLVER_BACKENDS.md#lammps)).

What is tested without LAMMPS is the adapter, not LAMMPS: 48 unit and 57
integration tests in `tests/unit/test_lammps.py`,
`tests/integration/test_lammps_runs.py` and
`tests/integration/test_lammps_service.py` run a deterministic fake process
(`tests/support/fake_lammps.py`) that evaluates Materia's own EAM in LAMMPS
metal units and writes LAMMPS-format logs and dumps. Energies, forces and
stresses returned through it agree with Materia's to 1e-12 relative, which
shows that positions, image flags, types, isotopic masses, velocities and the
pressure-to-stress conversion survive the round trip. Those are
implementation tests; they say nothing about LAMMPS's physics.

## What is not validated

* **No comparison with experimental STM images.** That requires reference data
  that is not shipped.
* **No comparison with published DFT surface energies, defect formation
  energies or relaxations.** The Tier 3 ground-state adapter now exists, but
  the required reference datasets and matched calculation protocols are not
  shipped.
* **No live LAMMPS comparison has been run.** The cases exist and are blocked
  until LAMMPS is installed; see above.
* **No validation of the procedural wafer against a real wafer.** It is
  synthetic by construction and never claims otherwise.
* **No validation of the AFM contrast mechanism**, only of its mathematics.
  The force model has no chemical bonding, so there is nothing to validate it
  against beyond its own definition.

These gaps are the most important entries in
[LIMITATIONS.md](LIMITATIONS.md) and the first items on
[ROADMAP.md](ROADMAP.md) that would close them.
