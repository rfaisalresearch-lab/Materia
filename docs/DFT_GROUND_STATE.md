# Ground-state DFT experiments

Materia runs self-consistent Kohn-Sham density functional theory through
GPAW as an **experiment**: an immutable, versioned specification of every
variable the calculation depends on, checked against physical and numerical
rules before anything starts, executed in GPAW's own interpreter, and stored
with every observable, its provenance and its convergence record. A
**convergence laboratory** measures how much a result still depends on its
numerical settings.

This is one capability of a larger aim: a research workbench that drives
established open-source solvers through one consistent, validated experiment
system and produces reproducible physical hypotheses. It is not an exact
universal simulator. A density functional has systematic errors, and every
record here says so.

Code: `materia/experiments/dft/` (`spec.py`, `boundary.py`, `datasets.py`,
`run.py`, `records.py`, `restart.py`, `convergence.py`),
`materia/solvers/gpaw_driver/ground_state_worker.py` (runs inside GPAW's
interpreter), `materia/project_format/arrays.py` (chunked grids),
`materia/provenance/classification.py` (result classifications). Tests:
`tests/unit/test_dft_experiment.py`,
`tests/integration/test_dft_experiment_service.py`,
`tests/validation/test_dft_ground_state_finite.py`,
`tests/validation/test_dft_ground_state_periodic.py`. Setup of GPAW itself:
[GPAW.md](GPAW.md).

## The specification

`GroundStateSpec` (schema `materia.dft.ground-state`, version 1.0) is a frozen
dataclass. Changing a variable returns a new specification; the old one is
untouched. It serialises to JSON, is rebuilt with `from_dict` (which refuses
unknown or missing fields, another schema and another version), and has a
SHA-256 `digest` over its canonical JSON that becomes the `inputs_digest` of
every result it produces.

| Section | Variable | Unit | Rule |
| --- | --- | --- | --- |
| identity | structure key, formula, geometry digest, atom ids, atomic numbers, positions, isotopes, formal charges, cell, periodic axes | A, mass number, e | frozen from the structure, not settable |
| physical system | `boundary`, `vacuum_axis` | | derived from the periodic axes |
| | `charge_e` | e | whole number; defaults to the sum of the atoms' formal charges |
| | `charged_periodic_policy` | | `uniform-background` required for, and only allowed for, a charged bulk cell |
| | `poisson` | | `moment-corrected` for clusters, `dipole-layer` for slabs, `gpaw-default` |
| | `external_field_V_per_A` | V/A | vector; only along open directions; fd and lcao |
| electronic state | `spin_polarized` | | required for an odd electron count in a finite system and for any non-zero moment |
| | `initial_magnetic_moments_muB` | mu_B | one per atom; a starting guess |
| | `occupations`, `smearing_eV` | eV | fixed (width 0), Fermi-Dirac or Marzari-Vanderbilt (width above 0, at most 1 eV) |
| | `n_bands` | bands | empty for GPAW's default; never fewer than the occupied states |
| numerical accuracy | `xc` | | LDA, PBE, RPBE, revPBE, PBEsol, BLYP, offered only when every element has a dataset for it |
| | `representation` | | `fd`, `pw` (bulk only), `lcao` |
| | `grid_spacing_A` | A | fd and lcao; 0.05 to 0.35 |
| | `cutoff_eV` | eV | pw; 100 to 3000 |
| | `basis` | | lcao; must be installed for every element |
| | `kpoints`, `kpoints_gamma_centered` | divisions | 1 along every open axis, at most 32 |
| | `energy_tol_eV_per_electron`, `density_tol_electrons_per_electron`, `eigenstates_tol_eV2_per_electron`, `forces_tol_eV_A` | as named | positive and no looser than 5e-3, 1e-2, 1e-4 and 0.5 |
| | `max_scf_iterations` | iterations | 1 to 2000 |
| | `random_seed` | | must be empty: GPAW's start is deterministic |
| requested observables | `observables` | | energy (always), forces, stress (pw only), density, spin density (spin only), electrostatic potential, eigenvalues, occupations, Fermi level, magnetic moment (spin only) |
| software | GPAW, ASE and Python versions, dataset distribution, every PAW dataset and basis file with its SHA-256 | | pinned when the specification is built; a mismatch at run time refuses |

Every settable variable goes into the keyword arguments handed verbatim to
`gpaw.GPAW` (`gpaw_parameters(spec)`). After the run the worker reads back the
parameters GPAW actually used, and a run whose echoed parameters differ from
those sent is reported as failed rather than stored. That is the check that no
variable is accepted and then ignored. The two exceptions are deliberate:
`observables` changes what is extracted, not the calculation, and
`random_seed` is refused because nothing in GPAW's ground state is random.

**Isotopes** set the nuclear masses of the atoms GPAW receives. The
Born-Oppenheimer electronic ground state does not depend on nuclear mass, so
no value in this calculation changes with the isotope, and a result stays
current when only an isotope changes. Vibrations and dynamics do depend on it.
The interface states this next to the physical-system settings.

## Boundary rules

| Class | Periodic axes | Allowed | Default |
| --- | --- | --- | --- |
| cluster | none | fd, lcao; Gamma only | fd 0.20 A, fixed occupations, moment-corrected Poisson |
| wire | one | fd, lcao; k-points along the axis | fd 0.20 A, Fermi-Dirac 0.1 eV |
| slab | two | fd, lcao; in-plane k-points; dipole layer | fd 0.20 A, Fermi-Dirac 0.1 eV, dipole layer |
| bulk | three | fd, lcao, pw; k-points; stress in pw | pw 400 eV, Fermi-Dirac 0.1 eV |

Default k-points give at least 30 A of divisions times lattice length along
each periodic axis. The earlier 20 A default was measured to be too coarse for
metals: with it, a copper bulk modulus from an energy-volume scan was 7 percent
low even on a fixed grid. Because the automatic grid follows the cell size, two
cells of different size can get different grids, and total energies from
different grids are not comparable: an aluminium energy-volume scan with the
automatic grid jumped from 8x8x8 to 7x7x7 part way and gave a bulk modulus 135
percent too high. The check therefore warns whenever the grid is the automatic
one; set `kpoints` explicitly and keep it fixed when comparing strained, scaled
or relaxed cells. See [ACCURACY_BENCHMARKS.md](ACCURACY_BENCHMARKS.md).

Stress converges more slowly with the plane-wave cutoff than energy: copper at
its PBE equilibrium reads -7.3 GPa at 400 eV and -0.02 GPa at 600 eV. The check
warns whenever stress is requested below 600 eV, which includes variable-cell
relaxation.

* **Open directions** end at the cell faces, where the grid imposes a zero
  boundary condition. Atoms must lie inside the cell along every open axis,
  with at least 2.5 A of vacuum on each side; below 4 A a warning says the
  result may still depend on the box.
* **Plane waves** are periodic in every direction by construction, so they are
  refused for anything but a bulk cell: GPAW would silently make a molecule
  periodic.
* **Slabs** need the vacuum axis perpendicular to the surface plane. The
  dipole-layer correction cancels the artificial field across the vacuum of an
  asymmetric slab; without it a warning is given.
* **Charged systems.** A charged cluster is solved with open boundaries; the
  moment-corrected solver removes its monopole and dipole analytically before
  the grid solve. A charged bulk cell requires the explicit
  `uniform-background` policy and carries a warning with the leading
  Makov-Payne estimate of its finite-size error, which is not corrected. A
  charged slab or wire is refused: with zero potential at the vacuum faces it
  would model charged plates, not an isolated charged surface. Only whole net
  charges are accepted. When the net charge differs from the formal charges
  labelled on the atoms, a warning says so; only the net charge enters the
  Kohn-Sham calculation.
* **Spin.** An odd electron count in a finite system, or with fixed
  occupations, cannot run spin-paired: that would put half an electron in each
  spin channel of one state. Spin-paired smeared occupations with an odd count
  per periodic cell are allowed, with a warning that no magnetic state was
  explored. Non-zero initial moments with spin polarisation off are refused
  rather than ignored. When spin polarisation defaults on because of an odd
  count, one Bohr magneton is shared over the atoms as the starting moment.
* **External field.** A uniform field is the gradient of a linear potential,
  which cannot be periodic, so it must be perpendicular to every periodic
  lattice vector: any direction for a cluster, the surface normal for a slab,
  perpendicular to the axis for a wire, never in a bulk cell and never in
  plane-wave mode. GPAW itself accepts a field along a periodic direction and
  builds a sawtooth; Materia refuses it. Fields above 2 V/A are refused
  because electrons would tunnel into the vacuum and the box would hide it.

## Observables

A converged run stores, under `dft::<run_id>::<quantity>`:

| Quantity | Value | Unit |
| --- | --- | --- |
| `energy` | free energy E - TS; extra: zero-width extrapolation, energy per atom, reference energy, contributions | eV |
| `forces` | per atom, also by atom id; largest and net force | eV/A |
| `stress` | 3 x 3 tensor (pw); pressure | eV/A^3, GPa |
| `density` | all-electron density including frozen cores on GPAW's fine grid | electrons/A^3 |
| `spin_density` | all-electron n_up - n_down | electrons/A^3 |
| `electrostatic_potential` | electrostatic potential energy of an electron, -e phi | eV |
| `eigenvalues`, `occupations` | [spin, IBZ k-point, band]; k-points and weights; band edges and Kohn-Sham gap | eV, 0 to 1 |
| `fermi_level` | Fermi level; highest occupied and lowest unoccupied levels | eV |
| `magnetic_moment` | total; local moments in atomic spheres, indicative only | mu_B |
| `charge_accounting` | nuclear charge, frozen-core and valence electrons, net charge, occupied electrons, density integral, spin up and down, background charge | e |
| `scf_history` | per iteration: energy, free energy, energy change, density and eigenstate residuals, magnetic moment, criteria met | as named |
| `run` | the run record: status, specification, observables, array summaries, log tail | |

Energies are relative to the sum of the PAW datasets' reference atomic
energies, which is GPAW's convention: only differences between calculations
with the same datasets mean anything.

The three grids and the eigenvalue and occupation tables are stored in the
project's chunked array store (`arrays/<key>/`, see
[PROJECT_FORMAT.md](PROJECT_FORMAT.md)), never inside a JSON result. Grid
point (i, j, k) sits at fractional coordinates (i/N1, j/N2, k/N3).

Every result records the model (`external:gpaw/ground-state-dft`) and tier
(`tier3-external-first-principles`), the complete specification and its
digest, the exact GPAW input and the parameters GPAW reported, the boundary
conditions, the convergence record, units, every PAW dataset with its SHA-256,
pinned and reported software versions, grid sizes, band and k-point counts,
symmetry operations, whether a restart was used, the approximations, the
warnings, and the classification `computational-prediction`. Whether a result
is **current** is computed when it is read.

**A calculation that does not converge yields no value.** Its record is stored
with origin `unsupported`, the reason, the SCF iterations it completed and the
log tail, and nothing else. The same applies to a failed or cancelled run.

## Transactions

* **Frozen input.** The specification holds the geometry, so a run reads
  nothing from live state after it starts.
* **Nothing is applied to a structure.** A ground-state run observes a
  structure; it never changes coordinates, forces or anything else, and never
  records an undo point. It adds one line to the project history, marked as
  not undoable so undo and redo pass over it.
* **Refusal** returns every reason, keyed to the variable concerned, and
  starts no process and stores nothing.
* **Cancellation** stops the GPAW process group within about 0.1 s and stores
  the cancelled record with no value.
* **Staleness.** A run is current while its structure still has the geometry
  digest frozen into its specification: ids, elements, positions, cell,
  periodicity, formal charges and magnetic moments. Any change makes it stale,
  with the change named; undo makes it current again. An isotope change leaves
  it current and says why. A run on a structure the project does not hold is
  `detached`. A run on an inactive structure is judged against its own slot.
* **Ownership.** A job belongs to the project it was started from. Replacing
  the project cancels it, and the outgoing project keeps the cancelled record.
* **Persistence.** Save and reopen preserve every result, array (bit for
  bit, checked by SHA-256), warning, convergence record and SCF history.
* **Restart data.** With restart reuse enabled, a converged state is written
  to a bounded cache on this machine (`~/.materia/dft-restart`, or
  `MATERIA_DFT_RESTART_DIR`), filed under a fingerprint of the atoms,
  discretisation, k-points, bands, electron count, spin setup, iteration
  limit, Poisson solver, GPAW version and dataset hashes. A later run finds it
  only if every one of those matches, so incompatible restart data is never
  used. GPAW accepts a changed functional, occupations, tolerances or field on
  a restarted calculator, and those are the variables that may differ. Restart
  files are not part of the project.

## Convergence laboratory

`convergence_study` varies one parameter over two to eight values, runs each as
a full ground-state calculation, keeps every run, and reports:

| Parameter | Unit | More accurate | Applies to |
| --- | --- | --- | --- |
| `grid_spacing_A` | A | smaller | fd, lcao |
| `cutoff_eV` | eV | larger | pw |
| `kpoint_density_A` | A | larger | periodic |
| `vacuum_A` | A per side | larger | cluster, wire, slab |
| `supercell` | repetitions | larger | periodic, neutral; k-points divided by the repetition |
| `n_bands` | bands | larger | all |
| `smearing_eV` | eV | smaller | smeared occupations |

Observables: energy per atom (default), total energy, largest force, Fermi
level, magnetic moment, pressure. The study reports the value at each point,
the change between successive points, the deviation of each from the most
accurate one, the **residual** (the change between the two most accurate
points) and whether it is within the tolerance chosen. A study with a point
that did not converge is incomplete and cannot meet its tolerance. It runs as
one cancellable background job; cancelling keeps the finished runs.

Only a smearing study reuses restart data from point to point; every other
parameter changes the discretisation or the atoms, so each point starts
fresh, and the study records which.

A study that meets its tolerance becomes **convergence evidence** for any
run whose specification differs from the study's base at most in that
parameter and sits at a value at least as accurate as the one from which the
study found the observable settled. Each result lists its evidence, or says
it has none. The residual is never used as an uncertainty of the physical
result: the functional, the frozen cores and every other approximation keep
their own errors.

## Result classifications

`materia/provenance/classification.py` defines the vocabulary every claim will
use: observation, computational prediction, candidate relation, empirical
invariant, conjecture, independently reproduced, experimentally supported.
Each has an evidence rule checked when a claim is stored (for example, an
independent reproduction must come from a different model or different
software). There is no theorem classification, and a statement that speaks of
a proof, a theorem or a law is refused: a theorem needs a formal proof. Claims
are saved with the project. DFT results are classified as computational
predictions. No equation-discovery engine exists; these are the structures one
will need.

## Interface

**Solvers > First principles: ground-state DFT (GPAW)** shows the code and the
datasets separately, then four groups of variables: physical system, electronic
state, numerical accuracy and requested observables, each control with its
unit and a one-line explanation, advanced settings folded away. The service
checks the specification as it is edited; each refusal appears beside its
control in red, warnings in amber, and **Run ground state** stays disabled
while anything refuses. The last run shows its status, whether it is current,
the energies, forces, pressure, Fermi level, gap, moment, electron accounting,
the SCF residual history, a planar average of any stored grid, warnings,
convergence evidence, and a **Provenance** inspector with the full record,
the PAW datasets, the approximations and the exact GPAW input. The
convergence laboratory runs a study and plots the observable against the
parameter with the verdict. Calculated and neutral states use Materia blue;
amber and red mark only warnings and failures.

## Python

```python
spec = dft.experiment(structure, xc="PBE", grid_spacing_A=0.18,
                      charge_e=1, observables=["energy", "forces", "density"])
dft.check(spec)["blocking"]             # every refusal, [] when it may run
dft.describe(spec)                      # sections, units, explanations
run = dft.ground_state(spec=spec)       # dict of Results, stored in the project
run["energy"].value, run["charge_accounting"].value["balanced"]
dft.array(name="density").data          # the stored grid
dft.state()                             # current, stale or detached, and why
dft.change(spec, smearing_eV=0.05)      # a new specification
study = dft.convergence_study(structure, parameter="cutoff_eV",
                              values=[300, 400, 500], tolerance=1e-3)
study.value["verdict"], dft.evidence()
claims.record("...", "computational-prediction", evidence=["dft::<run>::energy"])
```

## Service and HTTP

`dft/status`, `dft/spec`, `dft/run`, `dft/runs`, `dft/result`, `dft/array`,
`dft/study`, `dft/study/result` and `claims/list`, all through the same
service methods the interface and the Python API use.

## Validation

See [VALIDATION.md](VALIDATION.md#ground-state-dft-experiments).

## Limitations

* GPAW only, serial, collinear spin, scalar-relativistic datasets, no
  spin-orbit coupling.
* Only the datasets GPAW installs: LDA, PBE, RPBE and revPBE here. PBEsol and
  BLYP are refused because no datasets exist for them, not substituted.
* A ground state does not move the atoms. Relaxation of positions, and of the
  cell for bulk plane-wave calculations, is the relaxation experiment built on
  this one ([DFT_RELAXATION.md](DFT_RELAXATION.md)). No DOS or band-structure
  views, no vibrations, no excited states: those are the next phases in
  [ROADMAP.md](ROADMAP.md), in dependency order.
* Charged slabs and wires are refused; charged bulk cells carry an uncorrected
  finite-size error.
* The static field is uniform and restricted to open directions; no
  dielectric response is computed.
* Energies are relative to GPAW's reference atoms.
* Restart data is a local cache, not saved with the project.
* The fine-grid arrays are capped at 256 MB each; larger requests are refused.
* A converged calculation is converged with respect to its own settings only;
  convergence with respect to grid, cutoff, k-points, vacuum and supercell is
  shown only by a convergence study.
