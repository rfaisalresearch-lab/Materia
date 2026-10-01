# Changelog

All notable changes to this project are recorded here. The format follows
Keep a Changelog, and the project uses semantic versioning.

## [Unreleased]

### Added

* Harmonic vibrations (`materia/physics/phonons.py`, `docs/PHONONS.md`):
  finite-displacement force constants from a Materia potential, an ASE
  calculator or a force callback, with two- or four-point central
  differences; translational and rotational sum rules by exact orthogonal
  projection, refused for models that are not translation invariant or have
  fixed atoms; mass-weighted Gamma-point normal modes in THz, cm^-1 and meV,
  eigenvectors, displacement patterns, reduced masses and participation
  ratios; rigid-body, imaginary and unresolved modes classified against a
  measured eigenvalue resolution; zero-point energy only at a true minimum;
  refusal of non-stationary structures and of any wavevector other than
  Gamma. Every classical solver now runs the `phonons` task. Validated against
  the Lennard-Jones dimer, an independent Born-von Karman lattice sum for a
  32-atom fcc supercell, an Einstein solid and ASE's `Vibrations`.
* Rigid-ion potentials (`materia/physics/rigid_ion.py`, `docs/RIGID_ION.md`):
  fixed charges summed by the existing Ewald, slab and cluster electrostatics
  plus Buckingham pair terms, shipped with the BKS silica parameterisation and
  registered as `rigid-ion/bks-silica`, now the recommended relaxation and
  dynamics model for SiO2. The inner maximum of every pair interaction is found
  and any configuration inside it is refused instead of collapsing. Energy,
  forces, fixed-cell FIRE relaxation and dynamics record the energy split, an
  independent Ewald check and each pair's margin to its barrier; the solver
  panel prints them. Validated against a closed-form Born-Mayer crystal; example
  `examples/scripts/19_rigid_ion_silica.py`.
* Spatially resolved DFT LDOS and Tersoff-Hamann STM images
  (`materia/experiments/dft/ldos.py`, `docs/DFT_LDOS.md`): a frozen, versioned
  specification of the energy window about the Fermi level, spin and bands; a
  GPAW non-self-consistent step with point-group symmetry off; the map summed
  from pseudo-wavefunctions on the real-space grid with the PAW sphere radii
  recorded; verification of every echoed parameter, the state count against
  the returned eigenvalues, the map's integral against pseudo-norms, band
  coverage of the window and empty windows refused; constant-height and
  constant-current images limited to the vacuum above the PAW spheres; Python,
  service, HTTP and UI paths with a plane viewer, image tool, Cube and CSV
  export and provenance; example `examples/scripts/18_dft_ldos_stm.py`. Live
  validation is bit-identical to ASE's `STM` class on the same GPAW job.
* DFT equation of state (`materia/experiments/dft/eos.py`,
  `docs/DFT_EQUATION_OF_STATE.md`): ground states at 5 to 15 isotropically
  scaled volumes of a bulk crystal, all with the reference's pinned k-point grid
  and cutoff, verified point by point; a third-order Birch-Murnaghan fit kept
  only if every point converged, the minimum is inside the range, B0 and B' are
  physical and the fit reproduces every point; GPAW's stress compared with the
  fitted pressure and forces reported for unrelaxed internal parameters;
  checksummed storage with current, applied, stale, detached and corrupt
  states; the fitted equilibrium applied as one undoable change; Python,
  service and HTTP paths; energy and pressure plots, CSV export and a
  provenance inspector; example `examples/scripts/17_dft_equation_of_state.py`.
* Bond perception keeps a covalent-distance candidate only when the pair shares
  a Voronoi face of at least 5 percent of the solid angle, removing the Al-Al
  contacts of corundum, the metal-metal contacts of MoS2 and WSe2 and the second
  shell of bcc tungsten; every shipped crystal now shows its textbook
  coordination. `method="distance"` keeps the previous rule.
* Adatoms without an explicit height are placed at the covalent contact with the
  nearest surface atom instead of 2.0 A above the surface for every element.
* The ground-state check warns when stress is requested below 600 eV: the
  equation-of-state tool found copper's stress 7.3 GPa off at 400 eV, confirmed
  directly in GPAW.
* GPAW Kohn-Sham band structure (`materia/experiments/dft/bands.py`,
  `docs/DFT_BAND_STRUCTURE.md`): a frozen, versioned specification pinning the
  ground state, its symmetry choice and an explicit reciprocal-space path whose
  special points are generated once from ASE's Bravais-lattice tables and
  stored with every k-point; valid for bulk, slab and wire, refused for
  clusters; GPAW's fixed-density step on exactly those k-points with symmetry
  off; verification of every echoed parameter, the k-points in order, labels,
  breaks, band count, finite ordered eigenvalues and the path distance
  recomputed from the pinned cell; band edges along the path; transactional,
  checksummed storage with current, stale, detached and corrupt states;
  Python, service and HTTP paths; a desktop band plot with symmetry labels,
  segment dividers, Fermi level, spin selection, zoom, k-point table, CSV
  export and a provenance inspector; example
  `examples/scripts/16_dft_band_structure.py`. Five live GPAW cases cover
  silicon's indirect gap, endpoint agreement with an independent ground state,
  spin-polarised iron, a graphene sheet and a relaxed crystal. Orbital
  character is deferred.
* GPAW total DOS and PAW-projector PDOS (`docs/DFT_DOS.md`): a frozen,
  versioned spectral specification for a structure, stored ground state or
  stored relaxation; explicit energy reference, window, grid, Gaussian or
  tetrahedron integration, spin channels, non-self-consistent k-points, band
  count and atom and angular projections; exact parameter echo checks;
  independent reconstruction of the total DOS from returned eigenvalues and
  weights; refusal of incomplete bands and invalid curves; checksummed arrays,
  current, stale and detached state, persistence, Python and HTTP paths, a
  desktop plot with projections and spin channels, and CSV export. Five live
  GPAW validation cases cover H2, silicon, spin and a relaxed geometry.
* GPAW geometry relaxation and variable-cell relaxation in the DFT experiment
  system (`materia/experiments/dft/relaxation.py`, `docs/DFT_RELAXATION.md`): a
  frozen, versioned relaxation specification built on the ground-state one,
  with the optimiser (BFGS or FIRE), force and stress criteria, step limit,
  step length, symmetry, strain mask, hydrostatic strain and target pressure;
  fixed atoms from the structure; every variable echoed back by GPAW or the
  worker and compared; refusals keyed by variable, including variable cells
  outside bulk plane-wave mode, charged or constrained variable cells, fixed
  atoms with preserved symmetry and charged systems in a field; the step
  history, displacement, cell change and ground-state observables at the final
  geometry stored transactionally; one undoable apply that refuses stale
  geometry, with current, stale and detached state; `dft.relax_*` in Python,
  `/api/dft/relax/*` routes, and a Geometry relaxation group with a
  convergence view in the DFT panel. Validated with live GPAW 25.7.0 runs
  checked by independent ground-state calculations.
* LAMMPS driven end to end for EAM potentials (`materia/solvers/lammps/`,
  `docs/EXTERNAL_SOLVER_BACKENDS.md`): discovery of the exact executable or
  Python module with its version, packages and styles, failing closed with an
  install path; an immutable, versioned, unit-explicit run specification with
  refusals keyed by variable; byte-reproducible native data and input files
  with hashed potential copies; out-of-process runs with progress,
  cancellation, a time limit and bounded logs; strict parsing that refuses
  truncated, non-finite or inconsistent output, atom loss and id, type or mass
  changes; energy, forces, stress, fixed-cell relaxation and NVE, Langevin and
  Nose-Hoover dynamics; transactional storage, one undoable apply that refuses
  stale geometry, and current, stale and detached state; the `lammps` Python
  namespace, `/api/lammps/*` routes and a LAMMPS backend in the EAM panel whose
  trajectories replay in the existing player. Tested against a deterministic
  fake process; live parity with a real LAMMPS is blocked until one is
  installed.
* Ground-state DFT experiments through GPAW (`docs/DFT_GROUND_STATE.md`): an
  immutable, versioned specification of every variable (geometry and isotopes,
  boundary class, net charge and background policy, electrostatic boundary,
  optional uniform field along open directions, spin and initial moments,
  functional, representation, grid or cutoff or basis, k-points, bands,
  occupations, tolerances, requested observables, software and PAW dataset
  identity by SHA-256), each with a unit, an explanation, a support rule and a
  refusal that names it. Every variable reaches GPAW's own input, and the
  parameters GPAW reports back are checked against those sent. Boundary rules
  for clusters, wires, slabs and bulk cells; charged clusters and charged bulk
  cells with an explicit background; odd electron counts never run
  spin-paired. Energies, forces, plane-wave stress, all-electron density, spin
  density, electrostatic potential, eigenvalues, occupations, Fermi level,
  magnetic moment, electron accounting and SCF residual history from real
  runs; a run that does not converge keeps no value. Frozen submissions,
  cancellation, staleness with undo, isotope changes explained, ownership on
  project replacement, persistence, and a bounded restart cache reused only
  when compatible. A convergence laboratory for grid spacing, cutoff, k-point
  density, vacuum, supercell, bands and smearing, with residuals, tolerance
  verdicts and convergence evidence per result. Python `dft.experiment`,
  `dft.ground_state`, `dft.convergence_study` and friends, `/api/dft/*`
  routes, and a Solvers panel organised by physical system, electronic state,
  numerical accuracy and requested observables, with a provenance inspector.
* Chunked, checksummed storage of large arrays in the project file
  (`arrays/<key>/`), used for DFT grids and tables.
* Result classifications (`materia/provenance/classification.py`):
  observation, computational prediction, candidate relation, empirical
  invariant, conjecture, independently reproduced, experimentally supported,
  each with an evidence rule; no theorem classification. Claims persist with
  the project; `claims` namespace in Python.
* The future dependency order after the DFT ground state is recorded in
  `docs/ROADMAP.md`.

* Embedded-atom potentials for Cu, Au and W (`docs/EAM.md`): the Zhou,
  Johnson and Wadley (2004) setfl files, NIST retabulation, from OpenKIM with a
  public-domain dedication, shipped unmodified with SHA-256 checks. setfl
  parser with strict refusal of malformed files, LAMMPS-equivalent cubic
  Hermite interpolation, energy, forces, virial stress, Verlet neighbour list,
  FIRE relaxation and molecular dynamics. Runs work on a copy and apply their
  outcome as one undoable change only to an unchanged structure; cancelled runs
  keep no value. Python `eam` namespace, `/api/eam/*` routes, a Solvers panel
  section showing the file's identity, licence, citation and cutoff, staleness
  tracking, benchmarks, an example script, and 29 validation cases against
  LAMMPS (NIST) and ASE. Copper, gold and tungsten now recommend EAM;
  Lennard-Jones stays available as an approximation.

* Point-charge electrostatics (`docs/ELECTROSTATICS.md`): Ewald summation for
  crystals with tin-foil or vacuum surroundings and an optional, explicitly
  requested neutralising background; Yeh-Berkowitz corrected Ewald for slabs;
  direct sums for clusters; wires, charged slabs and undefined configurations
  refused. Energy, forces, site potentials and site fields, with an
  independent-split convergence check, recorded parameters, progress and
  cancellation. Explicit charge models (per element, per atom,
  formal-point-ion) stored with the structure and assigned undoably. Python
  `electrostatics` namespace, `/api/electrostatics/*` routes, a Solvers panel
  section, per-atom values in the inspector, an example script, benchmarks and
  11 validation cases against published Madelung constants.
* `docs/REQUIREMENTS_MATRIX.md`: the evidenced status of every requirement.

- **GPAW as a genuinely driven Tier-3 solver.** `external:gpaw` now runs real
  projector augmented-wave DFT: single-point total energy and forces, and
  nothing else. It is optional, it is never imported into the Materia process,
  and it runs in its own interpreter as a subprocess, which is what makes
  cancellation possible and what lets it live in a different environment from
  Materia entirely.
- Separate detection of the GPAW **code** and the **PAW datasets**, which fail
  independently. `import gpaw` succeeding is never treated as evidence that a
  calculation can run; an installation with the code and no datasets is
  reported as not operational, with the instruction that fixes it.
- `GPAWSettings`: a validated configuration covering task, functional, mode and
  its cutoff, grid spacing or basis, k-points, occupations and smearing,
  charge, spin, both convergence tolerances and the iteration cap, every
  setting carrying its unit in its name. Presets `smoke`, `molecule` and
  `surface`. Incompatible combinations are refused, never silently corrected:
  plane waves on a cell that is not periodic in all three directions, k-points
  along a vacuum direction, a charge that leaves an odd electron count in a
  spin-paired calculation.
- Four distinct outcomes, converged, not-converged, cancelled, failed, with
  an unconverged calculation returned as an unsupported result carrying its
  iteration count and the tolerance it missed, never as a number.
- Provenance recording GPAW and ASE versions, the interpreter, MPI world size,
  every setting, boundary conditions, SCF iterations, wall time, and the
  identity of each PAW dataset including its MD5 fingerprint.
- `lab.dft` in the Python API, `gpaw/status`, `gpaw/settings` and `gpaw/energy`
  service and HTTP routes, and a solver panel showing availability, live
  settings validation, SCF progress, cancellation and the last run.
- Result and provenance deserialisation, so solver results saved in a
  `.materia` file are read back when the project is reopened. They were written
  before but never loaded.
- `docs/GPAW.md`.

- **Surface reconstructions as a first-class subsystem.** Generators are
  registered against a material prototype and orientation, produce topology
  only, and hand geometry to the material's recommended interatomic model. The
  record written into `structure.info["reconstruction"]` carries the generator,
  the pairing, the relaxation, the measured geometry, a 1×1/2×1 symmetry check
  and a full provenance entry, and round-trips through the project format.
- **Si(100)-(2×1) end to end.** The dimer axis is derived from the slab's own
  back-bond geometry; Stillinger-Weber relaxation gives a 2.4035 Å symmetric
  dimer, 1.68 eV per dimer below the relaxed ideal truncation, independent of
  the construction guess and of cell size.
- **Comparison against published structural data.** Material files may declare
  `reconstructions[].reference_geometry` with a value, uncertainty, method and
  a required citation. `compare_to_reference` reports the signed deviation and
  whether it falls inside the quoted uncertainty. It never corrects a value and
  never asserts agreement.
- `reconstruction_model`, `reconstruction_fmax_eV_A` and
  `reconstruction_max_steps` on `make_surface` and `create_surface`, so the
  relaxation that sets a reconstruction's geometry can be controlled from the
  same call that builds it.
- `Project.record_change`, which records an undo point for a change already
  made in place, so an operation that may refuse logs nothing until it has
  actually done something, and `Project.key_of`, which finds the slot holding a
  structure by identity.
- `Structure.restore_from`, the in-place inverse of `copy`, so an operation
  that mutates a structure can roll itself back without changing identity.
- `Structure ▸ Reconstruct surface…` and `Structure ▸ Reconstruction report`;
  a reconstruction selector and a fixed-layer field in the material Build
  panel; a Reconstruction section in the properties panel.
- Python API: `MaterialHandle.reconstructions()`, `SurfaceHandle.reconstruct()`,
  `.reconstruction()`, `.compare_reconstruction()`, and the
  `reconstruction=` argument to `create_surface` and `make_surface`.
- Service endpoints `structure/reconstruct` and `structure/reconstruction`.
- `UnsupportedRequest`, an `ApiError` subclass carrying the machine-readable
  unsupported record.
- `docs/RECONSTRUCTIONS.md`.
- 25 software tests dedicated to the subsystem, plus guards in the material,
  interface-contract, HTTP and acceptance suites, and 17 physical-validation
  cases for the geometry.
- `examples/scripts/08_surface_reconstruction.py`, a console example, and the
  `si100_2x1_dimer` example project with its STM figure.

### Changed

- A material file's `implemented: true` on a reconstruction is now resolved
  against the generator registry rather than trusted. `silicon.json`'s
  `2x1-dimer` entry previously carried that flag with no generator behind it.
- `gallium_arsenide.json` no longer claims `relaxed-110` as implemented: no
  interatomic potential shipped with Materia covers GaAs, so there is nothing
  to relax the (110) cleavage surface with.
- Reconstruction relaxations default to a 0.005 eV/Å force criterion rather
  than the general 0.02, because the reported quantity is a bond length quoted
  to three decimals.
- A reconstructed slab's provenance no longer states that no reconstruction was
  applied.
- `docs/LIMITATIONS.md`, `docs/VALIDATION.md`, `docs/ROADMAP.md`,
  `docs/SCIENTIFIC_MODELS.md`, `docs/MATERIAL_SCHEMA.md`, `docs/PYTHON_API.md`
  and `README.md` updated to describe what is now generated and what is still
  refused.

### Fixed

* The solver panel's model list sent each solver's display name, which the
  service never recognised, so every named model failed with "Unknown model"
  and only "recommended" worked. It now sends the registry key, lists each
  model once and leaves out solvers that are not available.
* Silica with the "recommended" model was relaxed with Stillinger-Weber silicon
  treating oxygen as a radius-scaled impurity. It now uses the BKS rigid-ion
  model.
* The equation of state fitted GPAW's free energy E - TS of the smeared
  occupations while calling the result a 0 K static-lattice equation of state.
  It now fits the energy extrapolated to zero width, keeps and fits the free
  energy separately, and compares GPAW's stress with -dF/dV of the free-energy
  fit, of which the stress is the derivative. For copper at 0.1 eV the two fits
  differ by 0.06 percent in V0; the difference grows with the smearing width.
  The specification version is now 2.0.
* The 4H-SiC structure had the wrong stacking: two consecutive Si layers sat on
  the same site, putting a carbon atom 0.63 A below a silicon atom and giving
  coordination numbers from 2 to 5. It is now the ABAC (Ramsdell ABCB) stacking
  of P6_3mc with every atom tetrahedral, Si-C 1.886 and 1.890 A. A new
  validation case checks the contacts and first-shell coordination of every
  shipped crystal, which would have caught it; the density check could not.
* Lennard-Jones parameters derived from a material now reproduce its cohesive
  energy and lattice constant with the cutoff and shift actually used. They
  were derived from infinite lattice sums, so the potential as evaluated gave
  gold -3.660 instead of -3.81 eV/atom with its minimum 0.17 percent too wide.
* Tight-binding hoppings are now scaled from bond lengths that equal the
  library lattices exactly. Rounded reference lengths had scaled every hopping
  by up to 0.14 percent: the graphene bandwidth was 16.178 eV instead of
  6|t| = 16.2 eV, and Si, Ge and GaAs bands were off by 2 to 3e-4 eV against an
  independent implementation of Vogl's Hamiltonian, which they now match to
  2e-14 eV.
* The default DFT k-point density is now 30 A instead of 20 A, and the check
  warns whenever the automatic grid is in use. With the old default an
  aluminium energy-volume scan changed grid part way and gave a bulk modulus
  135 percent too high, and copper was 7 percent low even on a fixed grid.
* Supercell convergence studies record whether each point's k-point sampling is
  equivalent to the base cell's, and say so when it is not.
* `DECAY_PREFACTOR_INV_A_SQRT_EV` was half of sqrt(2 m e)/hbar. The STM
  simulator used the correct function, so no image changed.
* `Isotope.neutrons` raised NotImplementedError; isotopes now carry their atomic
  number and return N = A - Z.
* Radioactive elements carried the mass number of their reference isotope as a
  mass; they now carry that isotope's atomic mass (for example Pu-244,
  244.064 u), except Lr, Sg, Rg, Mc and Ts.
* The alpha-quartz dielectric constant was the amorphous thermal-oxide value
  3.9; it is now the crystalline 4.52 perpendicular to c (4.64 parallel), and
  the melting point says it is the melting of silica, not of alpha-quartz.
* Validation tolerances that hid real trends were tightened: the tight-binding
  silicon gap is located by a bounded search (1.1713 eV at 0.731 of Gamma to X,
  not near 0.85 as documented), the STM decay case checks the approach of the
  apparent decay to 2 kappa, and the thermostat case uses 3 percent instead of
  12.

* Undo and redo no longer mislabel an edit when a history line without an undo
  point follows it. Before: editing a structure, running an electronic
  calculation (which logs a line but records no undo point), then undoing,
  reported "Electronic structure note" as undone while actually restoring the
  edit, and left the edit's line marked as not undone. Log entries now carry
  `undoable`, and undo and redo pass over lines without an undo point.

* EAM no longer returns zero energy for coincident atoms. The pair search
  dropped every separation below 1e-9 A, so two atoms at the same point were
  treated as one; atoms 1e-8 A apart gave 1.28e10 eV. Only the true self-pair
  (same index, zero image) is now excluded, and distinct atoms within the
  1e-5 A collision tolerance, including periodic images, are refused on every
  evaluation, including cached-list reuse, with no result, history or
  coordinate change. The generic Solvers panel relaxation and dynamics now
  record their undo point only after success and restore the structure when a
  run is refused part way.

* The shared neighbour list (Stillinger-Weber, Lennard-Jones, tight binding,
  bond perception, strain, measurements) lost neighbours of atoms stored more
  than one lattice vector outside the periodic cell: moving one atom of a
  64-atom silicon crystal by two lattice vectors changed the Stillinger-Weber
  energy from -277.5424 to -268.8692 eV. Positions are now wrapped internally
  for the search, the wrap is folded into each pair's integer lattice shift,
  and the stored coordinates are never changed. The EAM pair search does the
  same. Distinct coincident atoms are refused by every energy model, and bond
  perception leaves them out, instead of either being silently deleted.

* Choosing a model that cannot relax or run dynamics (tight binding, point-charge
  electrostatics) in the Solvers panel now returns a refusal naming the models
  that can, instead of failing with an attribute error.
* Right-dock panels could force the dock 37 px wider than the window, clipping
  the right edge of the Solvers panel. Panels now shrink to the dock width.

- **`external:ase` labelled every result Tier 3.** ASE is an interface and
  computes nothing itself, so attaching an EMT calculator produced a classical
  result stamped as first-principles DFT. The adapter now declares no tier of
  its own and the fidelity is taken from the attached calculator; an
  unrecognised calculator gets no tier at all and says so rather than guessing.
- The placeholder `external:gpaw` adapter advertised sixteen capabilities it
  could not deliver, including band structure, LDOS, charge density, spin and
  charged systems. It is replaced by the real driver, which declares the two it
  actually provides.
- A cancelled or failed background job withheld its result, so the interface
  could not show why a run produced nothing. Results are now available for any
  finished job.

- **Undo and redo now reverse project-level operations, not just coordinates.**
  A snapshot used to hold one structure and a selection, which could not
  express structure creation or removal, a change of active structure, a wafer
  replacement or a region appearing. Undoing a build left the structure in the
  project; undoing a second build left both and kept the second active; undoing
  a wafer left the wafer. Snapshots now carry a `ProjectState`, structure
  keys, active key, selection, wafer, regions, active region and the key
  counter, alongside the deep copy of the one structure whose contents are
  about to change, and every project mutator captures one before it acts.
  Topology-only operations copy no atoms: the state holds references, which is
  sound because every in-place mutation records its own copy and undo unwinds
  in LIFO order.
- Restoring a structure's contents no longer swaps the object in its slot, so a
  key's structure object identity is stable for the life of the project and a
  `SurfaceHandle` still points at its slot after an undo.
- A relaxation that ran but did not converge was reported as `relaxed: true`
  with origin `calculated`, and its provenance called the geometry an energy
  minimum. The record now carries `geometry_status` of `converged`,
  `not-converged` or `unrelaxed`; `relaxed` means converged; a stalled run is
  `estimated` and says so with its residual force; and `compare_to_reference`
  returns `comparable: false` with `within_stated_uncertainty` of `None` on
  every row, so no interface or API path can present a half-relaxed bond length
  as a result checked against experiment.
- `Service.build_surface` registered a slab that `MaterialHandle.create_surface`
  had already registered, leaving two project keys pointing at one object; a
  later undo restored only one of them. A build now produces one entry, and
  `Project.add_structure` returns the existing key rather than aliasing a
  structure it already holds.
- `Service.reconstruct` mutated the active structure without an undo point of
  its own and overwrote the label of the preceding history entry. It now builds
  on a copy, installs it through `Project.set_structure`, and records one
  accurate entry that undo and redo reverse completely, coordinates, roles and
  the `reconstruction` and `surface` metadata together.
- `SurfaceHandle.reconstruct` added a history entry and an undo point before
  attempting the work, so a refused request left both behind. The undo point is
  now recorded after the change succeeds, via `Project.record_change`, and a
  refusal of any kind leaves the structure and the history untouched.
- `Project.record_change` looked up the *active* structure rather than the one
  that had actually changed, so reconstructing a handle the project did not
  hold wrote an undo point against whatever was active; undoing it then
  replaced that structure with the unrelated one. Undo points are now bound by
  object identity to the slot that was changed, snapshots carry that slot, and
  a detached handle records nothing at all.
- `SurfaceHandle.reconstruct` could leave a slab half-reconstructed. An invalid
  `steps` or `fmax` raised out of the middle of the minimisation, and a
  generator or solver failure left moved atoms and renamed roles behind with no
  record of them. The relaxation controls are validated before anything is
  mutated, and the whole operation now runs under a restore point that puts
  positions, roles, labels, fixed flags, bonds, cell and `info` back on any
  failure.
- Two Python-console example snippets had an unmatched parenthesis and raised a
  `SyntaxError` when run. A contract test now parses every shipped snippet and
  every example script.

## [0.1.0]

First release: the complete vertical slice from wafer to atom.

### Added

**Application**
- Native desktop window (WKWebView on macOS, WebView2 on Windows, WebKitGTK on
  Linux) with a native menu bar and native file dialogs.
- Double-clickable macOS application bundle, built by
  `tools/make_macos_app.py`.
- `materia serve` for headless machines, and a command line with `info`,
  `materials`, `run` and `bench`.

**Science**
- 21 material definitions with lattices, bases, orientations, terminations,
  declared reconstructions and properties cited to primary literature.
- Exact crystallography: primitive-lattice detection, Miller-indexed surfaces
  in the conventional-cell convention, automatic stable-termination selection.
- Procedural wafer model, deterministic in (seed, position), with
  region-of-interest extraction.
- Stillinger-Weber and Lennard-Jones potentials with analytic forces, FIRE
  minimisation, velocity-Verlet and BAOAB Langevin dynamics.
- Orthogonal sp3s* and pz tight binding with band structures, densities of
  states, site-projected LDOS and substitutional impurities.
- Tersoff-Hamann scanning tunnelling microscopy with a full instrumental noise
  chain; classical-force atomic force microscopy with Giessibl frequency-shift
  conversion.
- Adapters for GPAW, Quantum ESPRESSO, LAMMPS, CP2K, PySCF, Psi4, Wannier90,
  OpenMX and ASE, each reporting availability and preserving refused requests.

**Interface**
- Multiscale viewports: wafer, microstructure, WebGL atomic model, probe image,
  with a magnification callout linking every scale back to the wafer.
- Atom inspector with identity, nucleus, electrons, orbitals, educational shell
  diagram, bonds, local density of states, energy, probe response, solver record
  and reproducing Python.
- Embedded Python console executing real Python in restricted, trusted or
  subprocess modes.
- Versioned `.materia` project format with history, checkpoints, undo and redo.
- Import and export: XYZ, extended XYZ, CIF, POSCAR, PDB, LAMMPS data, CUBE,
  CSV, PNG, 16-bit TIFF, NPZ and HDF5.
- Plug-in system with a reference implementation exercising every extension
  point.

**Honesty machinery**
- A provenance record on every result: model, fidelity tier, origin,
  approximations, tolerances, boundary conditions, convergence, references,
  seed and software version.
- Explicit refusal with named alternatives and preserved requests wherever a
  model is out of its domain.

### Fixed during development

- Minimum-image convention by rounding failed for 60° hexagonal surface cells;
  now searches neighbouring images.
- Surface construction enumerated conventional rather than primitive lattice
  vectors, producing surface cells twice the correct size.
- Si(111) slabs were cut on the unstable side of the bilayer; termination is
  now selected by broken-bond count.
- The AFM frequency-shift integral used second-kind Chebyshev nodes with
  first-kind weights, under-estimating Δf by (n−1)/(n+1).
- α-quartz was generated in the wrong space-group setting, giving Si-O
  distances of 1.29 and 1.87 Å instead of 1.604 and 1.613 Å.
- Isotope tables were incomplete for eleven elements, so abundances did not sum
  to one and weighted atomic masses were wrong.
- Material lookup by formula was order-dependent when two materials shared one.
- Non-finite floats produced JSON no strict parser would accept.
