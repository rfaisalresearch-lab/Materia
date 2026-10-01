# Requirement matrix

Status of every meaningful requirement of the original specification, judged
from code, tests and interface behaviour rather than from other documents.
Statuses: **done** (implemented and verified), **partial**, **missing**,
**refused** (intentionally unsupported, with the reason in the code), and
**blocked** (needs an external dependency or a licensed dataset).

Updated as each capability passes its quality gate. Last audit: 2026-09-30.

Current inventory: 47 done, 13 partial, 7 missing, 4 refused and 1 blocked
across 72 rows. These counts are not an overall product-completion percentage.
The rows differ radically in scope, several narrow infrastructure components
have their own rows, and entire missing scientific domains may occupy one row.
The earlier 72.1% headline and the claim that the first release was 100%
complete are retracted. Service-level acceptance tests do not prove that every
end-user workflow is complete or discoverable in the actual interface. See
`docs/CAPABILITY_AUDIT.md` for the corrected interpretation.

## Scale and structure

| Requirement | Status | Evidence |
|---|---|---|
| Wafer, microstructure, nano, lattice, atomic scales | done | `multiscale/wafer.py`, `multiscale/region.py`, `tests/unit/test_structure_builder.py` |
| Region-of-interest atomistic generation linked to the wafer | done | `multiscale/region.py`, `project_format/project.py:extract_region` |
| 21 starter materials, open text definitions, schema | done | `materials/library/*.json`, `materials/schema.py`, `tests/unit/test_materials.py` |
| Diamond, zinc blende, wurtzite, fcc, bcc, hexagonal, layered | done | `structure_builder/primitive.py` |
| Amorphous structures | missing | SiO2 ships as alpha-quartz only |
| Si(100)-2x1 reconstruction | done | `structure_builder/reconstruction.py`, `tests/unit/test_reconstruction.py` |
| Si(111)-7x7, Au(111) herringbone, other reconstructions | refused | declared `implemented: false`, refusal path tested |
| Buckled Si(100) orderings | blocked | needs an electronic total energy (SCC or driven DFT relaxation) |
| Alloys and partial occupancy | partial, built and awaiting the final combined verification phase | exact-composition seeded substitutional alloys, selected sublattices, charged-site refusal and finite-supercell occupancy sampling in `structure_builder/alloys.py` and `python_api/api.py`; SQS optimisation, correlated occupancy and phase stability remain unavailable |
| Grain boundaries, dislocations, stacking faults as atomistic objects | missing | procedural field only |

## Physics and solvers

| Requirement | Status | Evidence |
|---|---|---|
| Tier 0 lattices, neighbours, bonds, PBC | done | `physics/neighbors.py`, `physics/bonds.py` |
| Stillinger-Weber, Lennard-Jones, FIRE, velocity Verlet, Langevin | done | `physics/potentials.py`, `solvers/classical.py` |
| Harmonic vibrations, Gamma-point phonons, dispersion and DOS | partial | Gamma-point modes, thermodynamics and their validation are complete; periodic real-space force constants, explicit q paths, DOS meshes, array persistence and Python API are built in `physics/phonon_dispersion.py` and await the final combined verification phase; LO-TO splitting, anharmonicity and DFT force sources remain unavailable |
| Minimum-energy paths and reaction barriers | partial, built and awaiting the final combined verification phase | fixed-cell NEB, IDPP or linear interpolation, climbing image, FIRE optimisation, endpoint stationarity checks, persistent paths and Python retrieval in `physics/neb.py` and `python_api/api.py`; DFT scheduling, parallel images, adaptive images and rate theory remain unavailable |
| Coulomb interactions, long-range electrostatics | done | `physics/electrostatics.py`, `solvers/electrostatics.py`, `tests/unit/test_electrostatics.py`, `tests/validation/test_electrostatics_validation.py` |
| Rigid-ion relaxation (Coulomb plus short-range repulsion) | done for fixed-cell relaxation and dynamics; one shipped parameterisation (BKS silica), others from Python with a stated source | `physics/rigid_ion.py`, `solvers/classical.py`, `tests/unit/test_rigid_ion.py`, `tests/integration/test_rigid_ion_service.py`, `tests/validation/test_rigid_ion_validation.py` |
| EAM or other many-body metal potential | done for Cu, Au, W | `physics/eam.py`, `potentials/eam/`, `tests/unit/test_eam.py`, `tests/integration/test_eam_service.py`, `tests/validation/test_eam_validation.py`; other metals still Lennard-Jones only |
| Atomistic collision setup, NVE run, persistent playback, frame velocities and fragment inspection | done for EAM-covered systems | `experiments/collisions.py`, `solvers/classical.py`, `desktop_ui/service.py:eam_trajectory`, `tests/unit/test_collisions.py`, `tests/integration/test_eam_service.py`; LAMMPS trajectories replay through the same player (`tests/integration/test_lammps_runs.py::test_collision_setup_replays_through_the_shared_contract`) |
| Tight binding, DOS, LDOS, band structure, simplified STM | done | `solvers/tight_binding.py`, `microscopy/stm.py` |
| Self-consistent charge, charged systems, band bending | partial | charged clusters and charged bulk cells (uniform background) run self-consistently through the DFT experiment (`experiments/dft/spec.py`, `tests/validation/test_dft_ground_state_finite.py`); SCC tight binding and band bending are missing, charged slabs and wires are refused |
| GPAW single-point energy and forces | done | `solvers/gpaw_driver/`, `tests/integration/test_gpaw.py`, `tests/validation/test_gpaw_physics.py` |
| GPAW ground state: density, spin density, electrostatic potential, eigenvalues, occupations, Fermi level, magnetic moment, stress, charged and spin-polarised systems | done | `experiments/dft/`, `solvers/gpaw_driver/ground_state_worker.py`, `tests/unit/test_dft_experiment.py`, `tests/integration/test_dft_experiment_service.py`, `tests/validation/test_dft_ground_state_*.py` |
| GPAW fixed-cell geometry relaxation (BFGS or FIRE, fixed atoms, symmetry preserve or off, charge, spin, field, k-points, smearing and SCF tolerances kept) | done | `experiments/dft/relaxation.py`, `experiments/dft/records.py`, `tests/unit/test_dft_relaxation.py`, `tests/integration/test_dft_relaxation_service.py`; live GPAW 25.7.0 cases in `tests/validation/test_dft_relaxation_live.py` pass here |
| GPAW variable-cell relaxation (FrechetCellFilter, strain mask, hydrostatic strain, target pressure) | done for bulk crystals in plane-wave mode; refused for slabs, wires, clusters, charged cells and fixed atoms | same files; live Si lattice constant case passes here |
| GPAW DOS, PDOS, LDOS, band structure | done | total DOS and PAW-projector PDOS (`experiments/dft/dos.py`, `docs/DFT_DOS.md`), band structure along stored explicit paths (`experiments/dft/bands.py`, `docs/DFT_BAND_STRUCTURE.md`) and spatially resolved LDOS with Tersoff-Hamann constant-height and constant-current images (`experiments/dft/ldos.py`, `docs/DFT_LDOS.md`), each with unit, integration and live GPAW validation; the LDOS map is bit-identical to ASE's `STM` class on the same job (`tests/validation/test_dft_ldos_live.py`). Orbital character (fat bands) is not offered and not part of this row |
| LAMMPS discovery: exact executable or Python module, version, packages, styles; fail closed with an install path | done | `solvers/lammps/environment.py`, `tests/unit/test_lammps.py` |
| LAMMPS frozen, versioned, unit-explicit run specification with refusals keyed by variable | done | `solvers/lammps/spec.py`, `tests/unit/test_lammps.py` |
| LAMMPS energy, forces, stress, fixed-cell relaxation, NVE, Langevin and Nose-Hoover MD with EAM (eam/alloy) | done against a deterministic fake process; live parity blocked | `solvers/lammps/`, `tests/integration/test_lammps_runs.py`; `tests/validation/test_lammps_live.py` skips as BLOCKED because no LAMMPS is installed here |
| LAMMPS native input, out-of-process run, progress, cancellation, time limit, bounded logs, command and hash audit | done | `solvers/lammps/native.py`, `runner.py`, `run.py`, `tests/integration/test_lammps_runs.py` |
| LAMMPS output validation: truncation, non-finite values, atom loss, id, type and mass changes, version, exit status | done | `solvers/lammps/run.py`, `parse.py`, `tests/integration/test_lammps_runs.py` |
| LAMMPS transactional storage, one undoable apply, current, stale and detached state, persistence, HTTP and Python paths | done | `solvers/lammps/records.py`, `desktop_ui/service.py:lammps_*`, `python_api/api.py:LAMMPSNamespace`, `tests/integration/test_lammps_service.py` |
| LAMMPS pair styles other than eam/alloy, barostats, variable cell, MPI and accelerator packages | refused | declared and refused in `solvers/lammps/spec.py:check` |
| QE, GROMACS, ORCA, AMS, CP2K, PySCF, Psi4, Wannier90, OpenMX | refused | detected and declared, not driven (`solvers/external.py`, `docs/EXTERNAL_SOLVER_BACKENDS.md`) |
| Higgs, hadron, nuclear or detector collision physics | refused | outside atomistic-model scope; requires dedicated event-generator and detector-transport backends |

## Microscopy

| Requirement | Status | Evidence |
|---|---|---|
| Constant-current and constant-height STM, Tersoff-Hamann | done | `microscopy/stm.py`, `tests/unit/test_microscopy.py` |
| Contact, non-contact and FM-AFM | partial | `microscopy/afm.py`; forces are Lennard-Jones plus Hamaker, no chemical bonding |
| Noise, drift, creep, line artefacts, filtering, history | done | `microscopy/noise.py`, `microscopy/scan.py` |
| k-resolved STM integration | missing | small surface-zone grid |
| Tip-apex models, CO tips | missing | tip is a work function plus an apex orbital weighting |
| dI/dV maps, force spectroscopy as records | partial | point spectroscopy and force curves exist; maps do not |

## Interaction

| Requirement | Status | Evidence |
|---|---|---|
| Hover, click, box and lasso selection | done | `desktop_ui/static/js/main.js`, `core_model/selection.py` |
| Select by element, region, plane, neighbour radius | done | `core_model/selection.py`, `desktop_ui/service.py:select` |
| Distance, angle, dihedral, coordination | done | `python_api/api.py:MeasureNamespace`, measure panel |
| Add, remove, substitute, vacancy, interstitial, adatom, move | done | `desktop_ui/service.py:edit`, `structure_builder/defects.py` |
| Apply strain | done | `edit("strain")` |
| Apply force to an atom | partial, built and awaiting the final combined verification phase | stable-id constant forces and harmonic restraints affect classical energy, relaxation, dynamics and Gamma-point modes, persist in structure data, record provenance, support undo and refuse ambiguous NEB or periodic-dispersion use; `physics/external_bias.py`, `python_api/api.py`, `docs/EXTERNAL_MECHANICS.md`; force schedules, moving restraints and UI controls remain unavailable |
| Apply electric field | partial | a static uniform field along open directions in the DFT ground state (`external_field_V_per_A`); refused along periodic directions and in plane-wave mode; no response functions and no field in the classical or tight-binding models |
| Change charge and spin with bookkeeping checks | done | `edit("set_charge")`, `edit("set_spin")` feed the DFT experiment's defaults; parity, moment and background rules in `experiments/dft/spec.py:check`; electron accounting in every result |
| Electron occupation changes | missing | |
| Before and after comparison | partial, backend built and awaiting the final combined verification phase | general stable-id comparison now covers boundary-aware displacement modes, rigid cluster alignment, cell deformation and finite strain, additions, removals, substitutions, heuristic bond changes, persistence and Python retrieval in `physics/structure_comparison.py` and `python_api/api.py`; no general comparison view yet |
| Undo and redo for every mutation | done | `project_format/project.py`, `tests/integration/test_project_history.py` |
| Context menus | missing | |

## Experiments and discovery

| Requirement | Status | Evidence |
| --- | --- | --- |
| Immutable, versioned experiment specification with units, validation, support rules, serialisation, hashing and provenance | done for the DFT ground state | `experiments/dft/spec.py`, `docs/DFT_GROUND_STATE.md` |
| Every variable reaches the real solver input, or is refused | done | `gpaw_parameters`, the worker's parameter echo check (`run.parameter_mismatch`), `tests/unit/test_dft_experiment.py` |
| Boundary classification and physical refusals (vacuum, plane waves, charged cells, spin parity, fields) | done | `experiments/dft/boundary.py`, `spec.py:check` |
| Chunked storage of volumetric data in the project | done | `project_format/arrays.py` |
| Staleness, cancellation, ownership, persistence, restart invalidation | done | `experiments/dft/records.py`, `restart.py`, `tests/integration/test_dft_experiment_service.py` |
| Convergence laboratory | done | `experiments/dft/convergence.py` |
| Result classifications without a theorem status | done | `provenance/classification.py` |
| Equation discovery | missing | deliberately not started; last in the dependency order in `docs/ROADMAP.md` |
| Automated structure search and phase exploration | partial, built and awaiting the final combined verification phase | deterministic fixed-cell basin hopping, FIRE relaxation, Metropolis acceptance, ranked and deduplicated candidate minima, checksummed arrays, project retrieval and optional undoable activation in `physics/structure_search.py` and `python_api/api.py`; variable cell, composition search, symmetry fingerprints and global-optimum proof remain unavailable |
| Uncertainty ensembles and independent-model replication | partial, built and awaiting the final combined verification phase | classical model-force ensembles, pairwise force RMS matrices, within-model energy-change spreads, explicit reference-zero handling, persistence and Python retrieval in `physics/model_ensemble.py` and `python_api/api.py`; calibrated uncertainty, parameter ensembles, DFT replication and experimental coverage remain unavailable |

## Python, persistence and formats

| Requirement | Status | Evidence |
|---|---|---|
| Real Python in restricted, trusted and subprocess modes | done | `python_api/execution.py` |
| Versioned project, save, reopen, checkpoints, recovery | done | `project_format/`, `tests/unit/test_project_and_io.py` |
| XYZ, EXTXYZ, CIF, POSCAR, PDB, LAMMPS data, Cube, CSV, HDF5 | done | `dataio/formats.py` |
| PNG and TIFF scan export | done | `visualization/image.py` |
| NumPy array export | done | verified `.npz` export for structures, stored arrays, results and scans through the UI, service and Python API; `dataio/npz.py`, `tests/unit/test_npz_export.py`, `tests/integration/test_npz_export_service.py` |
| Notebook export | done | deterministic Jupyter notebook with bounded previews, project, structure and array digest verification, atomic writes, hostile-text handling and Python API in `dataio/notebook.py` and `python_api/api.py`; `tests/unit/test_notebook_export.py` |

## Performance and accessibility

| Requirement | Status | Evidence |
|---|---|---|
| 1k, 10k, 100k atom benchmarks, measured | done | `materia/benchmarks.py`, `docs/PERFORMANCE.md` |
| 1,000,000 visible instances | partial | benchmark exists, renderer limit is `MAX_RENDER_ATOMS = 250000` |
| Keyboard navigation, high contrast, reduced motion, scalable text | partial | present in CSS and `main.js`; canvases are not screen-reader navigable |
| Rebindable shortcuts | partial | shortcuts are data in `menus.json`; no in-app rebinding |
