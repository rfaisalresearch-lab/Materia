# External solver backends

Materia is a workbench and provenance layer. It should make established solvers
faster to set up, compare, inspect and reproduce. It should not reimplement
decades of validated numerical physics under a new name.

The integration contract for every backend is the same:

1. Detect the exact executable or Python environment.
2. Validate the structure, boundary conditions and requested variables.
3. Freeze an immutable input specification with units and file identities.
4. Generate the native input without silently changing the scientific request.
5. Run with progress, cancellation and captured logs.
6. Parse convergence and results, including trajectories and volumetric arrays.
7. Record executable version, potential, basis, pseudopotential and input hashes.
8. Refuse unsupported combinations instead of substituting another model.

Detection alone is not execution. GPAW, LAMMPS and a user-supplied ASE
calculator are driven end to end; the other entries below remain declared and
detected but are not driven.

| Backend | Scientific role | Materia priority | Current state |
| --- | --- | --- | --- |
| LAMMPS | Scalable materials MD, reactive and many-body potentials, MPI and accelerator packages | First classical adapter | Driven for EAM (eam/alloy): energy, forces, stress, fixed-cell relaxation, NVE, Langevin and Nose-Hoover MD. See below |
| GROMACS | Biomolecular MD, constraints, ensembles, free energy and GPU throughput | First biomolecular adapter | Detected, not driven |
| Quantum ESPRESSO | Periodic plane-wave DFT, relaxation, molecular dynamics, NEB and phonons | First periodic DFT adapter after GPAW | Detected, not driven |
| ORCA | Molecular HF, DFT, correlated wavefunction methods, excited states and spectroscopy | First molecular quantum chemistry adapter | Detected, not driven |
| ASE | Common Python atoms, calculator and workflow interface | Interoperability layer | Energy and relaxation driven with a supplied calculator |
| Amsterdam Modeling Suite | Licensed molecular, periodic, DFTB, ReaxFF and workflow engines | Optional licensed adapter | Detected, not driven |
| Materials Studio | Licensed multiscale graphical environment and industrial workflows | File and provenance interchange first | Not integrated |

The priorities follow the official capability descriptions. LAMMPS documents
classical MD, many interaction models, ensembles, minimizers, parallelism,
restart and trajectory output. GROMACS documents molecular dynamics,
constraints, free-energy methods and CPU/GPU execution. Quantum ESPRESSO
documents plane-wave DFT, structural optimization, molecular dynamics, NEB and
phonons. ORCA documents energies, gradients, optimization, molecular dynamics,
DFT, post-HF methods and spectroscopy. ASE documents a shared Atoms and
Calculator interface plus optimization, dynamics, NEB, phonons and analysis.

Primary documentation:

- LAMMPS: https://docs.lammps.org/Intro_features.html
- GROMACS: https://manual.gromacs.org/current/reference-manual/algorithms/
- Quantum ESPRESSO: https://www.quantum-espresso.org/what-can-qe-do/
- ORCA: https://www.faccts.de/docs/orca/6.1/manual/
- ASE: https://wiki.fysik.dtu.dk/ase/about.html
- Amsterdam Modeling Suite ReaxFF: https://www.scm.com/amsterdam-modeling-suite/reaxff/
- BIOVIA Materials Studio: https://www.3ds.com/products/biovia/materials-studio

## Researcher-time target

Materia should save time in the parts that remain repetitive across solvers:
structure preparation, unit-safe variables, support checks, job templates,
progress, comparison, staleness, provenance, trajectory inspection and export.
It should expose the native input and log at every step so an expert can audit
or reproduce the calculation outside Materia.


## LAMMPS

The LAMMPS adapter is in `materia/solvers/lammps/`. It follows the contract
above in full for the embedded-atom potentials Materia ships or a user
supplies as setfl files. Everything listed under "Driven" has been exercised
through a deterministic fake process in the test suite; live parity against
a real LAMMPS is described under "Validation status".

### 1. Discovery

`lammps.status()` (Python) and `/api/lammps/status` (HTTP) report the LAMMPS
found, never an assumed one. Materia looks, in order, at
`MATERIA_LAMMPS_EXECUTABLE` (an executable; exclusive when set),
`MATERIA_LAMMPS_PYTHON` (an interpreter that imports `lammps`; exclusive when
set), the executables `lmp`, `lmp_serial`, `lmp_mpi`, `lmp_mac`,
`lmp_mac_mpi`, `lmp_omp` and `lammps` on `PATH`, and finally the `lammps`
module of the interpreter running Materia. Each candidate is asked directly:
an executable with `lmp -h` (version banner, git information, installed
packages and every compiled style), a module in a separate process
(`lammps.version()`, `installed_packages`, `available_styles`, and the
version header of a probe log). The exact version string, for example
`29 Aug 2024 - Update 1`, is recorded, as are the path, the route and every
candidate tried. An explicit choice that does not answer like LAMMPS is
reported, not replaced by another installation. With nothing found the
status says so and gives the install path: `conda install -c conda-forge
lammps`, or a source build with the MANYBODY package, then `PATH` or one of
the two variables. Materia's own EAM solver is never used in its place.

### 2. The frozen specification

`LAMMPSRunSpec` (`spec.py`, schema `materia.lammps.run`, version 1.0) is
immutable and hashed (SHA-256 of its canonical JSON). It holds:

| Part | Content |
| --- | --- |
| Structure | stable atom ids, atomic numbers, mass numbers, roles, positions (A), velocities (A/fs), fixed atoms, formula and the structure fingerprint |
| Cell and boundaries | cell matrix (A), periodicity, LAMMPS boundary (`p` periodic, `s` shrink-wrapped) |
| Atom types | one type per (element, mass) pair, so isotopes keep their masses; type, element, Z and mass (u = g/mol) |
| Potential | style `eam/alloy`, identifier, file name, path, SHA-256, size, elements in file order, cutoff, licence, citations |
| Units | `metal`, atom style `atomic` |
| Task | `energy`, `relax` or `md` |
| Relaxation | `min_style` (`cg`, `fire`, `sd`), `ftol_eV_A` (largest force on any mobile atom, `min_modify norm max`), `etol`, `max_iterations`, `max_evaluations` |
| Dynamics | `steps`, `timestep_fs`, `ensemble` (`nve`, `langevin`, `nvt`), `temperature_K`, `damping_fs`, `seed`, `initial_velocities` (`keep`, `create`), `pressure_bar`, `sample_every` |
| Numerics and output | `neighbor_skin_A`, `thermo_every`, `outputs` (`energy`, `forces`, `stress`, `thermo`, `trajectory`) |
| Installation | route, path, exact version, version number, git information, packages |

Settings that do not apply to the task are stored as `None`. Unknown or
mistyped settings raise `SpecError`. `check()` returns every refusal keyed by
variable, among them: a unit or atom style other than `metal` and `atomic`;
atom ids outside 1 to 2^31 - 1 or repeated; a type without a positive finite
mass; an element the potential file does not describe; a potential file whose
SHA-256 no longer matches; a cell that is not in the LAMMPS restricted
triclinic orientation (Materia does not rotate silently), a tilt beyond half
a box length, or a tilt involving a non-periodic direction; stress requested
for a cell that is not periodic in all three directions; a trajectory outside
`md`; a non-zero `pressure_bar` (no barostat: the cell is fixed); a
thermostat or created velocities at 0 K; a missing seed; `steps` not a
multiple of `sample_every`; a trajectory above 20,000 frames or 1 GiB; fixed
atoms with velocities; LAMMPS absent, a different version or path from the
one frozen, or a build without a required style (pair `eam/alloy`, the
minimiser, fix `nve`, `nvt`, `langevin`, `setforce`).

### 3. Native input

`native.py` writes, from the specification alone and byte for byte
reproducibly, `structure.data` (atom style `atomic`, box, masses, atoms with
image flags, velocities in A/ps), `in.lammps` and a copy of the potential
file whose SHA-256 is checked after copying. Periodic atoms outside the cell
are wrapped with image flags; LAMMPS returns unwrapped `xu yu zu`, so
coordinates are never shifted. The masses are set with `mass` commands after
`pair_coeff`, because `pair_style eam/alloy` would otherwise overwrite them
with the file's natural masses. Fixed atoms are frozen with `fix setforce`
and excluded from integration, then released before a final `run 0` so that
the reported forces are the physical ones. `lammps.native_input(spec)` and
`/api/lammps/spec` show the exact files without running anything.

### 4. Execution

Each run gets a private directory under `MATERIA_LAMMPS_RUN_DIR` or the
system temporary directory. LAMMPS runs in its own process group as
`lmp -in in.lammps -log log.lammps -echo log -nocite`, or through
`module_worker.py` with the same arguments for the Python module. Progress
comes from the thermodynamic rows it prints. Cancellation and the optional
time limit signal the whole process group and then kill it; application
shutdown stops every LAMMPS process Materia started. Only the last 400 lines
of standard output and error are held in memory, and the stored log excerpt
is the last 200 lines. The run record keeps the exact command, the version in
the log header, the pinned installation and packages, the SHA-256 and size of
every input file, the input script itself, the potential SHA-256, the wall
time and the log excerpt. The directory is deleted afterwards unless
`keep_files` is set.

### 5. Reading the output

Every check below must pass before any result exists (`run.py`): exit status
0; no `ERROR` line; the end-of-script marker reached; the log header naming
the frozen version; every thermodynamic row and dumped value finite; the
atom count constant in every row; the final dump and every trajectory frame
carrying exactly the written ids, types and masses; the printed steps and
frame timesteps exactly those requested; the final evaluation's potential
energy equal to the last energy of the run; and the kinetic energy recomputed
from the dumped velocities and masses equal to the one LAMMPS printed, which
checks the velocity and mass conversion on every run. A relaxation is
reported as converged only if the largest force on a mobile atom in the final
dump is within `ftol_eV_A`, whatever LAMMPS's stopping criterion says; both
are recorded.

Units are converted in one module (`units.py`): time and velocity by exact
factors of 1000, pressure to stress with `sigma = -P / nktv2p` using the
constant LAMMPS applies in metal units (`1.6021765e6` bar per eV/A^3, within
1e-7 of CODATA 2018), kinetic energy with LAMMPS's `mvv2e`.

### 6. Results, storage and ownership

Results are stored as `lammps::<run_id>::run`, `energy`, `forces`, and as
applicable `stress`, `thermo`, `relaxation`, `trajectory` and
`mean_temperature`; per-atom and trajectory data go to the checksummed array
store (`final_positions`, `final_velocities`, `forces`, `positions`,
`velocities`). The trajectory result and arrays follow the same contract as
EAM runs, so the existing player replays LAMMPS runs: `/api/eam/trajectory`
accepts a LAMMPS run identifier, and `/api/lammps/trajectory` is the same
payload.

A refused, failed or cancelled run stores nothing, changes no structure and
writes no history line. A finished run is stored in one step and, when
`apply` is set, its final state is written onto its structure as one undoable
change (forces; positions and forces for a converged relaxation; positions,
velocities and forces for dynamics), only if the structure still has the
geometry the run started from. `lammps.state()` reports `current`, `stale`
(with whether the run can still be applied) or `detached`, and
`lammps.apply()` refuses a stale geometry, an unconverged relaxation, a
detached structure and a second application.

### Driven

| Capability | LAMMPS commands | Status |
| --- | --- | --- |
| Energy and forces | `run 0`, `write_dump` | done (fake-process tests) |
| Stress, fully periodic cells | `thermo_style` `pxx` ... `pyz` | done (fake-process tests) |
| Fixed-cell relaxation | `minimize`, `min_style cg/fire/sd`, `min_modify norm max` | done (fake-process tests) |
| NVE dynamics | `fix nve`, `timestep`, `dump custom` | done (fake-process tests) |
| Langevin dynamics | `fix nve` + `fix langevin ... zero yes` | done (fake-process tests) |
| Nose-Hoover dynamics | `fix nvt` | done (fake-process tests) |
| Created velocities | `velocity create ... loop geom` | done (fake-process tests) |
| Fixed atoms | `group`, `fix setforce`, integration of the mobile group | done (fake-process tests) |
| Boundaries | `p` and `s`, orthogonal and restricted triclinic cells | done (fake-process tests) |

### Not driven

Pair styles other than `eam/alloy` (including `eam/fs`, Tersoff, ReaxFF,
MEAM and machine-learned potentials), barostats and variable-cell
relaxation, boundary styles `f` and `m`, atom styles other than `atomic`,
charges, molecular topology, MPI and accelerator packages, restart files,
replica and NEB workflows. Each is refused rather than approximated.

### Validation status

The adapter's own behaviour (discovery, specification, native files,
process control, parsing, unit conversion, validation, storage, ownership,
HTTP and Python paths) is tested against a deterministic fake LAMMPS in
`tests/support/fake_lammps.py`. Live parity against a real LAMMPS on the
shipped Cu potential is in `tests/validation/test_lammps_live.py`. On the
machine where this adapter was written no LAMMPS was installed, so that
validation is **blocked, not passed**: its three cases skip with the reason
`BLOCKED`. Install LAMMPS as above and run
`pytest tests/validation/test_lammps_live.py -rs` to perform it.
