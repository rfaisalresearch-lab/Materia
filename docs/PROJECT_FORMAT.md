# Project format

Version 1.0. A Materia project is a ZIP archive with the extension `.materia`.

## Layout

```
project.materia
├── manifest.json          schema version, metadata, wafer, regions, settings,
│                          selection, checkpoints, history
├── PROVENANCE.txt         plain-text reproducibility report
├── structures/<key>.json  full atomistic state
├── results/<key>.json     solver results with complete provenance
├── scans/<key>.json       scan metadata
├── scans/<key>.npz        raw channel arrays
├── arrays/<key>/          large arrays in checksummed chunks (grids, tables)
│   ├── manifest.json
│   └── chunk-00000.npy ...
├── claims/<id>.json       classified claims and the evidence they cite
└── scripts/<name>.py      scripts saved with the project
```

Everything is inspectable with standard tools:

```bash
unzip -l project.materia
unzip -p project.materia PROVENANCE.txt
unzip -p project.materia manifest.json | python -m json.tool
```

## What is stored, and what is regenerated

The wafer is stored as a **specification plus a seed**, not as atoms. An
atomistic region is stored explicitly, because you may have edited it, but its
`RegionSpec` and the wafer seed are stored too, so the pristine region can be
regenerated exactly.

This is what makes a project small. A 300 mm wafer with a 640-atom region is
about 12 kB.

## Autosave and crash recovery

Every state-changing desktop operation is atomically autosaved to
`~/.materia/recovery/autosave.materia`. The autosave uses the same documented
project format as an ordinary project and never replaces the user's named
project file. A later launch offers to restore, discard, or leave the recovery
file untouched. Explicitly saving, opening another project, or creating a new
project clears the superseded recovery file.

The recovery write uses a temporary ZIP followed by an atomic filesystem
replacement. A failed write leaves the previous valid autosave in place and is
reported in the warning panel.

## Structure record

```json
{
  "ids": [1, 2, 3],
  "numbers": [14, 14, 15],
  "positions": [[0,0,0], ...],
  "mass_numbers": [0, 29, 0],
  "formal_charges": [0.0, 0.0, 0.0],
  "partial_charges": [...],
  "magnetic_moments": [...],
  "velocities": [...],
  "forces": [[null, null, null], ...],
  "fixed": [false, ...],
  "roles": ["surface", "bulk", "dopant"],
  "labels": ["Si(a)", "Si(b)", "P@Si"],
  "cell": { "matrix": [[...]], "pbc": [true, true, false], "unit": "A" },
  "atom_meta": { "3": { "note": "..." } },
  "info": { "surface": {...}, "provenance": {...}, "defects": [...] },
  "next_id": 4,
  "bonds": [ { "a": 1, "b": 2, "order": 1.0, "length_A": 2.35,
               "origin": "distance-heuristic(scale=1.15)" } ],
  "units": { "positions": "A", "velocities": "A/fs", "forces": "eV/A",
             "charges": "e", "magnetic_moments": "mu_B" }
}
```

Atom ids are stable: they survive deletion, insertion and reordering, and a
selection saved in one session refers to the same atoms in the next.
`mass_numbers` of 0 means natural isotopic abundance. `forces` of `null` means
no solver has run.

### Embedded-atom runs

Each run stores `results/eam::<run_id>::energy` and `forces`, plus
`relaxation_history` or `trajectory` and `mean_temperature` for the other
tasks. The provenance parameters carry the potential identity (identifier,
file name, SHA-256, licence, citations), the setfl header, the cutoff
residuals and the task settings; `extra` carries the structure fingerprints
before and after the run and whether the result was applied. The parameter
files themselves are not copied into the project: the checksum identifies
them. No schema change was needed.

### LAMMPS runs

Only a finished LAMMPS run is stored; a refused, failed or cancelled one
leaves nothing. Each stores `results/lammps::<run_id>::run`, the audit
record: status, exact command, LAMMPS version from the log header, the
pinned installation and packages, the SHA-256 and size of `in.lammps`,
`structure.data` and the potential copy, the input script itself, wall time,
log warnings and the last 200 log lines. Its provenance parameters hold the
complete frozen specification (`run_spec`, schema `materia.lammps.run`
version 1.0), from which the data file can be regenerated and its recorded
hash checked. Alongside it are `energy`, `forces` and as requested `stress`,
`thermo`, `relaxation`, `trajectory` and `mean_temperature`, each with the
structure fingerprints before and after the run in `extra`. Arrays
`lammps::<run_id>::final_positions`, `final_velocities` and `forces` (kind
`per-atom`), and for dynamics `positions` and `velocities` (kind
`trajectory`, with the same metadata as EAM trajectories plus
`backend: lammps`), go to the array store below. No schema change was needed.

### DFT relaxations

Only a finished relaxation is stored; a refused, cancelled, timed-out or failed
one leaves nothing. Each stores `results/dftrelax::<run_id>::relaxation` (the
summary: status, stop reason, energies, forces, stress deviation, displacement
per atom id, cell before and after with the small-strain tensor, step history,
the optimiser echo and the geometry fingerprints) whose provenance holds the
complete relaxation specification (`relaxation_spec`, schema
`materia.dft.relaxation` version 1.0), and `run`, `energy`, `forces`,
`stress`, `charge_accounting`, `scf_history` and any requested observable at
the final geometry, exactly as a ground-state run stores them. Arrays
`dftrelax::<run_id>::initial_positions`, `final_positions`, `initial_cell`,
`final_cell`, `final_forces` (kinds `per-atom` and `table`) and
`step_positions`, `step_cells` (kind `trajectory`) go to the array store. An
apply is an ordinary undoable structure change with operation
`solver.dft.relax.<mode>`. No schema change was needed.

### Electrostatics

A point-charge model is stored in the structure record under
`info.point_charge_model`: `kind` (`per-element`, `per-atom`,
`formal-point-ion`), `by_element`, `by_atom_id`, `source` and `unit` (`e`).
Each electrostatics run stores `results/electrostatics::<run_id>::energy`,
`forces`, `site_potential`, `site_field` and `point_charges`, each with a full
provenance record including the settings and an input fingerprint. Older
Materia versions read these files unchanged: the model is part of the
free-form `info` block and the results are ordinary result records, so no
schema change or migration was needed.

### Ground-state DFT runs

Each run stores `results/dft::<run_id>::run` (the record: status, the complete
experiment specification and its SHA-256, the exact GPAW input and the
parameters GPAW reported, software and dataset identity, restart use,
warnings, SCF history for a run that did not converge) and, for a converged
run, `energy`, `forces`, `stress`, `fermi_level`, `magnetic_moment`,
`eigenvalues`, `occupations`, `density`, `spin_density`,
`electrostatic_potential`, `charge_accounting` and `scf_history` as requested.
Grids and the eigenvalue and occupation tables are not inside those JSON
files: the result's value names an entry in the array store below. A
convergence study stores `results/dftstudy::<study_id>`, pointing at every run
it made. See [DFT_GROUND_STATE.md](DFT_GROUND_STATE.md).

### DFT density of states

A completed DOS stores `results/dftdos::<run_id>::dos` plus the ground-state
energy, Fermi level, charge accounting, SCF history and run record used to
produce it. Its full `materia.dft.dos` version 1.0 specification records the
source structure or run, spectral grid, broadening, spin channels, non-self-
consistent k-points, band count and projections. The arrays `energies`,
`dos_total`, optional `dos_spin`, optional `pdos`, optional `pdos_spin`,
`eigenvalues` and `kpoint_weights` use the checksummed array store. Refused,
cancelled, timed-out, non-converged and invalid calculations store no numeric
record. See [DFT_DOS.md](DFT_DOS.md).

### DFT band structure

A completed band structure stores `results/dftbands::<run_id>::bands` plus the
ground-state energy, Fermi level, charge accounting, SCF history and run record
used to produce it. Its full `materia.dft.band-structure` version 1.0
specification records the source, the ground state, the ground-state symmetry
choice and the exact path: special points, their origin (generator, ASE
version, lattice and the cell they were generated for), branches, segment
intervals and every k-point. The arrays `eigenvalues` `[spin, k-point, band]`
(eV), `kpoints_frac`, `kpoints_cartesian` (1/A) and `distance` (1/A) use the
checksummed array store, and the `bands` record keeps the SHA-256 of each; a
record whose arrays no longer match is reported as corrupt and not used.
Refused, cancelled, timed-out, non-converged and unverifiable calculations
store nothing. See [DFT_BAND_STRUCTURE.md](DFT_BAND_STRUCTURE.md).

### DFT local density of states

A completed LDOS stores `results/dftldos::<run_id>::ldos` with the window, state
count, map integral, PAW sphere radii and the SHA-256 of every array, plus the
ground-state results used. Its `materia.dft.ldos` version 1.0 specification
records the source, ground state, window, spin channels and bands. The arrays
`ldos` (volumetric, states/A^3), optional `ldos_spin`, `eigenvalues` and
`kpoint_weights` use the checksummed array store. STM images are derived from
the stored map on request and not stored. See [DFT_LDOS.md](DFT_LDOS.md).

### DFT equation of state

A completed equation of state stores `results/dfteos::<run_id>::eos`: the fitted V0, B0, B', E0, every point with its run
identity and specification digest, the checks and the SHA-256 of every array.
Its `materia.dft.equation-of-state` version 2.0 specification records the
reference crystal and settings and every sampled volume. The fitted energy is
the zero-width extrapolated energy; version 1.0, which fitted the free energy,
is not read. The arrays `scales`, `volumes`, `energies`, `free_energies`,
`fit_pressures`, `free_energy_fit_pressures` and, where available,
`stress_pressures` and `max_forces` use the checksummed array store. Nothing is
stored for a refused, cancelled, failed or unverified run. Applying the
equilibrium volume is an ordinary undoable structure change. See
[DFT_EQUATION_OF_STATE.md](DFT_EQUATION_OF_STATE.md).

### Large arrays

`arrays/<key>/manifest.json` holds the schema (`materia.array`, version 1),
dtype, shape, the chunk list with each chunk's row range and SHA-256, the
SHA-256 of the whole array, the unit, a description, the kind (`volumetric`,
`table` or `trajectory`) and grid metadata where relevant: shape, cell,
periodicity, origin and the
convention that grid point (i, j, k) sits at fractional (i/N1, j/N2, k/N3).
`chunk-NNNNN.npy` files are consecutive slabs along the first axis in NumPy's
documented `.npy` format, at most 4 MB each, at full precision. Reading
verifies every checksum and refuses an array with a damaged, missing or
misplaced chunk. The manifest lists array keys under `arrays`.

Molecular-dynamics positions and velocities use the same store with kind
`trajectory`. Their metadata fixes atom ids, elements, roles, cell, frame
steps and physical times, so a saved run remains inspectable after reopening.

### Claims

`claims/<id>.json` holds a statement, its classification (observation,
computational-prediction, candidate-relation, empirical-invariant, conjecture,
independently-reproduced or experimentally-supported), the evidence it cites
and the conditions it was made under. The evidence rules are checked again
when the file is read. The manifest lists claim ids under `claims`.

These additions follow the stability promise below: new fields and files in
schema 1.0, ignored by readers that do not know them.

## History

Every state-changing operation appends a `LogEntry` with its operation name,
a human-readable label, the parameters that produced it, a timestamp, and
whether it has been undone. Lines that record something without an undo point,
such as a calculation that observed a structure, carry `undoable: false`, and
undo and redo pass over them, so the line marked as undone is always the one
whose snapshot was restored. The log is permanent and is saved with the
project; the undo *stack* is in memory only.

### What an undo has to put back

Reversing an edit means restoring more than one structure's coordinates.
Building a surface adds a dictionary entry and moves the active key; extracting
a region adds a region, a structure and an active region; creating a wafer
replaces the wafer. A snapshot therefore carries two parts:

| part | holds | cost |
| --- | --- | --- |
| `state` | the project topology: structure keys, active key, selection, wafer, regions, active region, key counter | references, a few pointers |
| `structure` + `structure_key` | a deep copy of the one structure whose *contents* the operation was about to change | a few megabytes, or nothing for a topology-only operation |

The topology holds **references** to the live structure and region objects
rather than copies. That is sound because of one invariant the project keeps:
*every in-place mutation of a structure records its own snapshot containing a
copy of that structure's prior contents.* Undo unwinds in LIFO order, so by the
time an older topology is restored, every content change made after it has
already been undone by its own entry. A topology-only operation (adding a
structure, creating a wafer) therefore copies no atoms at all.

Contents are restored **into the object the slot already holds**, not by
swapping in a new one, so a key's structure object identity never changes for
the life of the project and any handle pointing at it stays valid across undo.
No two keys ever share one mutable structure object.

Snapshots are capped by depth and by a memory budget, and the budget counts
only the deep copies. Larger systems would need a delta encoding; that is
recorded in [LIMITATIONS.md](LIMITATIONS.md) rather than silently degrading.

An operation that is refused or fails records nothing: the snapshot is taken
only once the operation is going to happen, so a refusal leaves no phantom
entry and no extra undo level.

## Checkpoints

A named, restorable state. Unlike undo, checkpoints are saved with the project
and survive reopening. This is what "save a checkpoint before you deform it,
then use it as the strain reference" means.

## Migration

`manifest.json` carries `schema_version`. On load, registered migrations run in
order until the manifest matches the current version. A project written by a
**newer** Materia is refused with a clear message and nothing is modified,
rather than being partially misread.

```python
@register("1.0", "1.1")
def add_field(manifest: dict) -> dict:
    manifest.setdefault("new_field", default)
    return manifest
```

## Stability promise

Within a major schema version: fields are added, never removed or repurposed,
and readers ignore fields they do not know. Across a major version: a migration
is provided, or the file is refused with an explanation.
