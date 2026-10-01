# DFT relaxation

A relaxation is a ground-state experiment ([DFT_GROUND_STATE.md](DFT_GROUND_STATE.md))
whose nuclei, and for a variable cell whose cell, move until the forces and
the stress meet stated criteria. It reuses the ground-state specification,
its boundary and refusal rules, the GPAW worker, the parameter echo, the
result and array storage, and the current, stale and detached states. It adds
the ionic search and one thing a ground state never does: writing a new
geometry onto a structure, as one undoable change.

Code: `materia/experiments/dft/relaxation.py` (specification, checks,
execution), `materia/experiments/dft/records.py` (storage, state, apply),
`materia/solvers/gpaw_driver/ground_state_worker.py` (the ionic loop inside
the GPAW interpreter).

## The specification

`RelaxationSpec`, schema `materia.dft.relaxation`, version 1.0, is frozen and
identified by the SHA-256 of its canonical JSON. It holds a complete
`GroundStateSpec` (every electronic variable, the frozen geometry, the
software and PAW dataset identity) and:

| Variable | Unit | Meaning |
| --- | --- | --- |
| `mode` | | `fixed-cell` moves the nuclei; `variable-cell` also changes the cell |
| `optimizer` | | `BFGS` or `FIRE`, from ASE |
| `fmax_eV_A` | eV/A | largest force on any mobile atom at convergence |
| `max_steps` | steps | optimiser steps allowed |
| `maxstep_A` | A | largest move of any atom in one step |
| `symmetry` | | `preserve` (GPAW point group and time reversal) or `off` |
| `stress_tol_eV_A3` | eV/A^3 | variable cell: largest deviation of a free stress component from -p |
| `cell_mask` | | variable cell: which of xx, yy, zz, yz, xz, xy may change |
| `hydrostatic_strain` | | variable cell: scale the cell uniformly |
| `target_pressure_GPa` | GPa | variable cell: external hydrostatic pressure p |
| `fixed_atom_ids` | | from the structure's fixed flags, not settable |

The structure's magnetic moments are recorded too, so the fingerprint of the
relaxed geometry can be computed exactly. Defaults: fixed cell, BFGS,
0.05 eV/A, 100 steps, 0.2 A, symmetry preserved (off when atoms are fixed);
a variable cell adds 0.005 eV/A^3, every strain component free, no
hydrostatic constraint and zero pressure. When the observables are not given,
forces are requested, and the stress for a variable cell.

## What reaches GPAW, and how it is checked

Every ground-state variable reaches `gpaw.GPAW` exactly as for a ground state,
and `symmetry` is added as GPAW's own `symmetry` parameter. GPAW echoes its
parameters after the run and every value sent must come back unchanged. The
ionic variables reach the worker's relaxation block: optimiser class, step
length, force and stress criteria, step limit, fixed atom indices
(`ase.constraints.FixAtoms`) and, for a variable cell,
`ase.filters.FrechetCellFilter` with the mask, the hydrostatic flag and the
pressure converted to eV/A^3. The worker reports back what it actually built,
read from those objects, and any difference fails the run.

Convergence is judged by Materia's criteria, not by the optimiser: the
largest force on a mobile atom at most `fmax_eV_A` and, for a variable cell,
every free stress component within `stress_tol_eV_A3` of the target -p (the
trace only, for hydrostatic strain). The worker's verdict is checked against
its own last step.

## Refusals

Every ground-state refusal still applies. In addition a relaxation is refused,
with the variable named, when:

* the observables do not include `forces`, or `stress` for a variable cell;
* the SCF force tolerance is looser than `fmax_eV_A`;
* every atom is fixed at fixed cell;
* atoms are fixed and symmetry is preserved: GPAW detects symmetry from the
  atoms alone, so a symmetry mapping a fixed atom onto a mobile one is broken
  by the first step (found in a live H2 run, where GPAW stopped with "Broken
  symmetry");
* a charged system is in an external field: the net force qE leaves no
  equilibrium geometry;
* a variable cell is asked for anything but a bulk crystal in plane-wave mode,
  the only case in which GPAW computes the stress;
* a variable cell has fixed atoms, which the strain would carry with it;
* a variable cell is charged: the background term in the stress is unphysical;
* a variable-cell variable is set for a fixed cell;
* the mask frees nothing, or hydrostatic strain is combined with a partial mask;
* `fmax_eV_A` is outside (0, 1], `max_steps` outside 1 to 1000, `maxstep_A`
  outside (0, 0.5], `stress_tol_eV_A3` outside (0, 0.1], or the pressure outside
  -100 to 100 GPa.

Warnings, which do not block: no SCF force tolerance, a force criterion below
DFT noise, a plane-wave cutoff below 500 eV for a variable cell (Pulay stress),
a spin-polarised variable cell (each new cell restarts the SCF from the
initial moments), and preserved symmetry.

## Execution and transactions

The run is one GPAW worker process. A cancelled, timed-out or failed run, or
one whose output is inconsistent (parameter or optimiser echo mismatch, atoms
lost, a fixed atom moved, the cell changed at fixed cell, a non-finite energy,
missing forces or stress, a verdict that disagrees with its last step, ionic
steps out of order), yields no result, stores nothing and writes no history.

A finished run is either `converged` or `not-converged` (step limit reached).
Both are stored in one step under `dftrelax::<run_id>::`: the `relaxation`
record, `run`, and the ground-state observables at the final geometry
(`energy`, `forces`, `stress`, `charge_accounting`, `scf_history` of the last
SCF, and any requested grid or table), built by the same code as a
ground-state run for the final geometry. Arrays: `initial_positions`,
`final_positions`, `initial_cell`, `final_cell`, `final_forces`,
`step_positions` and `step_cells`, checksummed like every array. A
not-converged run is stored with origin `estimated` and is never applied.

The `relaxation` record holds the status, the stop reason, optimiser steps,
total SCF iterations, initial and final energy and largest force, the final
stress deviation, the displacement of every atom (largest, rms, per atom id),
the cell before and after (volumes, lengths, angles, small-strain tensor), the
fixed atoms, the step history (energy, force, stress deviation, volume, SCF
iterations per step), the echo, and the fingerprints of the input geometry,
the output geometry and the final ground state. Provenance: model
`external:gpaw/dft-relaxation`, the full relaxation specification and its
SHA-256 as the input digest, the GPAW parameters sent and used, the optimiser
sent and used, software and PAW dataset identity.

## Ownership, apply and undo

State follows the geometry fingerprint of the ground state (ids, elements,
positions, cell, periodicity, formal charges, magnetic moments):

* **current**: the structure has the relaxed geometry;
* **stale**: it still has the starting geometry (then the run can be applied
  if it converged), or it changed since the run started (then it cannot);
* **detached**: the run was for a structure the project does not hold.

A converged relaxation is applied automatically when it finishes if its
structure is unchanged; otherwise it is kept and one history line that is not
an undo point records why. Applying writes the final positions, the final
cell for a variable cell and the final forces as one undoable change. Undo
restores coordinates, cell, forces, roles and metadata, and the run becomes
stale and applicable again; redo makes it current. Applying refuses a stale
geometry, a detached structure, a not-converged run and a second application,
and rolls back completely if recording the change fails.

## Interface

The DFT panel has a **Geometry relaxation** group under the ground-state
controls: mode (variable cell offered only for a bulk pw calculation),
optimiser, force criterion, step limit, symmetry, step length, and for a
variable cell the stress criterion, target pressure, hydrostatic strain and
the six strain components. Refusals appear beside their control. The result
view shows status, state, energies, forces, stress deviation, displacement,
volume and cell change, plots of the free energy and of the largest force
(and stress deviation) against the optimiser step with the criterion drawn,
an **Apply relaxed geometry** button when the run can be applied, and the
provenance with the optimiser sent and used.

## Python and HTTP

```python
spec = dft.relax_spec(structure, fmax_eV_A=0.02, forces_tol_eV_A=0.01)
dft.relax_check(spec)["by_field"]
run = dft.relax(spec=spec)
run["relaxation"].value["history"], run["relaxation"].convergence.message
dft.relax_state(), dft.relax_apply(run_id), dft.relax_runs()
dft.relax_array(name="step_positions")
```

Routes: `dft/relax/spec`, `dft/relax/run`, `dft/relax/runs`,
`dft/relax/result`, `dft/relax/apply`. A refused specification raises
`ApiError`; a cancelled, timed-out or failed run raises `DFTRelaxationFailed`.

## Validation

`tests/validation/test_dft_relaxation_live.py` runs real GPAW and checks each
relaxed geometry with an independent ground-state run. See
[VALIDATION.md](VALIDATION.md#dft-relaxation). The implementation tests use a
deterministic worker double (`tests/support/fake_gpaw_relax.py`) and an
in-process run of the ionic loop with ASE's EMT calculator; they are software
guarantees, not physical validation.

## Limitations

Local minima only, from the given starting geometry. No constrained
relaxation beyond fixed atoms (no fixed bonds, planes or directions). No
variable-cell relaxation of slabs, wires or clusters, and none of charged
cells. The k-point divisions and the cutoff stay fixed while the cell changes.
With symmetry preserved the structure cannot lower its symmetry. Serial GPAW
only. No restart reuse during a relaxation. The undo stack, as for every
change, is not saved with the project; the history log and the stored run
are.
