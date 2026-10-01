# GPAW: first-principles calculations

GPAW is Materia's driven Tier-3 solver: projector augmented-wave density
functional theory, optional, run in its own interpreter, and never reported as
working until Materia has checked that it actually can. It is reached two
ways.

* **The ground-state experiment** ([DFT_GROUND_STATE.md](DFT_GROUND_STATE.md))
  is the full interface: an immutable, versioned specification of every
  variable, clusters, wires, slabs and bulk crystals, charged and
  spin-polarised systems, an optional static field along open directions,
  energies, forces, stress, densities, spin density, electrostatic potential,
  eigenvalues, occupations, Fermi level, magnetic moment, electron accounting
  and SCF history, with staleness, persistence, restart reuse and a
  convergence laboratory. The Solvers panel, `lab.dft.ground_state` and the
  `/api/dft/*` routes use it.
* **The DOS experiment** ([DFT_DOS.md](DFT_DOS.md)) uses a structure, stored
  ground state or stored relaxation to compute total DOS and PAW-projector
  PDOS with explicit k-points, bands, broadening, spin channels, checksummed
  arrays and an independent reconstruction from the returned eigenvalues.
* **The LDOS experiment** ([DFT_LDOS.md](DFT_LDOS.md)) sums |psi|^2 of the
  Kohn-Sham states in an energy window on the real-space grid, with symmetry
  off, and takes Tersoff-Hamann constant-height and constant-current images
  from it.
* **The band-structure experiment** ([DFT_BAND_STRUCTURE.md](DFT_BAND_STRUCTURE.md))
  computes Kohn-Sham bands on an explicit, stored reciprocal-space path with
  symmetry off, and verifies the k-points, labels, bands and path distance
  GPAW reports before storing them.
* **The single-point adapter** `external:gpaw`, described in the rest of this
  document, is the earlier and narrower driver behind `lab.dft.energy`, the
  `gpaw/*` routes and the solver registry. It is kept unchanged for existing
  scripts and projects.

This document covers what both share: installation, why GPAW runs out of
process, shutdown, and the single-point adapter's own settings.

## What the single-point adapter does, and what it does not

| | |
| --- | --- |
| **Drives** | single-point total energy, forces |
| **Does not drive** | relaxation, dynamics, charge density, density of states, LDOS, band structure, STM, dipole, stress, spin-polarised and charged systems |

`external:gpaw` declares exactly `energy` and `forces`, and a request for
anything else is refused with the reason rather than answered. The experiment
system computes the densities, stress, charged and spin-polarised systems,
relaxation, DOS and PDOS that this adapter refuses.

**Charged and spin-polarised systems are refused by this adapter, not
attempted.** The configuration still has `charge` and `spin_polarized` fields,
and their only purpose is to produce a precise refusal naming the missing
capability. A non-zero charge, `spin_polarized=True`, a structure carrying a
net formal charge, and a structure carrying non-zero initial magnetic moments
are all declined before any GPAW process starts. `spin_polarized` is not
coerced: `bool("false")` is `True`, and a string must not be what decides
whether a calculation is spin-polarised, so a non-boolean is refused by type.

Electron bookkeeping is stated once and tested: `Structure.total_electrons()`
is **sum Z minus sum formal charge**, counted from nuclear charge, not from a
neutral reference. An earlier version of the validator subtracted the
requested charge from that figure a second time, so a one-electron H2+ looked
like a closed shell and was accepted as a spin-paired calculation. The double
subtraction is gone, and charged systems are refused before any parity
argument is needed.

**Functionals without datasets.** This adapter lists PBEsol and BLYP, but the
standard `gpaw-data` distribution has no datasets for them, so GPAW fails when
asked; the failure is reported, with no energy. The ground-state experiment
checks dataset availability before running and refuses such a functional with
the missing file named.

## Installation

GPAW is **not** a Materia dependency. Two things have to be present and they
fail independently:

* the **code**: the `gpaw` package;
* the **PAW datasets**: distributed separately under their own licence.

`import gpaw` succeeding tells you nothing about the second, so Materia probes
both and reports them separately. An installation with the code and no datasets
is reported as *not operational*, with the instruction that fixes it.

```bash
conda create -n materia-gpaw -c conda-forge python=3.13 gpaw gpaw-data
export MATERIA_GPAW_PYTHON=$(conda run -n materia-gpaw which python)
```

On Apple Silicon there is no `osx-arm64` build on conda-forge. Prefix the
create with `CONDA_SUBDIR=osx-64` to use the x86_64 build under Rosetta, or
build GPAW from source against libxc. The machine this was developed on runs
the x86_64 build; that is why Materia drives GPAW out of process rather than
importing it, since the two interpreters cannot even share an architecture.

Materia looks for an interpreter in this order: `MATERIA_GPAW_PYTHON`, the
python beside a `gpaw` script on `PATH`, the interpreter running Materia, then
conda environments whose name contains `gpaw`. An interpreter that has the code
but not the datasets is preferred as a diagnosis over one that has neither, so
the reason you are shown is the most advanced problem found.

## Why it runs out of process

Three reasons, all of which turned out to matter:

1. A working GPAW often lives in a different environment from Materia.
2. A calculation has to be cancellable without taking the application with it.
   The subprocess runs in its own session, so cancelling signals the whole
   process group; measured latency from request to exit is **0.1 s**.
3. A crash inside a linked Fortran or MPI library would otherwise be
   unrecoverable.

Progress arrives as one line per SCF iteration on the subprocess's standard
output, which is what drives the progress bar and what lets a cancelled run
still report how far it got.

## Configuration

Every setting carries its unit in its name and is validated before a job is
created.

| setting | unit | notes |
| --- | --- | --- |
| `task` | | `energy` only |
| `xc` | | LDA, PBE, RPBE, revPBE, PBEsol, BLYP |
| `mode` | | `pw`, `fd`, `lcao` |
| `cutoff_eV` | eV | plane-wave mode; refused below 100 eV |
| `grid_spacing_A` | Å | finite-difference mode; refused above 0.35 Å |
| `basis` | | LCAO mode |
| `kpoints` | divisions | Monkhorst-Pack |
| `occupations` | | `fermi-dirac`, `marzari-vanderbilt`, `fixed` |
| `smearing_eV` | eV | |
| `charge` | e | must be exactly 0; anything else is refused |
| `spin_polarized` | | must be `False`; `True` is refused, and a non-boolean is refused by type |
| `energy_tol_eV_per_electron` | eV/electron | |
| `density_tol_electrons` | electrons | |
| `max_iterations` | SCF iterations | |

Presets: `smoke` (smallest thing that still solves the Kohn-Sham equations),
`molecule` (open boundaries, real-space grid) and `surface` (in-plane k-points,
metallic smearing).

### Combinations that are refused, not corrected

**Plane waves on a cell that is not periodic in all three directions.** GPAW's
PW mode expands in plane waves over the cell regardless of what the structure
says, so it silently imposes periodic boundaries. Measured on H₂ in a 6 Å box:

| | total energy |
| --- | ---: |
| PW(300), `pbc=False` | −6.528966 eV |
| PW(300), `pbc=True` | −6.528966 eV |
| FD grid 0.18 Å, `pbc=False` | −6.656841 eV |

The first two are identical to every digit: the boundary condition the user
asked for was discarded. The third is the calculation they meant, and it
differs by 0.128 eV. Materia refuses the first case and says why.

Also refused: k-point divisions along a non-periodic direction; any non-zero
charge or spin polarisation, whether asked for in the configuration or carried
by the structure; a grid or cutoff too coarse to mean anything; any unknown
setting.

## Convergence, cancellation and failure

Four outcomes are kept distinct, because they mean different things:

| status | what it is | what is reported |
| --- | --- | --- |
| `converged` | a solution | energy and forces, origin `calculated` |
| `not-converged` | ran out of iterations | **no energy**, origin `unsupported`, with the iteration count and the tolerance it missed |
| `cancelled` | the user stopped it | **no energy**, with how far it got |
| `failed` | it raised | **no energy**, with GPAW's own error |

A half-solved Kohn-Sham problem is not a total energy, so an unconverged run is
never returned as a calculated result.

## Ownership, staleness and shutdown

A first-principles run takes minutes, and the project can change underneath it.
Three rules keep the answer attached to the question.

**A job belongs to the project it was submitted from.** When the calculation is
queued, Materia freezes a `GPAWSubmission`: the owning project, the structure
slot, a copy of the whole worker specification, the validated configuration,
and a fingerprint of both. The run writes its results only to that project.

**Replacing the project cancels its jobs.** Starting a new project, opening one
from disk, or accepting a crash recovery cancels every unfinished job belonging
to the outgoing project, with the reason recorded on the job and a warning in
the interface. A cancelled run writes nothing into the incoming project; the
outgoing project keeps the record that its own run was stopped. The alternative
(letting the result land in a project the user has closed) was rejected
because the answer would arrive somewhere nobody is looking.

**Forces are attached only to the geometry they were computed for.** Before
applying them, Materia recomputes the fingerprint of the structure now in the
slot and compares it with the submitted one. Atom ids alone are not enough, so
the digest covers atomic numbers, positions, cell vectors, periodicity, initial
magnetic moments, fixed flags, atom ids and ordering, total formal charge, and
every validated setting. If anything changed, the energy and its provenance are
kept and the forces are refused with a message naming what changed. A stale
result stays readable; it just never gets attached to a structure it does not
describe.

### The input digest

`input_digest(spec, settings)` is a SHA-256 over a canonical JSON encoding of
the worker structure specification and the validated configuration, truncated
to 16 hex characters. It is deterministic, it is recomputed rather than stored
on the structure, and it lands in `Provenance.inputs_digest` on every result of
the run.

### Run identity

Results are keyed `gpaw::<run_id>::energy` and `gpaw::<run_id>::forces`, where
`run_id` is a millisecond timestamp and a short random suffix. Nothing is
overwritten: repeated calculations accumulate, each with its own digest and
provenance, and sorting the run ids gives them in order. `service.gpaw_runs()`
lists them newest first.

### Shutdown

GPAW runs in its own session, which is what makes cancellation clean and what
would otherwise let it outlive the application. `JobQueue.shutdown()` is
idempotent: it stops accepting work, cancels every queued and running job,
signals every GPAW process group, waits a bounded grace period, force-kills
what is left, and removes the temporary run directories unless diagnostics were
asked for. A job caught by shutdown finishes as `cancelled` with that reason,
never as `done`. It is called from the native window's close handler, from the
headless server's exit path, and from an `atexit` hook as a last resort.

A `SIGKILL` of Materia itself, or `os._exit`, cannot be intercepted by anything
and will still leave a worker behind until it next writes to its closed pipe.

## What is recorded

Every result carries: the input digest, the run id, GPAW version, ASE version,
the interpreter and its Python version, MPI world size, functional, mode,
cutoff or grid spacing or basis, k-points, occupations and smearing, charge,
spin, both convergence tolerances,
the iteration cap, boundary conditions, SCF iterations used, whether it
converged, wall time, and the **identity of every PAW dataset**: element,
filename, MD5 fingerprint, type, and valence and core electron counts. GPAW's
own warnings are kept, deduplicated with a count.

The fidelity is `tier3-external-first-principles` and the notes say it is *a
first-principles approximation, not an exact solution*.

Results and their provenance are saved in the `.materia` file and are read back
when the project is reopened.

## Using it

### Interface

The Solvers panel section **First principles: ground-state DFT (GPAW)** runs
the ground-state experiment, not this adapter; see
[DFT_GROUND_STATE.md](DFT_GROUND_STATE.md#interface). It shows the code and the
datasets as separate rows, the interpreter, how it was found and the parallel
mode, and when GPAW is missing it shows the blocking reason and the
instruction that fixes it. The single-point adapter is reached from Python and
the `gpaw/*` routes.

### Python

```python
lab.dft.status()
lab.dft.available()
lab.dft.presets()

settings = lab.dft.settings("surface", surface, xc="PBE", kpoints=(4, 4, 1))
energy = lab.dft.energy(surface, settings=settings)
energy.value, energy.provenance.parameters["paw_datasets"]
```

`apply_forces=True` writes the converged forces onto the structure as one
undoable change. Nothing is written when the run does not converge.

### Service and HTTP

`gpaw/status`, `gpaw/settings`, `gpaw/energy`: the same implementation the
Python API and the interface use.

## Verification on this machine

Every number below was produced by a real calculation on the development
machine (macOS 27, Apple Silicon, GPAW 25.7.0 x86_64 under Rosetta, serial,
ASE 3.29.0, PAW datasets 558 files / 83 elements).

| check | result |
| --- | --- |
| H₂, PW(300), LDA, 6 Å box | −6.528966 eV, 15 SCF steps, 3.2 s |
| H₂, FD 0.18 Å, LDA, open boundaries | −6.656841 eV, 14 SCF steps, 2.9 s |
| Si(111) slab, 4 atoms, FD 0.25 Å, LDA, Γ | −6.207788 eV, 29 SCF steps, 4.4 s, max \|F\| 3.16 eV/Å |
| Cancellation latency | 0.10 s from request to process exit |
| Non-convergence at 3 iterations | refused, no energy reported |

### Physical validation

Separate from the software tests, in `tests/validation/test_gpaw_physics.py`.

**Forces are the gradient of the energy.** The Hellmann-Feynman relationship is
an identity, and the two quantities travel through different parts of the
driver. On H₂ at 0.74 Å with PBE and a 0.16 Å grid: analytic force
**0.387225 eV/Å** against a central-difference derivative of the driver's own
energies of **0.391169 eV/Å**, agreeing to **0.0039 eV/Å**, the finite-difference
truncation error.

**H₂ bond length against experiment.** Five SCF solutions from 0.72 to 0.80 Å,
parabola fit:

| | |
| --- | ---: |
| PBE, this driver | 0.7521 Å |
| Experiment (Huber & Herzberg 1979) | 0.7414 Å |
| Deviation | **+0.0107 Å (+1.44 %)** |

PBE is known to overestimate this bond, and the case bounds the overestimate
and reports it rather than asserting agreement.

**These numbers are for H₂ and a four-atom silicon slab.** Nothing here
generalises to an arbitrary material, cell size or functional, and none of
these settings is converged with respect to grid, cutoff, k-points or vacuum
unless you check that yourself.

## Related

* `docs/DFT_GROUND_STATE.md`: the ground-state experiment and convergence laboratory
* `docs/FIDELITY_TIERS.md`: what Tier 3 means
* `docs/LIMITATIONS.md`: what Materia still cannot do
* `docs/PROVENANCE.md`: how results are traced
