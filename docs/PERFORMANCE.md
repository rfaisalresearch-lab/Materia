# Performance

Every number here was produced by running `materia bench` on the machine below.
None is estimated, extrapolated or aspirational. Timings vary with hardware,
BLAS build and thread count; re-run the command to get numbers for yours.

## Machine

| | |
| --- | --- |
| Materia | 0.1.0 |
| Python | 3.13.13 |
| Platform | Darwin 27.0.0 arm64 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| BLAS | accelerate unknown |

## Core scenes

Si(111) slabs of the stated size, built with six stacking repeats.

| Scene | Atoms | Time | Note |
| --- | ---: | ---: | --- |
| neighbour list, 3.0 A cutoff | 972 | 0.003 s | 3726 directed pairs |
| bond perception | 972 | 0.005 s | 1863 bonds |
| Stillinger-Weber energy and forces | 972 | 0.006 s | E/atom = -4.155908 eV |
| FIRE relaxation, 20 steps | 972 | 0.013 s | max |F| = 0.0003 eV/A |
| neighbour list, 3.0 A cutoff | 10,092 | 0.040 s | 38686 directed pairs |
| bond perception | 10,092 | 0.061 s | 19343 bonds |
| Stillinger-Weber energy and forces | 10,092 | 0.064 s | E/atom = -4.155908 eV |
| FIRE relaxation, 20 steps | 10,092 | 0.138 s | max |F| = 0.0003 eV/A |
| neighbour list, 3.0 A cutoff | 99,372 | 0.437 s | 380926 directed pairs |
| bond perception | 99,372 | 0.677 s | 190463 bonds |
| Stillinger-Weber energy and forces | 99,372 | 0.682 s | E/atom = -4.155908 eV |
| STM constant current 128 x 128 | 128 | 1.251 s | corrugation 0.238 A |
| STM constant current 256 x 256 (cached electronic structure) | 128 | 4.081 s | corrugation 0.241 A |
| AFM frequency shift 160 x 160 | 128 | 2.077 s | range 2.348 Hz |

## Embedded-atom metals

Measured with `eam_benchmarks` in `materia/benchmarks.py` on the machine above:
fcc Cu with the shipped Zhou 2004 potential (cutoff 5.716 A). The first call
includes building the neighbour list; the repeat reuses the Verlet list built
with a 0.5 A skin; the dynamics rows are ten velocity-Verlet steps with a
Langevin thermostat.

| Scene | Atoms | Time | Note |
| --- | ---: | ---: | --- |
| EAM energy and forces, first call | 864 | 0.027 s | 46656 directed pairs, E/atom -3.539983 eV |
| EAM energy and forces, Verlet list reused | 864 | 0.008 s | 1 list build(s) |
| EAM molecular dynamics, 10 steps | 864 | 0.110 s | 11.0 ms per step |
| EAM energy and forces, first call | 10,976 | 0.264 s | 592704 directed pairs, E/atom -3.539983 eV |
| EAM energy and forces, Verlet list reused | 10,976 | 0.107 s | 1 list build(s) |
| EAM molecular dynamics, 10 steps | 10,976 | 1.478 s | 147.8 ms per step |
| EAM energy and forces, first call | 97,556 | 2.766 s | 5268024 directed pairs, E/atom -3.539983 eV |
| EAM energy and forces, Verlet list reused | 97,556 | 1.151 s | 1 list build(s) |
| EAM molecular dynamics, 10 steps | 97,556 | 16.195 s | 1619.5 ms per step |

Cost is linear in the number of atoms. Relaxation and dynamics of a few
thousand atoms run at interactive speed; a hundred thousand atoms cost about
1.6 s per dynamics step, which is a background job.

## Point-charge electrostatics

Measured with `electrostatics_benchmarks` in `materia/benchmarks.py` on the
machine above, at the default accuracy of 10^-8. Rock-salt supercells with unit
charges; bulk rows report the Madelung constant recovered from the computed
energy (reference 1.747564594633), so each timing row is also an accuracy
check. Slab rows are the same crystal made non-periodic along z, which the
method evaluates in an internally enlarged cell. Rows with a convergence check
include the second, independent Ewald split that decides whether a result is
labelled converged. Slabs and checked runs were measured up to about 10,000
ions; the 97,336-ion bulk run was measured once, without the check.

| Scene | Atoms | Time | Note |
| --- | ---: | ---: | --- |
| Ewald bulk | 1,000 | 0.114 s | 4060 k vectors, r_c 13.3 A, Madelung 1.747564596 |
| Ewald bulk, with convergence check | 1,000 | 0.211 s | 4060 k vectors, r_c 13.3 A, Madelung 1.747564596 |
| Ewald slab | 1,000 | 0.181 s | 3994 k vectors, r_c 20.8 A |
| Ewald slab, with convergence check | 1,000 | 0.433 s | 3994 k vectors, r_c 20.8 A |
| Ewald bulk | 10,648 | 3.166 s | 13072 k vectors, r_c 19.8 A, Madelung 1.747564612 |
| Ewald bulk, with convergence check | 10,648 | 6.703 s | 13072 k vectors, r_c 19.8 A, Madelung 1.747564612 |
| Ewald slab | 10,648 | 6.353 s | 14479 k vectors, r_c 30.0 A |
| Ewald slab, with convergence check | 10,648 | 15.707 s | 14479 k vectors, r_c 30.0 A |
| Ewald bulk | 97,336 | 95.159 s | 39558 k vectors, r_c 28.6 A, Madelung 1.747564597 |

Cost grows as about N^1.5, as expected for Ewald summation with a balanced
split. An interactive job up to a few thousand ions finishes in seconds; ten
thousand ions take several seconds and run as a cancellable background job
with progress; a hundred thousand ions take minutes. Particle-mesh Ewald would
be the next step if structures that large become routine.

## Renderer payload

The renderer thins above an atom budget and says so in the viewport. These are
the times to build the payload the viewport consumes, not the frame time.

| Scene | Atoms | Time | Note |
| --- | ---: | ---: | --- |
| renderer payload | 972 | 0.008 s | 972 instances, stride 1 |
| renderer payload | 10,092 | 0.069 s | 10092 instances, stride 1 |
| renderer payload | 99,372 | 0.050 s | 99372 instances, stride 1 |
| renderer payload (level of detail) | 1,002,252 | 0.136 s | 200451 instances, stride 5 |

## What the numbers mean

* **Neighbour lists scale close to linearly.** A hundred thousand atoms with a
  3 Å cutoff takes well under a second, because the list is built with a k-d
  tree over explicitly expanded periodic images rather than an O(N²) scan.
* **The classical potential scales linearly** in the same regime. The
  three-body term is batched by coordination number so the triplet enumeration
  is vectorised rather than looped per atom.
* **Scanning-probe simulation is dominated by the eigensolve**, not by the
  image. That is why the electronic structure is computed on a truncated slab,
  extracted only inside the bias window, and cached on a content digest of the
  structure and the settings. Re-imaging the same surface at a different
  resolution, palette or noise seed skips the eigensolve entirely; any edit to
  the structure invalidates the cache.
* **A first scan after an edit pays full price.** This is the honest cost of
  doing the electronic structure properly rather than texturing a lattice.

## Interactive budget

| Action | Typical |
| --- | --- |
| Extract a 3 nm region (~640 atoms) | under a second |
| Rotate, pan, zoom the atomic model | 60 frames per second to ~10⁵ instances |
| Energy or relaxation step on a region | milliseconds |
| First STM scan at 128 × 128 | a couple of seconds |
| Repeat STM scan, cached electronic structure | under a second for the solve, the rest is the image |

## Limits observed

* The neighbour list refuses to build a pair list beyond `max_pairs` and says
  how to reduce it, rather than exhausting memory.
* The renderer applies level of detail above its atom budget and reports the
  stride in the viewport.
* A million-atom region builds and renders with level of detail, but undo
  snapshots at that size are not practical; see
  [LIMITATIONS.md](LIMITATIONS.md).
