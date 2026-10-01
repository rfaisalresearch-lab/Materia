# Capability and evidence audit

Snapshot: 2026-09-30.

This report separates breadth, correctness, scientific validation and speed.
A large test count does not make a missing feature complete, and a fast
calculation does not establish physical accuracy.

## Completion

Materia does not currently have a defensible single completion percentage.
The earlier 100% first-release claim and weighted 70.6% to 72.1% full-product
claims are retracted. They used an uneven internal checklist whose rows were
not comparable in scope. Multiple narrow infrastructure pieces received
separate credit while large missing scientific domains received one row.

The 20 first-release items have service-level integration coverage, but that
does not establish that all twenty are complete as end-user workflows in the
real interface. Each acceptance scenario must be repeated manually from a
fresh project, with its visible result and exported artifact inspected, before
that vertical slice can receive a completion percentage.

The row counts below remain useful only as an inventory:

| Status | Rows |
| --- | ---: |
| Implemented within the stated narrow scope | 45 |
| Partial | 8 |
| Missing | 10 |
| Intentionally refused | 4 |
| Blocked by an unavailable dependency or dataset | 1 |

This inventory must not be presented as 72.1% of the original vision, as a
replacement percentage for established scientific packages, or as evidence
that the product is nearly finished.

### Current expanded-goal estimate

Against the current goal of a broad atomic, molecular, electronic, microscopy,
multiscale and collision platform that can substantially replace mature tools,
the engineering estimate is **19% complete**. Interface polish, documentation,
example count and test count receive no separate completion credit. They only
support the scientific capabilities below.

| Capability domain | Weight | Completion within domain | Earned |
| --- | ---: | ---: | ---: |
| Structures, materials, defects and surfaces | 10% | 45% | 4.50% |
| Classical atomistic physics, force fields and dynamics | 14% | 28% | 3.92% |
| Quantum and electronic-structure calculations | 18% | 20% | 3.60% |
| External solvers, high-performance execution and parity | 12% | 5% | 0.60% |
| Molecular, biomolecular and reactive chemistry | 10% | 0% | 0.00% |
| STM, AFM and spectroscopy | 10% | 35% | 3.50% |
| Wafer, device and multiscale physical modeling | 8% | 15% | 1.20% |
| Advanced analysis, discovery and inverse workflows | 6% | 12% | 0.72% |
| Subatomic, nuclear and detector physics | 5% | 0% | 0.00% |
| Independent validation and researcher-ready release quality | 7% | 12% | 0.84% |
| **Total** | **100%** |  | **18.88%, reported as 19%** |

This is a scope-weighted engineering estimate, not an experimental measurement.
It can change only when a scientific capability works end to end and passes an
appropriate validation gate. Adding interface controls, mock backends, refusal
messages or more examples does not increase it.

The complete row-by-row list, its status and code or test evidence is in
[REQUIREMENTS_MATRIX.md](REQUIREMENTS_MATRIX.md). The main remaining gaps are
amorphous structures, alloys and partial occupancy, explicit extended defects,
orbital character of bands, advanced tip models,
general force and occupation editing, context menus, equation discovery,
million-instance full rendering and complete
accessibility.

## What one Materia window consolidates

Materia currently combines 13 application roles. This is a workflow count,
not a claim that it replaces 13 mature specialist packages in their full
scope.

| Role | Working capability in Materia |
| --- | --- |
| 1. Wafer and multiscale modeler | procedural wafer, microstructure and linked atomistic region extraction |
| 2. Crystal and surface builder | seven crystal families, orientations, terminations and Si(100)-2x1 reconstruction |
| 3. Atomistic viewer and editor | selection, atom inspection, defects, dopants, adsorbates, movement, strain and measurement |
| 4. Classical relaxation and MD workbench | Stillinger-Weber, Lennard-Jones, EAM, FIRE, velocity Verlet and Langevin |
| 5. Collision and trajectory viewer | collision preparation, NVE trajectories, pause, scrub, velocity and fragment inspection |
| 6. Electrostatics workbench | direct, Ewald and slab-corrected Ewald energy, force, potential and field |
| 7. Semi-empirical electronic workbench | tight-binding bands, DOS, LDOS and simplified STM input |
| 8. First-principles workbench | GPAW ground state, fixed and variable-cell relaxation, DOS and PDOS, band structure |
| 9. External solver job front end | auditable GPAW and EAM-LAMMPS specifications, process control, logs and result import |
| 10. STM simulator | constant-current and constant-height imaging, spectroscopy points, noise and artefacts |
| 11. AFM simulator | contact, non-contact and FM-AFM with numeric image export |
| 12. Reproducibility and interchange hub | Python, projects, history, provenance, checkpoints, plots and scientific file conversion |
| 13. Harmonic vibration workbench | finite-displacement force constants, Gamma-point modes, imaginary-mode detection and zero-point energy |

The external boundary is important. GPAW is installed and has passed live
tests here. LAMMPS is driven end to end for EAM, but no real LAMMPS is
installed here, so its three live parity tests remain blocked. A user-supplied
ASE calculator can run. Quantum ESPRESSO, GROMACS, ORCA, AMS, CP2K, PySCF,
Psi4, Wannier90 and OpenMX are detected and declared but not driven. Materia
does not yet replace those packages.

## Measured performance

Fresh measurements from `materia bench` on this machine:

| Operation | Size | Time |
| --- | ---: | ---: |
| Neighbour list | 99,372 atoms | 0.433 s |
| Bond perception | 99,372 atoms | 0.665 s |
| Stillinger-Weber energy and forces | 99,372 atoms | 0.673 s |
| FIRE relaxation, at most 20 steps, starting near its minimum | 10,092 atoms | 0.155 s |
| STM constant-current image | 128 atoms, 128 by 128 pixels | 1.174 s |
| AFM frequency-shift image | 128 atoms, 160 by 160 pixels | 1.971 s |
| EAM MD, 10 Langevin steps | 10,976 atoms | 1.973 s |
| Ewald bulk | 10,648 ions | 3.345 s |

The one direct speedup measured in the same implementation and run is EAM
neighbor-list reuse:

| Atoms | First force evaluation | Reused list | Speedup | Time reduction |
| ---: | ---: | ---: | ---: | ---: |
| 864 | 0.0260 s | 0.0091 s | 2.85x | 64.9% |
| 10,976 | 0.3695 s | 0.1300 s | 2.84x | 64.8% |
| 97,556 | 3.7476 s | 1.2750 s | 2.94x | 66.0% |

The 97,336-ion Ewald calculation took 102.85 seconds and recovered the NaCl
Madelung constant as 1.747564597 against 1.747564594633, a relative difference
of about 1.4 parts per billion. This validates the numerical point-charge sum,
not the suitability of a point-charge model for every real material.

## Researcher time

Materia removes manual file handoffs from these implemented paths:

| Workflow | Manual handoffs inside Materia |
| --- | ---: |
| Build, edit, relax, inspect, scan and export | 0 |
| Ground state, relaxation, DOS, band structure, plot, CSV and provenance | 0 |
| Collision setup, dynamics, playback and fragment inspection | 0 |
| Wafer region extraction, atomistic editing and project replay | 0 |

That is real consolidation, but the percentage of human time saved has not
been measured in a controlled study. Therefore Materia cannot currently claim
that a task is 15 times faster than LAMMPS, Quantum ESPRESSO, GROMACS, ORCA,
ASE, AMS or Materials Studio. Solver runtime comparisons would also be invalid
unless the physical model, tolerances, hardware and requested outputs were
identical.

A publishable time-saving claim needs a preregistered task set, the same input
structures and scientific tolerances, experienced users, screen recordings and
timing from task start to a validated result. The first useful comparison is
Materia against an ASE plus GPAW script for structure preparation, relaxation,
DOS, provenance and export. The second is Materia against LAMMPS plus a
trajectory viewer for EAM relaxation and dynamics. Until those trials exist,
the defensible claim is fewer manual handoffs, not a fabricated multiplier.

## Scientific proof already present

The complete suite on this snapshot reports **1,448 passed, 3 skipped and 48
warnings** in 1110.32 seconds. The three skips are only the explicit live
LAMMPS parity cases, blocked because LAMMPS is not installed. The warnings are
an ASE use of a NumPy shape-setting API that NumPy 2.5 has deprecated.

The validation suite covers crystallography, Stillinger-Weber, tight binding,
STM decay, AFM quadrature, surface reconstruction, GPAW ground states,
relaxation, DOS and band structure, point-charge electrostatics, rigid-ion
silica and EAM metals. Highlights
include live GPAW H2 and silicon calculations, an independently recomputed
0.7336 eV silicon PBE Kohn-Sham gap on the stated grid, exact DOS state counts,
published Madelung constants and EAM comparisons with tabulated LAMMPS and ASE
reference values. [VALIDATION.md](VALIDATION.md) records every tolerance and
limitation.

There is no validation against experimental STM images, published DFT defect
formation energies, reactive chemistry, biomolecular force fields, nuclear
physics, particle collisions or a real local LAMMPS installation. Those are
not product capabilities today.

## License

Materia source code and documentation are licensed under the MIT License. The
top-level `LICENSE`, package metadata, citation metadata, application About
text, launcher metadata and plug-in examples agree. Material definitions stay
CC0-1.0. The three shipped OpenKIM EAM parameter files stay under their public
domain dedication. External solvers and PAW datasets retain their own licenses
and are not relicensed or bundled by Materia. A clean wheel build reports
`License: MIT`, carries the OSI MIT classifier and includes the MIT text.
This checkout has no configured Git remote, so the source is MIT licensed but
has not been verified as publicly published.
