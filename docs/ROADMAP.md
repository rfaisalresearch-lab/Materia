# Roadmap

Ordered by dependency: each item needs the ones above it. Not by how
interesting it is.

## Shipped in 0.1

The vertical slice: desktop application, material browser, crystal and surface
generation, procedural wafer with region extraction, silver greyscale STM,
movable tip, hover identification, atom inspection with shell diagram and
orbital information, defect and dopant editing, relaxation, documented STM and
AFM models, embedded real Python, save and load, undo and redo, structure and
image export, automated tests, and this documentation.

## The research workbench: established solvers behind one experiment system

The long-term aim is a variable-driven research workbench that produces
reproducible physical hypotheses by driving established open-source solvers
through one consistent, validated experiment system. It will not become an
exact universal simulator, and it will not label a computational pattern as a
theorem without a formal proof.

**Done: the self-consistent ground-state DFT experiment**
(`docs/DFT_GROUND_STATE.md`). An immutable, versioned specification of every
variable; boundary rules for clusters, wires, slabs and bulk crystals; charged
and spin-polarised systems; an optional uniform field along open directions;
energies, forces, stress, densities, spin density, electrostatic potential,
eigenvalues, occupations, Fermi level, magnetic moment, electron accounting
and SCF history from real GPAW runs; chunked storage of grids; staleness,
cancellation, persistence and restart reuse; a bounded convergence laboratory;
and the result classifications a later discovery engine will need.

**Done: geometry and variable-cell DFT relaxation**
(`docs/DFT_RELAXATION.md`). A versioned relaxation specification built on the
ground-state one; fixed-cell relaxation for every boundary class and
variable-cell relaxation for bulk crystals in plane-wave mode; every variable
echoed back by GPAW or the worker; fixed atoms, symmetry, charge, spin, field,
k-points, cutoff, smearing and SCF tolerances kept; transactional storage;
one undoable apply that refuses stale geometry; validated against live GPAW
runs, each checked by an independent ground-state calculation.

**Done: DFT density of states and projected density of states**
(`docs/DFT_DOS.md`). A frozen, versioned DOS specification can use a structure,
a stored ground state or a stored relaxation. It drives GPAW's fixed-density
step with an explicit energy grid, broadening, k-points, band count, spin
channels and PAW-projector selections. It independently recomputes the total
DOS from returned eigenvalues, refuses incomplete bands, stores checksummed
arrays transactionally, tracks current, stale and detached state, plots the
curves in the desktop interface and exports CSV.

**Done: DFT electronic band structure** (`docs/DFT_BAND_STRUCTURE.md`). A
frozen, versioned band-structure specification pins the ground state, its
symmetry choice and an explicit reciprocal-space path: special points
generated once from ASE's Bravais-lattice tables, branches, segment intervals
and every k-point, stored and never regenerated invisibly. Valid for bulk,
slab and wire; refused for clusters. GPAW computes exactly the stored k-points
with symmetry off; the k-points in order, band count, labels, breaks and the
path distance recomputed from the pinned cell are verified before
transactional, checksummed storage with current, stale, detached and corrupt
states. A desktop band plot, CSV export and provenance inspector complete it.
Validated live on silicon, iron, a graphene sheet and a relaxed crystal.
Orbital character is deferred.

Each item below needs the ones above it.

1. LDOS and orbital views, including physically normalised orbital character of bands.
2. Vibrations, phonons, zero-point energy and thermodynamic free energies.
   Harmonic Gamma-point modes and temperature-dependent harmonic free energy,
   internal energy, entropy and heat capacity for classical and ASE force
   models are done (`docs/PHONONS.md`). Periodic real-space force constants,
   explicit dispersion paths and DOS meshes are built and await the final
   combined verification phase (`docs/PHONON_DISPERSION.md`). DFT force
   sources and Brillouin-zone-converged bulk thermodynamics remain.
3. Transition states, NEB, reaction barriers and kinetic models. Classical
   fixed-cell NEB and climbing-image barriers are built and await the final
   combined verification phase (`docs/NEB.md`). DFT image scheduling, adaptive
   paths, transition-state frequencies and kinetic rate models remain.
4. Spin ordering, spin-orbit coupling, magnetic fields and relativistic
   provenance.
5. External electric fields beyond the static uniform field, dielectric
   response, optical spectra and TDDFT.
6. Self-consistent charge methods and scalable reactive dynamics.
7. Electron transport and open-system chemical potentials.
8. Automated structure search and phase exploration. Deterministic fixed-cell
   basin hopping with ranked, deduplicated, persistent classical minima is built
   and awaits the final combined verification phase (`docs/STRUCTURE_SEARCH.md`).
   Variable-cell, variable-composition and symmetry-aware exploration remain.
9. Uncertainty ensembles, sensitivity analysis and independent-solver
   replication. Classical force disagreement and reference-safe energy-change
   ensembles are built and await the final combined verification phase
   (`docs/MODEL_ENSEMBLES.md`). Calibrated parameter uncertainty, DFT
   replication and experimental coverage remain.
10. Dimensionally constrained equation discovery and candidate-law testing.

Of (1), spatially resolved LDOS and Tersoff-Hamann images are done
(`docs/DFT_LDOS.md`), validated bit for bit against ASE's STM code on the
same GPAW job. A defensible orbital-character projection remains.

Classical atomistic collision setup and trajectory playback are now complete
for EAM-covered systems. Positions and velocities persist as checksummed arrays,
and every frame supports pause, scrub and fragment inspection. Reactive chemistry,
biomolecular force fields, first-principles molecular dynamics and high-energy
particle collisions remain separate backend tracks.

External constant forces and harmonic positional restraints are built as
energy-consistent classical bias terms and await the final combined verification
phase (`docs/EXTERNAL_MECHANICS.md`). Time-dependent force schedules, moving
restraints, feedback controllers and direct experimental-apparatus models remain.

A general before-and-after structural analysis backend is built and awaits the
final combined verification phase (`docs/STRUCTURE_COMPARISON.md`). It matches
stable atom ids, separates periodic cell deformation from non-affine movement,
aligns clusters, and records atom, bond and finite-strain changes. A dedicated
interactive comparison view remains.

## Next: completing the foundations

**1. More surface reconstructions.**
Depends on: nothing new. The generator interface, the registry, the provenance
record, the comparison against published geometry and the refusal path all
exist and are exercised by Si(100)-2×1 (`docs/RECONSTRUCTIONS.md`). What
remains is generators. Si(111)-7×7 (dimer-adatom-stacking-fault) and the
Au(111) herringbone are the two that matter most for recognisability; both are
large-cell constructions rather than local pairings, so each is its own piece
of work. Until then they stay declared and refused.

Buckled Si(100), the c(4×2) and p(2×2) ground states, is blocked on a
different thing: an electronic total energy. No classical potential can produce
the dimer buckling, so this waits on a driven Tier-3 adapter rather than on
another generator.

**2. Amorphous structures.**
Depends on: (1)'s generator interface. Melt-quench through the existing
molecular dynamics for a-Si, and a Wooten-Winer-Weaire bond-switching
generator for a-SiO₂, which is what thermal gate oxide actually is.

**3. Ewald electrostatics.** Done: point-charge Ewald for crystals,
Yeh-Berkowitz corrected Ewald for slabs, direct sums for clusters, with
convergence checks, refusals and Madelung validation
(`docs/ELECTROSTATICS.md`), and rigid-ion relaxation and dynamics with the BKS
silica potential (`docs/RIGID_ION.md`). What remains on this line: a Coulomb
stress tensor for variable-cell relaxation, more rigid-ion and shell-model
parameterisations, an applied external field, and
particle-mesh Ewald if structures beyond a few tens of thousands of ions need
it.

**4. Embedded-atom potentials.** Done for Cu, Au and W (`docs/EAM.md`):
checksummed public-domain setfl files, LAMMPS-equivalent interpolation,
energy, forces, stress, relaxation and dynamics, validated against LAMMPS and
ASE. What remains on this line: more elements and a validated alloy file,
cell relaxation, and the funcfl and Finnis-Sinclair formats if a potential
worth shipping needs them. The text below is the original plan.

**4a. Embedded-atom potentials (original plan).**
Depends on: (3) for mixed systems. Removes the "metals are Lennard-Jones only"
limitation and makes gold, copper and tungsten quantitative.

## Then: electronic structure worth the name

**5. Self-consistent-charge DFTB.**
Depends on: (3). This is the single biggest step in fidelity: charge transfer,
band bending, response to a field, and charged systems all become possible, and
it removes the largest block of refusals in the program.

**6. k-resolved STM integration.**
Depends on: (5) or the existing Tier 2. Replaces the small surface-zone grid
with a proper Brillouin-zone integral.

**7. Spin polarisation.**
Depends on: (5). Independent spin channels, spin density, and spin-polarised
tunnelling.

**8. Tier-3 adapters driven end to end.**
Depends on: nothing new; the adapter interface exists. Run GPAW and Quantum
ESPRESSO from the interface with input generation, job control and result
import, so a user can move from Tier 2 to Tier 3 without leaving the program.
The ordered adapter sequence is LAMMPS, Quantum ESPRESSO, GROMACS and ORCA.
AMS and Materials Studio are optional licensed integrations. Each must satisfy
the execution and provenance contract in `docs/EXTERNAL_SOLVER_BACKENDS.md`.

## Then: instruments and analysis

**9. AFM with chemical bonding.**
Depends on: (5) or (8). Forces from an electronic-structure calculation instead
of Lennard-Jones. This is what makes simulated AFM contrast trustworthy.

**10. Tip-apex models.**
Depends on: (9). Explicit apex clusters, CO functionalisation, and apex
relaxation, which is what dominates real high-resolution AFM.

**11. Spectroscopy suite.**
Depends on: (5). dI/dV maps, inelastic tunnelling spectroscopy, and force
spectroscopy as first-class measurements with their own records.

**12. Phonons and thermal transport.**
Depends on: (4). Done: finite-displacement force constants with sum rules,
Gamma-point normal modes, imaginary-mode reporting and zero-point energy for
classical force models and ASE calculators, plus Gamma-sampled harmonic free
energy, internal energy, entropy and heat capacity (`docs/PHONONS.md`).
Built and awaiting the final combined verification phase: dispersion from
lattice-resolved force constants and phonon DOS. Remaining: the non-analytic
LO-TO term, DFT force sources, Brillouin-zone-converged bulk thermodynamics
and temperature-dependent corrugation.

## Then: scale

**13. Explicit extended defects.**
Depends on: (4). Dislocations, stacking faults and grain boundaries as
atomistic objects with Burgers vectors and misorientation, not as a procedural
field.

**14. Million-atom regions.**
Depends on: (13) for anything worth looking at at that size. Needs delta-encoded
undo, out-of-core structures, and a WebGPU renderer.

**15. Multi-region projects.**
Depends on: (14). Several atomistic regions from the same wafer, compared side
by side, with their provenance linked to a shared parent.

## Continuous

* Material coverage, each with a validation case that checks the generated cell
  against cited geometry.
* Validation against published DFT and experimental data as adapters mature.
* Performance measurement on more hardware, reported rather than estimated.
* Accessibility: making the viewports navigable rather than only their numeric
  panels.

## Explicitly not planned

* A cloud service, an account system, or telemetry of any kind.
* Bundling GPL solvers.
* Any feature that would require presenting an approximation as exact.
