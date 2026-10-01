# Known limitations

Written plainly, because a user who does not know these will draw wrong
conclusions. Nothing here is hidden behind a caveat in a tooltip; the
application states each of these at the point where it matters.

## Scientific

### The models

* **One first-principles code is driven for the ground state, relaxation and
  DOS.** GPAW runs as a ground-state experiment
  (`docs/DFT_GROUND_STATE.md`): energies, forces, plane-wave stress,
  densities, spin density, electrostatic potential, eigenvalues, occupations,
  Fermi level and magnetic moment for clusters, wires, slabs and bulk cells,
  including charged and spin-polarised systems. It relaxes positions at fixed
  cell for every boundary class, and the cell as well only for bulk crystals
  in plane-wave mode (`docs/DFT_RELAXATION.md`): no variable-cell relaxation
  of slabs, wires, clusters or charged cells, no constraints beyond fixed
  atoms, fixed atoms only with symmetry off, local minima only, and the
  k-point divisions and cutoff stay fixed while a cell changes. It is
  optional and must be installed separately with its PAW datasets. Total DOS
  and PAW-projector PDOS are implemented (`docs/DFT_DOS.md`), and so are
  Kohn-Sham band structures along explicit paths for bulk crystals, slabs and
  wires (`docs/DFT_BAND_STRUCTURE.md`): no band paths for clusters, band edges
  read along the sampled path only, bands ordered by energy rather than
  disentangled at crossings, a Fermi-level energy zero rather than the vacuum
  level, and no orbital character (fat bands), which is deferred because PAW
  projector weights are not normalised orbital populations. Spatially
  resolved LDOS and Tersoff-Hamann STM images are implemented
  (`docs/DFT_LDOS.md`) from pseudo-wavefunctions, which are exact in the
  vacuum and not inside the PAW spheres, with an s-wave tip and no conversion
  to amperes. The DFT path computes no vibrations, transition states,
  spin-orbit coupling, response functions or excited states; harmonic
  vibrations exist for classical force models and ASE calculators only
  (`docs/PHONONS.md`). Those are the
  next phases in `docs/ROADMAP.md`, in dependency order. The older single-point
  adapter `external:gpaw` still refuses charged and spin-polarised systems.
* **A density functional has systematic errors.** Converging a DFT result
  with respect to grid, cutoff, k-points or vacuum does not remove the
  functional's error: PBE overestimates the H2 bond by about 1.4 percent, and
  semilocal functionals underestimate band gaps by tens of percent. No result
  carries an uncertainty for that error; the provenance names the functional
  and says so.
* **Charged periodic DFT is limited.** Charged slabs and wires are refused.
  Charged bulk cells use a uniform compensating background and keep an
  uncorrected finite-size error, estimated in a warning, that decays only as
  one over the cell size.
* **Only the installed PAW datasets.** LDA, PBE, RPBE and revPBE here; a
  functional without datasets (PBEsol, BLYP) is refused, not substituted.
* **LAMMPS is driven for EAM only, and its live parity is unverified here.**
  Energy, forces, stress, fixed-cell relaxation and NVE, Langevin and
  Nose-Hoover dynamics are driven for `pair_style eam/alloy` in `metal` units
  with `atomic` atoms. Other pair styles, barostats and variable-cell
  relaxation, boundary styles `f` and `m`, charges, molecular topology, MPI,
  accelerator packages and restart files are refused. Cells must already be in
  the LAMMPS restricted triclinic orientation. A Langevin trajectory is
  reproducible only with the same seed, LAMMPS build and processor count. The
  adapter is tested against a fake process; the comparison with a real LAMMPS
  is blocked until one is installed (`docs/VALIDATION.md`). The desktop panel
  offers LAMMPS as a backend of the EAM section, not as a separate tool.
* **Every other external adapter is still undriven.** Quantum ESPRESSO,
  GROMACS, ORCA, AMS, CP2K, PySCF, Psi4, Wannier90 and OpenMX are declared,
  detected and refused.
  ASE runs only with a calculator object supplied from Python, and the fidelity
  of what it returns is taken from that calculator, because ASE is an interface
  and computes nothing itself.
* **No solver shipped with Materia is a first-principles calculation.** Tier 2
  tight binding is *fitted* to experimental band structures. It reproduces the
  quantities it was fitted to (silicon's 12.5 eV valence bandwidth, its 1.17 eV
  indirect gap, GaAs's 1.55 eV direct gap) and it is not predictive for
  systems outside its fit. Use a Tier-3 adapter for prediction.
* **The tight-binding model is not self-consistent.** No charge transfer, no
  band bending, no response to an applied field, no screening. A charged system
  is refused outright.
* **Surface-state energies are not quantitative.** The parameters were fitted
  to bulk bands. Dangling-bond states appear with the right orbital character
  at approximately the right place in the gap, and that is all. Every run on a
  surface warns about this.
* **Shallow-donor binding energies are not reproduced.** The 45 meV of P in Si
  needs the long-range Coulomb tail of the ionised donor and a central-cell
  correction. A short-range on-site shift in a finite non-self-consistent
  supercell has neither.
* **Substitutional impurities in the classical potential are geometric only.**
  The covalent-radius sigma rescaling gives the sign and rough size of the
  local bond-length change, nothing more. It carries no valence and no
  electronic information.
* **Three metals have a quantitative potential; the rest do not.** Cu, Au and
  W use the Zhou 2004 embedded-atom potentials (`docs/EAM.md`), validated
  against LAMMPS and ASE on the same files. Every other metal still has only
  the Lennard-Jones fit derived from cohesive energies, which is for
  interactive geometry: metallic bonding is not pairwise. No alloy potential
  ships, and the single-element files are never combined. The cell is not
  relaxed. The Cu and Au cutoffs coincide with a neighbour shell at
  equilibrium, which makes their lattice constants depend slightly on the
  minimiser.
* **Classical nuclei.** No zero-point energy, no quantum heat capacity. Below
  the Debye temperature the equipartition temperature is not the thermodynamic
  temperature of the real solid.
* **Electrostatics is point charges only.**
  Ewald, slab-corrected Ewald and direct sums give the exact Coulomb energy,
  forces, site potentials and site fields of charges the user supplies
  (`docs/ELECTROSTATICS.md`). There is no polarisation, no screening and no
  charge transfer. Point charges alone do not relax or run dynamics; the
  rigid-ion model does (`docs/RIGID_ION.md`), with one shipped
  parameterisation (BKS silica), a fixed cell and no shell-model
  polarisation. One-dimensional (wire)
  geometries and charged slabs are refused. Charged bulk cells need an
  explicit uniform background and carry an uncorrected Makov-Payne
  finite-size error. Nothing applies an external electric field yet.
* **Vibrations are harmonic and Gamma-point only.** Finite-displacement force
  constants give normal modes of isolated systems and the Gamma-point modes of
  periodic cells (`docs/PHONONS.md`). Dispersion is refused, polar crystals
  lack the LO-TO correction, the zero-point energy of a crystal is a
  Gamma-point sample, and no anharmonic or thermal quantity is computed.
* **Local optimisation finds local minima.** FIRE relaxes to the nearest
  minimum. In Stillinger-Weber silicon the ideal vacancy is itself a local
  minimum at 4.34 eV, because its four neighbours sit just beyond the cutoff;
  the reconstructed vacancy at 2.69 eV (216 atoms) is reached only from a
  start with those neighbours displaced inward. Materia does not search for
  lower minima on its own.
* **A PAW dataset is not an all-electron calculation.** Through Materia and
  directly, GPAW with its shipped copper dataset puts the PBE volume 1.2 percent
  and the bulk modulus 3 percent from the WIEN2k all-electron values; silicon
  and aluminium agree within 0.4 percent. See `docs/ACCURACY_BENCHMARKS.md`.
* **Automatic k-point grids follow the cell size.** Two cells of different size
  can get different automatic grids, and their total energies are then not
  comparable. The check warns whenever the automatic grid is in use; fix
  `kpoints` when comparing strained, scaled or relaxed cells. The equation-of-
  state experiment (`docs/DFT_EQUATION_OF_STATE.md`) pins the grid for you; it
  scales isotropically with fractional coordinates fixed, so it does not relax
  internal parameters or c/a.

### The instruments

* **STM currents are in arbitrary units.** The Tersoff-Hamann prefactor
  contains the tip density of states and the apex geometry. The setpoint is
  calibrated against the computed signal, which is what an experimental
  feedback loop does; absolute currents are not predicted.
* **The tip has no atomic structure.** Real tips have an unknown apex that
  dominates the apparent corrugation. Only the s-wave apex is within the
  Tersoff-Hamann derivation; `pz` and `dz2` are applied as a directional
  weighting and labelled as outside the theory.
* **No tip-induced band bending, no inelastic channels, no spin-polarised
  tunnelling.**
* **AFM contrast has no chemical bonding.** Lennard-Jones plus Hamaker gives
  dispersion and Pauli repulsion. True atomic contrast on semiconductors is
  dominated by covalent tip-sample bonding, which needs an electronic-structure
  force calculation. There is also no tip-apex relaxation, no CO-tip lateral
  flexing, no dissipation channel and no electrostatic force.
* **The scanning-probe electronic structure is computed on a truncated slab**
  (the topmost ~9 Å). Deeper atoms cannot reach the vacuum, but the truncation
  creates its own lower surface; its states are excluded from the vacuum sum.

### Structures

* **Displayed bonds are geometric, not chemical.** A pair is bonded when it is
  closer than 1.15 times the sum of covalent radii and shares a Voronoi face of
  at least 5 percent of the solid angle, which gives the textbook coordination
  of every shipped crystal, corundum, MoS2 and bcc tungsten included. Above
  20,000 atoms, and for flat or very small systems where a Voronoi
  decomposition is undefined, only the distance criterion is applied, and every
  bond records which rule produced it. Bond orders are always 1.
* **Adatoms start at a covalent contact.** Without an explicit height an adatom
  is placed at the sum of covalent radii from the nearest surface atom, a
  starting geometry to relax, not an adsorption site search.

* **One surface reconstruction is generated; the rest are declared and
  refused.** Si(100)-2×1 is built end to end: the generator derives the dimer
  pairing from the slab's own back-bond geometry and the recommended potential
  relaxes it, so the bond length reported is computed, not entered. Si(111)-7×7,
  the Au(111) herringbone, GaAs(100)-β2(2×4), the SiC √3×√3 R30 and 6√3 buffer
  are listed in their material files with `implemented: false` and a reference.
  Asking for one of those returns an explicit unsupported record; Materia builds
  the ideal truncation and says so. See `docs/RECONSTRUCTIONS.md`.
* **The Si(100) dimer comes out symmetric, and the real one is buckled.**
  Stillinger-Weber relaxation gives a 2.4035 Å symmetric dimer. LEED (Over
  *et al.*, Phys. Rev. B **55** (1997) 4731) finds 2.24 ± 0.08 Å with a 0.72 Å
  vertical buckling and a 19° tilt. The bond is 7.3 % long and outside the
  quoted uncertainty; the buckling is zero. The buckling is a Jahn-Teller
  distortion driven by charge transfer between the two dimer atoms, and a
  three-body potential of nuclear coordinates has no mechanism for it. The
  low-temperature c(4×2) and p(2×2) orderings that follow from alternating
  buckling are therefore not reproduced and not approximated. Materia reports
  the deviation in every comparison rather than absorbing it.
* **No amorphous materials.** SiO₂ ships as crystalline α-quartz. Thermal gate
  oxide is amorphous. An amorphous model is on the roadmap.
* **No grain boundaries, dislocations or stacking faults as explicit atomistic
  objects.** Grains exist only as a procedural field at the microstructure
  scale.
* **The procedural wafer is synthetic.** It is reproducible and statistically
  plausible; it is not a measurement of any real wafer, and it never claims to
  be. Grains are a Voronoi tessellation, which real microstructure is not.
* **No alloys or partial occupancy.** `occupancy` is parsed but every site is
  treated as fully occupied.

## Engineering

* **Coincident atoms are refused, not modelled.** Distinct atoms within
  1e-5 A of each other, or of each other's periodic images, have no energy in
  any classical, tight-binding or EAM model and every calculation on them is
  refused. Editing tools still allow such a geometry to be created; they warn,
  and undo restores the previous structure.
* **Undo stores full structure snapshots.** Fine for the tens of thousands of
  atoms Materia instantiates; a million-atom region would need delta encoding.
  The stack is capped by depth and by a memory budget. Operations that only
  change project topology (adding a structure, creating a wafer) copy no
  atoms, because the snapshot holds references and every content change carries
  its own copy.
* **The undo stack is in memory only.** The permanent history *log* is saved
  with the project and reloads with it, but the snapshots behind undo are not,
  so reopening a project starts with an empty undo stack. Checkpoints are the
  mechanism that survives a reopen.
* **Restricted Python mode is a guard rail, not a sandbox.** CPython cannot be
  sandboxed from inside. See [SECURITY.md](../SECURITY.md).
* **Plug-ins are not sandboxed.** Loading one executes its code.
* **The renderer is WebGL2, not WebGPU.** WebGPU is not yet uniformly available
  in the system web views that host the application. The renderer is isolated
  in one module.
* **The Tauri shell is not built or tested here.** The architecture is
  Tauri-shaped (a native shell around a web view plus a separate core), so the
  shell is swappable, but this release ships the pywebview shell, which is what
  has been run and verified.
* **The macOS bundle is not code signed.** Gatekeeper will ask for confirmation
  the first time it is opened from Finder.
* **No k-resolved STM integration.** The tunnelling sum uses a small
  surface-Brillouin-zone grid, converged for corrugation to within a couple of
  per cent, not a full integral.
* **Large scans are slow.** A 256×256 constant-current image of a 640-atom
  region takes tens of seconds on the machine in [PERFORMANCE.md](PERFORMANCE.md).
  The electronic structure is cached, so re-imaging is much faster, but the
  first scan after any edit pays full price.
* **Single project per window.** No comparison view of two projects side by
  side.
* **Accessibility is partial.** Keyboard navigation, rebindable-ready
  shortcuts, high-contrast and density modes, screen-reader labels on controls
  and numeric alternatives to every visual measurement are present. The
  viewports themselves are canvases and are not navigable by a screen reader;
  the numeric panels are the accessible path to the same information.

* **DFT grids are capped at 256 MB each.** A requested density, spin density
  or potential that would exceed it on GPAW's fine grid is refused; there is
  no out-of-core storage.
* **DFT restart data is a local cache.** It lives in `~/.materia/dft-restart`
  (bounded to 16 files and 2 GB) and is not saved with the project.
* **DFT runs are serial.** GPAW's MPI parallelism is detected and reported but
  not used.

## Validation

* Validation covers crystallography, the classical potential, the tight-binding
  band structures, the tunnelling decay, the AFM quadrature, the embedded-atom
  potentials, point-charge electrostatics and the ground-state, relaxation and
  DOS DFT experiments.
  See [VALIDATION.md](VALIDATION.md).
* There is **no** validation against experimental STM images, against published
  DFT surface energies, or against measured defect formation energies. Those
  require the Tier-3 adapters and reference datasets that are not shipped.
* The LAMMPS live parity cases are **blocked**, not passed: no LAMMPS was
  installed where the adapter was written, so they skip with that reason.
* A passing software test is not scientific validation. The suites are labelled
  separately and reported separately, and that distinction is the point.
