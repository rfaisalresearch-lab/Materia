# Embedded-atom potentials for metals

Materia computes energies, forces and stresses of metals with the
embedded-atom method, relaxes positions and runs molecular dynamics with them,
using parameter files read verbatim and checked by checksum. Three potentials
ship with the program: Cu, Au and W from the database of Zhou, Johnson and
Wadley (2004). Any other setfl file can be used from Python.

Code: `materia/physics/eam.py` (parser, interpolation, energy, forces,
stress), `materia/potentials/eam/` (the files, `manifest.json`, `LICENSE`),
`materia/python_api/api.py` (`EAMNamespace`), `materia/desktop_ui/service.py`
(`eam_*`). Tests: `tests/unit/test_eam.py`,
`tests/integration/test_eam_service.py`, `tests/validation/test_eam_validation.py`.

## The model

```
E = sum_i F_a(rho_i) + 1/2 sum_{i != j} phi_ab(r_ij),    rho_i = sum_{j != i} f_b(r_ij)
```

`F_a` is the embedding energy of an atom of element `a` in the host density
`rho`, `f_b` the density a neighbour of element `b` contributes, and
`phi_ab` a pair repulsion (Daw and Baskes 1984; Foiles, Baskes and Daw 1986).
Every function is read from the file. Nothing is fitted, mixed or inferred
in Materia.

Units: distances in A, energies in eV, forces in eV/A, stress in eV/A^3
(reported in GPa where shown). The density is in the arbitrary units of the
file and is never presented as a physical electron density.

## The shipped potentials

| Identifier | Element | Intended structure | File | SHA-256 (first 16) | Cutoff |
| --- | --- | --- | --- | --- | --- |
| `Cu-Zhou04` | Cu | fcc | `Cu_Zhou04.eam.alloy` | `34bcf92bc7e14b63` | 5.715752 A |
| `Au-Zhou04` | Au | fcc | `Au_Zhou04.eam.alloy` | `746b008d8849eea5` | 6.451132 A |
| `W-Zhou04` | W | bcc | `W_Zhou04.eam.alloy` | `4ae0749ce44dc313` | 6.128705 A |

**Source.** The OpenKIM archives
`EAM_Dynamo_ZhouJohnsonWadley_2004NISTretabulation_{Cu,Au,W}__MO_..._001`
(doi:10.25950/a2be92ad, 10.25950/7833e99e, 10.25950/2d54fc66), retrieved
2026-09-25. These are the NIST retabulation by L. M. Hale, generated with
`Zhou04_create_v2.f` to remove single and double precision fluctuations in
the original tables. The SHA-1 of each file matches the checksum recorded in
the archive's own provenance record.

**Licence.** Each OpenKIM archive contains a LICENSE file whose complete text
is "This work is dedicated to the public domain." The files are shipped
unmodified under that dedication, not under Materia's MIT licence
(`materia/potentials/eam/LICENSE`). The same files are also posted on the NIST
Interatomic Potentials Repository without a licence statement; Materia did not
take them from there.

**Citations.** X. W. Zhou, R. A. Johnson and H. N. G. Wadley, Phys. Rev. B 69
(2004) 144113; X. W. Zhou, H. N. G. Wadley, R. A. Johnson et al., Acta Mater.
49 (2001) 4005; M. S. Daw and M. I. Baskes, Phys. Rev. B 29 (1984) 6443.

**What the authors say about it.** The database was built for vapor-deposited
metal multilayers and, in their words as recorded by NIST, was not designed
for intermetallic compounds.

## Format and parsing

Only setfl, the format of LAMMPS `pair_style eam/alloy`, is implemented,
because it is what the three verified potentials use. funcfl and
Finnis-Sinclair files are not read, and no partial parser for them exists.

A file is refused, with the reason, if it is truncated, has values after its
last table, has a non-numeric or non-finite value, names an unknown or
repeated element, gives an atomic number that does not match its symbol, has
fewer than five points in a table, or declares a cutoff beyond its own radial
grid. A shipped file whose bytes no longer match its manifest checksum is
refused. A user-supplied file is identified by its SHA-256; its licence is
recorded as unknown.

## Interpolation and cutoff

Tabulated functions are interpolated with the piecewise cubic Hermite scheme
of LAMMPS `pair_style eam` and the OpenKIM EAM_Dynamo driver: fourth-order
finite-difference slopes, one cubic per interval, derivatives from the same
cubic. Forces are therefore the exact gradient of the interpolated energy.
Pairs closer than the cutoff given in the file interact; the rest do not. A
density above the last tabulated value is extrapolated linearly, as LAMMPS
does, and every result reports how many atoms needed it.

**The cutoff sits on a neighbour shell for Cu.** The tabulated functions are
not exactly zero at the cutoff: for Cu the density function is 2.9e-6 and
the pair term -1.0e-6 eV there. The fifth-neighbour shell of fcc Cu,
`sqrt(5/2) a`, reaches the 5.7158 A cutoff at `a_c = 3.614959 A`, and the
energy steps up by about 1.2e-5 eV per atom as that shell leaves. Below `a_c`
the energy falls all the way to the step; above it there is a stationary
point at 3.6149789 A. NIST's LAMMPS calculations report 3.61470, 3.61479 and
3.61498 A from different relaxation methods, which is consistent with this
step; the last of those is the stationary point above it. Au's
equilibrium lies 8e-7 A inside the corresponding crossing. Elastic constants
are therefore evaluated from the virial stress with strains of 1e-8, which do
not move any shell across the cutoff; energy second differences with larger
strains would straddle the step and give meaningless numbers for Au. This is
a property of the files. LAMMPS and OpenKIM behave the same way.

## Coincident atoms

Two distinct atoms, or an atom and a periodic image of another, closer than
1e-5 A are refused with the ids, elements, separation and lattice image of the
pair. 1e-5 A is 1000 fm, about ten times the diameter of the largest nuclei
and four orders of magnitude below any interatomic distance that occurs in a
material, so no real structure, relaxation step or dynamics step reaches it,
and it lies far above coordinate rounding. The same constant,
`COLLISION_TOLERANCE_A` in `materia/physics/neighbors.py`, is used by every
classical potential.

Only a true self-pair, the same atom in the zero lattice image, is left out of
the pair list, and it is identified by index and image, never by distance.
The check runs on every evaluation: before a neighbour list is built, and on
the stored list when it is reused. A list is stored only after a search that
succeeded, so a refused geometry leaves nothing behind, and the same potential
evaluates the corrected structure normally. A refused energy, relaxation or
dynamics run stores no number, changes no coordinates and adds no undo entry
or history line; a coincidence that appears part way through a run is
reported as an unsupported record and the structure is left as it was.

## Neighbours and performance

Pairs are found with a k-d tree over explicitly expanded periodic images, so
cells thinner than twice the cutoff are handled exactly, and positions stored
outside the cell are wrapped for the search with the wrap folded back into
each pair's image shift. The list is built at cutoff plus a 0.5 A skin and
reused until an atom has moved half the skin. Cost is linear in the number of
atoms: about 0.11 s per energy and force evaluation for 11,000 atoms and 1.2 s
for 98,000 on the benchmark machine (`docs/PERFORMANCE.md`).

## Tasks

| Task | What happens | Applied to the structure |
| --- | --- | --- |
| energy | energy, forces, and stress for fully periodic cells | forces |
| relax | FIRE on positions at fixed cell, to `fmax_eV_A` (default 0.01) within `max_steps` (default 1000) | positions and forces, only if converged |
| md | velocity Verlet, NVE or Langevin (BAOAB), seeded | final positions, velocities and forces |

Every run works on a copy. When it finishes, the outcome is written back as
one undoable change, and only if the structure is still exactly what was
submitted; otherwise the result is stored and marked as not applied, with the
reason. A relaxation that stops before its force tolerance is stored with
origin `estimated` and is not applied. A cancelled run stores an unsupported
record with no number and changes nothing. The cell is never relaxed.

## Selection

For the Solvers panel, `eam.run()` without a potential, and the `recommended`
model of `relax`, `energy` and `dynamics`, the choice is deterministic: the
shipped potential whose elements cover the structure's, preferring an exact
match, then fewer extra elements, then the lowest identifier. Copper, gold
and tungsten now recommend their EAM potential. A structure containing
elements split across files, such as Cu and Au, gets no potential: Materia
does not combine files, because the cross interactions would have to be
constructed rather than read. Lennard-Jones stays available for every metal,
labelled as an interactive pair approximation with no many-body metallic
bonding, and is no longer recommended where an EAM potential exists. Results
saved before this change keep the model that produced them.

## Staleness and persistence

Results are stored as `eam::<run_id>::energy`, `forces`, and for the other
tasks `relaxation_history` or `trajectory` and `mean_temperature`. Each
records the potential identifier, file name, SHA-256, licence, citations,
cutoff residuals, the file header, the task settings, and fingerprints of the
structure before and after the run. A result is current while the structure's
fingerprint equals the one the run produced, so any edit, undo or redo is
reflected at once. On reopening, the stored checksum is compared with the
installed file. No project schema change was needed.

Dynamics positions and velocities are stored at each requested sample in the
project's checksummed chunked array store. The trajectory result names those
arrays and keeps the aligned step, time, energy and temperature series. This
avoids the project writer's JSON array limit and makes damage detectable.

## Interface

**Solvers ▸ Embedded-atom metals (EAM)** shows the structure's elements, the
potential chosen for it or the reason there is none, and the potential's
family, elements, intended structure, file, SHA-256, cutoff, OpenKIM source,
origin, licence, retrieval date, citations and the authors' limitations. The
task selector offers energy and forces, relaxation (f_max, max steps) and
dynamics (steps, time step, temperature, thermostat, seed, sample interval).
Stored dynamics expose pause, play, frame scrubbing, playback speed, velocities
and contact-based fragment summaries in the 3D view. Runs are
background jobs with progress in the status bar and cancellation from the job
log. The last run shows its task, structure, potential and checksum, energy,
energy per atom, max |F|, energy change, steps, convergence, whether it was
applied, whether it is current, and the approximations.

## Python

```python
cu = materials.load("copper").bulk(repeat=(4, 4, 4))
run = eam.run(cu, "relax", fmax_eV_A=0.001)       # dict of Result
run["energy"].value, run["energy"].convergence.message
run["energy"].extra["applied"], eam.is_current()
eam.run(cu, "md", steps=500, temperature_K=600, thermostat="langevin", seed=1)
eam.catalog()                                       # identity, licence, cutoff
p = eam.potential("W-Zhou04")                       # EAMPotential
p.evaluate(structure)["stress_eV_A3"]
eam.potential("/path/to/NiAl.eam.alloy")            # user-supplied setfl file
```

## Validation

See [VALIDATION.md](VALIDATION.md#embedded-atom-metals). Lattice constants,
cohesive energies, structure ordering, C11, C12, C44, bulk moduli, relaxed
vacancy formation energies and relaxed surface energies agree with NIST's
LAMMPS calculations on the same potentials; energies and forces agree with
ASE's independent EAM calculator on the same files to 2e-11 eV per atom.

## Limitations

* Three elements ship. There is no shipped alloy potential, and no alloy is
  offered from the single-element files.
* Only setfl is read.
* The cell is not relaxed; there is no barostat. The stress is computed and
  reported but not used to change the cell.
* The Cu and Au cutoffs coincide with a neighbour shell at equilibrium, which
  makes their lattice constants depend on the minimisation method by up to
  3e-4 A (Cu).
* Agreement with the parameter publication's own property tables was not
  checked: the article was not accessible when this was written.
* An EAM potential has no electrons, no angular forces and no magnetism.
  W's bcc stability and elastic constants come from the fit, not from its
  d-band physics; surfaces, defects and alloys outside the fitting data are
  approximate.
* Nuclei are classical.
* Fragment labels in trajectory playback are covalent-radius contact groups,
  not electronic bond orders.
