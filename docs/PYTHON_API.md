# Python API

The embedded console, a notebook and a plain `.py` file all see the same
namespace, and it is the same code the graphical interface calls. Nothing here
is a stub.

## The namespace

| Name | What it is |
| --- | --- |
| `lab` | The `Lab` object: solvers, selection, checkpoints, undo |
| `project` | The active `Project`: structures, results, history, save |
| `materials` | The material registry |
| `build` | `bulk`, `surface`, `passivate`, `strain`, `interplanar_spacing` |
| `microscope` | Virtual STM and AFM |
| `measure` | Distances, angles, coordination, strain, g(r) |
| `eam` | Embedded-atom metals: energy, relaxation and dynamics |
| `collisions` | Atomistic collision setup, dynamics and fragment analysis |
| `lammps` | LAMMPS out of process: frozen specifications, audited and validated runs |
| `dft` | Versioned first-principles GPAW experiments |
| `view` | Hand results to the interface (collected headlessly) |
| `io` | Import and export |
| `defects` | Vacancies, dopants, interstitials, adatoms |
| `solvers` | The solver registry |
| `np` | NumPy |

## Materials

```python
silicon = materials.load("silicon", orientation="111")
silicon.summary()
silicon.property("band_gap")           # PropertyValue with unit and source
silicon.interplanar_spacing((1, 1, 1))
silicon.reconstructions()              # declared, each flagged supported or not
silicon.reconstructions((1, 0, 0))     # only those for one orientation

materials.list()
materials.search("carbide")
materials.add_from_file("/path/to/my_material.json")
materials.search_paths()
```

## Building

```python
surface = silicon.create_surface(
    size=(6, 6, 4),          # in-plane repeats, stacking repeats
    vacuum_angstrom=15,
    orientation=(1, 1, 1),   # overrides the material default
    fix_bottom_layers=1,     # hold the substrate during relaxation
)
crystal = silicon.bulk(repeat=(2, 2, 2))

len(surface)
surface.info()               # in-plane cell, spacing, termination, supercell matrix
surface.cell
surface.bonds()
```

### Reconstructions

A reconstruction can be applied when the slab is built, or afterwards. Both
paths run the generator and then relax with the material's recommended model;
the geometry reported is the relaxed one, never a stored coordinate.

```python
surface = silicon.create_surface(
    size=(4, 2, 8), orientation=(1, 0, 0), vacuum_angstrom=12,
    fix_bottom_layers=4, reconstruction="2x1-dimer",
)

record = surface.reconstruction()
record["n_dimers"], record["measured"]["bond_length_A"]
record["relaxation"]["model"], record["relaxation"]["converged"]
record["periodicity_check"]["is_2x1"]

for row in surface.compare_reconstruction()["rows"]:
    print(row["quantity"], row["measured"], row["reference"],
          row["within_stated_uncertainty"], row["source"])
```

On an existing slab:

```python
surface.reconstruct("2x1-dimer")                   # relaxes by default
surface.reconstruct("2x1-dimer", relax=False)      # topology only, labelled estimated
surface.reconstruct("2x1-dimer", initial_separation_A=2.8)
```

A reconstruction the material declares but this build cannot generate raises
`UnsupportedRequest`, which carries the machine-readable record. Nothing
approximate is returned in its place and no structure is added to the project.

A failed reconstruction leaves no trace at all. That holds for a bad argument,
for a reconstruction this build cannot generate, and for a failure part-way
through the generator or the relaxation: the relaxation controls are validated
before anything moves, and past that point the operation runs under a restore
point, so positions, roles, labels, bonds, metadata and the project history all
come back as they were.

Which history an undo point goes into follows what the project holds, by object
identity. A handle on the active structure records one as usual; a handle on a
structure the project holds but has not activated records one bound to that
slot, so undo restores *that* structure and leaves the active one alone; a
detached handle (`activate=False`, never registered) records none at all.

```python
from materia.python_api import UnsupportedRequest

try:
    silicon.create_surface(orientation=(1, 1, 1), reconstruction="7x7-DAS")
except UnsupportedRequest as exc:
    record = exc.as_dict()
    print(record["reason"], record["reference"], record["suggested"])
```

See [RECONSTRUCTIONS.md](RECONSTRUCTIONS.md) for what each generator does and
where its output sits against published structural data.

## Atoms and selection

```python
atom = surface.atoms[atom_id]
atom.symbol, atom.atomic_number, atom.mass, atom.mass_number
atom.position, atom.velocity, atom.force
atom.charge, atom.partial_charge, atom.magnetic_moment, atom.spin
atom.role, atom.label, atom.fixed
atom.electron_configuration
atom.coordination            # perceives bonds if needed
atom.neighbors

atom.set_charge(+1)
atom.set_spin(0.5)

surface.atoms.selected()
surface.atoms.selected_one()             # raises unless exactly one
surface.atoms.by_element("Si", "P")
surface.atoms.by_role("surface", "dopant")
surface.atoms.within((10.0, 10.0, 25.0), radius_A=6.0)
surface.atoms.neighbors_of(atom_id, radius_A=3.0)
surface.atoms.in_box(lo, hi)
surface.atoms.on_plane((1, 1, 1), offset_A=20.0, tolerance_A=0.5)
surface.atoms.table()                    # columns ready for CSV

lab.select(surface.atoms.by_element("P"))
lab.selection
```

## Editing

```python
site = surface.nearest_site((10.0, 10.0, 25.0))
surface.add_dopant("phosphorus", site=site)
surface.substitute(site, "P")
surface.create_vacancy(site)
surface.add_interstitial("Si", position=(5.0, 5.0, 20.0))
surface.add_adatom("Si", xy=(8.0, 8.0), height_A=2.2)
surface.passivate("H", side="bottom")
surface.strain([0.01, 0.01, 0.0])
surface.fix_below(z_A=12.0)
```

Every edit records an undo point and a history entry. `add_interstitial`
refuses a position that overlaps an existing atom, because that is an
unphysical configuration rather than a defect.

## Solving

```python
surface.energy()                                   # Result in eV
surface.relax(model="recommended", fmax=0.02, steps=400)
surface.dynamics(steps=500, dt_fs=1.0, temperature_K=300,
                 thermostat="langevin", seed=7)
out = surface.solve()                              # electronic structure
out["fermi_level"].value
out["total_dos"].value
out["local_dos"].value["ldos"]
out["partial_charges"].value

lab.band_structure(cell, path=[(0.5,0.5,0.5), (0,0,0), (0.5,0,0.5)],
                   labels=["L", "G", "X"], n_per_segment=50)
lab.solvers()                                      # what is registered
```

`model="recommended"` picks from the material's `recommended_models` and the
species present, including host-plus-impurity combinations. Naming a model
explicitly is always allowed.

## Embedded-atom metals

`eam` runs the embedded-atom potentials. See [EAM.md](EAM.md).

```python
cu = materials.load("copper").bulk(repeat=(4, 4, 4))
run = eam.run(cu, "relax", fmax_eV_A=0.001)   # also "energy" and "md"
run["energy"].value, run["energy"].convergence.message, run["energy"].extra["applied"]
eam.run(cu, "md", steps=500, temperature_K=600, seed=1)
eam.catalog()                  # shipped files: identity, SHA-256, licence, cutoff
eam.potential("W-Zhou04").evaluate(structure)["stress_eV_A3"]
eam.potential("/path/to/file.eam.alloy")    # user-supplied setfl
eam.runs(); eam.result(quantity="forces"); eam.is_current()
```

`run` works on a copy and writes back forces, converged relaxed positions or
the final dynamics state as one undoable change, only if the structure is
unchanged since the run started. Unknown settings, invalid values and
structures with uncovered elements raise `ApiError`.

## Atomistic collisions

`collisions` prepares momentum-conserving classical impact experiments. See
[COLLISION_DYNAMICS.md](COLLISION_DYNAMICS.md).

```python
projectile = materials.load("copper").bulk(repeat=(2, 2, 2), activate=False)
target = materials.load("copper").bulk(repeat=(3, 3, 3), activate=False)
impact = collisions.prepare(projectile, target, relative_speed_A_fs=0.08,
                            impact_parameter_A=1.5, gap_A=3.0)
run = collisions.run(impact, potential="Cu-Zhou04", steps=800,
                     dt_fs=0.1, sample_every=4)
parts = collisions.fragments(impact)
```

The trajectory is saved in checksummed project arrays. The desktop EAM panel
can pause, scrub and replay it and report fragments for each frame. This models
classical nuclei under the selected potential, not nuclear or particle physics.

## LAMMPS

`lammps` drives an installed LAMMPS out of process. See
[EXTERNAL_SOLVER_BACKENDS.md](EXTERNAL_SOLVER_BACKENDS.md#lammps).
`status()` names the exact executable or module, its version and packages,
or how to install it. `check()` lists every refusal keyed by variable,
`native_input()` shows the exact files LAMMPS will read, and `state()` says
whether a run is current, stale or detached and whether it can be applied.
`apply()` writes a run onto its structure as one undoable change and refuses
stale geometry. `lammps.run(cu, "energy")` and
`lammps.run(cu, "relax", ftol_eV_A=1e-3)` build the specification in one step.

```python
lammps.status()
cu = materials.load("copper").bulk(repeat=(4, 4, 4))
spec = lammps.spec(cu, "md", steps=2000, timestep_fs=1.0, ensemble="langevin",
                   temperature_K=600.0, damping_fs=100.0, seed=7,
                   initial_velocities="create", sample_every=20)
lammps.check(spec)["blocking"]
lammps.native_input(spec)["in.lammps"]
run = lammps.run(spec=spec, apply=False)
run["run"].value["command_line"], run["run"].value["inputs"]
lammps.state()
lammps.apply(run["run"].extra["run_id"])
lammps.runs(); lammps.result(quantity="stress"); lammps.array(name="positions")
```

Tasks are `energy`, `relax` (fixed cell, `min_style` `cg`, `fire` or `sd`,
`ftol_eV_A` is the largest force on a mobile atom) and `md` (`ensemble`
`nve`, `langevin` or `nvt`). Every setting carries its unit in its name; the
defaults are in `materia.solvers.lammps.spec.DEFAULTS`. The potential is an
EAM setfl file chosen exactly as for `eam`. A refused specification raises
`ApiError` and a failed or cancelled run raises `LAMMPSRunFailed` (with
`status`, `reason` and the audit record); neither stores anything or touches
the history. Materia never substitutes its own EAM solver when LAMMPS is
missing. Trajectories use the EAM trajectory contract, so the desktop player
replays them.

## First-principles ground state

`dft` runs self-consistent GPAW calculations as versioned experiments. See
[DFT_GROUND_STATE.md](DFT_GROUND_STATE.md).

```python
spec = dft.experiment(structure, xc="PBE", grid_spacing_A=0.18)
dft.check(spec)                         # ok, blocking, warnings, by_field
dft.describe(spec)                      # every variable with unit and explanation
run = dft.ground_state(spec=spec)       # dict of Results; also ground_state(structure, **variables)
run["energy"].value, run["forces"].extra["by_atom_id"]
run["charge_accounting"].value, run["scf_history"].value
dft.array(name="density")               # StoredArray: data, unit, meta
dft.runs(); dft.result(quantity="fermi_level"); dft.spec_of()
dft.state(); dft.is_current()           # current, stale or detached, and why
dft.change(spec, kpoints=[6, 6, 6])     # a new specification
study = dft.convergence_study(structure, parameter="kpoint_density_A",
                              values=[10, 15, 20], tolerance=1e-3)
dft.studies(); dft.evidence()
```

A refused specification raises `ApiError` and stores nothing. A run that does
not converge, fails or is cancelled stores only its record. Nothing here
changes a structure or records an undo point. `dft.energy()` is the earlier
single-point adapter, kept for existing scripts.

## DFT relaxation

`dft.relax` relaxes positions, and for a bulk plane-wave calculation the cell,
with GPAW. See [DFT_RELAXATION.md](DFT_RELAXATION.md). Every ground-state
variable is accepted, plus `mode` (`fixed-cell`, `variable-cell`),
`optimizer` (`BFGS`, `FIRE`), `fmax_eV_A`, `max_steps`, `maxstep_A`,
`symmetry` (`preserve`, `off`) and, for a variable cell, `stress_tol_eV_A3`,
`cell_mask`, `hydrostatic_strain` and `target_pressure_GPa`. Fixed atoms come
from the structure.

```python
spec = dft.relax_spec(structure, fmax_eV_A=0.02, forces_tol_eV_A=0.01)
dft.relax_check(spec)["by_field"]
dft.relax_describe(spec)["relaxation_settings"]
run = dft.relax(spec=spec)
run["relaxation"].value["history"]
run["relaxation"].value["displacement"]["max_A"]
run["relaxation"].value["cell"]["volume_change_percent"]
dft.relax_state()
dft.relax_runs()
dft.relax_array(name="step_positions")
crystal = dft.relax(bulk, mode="variable-cell", cutoff_eV=500.0, kpoints=[4, 4, 4],
                    stress_tol_eV_A3=0.001, target_pressure_GPa=0.0)
```

A refused specification raises `ApiError`; a cancelled, timed-out or failed
run raises `DFTRelaxationFailed` with `status`, `reason` and `audit`. Neither
stores anything, records history or changes the structure. A converged run is
applied to its structure as one undoable change if the structure is
unchanged; `dft.relax_apply(run_id)` applies one kept with `apply=False` and
refuses stale geometry, a detached structure, a not-converged run and a second
application.

## DFT density of states

`dft.dos` computes total DOS and PAW-projector PDOS with GPAW. See
[DFT_DOS.md](DFT_DOS.md).

```python
spec = dft.dos_spec(structure, energy_min_eV=-12.0, energy_max_eV=6.0,
                    energy_step_eV=0.02, width_eV=0.1,
                    dos_kpoints=[6, 6, 6], n_bands=24,
                    projections=[{"element": "Si", "angular": "p"}])
dft.dos_check(spec)["by_field"]
run = dft.dos(spec=spec)
run["dos"].value["checks"]
dft.dos_array(name="dos_total")
dft.dos_array(name="pdos")
dft.dos_runs()
dft.dos_state()
```

`dft.dos_spec(source_run=run_id)` inherits a stored ground state.
`source_kind="relaxation"` inherits a stored relaxation and its final geometry.
The stored source fixes all electronic settings. A failed or refused run stores
nothing and a successful run never changes the structure.

## DFT band structure

`dft.bands` computes the Kohn-Sham band structure along an explicit
reciprocal-space path with GPAW. See
[DFT_BAND_STRUCTURE.md](DFT_BAND_STRUCTURE.md).

```python
spec = dft.bands_spec(structure, xc="PBE", cutoff_eV=300.0, kpoints=[4, 4, 4],
                      occupations="fixed", smearing_eV=0.0, path="GXWKGL",
                      sampling_density_per_invA=10.0, n_bands=8)
spec.path_origin, spec.kpoints_frac, spec.ticks()
dft.bands_check(spec)["by_field"]
run = dft.bands(spec=spec)
run["bands"].value["band_edges"]
dft.bands_array(name="eigenvalues")      # [spin, k-point, band], eV, absolute
dft.bands_array(name="distance")         # 1/A, 2 pi included
dft.bands_runs()
dft.bands_state()                        # current, stale, detached or corrupt
```

The standard path is generated once when the specification is built and stored
as exact fractional coordinates; `dft.bands_change(spec, n_bands=12)` keeps it,
and only `path="standard"` regenerates it. `source_run` and `source_kind`
inherit a stored ground state or relaxation. Refusals raise `ApiError`; failed,
cancelled or unverifiable runs raise `DFTBandsFailed` and store nothing.

## DFT equation of state

`dft.eos` fits a Birch-Murnaghan equation of state to ground states at several
volumes, all with one pinned k-point grid and cutoff. See
[DFT_EQUATION_OF_STATE.md](DFT_EQUATION_OF_STATE.md).

```python
spec = dft.eos_spec(structure, xc="PBE", cutoff_eV=500.0, kpoints=[12, 12, 12],
                    volume_min_scale=0.94, volume_max_scale=1.06, n_points=7)
result = dft.eos(spec=spec)
result.value["V0_A3_per_atom"], result.value["B0_GPa"], result.value["B1"]
dft.eos_array(name="energies")
dft.eos_state()                          # current, applied, stale, detached or corrupt
dft.eos_apply()                          # one undoable change to the equilibrium volume
```

Failed, cancelled or unverified runs raise `DFTEOSFailed` and store nothing.

## DFT LDOS and STM images

`dft.ldos` computes the spatially resolved LDOS in an energy window about the
Fermi level; `dft.stm_image` takes Tersoff-Hamann images from a stored slab
LDOS. See [DFT_LDOS.md](DFT_LDOS.md).

```python
spec = dft.ldos_spec(slab, kpoints=[6, 6, 1], energy_min_eV=-1.0, energy_max_eV=0.0)
results = dft.ldos(spec=spec)
dft.ldos_array(name="ldos").data
dft.stm_image(mode="constant-height", height_A=3.0)["values"]
dft.stm_image(mode="constant-current", isovalue=1e-5)["values"]
dft.ldos_state()
```

Failed, cancelled or unverified runs raise `DFTLDOSFailed` and store nothing.

## Claims

```python
claims.classifications()                # the seven classes and their definitions
claims.record("The 2x2x1 slab energy per atom is converged to 1 meV",
              "computational-prediction", evidence=["dft::<run_id>::energy"])
claims.list()
```

A claim whose evidence does not support its class, or that speaks of a proof,
a theorem or a law, raises `ApiError`.

## Electrostatics

`electrostatics` computes the Coulomb energy, forces, site potentials and site
fields of explicit point charges. See [ELECTROSTATICS.md](ELECTROSTATICS.md).

```python
electrostatics.assign(by_element={"Ga": 1.0, "As": -1.0}, source="unit charges")
electrostatics.assign(formal=True)                 # formal charges as point ions
electrostatics.assign(by_atom_id={1: 0.4, 2: -0.4, ...}, source="...")
run = electrostatics.compute(accuracy=1e-10)       # SolverResult
run["energy"], run["forces"], run["site_potential"], run["site_field"]
electrostatics.status()   # charge model, geometry, blocking reasons
electrostatics.runs()     # stored runs, newest first
electrostatics.result(quantity="site_potential")
electrostatics.clear()
```

Every call takes an optional target (a structure or surface handle) as its
first argument; without one it acts on the active structure. `assign` and
`clear` are undoable. `compute` never changes the structure and stores its
results as `electrostatics::<run_id>::<quantity>`.

## Microscopy

```python
scan = microscope.stm_scan(
    surface,
    bias_volts=0.8,
    current_nA=0.5,
    resolution=(256, 256),
    mode="constant-current",    # or "constant-height"
    tip="W",
    noise="realistic",          # "quiet", "realistic", "noisy"
    seed=0,
)

scan.channel("topography")      # NumPy array
scan.channel_names()
scan.statistics()
scan.filtered("topography", "plane+median")
scan.detect_features()
scan.identify_at(x_A, y_A)
scan.provenance.summary()

spectrum = microscope.stm_spectroscopy(surface, x=5.0, y=5.0, height_A=5.0)
afm = microscope.afm_scan(surface, mode="fm-afm", height_A=4.0)
curve = microscope.force_curve(surface, x=5.0, y=5.0)
```

## Measuring

```python
measure.distance(a, b, surface)
measure.angle(a, b, c, surface)
measure.dihedral(i, j, k, l, surface)
measure.coordination(atom_id, surface)
measure.neighbour_list(cutoff_A=3.0, target=surface)
measure.radial_distribution(r_max_A=8.0, bins=200, target=surface)
measure.strain(reference_surface, surface)
```

## Import and export

```python
io.read("structure.cif")
io.write("surface.xyz", surface)              # xyz, extxyz, cif, poscar, pdb, lammps-data
io.write_image("scan.png", scan, palette="silver")
io.write_image("scan.tif", scan)              # 16-bit, full dynamic range
io.write_csv("atoms.csv", surface.atoms.table())
io.write_cube("density.cube", surface, volumetric_array)
io.formats()
```

## Project and reproducibility

```python
project.save_checkpoint("before-doping", "pristine surface")
project.restore_checkpoint("before-doping")
project.save("experiment.materia")
lab.undo()
lab.redo()
project.provenance_text()
```

## Display

```python
view.display(scan, palette="silver", title="Si(111)")
view.plot(x, y, title="dI/dV", xlabel="bias / V", ylabel="dI/dV")
view.table({"id": ids, "element": symbols})
view.message("done", "info")
```

Headless scripts still work: everything is collected in `view.items` and
returned with the script result.

## Execution modes

| Mode | What it does |
| --- | --- |
| `restricted` (default) | Curated builtins and an import allow-list |
| `trusted` | Full builtins and unrestricted imports, after explicit confirmation |
| `subprocess` | A separate interpreter with a hard timeout, against a project on disk |

Restricted mode blocks the common accidents: `os`, `sys`, `subprocess`,
`socket`, `open`, `eval`, `exec`, `compile`, `input`. **It is a guard rail, not
a security sandbox.** CPython cannot be sandboxed from inside. Treat a script
you are about to run the way you would treat any other program.

```bash
materia run analysis.py --project wafer.materia          # subprocess, isolated
materia run analysis.py --project wafer.materia --trusted
```

## Errors

Errors name the thing that is wrong and what would fix it:

```
ApiError: No electronic-structure model covers Au. Available tight-binding
models: pz-graphene, sp3s*-GaAs, sp3s*-Ge, sp3s*-Si; elements that can be added
as substitutional impurities: Al, As, B, C, Ga, Ge, In, N, O, P, S, Sb, Se, Sn.
```

`UnsupportedRequest` is the `ApiError` subclass raised when the program
declines to answer rather than approximate. `exc.as_dict()` gives the reason,
the literature reference for what was asked, and the models that could produce
it.

A refusal that is part of the physics comes back as an unsupported `Result`,
not an exception:

```python
out = surface.solve(self_consistent=True)
r = out.get("self_consistency")
if r is not None and not r.supported:
    print(r.unsupported_reason)
    print(r.suggested_models)
```
