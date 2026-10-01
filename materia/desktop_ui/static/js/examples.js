/** Runnable laboratory examples and their feature coverage. @module examples */

export const FEATURE_AREAS = [
  ['materials', 'Material library'],
  ['bulk', 'Bulk crystal construction'],
  ['surface', 'Surface construction'],
  ['viewport-3d', 'Interactive 3D viewport'],
  ['wafer', 'Procedural wafer'],
  ['roi', 'Atomistic region extraction'],
  ['defects', 'Defects and dopants'],
  ['reconstruction', 'Surface reconstruction'],
  ['selection', 'Atom selection'],
  ['measurements', 'Geometry measurements'],
  ['strain', 'Applied strain'],
  ['history', 'Undo, redo and checkpoints'],
  ['relaxation', 'Classical relaxation'],
  ['dynamics', 'Molecular dynamics'],
  ['electrostatics', 'Long-range electrostatics'],
  ['eam', 'Embedded-atom metals'],
  ['rigid-ion', 'Rigid-ion materials'],
  ['collision-setup', 'Collision setup'],
  ['collision-playback', 'Trajectory playback'],
  ['fragments', 'Fragment analysis'],
  ['stm', 'STM imaging'],
  ['afm', 'AFM imaging'],
  ['spectroscopy', 'Probe spectroscopy'],
  ['noise', 'Instrument noise'],
  ['tight-binding', 'Tight binding'],
  ['bands', 'Band structure'],
  ['quantum-states', 'Quantum energy states'],
  ['orbital-amplitudes', 'Orbital amplitudes'],
  ['electron-density', 'Electron and spin density'],
  ['occupations', 'Occupations and Fermi level'],
  ['batch', 'Batch experiments'],
  ['dft-ground', 'DFT ground state'],
  ['dft-relax', 'DFT relaxation'],
  ['dft-dos', 'DFT density of states'],
  ['dft-bands', 'DFT band structure'],
  ['dft-eos', 'DFT equation of state'],
  ['dft-ldos', 'DFT LDOS and STM'],
  ['lammps', 'LAMMPS backend'],
  ['import-export', 'Scientific import and export'],
  ['python', 'Python laboratory'],
  ['provenance', 'Provenance and limitations'],
  ['refusals', 'Unsupported request refusals'],
].map(([id, label]) => ({ id, label }));

function define(id, label, category, description, runtime, requires, view, features, code) {
  return { id, label, category, description, runtime, requires, view, features, code };
}

export const EXAMPLES = [
  define('model-silicon', '3D silicon diamond crystal', '3D models',
    'Build a repeated diamond silicon crystal and open it in the interactive atomic viewport.',
    'instant', 'Built in', 'atoms', ['materials', 'bulk', 'viewport-3d', 'python'], String.raw`
si = materials.load("silicon").bulk(repeat=(3, 3, 3))
print(si, len(si), "atoms")
print("cell lengths:", si.cell.lengths)
view.display(si, title="Silicon diamond crystal")
`),
  define('model-graphene', '3D graphene sheet', '3D models',
    'Build a wide hexagonal graphene sheet with vacuum above and below it.',
    'instant', 'Built in', 'atoms', ['materials', 'bulk', 'viewport-3d'], String.raw`
graphene = materials.load("graphene").bulk(repeat=(7, 7, 1))
print(graphene, len(graphene), "atoms")
view.display(graphene, title="Graphene sheet")
`),
  define('model-gold', '3D gold FCC crystal', '3D models',
    'Build a face-centred cubic gold crystal ready for EAM calculations.',
    'instant', 'Built in', 'atoms', ['materials', 'bulk', 'viewport-3d', 'eam'], String.raw`
gold = materials.load("gold").bulk(repeat=(4, 4, 4))
print(gold, len(gold), "atoms")
view.display(gold, title="Gold FCC crystal")
`),
  define('model-gaas', '3D gallium arsenide', '3D models',
    'Build the two-species zinc-blende lattice and inspect the Ga and As sublattices.',
    'instant', 'Built in', 'atoms', ['materials', 'bulk', 'viewport-3d'], String.raw`
gaas = materials.load("gallium_arsenide").bulk(repeat=(3, 3, 3))
print(gaas, len(gaas), "atoms")
print("formula:", gaas.structure.formula())
view.display(gaas, title="Gallium arsenide zinc blende")
`),
  define('model-mos2', '3D molybdenum disulfide', '3D models',
    'Build a layered MoS2 model that makes the S-Mo-S sheets visible in 3D.',
    'instant', 'Built in', 'atoms', ['materials', 'bulk', 'viewport-3d'], String.raw`
mos2 = materials.load("molybdenum_disulfide").bulk(repeat=(5, 5, 2))
print(mos2, len(mos2), "atoms")
view.display(mos2, title="Layered molybdenum disulfide")
`),
  define('model-quartz', '3D alpha-quartz silica', '3D models',
    'Build crystalline SiO2 with its tetrahedral network and rigid-ion model coverage.',
    'instant', 'Built in', 'atoms', ['materials', 'bulk', 'viewport-3d', 'rigid-ion'], String.raw`
quartz = materials.load("silicon_dioxide").bulk(repeat=(2, 2, 2))
print(quartz, len(quartz), "atoms")
view.display(quartz, title="Alpha-quartz silica")
`),
  define('wafer-region', 'Wafer to atomistic region', 'Build and edit',
    'Create a reproducible procedural wafer and extract one explicit atomistic region from it.',
    'seconds', 'Built in', 'atoms', ['wafer', 'roi', 'surface', 'viewport-3d', 'provenance'], String.raw`
wafer = lab.create_wafer(material="silicon", diameter_mm=300.0,
                         orientation=(1, 1, 1), seed=20260920)
region = lab.extract_region(x_mm=12.5, y_mm=-8.0, size_nm=(2.5, 2.5),
                            depth_layers=4, include_defects=True)
print("wafer:", wafer.spec.diameter_mm, "mm")
print("region:", len(region), "atoms")
view.display(region, title="Wafer region at 12.5, -8.0 mm")
`),
  define('silicon-stm', 'Silicon surface, relax and STM', 'Microscopy',
    'Build a Si(111) slab, relax it, simulate a constant-current STM image and display the scan.',
    'seconds', 'Built in', 'probe', ['surface', 'relaxation', 'stm', 'noise', 'tight-binding', 'provenance'], String.raw`
import numpy as np
si = materials.load("silicon", orientation="111")
surface = si.create_surface(size=(4, 4, 4), vacuum_angstrom=15)
relaxed = surface.relax(model="recommended", fmax=0.02)
print(relaxed.convergence.message)
scan = microscope.stm_scan(surface, bias_volts=1.0, current_nA=0.5,
                           resolution=(192, 192), noise="realistic", seed=7)
print("corrugation:", float(np.ptp(scan.channel("topography"))), "A")
view.display(scan, palette="silver", title="Si(111) constant current")
`),
  define('dopant', 'Phosphorus dopant', 'Build and edit',
    'Substitute a phosphorus donor, relax the structure and compare its local bonds.',
    'seconds', 'Built in', 'atoms', ['surface', 'defects', 'relaxation', 'measurements', 'history'], String.raw`
si = materials.load("silicon", orientation="111")
surface = si.create_surface(size=(4, 4, 4), vacuum_angstrom=15)
before = surface.copy()
site = surface.nearest_site((10.0, 10.0, 25.0))
surface.substitute(site, "P")
surface.relax(model="recommended", fmax=0.02)
project.save_checkpoint("phosphorus-relaxed")
for neighbour in surface.atoms[site].neighbors:
    old = measure.distance(site, neighbour.id, before)
    new = measure.distance(site, neighbour.id, surface)
    print(neighbour.symbol, round(old, 4), "to", round(new, 4), "A")
view.display(surface, title="Phosphorus donor in silicon")
`),
  define('vacancy', 'Surface vacancy', 'Build and edit',
    'Remove one top-layer silicon atom, relax the neighbours and inspect the defect in 3D.',
    'seconds', 'Built in', 'atoms', ['surface', 'defects', 'relaxation', 'viewport-3d', 'history'], String.raw`
si = materials.load("silicon", orientation="111")
surface = si.create_surface(size=(4, 4, 4), vacuum_angstrom=15)
top = max(atom.position[2] for atom in surface.atoms)
site = surface.nearest_site((10.0, 10.0, top))
surface.create_vacancy(site)
result = surface.relax(model="recommended", fmax=0.03)
print(result.convergence.message)
view.display(surface, title="Relaxed Si(111) vacancy")
`),
  define('reconstruction', 'Si(100) 2x1 reconstruction', 'Build and edit',
    'Generate the supported dimer reconstruction and compare its geometry with cited reference data.',
    'seconds', 'Built in', 'atoms', ['surface', 'reconstruction', 'relaxation', 'measurements', 'provenance'], String.raw`
si = materials.load("silicon")
surface = si.create_surface(size=(4, 2, 8), orientation=(1, 0, 0),
                            vacuum_angstrom=12, fix_bottom_layers=4,
                            reconstruction="2x1-dimer")
record = surface.reconstruction()
print(record["n_dimers"], "dimers")
print("status:", record["geometry_status"])
print("bond:", round(record["measured"]["bond_length_A"], 4), "A")
print("comparable:", surface.compare_reconstruction()["comparable"])
view.display(surface, title="Si(100) 2x1 dimers")
`),
  define('measure-selection', 'Selection and geometry measurements', 'Build and edit',
    'Select silicon atoms, inspect neighbours and report distance, angle and coordination values.',
    'instant', 'Built in', 'atoms', ['selection', 'measurements', 'viewport-3d'], String.raw`
cell = materials.load("silicon").bulk(repeat=(2, 2, 2))
ids = [atom.id for atom in cell.atoms]
project.selection = Selection(ids[:4], "first four sites")
print("distance:", measure.distance(ids[0], ids[1], cell), "A")
print("angle:", measure.angle(ids[1], ids[0], ids[2], cell), "degrees")
print("coordination:", measure.coordination(ids[0], cell))
view.display(cell, title="Selected silicon sites")
`),
  define('strain-history', 'Strain, undo and redo', 'Build and edit',
    'Apply anisotropic strain and prove that project history restores and reapplies it.',
    'instant', 'Built in', 'atoms', ['strain', 'history', 'measurements', 'viewport-3d'], String.raw`
crystal = materials.load("silicon").bulk(repeat=(2, 2, 2))
before = crystal.cell.lengths.copy()
strained = build["strain"](crystal.structure,
                           [[0.02, 0.0, 0.0], [0.0, -0.01, 0.0], [0.0, 0.0, 0.0]])
project.set_structure(strained, "Apply example strain", "example.strain",
                      {"xx": 0.02, "yy": -0.01})
print("before:", before)
print("strained:", crystal.cell.lengths)
print("undo:", lab.undo())
print("redo:", lab.redo())
view.display(crystal, title="Anisotropically strained silicon")
`),
  define('classical-md', 'Classical molecular dynamics', 'Classical physics',
    'Run finite-temperature silicon dynamics and inspect the stored trajectory record.',
    'seconds', 'Built in', 'atoms', ['dynamics', 'tight-binding', 'provenance'], String.raw`
silicon = materials.load("silicon").bulk(repeat=(2, 2, 2))
run = silicon.dynamics(model="recommended", steps=300, dt_fs=0.5,
                       temperature_K=300.0, thermostat="langevin",
                       friction_per_fs=0.01, seed=4, sample_every=3)
print(run.convergence.message)
print("final temperature:", run.extra.get("temperature_K"), "K")
view.display(silicon, title="Silicon after molecular dynamics")
`),
  define('electrostatics', 'GaAs electrostatics', 'Classical physics',
    'Assign an explicit ionic charge model and evaluate periodic long-range Coulomb energy.',
    'seconds', 'Built in', 'atoms', ['electrostatics', 'provenance', 'materials'], String.raw`
gaas = materials.load("gallium_arsenide").bulk(repeat=(2, 2, 2))
electrostatics.assign(gaas, by_element={"Ga": 1.0, "As": -1.0},
                      source="unit charges for a Madelung check")
report = electrostatics.status(gaas)
print("ready:", report["ready"], report["blocking"])
run = electrostatics.compute(gaas)
print(run["energy"])
view.display(gaas, title="Charged GaAs lattice")
`),
  define('eam-metal', 'Copper with an EAM potential', 'Classical physics',
    'Evaluate and relax copper with a checksummed many-body embedded-atom potential.',
    'seconds', 'Built in', 'atoms', ['eam', 'relaxation', 'provenance', 'materials'], String.raw`
copper = materials.load("copper").bulk(repeat=(3, 3, 3))
run = eam.run(copper, "relax", potential="Cu-Zhou04",
              fmax_eV_A=0.01, max_steps=400)
print(run["energy"].value, "eV")
print(run["energy"].convergence.message)
print(run["energy"].provenance.summary())
view.display(copper, title="EAM-relaxed copper")
`),
  define('rigid-ion-silica', 'Rigid-ion silica', 'Classical physics',
    'Relax alpha-quartz with BKS silica and inspect the short-range and Ewald terms.',
    'seconds', 'Built in', 'atoms', ['rigid-ion', 'electrostatics', 'relaxation', 'provenance'], String.raw`
quartz = materials.load("silicon_dioxide").bulk()
result = quartz.relax(model="recommended", fmax=1e-3, steps=2000)
energy = project.results["relax::energy"]
checks = energy.extra["potential_checks"]
print(result.convergence.message)
print("short range:", checks["components_eV"]["short_range"], "eV")
print("Coulomb:", checks["components_eV"]["coulomb"], "eV")
print(checks["ewald_message"])
view.display(quartz, title="BKS-relaxed alpha-quartz")
`),
  define('two-atom-head-on', 'Two copper atoms, head-on', 'Collisions',
    'Launch exactly one copper atom at another, save the sampled frames and identify final fragments.',
    'seconds', 'Built in', 'atoms', ['collision-setup', 'collision-playback', 'fragments', 'eam', 'dynamics', 'viewport-3d'], String.raw`
import numpy as np
from materia.core_model import Cell, Structure
projectile = Structure([29], np.array([[0.0, 0.0, 0.0]]), Cell.none())
target = Structure([29], np.array([[0.0, 0.0, 0.0]]), Cell.none())
impact = collisions.prepare(projectile, target, relative_speed_A_fs=0.04,
                            impact_parameter_A=0.0, gap_A=4.0, padding_A=8.0)
run = collisions.run(impact, potential="Cu-Zhou04", steps=800,
                     dt_fs=0.1, sample_every=4)
print("atoms:", len(impact))
print("frames:", run["trajectory"].value["frames"])
print("final fragments:", collisions.fragments()["fragments"])
print(impact.info["collision"]["warning"])
`),
  define('two-atom-glancing', 'Two copper atoms, glancing impact', 'Collisions',
    'Repeat the exact two-atom experiment with a transverse offset to compare scattering geometry.',
    'seconds', 'Built in', 'atoms', ['collision-setup', 'collision-playback', 'fragments', 'eam', 'dynamics'], String.raw`
import numpy as np
from materia.core_model import Cell, Structure
projectile = Structure([29], np.array([[0.0, 0.0, 0.0]]), Cell.none())
target = Structure([29], np.array([[0.0, 0.0, 0.0]]), Cell.none())
impact = collisions.prepare(projectile, target, relative_speed_A_fs=0.06,
                            impact_parameter_A=2.2, gap_A=4.0, padding_A=8.0)
run = collisions.run(impact, potential="Cu-Zhou04", steps=700,
                     dt_fs=0.1, sample_every=4)
print("centre-of-mass energy:", impact.info["collision"]["centre_of_mass_energy_eV"], "eV")
print("frames:", run["trajectory"].value["frames"])
print("final fragments:", collisions.fragments()["fragments"])
`),
  define('cluster-collision', 'Copper clusters collide', 'Collisions',
    'Collide two finite copper clusters, replay the saved path and inspect connected fragments.',
    'seconds', 'Built in', 'atoms', ['collision-setup', 'collision-playback', 'fragments', 'eam', 'dynamics'], String.raw`
projectile = materials.load("copper").bulk(repeat=(1, 1, 1), activate=False)
target = materials.load("copper").bulk(repeat=(1, 1, 1), activate=False)
impact = collisions.prepare(projectile, target, relative_speed_A_fs=0.06,
                            impact_parameter_A=1.0, gap_A=3.0, padding_A=8.0)
run = collisions.run(impact, potential="Cu-Zhou04", steps=600,
                     dt_fs=0.1, sample_every=3)
print("run:", run["energy"].extra["run_id"])
print("frames:", run["trajectory"].value["frames"])
print("fragments:", collisions.fragments()["fragments"])
`),
  define('afm', 'FM-AFM image and force curve', 'Microscopy',
    'Generate a frequency-modulation AFM map and a height-dependent force curve at one site.',
    'seconds', 'Built in', 'probe', ['afm', 'spectroscopy', 'noise', 'surface', 'provenance'], String.raw`
surface = materials.load("silicon", orientation="111").create_surface(
    size=(3, 3, 3), vacuum_angstrom=14)
scan = microscope.afm_scan(surface, mode="fm-afm", height_A=4.0,
                           resolution=(160, 160), noise="quiet", seed=2)
curve = microscope.force_curve(surface, x=4.0, y=4.0)
print("force curve points:", len(curve.value["height_above_surface_A"]))
print(curve.provenance.summary())
view.display(scan, palette="silver", title="Si(111) FM-AFM")
`),
  define('spectroscopy', 'Simulated dI/dV spectrum', 'Microscopy',
    'Evaluate a point tunnelling spectrum and plot differential conductance against bias.',
    'seconds', 'Built in', 'atoms', ['stm', 'spectroscopy', 'tight-binding', 'surface', 'provenance'], String.raw`
surface = materials.load("silicon", orientation="111").create_surface(
    size=(3, 3, 4), vacuum_angstrom=14)
sts = microscope.stm_spectroscopy(surface, x=5.0, y=5.0, height_A=5.0)
voltage = sts.value["bias_V"]
conductance = sts.value["dIdV"]
view.plot(voltage, conductance, title="Simulated dI/dV",
          xlabel="bias / V", ylabel="dI/dV / arbitrary unit")
print(sts.provenance.summary())
`),
  define('tight-binding-bands', 'Tight-binding silicon bands', 'Electronic structure',
    'Compute a fast semi-empirical silicon band path and plot the top valence band.',
    'seconds', 'Built in', 'atoms', ['tight-binding', 'bands', 'materials', 'provenance'], String.raw`
from materia.structure_builder.lattice import bulk
from materia.structure_builder.primitive import primitive_structure
si = materials.load("silicon")
cell = primitive_structure(bulk(si.definition))
project.add_structure(cell)
out = lab.band_structure(cell, path=[(0.5, 0.5, 0.5), (0, 0, 0), (0.5, 0, 0.5)],
                         labels=["L", "G", "X"], n_per_segment=50)
bands_data = out["band_structure"].value
print("gap:", round(out["band_gap"].value, 4), "eV")
view.plot(bands_data["k_coord"], [row[3] for row in bands_data["bands_eV"]],
          title="Silicon valence band maximum", xlabel="k", ylabel="E / eV")
`),
  define('quantum-state-explorer', 'Quantum state explorer', 'Quantum',
    'Resolve silicon quantum energy levels, the Fermi level, orbital amplitudes and the density of states with the built-in tight-binding model.',
    'seconds', 'Built in', 'atoms', ['tight-binding', 'quantum-states', 'orbital-amplitudes', 'occupations', 'spectroscopy', 'provenance'], String.raw`
import numpy as np
surface = materials.load("silicon").create_surface(
    size=(2, 2, 2), vacuum_angstrom=12)
states = surface.solve(model="recommended")
energies = states["eigenvalues"].value
vectors = states["eigenvectors"].value
fermi = states["fermi_level"].value
homo = int(np.where(energies <= fermi)[0][-1])
orbitals_per_atom = vectors.shape[0] // len(surface)
weights = np.abs(vectors[:, homo]) ** 2
atom_weights = weights.reshape(len(surface), orbitals_per_atom).sum(axis=1)
ranked = np.argsort(atom_weights)[::-1][:6]
print("quantum states:", len(energies))
print("Fermi level:", round(fermi, 6), "eV")
print("HOMO-LUMO gap:", round(states["hl_gap"].value, 6), "eV")
print("largest HOMO atom weights:")
for index in ranked:
    print(surface.atoms[int(index)].id, round(float(atom_weights[index]), 6))
dos = states["total_dos"].value
view.display(surface, title="Silicon quantum-state model")
view.plot(dos["energy_eV"], dos["dos"], title="Quantum density of states",
          xlabel="energy / eV", ylabel="states / eV")
`),
  define('quantum-density-dft', 'DFT electron-density observables', 'Quantum',
    'Set up a self-consistent quantum calculation that returns electron density, spin density, potential, eigenvalues, occupations and Fermi level.',
    'minutes', 'GPAW and PAW data', 'atoms', ['dft-ground', 'quantum-states', 'electron-density', 'occupations', 'provenance'], String.raw`
import numpy as np
from materia.core_model import Cell, Structure
hydrogen = Structure([1], np.array([[4.0, 4.0, 4.0]]),
                     Cell.cubic(8.0, pbc=(False, False, False)))
project.add_structure(hydrogen)
if not dft.available():
    print("GPAW unavailable:", dft.status()["blocking_reason"])
else:
    spec = dft.experiment(
        hydrogen, xc="PBE", grid_spacing_A=0.22, spin_polarized=True,
        initial_magnetic_moments_muB=[1.0],
        observables=["energy", "density", "spin_density",
                     "electrostatic_potential", "eigenvalues",
                     "occupations", "fermi_level", "magnetic_moment"])
    report = dft.check(spec)
    print("checks:", report["blocking"])
    if report["ok"]:
        run = dft.ground_state(spec=spec)
        print("energy:", run["energy"].value, "eV")
        print("Fermi level:", run["fermi_level"].value, "eV")
        print("magnetic moment:", run["magnetic_moment"].value, "muB")
        print("density shape:", run["density"].value["shape"])
        print("spin-density shape:", run["spin_density"].value["shape"])
`),
  define('batch-stm', 'Batch STM bias series', 'Microscopy',
    'Run the same surface at four biases and compare corrugation in one reproducible batch.',
    'seconds', 'Built in', 'probe', ['batch', 'stm', 'noise', 'surface', 'import-export'], String.raw`
import numpy as np
surface = materials.load("silicon", orientation="111").create_surface(
    size=(4, 4, 4), vacuum_angstrom=15)
last = None
for bias in (-1.5, -0.8, 0.8, 1.5):
    last = microscope.stm_scan(surface, bias_volts=bias,
                               resolution=(128, 128), seed=7)
    print(bias, "V:", float(np.ptp(last.channel("topography"))), "A")
view.display(last, palette="silver", title="Final STM bias in batch")
`),
  define('dft-ground', 'DFT ground state', 'DFT',
    'Run a frozen GPAW specification for H2, including electron accounting and provenance.',
    'minutes', 'GPAW and PAW data', 'atoms', ['dft-ground', 'provenance', 'python'], String.raw`
import numpy as np
from materia.core_model import Cell, Structure
h2 = Structure([1, 1], np.array([[3.0, 3.0, 2.63], [3.0, 3.0, 3.37]]),
               Cell.cubic(6.0, pbc=(False, False, False)))
project.add_structure(h2)
if not dft.available():
    print("GPAW unavailable:", dft.status()["blocking_reason"])
else:
    spec = dft.experiment(h2, xc="PBE", grid_spacing_A=0.22)
    print("checks:", dft.check(spec)["blocking"])
    run = dft.ground_state(spec=spec)
    print("energy:", run["energy"].value, "eV")
    print(run["charge_accounting"].value["message"])
`),
  define('dft-relax', 'DFT geometry relaxation', 'DFT',
    'Relax an H2 bond with GPAW and retain the optimizer history as an undoable result.',
    'minutes', 'GPAW and PAW data', 'atoms', ['dft-relax', 'history', 'provenance'], String.raw`
import numpy as np
from materia.core_model import Cell, Structure
h2 = Structure([1, 1], np.array([[3.0, 3.0, 2.60], [3.0, 3.0, 3.40]]),
               Cell.cubic(6.0, pbc=(False, False, False)))
project.add_structure(h2)
if not dft.available():
    print("GPAW unavailable:", dft.status()["blocking_reason"])
else:
    spec = dft.relax_spec(h2, xc="PBE", grid_spacing_A=0.22,
                          fmax_eV_A=0.03, forces_tol_eV_A=0.02)
    run = dft.relax(spec=spec)
    print(run["relaxation"].convergence.message)
    print(run["relaxation"].value)
`),
  define('dft-dos', 'DFT density of states', 'DFT',
    'Compute and store a verified GPAW density of states for crystalline silicon.',
    'minutes', 'GPAW and PAW data', 'atoms', ['dft-dos', 'dft-ground', 'provenance'], String.raw`
silicon = materials.load("silicon").bulk(repeat=(1, 1, 1))
if not dft.available():
    print("GPAW unavailable:", dft.status()["blocking_reason"])
else:
    spec = dft.dos_spec(silicon, xc="PBE", cutoff_eV=300.0,
                        kpoints=[2, 2, 2], n_bands=20,
                        energy_min_eV=-12.0, energy_max_eV=6.0)
    print("checks:", dft.dos_check(spec)["blocking"])
    print(dft.dos(spec=spec)["dos"].value["checks"])
`),
  define('dft-bands', 'DFT band structure', 'DFT',
    'Calculate explicit high-symmetry Kohn-Sham bands and retain the sampled path.',
    'minutes', 'GPAW and PAW data', 'atoms', ['dft-bands', 'dft-ground', 'bands', 'provenance'], String.raw`
silicon = materials.load("silicon").bulk(repeat=(1, 1, 1))
if not dft.available():
    print("GPAW unavailable:", dft.status()["blocking_reason"])
else:
    spec = dft.bands_spec(silicon, xc="PBE", cutoff_eV=300.0,
                          kpoints=[2, 2, 2], path="GXWKGL",
                          sampling_density_per_invA=6.0, n_bands=20)
    print("checks:", dft.bands_check(spec)["blocking"])
    print(dft.bands(spec=spec)["bands"].value["band_edges"])
`),
  define('dft-eos', 'DFT equation of state', 'DFT',
    'Sample several silicon volumes and fit a verified Birch-Murnaghan equation of state.',
    'long', 'GPAW and PAW data', 'atoms', ['dft-eos', 'dft-ground', 'provenance'], String.raw`
silicon = materials.load("silicon").bulk(repeat=(1, 1, 1))
if not dft.available():
    print("GPAW unavailable:", dft.status()["blocking_reason"])
else:
    spec = dft.eos_spec(silicon, xc="PBE", cutoff_eV=300.0,
                        kpoints=[3, 3, 3], volume_min_scale=0.96,
                        volume_max_scale=1.06, n_points=5)
    print("checks:", dft.eos_check(spec)["blocking"])
    print(dft.eos(spec=spec).value)
`),
  define('dft-ldos', 'DFT LDOS and STM image', 'DFT',
    'Create a spatial LDOS map with GPAW and derive a Tersoff-Hamann constant-height image.',
    'long', 'GPAW and PAW data', 'atoms', ['dft-ldos', 'dft-ground', 'stm', 'provenance'], String.raw`
import math
import numpy as np
from materia.core_model import Cell, Structure
a = 2.46
graphene = Structure([6, 6],
    np.array([[0.0, 0.0, 5.0], [0.0, a / math.sqrt(3), 5.0]]),
    Cell(np.array([[a, 0.0, 0.0], [-a / 2, a * math.sqrt(3) / 2, 0.0],
                   [0.0, 0.0, 16.0]]), (True, True, False)))
project.add_structure(graphene)
if not dft.available():
    print("GPAW unavailable:", dft.status()["blocking_reason"])
else:
    spec = dft.ldos_spec(graphene, xc="PBE", grid_spacing_A=0.22,
                         kpoints=[3, 3, 1], energy_min_eV=-1.0,
                         energy_max_eV=0.0, n_bands=16)
    print("checks:", dft.ldos_check(spec)["blocking"])
    run = dft.ldos(spec=spec)
    image = dft.stm_image(mode="constant-height", height_A=3.0)
    print(run["ldos"].value)
    print("image shape:", image["values"].shape)
`),
  define('lammps', 'LAMMPS checked backend', 'External solvers',
    'Inspect the local LAMMPS environment, freeze a run specification and report refusals before execution.',
    'external', 'LAMMPS for execution', 'atoms', ['lammps', 'provenance', 'refusals'], String.raw`
copper = materials.load("copper").bulk(repeat=(2, 2, 2))
print("environment:", lammps.status())
spec = lammps.spec(copper, task="energy", potential="Cu-Zhou04")
report = lammps.check(spec)
print("ready:", report["ok"])
print("blocking:", report["blocking"])
if report["ok"]:
    print(lammps.run(spec=spec)["energy"])
`),
  define('export-checkpoint', 'Checkpoint and scientific export', 'Workflow',
    'Create a checkpoint and export one structure, an atom table and a verified NumPy archive.',
    'instant', 'Built in', 'atoms', ['import-export', 'history', 'python', 'provenance'], String.raw`
structure = materials.load("silicon").bulk(repeat=(2, 2, 2))
project.save_checkpoint("silicon-built", "before export")
print(io.write("materia_example.xyz", structure))
table = structure.atoms.table()
print(io.write_csv("materia_example_atoms.csv", table, "Materia atom table"))
print(io.write_npz("materia_example_arrays.npz"))
print("formats:", io.formats())
`),
  define('unsupported', 'Scientific refusal paths', 'Workflow',
    'Ask for unavailable physics and inspect the explicit reason and suggested alternatives.',
    'instant', 'Built in', 'atoms', ['refusals', 'provenance', 'reconstruction', 'python'], String.raw`
silicon = materials.load("silicon", orientation="111")
try:
    silicon.create_surface(orientation=(1, 1, 1), reconstruction="7x7-DAS")
except UnsupportedRequest as error:
    record = error.as_dict()
    print("refused:", record["reason"])
    print("citation:", record.get("citation"))
surface = silicon.create_surface(size=(2, 2, 3))
result = surface.solve(self_consistent=True)
item = result.get("self_consistency")
if item is not None and not item.supported:
    print("unsupported:", item.unsupported_reason)
    print("alternatives:", item.suggested_models)
`),
];

export function exampleCoverage() {
  const covered = new Set(EXAMPLES.flatMap((example) => example.features));
  return {
    total: FEATURE_AREAS.length,
    covered: FEATURE_AREAS.filter((feature) => covered.has(feature.id)).length,
    missing: FEATURE_AREAS.filter((feature) => !covered.has(feature.id)),
  };
}
