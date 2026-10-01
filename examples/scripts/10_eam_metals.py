"""Embedded-atom metals: cohesive energies, a vacancy, and an honest refusal.

Run with:
    materia run examples/scripts/10_eam_metals.py
or paste into the embedded Python console.

The shipped potentials are the Zhou, Johnson and Wadley (2004) EAM files for
Cu, Au and W, retabulated by NIST and distributed by OpenKIM with a public
domain dedication. Every result records the file's checksum and citation.
"""

for name in ("copper", "gold", "tungsten"):
    cell = materials.load(name).bulk(repeat=(3, 3, 3))
    run = eam.run(cell, "energy")
    energy = run["energy"]
    identity = energy.provenance.parameters["potential"]
    print(f"{name:9s} {energy.value / len(cell):+.5f} eV/atom  "
          f"{identity['id']}  sha256 {identity['sha256'][:12]}")

base = materials.load("tungsten").bulk(repeat=(2, 2, 2), activate=False).structure
tabulated = base.cell.lengths[0] / 2
scan = []
for a in (3.15, 3.16, 3.1648, 3.17, 3.18):
    trial = base.copy()
    trial.cell = type(base.cell)(base.cell.matrix * a / tabulated, base.cell.pbc)
    trial.positions = base.positions * a / tabulated
    energy = eam.potential("W-Zhou04").energy(trial) / len(trial)
    scan.append((energy, a))
print(f"W bcc lattice constant from a scan: {min(scan)[1]} A "
      "(NIST LAMMPS reference 3.16485 A)")

copper = materials.load("copper").bulk(repeat=(4, 4, 4))
perfect = eam.run(copper, "energy")["energy"].value
n = len(copper)
copper.create_vacancy(int(copper.structure.ids[0]))
relaxed = eam.run(copper, "relax", fmax_eV_A=1e-4)["energy"]
print(relaxed.convergence.message)
print(f"Cu vacancy formation energy: {relaxed.value - (n - 1) / n * perfect:.3f} eV "
      "(NIST LAMMPS reference 1.279 eV, experiment about 1.28 eV)")

mixed = materials.load("gold").bulk(repeat=(2, 2, 2))
mixed.substitute(int(mixed.structure.ids[0]), "Cu")
try:
    eam.run(mixed, "energy")
except ApiError as exc:
    print("refused:", exc)
