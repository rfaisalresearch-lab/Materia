"""Point-charge electrostatics: a Madelung check, a slab and an honest refusal.

Run with:
    materia run examples/scripts/09_electrostatics.py
or paste into the embedded Python console.

Charges are never inferred from the material. Each calculation below states
where its charges come from, and every result carries its model, boundary
conditions, convergence check and references.
"""

import math

gaas = materials.load("gallium_arsenide")
cell = gaas.bulk(repeat=(2, 2, 2))
print(f"{len(cell)} atoms, periodic {cell.cell.pbc}")

electrostatics.assign(cell, by_element={"Ga": 1.0, "As": -1.0},
                      source="unit charges, to compare with the zinc-blende Madelung constant")
run = electrostatics.compute(cell)
energy = run["energy"]
print(energy)
print("model:", energy.provenance.model, "|", energy.provenance.boundary_conditions)
print("convergence:", energy.convergence.message)
for name, value in energy.extra["components_eV"].items():
    print(f"   {name:<16s} {value:14.6f} eV")

a = cell.cell.lengths[0] / 2
r_nn = a * math.sqrt(3) / 4
pairs = len(cell) / 2
madelung = -energy.value / pairs * r_nn / 14.3996454784
print(f"Madelung constant from the energy: {madelung:.7f} (published 1.638055)")

potential = electrostatics.result(quantity="site_potential")
print(f"site potential range {potential.value.min():.4f} to {potential.value.max():.4f} V")

slab = gaas.create_surface(size=(3, 3, 4), vacuum_angstrom=15, orientation=(1, 1, 0))
electrostatics.assign(slab, by_element={"Ga": 1.0, "As": -1.0},
                      source="unit charges on a nonpolar (110) slab")
slab_run = electrostatics.compute(slab)
slab_energy = slab_run["energy"]
print(slab_energy)
print("slab method:", slab_energy.provenance.parameters["method"])
print(f"internal vacuum gap {slab_energy.provenance.parameters['internal_vacuum_gap_A']:.1f} A")

electrostatics.assign(slab, by_element={"Ga": 1.0, "As": -0.5},
                      source="deliberately unbalanced charges")
refused = electrostatics.compute(slab)["energy"]
print("charged slab supported:", refused.supported)
print("reason:", refused.unsupported_reason)
