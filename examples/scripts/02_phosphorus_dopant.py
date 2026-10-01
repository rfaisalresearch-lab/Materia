"""Substitute a phosphorus donor and measure what changes.

Demonstrates the before-and-after comparison the acceptance scenario requires:
local geometry from the classical potential, local electronic structure from
tight binding, and the provenance of both.
"""

import numpy as np

silicon = materials.load("silicon", orientation="111")
surface = silicon.create_surface(size=(4, 4, 4), vacuum_angstrom=15)
before = surface.copy()

site = surface.nearest_site((10.0, 10.0, 25.0))
print("substituting atom", site, "at",
      [round(v, 3) for v in surface.atoms[site].position])
record = surface.substitute(site, "P")
print("record:", record)

project.save_checkpoint("phosphorus-unrelaxed", "P in place, geometry untouched")

result = surface.relax(model="recommended", fmax=0.02, steps=300)
print("relaxation:", result.convergence.message)
print("model:", result.provenance.model)

print("\nlocal geometry")
for neighbour in surface.atoms[site].neighbors:
    after = measure.distance(site, neighbour.id, surface)
    original = measure.distance(site, neighbour.id, before)
    print(f"   P-{neighbour.symbol}#{neighbour.id}: "
          f"{original:.4f} -> {after:.4f} A  ({(after - original) * 100:+.2f} pm)")

electronic = surface.solve()
print("\nelectronic structure")
print("   solver      :", electronic.solver)
print("   Fermi level :", round(electronic["fermi_level"].value, 4), "eV")

charges = electronic["partial_charges"].value
index = list(surface.structure.ids).index(site)
print(f"   partial charge on P  {charges[index]:+.4f} e")
print(f"   mean on Si           {np.delete(charges, index).mean():+.4f} e")

for message in electronic.log:
    print("   note:", message)

ldos = electronic["local_dos"].value
column = list(ldos["atom_ids"]).index(site)
view.plot(ldos["energy_eV"], ldos["ldos"][:, column],
          title=f"Local density of states at the P donor (atom {site})",
          xlabel="energy / eV", ylabel="states / eV")

print("\nThe shift of the donor level is not quantitative:")
for approximation in electronic["eigenvalues"].provenance.approximations:
    if "Harrison" in approximation or "binding" in approximation:
        print("   ", approximation)
