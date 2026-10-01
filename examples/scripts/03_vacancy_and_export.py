"""Create a surface vacancy, image it, and export everything.

Produces a structure file, a scan image and a reproducible project.
"""

import numpy as np

silicon = materials.load("silicon", orientation="111")
surface = silicon.create_surface(size=(4, 4, 4), vacuum_angstrom=15)

top = max(atom.position[2] for atom in surface.atoms)
victim = surface.nearest_site((10.0, 10.0, top))
print("removing atom", victim)
record = surface.create_vacancy(victim)
print("neighbours now marked:", record["neighbour_ids"])

surface.relax(model="recommended", fmax=0.03, steps=300)

scan = microscope.stm_scan(surface, bias_volts=1.0, resolution=(224, 224), seed=5)
view.display(scan, palette="silver", title="Si(111) with a surface vacancy")

x, y = record["position_A"][:2]
identity = scan.identify_at(x, y)
print("\nat the vacancy site the probe reports:")
print("   kind       :", identity["kind"])
print("   nearest    :", identity["nearest_element"], "#", identity["nearest_atom_id"])
print("   confidence :", round(identity["confidence"], 2))
print("   caveat     :", identity["caveat"])

io.write("si111_vacancy.xyz", surface)
io.write("si111_vacancy.cif", surface)
io.write_image("si111_vacancy.png", scan, palette="silver")
io.write_image("si111_vacancy.tif", scan)
io.write_csv("si111_vacancy_atoms.csv", surface.atoms.table())
project.save("si111_vacancy.materia")
print("\nexported structure, image, table and project")
