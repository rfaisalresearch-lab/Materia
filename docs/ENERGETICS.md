# Defect and surface energetics

`materia.physics.energetics` works with any Materia potential:

- `equilibrium_bulk(crystal, potential)`: the isotropic scale that minimises
  the energy, with the energy and volume per atom. It is refused for a
  crystal whose atoms feel forces (its shape is not fixed by symmetry) and
  when the minimum lies more than 10% away.
- `vacancy_formation(crystal, potential, supercells)`: constant-volume
  formation energy `E(N-1) - (N-1)/N E(N)`, relaxed and unrelaxed, in each
  supercell, marked converged only when the last two agree within 0.01 eV.
- `surface_energy(material, potential, miller, layers)`: relaxed and
  unrelaxed `(E_slab - N e_bulk) / 2A` in J/m^2 for slabs of increasing
  thickness, built at the material's geometry and scaled to the potential's
  own lattice constant, marked converged only when the two thickest slabs
  agree within 5 mJ/m^2.

`eam.load_lammps(name)` downloads an eam/alloy file from the LAMMPS
repository (GPL-2.0, never shipped). Some of those files give a wrong atomic
number in an element header, which LAMMPS ignores. Materia then uses the
element symbol and records the correction in the potential's notes;
`load_file` still refuses such files.

## Validation against the published potential

Mishin Cu EAM (Y. Mishin et al., Phys. Rev. B 63 (2001) 224106, Table III):

| Quantity | Materia | Published |
| --- | ---: | ---: |
| a0 | 3.6149 A | 3.615 A |
| Cohesive energy | 3.540 eV | 3.54 eV |
| Relaxed vacancy formation (500 atoms) | 1.273 eV | 1.27 eV |
| gamma(111) | 1.2395 J/m^2 | 1.239 |
| gamma(100) | 1.3453 J/m^2 | 1.345 |
| gamma(110) | 1.4755 J/m^2 | 1.475 |

The unrelaxed values do not change with supercell or slab size once the size
exceeds the cutoff, which the unit tests check exactly.

## A property of the shipped Zhou 2004 Cu potential

Its cutoff, 5.71575 A, coincides with the fifth-neighbour shell of fcc copper
at equilibrium (a sqrt(2.5) = 5.7158 A). The energy-volume curve therefore
has a kink at its minimum, so the pressure is not zero on both sides, and
small-strain elastic constants are noisy (which is why `elastic_tensor`
checks a doubled strain). Use the Mishin potential where smooth equilibrium
properties matter.
