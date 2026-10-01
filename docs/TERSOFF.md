# Tersoff bond-order potentials

`materia/physics/tersoff.py` evaluates Tersoff potentials natively, with
analytic forces, in exactly the form LAMMPS `pair_style tersoff` uses, from
LAMMPS-format parameter files. Registered models: `tersoff/si`,
`tersoff/sic`, `tersoff/sicge`, `tersoff/bnc`. The parameter files are part
of LAMMPS (GPL) and are not shipped: they are downloaded on first use into
`~/.cache/materia/lammps-potentials`, and the citation lines in each file are
copied into every result.

## Checked against LAMMPS

| System | Energy difference | Largest force difference |
| --- | --- | --- |
| Distorted Si, 64 atoms, Si.tersoff | 4e-12 eV | 2e-11 eV/A |
| Distorted 3C-SiC, 64 atoms, SiC.tersoff (two species, cross terms) | 1e-11 eV | 2e-10 eV/A |

Forces also agree with central differences of the energy to 2e-9 eV/A, and
diamond silicon has the published cohesive energy of 4.63 eV per atom.

## Limitations

* Tersoff form only; REBO, AIREBO, ReaxFF and Tersoff variants with
  different functional forms (tersoff/mod, tersoff/zbl) are not implemented.
* Pure Python and NumPy; fine for thousands of atoms, not for millions.
