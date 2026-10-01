# Rigid-ion potentials

A rigid-ion model gives every atom a fixed charge and adds a short-range
repulsion between ions, so an ionic solid has a stable arrangement that point
charges alone do not. Materia's rigid-ion models combine the Ewald
electrostatics of [ELECTROSTATICS.md](ELECTROSTATICS.md) with Buckingham pair
terms. They are Tier 1 classical models with energy, forces, fixed-cell
relaxation and molecular dynamics.

Code: `materia/physics/rigid_ion.py` (the model),
`materia/solvers/classical.py` (relaxation, dynamics, recorded checks),
`materia/solvers/__init__.py` (registration). Tests:
`tests/unit/test_rigid_ion.py`, `tests/integration/test_rigid_ion_service.py`,
`tests/validation/test_rigid_ion_validation.py`. Example:
`examples/scripts/19_rigid_ion_silica.py`.

## Model

Every pair of atoms at distance `r` interacts through

`phi(r) = k_e q_i q_j / r + A exp(-b r) - C / r^6`

with `k_e = 14.3996 eV A`. The Coulomb part is summed over every periodic
image by the electrostatics module: Ewald summation with tin-foil boundary
conditions for crystals, the Yeh-Berkowitz corrected Ewald sum for slabs and a
direct sum for clusters. Wires are refused. The Buckingham part is summed over
pairs closer than the cutoff (10 A by default) and shifted to zero there, so
the energy is continuous and the force jumps at the cutoff by `|dphi/dr|`,
which the provenance record states for each pair (below 1.1e-4 eV/A for BKS).

Ions are not polarisable and their charges do not respond to the environment.
The cell is held fixed during relaxation and dynamics because the Coulomb sum
has no stress tensor.

## Shipped parameterisation

| Id | Charges | Pair terms (A in eV, b in 1/A, C in eV A^6) | Source |
| --- | --- | --- | --- |
| `bks-silica` | Si +2.4 e, O -1.2 e | Si-O: 18003.7572, 4.87318, 133.5381; O-O: 1388.7730, 2.76000, 175.0000; no Si-Si term | van Beest, Kramer and van Santen, Phys. Rev. Lett. 64 (1990) 1955 |

BKS was fitted to Hartree-Fock energies of an H4SiO4 cluster and to the
elastic constants of alpha-quartz. It is the standard classical model for
silica, not a description of any other oxide. The solver is registered as
`rigid-ion/bks-silica` and is the recommended relaxation and dynamics model of
the `silicon_dioxide` material. The model chooser picks it for any structure
made of exactly Si and O. Each result records the parameters, their SHA-256
digest and the reference.

Other parameterisations are built in Python with `RigidIonParameters`, which
requires a stated source and refuses negative, non-finite or duplicated terms.

## The short-range collapse

Wherever `C > 0` the `-C / r^6` term dominates at short range and `phi` falls
to minus infinity as `r` goes to zero. Each such pair interaction has a
maximum at a distance `r_b`; closer than that, the two ions fall into each
other. This is the known failure of Buckingham potentials, often called the
Buckingham catastrophe. Materia finds `r_b` for every species pair from the
full pair interaction, bare Coulomb term included, by a scan followed by
Brent's method:

| Pair | Barrier `r_b` | Pair energy at `r_b` |
| --- | --- | --- |
| Si-O | 1.1936 A | -27.316 eV |
| O-O | 1.4387 A | +20.868 eV |

Any configuration with a pair inside its barrier is refused with the two atoms
named, before evaluation and on every step of relaxation or dynamics. A run
that crosses a barrier stops, and the service puts the structure back as it
was. Materia does not add a repulsive wall or otherwise change the published
potential. Every result reports each pair's closest approach and how far it is
outside its barrier.

## Recorded checks

A single point, a relaxation and the end of a dynamics run each attach
`potential_checks` to the energy result:

* the energy split into the shifted Buckingham sum and the Coulomb sum, with
  the Coulomb sum's real, reciprocal and self terms;
* an independent Ewald evaluation with the splitting parameter scaled by 0.8,
  and whether it agrees within 1e-6 eV per atom and 1e-4 eV/A (the per-step
  sums skip this repeat for speed);
* the closest pair of each kind and its margin to the collapse barrier.

The solver panel prints these lines after a run, and the service returns them
as `checks`.

## Validation

See [VALIDATION.md](VALIDATION.md#rigid-ion-potentials).

* Forces equal central differences of the energy to 1e-6 eV/A for a perturbed
  quartz crystal, slab and cluster, and sum to zero.
* The Buckingham sum equals an explicit sum over lattice images, written
  independently of the neighbour list, to 1e-12 relative.
* A Born-Mayer rock-salt crystal reproduces the closed form
  `-M k_e / r + 6 (phi(r) - phi(r_c))` per ion pair with the published
  Madelung constant to 1e-9 relative, and its minimum-energy distance matches
  the closed-form equilibrium condition to 1e-5 A.
* BKS quartz relaxed at the measured cell keeps its three Si and six O sites
  equivalent to 1e-4 A. Its Si-O bonds come out at 1.595 and 1.604 A against
  1.605 and 1.614 A in the measured structure; this is a plausibility check,
  not a validation of the BKS fit.
* Velocity-Verlet dynamics of 72-atom BKS quartz near 130 K conserves energy
  with a spread below 2e-4 eV per atom at 1 fs, and the spread falls by a
  factor of about 4 when the step is halved.

## Limitations

* Fixed cell only: there is no stress tensor, so lattice constants and elastic
  constants cannot be computed with this model yet.
* One shipped parameterisation (BKS silica). Rigid-ion models for other oxides
  need parameters from a stated source.
* No polarisation (no shell model), no charge transfer, no many-body terms.
* The Buckingham cutoff is a truncation with an energy shift; no long-range
  dispersion correction is added.
* Ewald summation, not particle-mesh Ewald: cost grows as about N^1.5.
