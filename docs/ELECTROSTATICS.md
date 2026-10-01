# Point-charge electrostatics

Materia computes the long-range Coulomb energy, forces, site potentials and
site fields of explicit point charges. It is a Tier 1 classical capability:
the charges are fixed numbers the user supplies, there are no electrons, no
polarisation and no screening. What it gives is the exact electrostatics of
that point-charge model, summed correctly for periodic crystals, slabs and
clusters, with every numerical choice recorded.

Code: `materia/physics/electrostatics.py` (the sums),
`materia/solvers/electrostatics.py` (provenance, convergence, refusal),
`materia/python_api/api.py` (`ElectrostaticsNamespace`),
`materia/desktop_ui/service.py` (`electrostatics_*`). Tests:
`tests/unit/test_electrostatics.py`,
`tests/integration/test_electrostatics_service.py`,
`tests/validation/test_electrostatics_validation.py`.

## Where the charges come from

Charges are never inferred from a material, an oxidation state table or a
bonding heuristic. A structure carries at most one charge model, recorded in
`structure.info["point_charge_model"]` and saved with the project.

| Model | Charges | Refused when |
| --- | --- | --- |
| `per-element` | one charge per element, typed by the user with a stated source | an element in the structure is not listed (it is never assumed neutral) |
| `per-atom` | one charge per atom id, from Python | an atom has no charge, or a listed id no longer exists |
| `formal-point-ion` | the structure's formal charges, taken literally | never, but it is labelled as the full ionic limit |

The formal-point-ion model is the textbook Madelung model of an ionic crystal.
Real ions in a solid carry less charge than their oxidation state, so energies
from it bound the magnitude of the ionic contribution from above. It is
available because it is the conventional reference, not because it is a
physical description of any material.

Assigning, changing or removing a charge model is one undoable operation.
Computing is read-only: it is logged in the project history and stores its
results, and it never changes the structure.

## Geometries

The periodicity of the cell decides the method. Nothing is guessed from the
atoms.

| Cell | Method | Net charge |
| --- | --- | --- |
| periodic in 3 directions | Ewald summation | refused unless a uniform neutralising background is explicitly requested |
| periodic in 2 directions (slab) | Ewald in an internally enlarged cell with the Yeh-Berkowitz dipole correction | refused |
| not periodic (cluster) | direct pairwise sum, exact | allowed |
| periodic in 1 direction (wire) | none | refused: no one-dimensional method is implemented |

### Bulk

`E = E_real + E_recip + E_self + E_background + E_surface`, with

* `E_real = 1/2 sum' q_i q_j erfc(alpha r) / r` over all pairs and images
* `E_recip = (2 pi / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) / k^2 |S(k)|^2`
* `E_self = -(alpha / sqrt(pi)) sum q_i^2`
* `E_background = -pi Q^2 / (2 V alpha^2)`, only with `background="uniform"`
* `E_surface = (2 pi / 3V) |M|^2`, only with `surrounding="vacuum"`

The default surrounding is a conductor at infinity (tin foil), which makes the
energy independent of which periodic image of each atom is stored. The vacuum
surrounding is the energy of a large spherical crystal in vacuum and does
depend on the stored images; that is physics, and the program says so. It is
refused for a charged cell, where the dipole depends on the origin.

A charged periodic cell has no finite Coulomb energy. With an explicitly
requested uniform background the energy is finite and is reported with the
background as its own term, but it depends on the cell size (Makov and Payne
1995). No finite-size correction is applied, because a correct one needs the
dielectric constant of the host.

### Slab

The stored cell height is ignored. The two in-plane vectors are kept and a new
third vector is built along the slab normal with a vacuum gap chosen so that
the interaction between a slab and its artificial images is below the
requested accuracy:

`gap = max(-ln(eps) / G_min, 2 h, 10 A)`

where `G_min` is the smallest in-plane reciprocal vector and `h` the slab
thickness. The Yeh-Berkowitz correction `(2 pi / V') M_z^2` then removes the
artificial field across the gap. For a neutral slab that correction is exact
for the laterally averaged potential; the remaining error decays as
`exp(-G_min gap)`, which is what the gap controls.

A charged slab is refused. A neutralising background would fill the vacuum
and the answer would depend on the height of that arbitrary box.

### Cluster

The direct sum over all pairs, O(N^2), with no truncation.

## Numerical parameters and convergence

For accuracy `eps` (default 1e-8) and `s = sqrt(-ln eps)`:

* the real-space cutoff balances the cost of the two sums,
  `r_c^6 = c s^6 V^2 / (2 pi^3 N)`, with `c = 0.11` the measured cost of one
  reciprocal term relative to one real-space pair in this implementation,
  held between 4 and 30 A;
* `alpha = s / r_c` and `k_c = 2 alpha s`.

The largest neglected real-space factor is `erfc(alpha r_c)` and the largest
neglected reciprocal factor is `exp(-k_c^2 / 4 alpha^2) = eps`. Both are
recorded, together with `alpha`, both cutoffs, the number of k vectors and the
number of real-space pairs. `alpha_per_A`, `real_cutoff_A` and
`kspace_cutoff_per_A` may be set explicitly.

**Convergence check (on by default).** The calculation is repeated with an
independent split, `alpha` scaled by 0.8 and, for a slab, a 30 % larger gap.
The exact answer does not depend on either choice, so the difference between
the two is a direct measure of numerical error. The result is labelled
converged, and its origin `calculated`, only if the energies agree within
`energy_tolerance_eV_per_atom` (default 1e-6 eV per atom) times the number of
atoms and the forces within `force_tolerance_eV_A` (default 1e-4 eV/A). The
energy difference is reported as the uncertainty. With the check switched off
the result is labelled `estimated` and not converged, because nothing
independent confirms it.

## Units

| Quantity | Unit | Note |
| --- | --- | --- |
| charge | e | |
| energy | eV | `k_e = 14.3996454784 eV A` (CODATA 2018 via SI constants) |
| site potential | V | at the nucleus, excluding the atom's own charge |
| site field | V/A | at the nucleus, excluding the atom's own charge |
| force | eV/A | `F_i = q_i E_i` |

## Interface

**Solvers ▸ Electrostatics (point charges)** shows the geometry and method,
the charge model with its source, the net point charge and a per-element
charge table. It assigns per-element or formal-point-ion charges, removes
them, and sets accuracy, surrounding, background and the convergence check
(surrounding and background are disabled for anything but a bulk crystal).
**Compute electrostatics** runs as a background job with progress in the
status bar and can be cancelled from the job log. The last run is shown with
its energy, every term, the split difference, the Ewald parameters, the
convergence message, a table of site potentials by element and charge, and
its approximations. If the structure or its charges have changed since, a
warning says so.

**Atom ▸ Energy** shows the point charge, site potential, site field and
Coulomb force at the selected atom from the newest run whose input
fingerprint matches the current structure and charges. After any edit it
shows nothing rather than stale values.

## Python

```python
electrostatics.assign(by_element={"Ga": 1.0, "As": -1.0}, source="unit charges")
run = electrostatics.compute()                       # SolverResult
run["energy"].value, run["energy"].unit             # float, "eV"
run["forces"].value, run["site_potential"].value    # arrays in atom order
run["energy"].extra["components_eV"]                # real, reciprocal, self, ...
run["energy"].convergence.message
electrostatics.status()                             # model, geometry, readiness
electrostatics.runs()                               # stored runs, newest first
electrostatics.result(quantity="site_field")        # a stored quantity
electrostatics.clear()                               # undoable
```

`compute(settings={...})` or keyword overrides accept `accuracy`,
`alpha_per_A`, `real_cutoff_A`, `kspace_cutoff_per_A`, `surrounding`,
`background`, `check_convergence`, `energy_tolerance_eV_per_atom` and
`force_tolerance_eV_A`; unknown keys are refused. A refused or cancelled run
returns an unsupported `energy` result with the reason and no number.

## Persistence

The charge model travels with the structure. Each run stores
`electrostatics::<run_id>::energy`, `forces`, `site_potential`, `site_field`
and `point_charges`, each with the full provenance record: charge model and
source, geometry, settings, every chosen parameter, boundary conditions,
references and a fingerprint of the inputs. Per-atom arrays larger than the
project file's inline limit (20,000 values) are omitted and marked as omitted;
the energy and all summary values are always kept. No schema migration was
needed: the charge model lives in the free-form structure `info` and results
use the existing result records.

## Validation

See [VALIDATION.md](VALIDATION.md#point-charge-electrostatics). Madelung
constants of rock salt, caesium chloride and zinc blende, the square lattice
of alternating charges, a point charge in a neutralising background, the
slab-to-bulk layer energy, and the isolated ion pair.

## Limitations

* Point charges only: no dipoles or higher multipoles, no polarisability, no
  charge penetration, no dielectric screening.
* The electrostatics solver alone does not relax a structure or run dynamics,
  since point charges would collapse. It refuses both and names the rigid-ion
  model ([RIGID_ION.md](RIGID_ION.md)), which adds the short-range repulsion.
* No stress tensor.
* No one-dimensional (wire) method.
* Charged slabs are refused. Charged bulk cells carry the Makov-Payne
  finite-size error uncorrected.
* Ewald, not particle-mesh Ewald: cost grows as about N^1.5. See
  [PERFORMANCE.md](PERFORMANCE.md) for measured times.
* Charges do not respond to anything. Charge transfer needs a self-consistent
  electronic model, which Materia does not yet have.
* Nothing yet applies an external electric field. The site fields reported
  here are the fields of the point charges themselves.
