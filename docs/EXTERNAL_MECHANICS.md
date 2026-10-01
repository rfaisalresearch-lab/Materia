# External forces and positional restraints

Materia can attach mechanical biases to atoms by stable atom id. The records
live with the structure, survive project save and load, and can be undone and
redone when the structure belongs to the project.

```python
surface = materials.load("silicon").bulk(repeat=(2, 2, 2))
atom_id = int(surface.structure.ids[0])

surface.apply_force(atom_id, [0.25, 0.0, 0.0])
surface.restrain(atom_id, 4.0, target_position_A=[0.1, 0.0, 0.0])
surface.relax(model="recommended")

active = surface.biases()
surface.clear_biases(atom_ids=[atom_id], kind="constant_force")
```

The same operations are available as `mechanics.apply_force`,
`mechanics.restrain`, `mechanics.list`, and `mechanics.clear` for a raw
structure or surface handle.

## Equations and units

A constant force in eV/A contributes

`E_ext = -F dot (r - r_ref)`

and adds `F` to the force on the selected atom. The reference position records
the arbitrary zero of this external potential. A restraint with spring constant
`k` in eV/A^2 contributes

`E_restraint = k |r - r_target|^2 / 2`

and adds `-k (r - r_target)` to the force.

All coordinates are unwrapped Cartesian angstrom coordinates. Crossing a
periodic boundary does not replace the displacement with a minimum-image
displacement. This convention makes pulling directions continuous for
trajectories whose coordinates remain unwrapped, but a caller that explicitly
wraps atoms must also update the reference or target if continuity is required.

## Solver scope

The composed energy and forces are used by classical single-point evaluation,
FIRE relaxation, velocity-Verlet dynamics, Langevin dynamics, and Gamma-point
finite-displacement harmonic analysis. The full bias definition is included in
result provenance.

Periodic phonon dispersion refuses biased structures because repeating a
single-atom load through a supercell and enforcing translational symmetry are
not uniquely defined. NEB also refuses biased endpoints because it would be
ambiguous whether and how the bias should act on every image. Clear the biases
before those calculations.

## Scientific limits

These are externally prescribed mechanical terms, not forces predicted from a
tip, field, pressure apparatus, or feedback controller. They do not model an
experimental actuator automatically. Constant-force energy has an arbitrary
zero and is only comparable when the same reference convention is used.

The implementation and its regression gates are built. The tests are recorded
but have not yet been run because the current feature push defers the combined
verification phase.
