# Before-and-after structure comparison

Materia can compare two structures by stable atom id without changing either
input. The analysis stores per-atom displacements, scalar movement summaries,
atom additions and removals, substitutions, cell deformation, finite strain,
and heuristic bond changes.

```python
before = materials.load("silicon").bulk(repeat=(2, 2, 2))
after = before.copy()
after.structure.positions[0, 0] += 0.1

change = compare.structures(before, after)
print(change.metrics["rms_displacement_A"])
print(change.bonds_formed, change.bonds_broken)
```

The same operation is available as `before.compare_to(after)`. Results and
numerical arrays receive a unique run id and remain retrievable through
`compare.result(...)` and `compare.array(...)`.

## Coordinate conventions

`mode="auto"` selects fractional comparison when both structures share the
same periodic-boundary flags and at least one direction is periodic. Fractional
positions from each cell are compared and mapped back into the reference cell.
This separates homogeneous cell deformation from non-affine atomic movement.

For two non-periodic structures, automatic mode uses a proper Kabsch rotation
and independent centering to remove rigid translation and rotation. Mixed
boundary conditions use raw Cartesian displacement.

The explicit modes are:

- `cartesian`: direct after minus before coordinates
- `minimum-image`: shortest displacement under an unchanged periodic cell
- `fractional`: wrapped fractional change mapped through the reference cell
- `rigid-align`: translation and proper-rotation alignment for clusters

`remove_translation=True` subtracts the mean matched-atom displacement after
the selected convention.

## Cell and topology changes

For invertible cells, the deformation gradient maps the reference cell to the
new cell. Materia also reports the finite Green-Lagrange strain, absolute and
relative volume change. Added and removed atoms are identified from stable ids.
An atomic-number change on a shared id is reported as a substitution.

Bond formation and breaking use the existing covalent-radius and shared-Voronoi
face heuristic. These are geometric candidate bonds, not electronic bond
orders. Set `compare_bonds=False` to omit that analysis.

## Limits

Stable ids are required for atom matching. The tool does not infer an atom
mapping when independently imported structures use unrelated ids. Fractional
comparison is a reference-cell convention and does not itself identify slips,
permutation-equivalent atoms, or symmetry-equivalent domains. Rigid alignment
is intentionally refused for periodic structures.

The implementation and its regression gates are built. The tests are recorded
but have not yet been run because the current feature push defers the combined
verification phase.
