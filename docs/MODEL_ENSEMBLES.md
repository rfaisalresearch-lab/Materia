# Model-disagreement ensembles

Materia can evaluate the same structure with multiple classical potentials and
record where their atomic forces disagree. It can also compare energy changes
between two states within each model, where arbitrary model-specific energy
zeros cancel.

```python
surface = materials.load("silicon").bulk()

forces = ensembles.forces(
    surface,
    models=["stillinger-weber-si", "lennard-jones"],
)

changed = surface.copy()
changed.structure.positions[0, 0] += 0.05
delta = ensembles.energy_change(
    surface,
    changed,
    models=["stillinger-weber-si", "lennard-jones"],
)
```

Every call receives a run id and persists results and checksummed arrays.
`ensembles.result(...)` and `ensembles.array(...)` retrieve them.

## Force disagreement

For every model, the backend stores the total energy and the full atomic force
array. It reports:

- the component-wise mean force
- the sample standard deviation of each force component
- the per-atom RMS vector disagreement around the ensemble mean
- the whole-structure pairwise force RMS matrix
- the maximum and mean per-atom disagreement

Absolute energies are retained for audit, but their cross-model range is
explicitly marked as not being an energy uncertainty. Different potentials can
use different reference zeros.

## Energy-change disagreement

For reference and target structures with identical stable atom-id and element
order, each model computes `E_target - E_reference`. The backend reports every
within-model change, their mean, sample standard deviation, range, and whether
all selected models agree on the sign.

This is safer than comparing raw total energies across unrelated potentials,
but the spread is still model disagreement. It is not a calibrated confidence
interval, experimental error bar, or guarantee that the true answer lies in
the selected range.

## Refusal and provenance

At least two uniquely labelled models are required. Every model must support
the structure and return finite energy and force values. Structures with
external mechanical biases are refused so the result describes intrinsic
model disagreement. Model names, parameters, approximations, references,
structure digests, and the full numerical model axis are persisted.

The implementation, public Python workflow, persistence, and regression gates
are built. The combined verification phase remains pending.
