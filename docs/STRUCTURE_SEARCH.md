# Automated structure search

Materia includes deterministic fixed-cell basin hopping for classical force
models. It is a candidate generator and ranker, not a proof of a global ground
state.

```python
surface = materials.load("silicon").bulk(repeat=(2, 2, 2))

search_result = surface.search_minima(
    trials=40,
    displacement_A=0.15,
    temperature_eV=0.08,
    fmax_eV_A=0.03,
    relaxation_steps=400,
    keep=8,
    seed=17,
)

best = search_result.best.structure
energies = search.array(name="candidate_energies")
```

The equivalent top-level call is `search.basin_hopping(surface, ...)`.
`activate_best=True` installs a copy of the lowest retained candidate as a new
undoable project structure. The input structure is never changed by the
search.

## Algorithm

The starting geometry is relaxed with the selected classical potential. Each
trial adds a seeded Gaussian Cartesian displacement to movable atoms, removes
the net translation when more than one atom can move, wraps periodic
coordinates, and performs a fixed-cell FIRE relaxation. A downhill move is
accepted. An uphill move is accepted with probability
`exp(-delta_energy / temperature_eV)`.

Candidate minima are sorted by energy. A candidate is treated as a duplicate
only when its energy and translation-removed, minimum-image Cartesian RMSD are
both within the explicit tolerances. The complete trial energy, residual
force, local convergence, acceptance decision, and ranked candidate positions
are stored as checksummed project arrays.

## Reproducibility and refusal rules

The random seed and every search and relaxation setting are stored in
provenance. The same starting structure, potential implementation, settings,
and numerical environment are intended to reproduce the same path.

Search refuses an empty structure, a structure with no movable atoms, an
unsupported composition, or a structure carrying external mechanical biases.
By default the starting relaxation and every retained minimum must converge.
`require_converged=False` is available for explicitly exploratory work and the
local convergence record is retained for every trial.

## Scientific limits

The current search keeps composition and cell fixed. It does not exchange atom
types, change atom count, mutate lattice vectors, use symmetry fingerprints,
identify every permutation-equivalent minimum, perform saddle searches, or
prove global optimality. The result explicitly records
`global_minimum_proven: false`. Quality is limited by the selected classical
potential and search budget.

The implementation and its regression gates are built. The tests are recorded
but have not yet been run because the current feature push defers the combined
verification phase.
