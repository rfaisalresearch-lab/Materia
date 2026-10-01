# Minimum-energy paths and reaction barriers

Materia now has a backend workflow for nudged elastic band calculations with
an optional climbing image. It is intended for diffusion hops, conformational
changes and reaction paths where the initial and final structures are known.

The path uses ASE's NEB geometry and FIRE optimiser. Every energy and force
comes from the selected in-process Materia potential. The input endpoints are
never mutated. A run stores the full image trajectory, physical forces,
energies, cumulative reaction coordinate and optimisation history as
checksummed project arrays.

## Python workflow

```python
initial = io.read("relaxed-state-a.xyz")
final = io.read("relaxed-state-b.xyz")

path = neb.run(
    initial,
    final,
    model="stillinger-weber-si",
    images=7,
    spring_eV_A2=0.1,
    fmax_eV_A=0.05,
    interpolation="idpp",
    climb=True,
)

print(path.forward_barrier_eV)
print(path.convergence.message)
```

The initial and final structures must contain the same atom ids and elements
in the same order, with the same cell, periodicity and fixed-atom mask. Fixed
atoms cannot move between endpoints. Endpoints must meet their own force
tolerance unless `allow_unrelaxed_endpoints=True` is set, in which case every
barrier is labelled estimated.

## Stored records

Each run receives an id and stores:

| Record | Contents |
| --- | --- |
| `barrier` | forward barrier, reverse barrier, reaction energy and saddle image |
| `energies` | total energy of every image |
| `reaction_coordinate` | cumulative Cartesian path length |
| `path_forces` | physical force on every atom in every image |
| `positions` array | complete image trajectory |
| `history` array | optimisation step, current barrier and maximum projected force |

Use `neb.runs()`, `neb.result()` and `neb.array()` to retrieve stored work.

## Scientific boundary

This is a local path optimiser. A converged path is not proof that no lower
path exists. Its answer depends on the endpoints, number of images and initial
interpolation. The climbing image approaches a first-order saddle only when
the path and endpoints are converged. Classical potentials cannot describe a
reaction whose bonding physics lies outside their parameterisation.

The current implementation does not schedule one DFT calculation per image,
does not distribute images over multiple processes, does not perform adaptive
image insertion, and does not calculate harmonic transition-state rates.
Those remain separate capabilities.

Verification is pending the final combined test phase. The committed tests
include an analytic symmetric double well with a known 2.5 eV barrier,
endpoint validation, cancellation, fixed atoms, non-convergence and immutable
inputs.
