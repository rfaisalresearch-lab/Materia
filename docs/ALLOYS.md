# Alloys and partial occupancy

Materia now builds reproducible finite substitutional alloys and samples
partially occupied crystallographic sites. These are structure-generation
operations. They do not silently claim that an unrelaxed random cell is a
thermodynamic phase.

## Exact-composition random alloys

```python
silicon = materials.load("silicon").bulk(repeat=(4, 4, 4))
alloy = silicon.alloy({"Si": 0.75, "Ge": 0.25}, seed=17)
print(alloy.structure.info["alloy"])
```

The requested fractions are converted to integer site counts with Hamilton
apportionment, so every selected site is assigned and the finite cell is as
close as possible to the requested composition. The assignments are shuffled
with the recorded seed. The input structure is unchanged and the new alloy is
registered as a separate project structure.

Pass `atom_ids` to limit substitution to a sublattice or region. Charged sites
are refused because changing their elements without a charge model would leave
physically inconsistent charges. The generation record includes the target
fractions, realized counts and fractions, selected stable atom ids, seed,
input digest and limitations.

## Partial occupancy

A material basis may assign an `occupancy` from zero through one to each
crystallographic site. A finite atomic model must choose which repeated sites
exist. Materia therefore requires an explicit seed rather than silently
turning partial occupancy into full occupancy.

```python
sample = materials.load("partially-occupied-material").bulk(
    repeat=(4, 4, 4),
    occupancy_seed=23,
)
```

Each basis-site family is sampled over the repeated conventional cells. Its
occupied count is the integer below or above the expectation, selected with
the seeded fractional probability. The record reports every target and
realized occupancy.

## Scientific boundary

A seeded random alloy is not a special quasirandom structure. Partial-site
families are sampled independently, with no inferred short-range order,
charge-balance constraint or defect chemistry. Neither operation relaxes the
cell or atoms, selects a stable phase, or supplies a force field for the new
elements. Those steps must be requested separately with a model that supports
the realized composition.

Verification is pending the final combined test phase. The committed cases
cover exact apportionment, deterministic seeds, selected sublattices, input
immutability, charge refusal, finite occupancy realization and project
registration.
