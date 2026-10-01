# DFT equation of state

Materia fits the equation of state of a bulk crystal from GPAW ground states at
several volumes, in `materia/experiments/dft/eos.py`. It gives the equilibrium
volume, the equilibrium cell lengths, the bulk modulus B0 and its pressure
derivative B', and can scale the structure to the equilibrium volume as one
undoable change. GPAW computes every energy; Materia pins the settings, checks
every point, fits and verifies.

## Why a dedicated experiment

A lattice constant or bulk modulus comes from comparing total energies of cells
of different size, and those energies are comparable only if every cell is
computed with the same discretisation. An automatic k-point grid follows the
cell size, so a hand-made volume scan can change grid part way: an aluminium
scan with the automatic grid did exactly that and gave a bulk modulus 135
percent too high (`ACCURACY_BENCHMARKS.md`). This experiment pins the grid and
the plane-wave cutoff of the reference crystal and verifies that every volume
uses them.

## What energy is fitted

The result is the static-lattice equation of state at 0 K, so the fitted energy
is GPAW's total energy extrapolated to zero smearing width. With smeared
occupations GPAW's primary energy is the free energy F = E - TS of the smeared
electrons, and TS changes with volume: for copper at 0.1 eV Fermi-Dirac
smearing it goes from -5.69 to -6.45 meV between V/V_ref = 0.94 and 1.06, and a
fit to F gives V0 = 12.0893 A^3 where the zero-width energy gives 12.0818 A^3
(PBE, 600 eV, 12x12x12). The free energy is therefore stored as its own curve
and fitted separately, and the difference between the two fitted volumes is
reported. GPAW's stress, like its forces, is the derivative of F at the finite
width, so the stress is compared with -dF/dV of the free-energy fit, never with
the zero-width curve. With fixed occupations the two energies are identical.
Specification version 1.0 fitted F and is not read.

## Frozen specification

`EOSSpec` uses schema `materia.dft.equation-of-state`, version `2.0`, with a
SHA-256 digest over every field:

| Field | Meaning |
| --- | --- |
| `ground_state` | the reference crystal's full ground-state specification; its k-point grid and cutoff are used at every volume; observables energy, forces and stress |
| `source` | structure, stored ground state or stored relaxation, with fingerprints and the source energy |
| `volume_min_scale`, `volume_max_scale` | smallest and largest volume as fractions of the reference, 0.7 to 1.3 |
| `n_points` | 5 to 15 evenly spaced volumes |
| `volume_scales` | every sampled volume, as stored |
| `fit` | `birch-murnaghan-3` |

Each point scales the cell and the atoms isotropically with fractional
coordinates fixed. Refused: slabs, wires and clusters (their volume is the
box's), real-space grids and LCAO (the number of grid points changes with the
cell and steps the curve), and charged cells (the compensating background adds a
volume-dependent term).

## Verification

Nothing is stored unless:

* every volume converged, with GPAW reporting back exactly the parameters sent,
  and every point's settings equal the reference's apart from the geometry
* every energy and force is finite and the volumes are the specified ones
* the third-order Birch-Murnaghan fit reproduces every point within 2 meV per
  atom
* the fitted minimum lies inside the sampled volumes
* B0 is positive and B' lies between 1 and 12

Two further checks are reported as evidence and as warnings. GPAW's stress gives
a pressure at every volume, compared with -dF/dV of the free-energy fit; a
difference above 1 GPa means the plane-wave basis is not converged (Pulay
stress). The largest
force at every volume is reported; forces above 0.05 eV/A mean the crystal has
free internal parameters that are not relaxed at each volume.

## States and apply

**current** while the structure has the reference geometry, **applied** once it
has been scaled to the fitted equilibrium, **stale** after any other change,
**detached** for a structure the project does not hold, and **corrupt** when a
stored array no longer matches its checksum. Applying scales every lattice
vector and every position by (V0 / V_ref)^(1/3) as one undoable change; it is
refused for stale, detached, corrupt or already applied results.

## Interface, Python, HTTP and storage

The Equation of state group of the DFT panel sets the reference, the volume
range and the number of points, and states the pinned grid, cutoff and volumes.
The result shows energy and pressure against volume with the fitted curves, a
table of V0, cell lengths, B0, B', E0, the fit residual, the stress comparison
and the largest force, an apply button, CSV export and a provenance inspector.

```python
spec = dft.eos_spec(structure, xc="PBE", cutoff_eV=500.0, kpoints=[12, 12, 12],
                    volume_min_scale=0.94, volume_max_scale=1.06, n_points=7)
dft.eos_check(spec)["by_field"]
result = dft.eos(spec=spec)
result.value["V0_A3_per_atom"], result.value["B0_GPa"], result.value["B1"]
dft.eos_array(name="energies")
dft.eos_state()
dft.eos_apply()
```

Routes: `dft/eos/spec`, `dft/eos/run`, `dft/eos/runs`, `dft/eos/result`,
`dft/eos/export`, `dft/eos/apply`. Results use `dfteos::<run_id>::eos`; the
arrays `scales`, `volumes`, `energies` (zero-width), `free_energies`,
`fit_pressures` (-dE/dV of the zero-width fit, the 0 K pressure),
`free_energy_fit_pressures` (-dF/dV) and, where available, `stress_pressures`
and `max_forces` are checksummed in the array store. The CSV has one row per
volume with both energies, both fitted pressures and GPAW's stress pressure.
`examples/scripts/17_dft_equation_of_state.py` is a runnable example.

## Validation

`tests/validation/test_dft_eos_live.py` runs the tool on silicon at 500 eV and
12x12x12 against the all-electron PBE volume and bulk modulus, applies the
equilibrium and confirms it with GPAW's stress in a fresh ground state, and runs
copper with the automatic grid to show that the pinned grid gives a smooth
curve. See `VALIDATION.md` for the measured values.

## Limits

* Fractional coordinates are fixed: crystals with free internal parameters
  (wurtzite u, quartz, corundum) are not relaxed at each volume. Use the
  variable-cell relaxation for those.
* Isotropic scaling only: c/a of a hexagonal crystal is not optimised.
* Static lattice at 0 K, without zero-point or thermal expansion.
* The functional's own error remains: PBE overestimates lattice constants.
