# DFT density of states

Materia computes total density of states and projected density of states with
GPAW. The implementation is a first-principles Tier 3 experiment. It does not
reuse the fitted tight-binding DOS and does not label Kohn-Sham states as
quasiparticle excitation energies.

## Scope

A DOS can describe the active structure, a stored converged ground state, or
the final geometry of a stored converged relaxation. It stores:

* the total DOS in states per eV per cell
* spin channels when the ground state is spin polarised
* atom and angular-momentum projections onto bound PAW projectors
* the energy grid, eigenvalues, irreducible k-point weights and Fermi level
* state-count, electron-count and independent recomputation checks
* the complete ground-state and DOS specifications, software versions, PAW
  dataset hashes, source run and current, stale or detached state

Spatial LDOS maps are a separate experiment ([DFT_LDOS.md](DFT_LDOS.md)); orbital rendering is not implemented. Band
structures along a reciprocal-space path are a separate experiment; see
[DFT_BAND_STRUCTURE.md](DFT_BAND_STRUCTURE.md).

## Frozen specification

`DOSSpec` uses schema `materia.dft.dos`, version `1.0`. It freezes the source,
ground-state specification, energy reference, window, spacing, broadening,
Gaussian width, spin channels, DOS k-point grid, Gamma centring, band count and
every projection. The SHA-256 digest covers the complete specification.

The energy zero can be the self-consistent Fermi level. An absolute energy zero
is offered only for a finite cluster without an external field. In periodic
cells, GPAW's Kohn-Sham zero is set by the average electrostatic potential and
is not a vacuum reference.

Gaussian broadening uses GPAW's definition
`exp(-((E - e) / w)^2) / (sqrt(pi) w)`. Its standard deviation is
`w / sqrt(2)`. The linear tetrahedron method is restricted to bulk crystals
with at least two k-point divisions on every axis.

## Execution and validation

The worker first converges the frozen ground state. It then calls GPAW's
`fixed_density` with the specified k-point grid and band count and requires all
bands to converge. Total DOS and PDOS are evaluated with GPAW's DOS calculator.

Every value sent to GPAW is compared with the value reported by the running
calculator. Materia independently recomputes the total DOS from the returned
eigenvalues and k-point weights and refuses the result if the curves disagree
by more than `1e-6` of the maximum. It also refuses non-finite or negative
curves, changed Fermi levels, mismatched spin channels, unavailable PAW
projectors and a band count that does not reach at least three Gaussian widths
above the requested window.

A refused, cancelled, timed-out, non-converged or invalid run stores no numeric
result. A completed DOS observes a structure and creates no undo point.

## Interface

The Density of states group in the DFT panel exposes the source, energy grid,
broadening, spin channels, k-points, bands and available element projections.
The result view plots total and projected curves, spin channels, the Fermi
level and the numerical checks. It exports the full-resolution curves to CSV.

## Python

```python
spec = dft.dos_spec(structure, xc="PBE", cutoff_eV=400.0,
                    kpoints=[4, 4, 4], dos_kpoints=[8, 8, 8],
                    energy_min_eV=-12.0, energy_max_eV=6.0,
                    energy_step_eV=0.02, width_eV=0.1,
                    projections=[{"element": "Si", "angular": "p"}])
dft.dos_check(spec)
run = dft.dos(spec=spec)
summary = run["dos"].value
energies = dft.dos_array(name="energies").data
total = dft.dos_array(name="dos_total").data
projected = dft.dos_array(name="pdos").data
dft.dos_state()
```

Use `source_run` with `source_kind="ground-state"` or
`source_kind="relaxation"` to inherit the exact electronic settings and
geometry of a stored run. Those inherited electronic variables cannot be
changed.

## HTTP and project storage

The local service exposes `dft/dos/spec`, `dft/dos/run`, `dft/dos/runs`,
`dft/dos/result` and `dft/dos/export`. Results use the
`dftdos::<run_id>::<quantity>` prefix. Arrays use the same prefix in the
checksummed chunk store.

## Live validation

`tests/validation/test_dft_dos_live.py` runs against GPAW 25.7.0 and its PAW
datasets. The current cases check H2 levels and two-electron count against an
independent ground state, silicon Gaussian and tetrahedron state counts and
gap, a spin-resolved hydrogen atom, and DOS from the final geometry of a real
relaxation. The tests skip with the installation reason when GPAW or its
datasets are unavailable.

## Limits

The curves depend on the exchange-correlation functional, PAW datasets,
k-point grid, band count and broadening. Semilocal Kohn-Sham gaps are not
quasiparticle gaps. PDOS is the weight inside PAW augmentation projectors, so
it need not sum to the total DOS. A converged numerical result does not remove
model error.
