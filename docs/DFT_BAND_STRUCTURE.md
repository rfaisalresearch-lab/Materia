# DFT band structure

Materia computes Kohn-Sham electronic band structures along explicit
reciprocal-space paths with GPAW. The implementation is a first-principles
Tier 3 experiment in `materia/experiments/dft/bands.py`. GPAW does the
computation; Materia freezes the request, checks what GPAW reports against it,
stores the verified result and keeps its provenance. It does not reuse the
fitted tight-binding bands, and it does not present Kohn-Sham bands as
quasiparticle or measured bands.

## Scope

A band structure can describe the active structure, a stored converged ground
state, or the final geometry of a stored converged relaxation. It stores:

* Kohn-Sham eigenvalues `[spin, k-point, band]` in eV, absolute GPAW energies,
  with the self-consistent Fermi level used as the energy zero for plots and
  export
* the path k-points in fractional reciprocal coordinates and in Cartesian
  coordinates (1/A, 2 pi included)
* the cumulative distance along the path in 1/A, with no distance across a
  branch break
* the labels of the special points, the branch breaks and the tick positions
* both spin channels for a spin-polarised ground state
* band edges and gaps read along the sampled path, per spin and together
* the complete ground-state and band-structure specifications, the path
  origin, software versions, PAW dataset hashes, array checksums, the source
  run and the current, stale, detached or corrupt state

Occupations are not stored: path k-points carry no Brillouin-zone integration
weight, so a state's occupation follows from the self-consistent Fermi level,
not from the path. Orbital character (fat bands) is deferred; see Limits.

## Frozen specification

`BandStructureSpec` uses schema `materia.dft.band-structure`, version `1.0`.
Its SHA-256 digest covers every field:

| Field | Meaning |
| --- | --- |
| `ground_state` | the full `materia.dft.ground-state` specification: exact geometry, functional, representation, cutoff or grid spacing, k-point grid, occupations, smearing, charge, spin, tolerances, software and dataset hashes; observables are energy and Fermi level only |
| `source` | structure, ground-state run or relaxation, with run id, spec digest, geometry and electronic fingerprints and source energy |
| `energy_reference` | `fermi-level`; `absolute` is refused for periodic cells, whose Kohn-Sham zero is the average electrostatic potential |
| `scf_symmetry` | `preserve` or `off`, sent to GPAW for the ground state; the path step always runs with symmetry off |
| `path` | branches of labels, such as `GXWKGLUWLK,UX`; a comma starts a new branch |
| `special_points` | label and fractional coordinates of every special point used |
| `path_origin` | generator (`ase` or `user`), ASE version, Bravais lattice, standard path and the cell and periodicity it was generated for |
| `sampling_density_per_invA` | intervals per 1/A of path length used to choose the intervals, or empty when set by hand |
| `segment_intervals` | equal steps along each segment, in path order |
| `kpoints_frac` | every k-point of the path, in order |
| `n_bands`, `extra_bands` | bands returned and converged per spin and k-point; unconverged buffer bands computed and discarded |

### Paths are generated once

A standard path is generated with ASE's Bravais-lattice tables
(Setyawan and Curtarolo special points) for the pinned cell and periodicity
when the specification is built. The special points, labels, intervals and
every k-point are stored in the specification. Changing any other variable
keeps them exactly. Regeneration happens only when `path="standard"` is asked
for, and produces a new specification with a new digest. A specification whose
stored k-points are not the ones its special points and intervals define, in
order, is refused, and so is a path generated for a different cell or
periodicity. If ASE is not installed in Materia's interpreter, no path is
generated and the specification is refused with that reason until
`special_points` and `path` are given explicitly.

### Validity by boundary

| Boundary | Path |
| --- | --- |
| bulk, periodic in three directions | 3D Bravais-lattice path, for example `GXWKGLUWLK,UX` for FCC |
| slab, periodic in two | 2D path in the plane, for example `GMKG` for a hexagonal sheet; any special point with a component along the open axis is refused |
| wire, periodic in one | `GX` along the periodic axis; components along open axes are refused |
| cluster, periodic in none | refused: there is no Brillouin zone and no dispersion, only discrete levels; use the ground-state eigenvalues or the DOS |

Other bounds: fractional coordinates within [-1, 1], labels of one capital
letter and at most seven more letters, digits or primes, no zero-length
segment, 1 to 400 intervals per segment, at most 1000 k-points, sampling
density 0.5 to 200 per 1/A, at least the occupied bands, 0 to 200 buffer bands.
All ground-state rules apply unchanged.

## Execution and validation

The worker converges the frozen ground state with the pinned symmetry choice,
then calls GPAW's `fixed_density` with the explicit list of k-points,
`symmetry="off"`, `nbands = n_bands + extra_bands` and
`convergence={"bands": n_bands}`. Before anything is kept, Materia requires:

* every ground-state parameter, including the symmetry setting, echoed back
  unchanged, and every path-step parameter, including the k-point list,
  echoed back unchanged
* the labels, breaks, reference and band count the worker used equal to the
  specification's
* all requested bands converged in the path step, which used the identity as
  its only symmetry operation
* the k-points GPAW computed (irreducible and full lists) equal to the path's,
  in the path's order, to 1e-10
* eigenvalues with exactly the specified spin channels, k-points and bands, all
  finite and ascending at every k-point
* GPAW's reciprocal cell equal to the pinned cell's, and the cumulative path
  distance the worker derived from GPAW's reciprocal cell equal to Materia's
  independent recomputation from the pinned cell to 1e-8 of the path length
* the Fermi level of the path step equal to the ground state's

A refused, cancelled, timed-out, non-converged, failed or unverifiable run
stores nothing. A complete run is stored in one step with one history line
that is not an undo point: a band structure observes a structure and never
changes it.

## States and storage

The state follows the geometry fingerprint as for the other DFT experiments:
**current** while the structure has the geometry the bands were computed for,
**stale** after any geometry change (undoing the change makes it current
again), and **detached** for a structure the project does not hold. Bands of a
relaxation are stale until the relaxation is applied. A record whose arrays do
not match the SHA-256 recorded when they were computed, or whose eigenvalue
shape or k-points disagree with the specification, is **corrupt**: it is not
plotted, exported or returned by the Python API.

## Interface

The Band structure group of the DFT panel sets the source, energy zero,
ground-state symmetry, path, sampling density, bands and buffer bands, lists
the stored special points with their origin, and shows every refusal beside
the control it concerns. The result view plots the bands against the true
cumulative path distance with symmetry labels (G shown as Gamma), segment
dividers, a heavier divider at a branch break, the Fermi level at zero, spin up
solid and spin down dashed with a spin selector, zoom by special-point range
and energy window, an enlarged view, a table of eigenvalues at any clicked
k-point, a summary with band edges and checks, CSV export and a provenance
inspector.

The CSV has one row per spin and k-point: spin, k index, distance, the three
fractional coordinates, label, a branch-start flag and every band's energy
relative to the Fermi level, all at full precision, with the specification
digest in the header.

## Python

```python
spec = dft.bands_spec(structure, xc="PBE", cutoff_eV=300.0, kpoints=[4, 4, 4],
                      kpoints_gamma_centered=True, occupations="fixed",
                      smearing_eV=0.0, path="GXWKGL", n_bands=8)
dft.bands_check(spec)["by_field"]
run = dft.bands(spec=spec)
summary = run["bands"].value
summary["band_edges"]["gap_eV"], summary["ticks"]
eigenvalues = dft.bands_array(name="eigenvalues").data
distance = dft.bands_array(name="distance").data
dft.bands_state()
```

`source_run` with `source_kind="ground-state"` or `"relaxation"` inherits the
exact geometry and electronic settings of a stored run; a relaxation's symmetry
setting becomes the default ground-state symmetry. Refusals raise `ApiError`;
cancelled, timed-out, failed or unverifiable runs raise `DFTBandsFailed`.
`examples/scripts/16_dft_band_structure.py` is a runnable example.

## HTTP and project storage

The local service exposes `dft/bands/spec`, `dft/bands/run`, `dft/bands/runs`,
`dft/bands/result` and `dft/bands/export`. Results use the
`dftbands::<run_id>::<quantity>` prefix; the arrays `eigenvalues`,
`kpoints_frac`, `kpoints_cartesian` and `distance` use the same prefix in the
checksummed chunk store.

## Live validation

`tests/validation/test_dft_bands_live.py` runs against GPAW 25.7.0 and its PAW
datasets with a deliberately light setup (PBE, 300 eV plane waves, 4x4x4
Gamma-centred grid). See [VALIDATION.md](VALIDATION.md) for the cases,
tolerances and the values measured here. The cases skip with the reason
`BLOCKED` when GPAW or its datasets are unavailable; a skip is not a pass.

## Limits

* Kohn-Sham bands of a semilocal functional are not quasiparticle bands. PBE
  underestimates band gaps; the silicon gap here is about 0.55 eV against a
  measured 1.17 eV. No experimental agreement is claimed, and the record says
  so explicitly.
* Band edges and gaps are extremes over the sampled path points. The true
  extremes can lie between points or off the path.
* Bands are sorted by energy at each k-point. Crossings are not disentangled by
  symmetry, so a plotted line of one band index can change character at a
  crossing.
* No spin-orbit coupling, no non-collinear magnetism, no hybrid functionals.
* The energy zero of a slab or wire is its Fermi level, not the vacuum level.
* **Orbital character is deferred.** GPAW's PAW projector weights are
  projections onto bound partial waves inside the augmentation spheres. They
  are not normalised orbital populations, they do not sum to one, and the
  interstitial weight belongs to no orbital. Fat bands built from them would
  need their own normalisation convention and validation, which this slice
  does not provide, so none are offered rather than shipping approximate ones.
* The band structure converges its own ground state; it does not load a stored
  wavefunction. With a stored source, the fresh ground state is compared with
  the source's energy and a difference above 1 meV is warned about.
