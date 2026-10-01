# DFT local density of states and STM images

Materia computes the spatially resolved local density of states (LDOS) with
GPAW and takes Tersoff-Hamann STM images from it, in
`materia/experiments/dft/ldos.py`. GPAW computes the wavefunctions; Materia
freezes the request, checks what GPAW reports, stores the verified map and
keeps its provenance. The image is the standard Tersoff-Hamann quantity; it is
not a prediction of absolute currents.

## What is computed

For an energy window [E_min, E_max] relative to the self-consistent Fermi
level,

    n(r) = (2 / n_spins) sum_s sum_k w_k sum_n [E_min < e_nks - E_F < E_max] |psi~_nks(r)|^2

in states per A^3 for both spins, on GPAW's coarse real-space grid. The window
is sharp and open: a state counts fully or not at all. A negative sample bias V
images the filled states from V to 0; a positive one the empty states from 0 to
V.

The worker converges the ground state, then runs GPAW's `fixed_density` on the
ground state's own k-point grid with point-group symmetry off and time
reversal kept, with the requested bands converged and a few unconverged buffer
bands above them that never enter the map. Point-group symmetry is off because
the irreducible k-points alone give a map that has lost the surface's
symmetry; time reversal leaves every |psi|^2 unchanged.

### Pseudo-wavefunctions

The map is built from GPAW's pseudo-wavefunctions. They equal the all-electron
wavefunctions outside the PAW augmentation spheres, whose radii are recorded
(carbon 0.64 A, for example), and differ inside them. The map is therefore
quantitative in the vacuum, where STM tips are, and not inside the atoms. Its
integral over the cell is the pseudo-norm of the states, close to but not equal
to the number of states: 0.951 of it for graphene. For hydrogen, which has no
core, a window covering every occupied state reproduces GPAW's pseudo valence
density; for heavier atoms GPAW's pseudo density also contains a smooth
pseudo-core charge that is not a Kohn-Sham state.

## Frozen specification

`LDOSSpec` uses schema `materia.dft.ldos`, version `1.0`: the ground state, the
source (structure, stored ground state or stored relaxation, with
fingerprints), `energy_min_eV`, `energy_max_eV`, `spin_channels` and
`n_bands`. Refused: windows narrower than 0.01 eV or reaching farther than
20 eV from E_F, resolved spin without spin polarisation, fewer bands than
occupied states, and grids too large to store.

## Verification

Nothing is stored unless:

* every ground-state and path-step parameter, including the symmetry setting,
  is echoed back unchanged, and the window and spin the worker used are the
  specified ones
* every requested band converged, the Fermi level did not move, and the
  converged bands reach above the window at every k-point
* the k-point weights sum to one, every eigenvalue and map value is finite and
  the map is nowhere negative
* the state count the worker reports equals the one recomputed from the
  returned eigenvalues and weights
* the map integrates to between 0.5 and 1.5 of the states in the window, the
  range of pseudo-norms
* the window holds at least one state; a k-point grid that misses the states
  near the window (graphene's K point on a 4x4 grid) is refused instead of
  stored as a zero map
* spin channels add up to the total, and every element has a recorded
  augmentation radius

## STM images

Images need a slab periodic along a and b with vacuum along c, c perpendicular
to both. A **constant-height** image is the map interpolated linearly along c
at a height above the topmost atom; a **constant-current** image is the height
at which the map equals a chosen value, searched from the vacuum downwards, as
ASE's `STM.scan` does. Heights are allowed only between max(2 A, largest PAW
radius + 1 A) above the topmost atom and 2 A below the cell face, where the
real-space boundary condition forces the wavefunctions to zero. A value that is
not reached inside that band in every column is refused. No conversion to
amperes is made: Tersoff-Hamann gives a proportionality, and an empirical
conversion would add a number Materia cannot verify.

## Interface, Python, HTTP and storage

The DFT panel's Local density of states and STM image group sets the source,
the window (with filled-state and empty-state presets), spin and bands. The
result shows the planar average along c on a logarithmic scale, a viewer for
any plane of the map, the constant-height and constant-current image tool,
Cube export of the map, CSV export of an image and a provenance inspector.

```python
spec = dft.ldos_spec(slab, kpoints=[6, 6, 1], energy_min_eV=-1.0, energy_max_eV=0.0)
results = dft.ldos(spec=spec)
dft.ldos_array(name="ldos").data            # states/A^3 on the coarse grid
dft.stm_image(mode="constant-height", height_A=3.0)["values"]
dft.stm_image(mode="constant-current", isovalue=1e-5)["values"]   # A above the surface
dft.ldos_state()
```

Routes: `dft/ldos/spec`, `dft/ldos/run`, `dft/ldos/runs`, `dft/ldos/result`,
`dft/ldos/slice`, `dft/ldos/image`, `dft/ldos/export`, `dft/ldos/image/export`.
Results use `dftldos::<run_id>::ldos`; the arrays `ldos`, optional `ldos_spin`,
`eigenvalues` and `kpoint_weights` are checksummed in the array store, and a
record whose arrays no longer match is reported as corrupt and not used.
`examples/scripts/18_dft_ldos_stm.py` is a runnable example.

## Validation

`tests/validation/test_dft_ldos_live.py` compares the graphene map, its
constant-current heights and constant-height values with ASE's `STM` class run
on the same job in a separate GPAW process (bit-identical here), checks that
a full window reproduces GPAW's pseudo density for H2, that a spin-polarised
hydrogen atom puts its one state in spin up, and that an LDOS of a real
relaxation is current once applied. See `VALIDATION.md`.

## Limits

* Tersoff-Hamann: an s-wave tip, no tip electronic structure, no tip-sample
  interaction, no bias-dependent barrier.
* Pseudo-wavefunctions: the map is not the all-electron LDOS inside the PAW
  spheres.
* A sharp window over discrete k-points: the map is a sum over the sampled
  states, not a smooth energy integral; refine the k-points for metals.
* Kohn-Sham states of a semilocal functional, not quasiparticles.
* Orbital character (fat bands) is not offered; PAW projector weights are not
  normalised orbital populations.
