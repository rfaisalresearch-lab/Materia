# Surface reconstructions

A clean surface is rarely the plane the crystal was cut on. Atoms at a
truncation carry dangling bonds, and they rearrange to get rid of them. Materia
treats that rearrangement as a first-class object with its own generator,
provenance, measurement and refusal behaviour, separate from the ideal-slab
builder.

## How a reconstruction is produced

Two stages, kept apart on purpose.

**Stage 1, the generator** establishes topology and periodicity from the
symmetry of the truncated slab itself: which sites pair, along which
crystallographic direction, and what the new surface cell is. It imposes
nothing else. For the dimer generator this means the pairing and a starting
partner separation, and that is all.

**Stage 2, the relaxation** finds the geometry, using the interatomic model
the material recommends. Every bond length, height and buckling amplitude
Materia reports for a reconstruction is the output of this relaxation.

The consequence matters: **no reconstructed atomic coordinate shipped with
Materia is a hand-entered number.** Published structural parameters are carried
in the material file, but only so that a computed geometry can be compared
against measurement. They never place an atom.

## What is implemented

| id | material | surface | generator | status |
| --- | --- | --- | --- | --- |
| `2x1-dimer` | silicon | (100) | dimer pairing, relaxed | **available** |
| `relaxed-110` | gallium arsenide | (110) |, | declared, refused |
| `2x4-beta2` | gallium arsenide | (100) |, | declared, refused |
| `herringbone-22xsqrt3` | gold | (111) |, | declared, refused |
| `ripples` | graphene | (001) |, | declared, refused |
| `7x7-DAS` | silicon | (111) |, | declared, refused |
| `sqrt3xsqrt3-R30` | silicon carbide 4h | (001) |, | declared, refused |
| `6sqrt3-buffer` | silicon carbide 4h | (001) |, | declared, refused |

The `available` flag the interface shows is resolved against the generator
registry and the material's own prototype, not against the `implemented` key in
the file. A material definition cannot claim a capability the build does not
have. `gallium_arsenide.json` carried `implemented: true` on `relaxed-110`
before this subsystem existed; the flag is now false, because no interatomic
potential shipped with Materia covers GaAs and there is nothing to relax it
with. A test refuses any shipped definition that claims a reconstruction the
build has no generator for.

### Si(100)-(2×1), the dimer reconstruction

Every site of the ideal Si(100) surface carries two dangling bonds. They lie in
a plane whose in-plane trace rotates by 90° from one atomic layer to the next.
Neighbouring sites along that trace can each close one dangling bond by moving
together, which halves the periodicity in that direction and leaves rows of
dimers, the 2×1 pattern reported by Schlier and Farnsworth in 1959.

The generator derives the dimer axis rather than assuming a cube direction: for
a tetrahedral site holding two back-bonds **b₁** and **b₂**, the two missing
orbitals span the plane of −(**b₁**+**b₂**) and **b₁**×**b₂**, so **b₁**×**b₂**
projected into the surface is the axis along which two such sites can pair
without twisting their remaining bonds. Sites are grouped into rows
perpendicular to it and paired along it.

The slab must therefore have an **even** in-plane repeat along the dimer axis.
An odd repeat is refused with a message that says so; it is not silently
rounded.

After relaxation the record carries a symmetry check: the slab must no longer
be invariant under the 1×1 translation it was cut with, and must be invariant
under the doubled one. With four repeats along the axis the second test is a
genuine one rather than the lattice vector itself.

#### What Stillinger-Weber gives, and what it cannot

Relaxing with the recommended Stillinger-Weber potential gives a **symmetric**
dimer with a bond of **2.4035 Å**, converged to 0.005 eV/Å, independent of the
starting separation over 2.2-3.4 Å and of the cell size, and lower in energy
than the relaxed ideal truncation by **1.68 eV per dimer**.

The LEED structure determination of Over *et al.*, Phys. Rev. B **55** (1997)
4731 finds a **buckled** dimer: bond length 2.24 ± 0.08 Å, vertical separation
of the up and down atom 0.72 ± 0.05 Å, tilt 19 ± 2°.

Materia reports both and the deviation between them. It does not claim
agreement:

* The computed bond is **0.16 Å (7.3 %) longer** than the LEED value and lies
  outside the quoted experimental uncertainty.
* The computed buckling is **exactly zero**, against 0.72 Å measured.

The zero is not a bug and is not a tolerance failure. The real dimer buckles
because charge transfers from one dimer atom to the other, a Jahn-Teller
distortion of an electronic degree of freedom. Stillinger-Weber is a
three-body potential of nuclear coordinates only. It has no mechanism that can
produce the buckling, so it returns the symmetric structure, and the
low-temperature c(4×2) and p(2×2) orderings that follow from alternating
buckling are **not reproduced and not approximated**. Asking for them is
answered with an unsupported record, not a guess.

Reproducing the buckling needs an electronic total energy: a Tier-3 adapter.

## Refusal behaviour

Four distinct outcomes, none of which produce an approximate structure:

| situation | outcome |
| --- | --- |
| id not declared by the material | `ReconstructionError` listing what *is* declared |
| declared but no generator in this build | `ReconstructionNotImplemented` with a machine-readable record |
| generator exists but the orientation is wrong | `ReconstructionError` naming both surfaces |
| slab cannot carry it (odd repeat, wrong termination, already reconstructed) | `ReconstructionError` saying what to change |

The unsupported record contains the reason, the literature reference for the
structure that was asked for, the solvers that could produce it, and a
`Result` whose value is `None` and whose origin is `unsupported`. The interface
shows it as a blocked dialog; the Python API raises `UnsupportedRequest`
carrying the same record; the service returns `{"ok": false, "unsupported": …}`
and adds an `unsupported` entry to the warning log. In every path, no structure
is added to the project.

## Provenance

A reconstructed slab carries `structure.info["reconstruction"]`:

```
id                     the reconstruction applied
material, miller       what it was applied to
generator              id, prototypes, periodicity, method, references
pairs                  atom-id pairs, stable across the session
n_dimers, n_rows       what the generator built
dimer_axis, row_axis   derived, not assumed
unreconstructed_spacing_A
initial_separation_A   the construction guess, recorded so it can be audited
relaxation_attempted   whether a relaxation ran at all
relaxed                whether it ran *and* converged
geometry_status        converged | not-converged | unrelaxed
relaxation             model, converged, steps, residual force, energy, wall time
measured               bond length, spread, buckling, contraction, and the
                       geometry status the numbers were taken from
periodicity_check      the 1×1 and 2×1 translation tests
provenance             full Provenance record
```

### Three geometry states, kept apart

A relaxation that runs is not the same as a relaxation that converges, and the
record never conflates them.

| `geometry_status` | origin | fidelity | comparable |
| --- | --- | --- | --- |
| `converged` | `calculated` | `tier1-classical` | yes |
| `not-converged` | `estimated` | `tier1-classical` | **no** |
| `unrelaxed` | `estimated` | `tier0-structural` | **no** |

A run that stops on its step budget, `reconstruction_max_steps` too small,
say, leaves the atoms somewhere between the construction guess and the
minimum. Its provenance states the residual force against the target and says
in plain words that the geometry is **not** an energy minimum and that no
distance measured from it is a prediction. `compare_to_reference` then returns
`comparable: false` with the reason, and every row's
`within_stated_uncertainty` is `None` rather than a boolean, so no caller can
read a half-relaxed bond length as a model result that was checked against
experiment. The measured numbers are still shown, hiding them would be its own
distortion, but as evidence about the run, not as a result. The interface
renders the same thing: a blocked note above a table whose verdict column reads
*not comparable*.

`relaxed` therefore means converged, and only a converged geometry is labelled
`calculated`. The ideal-truncation statement the slab builder wrote into its
own provenance is replaced when a reconstruction is applied, so a reconstructed
slab never carries the claim that no reconstruction was applied, nor the claim
that it was relaxed when the relaxation stalled.

The whole record round-trips through the project format.

### History and undo

Reconstructing an existing slab is an editing operation like any other. It
records exactly one undo point of its own, labelled for what it did, and undo
puts the ideal truncation back, coordinates, roles, and the `reconstruction`
and `surface` metadata together. A refused request records nothing: the undo
point is committed after the change has been made, not before it is attempted,
so a refusal cannot leave a phantom entry behind. Building a surface produces
one project entry, never two sharing one object.

Where the undo point lands is decided by what the project actually holds, by
object identity, not by which structure happens to be active:

| the handle's structure is | undo point | undo restores |
| --- | --- | --- |
| the active one | recorded | that structure, in place |
| held by the project but not active | recorded, bound to **its own slot** | that structure; the active one is untouched and stays active |
| detached (`activate=False`, never registered) | **none** | nothing, the project does not own the object |

Snapshots carry the slot they were taken from, so an undo can never write one
structure's geometry into another's slot. A detached handle can still be
reconstructed, it is the caller's own object, but it does not get to write
into a project history that knows nothing about it.

### Failure is transactional

A reconstruction either completes or leaves no trace. The relaxation controls
are validated before anything moves, so a bad `steps` or `fmax` is refused
while the slab is still untouched. Past that point the whole operation runs
under a restore point: if the generator fails after it has moved atoms, if the
solver fails part-way through a minimisation, or if anything fails while the
records are being written, positions, roles, labels, fixed flags, bonds, cell
and the whole of `info` are put back as they were, and no history entry is
made. `Structure.restore_from` does the restoring in place, so every reference
the caller still holds sees the rolled-back state. A slab that failed this way
can be reconstructed again immediately.

## Using it

### Interface

Material library → pick a material → **Build** → choose a reconstruction from
the dropdown (unavailable ones are listed and disabled with the reason) → set
`fixed layers` → **Build surface**. The measured geometry appears in the
Reconstruction section of the properties panel, and the comparison against
published data opens automatically.

On an existing slab: **Structure ▸ Reconstruct surface…**, or **Structure ▸
Reconstruction report** for the comparison table at any time.

### Python

```python
from materia.python_api import Lab, UnsupportedRequest

lab = Lab()
silicon = lab.materials.load("silicon")

for r in silicon.reconstructions():
    print(r["id"], r["supported"], r.get("reason", ""))

surface = silicon.create_surface(size=(4, 2, 8), orientation=(1, 0, 0),
                                 vacuum_angstrom=12.0, fix_bottom_layers=4,
                                 reconstruction="2x1-dimer")
record = surface.reconstruction()
print(record["measured"]["bond_length_A"], record["relaxation"]["model"])

for row in surface.compare_reconstruction()["rows"]:
    print(row["quantity"], row["measured"], row["reference"],
          row["within_stated_uncertainty"])

try:
    silicon.create_surface(orientation=(1, 1, 1), reconstruction="7x7-DAS")
except UnsupportedRequest as exc:
    print(exc.as_dict()["reason"])
```

`make_surface(..., reconstruction=...)` and
`apply_reconstruction(slab, material, id, ...)` are the lower-level entry
points; `relax_reconstruction=False` returns the generator's topology with the
geometry explicitly labelled estimated, and `reconstruction_model`,
`reconstruction_fmax_eV_A` and `reconstruction_max_steps` control the
relaxation that otherwise follows.

```python
surface.reconstruct("2x1-dimer")                  # relaxes by default
surface.reconstruct("2x1-dimer", relax=False)     # topology only, labelled estimated
surface.reconstruct("2x1-dimer", fmax=0.002, steps=2000)
```

A refused request raises and changes nothing, not the structure, not the
project history.

## Adding a reconstruction

1. Write a generator `fn(slab, material, options) -> dict` that alters `slab`
   in place and returns what it did. Derive directions from the slab; do not
   assume a cube axis. Refuse a slab that cannot carry the reconstruction with
   a message naming the fix.
2. `register(GeneratorInfo(...), fn)` in
   `materia/structure_builder/reconstruction.py`, listing the prototypes and
   orientation it applies to and citing the structure.
3. Declare it in the material file with `implemented: true`, a `method` line,
   and a `reference_geometry` block: each quantity needs a `value`, a `source`
   citation and, wherever the source states one, an `uncertainty`. Validation
   rejects a block without a value or a source.
4. Add software tests for the refusal paths and the records, and separate
   physical-validation cases for the geometry. Keep them apart: a test that the
   record serialises is not evidence about silicon.
5. If the recommended model cannot reproduce a feature of the real structure,
   say so in the material's `note` and in this file, and make the program
   report it rather than approximate it.

## Related

* `docs/SCIENTIFIC_MODELS.md`, the interatomic models used for the relaxation
* `docs/FIDELITY_TIERS.md`, what `tier0`/`tier1` mean
* `docs/VALIDATION.md`, the physical cases run against this generator
* `docs/LIMITATIONS.md`, what is still declared and refused
