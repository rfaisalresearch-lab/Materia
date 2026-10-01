# Figures

Every image in this directory is the program's own output, produced by
`python tools/make_examples.py`. None is an illustration, a mock-up or a
retouched capture. Each PNG carries the model, the channel, the palette limits
and the provenance summary in its text chunks:

```bash
python - <<'PY'
import zlib, struct
data = open("docs/screenshots/si111_stm.png", "rb").read()
i = 8
while i < len(data):
    length, tag = struct.unpack(">I4s", data[i:i+8])
    if tag == b"tEXt":
        key, _, value = data[i+8:i+8+length].partition(b"\x00")
        print(key.decode(), "=", value.decode()[:120])
    i += 12 + length
PY
```

## Scanning tunnelling microscopy

### `si111_stm.png`, Si(111), constant current, +1.0 V

The default view. A greyscale measurement with the full instrumental noise
chain: white detector noise, 1/f along the fast-scan direction, per-line offset
and gain, thermal drift, piezo creep and a vibration ripple, all from seed 42.
Bright maxima sit on the surface dangling bonds of the ideal truncation. No
element labels, no coloured spheres: the identity appears when you hover.

### `si111_stm_noiseless.png`, the same scan with the noise chain off

What the physical model produces before the instrument touches it. The
difference between this and the image above is the measurement chain, and the
program always keeps both channels.

### `si111_stm_filled_states.png`, Si(111) at −1.2 V

Negative bias probes filled states. The corrugation is several times larger
than at positive bias because the filled dangling-bond states are more strongly
localised. Nothing about the atoms changed between the two images; the bias
window selected different states.

### `si111_stm_current.png`, the current channel, gold palette

The same acquisition shown as the calibrated current rather than the feedback
height, in the conventional orange scanning-probe palette. The palette is
labelled and its limits are shown, because a colour map that invents contrast
is a way of lying.

### `si111_phosphorus.png`, a substitutional phosphorus donor

A subsurface P donor after relaxation. The donor site carries excess electron
density in the tight-binding solution and appears brighter at positive bias.
The magnitude is *not* quantitative: the impurity is an on-site energy shift
from free-atom term values, and the model does not contain the physics that
sets a shallow-donor binding energy. The inspector says so.

### `si111_vacancy.png`, a surface vacancy

A missing surface atom after relaxation. Hovering over the vacancy reports
`uncertain` with low confidence, because there is no atom there and the program
does not pretend otherwise.

### `graphene_stm.png`, graphene, +0.5 V

The honeycomb lattice through the π-band model. Both sublattices are equivalent
in a free-standing sheet, so the image shows the full honeycomb rather than the
triangular pattern a substrate-broken sheet gives.

### `si100_2x1_stm.png`, Si(100)-(2×1), +1.0 V

The dimer reconstruction, generated rather than stored. Bright rows run
vertically at 7.68 Å spacing, twice the 3.840 Å 1×1 surface lattice constant,
which is the 2× of 2×1, separated by dark trenches, with the individual dimers
resolved along each row at the 1× repeat. Nothing in this image was placed by
hand: the generator paired the surface sites along the direction it derived
from their back-bonds, Stillinger-Weber relaxation set the 2.4035 Å bond, and
the Tersoff-Hamann model produced the contrast.

The dimers here are **symmetric**. The real surface buckles, and the figure
does not, for the reason given in [RECONSTRUCTIONS.md](RECONSTRUCTIONS.md): the
potential that set the geometry has no electronic degrees of freedom to buckle
with. The row *periodicity* is a property of the topology and is correct; the
intra-dimer asymmetry is a property of the electrons and is absent.

### `gaas110_stm.png`, GaAs(110), −2.0 V

The non-polar cleavage plane, the standard STM surface for GaAs. At negative
bias the filled states are localised on the arsenic sublattice, so only one of
the two species is bright. This is the textbook demonstration that a
scanning-probe image is a map of the local density of states and not a map of
the atoms.

## Atomic force microscopy

### `si111_afm.png`, frequency-shift image, qPlus parameters

Constant-height frequency shift with a tungsten apex at 4.2 Å, f₀ = 30 kHz,
k = 1800 N/m, amplitude 1 Å. Displayed inverted so that a more negative
detuning reads bright. The contrast comes from a Lennard-Jones plus Hamaker
force model with no covalent tip-sample bonding, which is the dominant
mechanism in real atomic-resolution AFM, the figure is correct for the model
and the model is stated.

## Interface verification

Full-window interface captures are not committed in this release. The previous
draft of this document named four interface PNG files that did not exist, so
those claims have been removed. The interface is instead checked live through
**Help ▸ Run interface self-check**, alongside the HTTP workflow and acceptance
tests. Measurement figures above remain real program output with embedded
provenance metadata.
