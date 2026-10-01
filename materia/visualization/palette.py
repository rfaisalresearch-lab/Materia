"""Colour maps for scanning-probe data.

The default is ``silver``: a dark-charcoal-to-silver-white ramp that matches
the appearance of published grayscale STM and AFM micrographs.  Rainbow maps
are available but are never the default, because they introduce visual
features that are not in the data.  Every palette carries a printable scale
and a perceptual note.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

Stop = Tuple[float, Tuple[int, int, int]]


@dataclass(frozen=True)
class Palette:
    id: str
    name: str
    stops: Tuple[Stop, ...]
    perceptual: str
    note: str = ""
    default_for: Tuple[str, ...] = ()

    def lut(self, n: int = 256) -> np.ndarray:
        """``(n, 3)`` uint8 lookup table."""
        xs = np.array([s[0] for s in self.stops], dtype=float)
        cols = np.array([s[1] for s in self.stops], dtype=float)
        t = np.linspace(0.0, 1.0, n)
        out = np.empty((n, 3))
        for c in range(3):
            out[:, c] = np.interp(t, xs, cols[:, c])
        return np.clip(np.round(out), 0, 255).astype(np.uint8)

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "perceptual": self.perceptual,
                "note": self.note,
                "stops": [[s[0], list(s[1])] for s in self.stops]}


PALETTES: Dict[str, Palette] = {}


def _add(p: Palette) -> None:
    PALETTES[p.id] = p


_add(Palette(
    id="silver",
    name="Silver (default)",
    stops=((0.00, (11, 12, 14)), (0.18, (38, 40, 44)), (0.42, (94, 98, 104)),
           (0.68, (158, 162, 168)), (0.86, (206, 210, 216)), (1.00, (244, 246, 250))),
    perceptual="monotonic luminance",
    note="Matches the appearance of published grayscale scanning-probe micrographs.",
    default_for=("STM", "AFM"),
))
_add(Palette(
    id="gray",
    name="Neutral gray",
    stops=((0.0, (0, 0, 0)), (1.0, (255, 255, 255))),
    perceptual="monotonic luminance, linear",
    note="Strictly linear luminance; use when comparing against printed data.",
))
_add(Palette(
    id="gold",
    name="Gold (classic STM)",
    stops=((0.00, (12, 6, 0)), (0.30, (92, 42, 6)), (0.62, (196, 118, 28)),
           (0.85, (240, 190, 96)), (1.00, (255, 242, 200))),
    perceptual="monotonic luminance, warm hue ramp",
    note="The conventional orange STM look. Luminance is monotonic, so features "
         "are not created by the colour map.",
))
_add(Palette(
    id="copper",
    name="Copper",
    stops=((0.0, (8, 4, 2)), (0.5, (150, 82, 50)), (1.0, (255, 218, 185))),
    perceptual="monotonic luminance",
))
_add(Palette(
    id="viridis",
    name="Viridis",
    stops=((0.00, (68, 1, 84)), (0.25, (59, 82, 139)), (0.50, (33, 145, 140)),
           (0.75, (94, 201, 98)), (1.00, (253, 231, 37))),
    perceptual="monotonic luminance, colour-vision-deficiency safe",
    note="Use when a colour scale is needed; always display the scale bar.",
))
_add(Palette(
    id="blue-white-red",
    name="Blue-white-red (diverging)",
    stops=((0.0, (33, 66, 160)), (0.5, (247, 247, 247)), (1.0, (178, 34, 34))),
    perceptual="diverging; NOT monotonic in luminance",
    note="For signed quantities only (e.g. differential images). Never use it "
         "for topography: the white mid-point invents a feature at zero.",
))
_add(Palette(
    id="high-contrast",
    name="High contrast (accessibility)",
    stops=((0.0, (0, 0, 0)), (0.5, (255, 255, 0)), (1.0, (255, 255, 255))),
    perceptual="monotonic luminance, maximum contrast",
    note="Accessibility option for low-vision users.",
))


def get(palette_id: str) -> Palette:
    if palette_id not in PALETTES:
        raise KeyError(f"Unknown palette {palette_id!r}. Available: {sorted(PALETTES)}")
    return PALETTES[palette_id]


def list_palettes() -> List[dict]:
    return [p.as_dict() for p in PALETTES.values()]


def normalize(data: np.ndarray, mode: str = "minmax",
              percentile: Tuple[float, float] = (0.5, 99.5),
              vmin: float = None, vmax: float = None) -> Tuple[np.ndarray, float, float]:
    """Map data to 0..1 and return the limits actually used."""
    a = np.asarray(data, dtype=float)
    if vmin is None or vmax is None:
        if mode == "minmax":
            lo, hi = float(a.min()), float(a.max())
        elif mode == "percentile":
            lo, hi = (float(np.percentile(a, percentile[0])),
                      float(np.percentile(a, percentile[1])))
        elif mode == "sigma":
            m, sd = float(a.mean()), float(a.std())
            lo, hi = m - 2.5 * sd, m + 2.5 * sd
        else:
            raise ValueError(f"Unknown normalisation mode {mode!r}")
        lo = vmin if vmin is not None else lo
        hi = vmax if vmax is not None else hi
    else:
        lo, hi = float(vmin), float(vmax)
    if hi <= lo:
        hi = lo + 1e-12
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0), lo, hi


def apply_palette(data: np.ndarray, palette_id: str = "silver", *,
                  mode: str = "percentile", gamma: float = 1.0,
                  contrast: float = 1.0, brightness: float = 0.0,
                  vmin: float = None, vmax: float = None,
                  invert: bool = False) -> Tuple[np.ndarray, dict]:
    """Render a 2-D array to an ``(ny, nx, 3)`` uint8 RGB image."""
    norm, lo, hi = normalize(data, mode=mode, vmin=vmin, vmax=vmax)
    if invert:
        norm = 1.0 - norm
    if gamma != 1.0:
        norm = np.power(norm, max(gamma, 1e-6))
    if contrast != 1.0 or brightness:
        norm = np.clip((norm - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    lut = get(palette_id).lut(256)
    idx = np.clip(np.round(norm * 255), 0, 255).astype(np.int32)
    rgb = lut[idx]
    meta = {"palette": palette_id, "vmin": lo, "vmax": hi, "mode": mode,
            "gamma": gamma, "contrast": contrast, "brightness": brightness,
            "inverted": invert}
    return rgb, meta
