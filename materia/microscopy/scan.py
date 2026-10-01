"""Scan results: channels, feature detection and honest feature attribution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..elements import periodic_table as pt
from ..provenance import Convergence, Fidelity, Origin, Provenance, Result

FEATURE_KINDS = (
    "atomic-site",
    "electronic-feature",
    "adsorbate",
    "dopant-site",
    "defect",
    "uncertain",
)


@dataclass
class Feature:
    """A detected maximum in a scan channel, with its attribution."""

    x_A: float
    y_A: float
    value: float
    kind: str
    nearest_atom_id: Optional[int]
    nearest_element: Optional[str]
    lateral_offset_A: float
    confidence: float
    rationale: str

    def as_dict(self) -> dict:
        return {
            "x_A": self.x_A, "y_A": self.y_A, "value": self.value, "kind": self.kind,
            "nearest_atom_id": self.nearest_atom_id, "nearest_element": self.nearest_element,
            "lateral_offset_A": self.lateral_offset_A, "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass
class ScanResult:
    """A simulated scanning-probe measurement."""

    technique: str
    mode: str
    channels: Dict[str, np.ndarray]
    primary_channel: str
    units: Dict[str, str]
    extent_A: Tuple[float, float, float, float]
    resolution: Tuple[int, int]
    structure: Optional[Structure]
    provenance: Provenance
    convergence: Optional[Convergence] = None
    noise_record: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0
    electronic: Any = None
    unsupported_reason: str = ""
    suggested_models: List[str] = field(default_factory=list)
    _features: Optional[List[Feature]] = None
    _lattice_cache: Optional[tuple] = None
    _border_rejected: int = 0

    @staticmethod
    def unsupported_result(model: str, reason: str, settings: dict,
                           suggested: Optional[List[str]] = None) -> "ScanResult":
        return ScanResult(
            technique="STM", mode=settings.get("mode", ""), channels={},
            primary_channel="", units={}, extent_A=(0, 0, 0, 0), resolution=(0, 0),
            structure=None,
            provenance=Provenance(model=model, fidelity=Fidelity.NON_PHYSICAL,
                                  origin=Origin.UNSUPPORTED, notes=reason),
            settings=settings, unsupported_reason=reason,
            suggested_models=list(suggested or []),
        )

    @property
    def supported(self) -> bool:
        return not self.unsupported_reason

    def channel(self, name: Optional[str] = None) -> np.ndarray:
        key = name or self.primary_channel
        if key not in self.channels:
            raise KeyError(
                f"No channel {key!r} in this {self.technique} scan. "
                f"Available: {sorted(self.channels)}"
            )
        return self.channels[key]

    def channel_names(self) -> List[str]:
        return sorted(self.channels)

    def statistics(self, name: Optional[str] = None) -> dict:
        a = self.channel(name)
        return {
            "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std()),
            "peak_to_peak": float(np.ptp(a)),
            "unit": self.units.get(name or self.primary_channel, ""),
        }

    def pixel_size_A(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.extent_A
        nx, ny = self.resolution
        return ((x1 - x0) / max(nx - 1, 1), (y1 - y0) / max(ny - 1, 1))

    def pixel_to_xy(self, col: float, row: float) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.extent_A
        nx, ny = self.resolution
        return (x0 + (x1 - x0) * col / max(nx - 1, 1),
                y0 + (y1 - y0) * row / max(ny - 1, 1))

    def xy_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.extent_A
        nx, ny = self.resolution
        return ((x - x0) / max(x1 - x0, 1e-12) * max(nx - 1, 1),
                (y - y0) / max(y1 - y0, 1e-12) * max(ny - 1, 1))

    def filtered(self, name: Optional[str] = None, method: str = "plane+median",
                 median_size: int = 3) -> np.ndarray:
        """Standard SPM post-processing, applied on request only.

        ``plane`` subtracts a least-squares plane (removes sample tilt and
        linear drift), ``line`` subtracts the median of each scan line
        (removes line offsets), ``median`` applies a small median filter.
        The raw channel is always retained unmodified.
        """
        a = np.array(self.channel(name), dtype=float)
        steps = method.split("+")
        for step in steps:
            if step == "plane":
                ny, nx = a.shape
                yy, xx = np.mgrid[0:ny, 0:nx]
                A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(a.size)])
                coef, *_ = np.linalg.lstsq(A, a.ravel(), rcond=None)
                a = a - (A @ coef).reshape(a.shape)
            elif step == "line":
                a = a - np.median(a, axis=1, keepdims=True)
            elif step == "median":
                from scipy.ndimage import median_filter
                a = median_filter(a, size=median_size, mode="nearest")
            elif step == "none":
                pass
            else:
                raise ValueError(
                    f"Unknown filter step {step!r}. Available: plane, line, median, none."
                )
        return a

    def detect_features(self, name: Optional[str] = None, min_prominence_frac: float = 0.15,
                        neighbourhood: int = 3, force: bool = False) -> List[Feature]:
        """Find local maxima and attribute them, with explicit confidence."""
        if self._features is not None and not force:
            return self._features
        from scipy.ndimage import maximum_filter

        a = self.filtered(name, "plane")
        span = float(np.ptp(a)) or 1.0
        mx = maximum_filter(a, size=2 * neighbourhood + 1, mode="nearest")
        peaks = (a == mx) & (a >= a.min() + min_prominence_frac * span)

        border = np.zeros_like(peaks)
        border[:neighbourhood, :] = True
        border[-neighbourhood:, :] = True
        border[:, :neighbourhood] = True
        border[:, -neighbourhood:] = True
        self._border_rejected = int((peaks & border).sum())
        peaks &= ~border

        rows, cols = np.nonzero(peaks)
        feats: List[Feature] = []
        for r, c in zip(rows, cols):
            x, y = self.pixel_to_xy(float(c), float(r))
            feats.append(self._attribute(x, y, float(a[r, c])))
        feats.sort(key=lambda f: -f.value)
        self._features = feats
        return feats

    def _lateral_distance(self, x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Minimum-image lateral distances from (x, y) to every atom."""
        s = self.structure
        assert s is not None
        d = s.positions[:, :2] - np.array([x, y])
        if s.cell.is_periodic:
            d3 = np.column_stack([d, np.zeros(len(d))])
            d3 = s.cell.minimum_image(d3)
            d = d3[:, :2]
        return np.linalg.norm(d, axis=1), d

    def _surface_lattice(self):
        """Top-layer atom indices and the characteristic in-plane site spacing.

        Only the topmost atomic layer is considered: a probe sees it, and the
        layer beneath a Si(111) bilayer sits directly below its partner, so
        including it would make every site look like a degenerate pair.
        """
        if getattr(self, "_lattice_cache", None) is not None:
            return self._lattice_cache
        s = self.structure
        z = s.positions[:, 2]
        top = z.max()
        idx = np.nonzero(z >= top - 0.6)[0]
        spacing = 3.0
        if idx.size > 1:
            xy = s.positions[idx][:, :2]
            d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            d[d < 0.3] = np.inf
            nearest = d.min(axis=1)
            finite = nearest[np.isfinite(nearest)]
            if finite.size:
                spacing = float(np.median(finite))
        self._lattice_cache = (idx, spacing)
        return self._lattice_cache

    def _attribute(self, x: float, y: float, value: float) -> Feature:
        s = self.structure
        if s is None or len(s) == 0:
            return Feature(x, y, value, "uncertain", None, None, float("nan"), 0.0,
                           "No structure is associated with this scan.")
        surface_idx, spacing = self._surface_lattice()
        r, _ = self._lateral_distance(x, y)
        r_masked = np.full(len(s), np.inf)
        r_masked[surface_idx] = r[surface_idx]
        k = int(np.argmin(r_masked))
        d0 = float(r_masked[k])
        aid = int(s.ids[k])
        element = pt.symbol(int(s.numbers[k]))
        role = str(s.roles[k])

        order = np.argsort(r_masked)
        d1 = float(r_masked[order[1]]) if len(order) > 1 else float("inf")

        if role in ("adatom", "adsorbate"):
            kind = "adsorbate"
            conf = float(np.clip(0.9 - d0 / spacing, 0.3, 0.9))
            why = (f"The nearest surface species is a {element} {role} "
                   f"{d0:.2f} A away in the plane.")
        elif role == "dopant":
            kind = "dopant-site"
            conf = float(np.clip(0.85 - d0 / spacing, 0.3, 0.85))
            why = (f"The nearest surface site is a substitutional {element} dopant "
                   f"{d0:.2f} A away in the plane.")
        elif role == "vacancy-neighbour":
            kind, conf = "defect", 0.6
            why = (f"The nearest site borders a vacancy; the maximum is {d0:.2f} A "
                   f"from {element} #{aid}.")
        elif d0 <= 0.30 * spacing:
            kind = "atomic-site"
            conf = float(np.clip(1.0 - d0 / (0.45 * spacing), 0.35, 0.97))
            why = (f"The maximum lies {d0:.2f} A from {element} #{aid}, well inside "
                   f"the {spacing:.2f} A spacing between surface sites.")
        elif abs(d0 - d1) <= 0.20 * spacing and d0 <= 0.75 * spacing:
            kind, conf = "electronic-feature", 0.5
            why = (f"The maximum is nearly equidistant from two surface sites "
                   f"({d0:.2f} A and {d1:.2f} A), which is more consistent with a "
                   "bonding or dangling-bond state than with a single nucleus.")
        elif d0 <= 0.55 * spacing:
            kind = "atomic-site"
            conf = float(np.clip(0.75 - (d0 / spacing - 0.30), 0.3, 0.75))
            why = (f"The maximum is {d0:.2f} A from {element} #{aid}; that is within "
                   f"half the {spacing:.2f} A site spacing, but far enough off centre "
                   "that the attribution is not certain.")
        else:
            kind, conf = "uncertain", 0.3
            why = (f"The maximum is {d0:.2f} A from the nearest surface site "
                   f"({element} #{aid}) with a site spacing of {spacing:.2f} A: too "
                   "far to call it an atomic site and too asymmetric to call it a "
                   "bond state.")

        return Feature(x, y, value, kind, aid, element, d0, conf,
                       why + " This is an inference from the simulated structure and "
                             "the imaging model, not a measurement of the element.")

    def identify_at(self, x: float, y: float) -> dict:
        """Hover query: what is under the tip at (x, y)?"""
        if self.structure is None:
            return {"kind": "uncertain", "confidence": 0.0,
                    "message": "No structure associated with this scan."}
        f = self._attribute(x, y, float("nan"))
        s = self.structure
        out = f.as_dict()
        if f.nearest_atom_id is not None:
            atom = s.atom(f.nearest_atom_id)
            el = atom.element
            iso = el.most_abundant_isotope
            out.update({
                "atom": atom.as_dict(),
                "element_name": el.name,
                "atomic_number": el.number,
                "predicted_isotope": (f"{el.symbol}-{iso.mass_number}" if iso else None),
                "isotope_basis": ("most abundant natural isotope; the simulation does "
                                  "not distinguish isotopes in the imaging signal unless "
                                  "one was assigned explicitly"
                                  if atom.mass_number == 0 else
                                  f"explicitly assigned {el.symbol}-{atom.mass_number}"),
                "charge_state_e": atom.charge,
                "lattice_site": atom.label or "unlabelled",
                "role": atom.role,
                "coordination": atom.coordination,
                "height_above_lowest_A": float(atom.position[2] - s.positions[:, 2].min()),
            })
        px, py = self.xy_to_pixel(x, y)
        nx, ny = self.resolution
        if 0 <= px < nx and 0 <= py < ny and self.primary_channel:
            out["signal"] = float(self.channel()[int(round(py)), int(round(px))])
            out["signal_unit"] = self.units.get(self.primary_channel, "")
        out["caveat"] = (
            "Identification is model-based: the element is read from the simulated "
            "structure and matched to the nearest maximum. A scanning probe measures "
            "the local density of states, which does not identify chemical species on "
            "its own."
        )
        return out

    def as_dict(self, include_arrays: bool = False) -> dict:
        d = {
            "technique": self.technique,
            "mode": self.mode,
            "primary_channel": self.primary_channel,
            "channels": self.channel_names(),
            "units": dict(self.units),
            "extent_A": list(self.extent_A),
            "resolution": list(self.resolution),
            "provenance": self.provenance.as_dict(),
            "convergence": self.convergence.as_dict() if self.convergence else None,
            "noise": self.noise_record,
            "settings": self.settings,
            "wall_time_s": self.wall_time_s,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "suggested_models": list(self.suggested_models),
            "statistics": ({k: self.statistics(k) for k in self.channel_names()}
                           if self.channels else {}),
        }
        if include_arrays:
            d["data"] = {k: v.tolist() for k, v in self.channels.items()}
        return d

    def to_result(self) -> Result:
        return Result(
            name=f"{self.technique.lower()}_scan",
            value=self,
            unit=self.units.get(self.primary_channel, ""),
            provenance=self.provenance,
            convergence=self.convergence,
        )
