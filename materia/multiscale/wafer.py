"""Hierarchical wafer model.

A 300 mm silicon wafer contains of order 1e24 atoms.  No program can hold
that, and Materia does not pretend to.  Instead the wafer is represented
*procedurally*: a small set of parameters plus seeded noise functions define
grains, surface roughness, terraces, doping and defect densities everywhere on
the wafer, and an explicit atomistic model is instantiated only for the region
the user asks for.

The procedural functions are deterministic in ``(wafer.seed, position)``, so
the same wafer coordinate always yields the same microstructure and the same
atomistic region -- across sessions and across machines.  That is what makes a
saved project reproducible without storing the atoms.

Honesty notes
-------------
* The procedural microstructure is **synthetic**.  It is a plausible,
  parameterised stand-in for a real wafer's statistics, not a measurement of
  one.  Everything it produces carries ``origin="estimated"``.
* Dopant placement is a Poisson process at the specified concentration.  It
  reproduces the correct *average* density and the correct statistics of
  fluctuations, not the actual positions in any real wafer.
* Grain structure is a Voronoi tessellation of seeded points.  Real
  polycrystalline microstructure is not Voronoi, and single-crystal wafers have
  no grains at all -- the grain model is disabled for ``polycrystalline=False``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..materials.schema import MaterialDefinition
from ..provenance import Fidelity, Origin, Provenance, Result

STANDARD_WAFERS = {
    50.8: 279.0, 76.2: 381.0, 100.0: 525.0, 125.0: 625.0,
    150.0: 675.0, 200.0: 725.0, 300.0: 775.0, 450.0: 925.0,
}

SCALE_LEVELS = (
    ("wafer", 1.0e8, "mm", "Whole wafer: diameter, flat/notch, layer stack, device regions."),
    ("microstructure", 1.0e5, "um", "Grains, boundaries, roughness, patterned regions."),
    ("nanoscale", 1.0e3, "nm", "Steps, terraces, islands, voids, local strain."),
    ("lattice", 1.5e1, "A", "Unit cells, lattice sites, dopants, vacancies."),
    ("atomic", 5.0e0, "A", "Individual atoms, bonds, orbitals, probe signals."),
)


@dataclass
class Layer:
    """One layer of a wafer's film stack."""

    name: str
    material_id: str
    thickness_nm: float
    role: str = "film"
    crystalline: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class DeviceRegion:
    """A patterned region on the wafer surface, in wafer coordinates (mm)."""

    name: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    kind: str = "die"

    def contains(self, x_mm: float, y_mm: float) -> bool:
        return (abs(x_mm - self.x_mm) <= self.width_mm / 2
                and abs(y_mm - self.y_mm) <= self.height_mm / 2)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class WaferSpec:
    """Everything that defines a wafer, apart from the atoms."""

    material_id: str = "silicon"
    diameter_mm: float = 300.0
    thickness_um: Optional[float] = None
    orientation: Tuple[int, int, int] = (1, 0, 0)
    miscut_deg: float = 0.0
    miscut_azimuth_deg: float = 0.0
    edge_feature: str = "notch"
    dopant: Optional[str] = None
    dopant_concentration_cm3: float = 0.0
    dopant_type: str = ""
    temperature_K: float = 300.0
    roughness_rms_A: float = 0.8
    roughness_correlation_nm: float = 40.0
    polycrystalline: bool = False
    mean_grain_size_um: float = 50.0
    vacancy_density_cm2: float = 1.0e12
    adatom_density_cm2: float = 0.0
    contamination_density_cm2: float = 0.0
    contaminant: str = "O"
    layers: List[Layer] = field(default_factory=list)
    device_regions: List[DeviceRegion] = field(default_factory=list)
    termination: str = "ideal-bulk"
    seed: int = 20260920

    def __post_init__(self) -> None:
        if self.thickness_um is None:
            nearest = min(STANDARD_WAFERS, key=lambda d: abs(d - self.diameter_mm))
            self.thickness_um = (STANDARD_WAFERS[nearest]
                                 if abs(nearest - self.diameter_mm) < 1e-6
                                 else round(0.0026 * self.diameter_mm * 1000, 1))
        self.orientation = tuple(int(v) for v in self.orientation)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("layers", "device_regions")}
        d["orientation"] = list(self.orientation)
        d["layers"] = [l.as_dict() for l in self.layers]
        d["device_regions"] = [r.as_dict() for r in self.device_regions]
        return d

    @staticmethod
    def from_dict(d: dict) -> "WaferSpec":
        d = dict(d)
        layers = [Layer(**l) for l in d.pop("layers", [])]
        regions = [DeviceRegion(**r) for r in d.pop("device_regions", [])]
        spec = WaferSpec(**d)
        spec.layers = layers
        spec.device_regions = regions
        return spec


def _hash_seed(*parts) -> int:
    blob = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


class Wafer:
    """A procedurally-defined wafer, with deterministic local sampling."""

    def __init__(self, spec: WaferSpec, material: MaterialDefinition) -> None:
        self.spec = spec
        self.material = material
        self._grain_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def radius_mm(self) -> float:
        return self.spec.diameter_mm / 2.0

    def contains(self, x_mm: float, y_mm: float) -> bool:
        r = math.hypot(x_mm, y_mm)
        if r > self.radius_mm:
            return False
        if self.spec.edge_feature == "flat":
            flat_offset = self.radius_mm * 0.94
            if y_mm < -flat_offset:
                return False
        elif self.spec.edge_feature == "notch":
            notch_r = self.radius_mm * 0.985
            if (r > notch_r and abs(math.degrees(math.atan2(x_mm, -y_mm))) < 1.2):
                return False
        return True

    def outline(self, n: int = 512) -> np.ndarray:
        """Polygon of the wafer edge including the flat or notch, in mm."""
        pts = []
        for i in range(n):
            th = 2.0 * math.pi * i / n
            x, y = self.radius_mm * math.sin(th), -self.radius_mm * math.cos(th)
            if self.spec.edge_feature == "flat":
                flat = self.radius_mm * 0.94
                if y < -flat:
                    y = -flat
            elif self.spec.edge_feature == "notch":
                ang = abs(math.degrees(th if th <= math.pi else th - 2 * math.pi))
                if ang < 1.4:
                    scale = 1.0 - 0.02 * math.cos(ang / 1.4 * math.pi / 2) ** 2
                    x, y = x * scale, y * scale
            pts.append((x, y))
        return np.array(pts)

    def _grains(self, n_points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        if self._grain_cache is not None:
            return self._grain_cache
        rng = np.random.default_rng(_hash_seed(self.spec.seed, "grains"))
        r = self.radius_mm * np.sqrt(rng.random(n_points))
        th = rng.random(n_points) * 2 * np.pi
        centres = np.column_stack([r * np.cos(th), r * np.sin(th)])
        orientations = rng.random(n_points) * 360.0
        self._grain_cache = (centres, orientations)
        return self._grain_cache

    def grain_at(self, x_mm: float, y_mm: float) -> Optional[dict]:
        if not self.spec.polycrystalline:
            return None
        centres, orientations = self._grains()
        d = np.linalg.norm(centres - np.array([x_mm, y_mm]), axis=1)
        k = int(np.argmin(d))
        second = float(np.sort(d)[1]) if len(d) > 1 else float("inf")
        return {
            "grain_id": k,
            "in_plane_rotation_deg": float(orientations[k]),
            "distance_to_centre_mm": float(d[k]),
            "distance_to_boundary_mm": float(0.5 * (second - d[k])),
            "near_boundary": bool(0.5 * (second - d[k]) < 0.02 * self.spec.mean_grain_size_um * 1e-3),
        }

    def _value_noise(self, x_mm: float, y_mm: float, tag: str,
                     correlation_nm: float, octaves: int = 3) -> float:
        """Deterministic band-limited noise in [-1, 1] at a wafer coordinate."""
        total, amplitude, norm = 0.0, 1.0, 0.0
        lam_mm = correlation_nm * 1e-6
        for octave in range(octaves):
            scale = lam_mm / (2 ** octave)
            gx, gy = math.floor(x_mm / scale), math.floor(y_mm / scale)
            fx, fy = x_mm / scale - gx, y_mm / scale - gy
            corners = []
            for dx in (0, 1):
                for dy in (0, 1):
                    h = _hash_seed(self.spec.seed, tag, octave, gx + dx, gy + dy)
                    corners.append((h % 2000003) / 2000003.0 * 2.0 - 1.0)
            sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
            top = corners[0] * (1 - sy) + corners[1] * sy
            bot = corners[2] * (1 - sy) + corners[3] * sy
            total += amplitude * (top * (1 - sx) + bot * sx)
            norm += amplitude
            amplitude *= 0.5
        return total / max(norm, 1e-12)

    def surface_at(self, x_mm: float, y_mm: float) -> dict:
        """Local surface description at a wafer coordinate."""
        if not self.contains(x_mm, y_mm):
            return {"on_wafer": False,
                    "message": "Coordinate lies outside the wafer outline."}
        rough = self._value_noise(x_mm, y_mm, "roughness",
                                  self.spec.roughness_correlation_nm)
        height_A = rough * self.spec.roughness_rms_A * math.sqrt(2.0)
        terrace_width_nm = (
            float("inf") if self.spec.miscut_deg <= 0 else
            self._step_spacing_nm()
        )
        out = {
            "on_wafer": True,
            "x_mm": x_mm, "y_mm": y_mm,
            "local_height_A": height_A,
            "roughness_rms_A": self.spec.roughness_rms_A,
            "terrace_width_nm": terrace_width_nm,
            "miscut_deg": self.spec.miscut_deg,
            "temperature_K": self.spec.temperature_K,
            "orientation": list(self.spec.orientation),
            "grain": self.grain_at(x_mm, y_mm),
            "device_region": next((r.name for r in self.spec.device_regions
                                   if r.contains(x_mm, y_mm)), None),
            "dopant": self.spec.dopant,
            "dopant_concentration_cm3": self.spec.dopant_concentration_cm3,
            "origin": "estimated",
            "note": ("Procedural microstructure generated from the wafer seed. "
                     "Synthetic, reproducible, and not a measurement."),
        }
        return out

    def _step_spacing_nm(self) -> float:
        from ..structure_builder.lattice import lattice_planes
        d = lattice_planes(self.material, self.spec.orientation)
        return float(d / max(math.tan(math.radians(self.spec.miscut_deg)), 1e-12) / 10.0)

    def defect_budget(self, area_nm2: float) -> dict:
        """Expected number of each surface defect in an area, in nm^2."""
        area_cm2 = area_nm2 * 1e-14
        return {
            "vacancies": self.spec.vacancy_density_cm2 * area_cm2,
            "adatoms": self.spec.adatom_density_cm2 * area_cm2,
            "contaminants": self.spec.contamination_density_cm2 * area_cm2,
        }

    def provenance(self) -> Provenance:
        return Provenance(
            model="multiscale/procedural-wafer",
            fidelity=Fidelity.TIER0_STRUCTURAL,
            origin=Origin.ESTIMATED,
            approximations=[
                "The wafer is defined procedurally, not atom by atom: a full 300 mm "
                "wafer contains of order 1e24 atoms and is never instantiated.",
                "Surface roughness is band-limited value noise with the specified RMS "
                "and correlation length; it reproduces the statistics, not any real "
                "wafer's topography.",
                "Dopants and point defects are placed by a Poisson process at the "
                "specified densities.",
                ("Grains are a Voronoi tessellation of seeded points; real "
                 "microstructure is not Voronoi."
                 if self.spec.polycrystalline else
                 "Single crystal: no grain structure is generated."),
            ],
            parameters=self.spec.as_dict(),
            seed=self.spec.seed,
            references=[],
            notes="Deterministic in (seed, position): the same coordinate always "
                  "reproduces the same microstructure and the same atomistic region.",
        )

    def as_dict(self) -> dict:
        return {"spec": self.spec.as_dict(), "material_id": self.material.id,
                "radius_mm": self.radius_mm}
