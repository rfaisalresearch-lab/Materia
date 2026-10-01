"""Region-of-interest extraction: from wafer coordinates to atoms.

An :class:`AtomisticRegion` is the bridge between the procedural wafer and the
explicit atomistic model.  It records exactly where on the wafer it came from,
which seed produced it, and which procedural features were applied, so that
the same region can be regenerated bit-for-bit from the project file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..elements import periodic_table as pt
from ..provenance import Fidelity, Origin, Provenance, digest
from ..structure_builder.defects import add_adatom, add_dopant, create_vacancy
from ..structure_builder.surface import make_surface
from .wafer import Wafer, _hash_seed


@dataclass
class RegionSpec:
    """Where on the wafer an atomistic region was taken from."""

    x_mm: float = 0.0
    y_mm: float = 0.0
    size_nm: Tuple[float, float] = (3.0, 3.0)
    depth_layers: int = 6
    vacuum_A: float = 14.0
    include_defects: bool = True
    include_dopants: bool = True
    passivate_bottom: bool = False
    max_atoms: int = 20000
    seed_offset: int = 0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["size_nm"] = list(self.size_nm)
        return d

    @staticmethod
    def from_dict(d: dict) -> "RegionSpec":
        d = dict(d)
        d["size_nm"] = tuple(d.get("size_nm", (3.0, 3.0)))
        return RegionSpec(**d)


@dataclass
class AtomisticRegion:
    """An explicit atomistic model generated from a wafer coordinate."""

    structure: Structure
    spec: RegionSpec
    wafer_context: dict
    applied: dict
    provenance: Provenance
    region_id: str = ""

    def as_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "spec": self.spec.as_dict(),
            "wafer_context": self.wafer_context,
            "applied": self.applied,
            "provenance": self.provenance.as_dict(),
            "n_atoms": len(self.structure),
        }


def _repetitions_for(cell_matrix: np.ndarray, size_nm: Sequence[float]) -> Tuple[int, int]:
    ax = float(np.linalg.norm(cell_matrix[0]))
    ay = float(np.linalg.norm(cell_matrix[1]))
    nx = max(1, int(round(size_nm[0] * 10.0 / max(ax, 1e-9))))
    ny = max(1, int(round(size_nm[1] * 10.0 / max(ay, 1e-9))))
    return nx, ny


def extract_region(wafer: Wafer, spec: Optional[RegionSpec] = None) -> AtomisticRegion:
    """Instantiate atoms for one region of a wafer.

    The base slab is generated from the wafer's material and orientation; the
    procedural wafer state at ``(x_mm, y_mm)`` then decides which dopants,
    vacancies and adsorbates are present, using a seed derived from the wafer
    seed and the region coordinate.
    """
    spec = spec or RegionSpec()
    context = wafer.surface_at(spec.x_mm, spec.y_mm)
    if not context.get("on_wafer"):
        raise ValueError(
            f"Region centre ({spec.x_mm}, {spec.y_mm}) mm lies outside the "
            f"{wafer.spec.diameter_mm} mm wafer."
        )

    unit = make_surface(wafer.material, wafer.spec.orientation,
                        size=(1, 1, 1), vacuum_A=spec.vacuum_A)
    nx, ny = _repetitions_for(unit.cell.matrix, spec.size_nm)
    nz = max(1, int(spec.depth_layers))

    estimated = len(unit) * nx * ny * nz
    if estimated > spec.max_atoms:
        scale = math.sqrt(spec.max_atoms / max(estimated, 1))
        nx = max(1, int(nx * scale))
        ny = max(1, int(ny * scale))
        estimated = len(unit) * nx * ny * nz

    structure = make_surface(
        wafer.material, wafer.spec.orientation,
        size=(nx, ny, nz), vacuum_A=spec.vacuum_A,
        termination=wafer.spec.termination,
        fix_bottom_layers=1 if nz >= 3 else 0,
    )

    seed = _hash_seed(wafer.spec.seed, "region", round(spec.x_mm, 6),
                      round(spec.y_mm, 6), spec.seed_offset) % (2 ** 63)
    rng = np.random.default_rng(seed)

    cell = structure.cell.matrix
    area_nm2 = float(abs(cell[0][0] * cell[1][1] - cell[0][1] * cell[1][0])) / 100.0
    budget = wafer.defect_budget(area_nm2)
    applied: Dict[str, object] = {"area_nm2": area_nm2, "expected": dict(budget)}

    z = structure.positions[:, 2]
    surface_ids = [int(i) for i, zz in zip(structure.ids, z) if zz >= z.max() - 0.6]
    bulkish_ids = [int(i) for i, zz in zip(structure.ids, z)
                   if z.min() + 1.0 < zz < z.max() - 0.6]

    dopant_records: List[dict] = []
    if (spec.include_dopants and wafer.spec.dopant
            and wafer.spec.dopant_concentration_cm3 > 0):
        volume_cm3 = (abs(np.linalg.det(np.array([cell[0], cell[1],
                                                  [0, 0, float(z.max() - z.min())]])))
                      * 1e-24)
        expected = wafer.spec.dopant_concentration_cm3 * volume_cm3
        n = int(rng.poisson(expected))
        candidates = bulkish_ids or surface_ids
        n = min(n, len(candidates))
        if n:
            picks = rng.choice(len(candidates), size=n, replace=False)
            for p in picks:
                dopant_records.append(
                    add_dopant(structure, wafer.spec.dopant, site=candidates[int(p)]))
        applied["dopants"] = {"expected": expected, "placed": len(dopant_records),
                              "element": wafer.spec.dopant,
                              "concentration_cm3": wafer.spec.dopant_concentration_cm3}

    vacancy_records: List[dict] = []
    if spec.include_defects and budget["vacancies"] > 0:
        n = int(rng.poisson(budget["vacancies"]))
        live = [i for i in surface_ids if structure.has_id(i)]
        n = min(n, max(0, len(live) - 1))
        if n:
            picks = rng.choice(len(live), size=n, replace=False)
            for p in sorted(picks, reverse=True):
                vid = live[int(p)]
                if structure.has_id(vid):
                    vacancy_records.append(create_vacancy(structure, vid))
        applied["vacancies"] = {"expected": budget["vacancies"],
                                "placed": len(vacancy_records)}

    adatom_records: List[dict] = []
    for kind, count, element in (
        ("adatoms", budget["adatoms"], pt.symbol(structure.numbers[0])),
        ("contaminants", budget["contaminants"], wafer.spec.contaminant),
    ):
        if not spec.include_defects or count <= 0:
            continue
        n = int(rng.poisson(count))
        for _ in range(n):
            fx, fy = rng.random(), rng.random()
            p = fx * cell[0] + fy * cell[1]
            adatom_records.append(add_adatom(structure, element, (p[0], p[1])))
        applied[kind] = {"expected": count, "placed": n, "element": element}

    structure.info["wafer_local_height_A"] = context["local_height_A"]

    if spec.passivate_bottom:
        from ..structure_builder.surface import passivate
        applied["passivation"] = passivate(structure, "H", side="bottom")

    prov = wafer.provenance()
    prov.parameters = {**prov.parameters, "region": spec.as_dict()}
    prov.inputs_digest = digest({"wafer": wafer.spec.as_dict(), "region": spec.as_dict()})
    prov.notes += (f" Region seed {seed}; {len(structure)} atoms instantiated from a "
                   f"{wafer.spec.diameter_mm} mm wafer.")
    structure.info["region_provenance"] = prov.as_dict()
    structure.info["wafer_context"] = context

    region_id = f"r{digest({'x': spec.x_mm, 'y': spec.y_mm, 'seed': seed})}"
    return AtomisticRegion(
        structure=structure, spec=spec, wafer_context=context,
        applied=applied, provenance=prov, region_id=region_id,
    )


def scale_ladder(view_span_A: float) -> dict:
    """Which scale level a given view span corresponds to."""
    from .wafer import SCALE_LEVELS
    for name, span, unit, description in SCALE_LEVELS:
        if view_span_A >= span:
            return {"level": name, "unit": unit, "description": description,
                    "span_A": view_span_A}
    name, span, unit, description = SCALE_LEVELS[-1]
    return {"level": name, "unit": unit, "description": description,
            "span_A": view_span_A}


def format_span(span_A: float) -> Tuple[float, str]:
    """Human-readable magnitude and unit for a length in angstrom."""
    if span_A >= 1.0e7:
        return span_A / 1.0e7, "mm"
    if span_A >= 1.0e4:
        return span_A / 1.0e4, "um"
    if span_A >= 10.0:
        return span_A / 10.0, "nm"
    return span_A, "A"
