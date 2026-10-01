"""Material-definition schema and validation.

A material definition is a plain JSON document.  Users and plug-ins add
materials by dropping files into a search path; nothing is hard-coded.
See ``docs/MATERIAL_SCHEMA.md`` for the normative description.

Validation is performed by this module rather than by an external JSON-Schema
dependency so that error messages can name the offending field precisely and
so that the core has no extra runtime requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..version import MATERIAL_SCHEMA_VERSION

REQUIRED_TOP_LEVEL = ("schema_version", "id", "name", "formula", "structure")
REQUIRED_STRUCTURE = ("prototype", "lattice", "basis")
REQUIRED_LATTICE = ("a", "unit")
LATTICE_SYSTEMS = (
    "cubic", "tetragonal", "orthorhombic", "hexagonal", "trigonal",
    "monoclinic", "triclinic", "rhombohedral", "amorphous", "layered-2d",
)


class MaterialValidationError(ValueError):
    """A material definition violated the schema."""


@dataclass
class PropertyValue:
    """A single physical property with its unit, conditions and source."""

    value: Any
    unit: str = ""
    source: str = ""
    conditions: str = ""
    uncertainty: Optional[float] = None
    note: str = ""
    origin: str = "reference"

    @staticmethod
    def parse(key: str, raw: Any) -> "PropertyValue":
        if isinstance(raw, dict):
            if "value" not in raw:
                raise MaterialValidationError(f"property '{key}' has no 'value' field")
            return PropertyValue(
                value=raw["value"],
                unit=raw.get("unit", ""),
                source=raw.get("source", ""),
                conditions=raw.get("conditions", ""),
                uncertainty=raw.get("uncertainty"),
                note=raw.get("note", ""),
                origin=raw.get("origin", "reference"),
            )
        return PropertyValue(value=raw, note="bare value: no unit or source recorded")

    def as_dict(self) -> dict:
        return {
            "value": self.value, "unit": self.unit, "source": self.source,
            "conditions": self.conditions, "uncertainty": self.uncertainty,
            "note": self.note, "origin": self.origin,
        }

    def __str__(self) -> str:
        u = f" {self.unit}" if self.unit else ""
        return f"{self.value}{u}"


@dataclass
class BasisSite:
    """One site of the unit-cell basis, in fractional coordinates."""

    element: str
    fractional: Tuple[float, float, float]
    label: str = ""
    wyckoff: str = ""
    occupancy: float = 1.0
    role: str = "bulk"

    def as_dict(self) -> dict:
        return {
            "element": self.element, "fractional": list(self.fractional),
            "label": self.label, "wyckoff": self.wyckoff,
            "occupancy": self.occupancy, "role": self.role,
        }


@dataclass
class Lattice:
    a: float
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    unit: str = "A"
    system: str = "triclinic"
    temperature_K: Optional[float] = None
    source: str = ""

    def parameters(self) -> Tuple[float, float, float, float, float, float]:
        """Resolve (a, b, c, alpha, beta, gamma) applying system constraints."""
        a = self.a
        b = self.b if self.b is not None else a
        c = self.c if self.c is not None else a
        al, be, ga = self.alpha, self.beta, self.gamma
        if self.system == "cubic":
            b = c = a
            al = be = ga = 90.0
        elif self.system in ("hexagonal", "layered-2d"):
            b = a
            al = be = 90.0
            ga = 120.0
        elif self.system == "tetragonal":
            b = a
            al = be = ga = 90.0
        elif self.system == "orthorhombic":
            al = be = ga = 90.0
        return (a, b, c, al, be, ga)

    def as_dict(self) -> dict:
        return {
            "a": self.a, "b": self.b, "c": self.c,
            "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            "unit": self.unit, "system": self.system,
            "temperature_K": self.temperature_K, "source": self.source,
        }


@dataclass
class Termination:
    id: str
    orientation: Tuple[int, int, int]
    description: str = ""
    passivation: str = ""
    implemented: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["orientation"] = list(self.orientation)
        return d


@dataclass
class Reconstruction:
    """A declared surface reconstruction.

    ``reference_geometry`` holds published structural parameters keyed by the
    quantity name used in the measurement helpers, each with ``value``, optional
    ``uncertainty``, ``unit``, ``method`` and ``source``.  They exist so that a
    computed geometry can be compared against measurement and are never used to
    place atoms.
    """

    id: str
    orientation: Tuple[int, int, int]
    description: str = ""
    implemented: bool = False
    reference: str = ""
    note: str = ""
    method: str = ""
    reference_geometry: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["orientation"] = list(self.orientation)
        d["reference_geometry"] = {k: dict(v) for k, v in self.reference_geometry.items()}
        return d


@dataclass
class MaterialDefinition:
    """A validated material definition."""

    id: str
    name: str
    formula: str
    category: str
    lattice: Lattice
    basis: List[BasisSite]
    prototype: str
    space_group_number: Optional[int] = None
    space_group_symbol: str = ""
    aliases: List[str] = field(default_factory=list)
    orientations: List[Tuple[int, int, int]] = field(default_factory=list)
    terminations: List[Termination] = field(default_factory=list)
    reconstructions: List[Reconstruction] = field(default_factory=list)
    properties: Dict[str, PropertyValue] = field(default_factory=dict)
    recommended_models: Dict[str, str] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    license: str = ""
    provenance_note: str = ""
    source_path: str = ""
    schema_version: str = MATERIAL_SCHEMA_VERSION
    raw: Dict[str, Any] = field(default_factory=dict)

    def property(self, key: str) -> Optional[PropertyValue]:
        return self.properties.get(key)

    def elements(self) -> List[str]:
        out: List[str] = []
        for s in self.basis:
            if s.element not in out:
                out.append(s.element)
        return out

    def summary(self) -> dict:
        a, b, c, al, be, ga = self.lattice.parameters()
        return {
            "id": self.id,
            "name": self.name,
            "formula": self.formula,
            "category": self.category,
            "prototype": self.prototype,
            "system": self.lattice.system,
            "space_group": (
                f"{self.space_group_symbol} (#{self.space_group_number})"
                if self.space_group_number else self.space_group_symbol
            ),
            "lattice_A": [a, b, c],
            "angles_deg": [al, be, ga],
            "n_basis": len(self.basis),
            "elements": self.elements(),
            "orientations": [list(o) for o in self.orientations],
            "band_gap_eV": (self.properties["band_gap"].value
                            if "band_gap" in self.properties else None),
            "recommended_models": dict(self.recommended_models),
            "license": self.license,
        }

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "formula": self.formula,
            "aliases": list(self.aliases),
            "category": self.category,
            "structure": {
                "prototype": self.prototype,
                "space_group": {"number": self.space_group_number,
                                "symbol": self.space_group_symbol},
                "lattice": self.lattice.as_dict(),
                "basis": [s.as_dict() for s in self.basis],
            },
            "orientations": [list(o) for o in self.orientations],
            "terminations": [t.as_dict() for t in self.terminations],
            "reconstructions": [r.as_dict() for r in self.reconstructions],
            "properties": {k: v.as_dict() for k, v in self.properties.items()},
            "recommended_models": dict(self.recommended_models),
            "references": list(self.references),
            "license": self.license,
            "provenance_note": self.provenance_note,
        }


def _reference_geometry(raw: Dict[str, Any], source_path: str) -> Dict[str, Dict[str, Any]]:
    """Validate the published-geometry block of one reconstruction declaration."""
    block = raw.get("reference_geometry") or {}
    where = f" in {source_path}" if source_path else ""
    if not isinstance(block, dict):
        raise MaterialValidationError(
            f"reconstruction {raw.get('id')!r}: reference_geometry must be an object{where}"
        )
    out: Dict[str, Dict[str, Any]] = {}
    for key, entry in block.items():
        if not isinstance(entry, dict) or "value" not in entry:
            raise MaterialValidationError(
                f"reconstruction {raw.get('id')!r}: reference_geometry[{key!r}] needs a "
                f"'value'{where}"
            )
        if not entry.get("source"):
            raise MaterialValidationError(
                f"reconstruction {raw.get('id')!r}: reference_geometry[{key!r}] needs a "
                f"'source' citation{where}"
            )
        out[key] = {
            "value": float(entry["value"]),
            "uncertainty": float(entry["uncertainty"]) if entry.get("uncertainty") is not None else None,
            "unit": str(entry.get("unit", "A")),
            "method": str(entry.get("method", "")),
            "source": str(entry["source"]),
            "note": str(entry.get("note", "")),
        }
    return out


def validate(raw: Dict[str, Any], source_path: str = "") -> MaterialDefinition:
    """Validate a raw material dict and build a :class:`MaterialDefinition`."""
    where = f" in {source_path}" if source_path else ""
    for key in REQUIRED_TOP_LEVEL:
        if key not in raw:
            raise MaterialValidationError(f"missing required field '{key}'{where}")

    if str(raw["schema_version"]).split(".")[0] != MATERIAL_SCHEMA_VERSION.split(".")[0]:
        raise MaterialValidationError(
            f"material schema_version {raw['schema_version']} is incompatible with "
            f"{MATERIAL_SCHEMA_VERSION}{where}"
        )

    st = raw["structure"]
    for key in REQUIRED_STRUCTURE:
        if key not in st:
            raise MaterialValidationError(f"structure.{key} is required{where}")

    lat_raw = st["lattice"]
    for key in REQUIRED_LATTICE:
        if key not in lat_raw:
            raise MaterialValidationError(f"structure.lattice.{key} is required{where}")
    system = lat_raw.get("system", "triclinic")
    if system not in LATTICE_SYSTEMS:
        raise MaterialValidationError(
            f"unknown lattice system '{system}'{where}; expected one of {LATTICE_SYSTEMS}"
        )
    if lat_raw["unit"] != "A":
        raise MaterialValidationError(
            f"structure.lattice.unit must be 'A' (angstrom){where}, got {lat_raw['unit']!r}"
        )
    lattice = Lattice(
        a=float(lat_raw["a"]),
        b=None if lat_raw.get("b") is None else float(lat_raw["b"]),
        c=None if lat_raw.get("c") is None else float(lat_raw["c"]),
        alpha=float(lat_raw.get("alpha", 90.0)),
        beta=float(lat_raw.get("beta", 90.0)),
        gamma=float(lat_raw.get("gamma", 90.0)),
        unit=lat_raw["unit"],
        system=system,
        temperature_K=lat_raw.get("temperature_K"),
        source=lat_raw.get("source", ""),
    )
    if lattice.a <= 0:
        raise MaterialValidationError(f"lattice parameter a must be positive{where}")

    basis_raw = st["basis"]
    if not basis_raw:
        raise MaterialValidationError(f"structure.basis must contain at least one site{where}")
    basis: List[BasisSite] = []
    from ..elements import periodic_table as pt
    for i, site in enumerate(basis_raw):
        if "element" not in site or "fractional" not in site:
            raise MaterialValidationError(
                f"structure.basis[{i}] needs 'element' and 'fractional'{where}"
            )
        try:
            pt.element(site["element"])
        except Exception as exc:
            raise MaterialValidationError(
                f"structure.basis[{i}] unknown element {site['element']!r}{where}"
            ) from exc
        frac = site["fractional"]
        if len(frac) != 3:
            raise MaterialValidationError(
                f"structure.basis[{i}].fractional must have 3 components{where}"
            )
        occupancy = float(site.get("occupancy", 1.0))
        if not 0.0 <= occupancy <= 1.0:
            raise MaterialValidationError(
                f"structure.basis[{i}].occupancy must lie between 0 and 1{where}")
        basis.append(BasisSite(
            element=site["element"],
            fractional=tuple(float(x) for x in frac),
            label=site.get("label", ""),
            wyckoff=site.get("wyckoff", ""),
            occupancy=occupancy,
            role=site.get("role", "bulk"),
        ))

    sg = st.get("space_group", {}) or {}
    props = {k: PropertyValue.parse(k, v) for k, v in (raw.get("properties") or {}).items()}

    return MaterialDefinition(
        id=str(raw["id"]),
        name=str(raw["name"]),
        formula=str(raw["formula"]),
        category=str(raw.get("category", "unclassified")),
        lattice=lattice,
        basis=basis,
        prototype=str(st["prototype"]),
        space_group_number=sg.get("number"),
        space_group_symbol=sg.get("symbol", ""),
        aliases=list(raw.get("aliases", [])),
        orientations=[tuple(int(v) for v in o) for o in raw.get("orientations", [])],
        terminations=[
            Termination(
                id=t["id"], orientation=tuple(int(v) for v in t["orientation"]),
                description=t.get("description", ""), passivation=t.get("passivation", ""),
                implemented=bool(t.get("implemented", True)), note=t.get("note", ""),
            ) for t in raw.get("terminations", [])
        ],
        reconstructions=[
            Reconstruction(
                id=r["id"], orientation=tuple(int(v) for v in r["orientation"]),
                description=r.get("description", ""),
                implemented=bool(r.get("implemented", False)),
                reference=r.get("reference", ""), note=r.get("note", ""),
                method=r.get("method", ""),
                reference_geometry=_reference_geometry(r, source_path),
            ) for r in raw.get("reconstructions", [])
        ],
        properties=props,
        recommended_models=dict(raw.get("recommended_models", {})),
        references=list(raw.get("references", [])),
        license=str(raw.get("license", "")),
        provenance_note=str(raw.get("provenance_note", "")),
        source_path=source_path,
        schema_version=str(raw["schema_version"]),
        raw=raw,
    )
