"""Provenance, fidelity and uncertainty bookkeeping.

Every number Materia shows the user must be traceable to:

* the model that produced it (``model``),
* the approximations that model makes (``approximations``),
* its numerical settings (``tolerances``, ``boundary_conditions``),
* whether it converged (``convergence``),
* how it should be read (``origin``: calculated / interpolated / estimated /
  illustrative / reference / imported / measured),
* its unit, its uncertainty when one is defined, and the software version.

This module is deliberately dependency-free so it can be imported by every
other module, including the serialisation layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import Any, Dict, List, Optional

from ..version import __version__, PROJECT_SCHEMA_VERSION


class Origin(str, Enum):
    """How a value came to exist.  Shown verbatim in the UI."""

    CALCULATED = "calculated"
    """Produced by a solver run in this session from the current state."""
    INTERPOLATED = "interpolated"
    """Interpolated or extrapolated from calculated samples."""
    ESTIMATED = "estimated"
    """Produced by a heuristic or empirical rule, not a variational solution."""
    ILLUSTRATIVE = "illustrative"
    """Teaching visualisation only. Not a physical prediction."""
    REFERENCE = "reference"
    """Literature/tabulated value shipped with the program."""
    IMPORTED = "imported"
    """Read from a user-supplied file or external solver."""
    MEASURED = "measured"
    """Experimental data supplied by the user."""
    UNSUPPORTED = "unsupported"
    """The active model cannot produce this quantity. Value is None."""


class Fidelity(str, Enum):
    """Coarse capability tier of the producing model.  See docs/FIDELITY_TIERS.md."""

    TIER0_STRUCTURAL = "tier0-structural"
    TIER1_CLASSICAL = "tier1-classical"
    TIER2_SEMI_EMPIRICAL = "tier2-semi-empirical"
    TIER3_EXTERNAL = "tier3-external-first-principles"
    NON_PHYSICAL = "non-physical"


@dataclass
class Convergence:
    """Solver convergence record."""

    converged: bool
    iterations: int = 0
    residual: Optional[float] = None
    residual_metric: str = ""
    tolerance: Optional[float] = None
    message: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Provenance:
    """Full traceability record attached to a :class:`Result`."""

    model: str
    fidelity: Fidelity
    origin: Origin
    approximations: List[str] = field(default_factory=list)
    tolerances: Dict[str, Any] = field(default_factory=dict)
    boundary_conditions: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    dataset: str = ""
    dataset_license: str = ""
    seed: Optional[int] = None
    software_version: str = __version__
    schema_version: str = PROJECT_SCHEMA_VERSION
    platform: str = field(default_factory=lambda: f"{platform.python_implementation()} {platform.python_version()} / {platform.system()}")
    created_unix: float = field(default_factory=time.time)
    inputs_digest: str = ""
    notes: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["fidelity"] = self.fidelity.value
        d["origin"] = self.origin.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "Provenance":
        """Rebuild a provenance record saved with a project."""
        known = {f.name for f in dataclass_fields(Provenance)}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["fidelity"] = Fidelity(d.get("fidelity", Fidelity.NON_PHYSICAL.value))
        kwargs["origin"] = Origin(d.get("origin", Origin.ESTIMATED.value))
        return Provenance(**kwargs)

    def summary(self) -> str:
        parts = [f"{self.model} [{self.fidelity.value}, {self.origin.value}]"]
        if self.approximations:
            parts.append("approximations: " + "; ".join(self.approximations))
        if self.boundary_conditions:
            parts.append("BC: " + self.boundary_conditions)
        return " | ".join(parts)


@dataclass
class Result:
    """A value plus everything needed to interpret it honestly.

    ``value`` is ``None`` exactly when ``provenance.origin`` is
    :attr:`Origin.UNSUPPORTED`; in that case :attr:`unsupported_reason` and
    :attr:`suggested_models` explain what would be needed.
    """

    name: str
    value: Any
    unit: str
    provenance: Provenance
    uncertainty: Optional[float] = None
    uncertainty_kind: str = ""
    confidence: Optional[float] = None
    convergence: Optional[Convergence] = None
    unsupported_reason: str = ""
    suggested_models: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.provenance.origin is not Origin.UNSUPPORTED

    def as_dict(self, include_value: bool = True) -> dict:
        d = {
            "name": self.name,
            "unit": self.unit,
            "provenance": self.provenance.as_dict(),
            "uncertainty": self.uncertainty,
            "uncertainty_kind": self.uncertainty_kind,
            "confidence": self.confidence,
            "convergence": self.convergence.as_dict() if self.convergence else None,
            "unsupported_reason": self.unsupported_reason,
            "suggested_models": list(self.suggested_models),
            "supported": self.supported,
            "extra": self.extra,
        }
        if include_value:
            d["value"] = _jsonable(self.value)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Result":
        """Rebuild a result saved with a project.

        A value that was too large to store comes back as ``None`` with
        ``value_omitted`` recorded in ``extra``, so a reader can tell a value
        that was never saved from one that was genuinely absent.
        """
        provenance = Provenance.from_dict(d.get("provenance", {}))
        convergence = d.get("convergence")
        extra = dict(d.get("extra", {}) or {})
        if "value" not in d and provenance.origin is not Origin.UNSUPPORTED:
            extra.setdefault("value_omitted",
                             "The value was too large to store in the project file.")
        return Result(
            name=str(d.get("name", "")),
            value=d.get("value"),
            unit=str(d.get("unit", "")),
            provenance=provenance,
            uncertainty=d.get("uncertainty"),
            uncertainty_kind=str(d.get("uncertainty_kind", "")),
            confidence=d.get("confidence"),
            convergence=(Convergence(**convergence) if isinstance(convergence, dict)
                         else None),
            unsupported_reason=str(d.get("unsupported_reason", "")),
            suggested_models=list(d.get("suggested_models", []) or []),
            extra=extra,
        )

    def __str__(self) -> str:
        if not self.supported:
            return f"{self.name}: unsupported by {self.provenance.model} ({self.unsupported_reason})"
        v = self.value
        if isinstance(v, float):
            body = f"{v:.6g}"
        else:
            body = str(v)
        unc = f" +/- {self.uncertainty:.3g}" if self.uncertainty is not None else ""
        return f"{self.name} = {body}{unc} {self.unit} [{self.provenance.origin.value}]"


def unsupported(
    name: str,
    model: str,
    reason: str,
    suggested_models: Optional[List[str]] = None,
    unit: str = "",
    fidelity: Fidelity = Fidelity.NON_PHYSICAL,
) -> Result:
    """Build an explicit 'this model cannot answer that' result.

    Never return a fabricated number when a model is out of its domain: return
    this instead.  The UI renders it as a blocked field with the reason and the
    recommended solvers.
    """
    return Result(
        name=name,
        value=None,
        unit=unit,
        provenance=Provenance(
            model=model,
            fidelity=fidelity,
            origin=Origin.UNSUPPORTED,
            notes=reason,
        ),
        unsupported_reason=reason,
        suggested_models=list(suggested_models or []),
    )


def digest(payload: Any) -> str:
    """Stable content digest used for cache keys and reproducibility checks."""
    blob = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return str(obj)
