"""Solver registry and recommendation.

Solvers register themselves by name.  Plug-ins add solvers through the same
API (see :mod:`materia.plugin_system`).  Nothing else in the program hard
codes a solver name except the material definitions' ``recommended_models``
block, which is itself user-editable data.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .base import Capability, Solver

_REGISTRY: Dict[str, Callable[..., Solver]] = {}
_META: Dict[str, dict] = {}


def register(name: str, factory: Callable[..., Solver], *, overwrite: bool = False,
             meta: Optional[dict] = None) -> None:
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"Solver {name!r} is already registered")
    _REGISTRY[name] = factory
    _META[name] = dict(meta or {})


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)
    _META.pop(name, None)


def available() -> List[str]:
    return sorted(_REGISTRY)


def create(name: str, **kwargs) -> Solver:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown solver {name!r}. Available: {', '.join(available())}. "
            "External solvers appear here only when their optional package is installed."
        )
    return _REGISTRY[name](**kwargs)


def describe_all() -> List[dict]:
    out = []
    for name in available():
        try:
            d = create(name).describe()
        except Exception as exc:
            d = {"name": name, "description": f"unavailable: {exc}",
                 "capabilities": [], "fidelity": "unknown", "available": False}
        d.setdefault("available", True)
        d.update(_META.get(name, {}))
        d["key"] = name
        out.append(d)
    return out


def solvers_providing(capability: Capability) -> List[str]:
    """Names of registered solvers that declare ``capability``."""
    out = []
    for name in available():
        try:
            solver = create(name)
        except Exception:
            continue
        if solver.can(capability):
            out.append(name)
    return out


def resolve_recommended(material, task: str = "relax") -> Optional[str]:
    """Look up a material's recommended solver for a task, if it is registered."""
    want = (material.recommended_models or {}).get(task)
    if not want or want == "unsupported":
        return None
    if want in _REGISTRY:
        return want
    for name in available():
        if want.startswith(name) or name.startswith(want.split("-")[0]):
            return name
    return None
