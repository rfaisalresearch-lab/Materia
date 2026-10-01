"""Plug-in discovery and loading.

A plug-in is a Python package (or a single module) that exposes a
``register(context)`` function.  It may add materials, solvers, potentials,
importers, exporters, analysis tools, palettes and interface panels.  Nothing
about the core is special-cased for the built-ins: they use the same calls.

Discovery order
---------------
1. entry points in the ``materia.plugins`` group (installed packages)
2. ``$MATERIA_PLUGINS`` (os.pathsep-separated directories)
3. ``~/.materia/plugins``
4. the ``plugins/`` directory of this repository, for development

Contract
--------
.. code-block:: python

    PLUGIN_API_VERSION = "1.0"
    PLUGIN = {
        "id": "my_plugin",
        "name": "My plug-in",
        "version": "0.1.0",
        "description": "...",
        "author": "...",
        "license": "MIT",
        "api_version": "1.0",
    }

    def register(ctx):
        ctx.add_material_path("/path/to/materials")
        ctx.add_solver("my-solver", MySolverFactory)
        ctx.add_exporter("myfmt", write_myfmt)
        ctx.add_analysis("my-analysis", run_my_analysis)

Loading a plug-in executes its code with the privileges of the Materia
process.  Materia shows what it is about to load and requires the plug-in
directory to be one the user configured; it does not sandbox plug-ins, and
says so.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..version import PLUGIN_API_VERSION

ENV_VAR = "MATERIA_PLUGINS"
USER_DIR = Path(os.path.expanduser("~")) / ".materia" / "plugins"
REPO_DIR = Path(__file__).resolve().parents[2] / "plugins"
ENTRY_POINT_GROUP = "materia.plugins"


@dataclass
class PluginContext:
    """What a plug-in is given at registration time."""

    materials: Any = None
    exporters: Dict[str, Callable] = field(default_factory=dict)
    importers: Dict[str, Callable] = field(default_factory=dict)
    analyses: Dict[str, Callable] = field(default_factory=dict)
    palettes: Dict[str, Any] = field(default_factory=dict)
    panels: List[dict] = field(default_factory=list)
    solvers_added: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

    def add_material_path(self, path: str) -> None:
        from ..materials.loader import default_library
        lib = self.materials or default_library()
        lib.add_search_path(path)
        self.log.append(f"material search path: {path}")

    def add_material(self, raw: dict, source: str = "<plugin>") -> str:
        from ..materials.loader import default_library
        lib = self.materials or default_library()
        d = lib.register_raw(raw, source, overwrite=True)
        self.log.append(f"material: {d.id}")
        return d.id

    def add_solver(self, name: str, factory: Callable, meta: Optional[dict] = None) -> None:
        from ..solvers.registry import register
        register(name, factory, overwrite=True, meta={**(meta or {}), "plugin": True})
        self.solvers_added.append(name)
        self.log.append(f"solver: {name}")

    def add_exporter(self, fmt: str, fn: Callable) -> None:
        self.exporters[fmt] = fn
        self.log.append(f"exporter: {fmt}")

    def add_importer(self, fmt: str, fn: Callable) -> None:
        self.importers[fmt] = fn
        self.log.append(f"importer: {fmt}")

    def add_analysis(self, name: str, fn: Callable) -> None:
        self.analyses[name] = fn
        self.log.append(f"analysis: {name}")

    def add_palette(self, palette) -> None:
        from ..visualization.palette import PALETTES
        PALETTES[palette.id] = palette
        self.palettes[palette.id] = palette
        self.log.append(f"palette: {palette.id}")

    def add_panel(self, spec: dict) -> None:
        self.panels.append(spec)
        self.log.append(f"panel: {spec.get('id', '?')}")


@dataclass
class LoadedPlugin:
    id: str
    name: str
    version: str
    description: str
    path: str
    api_version: str
    license: str = ""
    author: str = ""
    ok: bool = True
    error: str = ""
    provided: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class PluginManager:
    def __init__(self, search_paths: Optional[List[str]] = None) -> None:
        self.paths: List[Path] = []
        for p in (search_paths or []):
            self.add_path(p)
        for chunk in os.environ.get(ENV_VAR, "").split(os.pathsep):
            if chunk.strip():
                self.add_path(chunk.strip())
        if USER_DIR.is_dir():
            self.paths.append(USER_DIR)
        if REPO_DIR.is_dir():
            self.paths.append(REPO_DIR)
        self.context = PluginContext()
        self.loaded: List[LoadedPlugin] = []

    def add_path(self, path: str) -> None:
        p = Path(path).expanduser().resolve()
        if p not in self.paths:
            self.paths.insert(0, p)

    def discover(self) -> List[dict]:
        """List candidate plug-ins without executing their register()."""
        found: List[dict] = []
        for directory in self.paths:
            if not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir()):
                if entry.name.startswith((".", "_")):
                    continue
                if entry.is_dir() and (entry / "__init__.py").exists():
                    found.append({"id": entry.name, "path": str(entry), "kind": "package"})
                elif entry.suffix == ".py":
                    found.append({"id": entry.stem, "path": str(entry), "kind": "module"})
        try:
            from importlib.metadata import entry_points
            eps = entry_points()
            group = (eps.select(group=ENTRY_POINT_GROUP)
                     if hasattr(eps, "select") else eps.get(ENTRY_POINT_GROUP, []))
            for ep in group:
                found.append({"id": ep.name, "path": f"entry-point:{ep.value}",
                              "kind": "entry-point"})
        except Exception:
            pass
        return found

    def load_all(self) -> List[LoadedPlugin]:
        self.loaded = []
        for candidate in self.discover():
            self.loaded.append(self.load(candidate))
        return self.loaded

    def load(self, candidate: dict) -> LoadedPlugin:
        path = candidate["path"]
        try:
            if candidate["kind"] == "entry-point":
                from importlib.metadata import entry_points
                eps = entry_points()
                group = (eps.select(group=ENTRY_POINT_GROUP)
                         if hasattr(eps, "select") else eps.get(ENTRY_POINT_GROUP, []))
                ep = next(e for e in group if e.name == candidate["id"])
                module = ep.load()
                if not hasattr(module, "register"):
                    module = importlib.import_module(module.__module__)
            else:
                module = self._import_from_path(candidate["id"], path)
            meta = getattr(module, "PLUGIN", {}) or {}
            api = str(meta.get("api_version", getattr(module, "PLUGIN_API_VERSION", "0")))
            if api.split(".")[0] != PLUGIN_API_VERSION.split(".")[0]:
                return LoadedPlugin(
                    id=candidate["id"], name=meta.get("name", candidate["id"]),
                    version=meta.get("version", "?"), description=meta.get("description", ""),
                    path=path, api_version=api, ok=False,
                    error=(f"Plug-in targets API {api}, this build provides "
                           f"{PLUGIN_API_VERSION}. Not loaded."))
            register = getattr(module, "register", None)
            if register is None:
                return LoadedPlugin(
                    id=candidate["id"], name=meta.get("name", candidate["id"]),
                    version=meta.get("version", "?"), description="", path=path,
                    api_version=api, ok=False,
                    error="No register(ctx) function; nothing to load.")
            before = len(self.context.log)
            register(self.context)
            provided = self.context.log[before:]
            return LoadedPlugin(
                id=meta.get("id", candidate["id"]), name=meta.get("name", candidate["id"]),
                version=meta.get("version", "?"), description=meta.get("description", ""),
                path=path, api_version=api, license=meta.get("license", ""),
                author=meta.get("author", ""), provided=provided)
        except Exception as exc:
            return LoadedPlugin(
                id=candidate["id"], name=candidate["id"], version="?",
                description="", path=path, api_version="?", ok=False,
                error=f"{type(exc).__name__}: {exc}\n"
                      + "".join(traceback.format_exc(limit=3)))

    @staticmethod
    def _import_from_path(name: str, path: str):
        target = Path(path)
        module_file = target / "__init__.py" if target.is_dir() else target
        spec = importlib.util.spec_from_file_location(f"materia_plugin_{name}",
                                                      str(module_file))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import plug-in from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def report(self) -> List[dict]:
        return [p.as_dict() for p in self.loaded]


_DEFAULT: Optional[PluginManager] = None


def default_manager() -> PluginManager:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PluginManager()
    return _DEFAULT
