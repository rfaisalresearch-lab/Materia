"""Material registry: discovery, loading and user extension.

Search order (first match wins for a given id):

1. paths passed explicitly to :meth:`MaterialLibrary.add_search_path`
2. ``$MATERIA_MATERIALS`` (os.pathsep-separated)
3. ``~/.materia/materials``
4. the built-in library shipped inside the package

Nothing about the program is specialised to the built-in list: a user adds a
material by dropping a JSON file into any search path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .schema import MaterialDefinition, MaterialValidationError, validate

BUILTIN_DIR = Path(__file__).resolve().parent / "library"
USER_DIR = Path(os.path.expanduser("~")) / ".materia" / "materials"
ENV_VAR = "MATERIA_MATERIALS"


class MaterialNotFound(KeyError):
    pass


class MaterialLibrary:
    """A lazily-populated registry of material definitions."""

    def __init__(self, search_paths: Optional[Iterable[os.PathLike]] = None,
                 include_builtin: bool = True) -> None:
        self._paths: List[Path] = []
        for p in (search_paths or []):
            self.add_search_path(p)
        env = os.environ.get(ENV_VAR, "")
        for chunk in env.split(os.pathsep):
            if chunk.strip():
                self.add_search_path(chunk.strip())
        if USER_DIR.is_dir():
            self._paths.append(USER_DIR)
        if include_builtin:
            self._paths.append(BUILTIN_DIR)
        self._cache: Dict[str, MaterialDefinition] = {}
        self._errors: Dict[str, str] = {}
        self._scanned = False

    def add_search_path(self, path: os.PathLike) -> None:
        p = Path(path).expanduser().resolve()
        if p not in self._paths:
            self._paths.insert(0, p)
            self._scanned = False

    @property
    def search_paths(self) -> List[Path]:
        return list(self._paths)

    def scan(self, force: bool = False) -> None:
        """Load every definition found on the search path."""
        if self._scanned and not force:
            return
        self._cache.clear()
        self._errors.clear()
        for directory in self._paths:
            if not directory.is_dir():
                continue
            for file in sorted(directory.glob("*.json")):
                try:
                    raw = json.loads(file.read_text())
                except Exception as exc:
                    self._errors[str(file)] = f"unreadable JSON: {exc}"
                    continue
                try:
                    definition = validate(raw, str(file))
                except MaterialValidationError as exc:
                    self._errors[str(file)] = str(exc)
                    continue
                self._cache.setdefault(definition.id, definition)
        self._scanned = True

    def ids(self) -> List[str]:
        self.scan()
        return sorted(self._cache)

    def all(self) -> List[MaterialDefinition]:
        self.scan()
        return [self._cache[k] for k in sorted(self._cache)]

    def errors(self) -> Dict[str, str]:
        self.scan()
        return dict(self._errors)

    def get(self, key: str) -> MaterialDefinition:
        """Fetch by id, alias, name or formula, case-insensitively.

        The four keys are tried in that order of specificity, and each pass
        visits materials in sorted order. Without this, a library containing
        both ``silicon`` and a derived ``silicon_strained_1pct`` - which share
        the formula ``Si`` - would resolve ``"Si"`` differently depending on
        which file happened to be scanned first. An ambiguous formula resolves
        to the material whose id is shortest, which is the base material.
        """
        self.scan()
        if key in self._cache:
            return self._cache[key]
        normalised = key.strip().lower()
        slug = normalised.replace(" ", "_").replace("-", "_")
        if slug in self._cache:
            return self._cache[slug]

        ordered = [self._cache[i] for i in sorted(self._cache)]
        for attribute in ("id", "aliases", "name", "formula"):
            matches = []
            for definition in ordered:
                if attribute == "aliases":
                    values = {a.lower() for a in definition.aliases}
                else:
                    values = {getattr(definition, attribute).lower()}
                if normalised in values:
                    matches.append(definition)
            if matches:
                return min(matches, key=lambda d: (len(d.id), d.id))

        raise MaterialNotFound(
            f"No material {key!r}. Known ids: {', '.join(sorted(self._cache))}"
        )

    def search(self, text: str) -> List[MaterialDefinition]:
        self.scan()
        t = text.strip().lower()
        if not t:
            return self.all()
        out = []
        for d in self.all():
            hay = " ".join([d.id, d.name, d.formula, d.category, " ".join(d.aliases),
                            d.prototype]).lower()
            if t in hay:
                out.append(d)
        return out

    def register(self, definition: MaterialDefinition, overwrite: bool = False) -> None:
        """Register an in-memory definition (used by plug-ins)."""
        self.scan()
        if definition.id in self._cache and not overwrite:
            raise ValueError(f"Material id {definition.id!r} already registered")
        self._cache[definition.id] = definition

    def register_raw(self, raw: dict, source: str = "<in-memory>", overwrite: bool = False) -> MaterialDefinition:
        d = validate(raw, source)
        self.register(d, overwrite=overwrite)
        return d


_DEFAULT: Optional[MaterialLibrary] = None


def default_library() -> MaterialLibrary:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MaterialLibrary()
    return _DEFAULT


def load(key: str) -> MaterialDefinition:
    return default_library().get(key)
