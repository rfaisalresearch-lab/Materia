"""Project-file migrations.

Each migration takes a manifest dict at schema version ``from_version`` and
returns one at the next version.  Loading applies them in order.  A project
saved by a newer Materia than the one reading it is refused with a clear
message rather than being partially misread.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from ..version import PROJECT_SCHEMA_VERSION


class MigrationError(Exception):
    pass


MIGRATIONS: List[Tuple[str, str, Callable[[dict], dict]]] = []


def register(from_version: str, to_version: str):
    def deco(fn):
        MIGRATIONS.append((from_version, to_version, fn))
        return fn
    return deco


def _version_tuple(v: str) -> Tuple[int, ...]:
    return tuple(int(p) for p in str(v).split("."))


def migrate_manifest(manifest: dict) -> dict:
    version = str(manifest.get("schema_version", "0.0"))
    if _version_tuple(version) > _version_tuple(PROJECT_SCHEMA_VERSION):
        raise MigrationError(
            f"This project was written by a newer Materia (project schema "
            f"{version}); this build understands up to {PROJECT_SCHEMA_VERSION}. "
            "Upgrade Materia to open it. Nothing has been modified."
        )
    guard = 0
    while version != PROJECT_SCHEMA_VERSION:
        for from_v, to_v, fn in MIGRATIONS:
            if from_v == version:
                manifest = fn(manifest)
                manifest["schema_version"] = to_v
                version = to_v
                break
        else:
            raise MigrationError(
                f"No migration path from project schema {version} to "
                f"{PROJECT_SCHEMA_VERSION}."
            )
        guard += 1
        if guard > 50:
            raise MigrationError("Migration loop detected.")
    return manifest
