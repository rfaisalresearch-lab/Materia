"""Atomic autosave storage and recovery discovery."""

from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path
from typing import Optional

from .project import Project


class RecoveryStore:
    def __init__(self, directory: Optional[str] = None) -> None:
        configured = directory or os.environ.get("MATERIA_RECOVERY_DIR")
        self.directory = Path(configured) if configured else Path.home() / ".materia" / "recovery"
        self.path = self.directory / "autosave.materia"
        self.last_error = ""

    def save(self, project: Project) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            path = project.save(str(self.path), update_modified=False)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        self.last_error = ""
        return path

    def discard(self) -> bool:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return False
        self.last_error = ""
        return True

    def load(self, library=None) -> Project:
        if not self.path.is_file():
            raise FileNotFoundError("No autosaved Materia project is available.")
        return Project.load(str(self.path), library=library)

    def info(self) -> dict:
        if not self.path.is_file():
            return {"available": False, "error": self.last_error}
        try:
            with zipfile.ZipFile(self.path, "r") as archive:
                manifest = json.loads(archive.read("manifest.json"))
            saved_at = self.path.stat().st_mtime
            return {
                "available": True,
                "path": str(self.path),
                "project_name": manifest.get("name", "Untitled project"),
                "project_modified": float(manifest.get("modified", saved_at)),
                "saved_at": saved_at,
                "saved_at_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(saved_at)),
                "structures": len(manifest.get("structures", [])),
                "scans": len(manifest.get("scans", [])),
                "history_entries": len(manifest.get("history", {}).get("log", [])),
                "error": self.last_error,
            }
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {
                "available": False,
                "corrupt": True,
                "path": str(self.path),
                "error": self.last_error,
            }
