"""Keeping converged wavefunctions for reuse, and refusing unsafe reuse.

A converged GPAW calculation can be written to a ``.gpw`` file and used as the
starting point of another.  That is only safe when the two calculations
describe the same atoms with the same discretisation, electron count and spin
setup: then the stored wavefunctions are a better first guess and the SCF
cycle re-converges to the new ground state from there.  When any of those
differ the stored file describes a different problem, and starting from it
could steer a magnetic or otherwise multi-minimum system into the wrong
state.

Each file is therefore filed under :func:`~.spec.restart_key`, a fingerprint
of exactly the variables that must match, and looked up by it.  A
specification whose key differs never sees the file: incompatible restart data
is invalidated by construction, not by a rule someone has to remember.

Restart files are a cache on this machine, not part of the project.  They
hold wavefunctions and can be large, so the store keeps a bounded number and
total size and deletes the oldest first.  A result never depends on whether a
restart was used beyond the SCF tolerance, and every result records whether
one was.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

ENV_VAR = "MATERIA_DFT_RESTART_DIR"

MAX_ENTRIES = 16
MAX_BYTES = 2 * 1024 ** 3

MAX_FILE_BYTES = 512 * 1024 ** 2


class RestartStore:
    """A bounded, content-addressed cache of converged GPAW states."""

    def __init__(self, directory: Optional[str] = None,
                 max_entries: int = MAX_ENTRIES, max_bytes: int = MAX_BYTES) -> None:
        root = directory or os.environ.get(ENV_VAR) or str(
            Path.home() / ".materia" / "dft-restart")
        self.directory = str(root)
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()

    @property
    def _index_path(self) -> str:
        return os.path.join(self.directory, "index.json")

    def _read(self) -> List[dict]:
        try:
            with open(self._index_path) as handle:
                entries = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return []
        return [e for e in entries if isinstance(e, dict)
                and os.path.isfile(str(e.get("path", "")))]

    def _write(self, entries: List[dict]) -> None:
        os.makedirs(self.directory, exist_ok=True)
        temporary = self._index_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(entries, handle, indent=1)
        os.replace(temporary, self._index_path)

    def path_for(self, key: str, run_id: str) -> str:
        os.makedirs(self.directory, exist_ok=True)
        return os.path.join(self.directory, f"{key}-{run_id}.gpw")

    def find(self, key: str) -> Optional[Dict[str, object]]:
        """The newest restart file filed under ``key``, or ``None``."""
        with self._lock:
            matches = [e for e in self._read() if e.get("key") == key]
        if not matches:
            return None
        return max(matches, key=lambda e: float(e.get("created_unix", 0.0)))

    def register(self, key: str, run_id: str, path: str, spec_digest: str) -> Optional[dict]:
        """File a restart written by a converged run, then enforce the bounds."""
        if not os.path.isfile(path):
            return None
        size = os.path.getsize(path)
        if size > MAX_FILE_BYTES:
            os.remove(path)
            return None
        entry = {"key": key, "run_id": run_id, "path": path, "spec_digest": spec_digest,
                 "size_bytes": size, "created_unix": time.time()}
        with self._lock:
            entries = [e for e in self._read() if e.get("path") != path]
            entries.append(entry)
            entries.sort(key=lambda e: float(e.get("created_unix", 0.0)))
            while entries and (len(entries) > self.max_entries or
                               sum(int(e.get("size_bytes", 0)) for e in entries)
                               > self.max_bytes):
                oldest = entries.pop(0)
                try:
                    os.remove(str(oldest["path"]))
                except OSError:
                    pass
            self._write(entries)
        return entry

    def discard(self, key: Optional[str] = None) -> int:
        """Delete every restart file, or those filed under ``key``."""
        with self._lock:
            entries = self._read()
            keep, drop = [], []
            for entry in entries:
                (drop if key is None or entry.get("key") == key else keep).append(entry)
            for entry in drop:
                try:
                    os.remove(str(entry["path"]))
                except OSError:
                    pass
            self._write(keep)
        return len(drop)

    def entries(self) -> List[dict]:
        with self._lock:
            return list(self._read())
