"""Export of a project's numerical data to one NumPy ``.npz`` archive.

Every entry is a plain numeric or text array, so the archive loads with
``numpy.load(path, allow_pickle=False)`` and needs nothing from Materia.
Entries are named by where they came from:

``structure/<field>``
    The active structure: atom ids, atomic numbers, positions (A), cell (A),
    periodic flags, fixed flags, velocities (A/fs) and forces (eV/A).
``arrays/<key>``
    Every stored large array (densities, potentials, DOS, LDOS maps and so on)
    exactly as stored.
``results/<key>`` and ``results/<key>/<field>``
    Every supported result whose value is numeric: scalars as 0-d arrays,
    arrays as they are, and each numeric field of a record such as a
    trajectory or a relaxation history.
``scans/<key>/<channel>``
    Every channel of every stored scan image.

``__manifest__`` holds, as JSON text, the schema, the Materia version, the
project name, and for each entry its unit, origin, shape, dtype, SHA-256 of
its bytes and, where there is one, the model and input digest of the result it
came from.  Values that are not numeric, or that are unsupported results, are
listed under ``skipped`` with the reason and are not written.

The archive is written to a temporary file, read back without pickling and
compared checksum by checksum with what was meant to be written, and only
then moved into place, so a failed export leaves no partial file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..version import __version__

SCHEMA = "materia.npz"
VERSION = 1
MANIFEST = "__manifest__"
PARTS = ("structure", "arrays", "results", "scans")


class NpzExportError(ValueError):
    """The export was refused or could not be verified."""


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()


def _numeric(value: Any) -> Optional[np.ndarray]:
    """``value`` as a numeric or boolean array, or ``None`` if it is not one."""
    if isinstance(value, (bool, np.bool_)):
        return np.asarray(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        return value if value.dtype.kind in "biuf" else None
    if isinstance(value, (list, tuple)) and len(value):
        try:
            array = np.asarray(value)
        except (ValueError, TypeError):
            return None
        return array if array.dtype.kind in "biuf" else None
    return None


def _structure_entries(structure) -> Dict[str, Tuple[np.ndarray, str, str]]:
    entries = {
        "structure/ids": (np.asarray(structure.ids), "", "atom ids"),
        "structure/numbers": (np.asarray(structure.numbers), "", "atomic numbers"),
        "structure/positions_A": (np.asarray(structure.positions, dtype=float), "A",
                                  "Cartesian positions"),
        "structure/cell_A": (np.asarray(structure.cell.matrix, dtype=float), "A",
                             "cell vectors as rows"),
        "structure/pbc": (np.asarray(structure.cell.pbc, dtype=bool), "", "periodic flags"),
    }
    for name, unit, text in (("fixed", "", "atoms held fixed"),
                             ("velocities", "A/fs", "velocities"),
                             ("forces", "eV/A", "forces from the last calculation")):
        value = getattr(structure, name, None)
        if value is not None:
            array = np.asarray(value)
            if array.dtype.kind in "biuf" and array.shape[:1] == (len(structure),):
                suffix = {"velocities": "_A_fs", "forces": "_eV_A"}.get(name, "")
                entries[f"structure/{name}{suffix}"] = (array, unit, text)
    return entries


def collect(project, parts=PARTS) -> Tuple[Dict[str, np.ndarray], dict]:
    """The arrays to write and the manifest describing them."""
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        raise NpzExportError(f"Unknown export part(s) {unknown}; choose from {list(PARTS)}.")
    arrays: Dict[str, np.ndarray] = {}
    entries: Dict[str, dict] = {}
    skipped: Dict[str, str] = {}

    def put(name: str, array: np.ndarray, unit: str, origin: str, **extra) -> None:
        array = np.array(array, order="C")
        arrays[name] = array
        entries[name] = {"unit": unit, "origin": origin, "shape": list(array.shape),
                         "dtype": str(array.dtype), "sha256": _sha256(array), **extra}

    if "structure" in parts:
        structure = project.structure
        if structure is None:
            skipped["structure"] = "the project has no active structure"
        else:
            for name, (array, unit, text) in _structure_entries(structure).items():
                put(name, array, unit, "active structure", description=text)
    if "arrays" in parts:
        for key, stored in sorted(project.arrays.items()):
            array = np.asarray(stored.data)
            if array.dtype.kind not in "biufc":
                skipped[f"arrays/{key}"] = f"dtype {array.dtype} is not numeric"
                continue
            put(f"arrays/{key}", array, stored.unit, "stored array",
                description=stored.description, kind=stored.kind,
                stored_sha256=stored.sha256())
    if "results" in parts:
        for key, result in sorted(project.results.items()):
            name = f"results/{key}"
            if not getattr(result, "supported", True):
                skipped[name] = "unsupported result: no value was computed"
                continue
            provenance = getattr(result, "provenance", None)
            extra = {"quantity": result.name}
            if provenance is not None:
                extra["model"] = provenance.model
                if getattr(provenance, "inputs_digest", ""):
                    extra["inputs_digest"] = provenance.inputs_digest
            value = result.value
            array = _numeric(value)
            if array is not None:
                put(name, array, result.unit, "result", **extra)
                continue
            if isinstance(value, dict):
                written = 0
                for field, item in value.items():
                    array = _numeric(item)
                    if array is None:
                        skipped[f"{name}/{field}"] = "field is not numeric"
                        continue
                    put(f"{name}/{field}", array, result.unit, "result field", **extra)
                    written += 1
                if not written:
                    skipped[name] = "record has no numeric field"
                continue
            skipped[name] = (f"value of type {type(value).__name__} is not numeric; export "
                             "structures with Export structure")
    if "scans" in parts:
        for key, scan in sorted(project.scans.items()):
            if not scan.supported or not scan.channel_names():
                skipped[f"scans/{key}"] = "unsupported scan: no channel was computed"
                continue
            for channel in scan.channel_names():
                put(f"scans/{key}/{channel}", np.asarray(scan.channel(channel)),
                    scan.units.get(channel, ""), "scan channel",
                    extent_A=[float(v) for v in scan.extent_A])
    manifest = {"schema": SCHEMA, "version": VERSION, "materia_version": __version__,
                "project": project.name,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "parts": list(parts), "entries": entries, "skipped": skipped}
    return arrays, manifest


def export(path: str, project, parts=PARTS, compressed: bool = True) -> dict:
    """Write the archive, verify it by reading it back, and return its manifest."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not path.lower().endswith(".npz"):
        raise NpzExportError("The file name must end in .npz.")
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        raise NpzExportError(f"The folder {folder} does not exist.")
    arrays, manifest = collect(project, tuple(parts))
    if not arrays:
        raise NpzExportError("There is nothing numeric to export: "
                             + "; ".join(f"{k}: {v}" for k, v in manifest["skipped"].items())
                             if manifest["skipped"] else
                             "There is nothing numeric to export.")
    if MANIFEST in arrays:
        raise NpzExportError(f"An entry is named {MANIFEST}, which is reserved.")
    payload = dict(arrays)
    payload[MANIFEST] = np.asarray(json.dumps(manifest, sort_keys=True))
    handle, temporary = tempfile.mkstemp(suffix=".npz", prefix=".materia-export-", dir=folder)
    try:
        with os.fdopen(handle, "wb") as stream:
            (np.savez_compressed if compressed else np.savez)(stream, **payload)
        verify(temporary, manifest)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return {"path": path, "entries": len(arrays), "skipped": manifest["skipped"],
            "bytes": os.path.getsize(path), "manifest": manifest}


def verify(path: str, manifest: Optional[dict] = None) -> dict:
    """Read an archive without pickling and check every entry against the manifest."""
    with np.load(path, allow_pickle=False) as archive:
        stored = json.loads(str(archive[MANIFEST]))
        if manifest is not None and stored["entries"] != manifest["entries"]:
            raise NpzExportError("The manifest read back differs from the one written.")
        names = set(archive.files) - {MANIFEST}
        if names != set(stored["entries"]):
            raise NpzExportError("The archive's entries do not match its manifest.")
        for name, entry in stored["entries"].items():
            array = archive[name]
            if (list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]
                    or _sha256(array) != entry["sha256"]):
                raise NpzExportError(f"Entry {name} does not match its manifest checksum.")
    return stored
