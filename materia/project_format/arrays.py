"""Large numerical arrays stored in chunks inside a project file.

Electron densities, spin densities and electrostatic potentials are grids of
10^5 to 10^8 numbers.  Written into a JSON result they would make the project
unreadable, and the ordinary result writer drops any value above 20,000
elements.  They are stored here instead, as a directory per array:

``arrays/<key>/manifest.json``
    dtype, shape, chunk layout, unit, description, grid metadata, and the
    SHA-256 of every chunk and of the whole array.
``arrays/<key>/chunk-00000.npy`` ...
    Consecutive slabs along the first axis, each at most :data:`CHUNK_BYTES`,
    in NumPy's documented ``.npy`` format.

Values are stored at full precision; reading verifies every checksum and
refuses an array whose chunks do not add up, rather than returning a
plausible-looking grid with a hole in it.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

ARRAY_SCHEMA = "materia.array"
ARRAY_VERSION = 1

CHUNK_BYTES = 4 * 1024 * 1024


class ArrayStoreError(ValueError):
    """A stored array that is missing, truncated or does not match its checksum."""


@dataclass
class StoredArray:
    """One array and what it means."""

    data: np.ndarray
    unit: str
    description: str = ""
    kind: str = "table"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> List[int]:
        return [int(n) for n in np.shape(self.data)]

    @property
    def nbytes(self) -> int:
        return int(np.asarray(self.data).nbytes)

    def sha256(self) -> str:
        array = np.ascontiguousarray(self.data)
        return hashlib.sha256(array.tobytes()).hexdigest()

    def summary(self) -> dict:
        array = np.asarray(self.data)
        finite = array[np.isfinite(array)] if array.size else array
        return {
            "shape": self.shape, "dtype": str(array.dtype), "unit": self.unit,
            "kind": self.kind, "description": self.description,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "nbytes": self.nbytes, "meta": dict(self.meta),
        }


def _npy(block: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(block), allow_pickle=False)
    return buffer.getvalue()


def write(archive: zipfile.ZipFile, key: str, stored: StoredArray) -> dict:
    """Write one array into an open project archive and return its manifest."""
    array = np.ascontiguousarray(stored.data)
    base = f"arrays/{key}"
    chunks: List[dict] = []
    if array.ndim == 0 or array.size == 0:
        payload = _npy(array)
        name = f"{base}/chunk-00000.npy"
        archive.writestr(name, payload)
        chunks.append({"file": "chunk-00000.npy", "start": 0, "stop": int(array.shape[0])
                       if array.ndim else 0,
                       "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes())
                       .hexdigest()})
    else:
        row_bytes = max(1, int(array[0:1].nbytes))
        rows = max(1, CHUNK_BYTES // row_bytes)
        for index, start in enumerate(range(0, array.shape[0], rows)):
            block = array[start:start + rows]
            name = f"chunk-{index:05d}.npy"
            archive.writestr(f"{base}/{name}", _npy(block))
            chunks.append({"file": name, "start": int(start),
                           "stop": int(start + block.shape[0]),
                           "sha256": hashlib.sha256(
                               np.ascontiguousarray(block).tobytes()).hexdigest()})
    manifest = {
        "schema": ARRAY_SCHEMA, "version": ARRAY_VERSION, "key": key,
        "dtype": str(array.dtype), "shape": [int(n) for n in array.shape],
        "chunk_axis": 0, "chunks": chunks, "sha256": stored.sha256(),
        "unit": stored.unit, "description": stored.description, "kind": stored.kind,
        "meta": _json_safe(stored.meta),
    }
    archive.writestr(f"{base}/manifest.json", json.dumps(manifest, indent=2))
    return manifest


def read(archive: zipfile.ZipFile, key: str) -> StoredArray:
    """Read and verify one array from an open project archive."""
    base = f"arrays/{key}"
    try:
        manifest = json.loads(archive.read(f"{base}/manifest.json"))
    except KeyError:
        raise ArrayStoreError(f"The project lists array {key} but does not contain "
                              "its manifest.") from None
    if manifest.get("schema") != ARRAY_SCHEMA:
        raise ArrayStoreError(f"{base}/manifest.json is not an array manifest.")
    if int(manifest.get("version", 0)) > ARRAY_VERSION:
        raise ArrayStoreError(
            f"Array {key} was written in array format version {manifest.get('version')}; "
            f"this Materia reads up to {ARRAY_VERSION}.")
    shape = tuple(int(n) for n in manifest["shape"])
    dtype = np.dtype(manifest["dtype"])
    blocks = []
    expected_start = 0
    for chunk in manifest["chunks"]:
        try:
            raw = archive.read(f"{base}/{chunk['file']}")
        except KeyError:
            raise ArrayStoreError(f"Array {key} is missing {chunk['file']}.") from None
        try:
            block = np.load(io.BytesIO(raw), allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise ArrayStoreError(
                f"Chunk {chunk['file']} of array {key} is not a readable NumPy array: "
                f"{exc}") from None
        if hashlib.sha256(np.ascontiguousarray(block).tobytes()).hexdigest() != chunk["sha256"]:
            raise ArrayStoreError(f"Chunk {chunk['file']} of array {key} does not match "
                                  "its SHA-256; the file is damaged.")
        if len(shape) and int(chunk["start"]) != expected_start:
            raise ArrayStoreError(f"Array {key} has a gap before {chunk['file']}.")
        expected_start = int(chunk["stop"])
        blocks.append(block)
    if len(shape) == 0 or (blocks and blocks[0].size == 0 and len(blocks) == 1):
        data = blocks[0].reshape(shape).astype(dtype, copy=False)
    else:
        data = np.concatenate(blocks, axis=0).astype(dtype, copy=False)
    if tuple(data.shape) != shape:
        raise ArrayStoreError(f"Array {key} has shape {data.shape}, its manifest says "
                              f"{shape}.")
    stored = StoredArray(data=data, unit=str(manifest.get("unit", "")),
                         description=str(manifest.get("description", "")),
                         kind=str(manifest.get("kind", "table")),
                         meta=dict(manifest.get("meta", {}) or {}))
    if stored.sha256() != manifest.get("sha256"):
        raise ArrayStoreError(f"Array {key} does not match its SHA-256 after assembly.")
    return stored


def _json_safe(value: Any) -> Any:
    from ..provenance.record import _jsonable

    return _jsonable(value)
