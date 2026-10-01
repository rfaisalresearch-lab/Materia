"""The Materia project: state, persistence and reproducibility.

On-disk format
--------------
A project is a ZIP archive with the extension ``.materia``:

``manifest.json``
    Schema version, metadata, wafer specification, region specifications,
    instrument settings, selection, history log and checkpoint index.
``structures/<key>.json``
    Full atomistic state, including ids, isotopes, charges and per-atom metadata.
``results/<key>.json``
    Solver results with their complete provenance records (values included
    unless they are large arrays).
``scans/<key>.json`` and ``scans/<key>.npz``
    Scan metadata and the raw channel arrays.
``scripts/<name>.py``
    Python scripts saved with the project.
``arrays/<key>/manifest.json`` and ``arrays/<key>/chunk-*.npy``
    Large numerical arrays such as electron densities, in checksummed chunks
    (:mod:`materia.project_format.arrays`).
``claims/<id>.json``
    Classified claims and the evidence they cite
    (:mod:`materia.provenance.classification`).
``PROVENANCE.txt``
    A plain-text reproducibility summary a reader can inspect without the app.

The format is versioned (:data:`materia.version.PROJECT_SCHEMA_VERSION`) and
:mod:`materia.project_format.migrations` upgrades older files on load.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core_model.selection import Selection
from ..core_model.structure import Structure
from ..materials.loader import MaterialLibrary, default_library
from ..multiscale.region import AtomisticRegion, RegionSpec, extract_region
from ..multiscale.wafer import Wafer, WaferSpec
from ..provenance import Convergence, Provenance, Result, digest
from ..version import PROJECT_SCHEMA_VERSION, __version__
from . import arrays as array_store
from .arrays import StoredArray
from .history import History, ProjectState, Snapshot

PROJECT_EXTENSION = ".materia"


class ProjectError(Exception):
    pass


@dataclass
class Checkpoint:
    """A named, restorable state."""

    name: str
    structure_key: str
    selection: Selection
    note: str = ""
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {"name": self.name, "structure_key": self.structure_key,
                "selection": self.selection.as_dict(), "note": self.note,
                "timestamp": self.timestamp}


class Project:
    """Everything the user is working on."""

    def __init__(self, name: str = "Untitled project",
                 library: Optional[MaterialLibrary] = None) -> None:
        self.name = name
        self.created = time.time()
        self.modified = time.time()
        self.library = library or default_library()
        self.wafer: Optional[Wafer] = None
        self.regions: Dict[str, AtomisticRegion] = {}
        self.active_region_id: Optional[str] = None
        self.structures: Dict[str, Structure] = {}
        self.active_structure_key: Optional[str] = None
        self.selection = Selection()
        self.results: Dict[str, Result] = {}
        self.arrays: Dict[str, StoredArray] = {}
        self.claims: Dict[str, Any] = {}
        self.scans: Dict[str, Any] = {}
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.scripts: Dict[str, str] = {}
        self.settings: Dict[str, Any] = {
            "stm": {}, "afm": {}, "view": {}, "solver": {},
        }
        self.history = History()
        self.notes = ""
        self._counter = 0

    def _key(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter:04d}"

    def touch(self) -> None:
        self.modified = time.time()

    @property
    def structure(self) -> Optional[Structure]:
        if self.active_structure_key is None:
            return None
        return self.structures.get(self.active_structure_key)

    @property
    def active_region(self) -> Optional[AtomisticRegion]:
        if self.active_region_id is None:
            return None
        return self.regions.get(self.active_region_id)

    def create_wafer(self, spec: WaferSpec) -> Wafer:
        material = self.library.get(spec.material_id)
        self._commit(f"Create {spec.diameter_mm:g} mm {material.name} wafer",
                     "wafer.create", spec.as_dict(), affected=None)
        self.wafer = Wafer(spec, material)
        self.touch()
        return self.wafer

    def extract_region(self, spec: Optional[RegionSpec] = None) -> AtomisticRegion:
        if self.wafer is None:
            raise ProjectError("No wafer defined. Create a wafer first.")
        region = extract_region(self.wafer, spec)
        self._commit(f"Extract atomistic region at "
                     f"({region.spec.x_mm:g}, {region.spec.y_mm:g}) mm",
                     "region.extract", region.spec.as_dict(),
                     result_summary=f"{len(region.structure)} atoms", affected=None)
        self.regions[region.region_id] = region
        self.active_region_id = region.region_id
        key = self.add_structure(region.structure, key=f"s_{region.region_id}",
                                 activate=True, log=False)
        region.structure.info["structure_key"] = key
        self.touch()
        return region

    def add_structure(self, structure: Structure, key: Optional[str] = None,
                      activate: bool = True, log: bool = True) -> str:
        """Register a structure and return its key.

        A structure already held under some key is not registered a second
        time: two keys sharing one object would drift apart the moment undo
        restored a snapshot into one of them, and every caller that reaches
        here twice with the same object means the single entry it already has.
        """
        existing = self.key_of(structure)
        if existing is not None:
            if activate and self.active_structure_key != existing:
                self.active_structure_key = existing
                self.selection = Selection()
            return existing
        key = key or self._key("struct")
        if log:
            self._commit(f"Add structure {key}", "structure.add",
                         {"key": key, "n_atoms": len(structure)}, affected=None)
        self.structures[key] = structure
        if activate:
            self.active_structure_key = key
            self.selection = Selection()
        self.touch()
        return key

    def set_structure(self, structure: Structure, label: str, operation: str,
                      parameters: Optional[dict] = None,
                      result_summary: str = "") -> None:
        """Replace the contents of the active structure, recording an undo point.

        The slot keeps the object it already has and adopts the new contents,
        so a key's structure object identity never changes for the life of the
        project and any handle pointing at it stays valid.
        """
        if self.active_structure_key is None:
            self.add_structure(structure)
            return
        self._commit(label, operation, parameters, result_summary,
                     affected=self.active_structure_key)
        self.structures[self.active_structure_key].restore_from(structure)
        self.selection = self.selection.prune(self.structure)
        self.touch()

    def key_of(self, structure: Structure) -> Optional[str]:
        """The slot holding ``structure``, by identity, or ``None`` if unheld."""
        return next((k for k, v in self.structures.items() if v is structure), None)

    def record_change(self, changed: Structure, before: Structure,
                      before_selection: Selection,
                      label: str, operation: str,
                      parameters: Optional[dict] = None,
                      result_summary: str = "") -> bool:
        """Record an undo point for a change already made to a structure in place.

        ``changed`` is the object that was mutated and ``before`` the copy of
        its prior state that the caller took.  Recording afterwards is what
        lets an operation that may refuse leave the history untouched when it
        does: nothing is logged until the change has actually happened.

        The undo point is bound to the slot that actually holds ``changed``, so
        undoing a change made to a structure that is not the active one puts it
        back where it came from.  A structure this project does not hold is the
        caller's own object: ``False`` is returned and nothing is logged, so a
        detached handle can never write into another structure's history.
        """
        key = self.key_of(changed)
        if key is None:
            return False
        snapshot = self._capture(label, operation, parameters, affected=key)
        snapshot.structure = before.copy()
        snapshot.selection = Selection(list(before_selection.ids), before_selection.query)
        snapshot.state.selection = snapshot.selection
        self.history.commit(snapshot, result_summary)
        if key == self.active_structure_key:
            self.selection = self.selection.prune(changed)
        self.touch()
        return True

    def begin(self, label: str, operation: str, parameters: Optional[dict] = None,
              result_summary: str = "") -> None:
        """Record an undo point before mutating the active structure in place."""
        self._commit(label, operation, parameters, result_summary,
                     affected=self.active_structure_key)
        self.touch()

    _ACTIVE = object()

    def _capture(self, label: str, operation: str,
                 parameters: Optional[dict] = None,
                 affected: Any = _ACTIVE) -> Snapshot:
        """The project as it is now, ready to be restored.

        ``affected`` names the slot whose *contents* the caller is about to
        change, and is deep-copied.  ``None`` means the operation only changes
        topology, so no structure needs copying at all.
        """
        key = self.active_structure_key if affected is Project._ACTIVE else affected
        structure = self.structures.get(key) if key is not None else None
        selection = Selection(list(self.selection.ids), self.selection.query)
        return Snapshot(
            structure=None if structure is None else structure.copy(),
            selection=selection,
            label=label, operation=operation, parameters=dict(parameters or {}),
            structure_key=key,
            state=ProjectState(
                structures=dict(self.structures),
                active_structure_key=self.active_structure_key,
                selection=selection,
                wafer=self.wafer,
                regions=dict(self.regions),
                active_region_id=self.active_region_id,
                counter=self._counter,
            ),
        )

    def _commit(self, label: str, operation: str,
                parameters: Optional[dict] = None, result_summary: str = "",
                affected: Any = _ACTIVE) -> None:
        """Capture the current state and log the operation about to happen."""
        self.history.commit(self._capture(label, operation, parameters, affected),
                            result_summary)

    def undo(self) -> str:
        pending = next((e for e in reversed(self.history.log)
                        if e.undoable and not e.undone), None)
        label = pending.label if pending else "current state"
        target = self.history.pending_undo()
        snap = self.history.undo(self._capture(
            label, pending.operation if pending else "",
            affected=target.structure_key if target else None))
        self._restore(snap)
        return label

    def redo(self) -> str:
        if not self.history.can_redo:
            raise IndexError("Nothing to redo")
        label = self.history.redo_labels()[-1]
        target = self.history.pending_redo()
        snap = self.history.redo(self._capture(
            label, "redo", affected=target.structure_key if target else None))
        self._restore(snap)
        return label

    def _restore(self, snap: Snapshot) -> None:
        """Put the whole project back to the state a snapshot recorded.

        Topology first, then the contents of the one structure the snapshot
        deep-copied.  Contents go back into the object the slot already holds
        rather than replacing it, so identity survives an undo and any handle
        still pointing at that structure sees the restored state.
        """
        state = snap.state
        if state is not None:
            self.structures = dict(state.structures)
            self.active_structure_key = state.active_structure_key
            self.wafer = state.wafer
            self.regions = dict(state.regions)
            self.active_region_id = state.active_region_id
            self._counter = state.counter
            self.selection = Selection(list(state.selection.ids), state.selection.query)
        key = snap.structure_key
        if snap.structure is not None and key is not None:
            if key in self.structures:
                self.structures[key].restore_from(snap.structure)
            else:
                self.structures[key] = snap.structure
        if state is None:
            self.selection = Selection(list(snap.selection.ids), snap.selection.query)
        self.touch()

    def save_checkpoint(self, name: str, note: str = "") -> Checkpoint:
        if self.structure is None:
            raise ProjectError("No active structure to checkpoint.")
        key = self._key("ckpt")
        self.structures[key] = self.structure.copy()
        cp = Checkpoint(name=name, structure_key=key,
                        selection=Selection(list(self.selection.ids), self.selection.query),
                        note=note)
        self.checkpoints[name] = cp
        from .history import LogEntry
        self.history.log.append(LogEntry(
            operation="checkpoint.save", label=f"Checkpoint '{name}'",
            parameters={"name": name}, result_summary=note, undoable=False))
        self.touch()
        return cp

    def restore_checkpoint(self, name: str) -> None:
        if name not in self.checkpoints:
            raise ProjectError(
                f"No checkpoint named {name!r}. Available: {sorted(self.checkpoints)}")
        cp = self.checkpoints[name]
        self._commit(f"Restore checkpoint '{name}'", "checkpoint.restore",
                     {"name": name}, affected=self.active_structure_key)
        if self.active_structure_key is None:
            self.add_structure(self.structures[cp.structure_key].copy(), log=False)
        else:
            self.structures[self.active_structure_key].restore_from(
                self.structures[cp.structure_key])
        self.selection = Selection(list(cp.selection.ids), cp.selection.query)
        self.touch()

    def add_result(self, key: str, result: Result) -> str:
        self.results[key] = result
        self.touch()
        return key

    def add_array(self, key: str, stored: StoredArray) -> str:
        """Hold a large array, saved in chunks rather than inside a result."""
        self.arrays[key] = stored
        self.touch()
        return key

    def add_claim(self, claim) -> str:
        """Hold a classified claim; see :mod:`materia.provenance.classification`."""
        claim.validate()
        self.claims[claim.claim_id] = claim
        self.touch()
        return claim.claim_id

    def add_scan(self, scan, key: Optional[str] = None) -> str:
        key = key or self._key("scan")
        self.scans[key] = scan
        self.touch()
        return key

    def manifest(self) -> dict:
        return {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "software_version": __version__,
            "name": self.name,
            "created": self.created,
            "modified": self.modified,
            "notes": self.notes,
            "wafer": None if self.wafer is None else self.wafer.spec.as_dict(),
            "regions": {k: v.as_dict() for k, v in self.regions.items()},
            "active_region_id": self.active_region_id,
            "structures": sorted(self.structures),
            "active_structure_key": self.active_structure_key,
            "selection": self.selection.as_dict(),
            "results": sorted(self.results),
            "arrays": sorted(self.arrays),
            "claims": sorted(self.claims),
            "scans": sorted(self.scans),
            "checkpoints": {k: v.as_dict() for k, v in self.checkpoints.items()},
            "scripts": sorted(self.scripts),
            "settings": self.settings,
            "history": self.history.as_dict(),
            "material_search_paths": [str(p) for p in self.library.search_paths],
            "counter": self._counter,
        }

    def provenance_text(self) -> str:
        lines = [
            "Materia project provenance",
            "=" * 60,
            f"project           : {self.name}",
            f"software version  : {__version__}",
            f"project schema    : {PROJECT_SCHEMA_VERSION}",
            f"created           : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created))}",
            f"modified          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.modified))}",
            "",
        ]
        if self.wafer:
            s = self.wafer.spec
            lines += [
                "Wafer",
                "-" * 60,
                f"  material        : {s.material_id}",
                f"  diameter        : {s.diameter_mm} mm",
                f"  thickness       : {s.thickness_um} um",
                f"  orientation     : ({''.join(str(v) for v in s.orientation)})",
                f"  dopant          : {s.dopant or 'none'} "
                f"{s.dopant_concentration_cm3:g} cm^-3",
                f"  temperature     : {s.temperature_K} K",
                f"  procedural seed : {s.seed}",
                "",
            ]
        if self.regions:
            lines += ["Regions", "-" * 60]
            for rid, region in self.regions.items():
                lines.append(f"  {rid}: ({region.spec.x_mm:g}, {region.spec.y_mm:g}) mm, "
                             f"{len(region.structure)} atoms")
            lines.append("")
        if self.results:
            lines += ["Results", "-" * 60]
            for key, res in self.results.items():
                lines.append(f"  {key}: {res.name} [{res.provenance.model}, "
                             f"{res.provenance.origin.value}]")
                for approx in res.provenance.approximations[:3]:
                    lines.append(f"      - {approx}")
            lines.append("")
        if self.arrays:
            lines += ["Arrays", "-" * 60]
            for key, stored in self.arrays.items():
                shape = "x".join(str(n) for n in stored.shape) or "scalar"
                lines.append(f"  {key}: {shape} {stored.unit}, {stored.kind}, "
                             f"sha256 {stored.sha256()[:16]}")
            lines.append("")
        if self.claims:
            lines += ["Claims", "-" * 60]
            for key, claim in self.claims.items():
                data = claim.as_dict()
                lines.append(f"  {key} [{data['classification']}]: {data['statement']}")
                for item in data["evidence"]:
                    lines.append(f"      evidence {item['kind']}: {item['reference']}")
            lines.append("")
        lines += ["History", "-" * 60]
        for entry in self.history.log:
            mark = "  (undone)" if entry.undone else ""
            lines.append(f"  {time.strftime('%H:%M:%S', time.localtime(entry.timestamp))} "
                         f"{entry.operation:<24} {entry.label}{mark}")
        lines += ["", "Every numeric result in this project carries a provenance record "
                      "naming the model, its approximations and its convergence state. "
                      "See results/*.json."]
        return "\n".join(lines)

    def save(self, path: str, update_modified: bool = True) -> str:
        if not path.endswith(PROJECT_EXTENSION):
            path += PROJECT_EXTENSION
        if update_modified:
            self.touch()
        tmp = path + ".tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(self.manifest(), indent=2))
            z.writestr("PROVENANCE.txt", self.provenance_text())
            for key, struct in self.structures.items():
                z.writestr(f"structures/{key}.json", json.dumps(struct.as_dict()))
            for key, res in self.results.items():
                z.writestr(f"results/{key}.json",
                           json.dumps(res.as_dict(include_value=_small(res.value)), indent=2))
            for key, stored in self.arrays.items():
                array_store.write(z, key, stored)
            for key, claim in self.claims.items():
                z.writestr(f"claims/{key}.json", json.dumps(claim.as_dict(), indent=2))
            for key, scan in self.scans.items():
                z.writestr(f"scans/{key}.json", json.dumps(scan.as_dict(), indent=2))
                buf = io.BytesIO()
                np.savez_compressed(buf, **{k: v for k, v in scan.channels.items()})
                z.writestr(f"scans/{key}.npz", buf.getvalue())
            for name, source in self.scripts.items():
                z.writestr(f"scripts/{name}", source)
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(path: str, library: Optional[MaterialLibrary] = None) -> "Project":
        from .migrations import migrate_manifest

        with zipfile.ZipFile(path, "r") as z:
            manifest = json.loads(z.read("manifest.json"))
            manifest = migrate_manifest(manifest)
            proj = Project(manifest.get("name", "Untitled project"), library=library)
            proj.created = manifest.get("created", time.time())
            proj.modified = manifest.get("modified", time.time())
            proj.notes = manifest.get("notes", "")
            proj.settings = manifest.get("settings", proj.settings)
            proj._counter = int(manifest.get("counter", 0))

            if manifest.get("wafer"):
                spec = WaferSpec.from_dict(manifest["wafer"])
                proj.wafer = Wafer(spec, proj.library.get(spec.material_id))

            for key in manifest.get("structures", []):
                name = f"structures/{key}.json"
                if name in z.namelist():
                    proj.structures[key] = Structure.from_dict(json.loads(z.read(name)))
            proj.active_structure_key = manifest.get("active_structure_key")
            proj.selection = Selection.from_dict(manifest.get("selection", {}))

            for key in manifest.get("results", []):
                name = f"results/{key}.json"
                if name in z.namelist():
                    proj.results[key] = Result.from_dict(json.loads(z.read(name)))

            for key in manifest.get("arrays", []) or []:
                proj.arrays[key] = array_store.read(z, key)

            if manifest.get("scans"):
                from ..microscopy.scan import ScanResult

                for key in manifest.get("scans", []):
                    metadata_name = f"scans/{key}.json"
                    arrays_name = f"scans/{key}.npz"
                    if metadata_name not in z.namelist() or arrays_name not in z.namelist():
                        continue
                    metadata = json.loads(z.read(metadata_name))
                    with np.load(io.BytesIO(z.read(arrays_name)), allow_pickle=False) as saved:
                        channels = {name: np.array(saved[name]) for name in saved.files}
                    convergence = metadata.get("convergence")
                    proj.scans[key] = ScanResult(
                        technique=metadata.get("technique", ""),
                        mode=metadata.get("mode", ""),
                        channels=channels,
                        primary_channel=metadata.get("primary_channel", ""),
                        units=dict(metadata.get("units") or {}),
                        extent_A=tuple(metadata.get("extent_A") or (0, 0, 0, 0)),
                        resolution=tuple(metadata.get("resolution") or (0, 0)),
                        structure=None,
                        provenance=Provenance.from_dict(metadata.get("provenance") or {}),
                        convergence=(Convergence(**convergence) if convergence else None),
                        noise_record=dict(metadata.get("noise") or {}),
                        settings=dict(metadata.get("settings") or {}),
                        wall_time_s=float(metadata.get("wall_time_s", 0.0)),
                        unsupported_reason=metadata.get("unsupported_reason", ""),
                        suggested_models=list(metadata.get("suggested_models") or []),
                    )

            if manifest.get("claims"):
                from ..provenance.classification import Claim

                for key in manifest.get("claims", []):
                    name = f"claims/{key}.json"
                    if name in z.namelist():
                        proj.claims[key] = Claim.from_dict(json.loads(z.read(name)))

            for rid, rdict in (manifest.get("regions") or {}).items():
                skey = f"s_{rid}"
                struct = proj.structures.get(skey)
                if struct is None:
                    continue
                prov = rdict.get("provenance", {})
                proj.regions[rid] = AtomisticRegion(
                    structure=struct,
                    spec=RegionSpec.from_dict(rdict.get("spec", {})),
                    wafer_context=rdict.get("wafer_context", {}),
                    applied=rdict.get("applied", {}),
                    provenance=Provenance(
                        model=prov.get("model", "unknown"),
                        fidelity=prov.get("fidelity", "tier0-structural"),
                        origin=prov.get("origin", "imported"),
                    ),
                    region_id=rid,
                )
            proj.active_region_id = manifest.get("active_region_id")

            for name, cp in (manifest.get("checkpoints") or {}).items():
                proj.checkpoints[name] = Checkpoint(
                    name=cp["name"], structure_key=cp["structure_key"],
                    selection=Selection.from_dict(cp.get("selection", {})),
                    note=cp.get("note", ""), timestamp=cp.get("timestamp", time.time()))

            for entry in z.namelist():
                if entry.startswith("scripts/"):
                    proj.scripts[entry.split("/", 1)[1]] = z.read(entry).decode()

            proj.history.load_log(manifest.get("history", {}).get("log", []))
        return proj


def _small(value: Any) -> bool:
    """Whether a result value is small enough to inline in the project file."""
    try:
        if isinstance(value, np.ndarray):
            return value.size <= 20000
        if isinstance(value, dict):
            return all(_small(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return len(value) <= 20000
    except Exception:
        return False
    return not isinstance(value, Structure)
