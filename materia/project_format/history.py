"""Undo/redo and the project history log.

Every state-changing operation goes through :class:`History.commit`, which
stores a :class:`Snapshot` of the project as it was, together with a
human-readable description and the machine-readable parameters that produced
it.  The log is saved with the project, so a project file records not just the
final state but how it was reached.

What a snapshot has to hold
---------------------------
Reversing an edit means putting back more than one structure's coordinates.
Building a surface adds a dictionary entry and moves the active key; extracting
a region adds a region, a structure and an active region; creating a wafer
replaces the wafer.  A snapshot therefore carries two parts:

``state``
    The project's *topology*: which keys exist, which is active, the selection,
    the wafer, the regions and the key counter.  The structure and region maps
    hold **references**, not copies, so this part costs a few pointers.
``structure`` and ``structure_key``
    A deep copy of the one structure whose *contents* the operation was about
    to change, or ``None`` for an operation that only changes topology.

Holding references in ``state`` is sound because of one invariant the project
maintains: **every in-place mutation of a structure records its own snapshot
containing a copy of that structure's prior contents.**  Undo unwinds in LIFO
order, so by the time an older topology is restored, every content change made
after it has already been undone by its own entry.

Memory
------
The deep copy is what costs: tens of thousands of atoms is a few megabytes.
The undo depth is capped by :attr:`History.max_depth` and by a memory budget,
and :meth:`Snapshot.nbytes` counts only the deep copy, because the references
in ``state`` are shared with the live project rather than duplicated.  Larger
systems would need a delta encoding; that is recorded as a known limitation
rather than silently degrading.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core_model.selection import Selection
from ..core_model.structure import Structure


@dataclass
class ProjectState:
    """The project topology an undo has to put back.

    The structure and region maps hold references to the live objects, not
    copies.  See the module docstring for why that is sound and what it costs.
    """

    structures: Dict[str, Structure] = field(default_factory=dict)
    active_structure_key: Optional[str] = None
    selection: Selection = field(default_factory=Selection)
    wafer: Any = None
    regions: Dict[str, Any] = field(default_factory=dict)
    active_region_id: Optional[str] = None
    counter: int = 0

    def describe(self) -> dict:
        return {"structures": sorted(self.structures),
                "active_structure_key": self.active_structure_key,
                "regions": sorted(self.regions),
                "active_region_id": self.active_region_id,
                "wafer": self.wafer is not None,
                "n_selected": len(self.selection)}


@dataclass
class Snapshot:
    """A point-in-time record of everything an undo must restore.

    ``structure_key`` names the project slot whose contents ``structure``
    holds, so that undoing a change made to a structure that is not the active
    one restores it where it came from instead of overwriting whatever happens
    to be active.  ``state`` carries the project topology as it was.
    """

    structure: Optional[Structure]
    selection: Selection
    label: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    structure_key: Optional[str] = None
    state: Optional[ProjectState] = None

    def nbytes(self) -> int:
        if self.structure is None:
            return 0
        s = self.structure
        return int(s.positions.nbytes + s.velocities.nbytes + s.forces.nbytes
                   + s.numbers.nbytes + s.ids.nbytes + 200 * len(s))

    def describe(self) -> dict:
        out = {"label": self.label, "operation": self.operation,
               "parameters": self.parameters, "timestamp": self.timestamp,
               "n_atoms": 0 if self.structure is None else len(self.structure)}
        if self.state is not None:
            out["state"] = self.state.describe()
        return out


@dataclass
class LogEntry:
    """One line of the permanent project history (never undone away).

    ``undoable`` is false for a line that records something without an undo
    point, such as a calculation that observed a structure without changing
    it.  Undo and redo pass over those lines, so the entry marked as undone
    is always the one whose snapshot was actually restored.
    """

    operation: str
    label: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    timestamp: float = field(default_factory=time.time)
    undone: bool = False
    undoable: bool = True

    def as_dict(self) -> dict:
        return {"operation": self.operation, "label": self.label,
                "parameters": self.parameters, "result_summary": self.result_summary,
                "timestamp": self.timestamp, "undone": self.undone,
                "undoable": self.undoable}

    @staticmethod
    def from_dict(d: dict) -> "LogEntry":
        return LogEntry(**d)


class History:
    """Undo/redo stacks plus a permanent operation log."""

    def __init__(self, max_depth: int = 100, memory_budget_mb: float = 512.0) -> None:
        self.max_depth = max_depth
        self.memory_budget_bytes = int(memory_budget_mb * 1024 * 1024)
        self._undo: List[Snapshot] = []
        self._redo: List[Snapshot] = []
        self.log: List[LogEntry] = []

    def commit(self, snapshot: Snapshot, result_summary: str = "") -> None:
        """Record the state *before* an operation, and log the operation.

        The caller builds the snapshot, because only the project knows which
        structure the operation is about to change and what its topology is.
        """
        self._undo.append(snapshot)
        self._redo.clear()
        self._trim()
        self.log.append(LogEntry(operation=snapshot.operation, label=snapshot.label,
                                 parameters=dict(snapshot.parameters),
                                 result_summary=result_summary))

    def _trim(self) -> None:
        while len(self._undo) > self.max_depth:
            self._undo.pop(0)
        total = sum(s.nbytes() for s in self._undo)
        while total > self.memory_budget_bytes and len(self._undo) > 1:
            total -= self._undo[0].nbytes()
            self._undo.pop(0)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self, current: Snapshot) -> Snapshot:
        if not self._undo:
            raise IndexError("Nothing to undo")
        self._redo.append(current)
        snap = self._undo.pop()
        for entry in reversed(self.log):
            if entry.undoable and not entry.undone:
                entry.undone = True
                break
        return snap

    def redo(self, current: Snapshot) -> Snapshot:
        if not self._redo:
            raise IndexError("Nothing to redo")
        self._undo.append(current)
        snap = self._redo.pop()
        for entry in self.log:
            if entry.undoable and entry.undone:
                entry.undone = False
                break
        return snap

    def pending_undo(self) -> Optional[Snapshot]:
        """The snapshot the next undo would restore, without consuming it."""
        return self._undo[-1] if self._undo else None

    def pending_redo(self) -> Optional[Snapshot]:
        """The snapshot the next redo would restore, without consuming it."""
        return self._redo[-1] if self._redo else None

    def undo_labels(self) -> List[str]:
        return [s.label for s in self._undo]

    def redo_labels(self) -> List[str]:
        return [s.label for s in self._redo]

    def as_dict(self) -> dict:
        return {"log": [e.as_dict() for e in self.log],
                "undo_depth": len(self._undo), "redo_depth": len(self._redo),
                "max_depth": self.max_depth,
                "memory_budget_mb": self.memory_budget_bytes / (1024 * 1024)}

    def load_log(self, entries: List[dict]) -> None:
        self.log = [LogEntry.from_dict(e) for e in entries]
