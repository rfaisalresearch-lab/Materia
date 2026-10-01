"""Record an operation in the project history without an undo point.

Read-only analyses change no state, so they belong in the audit trail but not
on the undo stack.
"""

from __future__ import annotations

from typing import Optional

from ..project_format.history import LogEntry
from ..project_format.project import Project


def note_operation(project: Project, label: str, operation: str,
                   parameters: Optional[dict] = None, summary: str = "") -> None:
    project.history.log.append(LogEntry(
        operation=operation, label=label, parameters=dict(parameters or {}),
        result_summary=summary, undoable=False))
    project.touch()
