"""Background job queue with progress and cancellation.

Scans and solver runs can take seconds to minutes.  They run on worker
threads so the interface stays responsive, report progress, and can be
cancelled.  Every job keeps its result until it is explicitly discarded, so
the interface can re-read it without recomputing.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class Cancelled(Exception):
    """Raised inside a job when the user cancels it."""


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    result: Any = None
    error: str = ""
    traceback: str = ""
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    owner: Any = None
    cancel_reason: str = ""

    def as_dict(self, include_result: bool = False) -> dict:
        d = {
            "id": self.id, "kind": self.kind, "label": self.label,
            "status": self.status, "progress": self.progress, "message": self.message,
            "created": self.created, "started": self.started, "finished": self.finished,
            "error": self.error, "traceback": self.traceback,
            "cancel_reason": self.cancel_reason,
            "elapsed_s": (None if self.started is None else
                          (self.finished or time.time()) - self.started),
        }
        if include_result:
            d["result"] = self.result
        return d


class JobQueue:
    def __init__(self, max_history: int = 100) -> None:
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        self._shutdown = threading.Event()
        self.max_history = max_history

    @property
    def is_shut_down(self) -> bool:
        return self._shutdown.is_set()

    def submit(self, kind: str, label: str,
               fn: Callable[[Job], Any], owner: Any = None) -> Job:
        """Queue work on a worker thread.

        ``owner`` marks what the job belongs to -- a project, typically -- so
        that the queue can cancel everything belonging to something that is
        being replaced, and so that a finished job can be checked against the
        thing it was started for.
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label, owner=owner)
        if self._shutdown.is_set():
            job.status = "cancelled"
            job.cancel_flag.set()
            job.cancel_reason = "Materia is shutting down; no new work was started."
            job.message = job.cancel_reason
            job.created = job.finished = time.time()
            with self._lock:
                self._jobs[job.id] = job
                self._order.append(job.id)
            return job
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self.max_history:
                self._jobs.pop(self._order.pop(0), None)

        def runner():
            if self._shutdown.is_set():
                job.status = "cancelled"
                job.cancel_reason = (
                    "Materia is shutting down; no new work was started.")
                job.message = job.cancel_reason
                job.finished = time.time()
                return
            job.status = "running"
            job.started = time.time()
            try:
                job.result = fn(job)
                if job.cancel_flag.is_set():
                    job.status = "cancelled"
                    if not job.message:
                        job.message = job.cancel_reason or "Cancelled."
                else:
                    job.status = "done"
                job.progress = 1.0
            except Cancelled:
                job.status = "cancelled"
                job.message = "Cancelled by the user."
            except BaseException as exc:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()
            finally:
                job.finished = time.time()

        thread = threading.Thread(target=runner, name=f"job-{job.id}", daemon=True)
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)
        thread.start()
        return job

    def shutdown(self, grace_s: float = 10.0, clean: bool = True) -> dict:
        """Stop everything, and do not come back.

        Idempotent, because it is called from more than one closing path and
        because a second close must not deadlock on the first.  New work is
        refused from the moment it starts; every queued or running job is
        cancelled; every GPAW and LAMMPS subprocess group is signalled, waited for, and
        then killed; and any job still unfinished when the grace period expires
        is recorded as cancelled with the reason, never as done.
        """
        from ..solvers.gpaw_driver import runner as gpaw_runner
        from ..solvers.lammps import runner as lammps_runner

        first = not self._shutdown.is_set()
        self._shutdown.set()
        reason = "Cancelled because Materia is shutting down."
        with self._lock:
            jobs = list(self._jobs.values())
            threads = list(self._threads)
        cancelled = []
        for job in jobs:
            if job.status in ("queued", "running"):
                job.cancel_flag.set()
                job.cancel_reason = reason
                job.message = reason
                cancelled.append(job.id)

        terminated = gpaw_runner.terminate_all(clean=clean)
        terminated += lammps_runner.terminate_all()

        deadline = time.time() + max(0.0, grace_s)
        for thread in threads:
            remaining = max(0.0, deadline - time.time())
            thread.join(timeout=remaining)

        terminated += gpaw_runner.terminate_all(clean=clean)
        terminated += lammps_runner.terminate_all()

        stranded = []
        for job in jobs:
            if job.status in ("queued", "running"):
                job.status = "cancelled"
                job.cancel_reason = reason
                job.message = reason
                job.finished = job.finished or time.time()
                stranded.append(job.id)
        return {"first_call": first, "cancelled": cancelled,
                "subprocesses_terminated": terminated, "stranded": stranded}

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str, reason: str = "Cancelled by the user.") -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False
        job.cancel_flag.set()
        job.cancel_reason = reason
        job.message = reason
        return True

    def cancel_owned_by(self, owner: Any,
                        reason: str = "Cancelled because its project was replaced."
                        ) -> List[str]:
        """Cancel every unfinished job belonging to ``owner``, by identity.

        Returns the ids that were still running.  Used when a project is
        replaced: a calculation started for a project that no longer exists has
        nowhere to put its answer.
        """
        cancelled = []
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.owner is owner and self.cancel(job.id, reason):
                cancelled.append(job.id)
        return cancelled

    def list(self, limit: int = 40) -> List[dict]:
        with self._lock:
            ids = self._order[-limit:]
        return [self._jobs[i].as_dict() for i in reversed(ids) if i in self._jobs]
