"""Running one GPAW calculation in its own process, and being able to stop it.

The calculation runs as a subprocess in its own session, so cancelling it
terminates the whole process group rather than leaving a detached solver
burning cores.  Progress arrives as one line per SCF iteration on the
subprocess's standard output.

Four outcomes are kept distinct, because they mean different things to anyone
reading the result: ``converged`` is a solution, ``not-converged`` is a run
that finished its iterations without meeting its tolerances, ``failed`` is a
calculation that raised, and ``cancelled`` is one the user stopped.  Only the
first is ever treated as a result.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...system import new_group_kwargs
from .environment import GPAWEnvironment

WORKER = Path(__file__).with_name("worker.py")

STATUS_CONVERGED = "converged"
STATUS_NOT_CONVERGED = "not-converged"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

TERMINATE_GRACE_S = 5.0

POLL_INTERVAL_S = 0.05

_MARKER = "@@MATERIA "

_ACTIVE_LOCK = threading.Lock()
_ACTIVE: "Dict[int, Tuple[subprocess.Popen, str]]" = {}


def _register(process: subprocess.Popen, workdir: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE[process.pid] = (process, workdir)


def _unregister(process: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.pop(process.pid, None)


def active_runs() -> int:
    """How many GPAW subprocesses this process is currently responsible for."""
    with _ACTIVE_LOCK:
        return len(_ACTIVE)


def terminate_all(grace_s: float = TERMINATE_GRACE_S,
                  clean: bool = True) -> int:
    """Stop every GPAW subprocess this process started, and say how many.

    Signals each process group, waits once for the whole set, then kills what
    is left.  Safe to call repeatedly and safe to call when nothing is running,
    because the application's shutdown path cannot know which it is.
    """
    with _ACTIVE_LOCK:
        running = list(_ACTIVE.values())
    if not running:
        return 0
    for process, _ in running:
        _stop(process)
    deadline = time.perf_counter() + max(0.0, grace_s)
    for process, _ in running:
        remaining = max(0.0, deadline - time.perf_counter())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    stopped = 0
    for process, workdir in running:
        if process.poll() is None:
            _stop(process, force=True)
            try:
                process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
        stopped += 1
        _unregister(process)
        if clean and workdir:
            shutil.rmtree(workdir, ignore_errors=True)
    return stopped


@dataclass
class GPAWRun:
    """Everything one subprocess run produced."""

    status: str
    result: Dict[str, Any] = field(default_factory=dict)
    stderr: str = ""
    returncode: Optional[int] = None
    wall_time_s: float = 0.0
    iterations: int = 0
    command: List[str] = field(default_factory=list)
    workdir: str = ""
    error: str = ""
    arrays: Dict[str, Any] = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.status == STATUS_CONVERGED

    def as_dict(self) -> dict:
        return {
            "status": self.status, "converged": self.converged,
            "returncode": self.returncode, "wall_time_s": self.wall_time_s,
            "iterations": self.iterations, "command": list(self.command),
            "error": self.error, "stderr_tail": self.stderr[-4000:],
        }


class GPAWUnavailable(RuntimeError):
    """Raised when there is no GPAW to run, carrying what is missing."""

    def __init__(self, environment: GPAWEnvironment) -> None:
        super().__init__(environment.blocking_reason())
        self.environment = environment
        self.reason = environment.blocking_reason()
        self.install_hint = environment.install_hint()


def run(structure_spec: Dict[str, Any], settings: Dict[str, Any],
        environment: GPAWEnvironment, *,
        progress: Optional[Callable[[dict], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        timeout_s: Optional[float] = None,
        keep_files: bool = False) -> GPAWRun:
    """Run one calculation and return what happened.

    ``cancelled`` is polled while the calculation runs; returning true
    terminates the process group and yields a ``cancelled`` run.
    """
    return run_job({"structure": structure_spec, "settings": settings}, environment,
                   progress=progress, cancelled=cancelled, timeout_s=timeout_s,
                   keep_files=keep_files)


def run_job(job: Dict[str, Any], environment: GPAWEnvironment, *,
            worker: Path = WORKER,
            progress: Optional[Callable[[dict], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None,
            keep_files: bool = False) -> GPAWRun:
    """Run ``worker`` on one job in its own process group.

    The job is written as JSON with ``log_path`` and ``arrays_path`` filled in
    inside a private working directory.  Arrays the worker saved there are
    loaded into :attr:`GPAWRun.arrays` before the directory is removed.
    """
    if not environment.operational:
        raise GPAWUnavailable(environment)

    workdir = tempfile.mkdtemp(prefix="materia-gpaw-")
    job_path = os.path.join(workdir, "job.json")
    result_path = os.path.join(workdir, "result.json")
    log_path = os.path.join(workdir, "gpaw.txt")
    arrays_path = os.path.join(workdir, "arrays.npz")
    with open(job_path, "w") as handle:
        json.dump({**job, "log_path": log_path, "arrays_path": arrays_path}, handle)

    command = [str(environment.interpreter), str(worker), job_path, result_path]
    started = time.perf_counter()
    stderr_chunks: List[str] = []
    iterations = 0
    was_cancelled = False

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=workdir, **new_group_kwargs(),
        env={**os.environ, "PYTHONWARNINGS": "ignore", "OMP_NUM_THREADS":
             os.environ.get("OMP_NUM_THREADS", "1")},
    )
    _register(process, workdir)

    events: "queue.Queue[dict]" = queue.Queue()

    def drain_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_chunks.append(line)

    def drain_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            if line.startswith(_MARKER):
                try:
                    events.put(json.loads(line[len(_MARKER):]))
                except json.JSONDecodeError:
                    continue

    readers = [threading.Thread(target=drain_stderr, daemon=True),
               threading.Thread(target=drain_stdout, daemon=True)]
    for reader in readers:
        reader.start()

    seen: List[dict] = []

    def pump() -> None:
        nonlocal iterations
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                return
            if event.get("event") == "scf":
                iterations = int(event.get("iteration", iterations))
                seen.append(event)
            if progress is not None:
                progress(event)

    try:
        while process.poll() is None:
            pump()
            if cancelled is not None and cancelled():
                was_cancelled = True
                _stop(process)
                break
            if timeout_s is not None and time.perf_counter() - started > timeout_s:
                _stop(process)
                stderr_chunks.append(
                    f"\nMateria stopped the calculation after {timeout_s:g} s.\n")
                break
            time.sleep(POLL_INTERVAL_S)
    finally:
        try:
            process.wait(timeout=TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            _stop(process, force=True)
            try:
                process.wait(timeout=TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                pass
        for reader in readers:
            reader.join(timeout=2.0)
        pump()
        _unregister(process)

    wall = time.perf_counter() - started
    stderr = "".join(stderr_chunks)
    result: Dict[str, Any] = {}
    if os.path.exists(result_path):
        try:
            with open(result_path) as handle:
                result = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            stderr += f"\nMateria could not read the result file: {exc}\n"
    arrays: Dict[str, Any] = {}
    if os.path.exists(arrays_path):
        try:
            import numpy as np

            with np.load(arrays_path, allow_pickle=False) as data:
                arrays = {name: np.array(data[name]) for name in data.files}
        except (OSError, ValueError) as exc:
            stderr += f"\nMateria could not read the arrays file: {exc}\n"

    if was_cancelled:
        status = STATUS_CANCELLED
    elif result.get("status") in (STATUS_CONVERGED, STATUS_NOT_CONVERGED, STATUS_FAILED):
        status = str(result["status"])
    else:
        status = STATUS_FAILED

    error = ""
    if status == STATUS_FAILED:
        error = str(result.get("error") or "").strip()
        if not error:
            tail = stderr.strip().splitlines()
            error = tail[-1] if tail else (
                f"GPAW exited with status {process.returncode} and produced no result.")

    run_record = GPAWRun(
        status=status, result=result, stderr=stderr,
        returncode=process.returncode, wall_time_s=wall,
        iterations=int(result.get("iterations", iterations) or iterations),
        command=command, workdir=workdir, error=error,
        arrays=arrays, events=seen,
    )
    if not keep_files:
        shutil.rmtree(workdir, ignore_errors=True)
        run_record.workdir = ""
    return run_record


def _stop(process: subprocess.Popen, force: bool = False) -> None:
    """Stop the child and everything it started, on any operating system."""
    from ...system import stop_tree

    stop_tree(process, force)
