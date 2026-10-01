"""Running LAMMPS in its own process group, watching it, and stopping it.

The process starts in its own process group (a new session on POSIX, a new
process group on Windows), so cancelling it stops the whole tree and leaves no
solver running.  Standard output is read line by line for
progress: the thermodynamic rows LAMMPS prints carry the step number.  Only
the last :data:`TAIL_LINES` lines of standard output and standard error are
kept in memory, so a run that prints without end cannot exhaust it.
"""

from __future__ import annotations

import collections
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

from ...system import new_group_kwargs
from .native import SECTION_FINAL, SECTION_MAIN, THERMO_KEYWORDS

STATUS_FINISHED = "finished"
STATUS_CANCELLED = "cancelled"
STATUS_TIMEOUT = "timeout"
STATUS_NOT_STARTED = "not-started"

TAIL_LINES = 400
TERMINATE_GRACE_S = 5.0
POLL_INTERVAL_S = 0.05

_ACTIVE_LOCK = threading.Lock()
_ACTIVE: Dict[int, subprocess.Popen] = {}


def active_runs() -> int:
    with _ACTIVE_LOCK:
        return len(_ACTIVE)


def terminate_all(grace_s: float = TERMINATE_GRACE_S) -> int:
    """Stop every LAMMPS process this Materia started, and say how many."""
    with _ACTIVE_LOCK:
        running = list(_ACTIVE.values())
    for process in running:
        _stop(process)
    deadline = time.perf_counter() + max(0.0, grace_s)
    for process in running:
        try:
            process.wait(timeout=max(0.0, deadline - time.perf_counter()))
        except subprocess.TimeoutExpired:
            _stop(process, force=True)
            try:
                process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
        with _ACTIVE_LOCK:
            _ACTIVE.pop(process.pid, None)
    return len(running)


@dataclass
class ProcessRun:
    status: str
    command: List[str]
    returncode: Optional[int] = None
    wall_time_s: float = 0.0
    stdout_tail: List[str] = field(default_factory=list)
    stderr_tail: List[str] = field(default_factory=list)
    stdout_lines: int = 0
    stderr_lines: int = 0
    error: str = ""

    def as_dict(self) -> dict:
        return {"status": self.status, "command": list(self.command),
                "returncode": self.returncode, "wall_time_s": self.wall_time_s,
                "stdout_lines": self.stdout_lines, "stderr_lines": self.stderr_lines,
                "stdout_tail": "\n".join(self.stdout_tail[-80:]),
                "stderr_tail": "\n".join(self.stderr_tail[-80:]), "error": self.error}


def run_process(command: List[str], workdir: str, *,
                total_steps: int,
                progress: Optional[Callable[[float, str], None]] = None,
                cancelled: Optional[Callable[[], bool]] = None,
                timeout_s: Optional[float] = None) -> ProcessRun:
    """Run ``command`` in ``workdir`` until it exits, is cancelled or times out."""
    started = time.perf_counter()
    stdout: Deque[str] = collections.deque(maxlen=TAIL_LINES)
    stderr: Deque[str] = collections.deque(maxlen=TAIL_LINES)
    counts = {"out": 0, "err": 0}
    state: Dict[str, object] = {"section": None, "in_table": False, "step": 0}
    try:
        process = subprocess.Popen(
            command, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, bufsize=1, **new_group_kwargs(),
            env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")})
    except OSError as exc:
        return ProcessRun(STATUS_NOT_STARTED, list(command),
                          error=f"LAMMPS could not be started: {exc}")
    with _ACTIVE_LOCK:
        _ACTIVE[process.pid] = process
    lock = threading.Lock()
    latest: Dict[str, object] = {}

    def observe(line: str) -> None:
        text = line.strip()
        if text == SECTION_MAIN:
            state["section"] = "main"
            return
        if text == SECTION_FINAL:
            state["section"] = "final"
            return
        tokens = text.split()
        if tokens[:1] == ["Step"]:
            state["in_table"] = True
            return
        if text.startswith("Loop time of"):
            state["in_table"] = False
            return
        if state["in_table"] and len(tokens) == len(THERMO_KEYWORDS):
            try:
                step = int(float(tokens[0]))
                energy = float(tokens[THERMO_KEYWORDS.index("pe")])
                temperature = float(tokens[THERMO_KEYWORDS.index("temp")])
            except ValueError:
                return
            with lock:
                latest.update(step=step, energy=energy, temperature=temperature,
                              section=state["section"])

    def drain_out() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            counts["out"] += 1
            stdout.append(line.rstrip("\n")[:2000])
            observe(line)

    def drain_err() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            counts["err"] += 1
            stderr.append(line.rstrip("\n")[:2000])

    readers = [threading.Thread(target=drain_out, daemon=True),
               threading.Thread(target=drain_err, daemon=True)]
    for reader in readers:
        reader.start()
    status = STATUS_FINISHED
    reported: Tuple = ()
    try:
        while process.poll() is None:
            if cancelled is not None and cancelled():
                status = STATUS_CANCELLED
                _stop(process)
                break
            if timeout_s is not None and time.perf_counter() - started > timeout_s:
                status = STATUS_TIMEOUT
                _stop(process)
                break
            with lock:
                snapshot = dict(latest)
            if progress is not None and snapshot:
                key = (snapshot.get("section"), snapshot.get("step"))
                if key != reported:
                    reported = key
                    step = int(snapshot.get("step", 0))
                    fraction = 0.98 if snapshot.get("section") == "final" else \
                        min(0.97, step / max(1, total_steps))
                    progress(fraction, f"LAMMPS step {step}: PE = "
                                       f"{snapshot.get('energy', 0.0):.8g} eV, T = "
                                       f"{snapshot.get('temperature', 0.0):.4g} K")
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
        with _ACTIVE_LOCK:
            _ACTIVE.pop(process.pid, None)
    if status == STATUS_FINISHED and cancelled is not None and cancelled():
        status = STATUS_CANCELLED
    return ProcessRun(status, list(command), process.returncode,
                      time.perf_counter() - started, list(stdout), list(stderr),
                      counts["out"], counts["err"])


def _stop(process: subprocess.Popen, force: bool = False) -> None:
    """Stop the child and everything it started, on any operating system."""
    from ...system import stop_tree

    stop_tree(process, force)
