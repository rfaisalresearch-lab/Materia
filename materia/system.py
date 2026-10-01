"""Operating-system differences in one place: macOS, Linux and Windows.

External engines run as child processes and live in conda environments whose
layout differs by platform:

* POSIX: ``<env>/bin/python`` and ``<env>/bin/<program>``;
* Windows: ``<env>\\python.exe``, programs in ``<env>\\Library\\bin``,
  ``<env>\\Scripts`` or ``<env>\\bin``, with a ``.exe`` suffix.

A run that is cancelled or times out must stop with all of its children:
POSIX uses a new session and a process-group signal, Windows a new process
group and ``taskkill /T``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

WINDOWS = os.name == "nt"
CONDA_ROOT_NAMES = ("miniforge3", "mambaforge", "miniconda3", "anaconda3", ".conda")


def conda_roots() -> List[Path]:
    """Directories that may hold conda environments, most usual first."""
    bases = [Path.home()]
    if WINDOWS:
        for variable in ("LOCALAPPDATA", "PROGRAMDATA", "ProgramFiles"):
            value = os.environ.get(variable)
            if value:
                bases.append(Path(value))
    roots = []
    for base in bases:
        for name in CONDA_ROOT_NAMES:
            envs = base / name / "envs"
            if envs.is_dir() and envs not in roots:
                roots.append(envs)
    return roots


def conda_environments(preferred: str = "") -> Iterator[Path]:
    """Every conda environment directory, the ``preferred`` name first."""
    for root in conda_roots():
        try:
            entries = [e for e in root.iterdir() if e.is_dir()]
        except OSError:
            continue
        for env in sorted(entries, key=lambda e: (e.name != preferred, e.name)):
            yield env


def environment_python(env: Path) -> Path:
    """The Python interpreter of a conda or virtual environment."""
    if WINDOWS:
        for candidate in (env / "python.exe", env / "Scripts" / "python.exe"):
            if candidate.is_file():
                return candidate
        return env / "python.exe"
    return env / "bin" / "python"


def environment_program(env: Path, name: str) -> Optional[Path]:
    """An executable installed in an environment, or None."""
    if WINDOWS:
        names = [name] if name.lower().endswith(".exe") else [name + ".exe", name]
        folders = [env / "Library" / "bin", env / "Scripts", env / "bin", env]
    else:
        names, folders = [name], [env / "bin"]
    for folder in folders:
        for item in names:
            candidate = folder / item
            if candidate.is_file() and (WINDOWS or os.access(candidate, os.X_OK)):
                return candidate
    return None


def is_executable(path: str) -> bool:
    return os.path.isfile(path) and (WINDOWS or os.access(path, os.X_OK))


def new_group_kwargs() -> Dict[str, object]:
    """Popen arguments that start a child in its own process group."""
    if WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
    return {"start_new_session": True}


def stop_tree(process: subprocess.Popen, force: bool = False) -> None:
    """Stop a child started with :func:`new_group_kwargs` and everything it started."""
    if process.poll() is not None:
        return
    if WINDOWS:
        command = ["taskkill", "/PID", str(process.pid), "/T"] + (["/F"] if force else [])
        try:
            subprocess.run(command, capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def utf8_environment() -> Dict[str, str]:
    """Environment variables that make child Python processes read and write UTF-8."""
    return {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def platform_name() -> str:
    if WINDOWS:
        return "windows"
    return "macos" if sys.platform == "darwin" else "linux"
