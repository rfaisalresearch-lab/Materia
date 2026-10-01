"""Finding a usable GPAW, and saying precisely what is missing when there is not one.

GPAW is optional and is never imported into the Materia process.  It is driven
in its own interpreter, for three reasons: a working GPAW often lives in a
different environment from Materia (on Apple Silicon the only prebuilt
conda-forge package is an x86_64 build, so the two interpreters cannot even
share an architecture), a first-principles run has to be cancellable without
taking the application with it, and a crash inside a linked Fortran or MPI
library would otherwise be unrecoverable.

Two things have to be present and they fail independently.  The **code** is the
``gpaw`` package.  The **datasets** are the PAW setups, distributed separately
under their own licence and installed either with ``gpaw install-data`` or as
the ``gpaw-data`` conda package.  ``import gpaw`` succeeding says nothing about
whether a calculation can run, so this module never reports GPAW as operational
on that basis alone: it probes the datasets too, and reports the two states
separately with the instructions that fix each one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

INTERPRETER_ENV_VAR = "MATERIA_GPAW_PYTHON"

PROBE_TIMEOUT_S = 60.0

_PROBE = r"""
import json, sys
out = {"python": sys.executable, "python_version": sys.version.split()[0]}
try:
    import gpaw
    out["gpaw_version"] = getattr(gpaw, "__version__", "")
except BaseException as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["stage"] = "import gpaw"
    print(json.dumps(out)); raise SystemExit(0)
try:
    import ase
    out["ase_version"] = getattr(ase, "__version__", "")
except BaseException as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["stage"] = "import ase"
    print(json.dumps(out)); raise SystemExit(0)
try:
    from gpaw.mpi import world
    out["mpi_world_size"] = int(world.size)
    out["mpi"] = bool(getattr(world, "size", 1) > 1)
except BaseException:
    out["mpi_world_size"] = 1
    out["mpi"] = False
try:
    import gpaw.mpi as _m
    out["mpi_enabled_build"] = bool(getattr(_m, "have_mpi", False))
except BaseException:
    out["mpi_enabled_build"] = False
paths = []
try:
    for p in getattr(gpaw, "setup_paths", []):
        paths.append(str(p))
except BaseException:
    pass
if not paths:
    for key in ("GPAW_SETUP_PATH",):
        value = __import__("os").environ.get(key)
        if value:
            paths.extend(value.split(":"))
out["setup_paths"] = paths
import os.path, glob
found, elements, functionals = [], set(), set()
for p in paths:
    if not os.path.isdir(p):
        continue
    names = [os.path.basename(n) for n in glob.glob(os.path.join(p, "*.gz"))]
    if not names:
        continue
    found.append({"path": p, "files": len(names)})
    for n in names:
        parts = n.split(".")
        if len(parts) >= 2:
            elements.add(parts[0])
            if "basis" not in n:
                functionals.add(parts[-2])
out["dataset_dirs"] = found
out["dataset_elements"] = sorted(elements)
out["dataset_functionals"] = sorted(f for f in functionals if f.isalpha() or "PBE" in f)
print(json.dumps(out))
"""


@dataclass
class GPAWEnvironment:
    """What was found when Materia looked for a usable GPAW."""

    interpreter: Optional[str] = None
    source: str = ""
    code_available: bool = False
    datasets_available: bool = False
    gpaw_version: str = ""
    ase_version: str = ""
    python_version: str = ""
    mpi: bool = False
    mpi_world_size: int = 1
    setup_paths: List[str] = field(default_factory=list)
    dataset_dirs: List[Dict[str, Any]] = field(default_factory=list)
    dataset_elements: List[str] = field(default_factory=list)
    dataset_functionals: List[str] = field(default_factory=list)
    error: str = ""
    stage: str = ""
    probed_unix: float = 0.0

    @property
    def operational(self) -> bool:
        """Both the code and the datasets are present.

        An importable ``gpaw`` with no PAW setups cannot run anything, so it is
        not reported as operational.
        """
        return bool(self.code_available and self.datasets_available)

    @property
    def dataset_file_count(self) -> int:
        return sum(int(d.get("files", 0)) for d in self.dataset_dirs)

    def blocking_reason(self) -> str:
        """Why a calculation cannot be run, or an empty string if it can."""
        if self.operational:
            return ""
        if self.interpreter is None:
            return (
                "No interpreter with GPAW was found. Materia runs GPAW in its own "
                f"interpreter; point {INTERPRETER_ENV_VAR} at the python that has "
                "GPAW installed."
            )
        if not self.code_available:
            detail = f" The probe reported: {self.error}" if self.error else ""
            return (
                f"GPAW is not importable in {self.interpreter}.{detail}"
            )
        return (
            "GPAW is installed but no PAW datasets were found, so no calculation "
            f"can run. Searched: {', '.join(self.setup_paths) or 'no setup path'}."
        )

    def install_hint(self) -> str:
        """The instruction that fixes whatever is missing, or how to change it."""
        if self.operational:
            return (
                f"GPAW {self.gpaw_version} is being used from {self.interpreter} "
                f"({self.source}). To use a different one, point "
                f"{INTERPRETER_ENV_VAR} at its interpreter."
            )
        if not self.code_available:
            return (
                "Install GPAW into an environment of its own and point "
                f"{INTERPRETER_ENV_VAR} at that interpreter, for example:\n"
                "    conda create -n materia-gpaw -c conda-forge python=3.13 gpaw gpaw-data\n"
                f"    export {INTERPRETER_ENV_VAR}=$(conda run -n materia-gpaw which python)\n"
                "On Apple Silicon conda-forge has no osx-arm64 build; prefix the "
                "create with CONDA_SUBDIR=osx-64 to use the x86_64 build under "
                "Rosetta, or build GPAW from source against libxc."
            )
        return (
            "Install the PAW datasets, which are distributed separately under "
            "their own licence:\n"
            "    conda install -c conda-forge gpaw-data\n"
            "or\n"
            "    gpaw install-data ~/gpaw-datasets\n"
            "and make sure GPAW_SETUP_PATH points at the directory it creates."
        )

    def as_dict(self) -> dict:
        return {
            "interpreter": self.interpreter,
            "source": self.source,
            "code_available": self.code_available,
            "datasets_available": self.datasets_available,
            "operational": self.operational,
            "gpaw_version": self.gpaw_version,
            "ase_version": self.ase_version,
            "python_version": self.python_version,
            "mpi": self.mpi,
            "mpi_world_size": self.mpi_world_size,
            "setup_paths": list(self.setup_paths),
            "dataset_dirs": [dict(d) for d in self.dataset_dirs],
            "dataset_file_count": self.dataset_file_count,
            "dataset_elements": list(self.dataset_elements),
            "dataset_functionals": list(self.dataset_functionals),
            "error": self.error,
            "stage": self.stage,
            "blocking_reason": self.blocking_reason(),
            "install_hint": self.install_hint(),
            "probed_unix": self.probed_unix,
        }


def candidate_interpreters() -> List[Dict[str, str]]:
    """Interpreters that might have GPAW, most explicit first."""
    out: List[Dict[str, str]] = []
    seen = set()

    def add(path: Optional[str], source: str) -> None:
        if not path:
            return
        resolved = str(Path(path))
        if resolved in seen or not os.path.exists(resolved):
            return
        seen.add(resolved)
        out.append({"interpreter": resolved, "source": source})

    add(os.environ.get(INTERPRETER_ENV_VAR), f"{INTERPRETER_ENV_VAR} environment variable")
    script = shutil.which("gpaw")
    if script:
        add(str(Path(script).with_name("python")), "python next to the gpaw script on PATH")
    add(sys.executable, "the interpreter running Materia")
    from ...system import conda_environments, environment_python

    for env in conda_environments():
        if "gpaw" in env.name.lower():
            add(str(environment_python(env)), f"conda environment {env.name}")
    return out


def probe(interpreter: str, timeout_s: float = PROBE_TIMEOUT_S) -> Dict[str, Any]:
    """Ask one interpreter what GPAW it has, without importing anything here."""
    try:
        completed = subprocess.run(
            [interpreter, "-c", _PROBE],
            capture_output=True, text=True, timeout=timeout_s,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
    except subprocess.TimeoutExpired:
        return {"error": f"probe timed out after {timeout_s:g} s", "stage": "probe"}
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "stage": "launch"}
    text = completed.stdout.strip().splitlines()
    for line in reversed(text):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    message = (completed.stderr or completed.stdout or "").strip()
    return {"error": message[-800:] or "the probe produced no output", "stage": "probe"}


def discover(refresh: bool = False) -> GPAWEnvironment:
    """The first interpreter that can actually run GPAW, or the best diagnosis.

    An interpreter with the code but no datasets is preferred over one with
    neither, so the reported reason is the most advanced problem found rather
    than whichever candidate happened to be first.
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    best: Optional[GPAWEnvironment] = None
    for candidate in candidate_interpreters():
        data = probe(candidate["interpreter"])
        found = GPAWEnvironment(
            interpreter=candidate["interpreter"],
            source=candidate["source"],
            code_available=bool(data.get("gpaw_version")),
            gpaw_version=str(data.get("gpaw_version", "")),
            ase_version=str(data.get("ase_version", "")),
            python_version=str(data.get("python_version", "")),
            mpi=bool(data.get("mpi", False)),
            mpi_world_size=int(data.get("mpi_world_size", 1) or 1),
            setup_paths=list(data.get("setup_paths", []) or []),
            dataset_dirs=list(data.get("dataset_dirs", []) or []),
            dataset_elements=list(data.get("dataset_elements", []) or []),
            dataset_functionals=list(data.get("dataset_functionals", []) or []),
            error=str(data.get("error", "")),
            stage=str(data.get("stage", "")),
            probed_unix=time.time(),
        )
        found.datasets_available = bool(found.dataset_dirs)
        if found.operational:
            _CACHE = found
            return found
        if best is None or (found.code_available and not best.code_available):
            best = found
    _CACHE = best or GPAWEnvironment(probed_unix=time.time())
    return _CACHE


_CACHE: Optional[GPAWEnvironment] = None


def reset_cache() -> None:
    """Forget the probe result, so the next discovery runs it again."""
    global _CACHE
    _CACHE = None
