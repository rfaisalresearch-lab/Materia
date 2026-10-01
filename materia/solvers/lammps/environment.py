"""Finding the LAMMPS this machine actually has, and saying exactly what it is.

LAMMPS is never imported into the Materia process.  It is found either as an
executable (``lmp`` and its build-specific names) or as a Python interpreter
that can import the ``lammps`` module, and in both cases it is asked directly
for its version, its installed packages and its compiled styles.  A name on
``PATH`` that does not answer like LAMMPS is not treated as LAMMPS.

Search order
------------
1. ``MATERIA_LAMMPS_EXECUTABLE``: the path of a LAMMPS executable.  When set,
   it is the only candidate: an explicit choice that does not work is
   reported, never replaced by another installation.
2. ``MATERIA_LAMMPS_PYTHON``: an interpreter that can import ``lammps``.  Also
   exclusive when set.
3. Executables on ``PATH``: ``lmp``, ``lmp_serial``, ``lmp_mpi``, ``lmp_mac``,
   ``lmp_mac_mpi``, ``lmp_omp``, ``lammps``.
4. The same executable names in conda environments under ``~/miniforge3``,
   ``~/miniconda3``, ``~/anaconda3`` and ``~/.conda``, the environment named
   ``materia-lammps`` first (``conda create -n materia-lammps -c conda-forge
   lammps``).
5. The ``lammps`` module importable by the interpreter running Materia.

The first candidate that answers is used.  Every candidate tried is recorded
with what it said, so a report can show why a particular installation was or
was not chosen.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

EXECUTABLE_ENV_VAR = "MATERIA_LAMMPS_EXECUTABLE"
PYTHON_ENV_VAR = "MATERIA_LAMMPS_PYTHON"
EXECUTABLE_NAMES: Tuple[str, ...] = (
    "lmp", "lmp_serial", "lmp_mpi", "lmp_mac", "lmp_mac_mpi", "lmp_omp", "lammps")

PROBE_TIMEOUT_S = 60.0

KIND_EXECUTABLE = "executable"
KIND_MODULE = "python-module"

INSTALL_HINT = (
    "Install LAMMPS with its Python module and executable from conda-forge "
    "(conda install -c conda-forge lammps) or build it from https://www.lammps.org "
    "with the MANYBODY package enabled. Then put lmp on PATH, or set "
    f"{EXECUTABLE_ENV_VAR} to the full path of the executable, or set "
    f"{PYTHON_ENV_VAR} to a Python interpreter that can import lammps.")

_BANNER = "Large-scale Atomic/Molecular Massively Parallel Simulator - "
_MONTHS = {m: i for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
    start=1)}

_MODULE_PROBE = r"""
import json, os, shutil, sys, tempfile
out = {"python": sys.executable, "python_version": sys.version.split()[0]}
try:
    import lammps as module
except BaseException as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["stage"] = "import lammps"
    print(json.dumps(out)); raise SystemExit(0)
out["module_path"] = str(getattr(module, "__file__", ""))
out["module_version"] = str(getattr(module, "__version__", ""))
work = tempfile.mkdtemp(prefix="materia-lammps-probe-")
log = os.path.join(work, "probe.log")
try:
    lmp = module.lammps(cmdargs=["-screen", "none", "-log", log, "-nocite"])
    out["version_id"] = int(lmp.version())
    out["packages"] = [str(p) for p in lmp.installed_packages]
    styles = {}
    for category in ("atom", "integrate", "minimize", "pair", "fix", "compute", "command"):
        try:
            styles[category] = [str(s) for s in lmp.available_styles(category)]
        except BaseException:
            pass
    out["styles"] = styles
    lmp.close()
    with open(log) as handle:
        out["version_line"] = handle.readline().strip()
except BaseException as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["stage"] = "start lammps"
finally:
    shutil.rmtree(work, ignore_errors=True)
print(json.dumps(out))
"""


def version_from_log_header(line: str) -> str:
    """``LAMMPS (29 Aug 2024 - Update 1)`` to ``29 Aug 2024 - Update 1``."""
    match = re.match(r"^\s*LAMMPS \((.+)\)\s*$", line or "")
    return match.group(1).strip() if match else ""


def version_id(version: str) -> Optional[int]:
    """``29 Aug 2024 - Update 1`` to ``20240829``, or ``None``."""
    match = re.match(r"^\s*(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{4})", version or "")
    if not match or match.group(2) not in _MONTHS:
        return None
    return int(match.group(3)) * 10000 + _MONTHS[match.group(2)] * 100 + int(match.group(1))


def parse_help(text: str) -> Dict[str, Any]:
    """Version, git information, packages and styles from ``lmp -h``."""
    lines = text.splitlines()
    out: Dict[str, Any] = {"version": "", "git_info": "", "packages": [], "styles": {}}
    for line in lines:
        if _BANNER in line:
            out["version"] = line.split(_BANNER, 1)[1].strip()
            break
    for line in lines:
        if line.startswith("Git info"):
            out["git_info"] = line.strip()
            break
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("Installed packages"):
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            packages: List[str] = []
            while index < len(lines) and lines[index].strip():
                packages.extend(lines[index].split())
                index += 1
            out["packages"] = packages
            continue
        match = re.match(r"^\*\s+([A-Za-z_ ]+?) styles?:?\s*$", line)
        if match:
            category = match.group(1).strip().lower()
            index += 1
            names: List[str] = []
            while index < len(lines):
                row = lines[index].strip()
                if row.startswith("*") or row.startswith("List of") or \
                        row.startswith("Installed packages"):
                    break
                names.extend(token for token in row.split() if token)
                index += 1
            out["styles"][category] = names
            continue
        index += 1
    return out


@dataclass
class LAMMPSEnvironment:
    """What was found when Materia looked for LAMMPS."""

    kind: str = ""
    path: str = ""
    source: str = ""
    version: str = ""
    version_id: Optional[int] = None
    git_info: str = ""
    module_path: str = ""
    module_version: str = ""
    python_version: str = ""
    packages: List[str] = field(default_factory=list)
    styles: Dict[str, List[str]] = field(default_factory=dict)
    error: str = ""
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    probed_unix: float = 0.0

    @property
    def available(self) -> bool:
        return bool(self.kind and self.path and self.version and not self.error)

    def has_style(self, category: str, name: str) -> bool:
        return name in self.styles.get(category, [])

    def missing_styles(self, required: Dict[str, Iterable[str]]) -> List[str]:
        out = []
        for category, names in required.items():
            for name in names:
                if not self.has_style(category, name):
                    out.append(f"{category} style {name}")
        return out

    def identity(self) -> Dict[str, Any]:
        """What a run specification pins: the exact installation it was frozen for."""
        return {"kind": self.kind, "path": self.path, "version": self.version,
                "version_id": self.version_id, "git_info": self.git_info,
                "module_path": self.module_path, "packages": sorted(self.packages)}

    def command(self, input_name: str, log_name: str) -> List[str]:
        arguments = ["-in", input_name, "-log", log_name, "-echo", "log", "-nocite"]
        if self.kind == KIND_EXECUTABLE:
            return [self.path] + arguments
        from pathlib import Path

        worker = Path(__file__).with_name("module_worker.py")
        return [self.path, str(worker)] + arguments

    def blocking_reason(self) -> str:
        if self.available:
            return ""
        if self.error:
            return f"LAMMPS was found at {self.path or 'the configured location'} but " \
                   f"could not be used: {self.error}"
        return "LAMMPS is not installed or not visible to Materia."

    def install_hint(self) -> str:
        return INSTALL_HINT

    def as_dict(self) -> Dict[str, Any]:
        return {"available": self.available, "kind": self.kind, "path": self.path,
                "source": self.source, "version": self.version,
                "version_id": self.version_id, "git_info": self.git_info,
                "module_path": self.module_path, "module_version": self.module_version,
                "python_version": self.python_version,
                "packages": list(self.packages),
                "styles": {k: list(v) for k, v in self.styles.items()},
                "error": self.error, "blocking_reason": self.blocking_reason(),
                "install_hint": self.install_hint(),
                "candidates": [dict(c) for c in self.candidates],
                "probed_unix": self.probed_unix}


def _current_interpreter_has_module() -> bool:
    try:
        return importlib.util.find_spec("lammps") is not None
    except (ImportError, ValueError):
        return False


def _conda_executables() -> List[str]:
    """LAMMPS executables in conda environments, ``materia-lammps`` first."""
    from ...system import conda_environments, environment_program

    found: List[str] = []
    for env in conda_environments("materia-lammps"):
        for name in EXECUTABLE_NAMES:
            candidate = environment_program(env, name)
            if candidate is not None:
                found.append(str(candidate))
    return found


def candidates(search_path: Optional[str] = None,
               include_current_interpreter: bool = True) -> List[Dict[str, str]]:
    """Where LAMMPS might be, in the order it is looked for."""
    explicit = os.environ.get(EXECUTABLE_ENV_VAR, "").strip()
    if explicit:
        return [{"kind": KIND_EXECUTABLE, "path": os.path.expanduser(explicit),
                 "source": EXECUTABLE_ENV_VAR}]
    interpreter = os.environ.get(PYTHON_ENV_VAR, "").strip()
    if interpreter:
        return [{"kind": KIND_MODULE, "path": os.path.expanduser(interpreter),
                 "source": PYTHON_ENV_VAR}]
    out: List[Dict[str, str]] = []
    seen = set()
    path_value = os.environ.get("PATH", "") if search_path is None else search_path
    for name in EXECUTABLE_NAMES:
        found = shutil.which(name, path=path_value)
        if found and os.path.realpath(found) not in seen:
            seen.add(os.path.realpath(found))
            out.append({"kind": KIND_EXECUTABLE, "path": found, "source": "PATH"})
    if search_path is None:
        for found in _conda_executables():
            if os.path.realpath(found) not in seen:
                seen.add(os.path.realpath(found))
                out.append({"kind": KIND_EXECUTABLE, "path": found,
                            "source": "conda environment"})
    if include_current_interpreter and _current_interpreter_has_module():
        out.append({"kind": KIND_MODULE, "path": sys.executable,
                    "source": "the interpreter running Materia"})
    return out


def probe_executable(path: str, timeout_s: float = PROBE_TIMEOUT_S) -> Dict[str, Any]:
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return {"error": f"{path} is not an executable file."}
    try:
        done = subprocess.run([path, "-h"], capture_output=True, text=True,
                              timeout=timeout_s, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"error": f"{path} -h did not answer within {timeout_s:g} s."}
    except OSError as exc:
        return {"error": f"{path} could not be started: {exc}"}
    parsed = parse_help(done.stdout)
    if not parsed["version"]:
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or [""]
        return {"error": f"{path} -h did not print a LAMMPS version banner "
                         f"(exit status {done.returncode}). {tail[0][:300]}".strip()}
    return parsed


def probe_module(python: str, timeout_s: float = PROBE_TIMEOUT_S) -> Dict[str, Any]:
    if not (os.path.isfile(python) and os.access(python, os.X_OK)):
        return {"error": f"{python} is not an executable interpreter."}
    try:
        done = subprocess.run([python, "-c", _MODULE_PROBE], capture_output=True,
                              text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"error": f"The LAMMPS module probe in {python} did not finish within "
                         f"{timeout_s:g} s."}
    except OSError as exc:
        return {"error": f"{python} could not be started: {exc}"}
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if not lines:
        tail = done.stderr.strip().splitlines()[-1:] or [""]
        return {"error": f"The LAMMPS module probe in {python} printed nothing usable "
                         f"(exit status {done.returncode}). {tail[0][:300]}".strip()}
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {"error": f"The LAMMPS module probe in {python} printed malformed JSON: {exc}"}
    if data.get("error"):
        return {"error": f"{data['error']} (while trying to {data.get('stage', 'probe')})",
                "module_path": data.get("module_path", "")}
    version = version_from_log_header(data.get("version_line", ""))
    if not version:
        return {"error": "The lammps module started but its log did not begin with a "
                         "LAMMPS version header."}
    return {"version": version, "git_info": "", "packages": data.get("packages", []),
            "styles": data.get("styles", {}), "module_path": data.get("module_path", ""),
            "module_version": data.get("module_version", ""),
            "python_version": data.get("python_version", ""),
            "version_id_reported": data.get("version_id")}


_CACHE: Dict[Tuple, LAMMPSEnvironment] = {}


def reset_cache() -> None:
    _CACHE.clear()


def discover(refresh: bool = False, search_path: Optional[str] = None,
             include_current_interpreter: bool = True,
             timeout_s: float = PROBE_TIMEOUT_S) -> LAMMPSEnvironment:
    """The LAMMPS to use, or an unavailable environment saying why there is none."""
    key = (os.environ.get(EXECUTABLE_ENV_VAR, ""), os.environ.get(PYTHON_ENV_VAR, ""),
           os.environ.get("PATH", "") if search_path is None else search_path,
           include_current_interpreter)
    if not refresh and key in _CACHE:
        return _CACHE[key]
    tried: List[Dict[str, Any]] = []
    chosen: Optional[LAMMPSEnvironment] = None
    first_failure: Optional[LAMMPSEnvironment] = None
    for candidate in candidates(search_path, include_current_interpreter):
        if candidate["kind"] == KIND_EXECUTABLE:
            found = probe_executable(candidate["path"], timeout_s)
        else:
            found = probe_module(candidate["path"], timeout_s)
        record = {**candidate, "ok": not found.get("error"),
                  "version": found.get("version", ""), "error": found.get("error", "")}
        tried.append(record)
        environment = LAMMPSEnvironment(
            kind=candidate["kind"], path=candidate["path"], source=candidate["source"],
            version=found.get("version", ""), version_id=version_id(found.get("version", "")),
            git_info=found.get("git_info", ""), module_path=found.get("module_path", ""),
            module_version=found.get("module_version", ""),
            python_version=found.get("python_version", ""),
            packages=list(found.get("packages", [])),
            styles={k: list(v) for k, v in (found.get("styles") or {}).items()},
            error=found.get("error", ""), probed_unix=time.time())
        if environment.available:
            chosen = environment
            break
        if first_failure is None:
            first_failure = environment
    result = chosen or first_failure or LAMMPSEnvironment(probed_unix=time.time())
    result.candidates = tried
    _CACHE[key] = result
    return result
