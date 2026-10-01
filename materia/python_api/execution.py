"""Script execution for the embedded Python laboratory.

This runs **real Python**.  There is no command whitelist and no simulated
console: the embedded editor executes the code you type against the same API
the graphical interface uses.

Execution modes
---------------
``restricted`` (default)
    The script runs in this process with a curated ``builtins`` mapping and an
    import hook that allows only the scientific modules listed in
    :data:`ALLOWED_MODULES`.  ``open``, ``eval``, ``exec``, ``compile``,
    ``input`` and the ``os``/``sys``/``subprocess``/``socket`` modules are not
    reachable from the script namespace.

    **This is a guard rail, not a security boundary.**  CPython cannot be
    sandboxed from inside: a determined script can reach the interpreter
    internals through object introspection.  Restricted mode stops accidents
    and casual mistakes.  It does not make it safe to run untrusted code.
    Treat scripts the way you treat any other program you are about to run.

``trusted``
    Full builtins and unrestricted imports.  The interface requires an
    explicit, per-session confirmation before selecting this mode, and the
    confirmation is recorded in the project history.

``subprocess``
    The script runs in a separate Python interpreter against a project loaded
    from disk, and the modified project is written back.  This gives real
    process isolation (and a hard timeout that actually works), at the cost of
    not sharing live objects with the interface.  It is what the ``materia
    run`` command uses for batch work.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .api import Lab, build_namespace

MODES = ("restricted", "trusted", "subprocess")

ALLOWED_MODULES = {
    "materia", "numpy", "scipy", "math", "cmath", "statistics", "random",
    "itertools", "functools", "collections", "dataclasses", "typing", "json",
    "re", "time", "datetime", "decimal", "fractions", "heapq", "bisect",
    "copy", "enum", "abc", "textwrap", "pprint", "string", "warnings",
    "ase", "h5py",
}

_SAFE_BUILTIN_NAMES = (
    "abs all any ascii bin bool bytearray bytes callable chr complex dict dir "
    "divmod enumerate filter float format frozenset getattr hasattr hash hex id "
    "int isinstance issubclass iter len list map max min next object oct ord pow "
    "print range repr reversed round set setattr slice sorted staticmethod str sum "
    "super tuple type vars zip classmethod property True False None "
    "ArithmeticError AssertionError AttributeError BaseException BufferError "
    "EOFError Exception FloatingPointError GeneratorExit ImportError IndentationError "
    "IndexError KeyError KeyboardInterrupt LookupError MemoryError NameError "
    "NotImplementedError OverflowError RecursionError ReferenceError RuntimeError "
    "StopIteration SyntaxError SystemError TypeError UnboundLocalError ValueError "
    "ZeroDivisionError"
).split()


class ScriptTimeout(Exception):
    pass


@dataclass
class ScriptResult:
    """Everything a script run produced."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    traceback: str = ""
    duration_s: float = 0.0
    view_items: List[dict] = field(default_factory=list)
    result_repr: str = ""
    mode: str = "restricted"

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "stdout": self.stdout, "stderr": self.stderr,
            "error": self.error, "traceback": self.traceback,
            "duration_s": self.duration_s, "view_items": self.view_items,
            "result_repr": self.result_repr, "mode": self.mode,
        }


def _restricted_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(
            f"Importing {name!r} is not allowed in restricted mode. "
            f"Allowed roots: {', '.join(sorted(ALLOWED_MODULES))}. "
            "Switch the console to trusted mode if you need it, and understand "
            "that a trusted script can do anything your user account can."
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


def _restricted_builtins() -> Dict[str, Any]:
    d = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)}
    d["__import__"] = _restricted_import
    d["__build_class__"] = builtins.__build_class__
    d["__name__"] = "materia_script"
    return d


class ScriptRunner:
    """Executes user scripts against a :class:`~materia.python_api.api.Lab`."""

    def __init__(self, lab: Lab, mode: str = "restricted") -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown execution mode {mode!r}; one of {MODES}")
        self.lab = lab
        self.mode = mode
        self.globals: Dict[str, Any] = {}
        self.reset_namespace()

    def reset_namespace(self) -> None:
        self.globals = build_namespace(self.lab)
        self.globals["__name__"] = "materia_script"
        self.globals["__builtins__"] = (
            _restricted_builtins() if self.mode == "restricted" else builtins.__dict__)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown execution mode {mode!r}; one of {MODES}")
        self.mode = mode
        self.reset_namespace()

    def run(self, code: str, echo_last_expression: bool = True) -> ScriptResult:
        """Execute ``code``, capturing output and the value of a final expression."""
        if self.mode == "subprocess":
            return run_in_subprocess(code, self.lab)
        t0 = time.perf_counter()
        out, err = io.StringIO(), io.StringIO()
        start_items = len(self.lab.view.items)
        result_repr = ""
        ok, error, tb = True, "", ""

        body, tail = _split_last_expression(code) if echo_last_expression else (code, None)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                if body.strip():
                    exec(compile(body, "<materia>", "exec"), self.globals)
                if tail:
                    value = eval(compile(tail, "<materia>", "eval"), self.globals)
                    if value is not None:
                        result_repr = repr(value)[:8000]
                        print(result_repr)
        except BaseException as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            tb = _strip_internal_frames(tb)
        duration = time.perf_counter() - t0
        return ScriptResult(
            ok=ok, stdout=out.getvalue(), stderr=err.getvalue(), error=error,
            traceback=tb, duration_s=duration,
            view_items=self.lab.view.items[start_items:],
            result_repr=result_repr, mode=self.mode,
        )


def _split_last_expression(code: str):
    """Split trailing expression so the console can echo its value, like a REPL."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code, None
    last = tree.body[-1]
    if len(tree.body) > 1:
        previous = tree.body[-2]
        prev_end = getattr(previous, "end_lineno", previous.lineno)
        if prev_end >= last.lineno:
            return code, None
    lines = code.splitlines()
    start = last.lineno - 1
    body = "\n".join(lines[:start])
    tail = textwrap.dedent("\n".join(lines[start:]))
    return body, tail


def _strip_internal_frames(tb: str) -> str:
    keep, skipping = [], False
    for line in tb.splitlines():
        if "python_api/execution.py" in line:
            skipping = True
            continue
        if skipping and line.startswith("    "):
            skipping = False
            continue
        skipping = False
        keep.append(line)
    return "\n".join(keep)


_SUBPROCESS_WRAPPER = r'''
import json, sys, time, io, contextlib, traceback
sys.path.insert(0, {root!r})
from materia.project_format import Project
from materia.python_api.api import Lab, build_namespace

project_path = {project!r}
script_path = {script!r}
out_path = {out!r}

project = Project.load(project_path) if project_path else Project()
lab = Lab(project)
g = build_namespace(lab)
g["__name__"] = "materia_script"
code = open(script_path).read()
buf_out, buf_err = io.StringIO(), io.StringIO()
ok, error, tb = True, "", ""
t0 = time.perf_counter()
try:
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        exec(compile(code, script_path, "exec"), g)
except BaseException as exc:
    ok = False
    error = "{{}}: {{}}".format(type(exc).__name__, exc)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
duration = time.perf_counter() - t0
if project_path:
    project.save(project_path)
json.dump({{"ok": ok, "stdout": buf_out.getvalue(), "stderr": buf_err.getvalue(),
           "error": error, "traceback": tb, "duration_s": duration,
           "view_items": lab.view.items}}, open(out_path, "w"), default=str)
'''


def run_in_subprocess(code: str, lab: Optional[Lab] = None, timeout_s: float = 600.0,
                      project_path: Optional[str] = None,
                      python: Optional[str] = None) -> ScriptResult:
    """Run a script in a separate interpreter with a hard timeout.

    When ``project_path`` is given the project is loaded there, modified by the
    script and written back, which is how batch runs stay reproducible.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "script.py")
        out = os.path.join(tmp, "result.json")
        runner = os.path.join(tmp, "runner.py")
        with open(script, "w") as fh:
            fh.write(code)
        with open(runner, "w") as fh:
            fh.write(_SUBPROCESS_WRAPPER.format(
                root=root, project=project_path, script=script, out=out))
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([python or sys.executable, runner],
                                  capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return ScriptResult(ok=False, mode="subprocess",
                                error=f"Script exceeded the {timeout_s:g} s time limit "
                                      "and was terminated.",
                                duration_s=time.perf_counter() - t0)
        if not os.path.exists(out):
            return ScriptResult(ok=False, mode="subprocess",
                                error="Script process did not produce a result.",
                                stdout=proc.stdout, stderr=proc.stderr,
                                duration_s=time.perf_counter() - t0)
        data = json.loads(open(out).read())
    return ScriptResult(mode="subprocess", **data)
