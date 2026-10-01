"""Driving LAMMPS as a real, optional, out-of-process classical solver.

``environment``  find the installed executable or Python module and its version
``spec``         the frozen, versioned, unit-explicit run specification
``native``       the LAMMPS data file and input script written from a specification
``runner``       the process: progress, cancellation, time limit and bounded logs
``parse``        strict readers for the log, thermodynamic tables and dumps
``run``          the whole pipeline, and results only when every check passes
``records``      storing runs in a project, their state, and applying them
``units``        every conversion between LAMMPS metal units and Materia units
"""

from .environment import (
    EXECUTABLE_ENV_VAR,
    INSTALL_HINT,
    PYTHON_ENV_VAR,
    LAMMPSEnvironment,
    discover,
    reset_cache,
)
from .records import ApplyRefused
from .run import LAMMPSOutcome, execute
from .spec import LAMMPSRunSpec, SpecError, SpecRefused, SpecReport, build, check

__all__ = [
    "EXECUTABLE_ENV_VAR", "PYTHON_ENV_VAR", "INSTALL_HINT", "LAMMPSEnvironment",
    "discover", "reset_cache", "LAMMPSRunSpec", "SpecError", "SpecRefused", "SpecReport",
    "build", "check", "execute", "LAMMPSOutcome", "ApplyRefused",
]
