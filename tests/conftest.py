"""Shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def isolate_recovery(tmp_path_factory):
    previous = os.environ.get("MATERIA_RECOVERY_DIR")
    os.environ["MATERIA_RECOVERY_DIR"] = str(tmp_path_factory.mktemp("recovery"))
    yield
    if previous is None:
        os.environ.pop("MATERIA_RECOVERY_DIR", None)
    else:
        os.environ["MATERIA_RECOVERY_DIR"] = previous


@pytest.fixture(scope="session", autouse=True)
def isolate_dft_restart(tmp_path_factory):
    """Keep restart files written by tests out of the user's cache."""
    previous = os.environ.get("MATERIA_DFT_RESTART_DIR")
    os.environ["MATERIA_DFT_RESTART_DIR"] = str(tmp_path_factory.mktemp("dft-restart"))
    yield
    if previous is None:
        os.environ.pop("MATERIA_DFT_RESTART_DIR", None)
    else:
        os.environ["MATERIA_DFT_RESTART_DIR"] = previous


@pytest.fixture(scope="session")
def library():
    from materia.materials import default_library
    return default_library()


@pytest.fixture(scope="session")
def silicon(library):
    return library.get("silicon")


@pytest.fixture(scope="session")
def si111(silicon):
    from materia.structure_builder.surface import make_surface
    return make_surface(silicon, (1, 1, 1), size=(3, 3, 4), vacuum_A=14.0)


@pytest.fixture
def si111_small(silicon):
    from materia.structure_builder.surface import make_surface
    return make_surface(silicon, (1, 1, 1), size=(2, 2, 3), vacuum_A=12.0)


@pytest.fixture
def service():
    from materia.desktop_ui.service import Service
    return Service()


@pytest.fixture(scope="session")
def gpaw_environment():
    """The GPAW Materia can see, or None when there is none to drive."""
    from materia.solvers.gpaw_driver import discover

    environment = discover()
    return environment if environment.operational else None


@pytest.fixture
def needs_gpaw(gpaw_environment):
    """Skip a case that has to actually run GPAW when GPAW is not installed."""
    if gpaw_environment is None:
        from materia.solvers.gpaw_driver import discover

        pytest.skip(f"GPAW is not usable here: {discover().blocking_reason()}")
    return gpaw_environment


FAKE_LAMMPS = ROOT / "tests" / "support" / "fake_lammps.py"


def _write_launcher(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    launcher = directory / name
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_LAMMPS}" "$@"\n')
    launcher.chmod(0o755)
    return launcher


@pytest.fixture
def no_lammps(tmp_path, monkeypatch):
    """A machine with no LAMMPS anywhere Materia looks."""
    from materia.solvers.lammps import environment

    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.delenv(environment.EXECUTABLE_ENV_VAR, raising=False)
    monkeypatch.delenv(environment.PYTHON_ENV_VAR, raising=False)
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(environment, "_current_interpreter_has_module", lambda: False)
    monkeypatch.setattr(environment, "_conda_executables", lambda: [])
    environment.reset_cache()
    yield
    environment.reset_cache()


@pytest.fixture
def fake_lammps(tmp_path, monkeypatch):
    """A deterministic fake LAMMPS executable, selected explicitly.

    See ``tests/support/fake_lammps.py``.  It is not LAMMPS; tests that use it
    exercise Materia's side of the adapter, never LAMMPS's physics.
    """
    from materia.solvers.lammps import environment

    launcher = _write_launcher(tmp_path / "fake-lammps-bin", "lmp")
    monkeypatch.setenv(environment.EXECUTABLE_ENV_VAR, str(launcher))
    monkeypatch.delenv(environment.PYTHON_ENV_VAR, raising=False)
    monkeypatch.setenv("MATERIA_LAMMPS_RUN_DIR", str(tmp_path / "lammps-runs"))
    for name in ("FAKE_LAMMPS_MODE", "FAKE_LAMMPS_VERSION", "FAKE_LAMMPS_PACKAGES"):
        monkeypatch.delenv(name, raising=False)
    environment.reset_cache()
    yield launcher
    environment.reset_cache()


@pytest.fixture
def fake_lammps_module(tmp_path, monkeypatch):
    """A fake ``lammps`` Python module in its own interpreter path."""
    from materia.solvers.lammps import environment

    package = tmp_path / "fake-lammps-module" / "lammps"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(FAKE_LAMMPS.parent)!r})\n"
        "from fake_lammps import ModuleFacade as lammps\n"
        "__version__ = 20240829\n")
    monkeypatch.delenv(environment.EXECUTABLE_ENV_VAR, raising=False)
    monkeypatch.setenv(environment.PYTHON_ENV_VAR, sys.executable)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(package.parent), str(ROOT)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    monkeypatch.setenv("MATERIA_LAMMPS_RUN_DIR", str(tmp_path / "lammps-runs"))
    for name in ("FAKE_LAMMPS_MODE", "FAKE_LAMMPS_VERSION", "FAKE_LAMMPS_PACKAGES"):
        monkeypatch.delenv(name, raising=False)
    environment.reset_cache()
    yield package
    environment.reset_cache()
