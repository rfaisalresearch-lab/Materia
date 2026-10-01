"""Platform layer: conda layouts and process-tree stopping on POSIX and Windows."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from materia import system as S


def make(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_windows_conda_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "WINDOWS", True)
    monkeypatch.setattr(S.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    env = tmp_path / "miniforge3" / "envs" / "materia-lammps"
    python = make(env / "python.exe")
    lmp = make(env / "Library" / "bin" / "lmp.exe")
    other = make(tmp_path / "local" / "miniconda3" / "envs" / "materia-qe" / "Library" / "bin"
                 / "pw.x.exe")
    assert S.environment_python(env) == python
    assert S.environment_program(env, "lmp") == lmp
    assert S.environment_program(env, "missing") is None
    names = [e.name for e in S.conda_environments("materia-qe")]
    assert set(names) == {"materia-lammps", "materia-qe"}
    assert S.environment_program(other.parents[2], "pw.x") == other
    assert S.new_group_kwargs() == {"creationflags": 0x200}


def test_posix_conda_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "WINDOWS", False)
    monkeypatch.setattr(S.Path, "home", classmethod(lambda cls: tmp_path))
    env = tmp_path / "miniforge3" / "envs" / "materia-qe"
    program = make(env / "bin" / "pw.x")
    assert S.environment_program(env, "pw.x") is None
    program.chmod(0o755)
    assert S.environment_program(env, "pw.x") == program
    assert S.environment_python(env) == env / "bin" / "python"
    assert S.new_group_kwargs() == {"start_new_session": True}


def test_windows_stop_uses_taskkill(monkeypatch):
    calls = []
    monkeypatch.setattr(S, "WINDOWS", True)
    monkeypatch.setattr(S.subprocess, "run", lambda command, **kw: calls.append(command))

    class Running:
        pid = 4242

        def poll(self):
            return None

    S.stop_tree(Running(), force=True)
    assert calls == [["taskkill", "/PID", "4242", "/T", "/F"]]


@pytest.mark.skipif(S.WINDOWS, reason="POSIX process groups")
def test_stop_tree_ends_children():
    child = subprocess.Popen([sys.executable, "-c", "import subprocess, sys, time;"
                              "subprocess.Popen([sys.executable, '-c', 'import time; "
                              "time.sleep(60)']); time.sleep(60)"], **S.new_group_kwargs())
    time.sleep(0.5)
    S.stop_tree(child, force=True)
    assert child.wait(timeout=10) != 0
