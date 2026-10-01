"""macOS launcher generation and environment validation."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name == "nt", reason="the macOS launcher is a POSIX shell script")
def test_macos_launcher_builds_and_self_checks(tmp_path):
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, str(root / "tools" / "make_macos_app.py"),
         "--output", str(tmp_path), "--name", "MateriaTest"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    bundle = tmp_path / "MateriaTest.app"
    launcher = bundle / "Contents" / "MacOS" / "MateriaTest"
    plist_path = bundle / "Contents" / "Info.plist"
    assert launcher.is_file()
    assert launcher.stat().st_mode & 0o111
    subprocess.run(["/bin/sh", "-n", str(launcher)], check=True)
    checked = subprocess.run(
        [str(launcher), "--check"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Materia launcher ready" in checked.stdout
    assert f"project={root}" in checked.stdout
    with plist_path.open("rb") as stream:
        info = plistlib.load(stream)
    assert info["CFBundleExecutable"] == "MateriaTest"
    assert info["CFBundleIdentifier"] == "org.materia.app"


def test_launcher_accepts_a_project_path(tmp_path):
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, str(root / "tools" / "make_macos_app.py"),
         "--output", str(tmp_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    launcher = tmp_path / "Materia.app" / "Contents" / "MacOS" / "Materia"
    text = launcher.read_text()
    assert '*.materia)' in text
    assert 'set -- --project "$PROJECT_FILE" "$@"' in text
