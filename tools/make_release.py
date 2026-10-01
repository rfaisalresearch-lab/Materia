#!/usr/bin/env python3
"""Build the downloadable macOS and Windows packages of Materia.

Both packages hold the same source, taken from the committed tree with
``git archive`` so that nothing untracked or machine-specific is shipped, and
an installer for their platform:

* ``Materia-<version>-windows.zip``: double-click ``Install Materia.cmd``. It
  finds Python 3.10 to 3.13 (offering to install Python 3.12 with winget when
  none is present), copies Materia to ``%LOCALAPPDATA%\\Materia``, creates a
  virtual environment, installs Materia with its desktop window and every
  optional engine that publishes Windows wheels, and adds Start Menu and
  Desktop shortcuts.
* ``Materia-<version>-macos.zip``: double-click ``Install Materia.command``. It
  does the same in ``~/Applications/Materia`` and builds ``Materia.app``.

Usage:
    python tools/make_release.py [--output DIR]
"""

from __future__ import annotations

import argparse
import io
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

EXCLUDED = ("Materia.app/",)
WINDOWS_ENGINES = ["rdkit", "openmm", "pdbfixer", "mkl", "devsim", "xraydb", "phonopy",
                   "radioactivedecay", "particle"]
MACOS_ENGINES = [e for e in WINDOWS_ENGINES if e != "mkl"] + ["pyscf", "pyscf-properties", "tblite",
                                                            "matscipy"]

INSTALL_PS1 = r"""$ErrorActionPreference = 'Stop'
$Version = '__VERSION__'
$Source = Split-Path -Parent $MyInvocation.MyCommand.Path
$Target = if ($env:MATERIA_INSTALL_DIR) { $env:MATERIA_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "Materia\$Version" }
Write-Host "Installing Materia $Version into $Target"

function Find-Python {
    foreach ($minor in '3.12', '3.13', '3.11', '3.10') {
        try {
            $path = & py "-$minor" -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $path) { return $path.Trim() }
        } catch { }
    }
    foreach ($name in 'python', 'python3') {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command -and $command.Source -notmatch 'WindowsApps') {
            $ok = & $command.Source -c 'import sys; print(int((3, 10) <= sys.version_info[:2] <= (3, 13)))'
            if ($ok -eq '1') { return $command.Source }
        }
    }
    return $null
}

$Python = Find-Python
if (-not $Python) {
    Write-Host 'Python 3.10 to 3.13 was not found.'
    $answer = Read-Host 'Install Python 3.12 from the Microsoft winget catalogue now? (y/n)'
    if ($answer -ne 'y') { throw 'Materia needs Python. Install it from https://www.python.org and run this installer again.' }
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $Python = Find-Python
    if (-not $Python) { throw 'Python was installed but cannot be found yet. Open a new window and run the installer again.' }
}
Write-Host "Using Python at $Python"

if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -Recurse -Force (Join-Path $Source 'materia-source\*') $Target
Copy-Item -Force (Join-Path $Source 'Materia.ico') $Target

& $Python -m venv (Join-Path $Target '.venv')
$VenvPython = Join-Path $Target '.venv\Scripts\python.exe'
$Windowed = Join-Path $Target '.venv\Scripts\pythonw.exe'
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install "$Target[adapters,bigdata,desktop]"
if ($LASTEXITCODE -ne 0) { throw 'Installing Materia and its core dependencies failed.' }

$Installed = @()
$Missing = @()
foreach ($engine in @(__ENGINES__)) {
    & $VenvPython -m pip install $engine 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $Installed += $engine } else { $Missing += $engine }
}

$Shell = New-Object -ComObject WScript.Shell
$Menu = Join-Path ([Environment]::GetFolderPath('Programs')) 'Materia.lnk'
$Desk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Materia.lnk'
foreach ($path in @($Menu, $Desk)) {
    $link = $Shell.CreateShortcut($path)
    $link.TargetPath = $Windowed
    $link.Arguments = '-X utf8 -m materia.desktop_ui.desktop'
    $link.WorkingDirectory = $Target
    $link.IconLocation = (Join-Path $Target 'Materia.ico')
    $link.Description = "Materia $Version"
    $link.Save()
}
Set-Content -Path (Join-Path $Target 'Materia.cmd') -Value "@echo off`r`n`"%~dp0.venv\Scripts\python.exe`" -X utf8 -m materia.cli %*"

Write-Host ''
Write-Host "Materia $Version is installed. Start it from the Start Menu or the Desktop shortcut."
Write-Host ("Optional engines installed: " + ($Installed -join ', '))
if ($Missing.Count -gt 0) {
    Write-Host ("Not available as Windows packages, refused by Materia until installed: " + ($Missing -join ', '))
}
Write-Host 'PySCF, tblite, GPAW and Quantum ESPRESSO have no native Windows builds; use them through WSL.'
"""

INSTALL_CMD = "@echo off\r\npowershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0install.ps1\"\r\npause\r\n"

UNINSTALL_PS1 = r"""$Version = '__VERSION__'
$Target = Join-Path $env:LOCALAPPDATA "Materia\$Version"
foreach ($path in @((Join-Path ([Environment]::GetFolderPath('Programs')) 'Materia.lnk'),
                    (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Materia.lnk'))) {
    if (Test-Path $path) { Remove-Item -Force $path }
}
if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
Write-Host "Materia $Version was removed. Projects you saved elsewhere are untouched."
"""

UNINSTALL_CMD = ("@echo off\r\npowershell -NoProfile -ExecutionPolicy Bypass -File "
                 "\"%~dp0uninstall.ps1\"\r\npause\r\n")

INSTALL_COMMAND = r"""#!/bin/bash
set -euo pipefail
VERSION='__VERSION__'
SOURCE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${MATERIA_INSTALL_DIR:-$HOME/Applications/Materia/$VERSION}"
APPS="${MATERIA_APPS_DIR:-$HOME/Applications}"
echo "Installing Materia $VERSION into $TARGET"
PYTHON=""
for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)'; then
      PYTHON="$(command -v "$candidate")"
      break
    fi
  fi
done
if [ -z "$PYTHON" ]; then
  echo "Python 3.10 to 3.13 was not found. Install it from https://www.python.org or with Homebrew (brew install python@3.12) and run this installer again."
  exit 1
fi
echo "Using Python at $PYTHON"
rm -rf "$TARGET"
mkdir -p "$TARGET"
cp -R "$SOURCE/materia-source/." "$TARGET/"
"$PYTHON" -m venv "$TARGET/.venv"
"$TARGET/.venv/bin/python" -m pip install --upgrade pip
"$TARGET/.venv/bin/python" -m pip install "$TARGET[adapters,bigdata,desktop]"
INSTALLED=""
MISSING=""
for engine in __ENGINES__; do
  if "$TARGET/.venv/bin/python" -m pip install "$engine" >/dev/null 2>&1; then
    INSTALLED="$INSTALLED $engine"
  else
    MISSING="$MISSING $engine"
  fi
done
"$TARGET/.venv/bin/python" "$TARGET/tools/make_macos_app.py" --output "$APPS"
echo ""
echo "Materia $VERSION is installed as $APPS/Materia.app"
echo "Optional engines installed:$INSTALLED"
if [ -n "$MISSING" ]; then
  echo "Not installed, refused by Materia until installed:$MISSING"
fi
echo "Quantum ESPRESSO, LAMMPS, GPAW, Psi4 and MACE run in conda environments; see docs/THIRD_PARTY.md."
"""

README = """Materia {version}

{platform_steps}

Materia is an open-source atomistic, molecular and electronic-structure
workbench released under the MIT licence. It drives mature open-source engines
installed separately under their own licences; see docs/THIRD_PARTY.md.
Every result carries its provenance, and calculations outside a model's
validity are refused rather than approximated. The validated capabilities are
listed in docs/RELEASE_NOTES.md.
"""

WINDOWS_STEPS = """Windows 10 or 11

1. Unzip this folder.
2. Double-click "Install Materia.cmd". It needs an internet connection. If
   Python 3.10 to 3.13 is missing it offers to install Python 3.12 with
   winget.
3. Start Materia from the Start Menu or the Desktop shortcut.

The window uses Microsoft Edge WebView2, which Windows 10 and 11 include.
"Uninstall Materia.cmd" removes the program and its shortcuts.
PySCF, tblite, GPAW and Quantum ESPRESSO have no native Windows builds; run
them through the Windows Subsystem for Linux if you need them."""

MACOS_STEPS = """macOS 12 or later

1. Unzip this folder.
2. Right-click "Install Materia.command" and choose Open (the installer is
   not code signed, so macOS asks once). It needs an internet connection and
   Python 3.10 to 3.13 from python.org or Homebrew.
3. Open Materia from ~/Applications/Materia.app."""


def source_files() -> dict:
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name.startswith(EXCLUDED):
                continue
            files[member.name] = (tar.extractfile(member).read(), member.mode)
    return files


def build_ico(sizes=(16, 24, 32, 48, 64, 128, 256)) -> bytes:
    from make_macos_app import render_icon

    images = [render_icon(size) for size in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, data = b"", b""
    for size, png in zip(sizes, images):
        dimension = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dimension, dimension, 0, 0, 1, 32, len(png),
                               offset + len(data))
        data += png
    return header + entries + data


def write_zip(path: Path, root: str, files: dict, extra: dict) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, (data, mode) in sorted(files.items()):
            info = zipfile.ZipInfo(f"{root}/materia-source/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = (mode & 0o777 | 0o100000) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
        for name, (data, mode) in extra.items():
            info = zipfile.ZipInfo(f"{root}/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = (mode | 0o100000) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist"))
    args = parser.parse_args()
    from materia.version import __version__ as version

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = source_files()
    root = f"Materia-{version}"
    windows_engines = ", ".join(f"'{e}'" for e in WINDOWS_ENGINES)
    windows = {
        "Install Materia.cmd": (INSTALL_CMD.encode(), 0o644),
        "Uninstall Materia.cmd": (UNINSTALL_CMD.encode(), 0o644),
        "install.ps1": (INSTALL_PS1.replace("__VERSION__", version)
                        .replace("__ENGINES__", windows_engines).encode("utf-8-sig"), 0o644),
        "uninstall.ps1": (UNINSTALL_PS1.replace("__VERSION__", version).encode("utf-8-sig"),
                          0o644),
        "Materia.ico": (build_ico(), 0o644),
        "README.txt": (README.format(version=version, platform_steps=WINDOWS_STEPS)
                       .replace("\n", "\r\n").encode(), 0o644),
    }
    macos = {
        "Install Materia.command": (INSTALL_COMMAND.replace("__VERSION__", version)
                                    .replace("__ENGINES__", " ".join(MACOS_ENGINES)).encode(),
                                    0o755),
        "README.txt": (README.format(version=version, platform_steps=MACOS_STEPS).encode(),
                       0o644),
    }
    targets = {"windows": windows, "macos": macos}
    for platform, extra in targets.items():
        path = out / f"Materia-{version}-{platform}.zip"
        write_zip(path, root, files, extra)
        print(f"{path} ({path.stat().st_size / 1e6:.1f} MB, {len(files)} source files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
