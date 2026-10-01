#!/usr/bin/env python3
"""Package Materia as a double-clickable macOS application bundle.

The bundle launches the native window entry point with the interpreter that
built it, so the application carries its own environment. It is not code
signed: macOS will ask for confirmation the first time it is opened from
Finder unless you sign it yourself.

Usage:
    python tools/make_macos_app.py [--output DIR] [--name Materia]
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shlex
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render_icon(size: int) -> bytes:
    """Draw the application icon: a probe tip above a silver lattice."""
    import numpy as np

    n = size
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    u = (xx + 0.5) / n
    v = (yy + 0.5) / n

    margin = 0.085
    span = 1.0 - 2.0 * margin
    sx = (u - 0.5) / (span * 0.5)
    sy = (v - 0.5) / (span * 0.5)
    squircle = np.abs(sx) ** 4.6 + np.abs(sy) ** 4.6
    edge = max(2.0 / n, 0.004)
    alpha = np.clip((1.0 - squircle) / (edge * 6.0) + 0.5, 0.0, 1.0)

    rgb = np.zeros((n, n, 3), dtype=np.float64)
    vertical = np.clip((v - margin) / span, 0.0, 1.0)
    base = 0.115 - 0.055 * vertical
    rgb[..., 0] = base
    rgb[..., 1] = base + 0.006
    rgb[..., 2] = base + 0.016

    rows, cols = 3, 4
    spacing = span / cols
    radius = spacing * 0.30
    for row in range(rows):
        cy = 0.60 + row * spacing * 0.72
        if cy > 1.0 - margin:
            break
        offset = 0.5 * spacing if row % 2 else 0.0
        depth = 1.0 - row * 0.30
        for col in range(-1, cols + 2):
            cx = margin + offset + col * spacing
            d = np.hypot(u - cx, (v - cy) * 1.05)
            core = np.clip(1.0 - (d / radius) ** 2, 0.0, 1.0) ** 0.55
            halo = np.exp(-(d / (radius * 2.1)) ** 2) * 0.30
            bright = np.clip(core * 0.92 + halo, 0.0, 1.2) * depth
            rgb[..., 0] += bright * 0.84
            rgb[..., 1] += bright * 0.87
            rgb[..., 2] += bright * 0.91

    apex_y = 0.505
    tip_top = margin + 0.03
    half_width = 0.115
    ramp = np.clip((apex_y - v) / (apex_y - tip_top), 0.0, 1.0)
    width = half_width * ramp
    dx = np.abs(u - 0.5)
    body = (v <= apex_y) & (v >= tip_top) & (dx <= width)
    outline = body & (np.abs(dx - width) <= max(1.5 / n, 0.006))
    shade = 0.30 + 0.34 * np.clip((u - 0.5) / half_width + 0.5, 0.0, 1.0)
    rgb[body] = np.stack([shade, shade + 0.01, shade + 0.03], axis=-1)[body] * 0.62
    rgb[outline] = np.array([0.80, 0.83, 0.87])

    beam = (np.exp(-((u - 0.5) / (0.014 + 0.004)) ** 2)
            * np.clip((v - apex_y) / 0.05, 0.0, 1.0)
            * np.clip((0.60 - v) / 0.06, 0.0, 1.0))
    rgb[..., 0] += beam * 0.62
    rgb[..., 1] += beam * 0.50
    rgb[..., 2] += beam * 0.20

    rim = np.clip((squircle - 0.72) / 0.28, 0.0, 1.0) * 0.16
    rgb += rim[..., None]

    rgb = np.clip(rgb, 0.0, 1.0)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return encode_png((rgba * 255).astype(np.uint8))


def encode_png(rgba) -> bytes:
    import numpy as np

    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def build_icns(target: Path) -> bool:
    iconset = target.parent / "Materia.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    for size in (16, 32, 64, 128, 256, 512):
        (iconset / f"icon_{size}x{size}.png").write_bytes(render_icon(size))
        (iconset / f"icon_{size}x{size}@2x.png").write_bytes(render_icon(size * 2))
    if shutil.which("iconutil") is None:
        shutil.rmtree(iconset)
        return False
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(target)],
                   check=True)
    shutil.rmtree(iconset)
    return True


LAUNCHER = """#!/bin/sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=""
SEARCH_ROOT="$HERE"
while [ "$SEARCH_ROOT" != "/" ]; do
  if [ -f "$SEARCH_ROOT/pyproject.toml" ] && [ -d "$SEARCH_ROOT/materia" ]; then
    PROJECT_ROOT="$SEARCH_ROOT"
    break
  fi
  SEARCH_ROOT="$(dirname "$SEARCH_ROOT")"
done
if [ -z "$PROJECT_ROOT" ] && [ -d __ROOT__ ]; then
  PROJECT_ROOT=__ROOT__
fi
LOG_DIR="${TMPDIR:-/tmp}/materia"
if [ -n "$PROJECT_ROOT" ]; then
  LOG_DIR="$PROJECT_ROOT/.materia"
fi
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher.log"
fail() {
  MESSAGE="$1"
  printf '%s\n' "$MESSAGE" >> "$LOG_FILE"
  printf '%s\n' "$MESSAGE" >&2
  if command -v osascript >/dev/null 2>&1; then
    osascript -e 'display alert "Materia could not start" message "Open .materia/launcher.log in the Atomic Model folder for details." as critical' >/dev/null 2>&1 &
  fi
  exit 2
}
if [ -z "$PROJECT_ROOT" ]; then
  fail "The Atomic Model project folder could not be found. Keep Materia.app inside the project folder."
fi
PYTHON="${MATERIA_PYTHON:-}"
if [ -z "$PYTHON" ] && [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi
if [ -z "$PYTHON" ] && [ -x __PYTHON__ ]; then
  PYTHON=__PYTHON__
fi
if [ -z "$PYTHON" ]; then
  PYTHON="$(command -v python3 || true)"
fi
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
  fail "No usable Python interpreter was found. Create .venv and install the project dependencies."
fi
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PYTHON" -c 'import materia, numpy, scipy, webview' >/dev/null 2>&1; then
  fail "The selected Python environment is missing Materia, NumPy, SciPy, or pywebview. Install the project dependencies in .venv."
fi
if [ "${1:-}" = "--check" ]; then
  printf 'Materia launcher ready\nproject=%s\npython=%s\n' "$PROJECT_ROOT" "$PYTHON"
  exit 0
fi
case "${1:-}" in
  *.materia)
    PROJECT_FILE="$1"
    shift
    set -- --project "$PROJECT_FILE" "$@"
    ;;
esac
: > "$LOG_FILE"
"$PYTHON" -m materia.desktop_ui.desktop "$@" >> "$LOG_FILE" 2>&1
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  fail "Materia exited with status $STATUS. See $LOG_FILE for the startup log."
fi
exit 0
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist"))
    parser.add_argument("--name", default="Materia")
    args = parser.parse_args()

    from materia.version import __version__

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / f"{args.name}.app"
    if bundle.exists():
        shutil.rmtree(bundle)
    macos = bundle / "Contents" / "MacOS"
    resources = bundle / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    launcher = macos / args.name
    launcher_text = LAUNCHER.replace("__PYTHON__", shlex.quote(sys.executable))
    launcher_text = launcher_text.replace("__ROOT__", shlex.quote(str(ROOT)))
    launcher.write_text(launcher_text)
    launcher.chmod(0o755)

    has_icon = build_icns(resources / "Materia.icns")

    info = {
        "CFBundleName": args.name,
        "CFBundleDisplayName": args.name,
        "CFBundleExecutable": args.name,
        "CFBundleIdentifier": "org.materia.app",
        "CFBundleVersion": __version__,
        "CFBundleShortVersionString": __version__,
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "LSApplicationCategoryType": "public.app-category.education",
        "NSHumanReadableCopyright": "MIT. Runs locally; no telemetry.",
        "CFBundleDocumentTypes": [{
            "CFBundleTypeName": "Materia project",
            "CFBundleTypeExtensions": ["materia"],
            "CFBundleTypeRole": "Editor",
            "LSHandlerRank": "Owner",
        }],
    }
    if has_icon:
        info["CFBundleIconFile"] = "Materia.icns"
    (bundle / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))

    print(f"built {bundle}")
    print(f"  interpreter : {sys.executable}")
    print(f"  icon        : {'Materia.icns' if has_icon else 'none (iconutil missing)'}")
    print("  open it with:  open " + str(bundle).replace(" ", "\\ "))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
