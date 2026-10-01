"""Dependency-free PNG and TIFF writers for scan export.

Scan images are written with the standard library only (``zlib`` and
``struct``), so exporting data never depends on an optional imaging package.
16-bit grayscale TIFF preserves the full dynamic range of the measurement;
PNG is for figures.
"""

from __future__ import annotations

import struct
import zlib
from typing import Optional, Sequence, Tuple

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_png(path: str, rgb: np.ndarray, *, text: Optional[dict] = None,
              dpi: Optional[int] = None) -> str:
    """Write an ``(ny, nx, 3)`` uint8 array as a PNG.

    ``text`` entries become tEXt chunks, which is how the provenance summary
    travels with an exported figure.
    """
    a = np.asarray(rgb)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    ny, nx, nc = a.shape
    if nc not in (3, 4):
        raise ValueError("write_png expects 3 or 4 colour channels")

    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(ny))
    out = [b"\x89PNG\r\n\x1a\n"]
    out.append(_chunk(b"IHDR", struct.pack(">IIBBBBB", nx, ny, 8,
                                           6 if nc == 4 else 2, 0, 0, 0)))
    if dpi:
        ppm = int(round(dpi / 0.0254))
        out.append(_chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1)))
    for key, value in (text or {}).items():
        k = str(key).encode("latin-1", "replace")[:79]
        v = str(value).encode("latin-1", "replace")
        out.append(_chunk(b"tEXt", k + b"\x00" + v))
    out.append(_chunk(b"IDAT", zlib.compress(raw, 6)))
    out.append(_chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(b"".join(out))
    return path


def write_tiff16(path: str, data: np.ndarray, *, description: str = "") -> str:
    """Write a single-channel 16-bit grayscale TIFF (little-endian, uncompressed).

    The data are linearly mapped to 0..65535; the mapping is written into the
    ImageDescription tag so the physical values can be recovered.
    """
    a = np.asarray(data, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        hi = lo + 1e-12
    scaled = np.clip((a - lo) / (hi - lo) * 65535.0, 0, 65535).astype("<u2")
    ny, nx = scaled.shape
    desc = (f"Materia 16-bit export; value = {lo!r} + level/65535 * "
            f"({hi!r} - {lo!r}). {description}").encode("latin-1", "replace") + b"\x00"

    header = struct.pack("<2sHI", b"II", 42, 8)
    n_tags = 10
    ifd_offset = 8
    data_offset = ifd_offset + 2 + n_tags * 12 + 4
    desc_offset = data_offset
    res_offset = desc_offset + len(desc)
    pixel_offset = res_offset + 16

    def tag(tag_id, dtype, count, value):
        if dtype == 3 and count == 1:
            payload = struct.pack("<HH", value, 0)
        else:
            payload = struct.pack("<I", value)
        return struct.pack("<HHI", tag_id, dtype, count) + payload

    ifd = struct.pack("<H", n_tags)
    ifd += tag(256, 3, 1, nx)
    ifd += tag(257, 3, 1, ny)
    ifd += tag(258, 3, 1, 16)
    ifd += tag(259, 3, 1, 1)
    ifd += tag(262, 3, 1, 1)
    ifd += tag(270, 2, len(desc), desc_offset)
    ifd += tag(273, 4, 1, pixel_offset)
    ifd += tag(277, 3, 1, 1)
    ifd += tag(278, 3, 1, ny)
    ifd += tag(279, 4, 1, nx * ny * 2)
    ifd += struct.pack("<I", 0)

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(ifd)
        fh.write(desc)
        fh.write(b"\x00" * 16)
        fh.write(scaled.tobytes())
    return path


def add_scale_bar(rgb: np.ndarray, extent_A: Sequence[float],
                  length_A: Optional[float] = None,
                  colour: Tuple[int, int, int] = (250, 250, 250),
                  margin_px: int = 12, thickness_px: int = 4) -> Tuple[np.ndarray, dict]:
    """Burn a scale bar into a rendered image.  Returns the image and its label."""
    out = np.array(rgb, dtype=np.uint8, copy=True)
    ny, nx = out.shape[:2]
    x0, y0, x1, y1 = extent_A
    width_A = float(x1 - x0)
    if length_A is None:
        target = width_A / 4.0
        nice = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        length_A = min(nice, key=lambda v: abs(v - target))
    px = int(round(length_A / max(width_A, 1e-12) * nx))
    px = max(4, min(px, nx - 2 * margin_px))
    y = ny - margin_px - thickness_px
    out[y:y + thickness_px, margin_px:margin_px + px] = colour
    out[y - 1:y + thickness_px + 1, margin_px - 1:margin_px + 1] = colour
    out[y - 1:y + thickness_px + 1, margin_px + px - 1:margin_px + px + 1] = colour
    return out, {"length_A": length_A, "pixels": px,
                 "label": (f"{length_A:g} A" if length_A < 10
                           else f"{length_A / 10:g} nm")}
