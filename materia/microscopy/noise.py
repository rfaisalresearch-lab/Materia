"""Instrumental noise and artefact models for simulated scanning-probe data.

Everything here is *instrumental*, not physical: it describes the measurement
chain, not the sample.  Each component is individually switchable and every
generator is driven by an explicit seed so that a scan is bit-reproducible
from the project file.

Components
----------
``white``
    Gaussian detector noise, uncorrelated between pixels.  Stands in for
    Johnson noise of the preamplifier and shot noise of the tunnel current.
``pink``
    1/f noise along the fast-scan direction, produced by filtering white noise
    in the frequency domain.  This is what gives real STM images their
    streaky texture.
``line_offset``
    A small random offset and gain error per scan line -- the single most
    recognisable feature of raster-scanned probe data.
``drift``
    Slow linear thermal drift in x, y and z across the image.
``creep``
    Piezo creep: an exponentially decaying displacement after the scan starts.
``vibration``
    A sinusoidal ripple from building vibration, at a chosen spatial frequency
    with a random phase.
``dropouts``
    Occasional single-line disturbances (tip instabilities). Off by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class NoiseModel:
    """Configuration of the instrumental noise chain."""

    seed: int = 0
    white_rms_frac: float = 0.012
    pink_rms_frac: float = 0.018
    line_offset_rms_frac: float = 0.010
    line_gain_rms: float = 0.004
    drift_A: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    creep_A: float = 0.0
    creep_lines: float = 20.0
    vibration_frac: float = 0.0
    vibration_cycles: float = 7.0
    dropout_probability: float = 0.0

    def describe(self) -> dict:
        d = {
            "seed": self.seed,
            "white_rms_frac": self.white_rms_frac,
            "pink_rms_frac": self.pink_rms_frac,
            "line_offset_rms_frac": self.line_offset_rms_frac,
            "line_gain_rms": self.line_gain_rms,
            "drift_A": list(self.drift_A),
            "creep_A": self.creep_A,
            "creep_lines": self.creep_lines,
            "vibration_frac": self.vibration_frac,
            "vibration_cycles": self.vibration_cycles,
            "dropout_probability": self.dropout_probability,
        }
        d["note"] = (
            "Instrumental noise model. These components describe the measurement "
            "chain, not the sample. The unfiltered physical signal is always kept "
            "as a separate channel."
        )
        return d

    @staticmethod
    def quiet(seed: int = 0) -> "NoiseModel":
        return NoiseModel(seed=seed, white_rms_frac=0.0, pink_rms_frac=0.0,
                          line_offset_rms_frac=0.0, line_gain_rms=0.0)

    @staticmethod
    def realistic(seed: int = 0) -> "NoiseModel":
        return NoiseModel(
            seed=seed, white_rms_frac=0.014, pink_rms_frac=0.020,
            line_offset_rms_frac=0.012, line_gain_rms=0.005,
            drift_A=(0.0, 0.0, 0.03), creep_A=0.04, creep_lines=18.0,
            vibration_frac=0.004, vibration_cycles=9.0,
        )

    @staticmethod
    def noisy(seed: int = 0) -> "NoiseModel":
        return NoiseModel(
            seed=seed, white_rms_frac=0.035, pink_rms_frac=0.045,
            line_offset_rms_frac=0.030, line_gain_rms=0.012,
            drift_A=(0.0, 0.0, 0.08), creep_A=0.10, creep_lines=25.0,
            vibration_frac=0.010, vibration_cycles=11.0,
            dropout_probability=0.01,
        )


def _pink_lines(rng: np.random.Generator, ny: int, nx: int) -> np.ndarray:
    """1/f noise along the fast-scan (x) direction, independent per line."""
    freqs = np.fft.rfftfreq(nx, d=1.0)
    amp = np.zeros_like(freqs)
    amp[1:] = 1.0 / np.sqrt(freqs[1:])
    spectrum = (rng.normal(size=(ny, len(freqs))) + 1j * rng.normal(size=(ny, len(freqs)))) * amp
    out = np.fft.irfft(spectrum, n=nx, axis=1)
    std = out.std()
    return out / std if std > 0 else out


def apply_noise(signal: np.ndarray, model: NoiseModel,
                scan_range_A: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, dict]:
    """Apply the instrumental chain to a clean 2-D signal.

    Returns the noisy signal and a record of what was applied.
    ``signal`` is indexed ``[line, pixel]`` with ``line`` the slow-scan axis.
    """
    sig = np.array(signal, dtype=float)
    ny, nx = sig.shape
    rng = np.random.default_rng(model.seed)
    span = float(np.ptp(sig))
    if span <= 0:
        span = float(np.abs(sig).max()) or 1.0
    applied: Dict[str, float] = {}

    if model.white_rms_frac > 0:
        sig = sig + rng.normal(0.0, model.white_rms_frac * span, sig.shape)
        applied["white_rms_A"] = model.white_rms_frac * span
    if model.pink_rms_frac > 0:
        sig = sig + model.pink_rms_frac * span * _pink_lines(rng, ny, nx)
        applied["pink_rms_A"] = model.pink_rms_frac * span
    if model.line_offset_rms_frac > 0:
        offs = rng.normal(0.0, model.line_offset_rms_frac * span, ny)
        offs = np.cumsum(offs) / np.sqrt(np.arange(1, ny + 1))
        sig = sig + offs[:, None]
        applied["line_offset_rms_A"] = model.line_offset_rms_frac * span
    if model.line_gain_rms > 0:
        gains = 1.0 + rng.normal(0.0, model.line_gain_rms, ny)
        mean = sig.mean()
        sig = mean + (sig - mean) * gains[:, None]
        applied["line_gain_rms"] = model.line_gain_rms
    if any(model.drift_A):
        dz = model.drift_A[2]
        if dz:
            ramp = np.linspace(0.0, dz, ny)[:, None]
            sig = sig + ramp
            applied["z_drift_A"] = dz
    if model.creep_A:
        lines = np.arange(ny)[:, None]
        sig = sig + model.creep_A * np.exp(-lines / max(model.creep_lines, 1e-6))
        applied["creep_A"] = model.creep_A
    if model.vibration_frac:
        phase = rng.uniform(0, 2 * np.pi)
        yy = np.linspace(0, 2 * np.pi * model.vibration_cycles, ny)[:, None]
        sig = sig + model.vibration_frac * span * np.sin(yy + phase)
        applied["vibration_A"] = model.vibration_frac * span
    if model.dropout_probability:
        hits = rng.random(ny) < model.dropout_probability
        n_hits = int(hits.sum())
        if n_hits:
            for line in np.nonzero(hits)[0]:
                start = rng.integers(0, nx)
                jump = rng.normal(0.0, 0.08 * span)
                sig[line, start:] += jump
            applied["dropout_lines"] = n_hits
    return sig, applied


def scan_noise_provenance(model: NoiseModel, applied: dict) -> dict:
    return {"model": model.describe(), "applied": applied,
            "reproducible": True,
            "note": "Re-running with the same seed reproduces this noise exactly."}
