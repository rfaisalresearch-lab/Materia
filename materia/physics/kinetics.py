"""Reaction rates from stationary points: harmonic transition-state theory.

Harmonic transition-state theory (G. H. Vineyard, J. Phys. Chem. Solids 3
(1957) 121):

``k = (prod_i nu_i^min / prod_j nu_j^saddle) exp(-dE / kT)``

over the real vibrational frequencies of the minimum (3N - r of them, r the
rigid-body count) and of the saddle point (one fewer: its unstable mode is
imaginary).  ``dE`` is the static barrier, from NEB or two single points.

Optional corrections, each a stated approximation:

* Kramers (Physica 7 (1940) 284) friction in the spatial-diffusion regime,
  ``kappa = sqrt(1 + (gamma / 2 omega_b)^2) - gamma / 2 omega_b`` with the
  saddle's imaginary angular frequency ``omega_b``; valid when friction is
  not so weak that energy diffusion limits the rate.
* Wigner (Z. Phys. Chem. B 19 (1932) 203) leading-order tunnelling,
  ``kappa = 1 + (hbar omega_b / kT)^2 / 24``, meaningful only while it stays
  close to one.

Inputs are :class:`~materia.physics.phonons.PhononResult` objects for the two
stationary points.  A minimum with an imaginary mode, a saddle without
exactly one, or mismatched mode counts are refused.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..core_model.units import BOLTZMANN_EV_K, HBAR_EV_FS
from ..provenance import Fidelity, Origin, Provenance, Result


class KineticsError(ValueError):
    pass


def _real(result) -> np.ndarray:
    return np.array([f for f, kind in zip(result.frequencies_THz, result.kinds)
                     if kind == "real"])


def harmonic_tst_rate(minimum, saddle, barrier_eV: float, temperature_K: float,
                      friction_per_s: Optional[float] = None, wigner: bool = False) -> Result:
    """Rate in 1/s for leaving the minimum through the saddle."""
    if temperature_K <= 0:
        raise KineticsError("temperature_K must be positive.")
    if barrier_eV <= 0:
        raise KineticsError("The barrier must be positive; otherwise there is no activated "
                            "process to describe.")
    if minimum.imaginary_modes():
        raise KineticsError("The minimum has imaginary modes; it is not a minimum.")
    imaginary = saddle.imaginary_modes()
    if len(imaginary) != 1:
        raise KineticsError(f"The saddle has {len(imaginary)} imaginary modes; a first-order "
                            "saddle point has exactly one.")
    undetermined = [k for k in ("zero-within-resolution",)
                    if k in minimum.kinds or k in saddle.kinds]
    if undetermined:
        raise KineticsError("A mode at one of the stationary points is zero within the "
                            "numerical resolution; the prefactor would be undefined.")
    real_min, real_saddle = _real(minimum), _real(saddle)
    if len(real_min) != len(real_saddle) + 1:
        raise KineticsError(f"The minimum has {len(real_min)} real modes and the saddle "
                            f"{len(real_saddle)}; a saddle should have exactly one fewer.")
    log_prefactor = (np.sum(np.log(real_min * 1e12)) - np.sum(np.log(real_saddle * 1e12)))
    kt = BOLTZMANN_EV_K * temperature_K
    rate = math.exp(log_prefactor - barrier_eV / kt)
    omega_b = 2 * math.pi * abs(float(saddle.frequencies_THz[imaginary[0]])) * 1e12
    factors = {}
    if friction_per_s is not None:
        if friction_per_s < 0:
            raise KineticsError("friction_per_s cannot be negative.")
        x = friction_per_s / (2 * omega_b)
        factors["kramers"] = math.sqrt(1 + x * x) - x
    if wigner:
        u = HBAR_EV_FS * 1e-15 * omega_b / kt
        factors["wigner"] = 1 + u * u / 24
        if factors["wigner"] > 1.5:
            raise KineticsError(f"The Wigner correction is {factors['wigner']:.2f}; tunnelling is "
                                "too strong for its leading-order form.")
    corrected = rate * math.prod(factors.values()) if factors else rate
    prov = Provenance(
        model="kinetics/harmonic-tst", fidelity=minimum.provenance.fidelity,
        origin=Origin.CALCULATED,
        approximations=["Harmonic transition-state theory: no recrossing, harmonic wells and "
                        "saddle, classical vibrations (Vineyard 1957)."] +
                       (["Kramers friction factor in the spatial-diffusion regime."]
                        if "kramers" in factors else []) +
                       (["Wigner leading-order tunnelling correction."] if wigner else []),
        parameters={"barrier_eV": barrier_eV, "temperature_K": temperature_K,
                    "friction_per_s": friction_per_s, "factors": factors,
                    "minimum_model": minimum.provenance.model,
                    "saddle_model": saddle.provenance.model},
        references=["G. H. Vineyard, J. Phys. Chem. Solids 3 (1957) 121",
                    "H. A. Kramers, Physica 7 (1940) 284",
                    "E. Wigner, Z. Phys. Chem. B 19 (1932) 203"])
    return Result("rate_constant", corrected, "1/s", prov,
                  extra={"prefactor_Hz": math.exp(log_prefactor),
                         "uncorrected_rate_per_s": rate, "factors": factors,
                         "imaginary_frequency_THz": float(saddle.frequencies_THz[imaginary[0]]),
                         "half_life_s": math.log(2) / corrected})
