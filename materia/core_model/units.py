"""Physical constants and unit handling.

Materia uses a single explicit internal unit system and converts at the
boundaries.  Every quantity that crosses a module boundary, is serialised, or
is shown in the UI carries its unit string.

Internal ("atomic-scale engineering") units
-------------------------------------------
======================  =====================================================
length                  angstrom (A)
energy                  electronvolt (eV)
mass                    unified atomic mass unit (u)
time                    femtosecond (fs)  [derived: 1 fs = sqrt(u A^2/eV)*...]
force                   eV/A
charge                  elementary charge (e)
temperature             kelvin (K)
electric field          V/A
current                 ampere (A_current) -- nanoampere at the UI boundary
frequency               hertz (Hz); AFM detuning reported in Hz
======================  =====================================================

CODATA 2018 values are used for all constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

ELEMENTARY_CHARGE_C = 1.602176634e-19
PLANCK_J_S = 6.62607015e-34
HBAR_J_S = PLANCK_J_S / (2.0 * 3.141592653589793)
BOLTZMANN_J_K = 1.380649e-23
AVOGADRO = 6.02214076e23
ELECTRON_MASS_KG = 9.1093837015e-31
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
BOHR_RADIUS_M = 5.29177210903e-11
VACUUM_PERMITTIVITY = 8.8541878128e-12

BOLTZMANN_EV_K = BOLTZMANN_J_K / ELEMENTARY_CHARGE_C
HARTREE_EV = 27.211386245988
BOHR_A = BOHR_RADIUS_M * 1e10
RYDBERG_EV = HARTREE_EV / 2.0
HBAR_EV_FS = HBAR_J_S / ELEMENTARY_CHARGE_C * 1e15
ELECTRON_MASS_U = ELECTRON_MASS_KG / ATOMIC_MASS_UNIT_KG

COULOMB_K_EV_A = (
    ELEMENTARY_CHARGE_C**2
    / (4.0 * 3.141592653589793 * VACUUM_PERMITTIVITY)
    / ELEMENTARY_CHARGE_C
    * 1e10
)

HBAR2_OVER_2M_EV_A2 = (HBAR_J_S**2 / (2.0 * ELECTRON_MASS_KG)) / ELEMENTARY_CHARGE_C * 1e20

U_A2_FS2_TO_EV = ATOMIC_MASS_UNIT_KG * (1e-10 / 1e-15) ** 2 / ELEMENTARY_CHARGE_C
EV_TO_U_A2_FS2 = 1.0 / U_A2_FS2_TO_EV

DECAY_PREFACTOR_INV_A_SQRT_EV = (2.0 * ELECTRON_MASS_KG * ELEMENTARY_CHARGE_C) ** 0.5 \
    / HBAR_J_S * 1e-10


def kappa_inv_angstrom(barrier_eV: float) -> float:
    """Vacuum tunnelling decay constant kappa = sqrt(2 m phi)/hbar, in 1/A.

    The tunnelling current then decays as exp(-2 kappa z).
    For phi = 4 eV this gives kappa ~ 1.024 1/A, i.e. roughly one decade of
    current per angstrom, which is the standard experimental rule of thumb.
    """
    if barrier_eV <= 0:
        raise ValueError("Tunnel barrier height must be positive")
    return (2.0 * ELECTRON_MASS_KG * barrier_eV * ELEMENTARY_CHARGE_C) ** 0.5 / HBAR_J_S * 1e-10


_LENGTH_TO_A: Dict[str, float] = {
    "A": 1.0, "angstrom": 1.0, "Ang": 1.0,
    "nm": 10.0, "um": 1.0e4, "micron": 1.0e4, "mm": 1.0e7, "m": 1.0e10,
    "bohr": BOHR_A, "a0": BOHR_A, "pm": 0.01,
}
_ENERGY_TO_EV: Dict[str, float] = {
    "eV": 1.0, "meV": 1e-3, "keV": 1e3,
    "hartree": HARTREE_EV, "Ha": HARTREE_EV, "Ry": RYDBERG_EV,
    "J": 1.0 / ELEMENTARY_CHARGE_C,
    "kJ/mol": 1.0 / (ELEMENTARY_CHARGE_C * AVOGADRO * 1e-3),
    "kcal/mol": 4.184 / (ELEMENTARY_CHARGE_C * AVOGADRO * 1e-3),
    "K": BOLTZMANN_EV_K,
}
_CURRENT_TO_A: Dict[str, float] = {"A": 1.0, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9, "pA": 1e-12}


class UnitError(ValueError):
    """Raised when a unit is unknown or dimensionally inconsistent."""


def to_angstrom(value: float, unit: str) -> float:
    try:
        return value * _LENGTH_TO_A[unit]
    except KeyError as exc:
        raise UnitError(f"Unknown length unit {unit!r}. Known: {sorted(_LENGTH_TO_A)}") from exc


def from_angstrom(value_A: float, unit: str) -> float:
    try:
        return value_A / _LENGTH_TO_A[unit]
    except KeyError as exc:
        raise UnitError(f"Unknown length unit {unit!r}") from exc


def to_eV(value: float, unit: str) -> float:
    try:
        return value * _ENERGY_TO_EV[unit]
    except KeyError as exc:
        raise UnitError(f"Unknown energy unit {unit!r}. Known: {sorted(_ENERGY_TO_EV)}") from exc


def from_eV(value_eV: float, unit: str) -> float:
    try:
        return value_eV / _ENERGY_TO_EV[unit]
    except KeyError as exc:
        raise UnitError(f"Unknown energy unit {unit!r}") from exc


def to_ampere(value: float, unit: str) -> float:
    try:
        return value * _CURRENT_TO_A[unit]
    except KeyError as exc:
        raise UnitError(f"Unknown current unit {unit!r}") from exc


@dataclass(frozen=True)
class Quantity:
    """A scalar with an explicit unit.  Used at API and serialisation edges."""

    value: float
    unit: str

    def __str__(self) -> str:
        return f"{self.value:g} {self.unit}"

    def as_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit}


INTERNAL_UNITS = {
    "length": "A",
    "energy": "eV",
    "mass": "u",
    "time": "fs",
    "force": "eV/A",
    "charge": "e",
    "temperature": "K",
    "efield": "V/A",
    "current": "A",
    "frequency": "Hz",
    "velocity": "A/fs",
    "pressure": "eV/A^3",
    "density": "1/A^3",
}
