"""Convenient lookup layer over :mod:`materia.elements.data`."""

from __future__ import annotations

from typing import Optional, Union

from .data import BY_NAME, BY_NUMBER, BY_SYMBOL, ELEMENTS, Element, Isotope


class UnknownElement(KeyError):
    """Raised for an unrecognised element symbol, name or atomic number."""


def element(key: Union[str, int, Element]) -> Element:
    """Look up an element by symbol ('Si'), name ('silicon') or Z (14)."""
    if isinstance(key, Element):
        return key
    if (not isinstance(key, (str, bytes)) and hasattr(key, "__index__")) or (
            isinstance(key, float) and float(key).is_integer()):
        z = int(key)
        try:
            return BY_NUMBER[z]
        except KeyError as exc:
            raise UnknownElement(f"No element with atomic number {z}") from exc
    if isinstance(key, str):
        k = key.strip()
        if k in BY_SYMBOL:
            return BY_SYMBOL[k]
        cap = k.capitalize()
        if cap in BY_SYMBOL:
            return BY_SYMBOL[cap]
        if k.lower() in BY_NAME:
            return BY_NAME[k.lower()]
        if k.isdigit():
            return element(int(k))
    raise UnknownElement(f"Unknown element {key!r}")


def atomic_number(key: Union[str, int, Element]) -> int:
    return element(key).number


def symbol(key: Union[str, int, Element]) -> str:
    return element(key).symbol


def name(key: Union[str, int, Element]) -> str:
    return element(key).name


def mass(key: Union[str, int, Element], mass_number: Optional[int] = None) -> float:
    """Atomic mass in u.

    With ``mass_number=None`` the IUPAC standard atomic weight (natural
    abundance mixture) is returned.  With an explicit mass number the exact
    nuclide mass from AME2020 is returned.
    """
    el = element(key)
    if mass_number:
        return el.isotope(int(mass_number)).atomic_mass_u
    if el.standard_atomic_weight is None:
        raise UnknownElement(f"No standard atomic weight tabulated for {el.symbol}")
    return el.standard_atomic_weight


def covalent_radius(key: Union[str, int, Element]) -> Optional[float]:
    return element(key).covalent_radius_A


def vdw_radius(key: Union[str, int, Element]) -> Optional[float]:
    return element(key).vdw_radius_A


def isotopes(key: Union[str, int, Element]):
    return element(key).isotopes


def has_isotope_data(key: Union[str, int, Element]) -> bool:
    return bool(element(key).isotopes)


def all_symbols():
    return [e.symbol for e in ELEMENTS]


__all__ = [
    "Element", "Isotope", "ELEMENTS", "UnknownElement",
    "element", "atomic_number", "symbol", "name", "mass",
    "covalent_radius", "vdw_radius", "isotopes", "has_isotope_data", "all_symbols",
]
