"""Validated configuration for a GPAW calculation.

Every setting carries its unit in its name.  Validation happens before a job is
created, so a bad combination is refused while the project is still untouched
rather than surfacing as a traceback out of a solver twenty minutes in.

Nothing here silently corrects a value.  A plane-wave calculation on a slab
that is not periodic in all three directions is refused, not quietly switched
to a finite-difference grid, because the two are different calculations and the
user has to know which one produced the number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

FUNCTIONALS: Dict[str, str] = {
    "LDA": "LDA",
    "PBE": "PBE",
    "RPBE": "RPBE",
    "revPBE": "revPBE",
    "PBEsol": "PBE",
    "BLYP": "PBE",
}

MODES: Dict[str, str] = {
    "pw": "cutoff_eV",
    "fd": "grid_spacing_A",
    "lcao": "basis",
}

OCCUPATION_SCHEMES = ("fermi-dirac", "marzari-vanderbilt", "fixed")

TASKS = ("energy",)


class GPAWSettingsError(ValueError):
    """A configuration that would not produce a meaningful calculation."""


@dataclass
class GPAWSettings:
    """One GPAW calculation, fully specified.

    ``task`` is ``"energy"``: a single self-consistent field solution giving the
    total energy and the forces on every atom.  Relaxation is not offered by
    this driver, and asking for it is refused rather than approximated.
    """

    task: str = "energy"
    xc: str = "PBE"
    mode: str = "pw"
    cutoff_eV: float = 340.0
    grid_spacing_A: float = 0.20
    basis: str = "dzp"
    kpoints: Tuple[int, int, int] = (1, 1, 1)
    occupations: str = "fermi-dirac"
    smearing_eV: float = 0.05
    charge: float = 0.0
    spin_polarized: bool = False
    energy_tol_eV_per_electron: float = 5.0e-4
    density_tol_electrons: float = 1.0e-4
    max_iterations: int = 333
    label: str = "materia-gpaw"

    def as_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["kpoints"] = list(self.kpoints)
        return out

    def units(self) -> Dict[str, str]:
        return {
            "cutoff_eV": "eV", "grid_spacing_A": "A", "smearing_eV": "eV",
            "charge": "e", "energy_tol_eV_per_electron": "eV/electron",
            "density_tol_electrons": "electrons", "kpoints": "Monkhorst-Pack divisions",
            "max_iterations": "SCF iterations",
        }

    def summary(self) -> str:
        if self.mode == "pw":
            discretisation = f"plane waves to {self.cutoff_eV:g} eV"
        elif self.mode == "fd":
            discretisation = f"real-space grid at {self.grid_spacing_A:g} A"
        else:
            discretisation = f"LCAO {self.basis}"
        kpts = "x".join(str(k) for k in self.kpoints)
        return (f"{self.xc}, {discretisation}, {kpts} k-points, "
                f"{self.occupations}"
                + (f" {self.smearing_eV:g} eV" if self.occupations != "fixed" else "")
                + (f", charge {self.charge:+g} e" if self.charge else "")
                + (", spin-polarised" if self.spin_polarized else ""))


def _positive(name: str, value: Any, unit: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise GPAWSettingsError(f"{name} must be a number in {unit}, got {value!r}.") from None
    if not number > 0 or number != number or number in (float("inf"), float("-inf")):
        raise GPAWSettingsError(
            f"{name} must be a positive, finite value in {unit}, got {value!r}.")
    return number


def validate(raw: Optional[Dict[str, Any]] = None,
             structure: Any = None) -> GPAWSettings:
    """Build settings from a raw dict, refusing anything that would not run.

    ``structure`` is optional; when given, settings are checked against its
    periodicity and its elements as well as against themselves.
    """
    raw = dict(raw or {})
    unknown = sorted(set(raw) - {f.name for f in fields(GPAWSettings)})
    if unknown:
        known = ", ".join(sorted(f.name for f in fields(GPAWSettings)))
        raise GPAWSettingsError(
            f"Unknown GPAW setting(s): {', '.join(unknown)}. Accepted: {known}.")

    settings = GPAWSettings()
    for key, value in raw.items():
        setattr(settings, key, value)

    if settings.task not in TASKS:
        raise GPAWSettingsError(
            f"task must be one of {', '.join(TASKS)}, got {settings.task!r}. "
            "This driver runs single-point energy and forces; relaxation and "
            "dynamics are not implemented and are refused rather than approximated.")

    if settings.xc not in FUNCTIONALS:
        raise GPAWSettingsError(
            f"xc must be one of {', '.join(sorted(FUNCTIONALS))}, got {settings.xc!r}.")

    if settings.mode not in MODES:
        raise GPAWSettingsError(
            f"mode must be one of {', '.join(sorted(MODES))}, got {settings.mode!r}.")

    if settings.mode == "pw":
        settings.cutoff_eV = _positive("cutoff_eV", settings.cutoff_eV, "eV")
        if settings.cutoff_eV < 100.0:
            raise GPAWSettingsError(
                f"cutoff_eV = {settings.cutoff_eV:g} eV is below 100 eV, which cannot "
                "represent a PAW pseudo-wavefunction. Raise it rather than accepting "
                "a meaningless energy.")
    elif settings.mode == "fd":
        settings.grid_spacing_A = _positive("grid_spacing_A", settings.grid_spacing_A, "A")
        if settings.grid_spacing_A > 0.35:
            raise GPAWSettingsError(
                f"grid_spacing_A = {settings.grid_spacing_A:g} A is coarser than 0.35 A; "
                "the real-space representation would not be converged in any useful "
                "sense.")
    elif not str(settings.basis).strip():
        raise GPAWSettingsError("basis must name an LCAO basis set, for example 'dzp'.")

    kpoints = settings.kpoints
    if isinstance(kpoints, int):
        kpoints = (kpoints, kpoints, kpoints)
    try:
        kpoints = tuple(int(k) for k in kpoints)
    except (TypeError, ValueError):
        raise GPAWSettingsError(
            f"kpoints must be three Monkhorst-Pack divisions, got {settings.kpoints!r}."
        ) from None
    if len(kpoints) != 3 or any(k < 1 for k in kpoints):
        raise GPAWSettingsError(
            f"kpoints must be three divisions of at least 1, got {list(kpoints)}.")
    settings.kpoints = kpoints

    if settings.occupations not in OCCUPATION_SCHEMES:
        raise GPAWSettingsError(
            f"occupations must be one of {', '.join(OCCUPATION_SCHEMES)}, "
            f"got {settings.occupations!r}.")
    try:
        settings.smearing_eV = float(settings.smearing_eV)
    except (TypeError, ValueError):
        raise GPAWSettingsError(
            f"smearing_eV must be a width in eV, got {settings.smearing_eV!r}.") from None
    if settings.smearing_eV < 0:
        raise GPAWSettingsError(
            f"smearing_eV must not be negative, got {settings.smearing_eV:g} eV.")
    if settings.occupations != "fixed" and settings.smearing_eV == 0.0:
        raise GPAWSettingsError(
            f"occupations={settings.occupations!r} with smearing_eV = 0 is a fixed "
            "occupation by another name. Choose occupations='fixed' explicitly, or "
            "give a positive width.")

    settings.charge = _validated_charge(settings.charge)
    settings.spin_polarized = _validated_spin(settings.spin_polarized)

    settings.energy_tol_eV_per_electron = _positive(
        "energy_tol_eV_per_electron", settings.energy_tol_eV_per_electron, "eV/electron")
    settings.density_tol_electrons = _positive(
        "density_tol_electrons", settings.density_tol_electrons, "electrons")

    if isinstance(settings.max_iterations, bool) or not isinstance(
            settings.max_iterations, int):
        raise GPAWSettingsError(
            f"max_iterations must be a whole number of SCF iterations, "
            f"got {settings.max_iterations!r}.")
    if settings.max_iterations < 1:
        raise GPAWSettingsError(
            f"max_iterations must be at least 1, got {settings.max_iterations}.")

    if structure is not None:
        _check_against_structure(settings, structure)
    elif settings.charge != 0.0:
        raise GPAWSettingsError(
            f"charge = {settings.charge:+g} e was requested. " + CHARGE_REFUSAL)
    return settings


CHARGE_REFUSAL = (
    "Charged systems are not supported by the GPAW driver (capability "
    "'charged_system'). GPAW can do them; this driver has never been run "
    "against one, so it declines rather than handing GPAW an input whose "
    "result nobody has checked. Neutralise the structure, or drive GPAW "
    "yourself through the ASE adapter and classify the result."
)

SPIN_REFUSAL = (
    "Spin-polarised calculations are not supported by the GPAW driver "
    "(capability 'spin_polarized'). GPAW can do them; this driver has never "
    "been run against one, so it declines rather than reporting a number from "
    "a path it has not exercised. Drive GPAW yourself through the ASE adapter "
    "if you need open-shell results."
)


def _validated_charge(value: Any) -> float:
    """Parse the system charge without silently coercing another type."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer,
                                                         np.floating)):
        raise GPAWSettingsError(
            f"charge must be a number of electrons in e, got {value!r}.")
    number = float(value)
    if not np.isfinite(number):
        raise GPAWSettingsError(f"charge must be finite, got {value!r}.")
    return number


def _validated_spin(value: Any) -> bool:
    """Spin polarisation, which this driver only accepts as ``False``.

    A string is refused rather than coerced: ``bool("false")`` is ``True``, and
    silently turning a configuration file's ``"false"`` into spin polarisation
    is exactly the kind of quiet reinterpretation this driver must not do.
    """
    if not isinstance(value, bool):
        raise GPAWSettingsError(
            f"spin_polarized must be a boolean, got {type(value).__name__} "
            f"{value!r}. It is not coerced: bool('false') is True, and a string "
            "must not decide whether a calculation is spin-polarised.")
    if value:
        raise GPAWSettingsError("spin_polarized=True was requested. " + SPIN_REFUSAL)
    return False


def _check_against_structure(settings: GPAWSettings, structure: Any) -> None:
    """Refuse combinations the structure itself makes meaningless."""
    pbc = tuple(bool(p) for p in structure.cell.pbc)
    axes = "abc"

    if settings.mode == "pw" and not all(pbc):
        open_axes = ", ".join(axes[i] for i in range(3) if not pbc[i])
        raise GPAWSettingsError(
            f"mode='pw' needs periodic boundaries in all three directions, and this "
            f"structure is open along {open_axes}. Use mode='fd' for a slab or a "
            "molecule. Materia will not switch the mode for you: a plane-wave and a "
            "real-space calculation are different calculations.")

    for index, divisions in enumerate(settings.kpoints):
        if divisions > 1 and not pbc[index]:
            raise GPAWSettingsError(
                f"kpoints asks for {divisions} divisions along {axes[index]}, which is "
                "not a periodic direction. Sampling a Brillouin zone that does not "
                "exist is meaningless; use 1 division there.")

    if len(structure) == 0:
        raise GPAWSettingsError("The structure has no atoms.")

    carried = float(structure.total_charge())
    if abs(carried - settings.charge) > 1e-9:
        raise GPAWSettingsError(
            f"The structure carries a net formal charge of {carried:+g} e, but "
            f"the calculation requested charge = {settings.charge:+g} e. "
            + CHARGE_REFUSAL)
    if abs(settings.charge) > 1e-9:
        raise GPAWSettingsError(
            f"The structure and calculation both specify charge = "
            f"{settings.charge:+g} e. " + CHARGE_REFUSAL)

    moments = np.asarray(structure.magnetic_moments, dtype=float)
    if np.any(np.abs(moments) > 1e-9):
        where = int(np.argmax(np.abs(moments)))
        raise GPAWSettingsError(
            f"The structure carries initial magnetic moments, the largest being "
            f"{moments[where]:+g} on atom {int(structure.ids[where])}, which only "
            "mean anything in a spin-polarised calculation. " + SPIN_REFUSAL)


PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "task": "energy", "xc": "LDA", "mode": "pw", "cutoff_eV": 300.0,
        "kpoints": (1, 1, 1), "occupations": "fixed", "smearing_eV": 0.0,
        "energy_tol_eV_per_electron": 1.0e-3, "max_iterations": 120,
    },
    "molecule": {
        "task": "energy", "xc": "PBE", "mode": "fd", "grid_spacing_A": 0.20,
        "kpoints": (1, 1, 1), "occupations": "fixed", "smearing_eV": 0.0,
        "energy_tol_eV_per_electron": 5.0e-4, "max_iterations": 333,
    },
    "surface": {
        "task": "energy", "xc": "PBE", "mode": "fd", "grid_spacing_A": 0.20,
        "kpoints": (4, 4, 1), "occupations": "fermi-dirac", "smearing_eV": 0.1,
        "energy_tol_eV_per_electron": 5.0e-4, "max_iterations": 333,
    },
}

PRESET_NOTES: Dict[str, str] = {
    "smoke": "Smallest calculation that still solves the Kohn-Sham equations. For "
             "checking that GPAW runs at all, not for a physical result.",
    "molecule": "Finite-difference grid with open boundaries, for an isolated "
                "molecule or cluster in a box.",
    "surface": "Finite-difference grid with in-plane k-point sampling and a metallic "
               "smearing, for a slab that is periodic in x and y only.",
}


def preset(name: str, structure: Any = None, **overrides: Any) -> GPAWSettings:
    """A conservative starting configuration, validated like any other."""
    if name not in PRESETS:
        raise GPAWSettingsError(
            f"Unknown preset {name!r}. Available: {', '.join(sorted(PRESETS))}.")
    raw = dict(PRESETS[name])
    raw.update(overrides)
    return validate(raw, structure)


def describe_presets() -> List[dict]:
    return [{"name": name, "settings": dict(values), "note": PRESET_NOTES.get(name, "")}
            for name, values in PRESETS.items()]
