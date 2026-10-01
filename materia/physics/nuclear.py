"""Nuclear masses, binding, decay energetics, decay chains and particle data.

Three sources, each named in every result:

AME2020 atomic mass evaluation (Materia-native analysis of published data)
    The complete ``mass_1.mas20`` table of W. J. Huang, M. Wang, F. G. Kondev,
    G. Audi and S. Naimi, Chin. Phys. C 45 (2021) 030002 and 030003,
    distributed by the IAEA Atomic Mass Data Center, shipped unchanged with its
    SHA-256 recorded.  Values marked ``#`` in the table are estimates from
    systematics, not measurements, and are flagged as such.  From the masses
    Materia computes binding energies, separation energies and decay Q values
    with the stated mass of the electron and the atomic-mass-unit energy
    equivalent; electron binding energies are neglected, as is conventional
    for atomic-mass differences.

Semi-empirical mass formula (Materia-native model)
    ``B = a_v A - a_s A^2/3 - a_c Z(Z-1)/A^1/3 - a_a (N-Z)^2/A + delta``
    with the pairing term ``+-a_p / A^1/2``, coefficients fitted here by least
    squares to the measured (not estimated) AME2020 binding energies with
    ``A >= 16``.  It is a liquid-drop model with no shell effects and is
    labelled semi-empirical.

``radioactivedecay`` (external engine, MIT licence)
    Decay chains and inventories from the ICRP Publication 107 data set:
    half-lives, decay modes, branching fractions and progeny, solved exactly
    for the Bateman equations by the package.

``particle`` (external engine, BSD-3-Clause, scikit-hep)
    Masses, widths, charges, spins and lifetimes of particles from the Particle
    Data Group tables bundled with the package.

Out of scope and refused: nuclear reactions and cross sections, fission
yields, shell-model or ab initio nuclear structure, and collision or detector
simulation.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..elements import periodic_table as pt
from ..provenance import Fidelity, Origin, Provenance, Result

AME_FILE = Path(__file__).resolve().parents[1] / "elements" / "nuclear_data" / "mass_1.mas20.txt"
AME_SHA256 = "e8599c6d7f724fac91934e59f1b9de8fb8f63e820f4b39456b790665ed2a3307"
AME_REFERENCE = ("W. J. Huang, M. Wang, F. G. Kondev, G. Audi and S. Naimi, Chin. Phys. C 45 "
                 "(2021) 030002; M. Wang et al., Chin. Phys. C 45 (2021) 030003 (AME2020)")
U_KEV = 931494.10242
ELECTRON_MASS_U = 5.48579909065e-4
ICRP_REFERENCE = "ICRP Publication 107, Ann. ICRP 38(3) (2008), via radioactivedecay"
PDG_REFERENCE = "Particle Data Group, via the scikit-hep particle package"


class NuclearError(ValueError):
    """The nuclide or request is unknown or outside what is evaluated."""


@dataclass(frozen=True)
class Nuclide:
    Z: int
    N: int
    A: int
    symbol: str
    mass_excess_keV: Optional[float]
    mass_excess_uncertainty_keV: Optional[float]
    binding_per_nucleon_keV: Optional[float]
    atomic_mass_u: Optional[float]
    atomic_mass_uncertainty_u: Optional[float]
    estimated: bool

    @property
    def name(self) -> str:
        return f"{self.symbol}-{self.A}"


def _value(text: str) -> Tuple[Optional[float], bool]:
    text = text.strip()
    if not text or "*" in text:
        return None, False
    estimated = "#" in text
    return float(text.replace("#", ".")), estimated


@lru_cache(maxsize=1)
def table() -> Dict[Tuple[int, int], Nuclide]:
    """Every AME2020 nuclide keyed by (Z, A), after checking the file's checksum."""
    raw = AME_FILE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != AME_SHA256:
        raise NuclearError("The shipped AME2020 table does not match its recorded checksum.")
    out: Dict[Tuple[int, int], Nuclide] = {}
    for line in raw.decode("ascii").splitlines():
        if len(line) < 120:
            continue
        try:
            n, z, a = int(line[4:9]), int(line[9:14]), int(line[14:19])
        except ValueError:
            continue
        symbol = line[20:23].strip()
        excess, e1 = _value(line[28:42])
        excess_unc, _ = _value(line[42:54])
        binding, e2 = _value(line[54:67])
        try:
            mass_prefix = int(line[106:109])
        except ValueError:
            continue
        micro, e3 = _value(line[110:123])
        mass_unc, _ = _value(line[123:135])
        atomic_mass = None if micro is None else mass_prefix + micro * 1e-6
        out[(z, a)] = Nuclide(z, n, a, symbol, excess, excess_unc, binding,
                              atomic_mass, None if mass_unc is None else mass_unc * 1e-6,
                              bool(e1 or e2 or e3))
    return out


def _key(element: Union[int, str], mass_number: int) -> Tuple[int, int]:
    if isinstance(element, str) and element.strip().lower() in ("n", "neutron"):
        return 0, int(mass_number)
    return pt.atomic_number(element) if isinstance(element, str) else int(element), int(mass_number)


def nuclide(element: Union[int, str], mass_number: int) -> Nuclide:
    key = _key(element, mass_number)
    entry = table().get(key)
    if entry is None:
        raise NuclearError(f"AME2020 has no mass for Z = {key[0]}, A = {key[1]}.")
    return entry


def _mass(z: int, a: int) -> float:
    entry = nuclide(z, a)
    if entry.atomic_mass_u is None:
        raise NuclearError(f"AME2020 gives no atomic mass for {entry.name}.")
    return entry.atomic_mass_u


def _provenance(model: str, origin: Origin, estimated: bool, references: List[str],
                notes: str = "") -> Provenance:
    approximations = ["Atomic masses: electron binding energies are neglected in mass "
                      "differences, as is conventional."]
    if estimated:
        approximations.append("At least one mass used is an AME2020 estimate from "
                              "systematics ('#'), not a measurement.")
    return Provenance(model=model, fidelity=Fidelity.TIER0_STRUCTURAL, origin=origin,
                      approximations=approximations, references=references,
                      dataset="AME2020 mass_1.mas20", parameters={"ame_sha256": AME_SHA256},
                      notes=notes)


def binding_energy(element: Union[int, str], mass_number: int) -> Result:
    """Total nuclear binding energy in MeV, ``B = Z M(1H) + N m_n - M(A, Z)``."""
    z, a = _key(element, mass_number)
    entry = nuclide(z, a)
    value = (z * _mass(1, 1) + (a - z) * _mass(0, 1) - _mass(z, a)) * U_KEV / 1000.0
    tabulated = None if entry.binding_per_nucleon_keV is None else \
        entry.binding_per_nucleon_keV * a / 1000.0
    return Result("binding_energy", value, "MeV",
                  _provenance("nuclear/ame2020-masses", Origin.CALCULATED, entry.estimated,
                              [AME_REFERENCE]),
                  uncertainty=(entry.atomic_mass_uncertainty_u or 0.0) * U_KEV / 1000.0,
                  uncertainty_kind="stddev",
                  extra={"nuclide": entry.name, "per_nucleon_MeV": value / a,
                         "tabulated_MeV": tabulated, "estimated": entry.estimated})


DECAYS = {
    "alpha": (-2, -4), "beta-": (1, 0), "beta+": (-1, 0), "ec": (-1, 0),
    "n": (0, -1), "p": (-1, -1), "2beta-": (2, 0),
}


def q_value(element: Union[int, str], mass_number: int, mode: str) -> Result:
    """Decay energy in MeV from atomic masses; positive means energetically allowed.

    ``beta+`` subtracts two electron masses, ``ec`` does not; ``alpha``,
    ``n`` and ``p`` emit a 4He atom, a neutron and a 1H atom.
    """
    if mode not in DECAYS:
        raise NuclearError(f"mode must be one of {sorted(DECAYS)}.")
    z, a = _key(element, mass_number)
    dz, da = DECAYS[mode]
    parent = _mass(z, a)
    daughter = _mass(z + dz, a + da)
    emitted = {"alpha": _mass(2, 4), "n": _mass(0, 1), "p": _mass(1, 1)}.get(mode, 0.0)
    q = parent - daughter - emitted
    if mode == "beta+":
        q -= 2 * ELECTRON_MASS_U
    estimated = nuclide(z, a).estimated or nuclide(z + dz, a + da).estimated
    return Result(f"q_value_{mode}", q * U_KEV / 1000.0, "MeV",
                  _provenance("nuclear/ame2020-masses", Origin.CALCULATED, estimated,
                              [AME_REFERENCE]),
                  extra={"parent": nuclide(z, a).name,
                         "daughter": nuclide(z + dz, a + da).name, "mode": mode,
                         "allowed": q > 0})


def separation_energy(element: Union[int, str], mass_number: int, particle: str = "n") -> Result:
    """Neutron (``n``), proton (``p``), two-neutron (``2n``) or alpha separation energy in MeV."""
    z, a = _key(element, mass_number)
    removal = {"n": (0, 1, _mass(0, 1)), "p": (1, 1, _mass(1, 1)),
               "2n": (0, 2, 2 * _mass(0, 1)), "alpha": (2, 4, _mass(2, 4))}
    if particle not in removal:
        raise NuclearError("particle must be n, p, 2n or alpha.")
    dz, da, mass = removal[particle]
    value = (_mass(z - dz, a - da) + mass - _mass(z, a)) * U_KEV / 1000.0
    estimated = nuclide(z, a).estimated or nuclide(z - dz, a - da).estimated
    return Result(f"separation_energy_{particle}", value, "MeV",
                  _provenance("nuclear/ame2020-masses", Origin.CALCULATED, estimated,
                              [AME_REFERENCE]), extra={"nuclide": nuclide(z, a).name})


@dataclass(frozen=True)
class MassFormula:
    a_v: float
    a_s: float
    a_c: float
    a_a: float
    a_p: float
    rms_MeV: float
    fitted_nuclides: int

    def binding_MeV(self, z: int, a: int) -> float:
        n = a - z
        pairing = 0.0
        if z % 2 == 0 and n % 2 == 0:
            pairing = self.a_p / math.sqrt(a)
        elif z % 2 == 1 and n % 2 == 1:
            pairing = -self.a_p / math.sqrt(a)
        return (self.a_v * a - self.a_s * a ** (2 / 3) - self.a_c * z * (z - 1) / a ** (1 / 3)
                - self.a_a * (n - z) ** 2 / a + pairing)


@lru_cache(maxsize=1)
def fitted_mass_formula(min_A: int = 16) -> MassFormula:
    """Liquid-drop coefficients (MeV) fitted to measured AME2020 binding energies."""
    rows, targets = [], []
    for (z, a), entry in table().items():
        if z < 1 or a < min_A or entry.estimated or entry.binding_per_nucleon_keV is None:
            continue
        n = a - z
        sign = 1.0 if (z % 2 == 0 and n % 2 == 0) else -1.0 if (z % 2 and n % 2) else 0.0
        rows.append([a, -a ** (2 / 3), -z * (z - 1) / a ** (1 / 3), -(n - z) ** 2 / a,
                     sign / math.sqrt(a)])
        targets.append(entry.binding_per_nucleon_keV * a / 1000.0)
    matrix, target = np.array(rows), np.array(targets)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    rms = float(np.sqrt(np.mean((matrix @ coefficients - target) ** 2)))
    return MassFormula(*map(float, coefficients), rms_MeV=rms, fitted_nuclides=len(targets))


def mass_formula_binding(element: Union[int, str], mass_number: int) -> Result:
    z, a = _key(element, mass_number)
    model = fitted_mass_formula()
    measured = None
    entry = table().get((z, a))
    if entry is not None and entry.binding_per_nucleon_keV is not None:
        measured = entry.binding_per_nucleon_keV * a / 1000.0
    prov = Provenance(
        model="nuclear/semi-empirical-mass-formula", fidelity=Fidelity.TIER2_SEMI_EMPIRICAL,
        origin=Origin.ESTIMATED,
        approximations=["Liquid-drop model: volume, surface, Coulomb, asymmetry and pairing "
                        "terms only; no shell or deformation effects.",
                        f"Coefficients fitted to {model.fitted_nuclides} measured AME2020 "
                        f"nuclides with A >= 16; rms residual {model.rms_MeV:.2f} MeV."],
        parameters={"a_v": model.a_v, "a_s": model.a_s, "a_c": model.a_c, "a_a": model.a_a,
                    "a_p": model.a_p},
        references=["C. F. von Weizsaecker, Z. Phys. 96 (1935) 431", AME_REFERENCE])
    return Result("binding_energy", model.binding_MeV(z, a), "MeV", prov,
                  uncertainty=model.rms_MeV, uncertainty_kind="model-spread",
                  extra={"measured_MeV": measured})


def _decay_engine():
    try:
        return importlib.import_module("radioactivedecay")
    except ImportError:
        raise NuclearError("radioactivedecay is not installed: pip install radioactivedecay") \
            from None


def decay(inventory: Dict[str, float], time: float, time_unit: str = "y",
          quantity: str = "Bq") -> Result:
    """An inventory after ``time``, with every progeny, from radioactivedecay (ICRP-107).

    ``inventory`` maps nuclide names such as ``"U-238"`` to amounts in
    ``quantity`` (``Bq``, ``Ci``, ``num`` or ``mol``).  Returns activities in Bq
    and numbers of atoms.  radioactivedecay's year is 365.2422 days.
    """
    rd = _decay_engine()
    try:
        start = rd.Inventory(dict(inventory), quantity)
    except (ValueError, KeyError) as exc:
        raise NuclearError(f"radioactivedecay rejected the inventory: {exc}") from None
    after = start.decay(float(time), time_unit)
    prov = Provenance(model="radioactivedecay/icrp-107", fidelity=Fidelity.TIER0_STRUCTURAL,
                      origin=Origin.CALCULATED,
                      approximations=["Decay data from ICRP Publication 107; the Bateman "
                                      "equations are solved by radioactivedecay.",
                                      "No neutron capture, no spontaneous-fission products."],
                      parameters={"engine": "radioactivedecay",
                                  "engine_version": rd.__version__,
                                  "time": float(time), "time_unit": time_unit,
                                  "initial": dict(inventory), "quantity": quantity},
                      references=[ICRP_REFERENCE,
                                  "A. Malins and T. Lemoine, J. Open Source Softw. 7 (2022) 3318"])
    activities = {str(k): float(v) for k, v in after.activities("Bq").items()}
    numbers = {str(k): float(v) for k, v in after.numbers().items()}
    return Result("inventory", {"activities_Bq": activities, "numbers": numbers}, "mixed",
                  prov)


def decay_data(name: str) -> dict:
    """Half-life, decay modes, branching fractions and progeny of one radionuclide."""
    rd = _decay_engine()
    try:
        entry = rd.Nuclide(name)
    except (ValueError, KeyError) as exc:
        raise NuclearError(f"radioactivedecay has no nuclide {name!r}: {exc}") from None
    return {"nuclide": entry.nuclide, "half_life_s": entry.half_life("s"),
            "half_life_readable": entry.half_life("readable"),
            "progeny": entry.progeny(), "branching_fractions": entry.branching_fractions(),
            "decay_modes": entry.decay_modes(), "source": ICRP_REFERENCE}


def particle_data(identifier: Union[int, str]) -> dict:
    """PDG data for one particle, by PDG id or name, from the scikit-hep package."""
    try:
        particle = importlib.import_module("particle")
    except ImportError:
        raise NuclearError("particle is not installed: pip install particle") from None
    try:
        entry = (particle.Particle.from_pdgid(int(identifier)) if isinstance(identifier, int)
                 else particle.Particle.from_name(str(identifier)))
    except Exception as exc:
        raise NuclearError(f"No PDG particle {identifier!r}: {exc}") from None
    return {"name": entry.name, "pdg_id": int(entry.pdgid), "mass_MeV": entry.mass,
            "mass_uncertainty_MeV": (entry.mass_upper, entry.mass_lower),
            "width_MeV": entry.width, "charge_e": entry.charge,
            "spin_J": entry.J, "lifetime_ns": entry.lifetime,
            "source": PDG_REFERENCE, "engine_version": particle.__version__}


LIGHT = {"n": (0, 1), "p": (1, 1), "d": (1, 2), "t": (1, 3), "h": (2, 3), "a": (2, 4),
         "alpha": (2, 4)}


def _nuclide_key(name: str) -> Tuple[int, int]:
    name = name.strip()
    if name.lower() in LIGHT:
        return LIGHT[name.lower()]
    match = __import__("re").fullmatch(r"([A-Za-z]{1,2})-?(\d{1,3})", name)
    if not match:
        raise NuclearError(f"Cannot read the nuclide {name!r}; write e.g. 'U-235' or 'd'.")
    return pt.atomic_number(match.group(1).capitalize()), int(match.group(2))


def reaction_q_value(reactants: List[str], products: List[str]) -> Result:
    """Q value in MeV of a nuclear reaction from AME2020 atomic masses.

    Charge (proton number) and nucleon number must balance; with balanced
    charge the electron masses cancel in atomic-mass differences.
    """
    left = [_nuclide_key(n) for n in reactants]
    right = [_nuclide_key(n) for n in products]
    if sum(z for z, _ in left) != sum(z for z, _ in right):
        raise NuclearError("Charge does not balance between reactants and products.")
    if sum(a for _, a in left) != sum(a for _, a in right):
        raise NuclearError("Nucleon number does not balance between reactants and products.")
    q = (sum(_mass(z, a) for z, a in left) - sum(_mass(z, a) for z, a in right)) * U_KEV / 1000
    estimated = any(nuclide(z, a).estimated for z, a in left + right)
    return Result("reaction_q_value", q, "MeV",
                  _provenance("nuclear/ame2020-masses", Origin.CALCULATED, estimated,
                              [AME_REFERENCE]),
                  extra={"reactants": list(reactants), "products": list(products),
                         "exothermic": q > 0})


ELECTRON_MASS_MEV = 0.51099895000
K_BETHE = 0.307075
STOPPING_MATERIALS = {
    "water": {"Z_over_A": 0.55509, "I_eV": 75.0, "density_g_cm3": 1.0},
    "air": {"Z_over_A": 0.49919, "I_eV": 85.7, "density_g_cm3": 1.20479e-3},
    "silicon": {"Z_over_A": 14 / 28.0855, "I_eV": 173.0, "density_g_cm3": 2.33},
    "aluminum": {"Z_over_A": 13 / 26.9815384, "I_eV": 166.0, "density_g_cm3": 2.699},
    "copper": {"Z_over_A": 29 / 63.546, "I_eV": 322.0, "density_g_cm3": 8.96},
    "lead": {"Z_over_A": 82 / 207.2, "I_eV": 823.0, "density_g_cm3": 11.35},
}
STOPPING_REFERENCES = [
    "H. Bethe, Ann. Phys. 397 (1930) 325; Particle Data Group, Passage of particles "
    "through matter (Rev. Part. Phys.)",
    "ICRU Report 37 (1984) and ICRU Report 49 (1993): mean excitation energies",
]


def bethe_stopping_power(kinetic_MeV: float, material: str = "water",
                         particle_mass_MeV: float = 938.27208816, charge: int = 1) -> Result:
    """Electronic mass stopping power in MeV cm^2/g from the Bethe formula.

    ``-dE/dx = K z^2 (Z/A) (1/beta^2) [1/2 ln(2 m_e c^2 b^2 g^2 T_max / I^2) - beta^2]``
    without shell, Barkas, Bloch or density-effect corrections.  It is accurate
    to about one percent for protons from a few MeV to a few hundred MeV in
    light materials, and is refused below 1 MeV per nucleon where the
    neglected corrections dominate.
    """
    if material not in STOPPING_MATERIALS:
        raise NuclearError(f"material must be one of {sorted(STOPPING_MATERIALS)}.")
    nucleons = particle_mass_MeV / 931.494
    if kinetic_MeV / nucleons < 1.0:
        raise NuclearError("Below 1 MeV per nucleon the Bethe formula without shell and "
                           "Barkas corrections is not reliable; refused.")
    if kinetic_MeV / particle_mass_MeV > 1.0:
        raise NuclearError("Above about 1 GeV per nucleon the density effect, not included "
                           "here, matters; refused.")
    data = STOPPING_MATERIALS[material]
    gamma = 1 + kinetic_MeV / particle_mass_MeV
    beta2 = 1 - 1 / gamma ** 2
    bg2 = beta2 * gamma ** 2
    ratio = ELECTRON_MASS_MEV / particle_mass_MeV
    t_max = 2 * ELECTRON_MASS_MEV * bg2 / (1 + 2 * gamma * ratio + ratio ** 2)
    i_mev = data["I_eV"] * 1e-6
    value = K_BETHE * charge ** 2 * data["Z_over_A"] / beta2 * (
        0.5 * math.log(2 * ELECTRON_MASS_MEV * bg2 * t_max / i_mev ** 2) - beta2)
    prov = Provenance(model="nuclear/bethe-stopping", fidelity=Fidelity.TIER2_SEMI_EMPIRICAL,
                      origin=Origin.CALCULATED,
                      approximations=["Bethe formula without shell, Barkas, Bloch or density "
                                      "corrections; mean excitation energy from ICRU tables."],
                      parameters={"material": material, **data, "kinetic_MeV": kinetic_MeV,
                                  "particle_mass_MeV": particle_mass_MeV, "charge": charge},
                      references=list(STOPPING_REFERENCES))
    return Result("mass_stopping_power", value, "MeV cm^2/g", prov,
                  extra={"linear_MeV_per_cm": value * data["density_g_cm3"],
                         "beta": math.sqrt(beta2), "t_max_MeV": t_max})


def csda_range(kinetic_MeV: float, material: str = "water",
               particle_mass_MeV: float = 938.27208816, charge: int = 1,
               floor_MeV_per_nucleon: float = 1.0) -> Result:
    """Continuous-slowing-down range in g/cm^2 by integrating 1/S from the floor upward.

    The part of the range below the floor (1 MeV per nucleon by default), where
    the Bethe formula is not used, is not included and is reported as a lower
    bound on the missing length.
    """
    nucleons = particle_mass_MeV / 931.494
    floor = floor_MeV_per_nucleon * nucleons
    if kinetic_MeV <= floor:
        raise NuclearError("The energy is below the floor of the integration.")
    grid = np.geomspace(floor, kinetic_MeV, 400)
    inverse = np.array([1.0 / bethe_stopping_power(e, material, particle_mass_MeV,
                                                   charge).value for e in grid])
    value = float(np.trapezoid(inverse, grid))
    below = floor / bethe_stopping_power(floor, material, particle_mass_MeV, charge).value
    prov = Provenance(model="nuclear/csda-range", fidelity=Fidelity.TIER2_SEMI_EMPIRICAL,
                      origin=Origin.CALCULATED,
                      approximations=[f"Continuous slowing down with Bethe stopping from "
                                      f"{floor:g} MeV upward; the range below is omitted."],
                      parameters={"material": material, "kinetic_MeV": kinetic_MeV,
                                  "floor_MeV": floor},
                      references=list(STOPPING_REFERENCES))
    return Result("csda_range", value, "g/cm^2", prov,
                  extra={"length_cm": value / STOPPING_MATERIALS[material]["density_g_cm3"],
                         "omitted_below_floor_g_cm2_at_most": below})
