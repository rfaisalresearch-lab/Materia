"""X-ray powder diffraction, attenuation and absorption edges from XrayDB data.

Reference data come from XrayDB (M. Newville, MIT licence), installed with
``pip install xraydb``: Waasmaier-Kirfel atomic form factors
(Acta Cryst. A51 (1995) 416), Chantler anomalous scattering factors (J. Phys.
Chem. Ref. Data 29 (2000) 597), Elam-Ravel-Sieber cross sections, edges and
emission lines (Radiat. Phys. Chem. 63 (2002) 121).  Nothing is copied into
Materia; a missing package is a refusal.

Powder diffraction is kinematic.  For every reflection with ``d >= lambda / 2``
inside the requested angular range the structure factor

``F(hkl) = sum_j o_j [f0_j(s) + f'_j(E) + i f''_j(E)] exp(-B_j s^2) exp(2 pi i h . x_j)``

with ``s = sin(theta) / lambda`` is summed over the cell, symmetry-equivalent
reflections are merged by equal d spacing and equal ``|F|^2`` (Friedel pairs
included), and the integrated intensity is

``I = m |F|^2 (1 + cos^2 2theta) / (sin^2 theta cos theta)``

(Lorentz-polarisation for an unpolarised beam without monochromator).  No
absorption, preferred orientation, extinction or texture is modelled.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..elements import periodic_table as pt
from ..provenance import Fidelity, Origin, Provenance, Result
from .potentials import UnsupportedSystem

HC_EV_A = 12398.419843320026
WAVELENGTHS_A = {"CuKa1": 1.540593, "CuKa": 1.541838, "MoKa1": 0.709317, "CoKa1": 1.788996,
                 "CrKa1": 2.289760, "AgKa1": 0.559421}
MAX_REFLECTIONS = 200_000
REFERENCES = ["D. Waasmaier and A. Kirfel, Acta Cryst. A51 (1995) 416",
              "C. T. Chantler, J. Phys. Chem. Ref. Data 29 (2000) 597",
              "W. T. Elam, B. D. Ravel and J. R. Sieber, Radiat. Phys. Chem. 63 (2002) 121",
              "M. Newville, XrayDB, https://github.com/xraypy/XrayDB"]


class XrayError(ValueError):
    pass


def _xraydb():
    try:
        import xraydb
    except ImportError:
        raise UnsupportedSystem("X-ray reference data need XrayDB: pip install xraydb.") from None
    return xraydb


def wavelength(source) -> float:
    if isinstance(source, str):
        if source not in WAVELENGTHS_A:
            raise XrayError(f"Unknown source {source!r}; use one of {sorted(WAVELENGTHS_A)} or "
                            "a wavelength in A.")
        return WAVELENGTHS_A[source]
    value = float(source)
    if not 0.1 <= value <= 5.0:
        raise XrayError("The wavelength must lie between 0.1 and 5 A.")
    return value


def _factors(symbols: Sequence[str], energy_eV: float, anomalous: bool):
    db = _xraydb()
    unique = sorted(set(symbols))
    real = {s: float(db.f1_chantler(s, energy_eV)) if anomalous else 0.0 for s in unique}
    imaginary = {s: float(db.f2_chantler(s, energy_eV)) if anomalous else 0.0 for s in unique}

    def f(symbol: str, s: float) -> complex:
        return complex(float(db.f0(symbol, s)[0]) + real[symbol], imaginary[symbol])

    return f, real, imaginary


def powder_pattern(structure: Structure, source="CuKa1", two_theta_range=(10.0, 120.0),
                   anomalous: bool = True, debye_waller_A2: Optional[Mapping[str, float]] = None,
                   occupancies: Optional[Sequence[float]] = None,
                   d_tolerance_A: float = 1e-5, intensity_tolerance: float = 1e-4) -> Result:
    """Kinematic powder pattern: peak positions, multiplicities, |F|^2 and intensities."""
    if not all(structure.cell.pbc):
        raise XrayError("Powder diffraction needs a crystal periodic in three directions.")
    lam = wavelength(source)
    low, high = (float(v) for v in two_theta_range)
    if not 0 < low < high < 180:
        raise XrayError("two_theta_range must lie inside (0, 180) degrees and be increasing.")
    symbols = [pt.symbol(int(z)) for z in structure.numbers]
    occ = np.ones(len(symbols)) if occupancies is None else np.asarray(occupancies, dtype=float)
    if occ.shape != (len(symbols),) or np.any(occ < 0) or np.any(occ > 1):
        raise XrayError("occupancies must give one value in [0, 1] per site.")
    b_factors = {s: float((debye_waller_A2 or {}).get(s, 0.0)) for s in set(symbols)}
    if any(b < 0 for b in b_factors.values()):
        raise XrayError("Debye-Waller B factors cannot be negative.")
    energy = HC_EV_A / lam
    form, f_prime, f_double = _factors(symbols, energy, anomalous)
    cell = np.asarray(structure.cell.matrix, dtype=float)
    reciprocal = np.linalg.inv(cell).T
    fractional = np.asarray(structure.positions, dtype=float) @ np.linalg.inv(cell)
    d_min = lam / (2 * math.sin(math.radians(high) / 2))
    limits = [int(math.ceil(np.linalg.norm(cell[i]) / d_min)) + 1 for i in range(3)]
    count = np.prod([2 * n + 1 for n in limits])
    if count > MAX_REFLECTIONS:
        raise XrayError(f"{count} reflections would be enumerated; reduce the angular range or "
                        "use a smaller cell.")
    hkl = np.array(list(itertools.product(*[range(-n, n + 1) for n in limits])))
    hkl = hkl[np.any(hkl != 0, axis=1)]
    g = np.linalg.norm(hkl @ reciprocal, axis=1)
    d = 1 / g
    sin_theta = lam / (2 * d)
    keep = sin_theta <= 1.0
    hkl, d, sin_theta = hkl[keep], d[keep], sin_theta[keep]
    two_theta = 2 * np.degrees(np.arcsin(sin_theta))
    keep = (two_theta >= low) & (two_theta <= high)
    hkl, d, sin_theta, two_theta = hkl[keep], d[keep], sin_theta[keep], two_theta[keep]
    s = sin_theta / lam
    phases = np.exp(2j * np.pi * hkl @ fractional.T)
    unique = sorted(set(symbols))
    table = {}
    for symbol in unique:
        values = np.array([form(symbol, float(v)) for v in s])
        table[symbol] = values * np.exp(-b_factors[symbol] * s ** 2)
    amplitudes = np.zeros(len(hkl), dtype=complex)
    for j, symbol in enumerate(symbols):
        amplitudes += occ[j] * table[symbol] * phases[:, j]
    intensity_f = np.abs(amplitudes) ** 2
    theta = np.radians(two_theta / 2)
    lp = (1 + np.cos(2 * theta) ** 2) / (np.sin(theta) ** 2 * np.cos(theta))
    order = np.lexsort((-intensity_f, -d))
    peaks: List[dict] = []
    scale = max(1.0, float(intensity_f.max())) if len(intensity_f) else 1.0
    for index in order:
        if peaks and abs(peaks[-1]["d_A"] - d[index]) < d_tolerance_A and \
                abs(peaks[-1]["structure_factor_sq"] - intensity_f[index]) <= \
                intensity_tolerance * scale:
            peaks[-1]["multiplicity"] += 1
            peaks[-1]["members"].append(hkl[index].tolist())
            continue
        peaks.append({"two_theta_deg": float(two_theta[index]), "d_A": float(d[index]),
                      "structure_factor_sq": float(intensity_f[index]),
                      "lorentz_polarisation": float(lp[index]), "multiplicity": 1,
                      "members": [hkl[index].tolist()]})
    absent = 1e-6 * scale
    allowed = [p for p in peaks if p["structure_factor_sq"] > absent]
    for p in allowed:
        members = [tuple(int(x) for x in v) for v in p["members"]]
        if _is_cubic(cell):
            p["hkl"] = sorted((abs(x) for x in members[0]), reverse=True)
        else:
            p["hkl"] = list(max(members, key=lambda v: (sum(x >= 0 for x in v), v)))
        p["intensity"] = p["multiplicity"] * p["structure_factor_sq"] * p["lorentz_polarisation"]
    top = max((p["intensity"] for p in allowed), default=1.0)
    for p in allowed:
        p["relative_intensity"] = 100.0 * p["intensity"] / top
        del p["members"]
    allowed.sort(key=lambda p: p["two_theta_deg"])
    prov = Provenance(
        model="xray/kinematic-powder", fidelity=Fidelity.TIER0_STRUCTURAL,
        origin=Origin.CALCULATED,
        approximations=[
            "Kinematic diffraction from independent spherical atoms (Waasmaier-Kirfel form "
            "factors" + (" with Chantler anomalous corrections" if anomalous else "") + ").",
            "Lorentz-polarisation for an unpolarised beam, no monochromator; no absorption, "
            "extinction, preferred orientation or peak profile.",
            "Isotropic Debye-Waller factors as given (zero by default: a static lattice)."],
        parameters={"wavelength_A": lam, "energy_eV": energy,
                    "two_theta_range_deg": [low, high], "anomalous": anomalous,
                    "debye_waller_A2": b_factors,
                    "f_prime": f_prime, "f_double_prime": f_double},
        references=list(REFERENCES))
    return Result("powder_diffraction", allowed, "mixed", prov,
                  extra={"systematic_absences": len(peaks) - len(allowed),
                         "wavelength_A": lam})


def _is_cubic(cell: np.ndarray) -> bool:
    lengths = np.linalg.norm(cell, axis=1)
    gram = cell @ cell.T / np.outer(lengths, lengths)
    return bool(np.allclose(lengths, lengths[0], rtol=1e-6) and
                np.allclose(gram, np.eye(3), atol=1e-6))


def profile(peaks: Sequence[dict], two_theta_deg: np.ndarray, fwhm_deg: float = 0.1,
            eta: float = 0.5) -> np.ndarray:
    """Pseudo-Voigt profile of the given peaks on a 2-theta grid, peak areas preserved."""
    if not 0 <= eta <= 1 or fwhm_deg <= 0:
        raise XrayError("eta must lie in [0, 1] and fwhm_deg must be positive.")
    x = np.asarray(two_theta_deg, dtype=float)
    out = np.zeros_like(x)
    sigma = fwhm_deg / (2 * math.sqrt(2 * math.log(2)))
    gamma = fwhm_deg / 2
    for p in peaks:
        u = x - p["two_theta_deg"]
        gauss = np.exp(-0.5 * (u / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        lorentz = gamma / (math.pi * (u ** 2 + gamma ** 2))
        out += p["intensity"] * ((1 - eta) * gauss + eta * lorentz)
    return out


def density_g_cm3(structure: Structure) -> float:
    mass_g = float(np.sum(structure.masses())) * 1.66053906660e-24
    return mass_g / (structure.cell.volume * 1e-24)


def formula(structure: Structure) -> str:
    counts: Dict[str, int] = {}
    for z in structure.numbers:
        symbol = pt.symbol(int(z))
        counts[symbol] = counts.get(symbol, 0) + 1
    return "".join(f"{s}{n}" for s, n in sorted(counts.items()))


def attenuation(material, energies_eV: Sequence[float], density: Optional[float] = None,
                thickness_um: Optional[float] = None) -> Result:
    """Mass and linear attenuation, attenuation length and optional transmission.

    ``material`` is a chemical formula with a ``density`` in g/cm^3, or a
    periodic structure whose formula and density are taken from the cell.
    Cross sections are the Elam total (photoabsorption, coherent and
    incoherent scattering).
    """
    db = _xraydb()
    if isinstance(material, Structure):
        if not all(material.cell.pbc):
            raise XrayError("A structure must be a periodic crystal to define a density.")
        name, rho = formula(material), density_g_cm3(material)
    else:
        if density is None or density <= 0:
            raise XrayError("Give the density in g/cm^3 for a formula.")
        name, rho = str(material), float(density)
    energies = np.asarray(energies_eV, dtype=float)
    if np.any(energies < 100) or np.any(energies > 8e5):
        raise XrayError("Energies must lie between 100 eV and 800 keV, the range of the "
                        "Elam tables.")
    try:
        mu = np.atleast_1d(db.material_mu(name, energies, density=rho))
    except Exception as exc:
        raise XrayError(f"XrayDB could not evaluate {name}: {exc}") from None
    value = {"formula": name, "density_g_cm3": rho, "energies_eV": energies.tolist(),
             "mass_attenuation_cm2_g": (mu / rho).tolist(),
             "linear_attenuation_per_cm": mu.tolist(),
             "attenuation_length_um": (1e4 / mu).tolist()}
    if thickness_um is not None:
        value["transmission"] = np.exp(-mu * float(thickness_um) * 1e-4).tolist()
    prov = Provenance(model="xray/elam-attenuation", fidelity=Fidelity.TIER0_STRUCTURAL,
                      origin=Origin.REFERENCE,
                      approximations=["Independent-atom cross sections (Elam tables): no "
                                      "near-edge or chemical-state structure."],
                      parameters={"formula": name, "density_g_cm3": rho},
                      references=list(REFERENCES))
    return Result("xray_attenuation", value, "mixed", prov)


def edges_and_lines(element: str) -> dict:
    """Absorption edges (eV, fluorescence yield, jump ratio) and emission lines."""
    db = _xraydb()
    try:
        edges = db.xray_edges(element)
        lines = db.xray_lines(element)
    except Exception as exc:
        raise XrayError(f"No X-ray data for {element!r}: {exc}") from None
    return {"element": element,
            "edges": {k: {"energy_eV": float(v.energy), "fluorescence_yield": float(v.fyield),
                          "jump_ratio": float(v.jump_ratio)} for k, v in edges.items()},
            "lines": {k: {"energy_eV": float(v.energy), "intensity": float(v.intensity),
                          "transition": f"{v.initial_level}-{v.final_level}"}
                      for k, v in lines.items()}}
