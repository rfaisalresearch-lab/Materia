"""Ideal-gas thermochemistry of a molecule: enthalpy, entropy and Gibbs energy.

Rigid-rotor, harmonic-oscillator, ideal-gas partition functions (D. A.
McQuarrie, Statistical Mechanics, Harper and Row, 1976, chapters 5 and 8):

* translation: ``H = 5/2 kT`` (including ``PV = kT``),
  ``S = k [ln((2 pi m kT / h^2)^(3/2) kT / P) + 5/2]``;
* rotation, linear: ``U = kT``, ``S = k [ln(8 pi^2 I kT / (sigma h^2)) + 1]``;
  nonlinear: ``U = 3/2 kT``,
  ``S = k [ln(sqrt(pi I_A I_B I_C) / sigma (8 pi^2 kT / h^2)^(3/2)) + 3/2]``;
* vibration: zero-point energy plus Bose-Einstein populations of each
  harmonic mode;
* electronic: ground state only, ``S = k ln(2S + 1)``.

The rotational symmetry number ``sigma`` comes from the point group, which
PySCF detects from the geometry, or is supplied.  Frequencies come from any
harmonic analysis (:func:`materia.physics.phonons.harmonic_analysis` with any
potential, or :func:`materia.physics.molecular.vibrational_spectra`).  The
geometry must be a minimum: imaginary frequencies, or a mode count other than
``3N - 6`` (``3N - 5`` for linear molecules), are refused.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..core_model.units import AVOGADRO, BOLTZMANN_J_K, ELEMENTARY_CHARGE_C, PLANCK_J_S
from ..elements import periodic_table as pt
from ..provenance import Fidelity, Origin, Provenance, Result

AMU_KG = 1.66053906660e-27
LIGHT_CM_S = 2.99792458e10
EV_TO_KJ_MOL = ELEMENTARY_CHARGE_C * AVOGADRO / 1000.0
LINEAR_TOLERANCE = 1e-3


class ThermochemistryError(ValueError):
    pass


def symmetry_number(structure: Structure, tolerance_A: float = 0.01) -> Tuple[int, str]:
    """Rotational symmetry number and the point group PySCF detects."""
    try:
        from pyscf.symm import geom
    except ImportError as exc:
        raise ThermochemistryError("Point-group detection needs PySCF; give "
                                   "symmetry_number explicitly.") from exc
    atoms = [(pt.symbol(int(z)), np.asarray(p, dtype=float))
             for z, p in zip(structure.numbers, structure.positions)]
    previous = geom.TOLERANCE
    geom.TOLERANCE = tolerance_A
    try:
        group = geom.detect_symm(atoms)[0]
    finally:
        geom.TOLERANCE = previous
    if group in ("C1", "Ci", "Cs", "Coov"):
        return 1, group
    if group == "Dooh":
        return 2, group
    if group in ("T", "Td"):
        return 12, group
    if group in ("O", "Oh"):
        return 24, group
    if group in ("I", "Ih"):
        return 60, group
    head, digits = group[0], "".join(c for c in group if c.isdigit())
    n = int(digits) if digits else 1
    if head == "C":
        return n, group
    if head == "D":
        return 2 * n, group
    if head == "S":
        return n // 2, group
    raise ThermochemistryError(f"No symmetry number is known for point group {group}.")


def principal_moments(structure: Structure) -> np.ndarray:
    """Principal moments of inertia in amu A^2, ascending."""
    masses = np.asarray(structure.masses(), dtype=float)
    positions = np.asarray(structure.positions, dtype=float)
    centre = masses @ positions / masses.sum()
    r = positions - centre
    tensor = np.einsum("i,ij,ik->jk", masses, r, r)
    inertia = np.trace(tensor) * np.eye(3) - tensor
    return np.linalg.eigvalsh(inertia)


def ideal_gas(structure: Structure, electronic_energy_eV: float,
              frequencies_cm1: Sequence[float], temperature_K: float = 298.15,
              pressure_Pa: float = 101325.0, spin_multiplicity: int = 1,
              sigma: Optional[int] = None, model: str = "",
              fidelity: Fidelity = Fidelity.NON_PHYSICAL) -> Result:
    """Enthalpy, entropy, Gibbs energy and heat capacity of one ideal-gas molecule.

    ``model`` and ``fidelity`` describe where the energy and frequencies came
    from; the default fidelity marks that source as unknown.
    """
    if temperature_K <= 0 or pressure_Pa <= 0:
        raise ThermochemistryError("Temperature and pressure must be positive.")
    if any(structure.cell.pbc):
        raise ThermochemistryError("Ideal-gas thermochemistry is for an isolated molecule; the "
                                   "structure is periodic.")
    n = len(structure)
    if n < 1 or spin_multiplicity < 1:
        raise ThermochemistryError("Need at least one atom and a multiplicity of at least 1.")
    moments = principal_moments(structure)
    if n == 1:
        geometry = "atom"
        expected = 0
    elif moments[0] < LINEAR_TOLERANCE * max(moments[2], 1e-12):
        geometry = "linear"
        expected = 3 * n - 5
    else:
        geometry = "nonlinear"
        expected = 3 * n - 6
    nu = np.asarray(list(frequencies_cm1), dtype=float)
    if np.any(nu <= 0):
        raise ThermochemistryError("Imaginary or zero frequencies: the geometry is not a "
                                   "minimum, and the harmonic partition function is undefined.")
    if len(nu) != expected:
        raise ThermochemistryError(f"A {geometry} molecule of {n} atoms has {expected} "
                                   f"vibrations; {len(nu)} frequencies were given.")
    group = None
    if sigma is None and n > 1:
        sigma, group = symmetry_number(structure)
    sigma = int(sigma or 1)
    k, h, t = BOLTZMANN_J_K, PLANCK_J_S, float(temperature_K)
    kt = k * t
    mass = float(np.sum(structure.masses())) * AMU_KG
    s_trans = k * (math.log((2 * math.pi * mass * kt / h ** 2) ** 1.5 * kt / pressure_Pa) + 2.5)
    h_trans = 2.5 * kt
    cp_trans = 2.5 * k
    inertia = moments * AMU_KG * 1e-20
    if geometry == "atom":
        s_rot = h_rot = cp_rot = 0.0
    elif geometry == "linear":
        s_rot = k * (math.log(8 * math.pi ** 2 * inertia[2] * kt / (sigma * h ** 2)) + 1)
        h_rot, cp_rot = kt, k
    else:
        s_rot = k * (math.log(math.sqrt(math.pi * np.prod(inertia)) / sigma
                              * (8 * math.pi ** 2 * kt / h ** 2) ** 1.5) + 1.5)
        h_rot, cp_rot = 1.5 * kt, 1.5 * k
    quanta = h * nu * LIGHT_CM_S
    x = quanta / kt
    zero_point = 0.5 * float(quanta.sum())
    h_vib = float(np.sum(quanta / np.expm1(x)))
    s_vib = k * float(np.sum(x / np.expm1(x) - np.log(-np.expm1(-x))))
    cp_vib = k * float(np.sum(x ** 2 * np.exp(-x) / np.expm1(-x) ** 2))
    s_elec = k * math.log(spin_multiplicity)
    to_ev = 1 / ELEMENTARY_CHARGE_C
    entropy = (s_trans + s_rot + s_vib + s_elec) * to_ev
    enthalpy = electronic_energy_eV + (zero_point + h_trans + h_rot + h_vib) * to_ev
    gibbs = enthalpy - t * entropy
    per_mol_k = ELEMENTARY_CHARGE_C * AVOGADRO
    value = {"enthalpy_eV": enthalpy, "entropy_eV_K": entropy, "gibbs_energy_eV": gibbs,
             "zero_point_energy_eV": zero_point * to_ev,
             "thermal_enthalpy_correction_eV": (zero_point + h_trans + h_rot + h_vib) * to_ev,
             "heat_capacity_p_eV_K": (cp_trans + cp_rot + cp_vib) * to_ev,
             "entropy_J_mol_K": entropy * per_mol_k,
             "entropy_terms_J_mol_K": {"translational": s_trans * AVOGADRO,
                                       "rotational": s_rot * AVOGADRO,
                                       "vibrational": s_vib * AVOGADRO,
                                       "electronic": s_elec * AVOGADRO},
             "geometry": geometry, "symmetry_number": sigma, "point_group": group,
             "principal_moments_amu_A2": moments.tolist()}
    prov = Provenance(
        model=f"thermochemistry/rrho-ideal-gas[{model}]" if model else
        "thermochemistry/rrho-ideal-gas",
        fidelity=fidelity,
        origin=Origin.CALCULATED,
        approximations=[
            "Ideal gas, rigid rotor and harmonic oscillator: no anharmonicity, hindered "
            "rotation or centrifugal distortion.",
            "Only the electronic ground state, with degeneracy 2S + 1.",
            "Low frequencies are treated as harmonic vibrations; their entropy is "
            "unreliable below about 100 cm^-1."],
        parameters={"temperature_K": t, "pressure_Pa": pressure_Pa,
                    "spin_multiplicity": spin_multiplicity, "symmetry_number": sigma,
                    "frequencies_cm1": nu.tolist()},
        references=["D. A. McQuarrie, Statistical Mechanics (Harper and Row, 1976)"])
    low = int(np.sum(nu < 100))
    return Result("ideal_gas_thermochemistry", value, "mixed", prov,
                  extra={"low_frequency_modes": low,
                         "kj_per_mol_per_eV": EV_TO_KJ_MOL})
