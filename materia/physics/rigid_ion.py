"""Rigid-ion potentials: fixed point charges plus a short-range Buckingham term.

Model
-----
Every atom carries a fixed charge ``q`` set by the parameterisation, and every
pair of atoms interacts through

``phi(r) = k_e q_i q_j / r + A exp(-b r) - C / r^6``

with ``k_e = 14.3996 eV A``.  The Coulomb part is summed over all periodic
images with :mod:`materia.physics.electrostatics` (Ewald for crystals,
Yeh-Berkowitz corrected Ewald for slabs, direct sums for clusters).  The
Buckingham part is summed over pairs closer than a cutoff and shifted so that
it is zero there; the energy is continuous at the cutoff and the force has a
jump of ``|dphi/dr|`` at the cutoff, which is recorded.  Ions are not
polarisable and charges do not respond to their environment.

The short-range collapse
------------------------
Wherever ``C > 0`` the ``-C / r^6`` term wins at short range and ``phi``
falls to minus infinity as ``r`` goes to zero.  For each such species pair
the pair interaction has a maximum at a distance ``r_b``; closer than that,
two ions fall into each other and the model has no physical meaning.  This is
the known failure of Buckingham potentials, the "Buckingham catastrophe".
Materia computes ``r_b`` and the barrier height for every pair of species
from the full pair interaction, including the bare Coulomb term, and refuses
any configuration with a pair inside its barrier.  It does not add a repulsive
wall or change the potential.

Shipped parameterisation
------------------------
``bks-silica``: van Beest, Kramer and van Santen (1990), fitted to Hartree-Fock
calculations on an H4SiO4 cluster and to the elastic constants of alpha-quartz.
``q_Si = +2.4 e``, ``q_O = -1.2 e``; Si-O and O-O Buckingham terms and no
Si-Si short-range term.  Any other parameterisation can be supplied through
:class:`RigidIonParameters` with its source stated.

References
----------
B. W. H. van Beest, G. J. Kramer and R. A. van Santen, Phys. Rev. Lett. 64 (1990) 1955.
R. A. Buckingham, Proc. R. Soc. Lond. A 168 (1938) 264.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from ..core_model.structure import Structure
from ..core_model.units import COULOMB_K_EV_A
from ..elements import periodic_table as pt
from ..provenance import Fidelity
from . import electrostatics as es
from .potentials import Potential, UnsupportedSystem

DEFAULT_CUTOFF_A = 10.0
DEFAULT_EWALD_ACCURACY = 1e-8
BARRIER_SCAN_MIN_A = 0.05
BARRIER_SCAN_POINTS = 4000


class RigidIonCollapse(UnsupportedSystem):
    """A pair of ions sits inside the short-range barrier of the Buckingham term."""


@dataclass(frozen=True)
class BuckinghamPair:
    """``A exp(-b r) - C / r^6`` for one unordered pair of species."""

    A_eV: float
    b_per_A: float
    C_eV_A6: float

    def energy(self, r: np.ndarray) -> np.ndarray:
        return self.A_eV * np.exp(-self.b_per_A * r) - self.C_eV_A6 / r ** 6

    def derivative(self, r: np.ndarray) -> np.ndarray:
        return -self.A_eV * self.b_per_A * np.exp(-self.b_per_A * r) + 6.0 * self.C_eV_A6 / r ** 7

    def as_dict(self) -> dict:
        return {"A_eV": self.A_eV, "b_per_A": self.b_per_A, "C_eV_A6": self.C_eV_A6}


def pair_key(a: str, b: str) -> Tuple[str, str]:
    """Canonical key for an unordered pair of element symbols."""
    return tuple(sorted((a, b)))


@dataclass(frozen=True)
class RigidIonParameters:
    """A complete rigid-ion parameterisation: charges, pair terms and cutoff."""

    id: str
    name: str
    charges_e: Mapping[str, float]
    pairs: Mapping[Tuple[str, str], BuckinghamPair]
    source: str
    references: Tuple[str, ...] = ()
    fitted_to: str = ""
    cutoff_A: float = DEFAULT_CUTOFF_A

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("A rigid-ion parameterisation needs a stated source.")
        charges = {pt.symbol(k): float(v) for k, v in dict(self.charges_e).items()}
        if not charges:
            raise ValueError("A rigid-ion parameterisation needs at least one charge.")
        pairs = {}
        for (a, b), term in dict(self.pairs).items():
            a, b = pt.symbol(a), pt.symbol(b)
            if a not in charges or b not in charges:
                raise ValueError(f"Pair {a}-{b} names an element with no charge.")
            if not isinstance(term, BuckinghamPair):
                term = BuckinghamPair(**term)
            values = (term.A_eV, term.b_per_A, term.C_eV_A6)
            if not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"Pair {a}-{b} has a non-finite parameter.")
            if term.A_eV < 0 or term.b_per_A <= 0 or term.C_eV_A6 < 0:
                raise ValueError(f"Pair {a}-{b} needs A >= 0, b > 0 and C >= 0.")
            key = pair_key(a, b)
            if key in pairs:
                raise ValueError(f"Pair {a}-{b} is given twice.")
            pairs[key] = BuckinghamPair(float(term.A_eV), float(term.b_per_A),
                                        float(term.C_eV_A6))
        if not (math.isfinite(self.cutoff_A) and self.cutoff_A > 0):
            raise ValueError("cutoff_A must be a positive number.")
        object.__setattr__(self, "charges_e", charges)
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "references", tuple(self.references))

    @property
    def elements(self) -> List[str]:
        return sorted(self.charges_e)

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "source": self.source,
                "charges_e": dict(sorted(self.charges_e.items())),
                "pairs": {f"{a}-{b}": term.as_dict()
                          for (a, b), term in sorted(self.pairs.items())},
                "cutoff_A": self.cutoff_A, "fitted_to": self.fitted_to,
                "references": list(self.references)}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    def with_cutoff(self, cutoff_A: float) -> "RigidIonParameters":
        return RigidIonParameters(self.id, self.name, self.charges_e, self.pairs, self.source,
                                  self.references, self.fitted_to, float(cutoff_A))


BKS_REFERENCE = ("B. W. H. van Beest, G. J. Kramer and R. A. van Santen, "
                 "Phys. Rev. Lett. 64 (1990) 1955")
BUCKINGHAM_REFERENCE = "R. A. Buckingham, Proc. R. Soc. Lond. A 168 (1938) 264"

BKS_SILICA = RigidIonParameters(
    id="bks-silica",
    name="BKS silica (van Beest, Kramer and van Santen 1990)",
    charges_e={"Si": 2.4, "O": -1.2},
    pairs={("Si", "O"): BuckinghamPair(18003.7572, 4.87318, 133.5381),
           ("O", "O"): BuckinghamPair(1388.7730, 2.76000, 175.0000)},
    source=BKS_REFERENCE,
    references=(BKS_REFERENCE,),
    fitted_to=("Hartree-Fock energies of an H4SiO4 cluster and the experimental "
               "elastic constants of alpha-quartz"),
)

SHIPPED: Dict[str, RigidIonParameters] = {BKS_SILICA.id: BKS_SILICA}


def catalog() -> List[dict]:
    """The shipped parameterisations."""
    return [{"id": p.id, "name": p.name, "elements": p.elements, "digest": p.digest(),
             "references": list(p.references)} for p in SHIPPED.values()]


def select_shipped(elements) -> Optional[str]:
    """The shipped parameterisation whose elements are exactly these, or ``None``."""
    wanted = set(elements)
    for key, p in SHIPPED.items():
        if wanted == set(p.elements):
            return key
    return None


def pair_interaction(params: RigidIonParameters, a: str, b: str, r: np.ndarray) -> np.ndarray:
    """The full unshifted pair interaction, bare Coulomb plus Buckingham, in eV."""
    r = np.asarray(r, dtype=float)
    value = COULOMB_K_EV_A * params.charges_e[a] * params.charges_e[b] / r
    term = params.pairs.get(pair_key(a, b))
    return value + (term.energy(r) if term is not None else 0.0)


def _pair_slope(params: RigidIonParameters, a: str, b: str, r: float) -> float:
    slope = -COULOMB_K_EV_A * params.charges_e[a] * params.charges_e[b] / r ** 2
    term = params.pairs.get(pair_key(a, b))
    return slope + (float(term.derivative(np.array(r))) if term is not None else 0.0)


def barrier(params: RigidIonParameters, a: str, b: str) -> Optional[dict]:
    """The innermost maximum of the pair interaction, or ``None`` if it has none.

    With ``C > 0`` the interaction falls to minus infinity at short range, so
    it rises from there to a maximum at ``r_b``.  The slope is positive below
    ``r_b`` and changes sign there; the root is refined with Brent's method.
    """
    term = params.pairs.get(pair_key(a, b))
    if term is None or term.C_eV_A6 == 0.0:
        return None
    grid = np.geomspace(BARRIER_SCAN_MIN_A, params.cutoff_A, BARRIER_SCAN_POINTS)
    slopes = np.array([_pair_slope(params, a, b, r) for r in grid])
    if slopes[0] <= 0:
        raise ValueError(f"The {a}-{b} interaction is not yet collapsing at "
                         f"{BARRIER_SCAN_MIN_A} A; the barrier search cannot start.")
    change = np.nonzero((slopes[:-1] > 0) & (slopes[1:] <= 0))[0]
    if len(change) == 0:
        return {"r_A": float(params.cutoff_A), "height_eV": None, "inside_cutoff": False}
    k = int(change[0])
    r_b = brentq(lambda r: _pair_slope(params, a, b, r), grid[k], grid[k + 1],
                 xtol=1e-12, rtol=1e-12)
    return {"r_A": float(r_b), "height_eV": float(pair_interaction(params, a, b, r_b)),
            "inside_cutoff": True}


def barriers(params: RigidIonParameters) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for i, a in enumerate(params.elements):
        for b in params.elements[i:]:
            found = barrier(params, a, b)
            if found is not None:
                out["-".join(pair_key(a, b))] = found
    return out


class RigidIon(Potential):
    """Fixed point charges plus Buckingham short-range pair terms."""

    fidelity = Fidelity.TIER1_CLASSICAL

    def __init__(self, params: RigidIonParameters = BKS_SILICA,
                 ewald_accuracy: float = DEFAULT_EWALD_ACCURACY) -> None:
        self.params = params
        self.name = f"rigid-ion/{params.id}"
        self.cutoff_A = float(params.cutoff_A)
        self.ewald = es.EwaldSettings(accuracy=float(ewald_accuracy),
                                      check_convergence=False).validate()
        self.charge_model = es.ChargeModel.per_element(dict(params.charges_e),
                                                       source=params.source)
        self.barriers = barriers(params)
        self._shift = {key: float(term.energy(np.array(self.cutoff_A)))
                       for key, term in params.pairs.items()}
        self._force_jump = {key: abs(float(term.derivative(np.array(self.cutoff_A))))
                            for key, term in params.pairs.items()}

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        present = {pt.symbol(int(z)) for z in structure.numbers}
        missing = sorted(present - set(self.params.charges_e))
        if missing:
            return False, (f"The {self.params.id} rigid-ion model has no parameters for "
                           f"{', '.join(missing)}. It covers {', '.join(self.params.elements)}.")
        if len(structure) == 0:
            return False, "The structure has no atoms."
        charges = self.charge_model.charges(structure)
        geometry = es.classify(structure.cell)
        reason = es.refusal(structure, charges, geometry, self.ewald)
        if reason:
            return False, (f"The {self.params.id} charges give this structure a net charge "
                           f"of {charges.sum():+.6g} e. {reason}" if abs(charges.sum()) > 1e-8
                           else reason)
        return True, ""

    def supports(self, structure: Structure) -> Tuple[bool, str]:
        ok, why = super().supports(structure)
        if not ok:
            return ok, why
        try:
            self._guard(structure, self.neighbours(structure, self.cutoff_A))
        except RigidIonCollapse as exc:
            return False, str(exc)
        return True, ""

    def _symbols(self, structure: Structure) -> np.ndarray:
        return np.array([pt.symbol(int(z)) for z in structure.numbers])

    def _guard(self, structure: Structure, nl) -> None:
        if not self.barriers or len(nl) == 0:
            return
        symbols = self._symbols(structure)
        for name, found in self.barriers.items():
            a, b = name.split("-")
            mask = (((symbols[nl.i] == a) & (symbols[nl.j] == b))
                    | ((symbols[nl.i] == b) & (symbols[nl.j] == a)))
            inside = mask & (nl.d < found["r_A"])
            if inside.any():
                k = int(np.flatnonzero(inside)[np.argmin(nl.d[inside])])
                i, j = int(nl.i[k]), int(nl.j[k])
                raise RigidIonCollapse(
                    f"Atoms #{int(structure.ids[i])} ({symbols[i]}) and "
                    f"#{int(structure.ids[j])} ({symbols[j]}) are {nl.d[k]:.4f} A apart, "
                    f"inside the {name} barrier at {found['r_A']:.4f} A. Closer than that the "
                    "-C/r^6 term of the Buckingham potential wins and the pair collapses "
                    "(the Buckingham catastrophe), so the model has no physical energy here.")

    def short_range(self, structure: Structure) -> Tuple[float, np.ndarray, dict]:
        """The shifted Buckingham energy and forces, and the closest pair of each kind."""
        n = len(structure)
        forces = np.zeros((n, 3))
        closest: Dict[str, float] = {}
        if n < 2 and not any(structure.cell.pbc):
            return 0.0, forces, closest
        nl = self.neighbours(structure, self.cutoff_A)
        self._guard(structure, nl)
        if len(nl) == 0:
            return 0.0, forces, closest
        symbols = self._symbols(structure)
        energy = 0.0
        for key, term in self.params.pairs.items():
            a, b = key
            mask = (((symbols[nl.i] == a) & (symbols[nl.j] == b))
                    | ((symbols[nl.i] == b) & (symbols[nl.j] == a)))
            if not mask.any():
                continue
            d = nl.d[mask]
            closest["-".join(key)] = float(d.min())
            energy += 0.5 * float((term.energy(d) - self._shift[key]).sum())
            slope = term.derivative(d)
            np.add.at(forces, nl.i[mask], (slope / d)[:, None] * nl.D[mask])
        return energy, forces, closest

    def coulomb(self, structure: Structure,
                settings: Optional[es.EwaldSettings] = None) -> es.ElectrostaticsOutput:
        try:
            return es.compute(structure, self.charge_model, settings or self.ewald)
        except (es.UndefinedElectrostatics, es.ChargeModelError) as exc:
            raise UnsupportedSystem(str(exc)) from None

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        self.check(structure)
        e_short, f_short, _ = self.short_range(structure)
        coulomb = self.coulomb(structure)
        return e_short + float(coulomb.energy_eV), f_short + coulomb.forces_eV_A

    def diagnostics(self, structure: Structure) -> dict:
        """Energy components, an independent Ewald check and the closest pairs."""
        e_short, _, closest = self.short_range(structure)
        checked = self.coulomb(structure, es.EwaldSettings(
            accuracy=self.ewald.accuracy, check_convergence=True).validate())
        margins = {name: closest[name] - found["r_A"]
                   for name, found in self.barriers.items() if name in closest}
        return {
            "components_eV": {"short_range": e_short,
                              "coulomb": float(checked.energy_eV),
                              **{f"coulomb_{k}": float(v)
                                 for k, v in checked.components_eV.items()}},
            "geometry": checked.geometry.kind,
            "ewald_parameters": dict(checked.parameters),
            "ewald_check": checked.check,
            "ewald_converged": bool(checked.converged),
            "ewald_message": checked.convergence_message,
            "closest_pair_A": closest,
            "barrier_margin_A": margins,
        }

    def describe(self) -> dict:
        p = self.params
        barrier_text = "; ".join(
            f"{name} at {found['r_A']:.4f} A, {found['height_eV']:.3f} eV"
            for name, found in self.barriers.items() if found["inside_cutoff"])
        jump = ", ".join(f"{'-'.join(k)} {v:.2e} eV/A" for k, v in self._force_jump.items())
        return {
            "model": self.name,
            "fidelity": self.fidelity.value,
            "cutoff_A": self.cutoff_A,
            "parameters": {"parameterisation": p.as_dict(), "parameter_digest": p.digest(),
                           "ewald": self.ewald.as_dict(), "barriers": self.barriers},
            "approximations": [
                f"Rigid ions with fixed charges {', '.join(f'{k} {v:+g} e' for k, v in sorted(p.charges_e.items()))}: "
                "no electronic polarisation and no charge transfer.",
                f"Buckingham terms truncated at {self.cutoff_A:g} A and shifted to zero "
                f"there; the force jumps at the cutoff by {jump}.",
                "Coulomb sum by Ewald summation with tin-foil boundary conditions for "
                "crystals, the Yeh-Berkowitz correction for slabs and a direct sum for "
                "clusters, each without a per-step convergence check; the check is run "
                "on the final configuration.",
                "Configurations with a pair inside the Buckingham barrier are refused "
                f"({barrier_text or 'none'}); no repulsive wall is added.",
                f"Parameters fitted to {p.fitted_to}." if p.fitted_to else
                f"Parameters from {p.source}.",
            ],
            "references": list(p.references) + [BUCKINGHAM_REFERENCE] + list(es.REFERENCES[:2]),
        }
